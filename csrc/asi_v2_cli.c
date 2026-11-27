/*
 * asi_v2_cli.c — Standalone C11 Command-Line Runner for ASI v2.6 Cartridges
 *
 * Works like `llama-cli` for `.asi` cartridges using `liblila_asi.so`:
 *   ./lila-asi-cli <model.asi> --info
 *   ./lila-asi-cli <model.asi> --bench [-t N] [-n N] [--mode fast|balanced|exact]
 *   ./lila-asi-cli <model.asi> -p "Prompt text" [-n N] [-t N] [--temp T] [--mode fast|balanced|exact]
 *   ./lila-asi-cli <model.asi> --tokens "151644,872,198,..." [-n N] [-t N] [--temp T]
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "lila_asi.h"
#include "asi_platform.h"
#include "asi_cpu.h"
#include "asi_state_centric.h"
#include "asi_grammar_sparse.h"

#include <ctype.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double now_ms(void)
{
    return (double)asi_monotonic_ns() * 1e-6;
}

static const char *arch_id_name(uint32_t arch_id)
{
    switch (arch_id)
    {
        case ASI_ARCH_DENSE_LLAMA:   return "dense_llama";
        case ASI_ARCH_HYBRID_QWEN35: return "hybrid_qwen35 (Gated DeltaNet + Full Attention)";
        case ASI_ARCH_DENSE_GEMMA:   return "dense_gemma";
        case ASI_ARCH_DENSE_QWEN2:   return "dense_qwen2";
        case ASI_ARCH_DENSE_QWEN3:   return "dense_qwen3";
        default:                     return "unknown";
    }
}

static const char *section_type_name(uint32_t stype)
{
    switch (stype)
    {
        case ASI_SEC_MODEL_CONFIG:    return "MODEL_CONFIG";
        case ASI_SEC_WEIGHTS:         return "WEIGHTS";
        case ASI_SEC_MEMORY_FABRIC:   return "MEMORY_FABRIC";
        case ASI_SEC_TOKENIZER:       return "TOKENIZER";
        case ASI_SEC_BYTECODE_LEGACY: return "BYTECODE_LEGACY";
        case ASI_SEC_HARNESS_LEGACY:  return "HARNESS_LEGACY";
        case ASI_SEC_PERSONALITY:     return "PERSONALITY";
        case ASI_SEC_METADATA:        return "METADATA";
        case ASI_SEC_CHAT_TEMPLATE:   return "CHAT_TEMPLATE";
        case ASI_SEC_VISION:          return "VISION";
        case ASI_SEC_AUDIO:           return "AUDIO";
        case ASI_SEC_ADAPTERS:        return "ADAPTERS";
        case ASI_SEC_RUNTIME_HINTS:   return "RUNTIME_HINTS";
        case ASI_SEC_SIGNATURE:       return "SIGNATURE";
        case ASI_SEC_GRAPH:           return "GRAPH";
        case ASI_SEC_PAIRING:         return "PAIRING";
        default:                      return "OTHER";
    }
}

/* Translate CLI --mode preset into coordinated LILA_ASI_* switches consumed by session_create. */
static void apply_mode_env(const char *mode)
{
    if (!mode || !mode[0]) return;
    if (strcmp(mode, "fast") == 0)
    {
        asi_setenv("LILA_ASI_MODE", "fast");             /* Default high-speed preset (+sparse ffn_down) */
        asi_setenv("LILA_ASI_LMHEAD_MODE", "1bit1024");  /* 1-bit Hadamard Sieve over 248K vocab */
        asi_setenv("LILA_ASI_LMHEAD_K", "2048");         /* Top-K exact Stage-2 candidates */
        asi_setenv("LILA_ASI_SPECULATIVE", "1");         /* Self-speculative multi-token draft verify */
        asi_setenv("LILA_ASI_SPEC_GATE", "1");           /* Share Sieve screening across draft lanes */
    }
    else if (strcmp(mode, "balanced") == 0)
    {
        asi_setenv("LILA_ASI_MODE", "balanced");         /* Tighter quality preset (dense ffn_down) */
        asi_setenv("LILA_ASI_LMHEAD_MODE", "2bit1024");  /* 2-bit sign+magnitude Sieve sketch */
        asi_setenv("LILA_ASI_LMHEAD_K", "1024");         /* Narrower K=1024 Stage-2 refinement */
        asi_setenv("LILA_ASI_SPECULATIVE", "1");
        asi_setenv("LILA_ASI_SPEC_GATE", "1");
    }
    else if (strcmp(mode, "exact") == 0)
    {
        asi_setenv("LILA_ASI_MODE", "exact");            /* Zero-approximation parity preset */
        asi_setenv("LILA_ASI_LMHEAD_MODE", "off");       /* Full exhaustive lm_head evaluation */
        asi_setenv("LILA_ASI_SPECULATIVE", "0");         /* Disable draft speculation */
        asi_setenv("LILA_ASI_SPEC_GATE", "0");
    }
}

/*
 * Lightweight embedded tokenizer loader from ASI_SEC_TOKENIZER JSON section.
 * Extracts vocab pieces and decode strings (converting GPT-2 byte-level UTF-8
 * markers like \xc4\xa0 -> ' ' and \xc4\x8a -> '\n') and provides greedy
 * longest-match tokenization for standalone C CLI usage.
 */
typedef struct {
    uint32_t vocab_size;
    char   **raw_pieces;
    char   **decoded_pieces;
    int32_t  eos_id;
    int32_t  im_start_id;
    int32_t  im_end_id;
    uint16_t b2u[256];
    int16_t  u2b[512];
} SimpleTokenizer;

static void init_gpt2_byte_tables(SimpleTokenizer *tok)
{
    for (int i = 0; i < 512; ++i) tok->u2b[i] = -1;
    bool in_bs[256] = {false};
    for (int b = 33; b <= 126; ++b) in_bs[b] = true;
    for (int b = 161; b <= 172; ++b) in_bs[b] = true;
    for (int b = 174; b <= 255; ++b) in_bs[b] = true;
    int n = 0;
    for (int b = 0; b < 256; ++b)
    {
        uint16_t cp = in_bs[b] ? (uint16_t)b : (uint16_t)(256 + n++);
        tok->b2u[b] = cp;
        if (cp < 512) tok->u2b[cp] = (int16_t)b;
    }
}

static void decode_gpt2_piece(const SimpleTokenizer *tok, const char *src, char *dst, size_t cap)
{
    size_t si = 0, di = 0;
    while (src[si] && di + 4 < cap)
    {
        uint8_t b0 = (uint8_t)src[si];
        uint8_t b1 = (uint8_t)src[si + 1];
        /* SentencePiece U+2581 ('▁') = 0xE2 0x96 0x81 */
        if (b0 == 0xE2 && b1 == 0x96 && (uint8_t)src[si + 2] == 0x81)
        {
            dst[di++] = ' ';
            si += 3;
            continue;
        }
        uint32_t cp = 0;
        size_t adv = 1;
        if (b0 < 0x80)
        {
            cp = b0;
            adv = 1;
        }
        else if ((b0 & 0xE0) == 0xC0 && (b1 & 0xC0) == 0x80)
        {
            cp = ((uint32_t)(b0 & 0x1F) << 6) | (uint32_t)(b1 & 0x3F);
            adv = 2;
        }
        if (cp < 512 && tok->u2b[cp] >= 0)
        {
            uint8_t byte_val = (uint8_t)tok->u2b[cp];
            if (byte_val != 0)
            {
                dst[di++] = (char)byte_val;
            }
            si += adv;
            continue;
        }
        dst[di++] = src[si++];
    }
    dst[di] = '\0';
}

static int load_simple_tokenizer(const LilaAsiModel *model, uint32_t vocab_size, SimpleTokenizer *out_tok)
{
    memset(out_tok, 0, sizeof(*out_tok));
    init_gpt2_byte_tables(out_tok);
    out_tok->vocab_size = vocab_size;
    out_tok->eos_id = 248044;
    out_tok->im_start_id = 248045;
    out_tok->im_end_id = 248046;

    const void *sec_ptr = NULL;
    size_t sec_size = 0;
    if (lila_asi_get_section(model, ASI_SEC_TOKENIZER, &sec_ptr, &sec_size) != 0 || !sec_ptr || sec_size == 0)
    {
        return -1;
    }

    const char *json = (const char *)sec_ptr;
    const char *end = json + sec_size;

    const char *tok_key = (const char *)asi_memmem(json, sec_size, "\"tokens\":[", 10);
    if (!tok_key) return -1;

    out_tok->raw_pieces = (char **)calloc(vocab_size, sizeof(char *));
    out_tok->decoded_pieces = (char **)calloc(vocab_size, sizeof(char *));
    if (!out_tok->raw_pieces || !out_tok->decoded_pieces) return -1;

    const char *p = tok_key + 10;
    uint32_t idx = 0;
    char buf[1024];
    char dec[1024];

    while (p < end && idx < vocab_size)
    {
        while (p < end && (*p == ' ' || *p == ',' || *p == '\n' || *p == '\r' || *p == '\t')) p++;
        if (p >= end) break;
        if (*p == ']') break;
        if (*p != '"')
        {
            p++;
            continue;
        }
        p++; /* skip opening quote */
        size_t bi = 0;
        while (p < end && *p != '"')
        {
            if (*p == '\\' && p + 1 < end)
            {
                p++;
                char esc = *p++;
                if (esc == 'n')      { if (bi + 1 < sizeof(buf)) buf[bi++] = '\n'; }
                else if (esc == 'r') { if (bi + 1 < sizeof(buf)) buf[bi++] = '\r'; }
                else if (esc == 't') { if (bi + 1 < sizeof(buf)) buf[bi++] = '\t'; }
                else if (esc == 'b') { if (bi + 1 < sizeof(buf)) buf[bi++] = '\b'; }
                else if (esc == 'f') { if (bi + 1 < sizeof(buf)) buf[bi++] = '\f'; }
                else if (esc == '"' || esc == '\\' || esc == '/')
                {
                    if (bi + 1 < sizeof(buf)) buf[bi++] = esc;
                }
                else if (esc == 'u' && p + 4 <= end)
                {
                    char hex[5] = { p[0], p[1], p[2], p[3], 0 };
                    uint32_t cp = (uint32_t)strtoul(hex, NULL, 16);
                    p += 4;
                    if (cp == 0)
                    {
                        /* Skip embedded NUL byte in GPT-2 byte-level vocabulary so C strings don't truncate */
                    }
                    else if (cp < 0x80)
                    {
                        if (bi + 1 < sizeof(buf)) buf[bi++] = (char)cp;
                    }
                    else if (cp < 0x800)
                    {
                        if (bi + 2 < sizeof(buf))
                        {
                            buf[bi++] = (char)(0xC0 | (cp >> 6));
                            buf[bi++] = (char)(0x80 | (cp & 0x3F));
                        }
                    }
                    else
                    {
                        if (bi + 3 < sizeof(buf))
                        {
                            buf[bi++] = (char)(0xE0 | (cp >> 12));
                            buf[bi++] = (char)(0x80 | ((cp >> 6) & 0x3F));
                            buf[bi++] = (char)(0x80 | (cp & 0x3F));
                        }
                    }
                }
            }
            else
            {
                if (bi + 1 < sizeof(buf)) buf[bi++] = *p;
                p++;
            }
        }
        if (p < end && *p == '"') p++;
        buf[bi] = '\0';
        decode_gpt2_piece(out_tok, buf, dec, sizeof(dec));
        out_tok->raw_pieces[idx] = strdup(buf);
        out_tok->decoded_pieces[idx] = strdup(dec);
        if (strcmp(buf, "<|endoftext|>") == 0) out_tok->eos_id = (int32_t)idx;
        else if (strcmp(buf, "<|im_start|>") == 0) out_tok->im_start_id = (int32_t)idx;
        else if (strcmp(buf, "<|im_end|>") == 0) out_tok->im_end_id = (int32_t)idx;
        idx++;
    }
    /* LILA_ASI_DEBUG_TOKENS=1 logs parsed vocabulary size from ASI_SEC_TOKENIZER. */
    if (getenv("LILA_ASI_DEBUG_TOKENS"))
    {
        fprintf(stderr, "loaded %u / %u vocab tokens\n", idx, vocab_size);
    }
    return 0;
}

static void free_simple_tokenizer(SimpleTokenizer *tok)
{
    if (!tok) return;
    if (tok->raw_pieces)
    {
        for (uint32_t i = 0; i < tok->vocab_size; ++i) free(tok->raw_pieces[i]);
        free(tok->raw_pieces);
    }
    if (tok->decoded_pieces)
    {
        for (uint32_t i = 0; i < tok->vocab_size; ++i) free(tok->decoded_pieces[i]);
        free(tok->decoded_pieces);
    }
}

/* Greedy longest-match tokenizer over decoded_pieces (restricted to piece length == 1 or no internal spaces across word boundaries) */
static uint32_t simple_encode(const SimpleTokenizer *tok,
                              const char *text,
                              int32_t *out_ids,
                              uint32_t max_ids)
{
    size_t len = strlen(text);
    size_t pos = 0;
    uint32_t count = 0;
    while (pos < len && count < max_ids)
    {
        /* Determine current pre-token chunk boundary (like GPT-2 regex: optional leading space + letters, or digits, or punct, or newline) */
        size_t chunk_end = pos + 1;
        if (text[pos] == '\n' || text[pos] == '\r')
        {
            chunk_end = pos + 1;
        }
        else
        {
            size_t p = pos;
            if (text[p] == ' ' && p + 1 < len && text[p + 1] != ' ' && text[p + 1] != '\n') p++;
            if (p < len && isalpha((unsigned char)text[p]))
            {
                while (p < len && isalpha((unsigned char)text[p])) p++;
                chunk_end = p;
            }
            else if (p < len && isdigit((unsigned char)text[p]))
            {
                while (p < len && isdigit((unsigned char)text[p])) p++;
                chunk_end = p;
            }
            else if (p < len && !isspace((unsigned char)text[p]))
            {
                while (p < len && !isspace((unsigned char)text[p]) && !isalnum((unsigned char)text[p])) p++;
                chunk_end = p;
            }
            else
            {
                chunk_end = pos + 1;
            }
        }
        size_t chunk_len = chunk_end - pos;

        while (pos < chunk_end && count < max_ids)
        {
            size_t rem_len = chunk_end - pos;
            size_t best_len = 0;
            int32_t best_id = -1;
            for (uint32_t i = 0; i < tok->vocab_size; ++i)
            {
                const char *piece = tok->decoded_pieces[i];
                if (!piece || !piece[0]) continue;
                size_t plen = strlen(piece);
                if (plen > best_len && plen <= rem_len && memcmp(text + pos, piece, plen) == 0)
                {
                    best_len = plen;
                    best_id = (int32_t)i;
                }
            }
            if (best_id >= 0 && best_len > 0)
            {
                out_ids[count++] = best_id;
                pos += best_len;
            }
            else
            {
                pos++;
            }
        }
        (void)chunk_len;
    }
    return count;
}

static int parse_hex_pubkey(const char *hex, uint8_t out_pk[32])
{
    if (!hex || strlen(hex) != 64) return -1;
    for (size_t i = 0; i < 32; ++i)
    {
        unsigned int v = 0;
        if (!isxdigit((unsigned char)hex[i * 2]) ||
            !isxdigit((unsigned char)hex[i * 2 + 1]) ||
            sscanf(hex + i * 2, "%2x", &v) != 1)
            return -1;
        out_pk[i] = (uint8_t)v;
    }
    return 0;
}

static void print_usage(const char *prog)
{
    printf("Lila ASI v2.6 Standalone C11 CLI (Sieve + E05 Grammar-Sparse + Anchor + Vision + Torrent)\n\n");
    printf("Usage:\n");
    printf("  %s <model.asi> --info\n", prog);
    printf("  %s <model.asi> --verify [--pubkey <64-hex>]\n", prog);
    printf("  %s <model.asi> --bench [-t threads] [-n tokens] [--mode fast|balanced|exact]\n", prog);
    printf("  %s <model.asi> -p \"<prompt>\" [-n max_tokens] [-t threads] [--temp T] [--mode fast|balanced|exact] [--raw-prompt]\n", prog);
    printf("  %s <model.asi> --tokens \"id1,id2,...\" [-n max_tokens] [-t threads] [--temp T]\n\n", prog);
    printf("Options:\n");
    printf("  --info                     Inspect .asi v2 cartridge header, architecture, and sections\n");
    printf("  --verify                   Verify Ed25519 signature and section integrity\n");
    printf("  --pubkey <64-hex>          Require signature from this 32-byte Ed25519 public key\n");
    printf("  --bench                    Run prefill + decode speed & confidence benchmark\n");
    printf("  -p, --prompt <text>        Prompt text to evaluate and generate from\n");
    printf("  --raw-prompt               Do not wrap prompt in ChatML (<|im_start|>user...<|im_start|>assistant)\n");
    printf("  --tokens <csv>             Comma-separated token IDs to evaluate directly\n");
    printf("  -n, --max-tokens <int>     Maximum tokens to generate (default: 128)\n");
    printf("  -t, --threads <int>        Worker threads (default: 2)\n");
    printf("  --ctx <int>                Max context length (default: 2048)\n");
    printf("  --temp <float>             Sampling temperature, <=0 for greedy (default: 0.0)\n");
    printf("  --mode <fast|balanced|exact> ASI preset mode (default: fast)\n");
}

static int cmd_verify(LilaAsiModel *model, const char *path, const char *pubkey_hex)
{
    uint8_t trusted_pk[32];
    const uint8_t *pk_ptr = NULL;
    if (pubkey_hex && pubkey_hex[0] != '\0')
    {
        if (parse_hex_pubkey(pubkey_hex, trusted_pk) != 0)
        {
            fprintf(stderr, "Invalid --pubkey (expected 64 hex chars).\n");
            return 1;
        }
        pk_ptr = trusted_pk;
    }
    AsiV2SignatureInfo info;
    char err_buf[256] = {0};
    if (lila_asi_verify_signature(model, pk_ptr, &info, err_buf, sizeof(err_buf)) != 0)
    {
        fprintf(stderr, "SIGNATURE VERIFY FAILED (%s): %s\n", path, err_buf);
        return 1;
    }
    printf("SIGNATURE VERIFIED: %s\n", path);
    printf("Publisher       : %s\n", info.publisher);
    printf("Signed At Unix  : %llu\n", (unsigned long long)info.signed_at_unix);
    printf("Public Key      : ");
    for (int i = 0; i < 32; ++i) printf("%02x", info.public_key[i]);
    printf("\n");
    return 0;
}

static int cmd_info(LilaAsiModel *model, const char *path, const char *mode)
{
    AsiV2Header hdr;
    AsiV2ModelConfig cfg;
    if (lila_asi_model_header(model, &hdr) != 0 || lila_asi_model_info(model, &cfg) != 0)
    {
        fprintf(stderr, "Failed to read ASI v2 header/config.\n");
        return 1;
    }

    char cpu_buf[128] = {0};
    lila_asi_cpu_info(cpu_buf, sizeof(cpu_buf));

    printf("=== ASI v2 Cartridge Inspection ===\n");
    printf("File            : %s\n", path);
    printf("Total Size      : %.2f MiB (%llu bytes)\n",
           (double)hdr.total_size / (1024.0 * 1024.0),
           (unsigned long long)hdr.total_size);
    printf("Format Version  : ASI v%u (magic=0x%08X, page=%u B)\n",
           hdr.version, hdr.magic, ASI_V2_PAGE_SIZE);
    printf("Runtime CPU     : %s\n", cpu_buf);
    printf("Architecture    : %s (arch_id=%u)\n", arch_id_name(cfg.arch_id), cfg.arch_id);
    printf("Active Preset   : %s\n", mode);
    printf("Identity SHA192 : ");
    for (int i = 0; i < 24; ++i) printf("%02x", hdr.identity_hash[i]);
    printf("\n");
    AsiV2SignatureInfo sinfo;
    char serr[128] = {0};
    if (lila_asi_verify_signature(model, NULL, &sinfo, serr, sizeof(serr)) == 0)
    {
        printf("Signature       : VERIFIED (publisher=%s, pubkey=", sinfo.publisher);
        for (int i = 0; i < 32; ++i) printf("%02x", sinfo.public_key[i]);
        printf(")\n\n");
    }
    else
    {
        printf("Signature       : %s\n\n", (hdr.flags & ASI_V2_FLAG_HAS_SIGNATURE) ? serr : "unsigned");
    }

    printf("--- Model Hyperparameters ---\n");
    printf("Layers          : %u\n", cfg.n_layers);
    printf("Hidden Size     : %u\n", cfg.hidden_size);
    printf("Intermediate    : %u\n", cfg.intermediate_size);
    printf("Attention Heads : %u (KV Heads: %u, Head Dim: %u)\n",
           cfg.n_heads, cfg.n_kv_heads, cfg.head_dim);
    printf("Vocab Size      : %u\n", cfg.vocab_size);
    printf("Max Seq Len     : %u\n", cfg.max_seq_len);
    printf("RoPE Dim/Theta  : %u / %.1f\n", cfg.rope_dim, cfg.rope_theta);
    if (cfg.arch_id == ASI_ARCH_HYBRID_QWEN35)
    {
        printf("Full-Attn Every : %u layers (SSM conv=%u, state=%u, groups=%u, inner=%u)\n",
               cfg.full_attention_interval, cfg.ssm_conv_kernel,
               cfg.ssm_state_size, cfg.ssm_group_count, cfg.ssm_inner_size);
    }
    printf("\n--- Cartridge Sections (%u) ---\n", hdr.n_sections);
    for (uint32_t st = 1; st <= 14; ++st)
    {
        const void *ptr = NULL;
        size_t sz = 0;
        if (lila_asi_get_section(model, st, &ptr, &sz) == 0 && ptr != NULL)
        {
            printf("  [0x%02X] %-16s : %10.2f KiB (%zu bytes)\n",
                   st, section_type_name(st), (double)sz / 1024.0, sz);
        }
    }
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 2 || strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)
    {
        print_usage(argv[0]);
        return 0;
    }

    const char *asi_path = argv[1];
    bool do_info = false;
    bool do_verify = false;
    bool do_bench = false;
    bool raw_prompt = false;
    const char *verify_pubkey = NULL;
    char prompt_buf[2048] = {0};
    const char *prompt_text = NULL;
    const char *tokens_csv = NULL;
    const char *mode = "fast";
    uint32_t max_tokens = 128;
    uint32_t n_threads = 2;
    uint32_t max_ctx = 2048;
    float temperature = 0.0f;

    for (int i = 2; i < argc; ++i)
    {
        if (strcmp(argv[i], "--info") == 0) do_info = true;
        else if (strcmp(argv[i], "--verify") == 0) do_verify = true;
        else if (strcmp(argv[i], "--pubkey") == 0 && i + 1 < argc) verify_pubkey = argv[++i];
        else if (strcmp(argv[i], "--bench") == 0) do_bench = true;
        else if (strcmp(argv[i], "--raw-prompt") == 0) raw_prompt = true;
        else if ((strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--prompt") == 0) && i + 1 < argc)
        {
            prompt_buf[0] = '\0';
            while (i + 1 < argc && argv[i + 1][0] != '-')
            {
                if (prompt_buf[0] != '\0') strncat(prompt_buf, " ", sizeof(prompt_buf) - strlen(prompt_buf) - 1);
                strncat(prompt_buf, argv[++i], sizeof(prompt_buf) - strlen(prompt_buf) - 1);
            }
            prompt_text = prompt_buf;
        }
        else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc)
            tokens_csv = argv[++i];
        else if ((strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--max-tokens") == 0) && i + 1 < argc)
            max_tokens = (uint32_t)atoi(argv[++i]);
        else if ((strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--threads") == 0) && i + 1 < argc)
            n_threads = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--ctx") == 0 && i + 1 < argc)
            max_ctx = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc)
            temperature = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc)
            mode = argv[++i];
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
        {
            print_usage(argv[0]);
            return 0;
        }
    }

    apply_mode_env(mode);

    char err_buf[256] = {0};
    double t_open0 = now_ms();
    LilaAsiModel *model = lila_asi_open(asi_path, err_buf, sizeof(err_buf));
    double t_open_ms = now_ms() - t_open0;
    if (!model)
    {
        fprintf(stderr, "Error opening %s: %s\n", asi_path, err_buf);
        return 1;
    }

    if (do_verify)
    {
        int rc = cmd_verify(model, asi_path, verify_pubkey);
        lila_asi_close(model);
        return rc;
    }

    if (do_info)
    {
        int rc = cmd_info(model, asi_path, mode);
        lila_asi_close(model);
        return rc;
    }

    AsiV2ModelConfig cfg;
    lila_asi_model_info(model, &cfg);

    double t_sess0 = now_ms();
    LilaAsiSession *session = lila_asi_session_create(model, max_ctx, n_threads, err_buf, sizeof(err_buf));
    double t_sess_ms = now_ms() - t_sess0;
    if (!session)
    {
        fprintf(stderr, "Error creating session: %s\n", err_buf);
        lila_asi_close(model);
        return 1;
    }

    float *logits = (float *)malloc((size_t)cfg.vocab_size * sizeof(float));
    if (!logits)
    {
        lila_asi_session_free(session);
        lila_asi_close(model);
        return 1;
    }

    if (do_bench)
    {
        int32_t bench_prompt[32];
        for (int i = 0; i < 32; ++i) bench_prompt[i] = 100 + i;

        double t_pf0 = now_ms();
        if (lila_asi_eval_batch(session, bench_prompt, 32, 0, logits) != 0)
        {
            fprintf(stderr, "Benchmark prefill failed.\n");
            free(logits);
            lila_asi_session_free(session);
            lila_asi_close(model);
            return 1;
        }
        double t_pf_ms = now_ms() - t_pf0;

        uint32_t cp_id = 0;
        uint64_t rng = 0x123456789ABCDEF0ULL;
        lila_asi_cp_fork(session, 32, rng, "bench_prefill", &cp_id);

        uint32_t n_dec = (max_tokens > 0 && max_tokens != 128) ? max_tokens : 32;
        double margin_sum = 0.0;
        double entropy_sum = 0.0;

        double t_dec0 = now_ms();
        for (uint32_t step = 0; step < n_dec; ++step)
        {
            AsiSampleStats st;
            int32_t next_id = lila_asi_sample_ex(logits, cfg.vocab_size, 0.0f, 64, 1.0f, &rng, &st);
            margin_sum += st.margin;
            entropy_sum += st.entropy;
            if (lila_asi_eval_token(session, next_id, 32 + step, logits) != 0) break;
        }
        double t_dec_ms = now_ms() - t_dec0;

        printf("=== ASI v2.6 Native Benchmark (Sieve + Anchor + Torrent) ===\n");
        printf("Cartridge       : %s\n", asi_path);
        printf("Preset Mode     : %s (threads=%u, ctx=%u)\n", mode, n_threads, max_ctx);
        printf("Open + Sieve    : %.1f ms (mmap=%.1f ms, session=%.1f ms)\n",
               t_open_ms + t_sess_ms, t_open_ms, t_sess_ms);
        printf("Prefill (32 tok): %.2f ms (%.2f tok/s)\n",
               t_pf_ms, 32000.0 / t_pf_ms);
        printf("Decode  (%2u tok): %.2f ms (%.2f ms/tok, %.2f tok/s)\n",
               n_dec, t_dec_ms, t_dec_ms / n_dec, (double)n_dec * 1000.0 / t_dec_ms);
        printf("E06 Confidence  : mean_margin=%.2f, mean_entropy=%.3f nats, cp_slot=%u\n",
               margin_sum / n_dec, entropy_sum / n_dec, cp_id);

        free(logits);
        lila_asi_session_free(session);
        lila_asi_close(model);
        return 0;
    }

    SimpleTokenizer tok;
    bool has_tok = (load_simple_tokenizer(model, cfg.vocab_size, &tok) == 0);

    int32_t prompt_ids[2048];
    uint32_t n_prompt = 0;

    if (tokens_csv)
    {
        char *copy = (char *)malloc(strlen(tokens_csv) + 1);
        if (copy) strcpy(copy, tokens_csv);
        char *save = NULL;
        char *part = copy ? asi_strtok_r(copy, ",", &save) : NULL;
        while (part && n_prompt < 2048)
        {
            prompt_ids[n_prompt++] = (int32_t)atoi(part);
            part = asi_strtok_r(NULL, ",", &save);
        }
        free(copy);
    }
    else if (prompt_text && has_tok)
    {
        if (raw_prompt)
        {
            n_prompt = simple_encode(&tok, prompt_text, prompt_ids, 2040);
        }
        else
        {
            /* Wrap in ChatML <|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n */
            prompt_ids[n_prompt++] = tok.im_start_id;
            n_prompt += simple_encode(&tok, "user\n", prompt_ids + n_prompt, 16);
            n_prompt += simple_encode(&tok, prompt_text, prompt_ids + n_prompt, 1980);
            prompt_ids[n_prompt++] = tok.im_end_id;
            n_prompt += simple_encode(&tok, "\n", prompt_ids + n_prompt, 8);
            prompt_ids[n_prompt++] = tok.im_start_id;
            n_prompt += simple_encode(&tok, "assistant\n", prompt_ids + n_prompt, 16);
            /* LILA_ASI_DEBUG_TOKENS=1 logs parsed vocabulary size from ASI_SEC_TOKENIZER. */
    if (getenv("LILA_ASI_DEBUG_TOKENS"))
            {
                fprintf(stderr, "prompt_text='%s'\n", prompt_text);
                for (uint32_t ti = 0; ti < n_prompt; ++ti)
                    fprintf(stderr, "tok[%u]=%d (%s)\n", ti, prompt_ids[ti],
                            (uint32_t)prompt_ids[ti] < tok.vocab_size ? tok.decoded_pieces[prompt_ids[ti]] : "?");
            }
        }
    }
    else
    {
        print_usage(argv[0]);
        if (has_tok) free_simple_tokenizer(&tok);
        free(logits);
        lila_asi_session_free(session);
        lila_asi_close(model);
        return 1;
    }

    if (n_prompt == 0)
    {
        fprintf(stderr, "Empty prompt tokens.\n");
        if (has_tok) free_simple_tokenizer(&tok);
        free(logits);
        lila_asi_session_free(session);
        lila_asi_close(model);
        return 1;
    }

    double t_pf0 = now_ms();
    if (lila_asi_eval_batch(session, prompt_ids, n_prompt, 0, logits) != 0)
    {
        fprintf(stderr, "Prefill failed.\n");
        if (has_tok) free_simple_tokenizer(&tok);
        free(logits);
        lila_asi_session_free(session);
        lila_asi_close(model);
        return 1;
    }
    double t_pf_ms = now_ms() - t_pf0;

    uint64_t rng = 0x9E3779B97F4A7C15ULL;
    uint32_t pos = n_prompt;
    uint32_t n_gen = 0;
    double t_dec0 = now_ms();

    for (uint32_t step = 0; step < max_tokens; ++step)
    {
        AsiSampleStats st;
        int32_t next_id = lila_asi_sample_ex(logits, cfg.vocab_size, temperature, 64, 0.95f, &rng, &st);
        if (next_id < 0) break;
        if (has_tok && (next_id == tok.eos_id || next_id == tok.im_end_id))
        {
            break;
        }
        if (has_tok && (uint32_t)next_id < tok.vocab_size && tok.decoded_pieces[next_id])
        {
            fputs(tok.decoded_pieces[next_id], stdout);
            fflush(stdout);
        }
        else
        {
            printf("[%d]", next_id);
            fflush(stdout);
        }
        n_gen++;
        if (step + 1 < max_tokens)
        {
            if (lila_asi_eval_token(session, next_id, pos++, logits) != 0) break;
        }
    }
    double t_dec_ms = now_ms() - t_dec0;
    printf("\n\n[ASI v2.6 | mode=%s | prefill=%u tok in %.1f ms (%.1f tok/s) | decode=%u tok in %.1f ms (%.2f tok/s)]\n",
           mode,
           n_prompt, t_pf_ms, (double)n_prompt * 1000.0 / (t_pf_ms > 0.0 ? t_pf_ms : 1.0),
           n_gen, t_dec_ms, (double)n_gen * 1000.0 / (t_dec_ms > 0.0 ? t_dec_ms : 1.0));

    if (has_tok) free_simple_tokenizer(&tok);
    free(logits);
    lila_asi_session_free(session);
    lila_asi_close(model);
    return 0;
}
