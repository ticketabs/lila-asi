#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "asi_grammar_sparse.h"
#include "asi_platform.h"
#include "q4k.h"

#include <ctype.h>
#include <math.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

static inline double gs_now_ms(void) {
    return (double)asi_monotonic_ns() * 1e-6;
}

/* ============================================================================
 * 1. GPT-2 Byte-Level BPE Decoder & Pure C11 ASI_SEC_TOKENIZER JSON Parser
 * ============================================================================ */

static void build_gpt2_unicode_to_byte(int16_t u2b[512]) {
    for (int i = 0; i < 512; ++i) u2b[i] = -1;
    uint8_t is_direct[256];
    memset(is_direct, 0, sizeof(is_direct));
    for (int b = 33; b <= 126; ++b) { is_direct[b] = 1; u2b[b] = (int16_t)b; }
    for (int b = 161; b <= 172; ++b) { is_direct[b] = 1; u2b[b] = (int16_t)b; }
    for (int b = 174; b <= 255; ++b) { is_direct[b] = 1; u2b[b] = (int16_t)b; }
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (!is_direct[b]) {
            u2b[256 + n] = (int16_t)b;
            ++n;
        }
    }
}

static inline int hex_val(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
    if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
    return -1;
}

static size_t append_utf8(char *dst, size_t cap, size_t len, uint32_t cp) {
    if (cp < 0x80u) {
        if (len + 1u < cap) dst[len] = (char)cp;
        return len + 1u;
    } else if (cp < 0x800u) {
        if (len + 2u < cap) {
            dst[len]     = (char)(0xC0u | (cp >> 6));
            dst[len + 1] = (char)(0x80u | (cp & 0x3Fu));
        }
        return len + 2u;
    } else if (cp < 0x10000u) {
        if (len + 3u < cap) {
            dst[len]     = (char)(0xE0u | (cp >> 12));
            dst[len + 1] = (char)(0x80u | ((cp >> 6) & 0x3Fu));
            dst[len + 2] = (char)(0x80u | (cp & 0x3Fu));
        }
        return len + 3u;
    } else {
        if (len + 4u < cap) {
            dst[len]     = (char)(0xF0u | (cp >> 18));
            dst[len + 1] = (char)(0x80u | ((cp >> 12) & 0x3Fu));
            dst[len + 2] = (char)(0x80u | ((cp >> 6) & 0x3Fu));
            dst[len + 3] = (char)(0x80u | (cp & 0x3Fu));
        }
        return len + 4u;
    }
}

/* Parse a JSON string starting at *p (which must point to '"').
 * Writes unescaped UTF-8 bytes into out_buf (NUL-terminated) and advances *p past closing '"'. */
static int parse_json_string(const char **pp, const char *end,
                             char *out_buf, size_t out_cap, size_t *out_len) {
    const char *p = *pp;
    if (p >= end || *p != '"') return -1;
    ++p;
    size_t len = 0;
    while (p < end) {
        char c = *p++;
        if (c == '"') {
            if (len < out_cap) out_buf[len] = '\0';
            else if (out_cap > 0) out_buf[out_cap - 1] = '\0';
            *out_len = (len < out_cap) ? len : (out_cap > 0 ? out_cap - 1 : 0);
            *pp = p;
            return 0;
        }
        if (c == '\\') {
            if (p >= end) return -1;
            char esc = *p++;
            switch (esc) {
                case '"':  if (len + 1 < out_cap) out_buf[len] = '"';  ++len; break;
                case '\\': if (len + 1 < out_cap) out_buf[len] = '\\'; ++len; break;
                case '/':  if (len + 1 < out_cap) out_buf[len] = '/';  ++len; break;
                case 'b':  if (len + 1 < out_cap) out_buf[len] = '\b'; ++len; break;
                case 'f':  if (len + 1 < out_cap) out_buf[len] = '\f'; ++len; break;
                case 'n':  if (len + 1 < out_cap) out_buf[len] = '\n'; ++len; break;
                case 'r':  if (len + 1 < out_cap) out_buf[len] = '\r'; ++len; break;
                case 't':  if (len + 1 < out_cap) out_buf[len] = '\t'; ++len; break;
                case 'u': {
                    if (p + 4 > end) return -1;
                    int h0 = hex_val(p[0]), h1 = hex_val(p[1]),
                        h2 = hex_val(p[2]), h3 = hex_val(p[3]);
                    if (h0 < 0 || h1 < 0 || h2 < 0 || h3 < 0) return -1;
                    uint32_t cp = (uint32_t)((h0 << 12) | (h1 << 8) | (h2 << 4) | h3);
                    p += 4;
                    if (cp >= 0xD800u && cp <= 0xDBFFu && p + 6 <= end &&
                        p[0] == '\\' && p[1] == 'u') {
                        int l0 = hex_val(p[2]), l1 = hex_val(p[3]),
                            l2 = hex_val(p[4]), l3 = hex_val(p[5]);
                        if (l0 >= 0 && l1 >= 0 && l2 >= 0 && l3 >= 0) {
                            uint32_t lo = (uint32_t)((l0 << 12) | (l1 << 8) | (l2 << 4) | l3);
                            if (lo >= 0xDC00u && lo <= 0xDFFFu) {
                                cp = 0x10000u + (((cp - 0xD800u) << 10) | (lo - 0xDC00u));
                                p += 6;
                            }
                        }
                    }
                    len = append_utf8(out_buf, out_cap, len, cp);
                    break;
                }
                default:
                    if (len + 1 < out_cap) out_buf[len] = esc;
                    ++len;
                    break;
            }
        } else {
            if (len + 1 < out_cap) out_buf[len] = c;
            ++len;
        }
    }
    return -1;
}

/* Decode GPT-2 byte-level BPE UTF-8 string into raw bytes. */
static size_t decode_gpt2_bpe_bytes(const char *src, size_t src_len,
                                    const int16_t u2b[512],
                                    char *dst, size_t dst_cap) {
    size_t i = 0, out = 0;
    while (i < src_len) {
        uint8_t b0 = (uint8_t)src[i];
        uint32_t cp = 0;
        size_t adv = 1;
        if (b0 < 0x80u) {
            cp = b0;
            adv = 1;
        } else if ((b0 & 0xE0u) == 0xC0u && i + 1 < src_len) {
            cp = ((uint32_t)(b0 & 0x1Fu) << 6) | (uint32_t)((uint8_t)src[i + 1] & 0x3Fu);
            adv = 2;
        } else if ((b0 & 0xF0u) == 0xE0u && i + 2 < src_len) {
            cp = ((uint32_t)(b0 & 0x0Fu) << 12) |
                 ((uint32_t)((uint8_t)src[i + 1] & 0x3Fu) << 6) |
                 (uint32_t)((uint8_t)src[i + 2] & 0x3Fu);
            adv = 3;
        } else if ((b0 & 0xF8u) == 0xF0u && i + 3 < src_len) {
            cp = ((uint32_t)(b0 & 0x07u) << 18) |
                 ((uint32_t)((uint8_t)src[i + 1] & 0x3Fu) << 12) |
                 ((uint32_t)((uint8_t)src[i + 2] & 0x3Fu) << 6) |
                 (uint32_t)((uint8_t)src[i + 3] & 0x3Fu);
            adv = 4;
        } else {
            cp = b0;
            adv = 1;
        }
        if (cp < 512u && u2b[cp] >= 0) {
            if (out + 1 < dst_cap) dst[out] = (char)(uint8_t)u2b[cp];
            ++out;
        } else {
            for (size_t k = 0; k < adv; ++k) {
                if (out + 1 < dst_cap) dst[out] = src[i + k];
                ++out;
            }
        }
        i += adv;
    }
    if (out < dst_cap) dst[out] = '\0';
    else if (dst_cap > 0) dst[dst_cap - 1] = '\0';
    return (out < dst_cap) ? out : (dst_cap > 0 ? dst_cap - 1 : 0);
}

static const char *find_json_key(const char *buf, size_t size, const char *key) {
    char pat[128];
    int n = snprintf(pat, sizeof(pat), "\"%s\"", key);
    if (n <= 0 || (size_t)n >= sizeof(pat)) return NULL;
    const char *end = buf + size;
    const char *p = buf;
    while (p < end) {
        const char *hit = (const char *)asi_memmem(p, (size_t)(end - p), pat, (size_t)n);
        if (!hit) return NULL;
        const char *q = hit + n;
        while (q < end && (*q == ' ' || *q == '\t' || *q == '\r' || *q == '\n')) ++q;
        if (q < end && *q == ':') {
            ++q;
            while (q < end && (*q == ' ' || *q == '\t' || *q == '\r' || *q == '\n')) ++q;
            return q;
        }
        p = hit + 1;
    }
    return NULL;
}

int asi_grammar_vocab_from_arrays(AsiGrammarVocab *out_vocab,
                                  uint32_t vocab_size,
                                  const char *const *tokens,
                                  const uint32_t *token_lens) {
    if (!out_vocab || !tokens || vocab_size == 0u) return -1;
    memset(out_vocab, 0, sizeof(*out_vocab));
    out_vocab->vocab_size = vocab_size;
    out_vocab->eos_token_id = -1;
    out_vocab->im_start_id = -1;
    out_vocab->im_end_id = -1;
    out_vocab->tool_call_open_id = -1;
    out_vocab->tool_call_close_id = -1;

    out_vocab->tokens = (char **)calloc(vocab_size, sizeof(char *));
    out_vocab->token_lens = (uint32_t *)calloc(vocab_size, sizeof(uint32_t));
    if (!out_vocab->tokens || !out_vocab->token_lens) {
        asi_grammar_vocab_free(out_vocab);
        return -1;
    }

    for (uint32_t i = 0; i < vocab_size; ++i) {
        if (!tokens[i]) continue;
        uint32_t len = token_lens ? token_lens[i] : (uint32_t)strlen(tokens[i]);
        char *copy = (char *)malloc((size_t)len + 1u);
        if (!copy) {
            asi_grammar_vocab_free(out_vocab);
            return -1;
        }
        if (len > 0u) memcpy(copy, tokens[i], len);
        copy[len] = '\0';
        out_vocab->tokens[i] = copy;
        out_vocab->token_lens[i] = len;

        if (len == 11u && memcmp(copy, "<tool_call>", 11u) == 0)
            out_vocab->tool_call_open_id = (int32_t)i;
        else if (len == 12u && memcmp(copy, "</tool_call>", 12u) == 0)
            out_vocab->tool_call_close_id = (int32_t)i;
        else if (len == 12u && memcmp(copy, "<|im_start|>", 12u) == 0)
            out_vocab->im_start_id = (int32_t)i;
        else if (len == 10u && memcmp(copy, "<|im_end|>", 10u) == 0)
            out_vocab->im_end_id = (int32_t)i;
    }
    return 0;
}

int asi_grammar_vocab_from_asi_json(AsiGrammarVocab *out_vocab,
                                    const void *sec_ptr,
                                    size_t sec_size) {
    if (!out_vocab || !sec_ptr || sec_size == 0u) return -1;
    memset(out_vocab, 0, sizeof(*out_vocab));
    out_vocab->eos_token_id = -1;
    out_vocab->im_start_id = -1;
    out_vocab->im_end_id = -1;
    out_vocab->tool_call_open_id = -1;
    out_vocab->tool_call_close_id = -1;

    const char *buf = (const char *)sec_ptr;
    const char *end = buf + sec_size;

    const char *eos_p = find_json_key(buf, sec_size, "eos_token_id");
    if (eos_p && *eos_p >= '0' && *eos_p <= '9') {
        out_vocab->eos_token_id = (int32_t)strtol(eos_p, NULL, 10);
    }

    const char *tok_p = find_json_key(buf, sec_size, "tokens");
    if (!tok_p || *tok_p != '[') return -1;

    /* Pass 1: parse tokens into growable arrays */
    uint32_t cap = 4096u;
    uint32_t n_toks = 0u;
    char **raw_toks = (char **)calloc(cap, sizeof(char *));
    uint32_t *raw_lens = (uint32_t *)calloc(cap, sizeof(uint32_t));
    if (!raw_toks || !raw_lens) {
        free(raw_toks);
        free(raw_lens);
        return -1;
    }

    char tmp[1024];
    const char *p = tok_p + 1;
    while (p < end) {
        while (p < end && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n' || *p == ',')) ++p;
        if (p >= end) { free(raw_toks); free(raw_lens); return -1; }
        if (*p == ']') break;
        size_t slen = 0;
        if (parse_json_string(&p, end, tmp, sizeof(tmp), &slen) != 0) {
            for (uint32_t i = 0; i < n_toks; ++i) free(raw_toks[i]);
            free(raw_toks);
            free(raw_lens);
            return -1;
        }
        if (n_toks == cap) {
            uint32_t new_cap = cap * 2u;
            char **nt = (char **)realloc(raw_toks, (size_t)new_cap * sizeof(char *));
            uint32_t *nl = (uint32_t *)realloc(raw_lens, (size_t)new_cap * sizeof(uint32_t));
            if (!nt || !nl) {
                free(nt ? nt : raw_toks);
                free(nl ? nl : raw_lens);
                return -1;
            }
            raw_toks = nt;
            raw_lens = nl;
            cap = new_cap;
        }
        char *copy = (char *)malloc(slen + 1u);
        if (!copy) {
            for (uint32_t i = 0; i < n_toks; ++i) free(raw_toks[i]);
            free(raw_toks);
            free(raw_lens);
            return -1;
        }
        if (slen > 0u) memcpy(copy, tmp, slen);
        copy[slen] = '\0';
        raw_toks[n_toks] = copy;
        raw_lens[n_toks] = (uint32_t)slen;
        ++n_toks;
    }

    /* Optional token_type array */
    uint8_t *ttypes = (uint8_t *)malloc(n_toks);
    if (!ttypes) {
        for (uint32_t i = 0; i < n_toks; ++i) free(raw_toks[i]);
        free(raw_toks);
        free(raw_lens);
        return -1;
    }
    memset(ttypes, 1, n_toks);

    const char *tt_p = find_json_key(buf, sec_size, "token_type");
    if (tt_p && *tt_p == '[') {
        const char *q = tt_p + 1;
        uint32_t idx = 0;
        while (q < end && idx < n_toks) {
            while (q < end && (*q == ' ' || *q == '\t' || *q == '\r' || *q == '\n' || *q == ',')) ++q;
            if (q >= end || *q == ']') break;
            char *next_q = NULL;
            long v = strtol(q, &next_q, 10);
            if (next_q == q) break;
            ttypes[idx++] = (uint8_t)v;
            q = next_q;
        }
    }

    int16_t u2b[512];
    build_gpt2_unicode_to_byte(u2b);

    out_vocab->vocab_size = n_toks;
    out_vocab->tokens = (char **)calloc(n_toks, sizeof(char *));
    out_vocab->token_lens = (uint32_t *)calloc(n_toks, sizeof(uint32_t));
    if (!out_vocab->tokens || !out_vocab->token_lens) {
        for (uint32_t i = 0; i < n_toks; ++i) free(raw_toks[i]);
        free(raw_toks);
        free(raw_lens);
        free(ttypes);
        asi_grammar_vocab_free(out_vocab);
        return -1;
    }

    char dec_buf[1024];
    for (uint32_t i = 0; i < n_toks; ++i) {
        const char *s = raw_toks[i];
        uint32_t sl = raw_lens[i];
        uint8_t tt = ttypes[i];

        if (sl == 12u && memcmp(s, "<|im_start|>", 12u) == 0) out_vocab->im_start_id = (int32_t)i;
        else if (sl == 10u && memcmp(s, "<|im_end|>", 10u) == 0) out_vocab->im_end_id = (int32_t)i;
        else if (sl == 11u && memcmp(s, "<tool_call>", 11u) == 0) out_vocab->tool_call_open_id = (int32_t)i;
        else if (sl == 12u && memcmp(s, "</tool_call>", 12u) == 0) out_vocab->tool_call_close_id = (int32_t)i;

        if (tt == 3u || tt == 5u) {
            /* Control or unused token: disable inside JSON grammar */
            free(raw_toks[i]);
            continue;
        }
        if (tt == 4u) {
            /* User-defined special token (e.g. <tool_call>, </tool_call>): keep literal */
            out_vocab->tokens[i] = raw_toks[i];
            out_vocab->token_lens[i] = sl;
            continue;
        }
        /* Normal GPT-2 byte-level BPE token */
        size_t dlen = decode_gpt2_bpe_bytes(s, sl, u2b, dec_buf, sizeof(dec_buf));
        free(raw_toks[i]);
        char *copy = (char *)malloc(dlen + 1u);
        if (!copy) {
            for (uint32_t j = i + 1u; j < n_toks; ++j) free(raw_toks[j]);
            free(raw_toks);
            free(raw_lens);
            free(ttypes);
            asi_grammar_vocab_free(out_vocab);
            return -1;
        }
        if (dlen > 0u) memcpy(copy, dec_buf, dlen);
        copy[dlen] = '\0';
        out_vocab->tokens[i] = copy;
        out_vocab->token_lens[i] = (uint32_t)dlen;
    }

    free(raw_toks);
    free(raw_lens);
    free(ttypes);
    return 0;
}

void asi_grammar_vocab_free(AsiGrammarVocab *vocab) {
    if (!vocab) return;
    if (vocab->tokens) {
        for (uint32_t i = 0; i < vocab->vocab_size; ++i) {
            free(vocab->tokens[i]);
        }
        free(vocab->tokens);
    }
    free(vocab->token_lens);
    memset(vocab, 0, sizeof(*vocab));
}

/* ============================================================================
 * 2. Compact Character-Level DFA (Full 0..255 byte flat transition table)
 * ============================================================================ */

typedef struct {
    int32_t  next[256];
    uint8_t  out_chars[256];
    uint16_t out_count;
    uint8_t  accepting;
} DfaState;

typedef struct {
    DfaState *states;
    uint32_t n_states;
    uint32_t cap_states;
    int32_t  start_state;
} CharDfa;

static int dfa_init(CharDfa *dfa) {
    memset(dfa, 0, sizeof(*dfa));
    dfa->cap_states = 256u;
    dfa->states = (DfaState *)malloc((size_t)dfa->cap_states * sizeof(DfaState));
    if (!dfa->states) return -1;
    dfa->n_states = 1u;
    dfa->start_state = 0;
    for (int c = 0; c < 256; ++c) dfa->states[0].next[c] = -1;
    dfa->states[0].out_count = 0;
    dfa->states[0].accepting = 0;
    return 0;
}

static void dfa_free(CharDfa *dfa) {
    free(dfa->states);
    memset(dfa, 0, sizeof(*dfa));
}

static int32_t dfa_new_state(CharDfa *dfa, uint8_t accepting) {
    if (dfa->n_states == dfa->cap_states) {
        uint32_t new_cap = dfa->cap_states * 2u;
        DfaState *ns = (DfaState *)realloc(dfa->states, (size_t)new_cap * sizeof(DfaState));
        if (!ns) return -1;
        dfa->states = ns;
        dfa->cap_states = new_cap;
    }
    int32_t sid = (int32_t)dfa->n_states++;
    DfaState *st = &dfa->states[sid];
    for (int c = 0; c < 256; ++c) st->next[c] = -1;
    st->out_count = 0;
    st->accepting = accepting;
    return sid;
}

static int dfa_add_edge(CharDfa *dfa, int32_t src, uint8_t ch, int32_t dst) {
    if (src < 0 || dst < 0) return -1;
    DfaState *st = &dfa->states[src];
    if (st->next[ch] >= 0) {
        return (st->next[ch] == dst) ? 0 : -1;
    }
    st->next[ch] = dst;
    st->out_chars[st->out_count++] = ch;
    return 0;
}

static int32_t dfa_add_literal_chain(CharDfa *dfa, int32_t src, const char *lit) {
    int32_t cur = src;
    for (const char *p = lit; *p; ++p) {
        uint8_t ch = (uint8_t)*p;
        int32_t nxt = dfa->states[cur].next[ch];
        if (nxt < 0) {
            nxt = dfa_new_state(dfa, 0);
            if (nxt < 0) return -1;
            if (dfa_add_edge(dfa, cur, ch, nxt) != 0) return -1;
        }
        cur = nxt;
    }
    return cur;
}

static int dfa_add_trie(CharDfa *dfa, int32_t src,
                        const char *const *strings, uint32_t n_strings,
                        int32_t target_end, int32_t *out_ends) {
    for (uint32_t i = 0; i < n_strings; ++i) {
        const char *s = strings[i];
        size_t slen = strlen(s);
        if (slen == 0u) return -1;
        int32_t cur = src;
        for (size_t k = 0; k < slen; ++k) {
            uint8_t ch = (uint8_t)s[k];
            int is_last = (k + 1u == slen);
            int32_t nxt = dfa->states[cur].next[ch];
            if (nxt < 0) {
                nxt = (is_last && target_end >= 0) ? target_end : dfa_new_state(dfa, 0);
                if (nxt < 0) return -1;
                if (dfa_add_edge(dfa, cur, ch, nxt) != 0) return -1;
            } else if (is_last && target_end >= 0 && nxt != target_end) {
                return -1;
            }
            cur = nxt;
        }
        if (out_ends) out_ends[i] = cur;
    }
    return 0;
}

static inline int32_t dfa_advance_bytes(const CharDfa *dfa, int32_t state,
                                        const char *bytes, uint32_t len) {
    int32_t cur = state;
    const DfaState *states = dfa->states;
    for (uint32_t i = 0; i < len; ++i) {
        uint8_t ch = (uint8_t)bytes[i];
        cur = states[cur].next[ch];
        if (cur < 0) return -1;
    }
    return cur;
}

static uint32_t dfa_forced_prefix(const CharDfa *dfa, int32_t state,
                                  char *out_buf, uint32_t max_len) {
    uint32_t len = 0u;
    int32_t cur = state;
    while (len < max_len) {
        const DfaState *st = &dfa->states[cur];
        if (st->accepting || st->out_count != 1u) break;
        uint8_t ch = st->out_chars[0];
        out_buf[len++] = (char)ch;
        cur = st->next[ch];
        if (cur < 0) break;
    }
    if (len < max_len) out_buf[len] = '\0';
    return len;
}

/* Safe printable ASCII characters inside JSON strings (RFC 8259: 0x20..0x7E except '"' and '\\') */
static const char kSafeJsonStrChars[] =
    " !#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`abcdefghijklmnopqrstuvwxyz{|}~";

typedef struct {
    int32_t states[32];
    uint32_t count;
} ExitList;

static int attach_literal_exits(CharDfa *dfa, ExitList *exits, const char *lit) {
    size_t len = strlen(lit);
    if (len == 0u || exits->count == 0u) return -1;
    int32_t merged = dfa_new_state(dfa, 0);
    if (merged < 0) return -1;
    for (uint32_t i = 0; i < exits->count; ++i) {
        if (dfa_add_edge(dfa, exits->states[i], (uint8_t)lit[0], merged) != 0) return -1;
    }
    int32_t end = (len == 1u) ? merged : dfa_add_literal_chain(dfa, merged, lit + 1);
    if (end < 0) return -1;
    exits->states[0] = end;
    exits->count = 1u;
    return 0;
}

static int add_hex_edges(CharDfa *dfa, int32_t src, int32_t dst) {
    static const char kHexDigits[] = "0123456789abcdefABCDEF";
    for (const char *p = kHexDigits; *p; ++p) {
        if (dfa_add_edge(dfa, src, (uint8_t)*p, dst) != 0) return -1;
    }
    return 0;
}

static int compile_string_arg(CharDfa *dfa, int32_t src, int32_t end, uint32_t max_len) {
    /* max_len == 0u -> unbounded RFC-8259 JSON string with escapes (\" \\ \/ \b \f \n \r \t \uXXXX),
     * empty string "", and full UTF-8 bytes (0x20..0xFF). */
    if (max_len == 0u) {
        int32_t body = dfa_new_state(dfa, 0);
        int32_t esc  = dfa_new_state(dfa, 0);
        int32_t u1   = dfa_new_state(dfa, 0);
        int32_t u2   = dfa_new_state(dfa, 0);
        int32_t u3   = dfa_new_state(dfa, 0);
        if (body < 0 || esc < 0 || u1 < 0 || u2 < 0 || u3 < 0) return -1;

        if (dfa_add_edge(dfa, src, (uint8_t)'"', body) != 0) return -1;
        /* Closing quote (also supports empty string "") */
        if (dfa_add_edge(dfa, body, (uint8_t)'"', end) != 0) return -1;
        /* Escape prefix '\\' */
        if (dfa_add_edge(dfa, body, (uint8_t)'\\', esc) != 0) return -1;
        /* Unescaped printable ASCII + UTF-8 continuation/lead bytes (0x20..0xFF except '"' and '\\') */
        for (uint32_t b = 0x20u; b <= 0xFFu; ++b) {
            if (b == (uint32_t)'"' || b == (uint32_t)'\\') continue;
            if (dfa_add_edge(dfa, body, (uint8_t)b, body) != 0) return -1;
        }
        /* Standard 1-char JSON escapes */
        static const char kEscChars[] = "\"\\/bfnrt";
        for (const char *p = kEscChars; *p; ++p) {
            if (dfa_add_edge(dfa, esc, (uint8_t)*p, body) != 0) return -1;
        }
        /* \uXXXX 4-hex-digit escape */
        if (dfa_add_edge(dfa, esc, (uint8_t)'u', u1) != 0) return -1;
        if (add_hex_edges(dfa, u1, u2) != 0) return -1;
        if (add_hex_edges(dfa, u2, u3) != 0) return -1;
        if (add_hex_edges(dfa, u3, body) != 0) return -1;
        return 0;
    }

    /* Bounded string chain (1..max_len chars) for deterministic termination tests */
    if (max_len > 128u) max_len = 128u;
    int32_t open_q = dfa_new_state(dfa, 0);
    if (open_q < 0) return -1;
    if (dfa_add_edge(dfa, src, (uint8_t)'"', open_q) != 0) return -1;
    int32_t cur = open_q;
    for (uint32_t step = 0; step < max_len; ++step) {
        int32_t nxt = dfa_new_state(dfa, 0);
        int32_t esc = dfa_new_state(dfa, 0);
        if (nxt < 0 || esc < 0) return -1;
        for (const char *p = kSafeJsonStrChars; *p; ++p) {
            if (dfa_add_edge(dfa, cur, (uint8_t)*p, nxt) != 0) return -1;
        }
        if (dfa_add_edge(dfa, cur, (uint8_t)'\\', esc) != 0) return -1;
        static const char kEscChars[] = "\"\\/bfnrt";
        for (const char *p = kEscChars; *p; ++p) {
            if (dfa_add_edge(dfa, esc, (uint8_t)*p, nxt) != 0) return -1;
        }
        if (step >= 1u) {
            if (dfa_add_edge(dfa, cur, (uint8_t)'"', end) != 0) return -1;
        }
        cur = nxt;
    }
    return dfa_add_edge(dfa, cur, (uint8_t)'"', end);
}

static int compile_int_exits(CharDfa *dfa, int32_t src, uint32_t max_digits, ExitList *out) {
    if (max_digits == 0u) max_digits = 6u;
    if (max_digits > 10u) max_digits = 10u;
    out->count = 0u;

    int32_t zero_st = dfa_new_state(dfa, 0);
    if (zero_st < 0) return -1;
    if (dfa_add_edge(dfa, src, (uint8_t)'0', zero_st) != 0) return -1;
    out->states[out->count++] = zero_st;

    int32_t d1 = dfa_new_state(dfa, 0);
    if (d1 < 0) return -1;
    for (char c = '1'; c <= '9'; ++c) {
        if (dfa_add_edge(dfa, src, (uint8_t)c, d1) != 0) return -1;
    }
    /* Optional leading '-' -> must be followed by 1..9 (or 0 for -0.x in float, here 1..9) */
    int32_t neg_st = dfa_new_state(dfa, 0);
    if (neg_st < 0) return -1;
    if (dfa_add_edge(dfa, src, (uint8_t)'-', neg_st) != 0) return -1;
    for (char c = '1'; c <= '9'; ++c) {
        if (dfa_add_edge(dfa, neg_st, (uint8_t)c, d1) != 0) return -1;
    }

    out->states[out->count++] = d1;
    int32_t cur = d1;
    for (uint32_t i = 1u; i < max_digits; ++i) {
        int32_t nxt = dfa_new_state(dfa, 0);
        if (nxt < 0) return -1;
        for (char c = '0'; c <= '9'; ++c) {
            if (dfa_add_edge(dfa, cur, (uint8_t)c, nxt) != 0) return -1;
        }
        out->states[out->count++] = nxt;
        cur = nxt;
    }
    return 0;
}

static int compile_arg_value(CharDfa *dfa, int32_t src, const AsiToolArgSchema *arg,
                             uint32_t def_max_str, uint32_t def_max_digits,
                             ExitList *out) {
    out->count = 0u;
    switch (arg->type) {
        case ASI_ARG_ENUM: {
            if (arg->n_enum_values == 0u || arg->n_enum_values > ASI_GRAMMAR_MAX_ENUM_VALS)
                return -1;
            int32_t end = dfa_new_state(dfa, 0);
            if (end < 0) return -1;
            char quoted_buf[ASI_GRAMMAR_MAX_ENUM_VALS][128];
            const char *quoted_ptrs[ASI_GRAMMAR_MAX_ENUM_VALS];
            for (uint32_t i = 0; i < arg->n_enum_values; ++i) {
                int n = snprintf(quoted_buf[i], sizeof(quoted_buf[i]), "\"%s\"",
                                 arg->enum_values[i]);
                if (n <= 0 || (size_t)n >= sizeof(quoted_buf[i])) return -1;
                quoted_ptrs[i] = quoted_buf[i];
            }
            if (dfa_add_trie(dfa, src, quoted_ptrs, arg->n_enum_values, end, NULL) != 0)
                return -1;
            out->states[0] = end;
            out->count = 1u;
            return 0;
        }
        case ASI_ARG_BOOL: {
            int32_t end = dfa_new_state(dfa, 0);
            if (end < 0) return -1;
            const char *bools[2] = {"true", "false"};
            if (dfa_add_trie(dfa, src, bools, 2u, end, NULL) != 0) return -1;
            out->states[0] = end;
            out->count = 1u;
            return 0;
        }
        case ASI_ARG_STRING: {
            int32_t end = dfa_new_state(dfa, 0);
            if (end < 0) return -1;
            uint32_t ml = arg->max_len ? arg->max_len : def_max_str;
            if (compile_string_arg(dfa, src, end, ml) != 0) return -1;
            out->states[0] = end;
            out->count = 1u;
            return 0;
        }
        case ASI_ARG_INT: {
            uint32_t md = arg->max_digits ? arg->max_digits : def_max_digits;
            return compile_int_exits(dfa, src, md, out);
        }
        case ASI_ARG_FLOAT: {
            uint32_t md = arg->max_digits ? arg->max_digits : def_max_digits;
            if (compile_int_exits(dfa, src, md, out) != 0) return -1;
            int32_t dot_st = dfa_new_state(dfa, 0);
            if (dot_st < 0) return -1;
            for (uint32_t i = 0; i < out->count; ++i) {
                if (dfa_add_edge(dfa, out->states[i], (uint8_t)'.', dot_st) != 0) return -1;
            }
            int32_t cur = dot_st;
            for (int f = 0; f < 4; ++f) {
                int32_t nxt = dfa_new_state(dfa, 0);
                if (nxt < 0) return -1;
                for (char c = '0'; c <= '9'; ++c) {
                    if (dfa_add_edge(dfa, cur, (uint8_t)c, nxt) != 0) return -1;
                }
                if (out->count < 32u) out->states[out->count++] = nxt;
                cur = nxt;
            }
            return 0;
        }
    }
    return -1;
}

static int attach_exits_to_target(CharDfa *dfa, const ExitList *exits, const char *lit, int32_t target) {
    size_t len = strlen(lit);
    if (len == 0u || exits->count == 0u || target < 0) return -1;
    if (len == 1u) {
        for (uint32_t i = 0; i < exits->count; ++i) {
            if (dfa_add_edge(dfa, exits->states[i], (uint8_t)lit[0], target) != 0) return -1;
        }
        return 0;
    }
    int32_t merged = dfa_new_state(dfa, 0);
    if (merged < 0) return -1;
    for (uint32_t i = 0; i < exits->count; ++i) {
        if (dfa_add_edge(dfa, exits->states[i], (uint8_t)lit[0], merged) != 0) return -1;
    }
    int32_t cur = merged;
    for (size_t c = 1u; c < len; ++c) {
        if (c + 1u == len) {
            if (dfa_add_edge(dfa, cur, (uint8_t)lit[c], target) != 0) return -1;
        } else {
            int32_t nxt = dfa_new_state(dfa, 0);
            if (nxt < 0) return -1;
            if (dfa_add_edge(dfa, cur, (uint8_t)lit[c], nxt) != 0) return -1;
            cur = nxt;
        }
    }
    return 0;
}

static int compile_tool_schemas(CharDfa *dfa,
                                const AsiToolSchema *tools,
                                uint32_t n_tools,
                                const AsiGrammarConfig *cfg) {
    if (!tools || n_tools == 0u || n_tools > ASI_GRAMMAR_MAX_TOOLS) return -1;
    if (dfa_init(dfa) != 0) return -1;

    int wrap_xml = cfg ? cfg->wrap_xml : 1;
    uint32_t def_str = cfg ? cfg->default_max_str : 0u;
    uint32_t def_dig = (cfg && cfg->default_max_digits) ? cfg->default_max_digits : 6u;

    const char *prefix = wrap_xml ? "<tool_call>\n{\"name\": \"" : "{\"name\": \"";
    const char *suffix = wrap_xml ? "}}\n</tool_call>" : "}}";

    int32_t after_name_open = dfa_add_literal_chain(dfa, dfa->start_state, prefix);
    if (after_name_open < 0) return -1;

    char (*quoted_names)[128] = (char (*)[128])calloc(n_tools, sizeof(*quoted_names));
    const char **qname_ptrs = (const char **)calloc(n_tools, sizeof(const char *));
    int32_t *name_ends = (int32_t *)calloc(n_tools, sizeof(int32_t));
    if (!quoted_names || !qname_ptrs || !name_ends) {
        free(quoted_names);
        free(qname_ptrs);
        free(name_ends);
        return -1;
    }

    int rc = -1;
    for (uint32_t i = 0; i < n_tools; ++i) {
        if (!tools[i].name || !tools[i].name[0]) goto cleanup;
        int n = snprintf(quoted_names[i], sizeof(quoted_names[i]), "%s\"", tools[i].name);
        if (n <= 0 || (size_t)n >= sizeof(quoted_names[i])) goto cleanup;
        qname_ptrs[i] = quoted_names[i];
    }
    if (dfa_add_trie(dfa, after_name_open, qname_ptrs, n_tools, -1, name_ends) != 0)
        goto cleanup;

    int32_t final_accept = dfa_new_state(dfa, 1);
    if (final_accept < 0) goto cleanup;

    for (uint32_t t = 0; t < n_tools; ++t) {
        uint32_t n_args = tools[t].n_args;
        if (n_args > ASI_GRAMMAR_MAX_ARGS) goto cleanup;

        ExitList open_exits;
        open_exits.states[0] = name_ends[t];
        open_exits.count = 1u;
        if (attach_literal_exits(dfa, &open_exits, ", \"arguments\": {") != 0) goto cleanup;
        int32_t open_brace_st = open_exits.states[0];

        if (n_args == 0u) {
            if (attach_exits_to_target(dfa, &open_exits, suffix, final_accept) != 0)
                goto cleanup;
            continue;
        }

        /* Compile each argument's value DFA once from a dedicated entry state */
        int32_t val_entry[ASI_GRAMMAR_MAX_ARGS];
        ExitList val_exits[ASI_GRAMMAR_MAX_ARGS];
        const char *key_ptrs[ASI_GRAMMAR_MAX_ARGS];
        char key_lits[ASI_GRAMMAR_MAX_ARGS][128];

        for (uint32_t a = 0; a < n_args; ++a) {
            if (!tools[t].args[a].name || !tools[t].args[a].name[0]) goto cleanup;
            int n = snprintf(key_lits[a], sizeof(key_lits[a]), "\"%s\": ", tools[t].args[a].name);
            if (n <= 0 || (size_t)n >= sizeof(key_lits[a])) goto cleanup;
            key_ptrs[a] = key_lits[a];

            val_entry[a] = dfa_new_state(dfa, 0);
            if (val_entry[a] < 0) goto cleanup;
            if (compile_arg_value(dfa, val_entry[a], &tools[t].args[a],
                                  def_str, def_dig, &val_exits[a]) != 0)
                goto cleanup;
        }

        /* Route from open_brace_st (before any arg) to legal first args [0 .. r0] */
        uint32_t r0 = n_args;
        for (uint32_t j = 0; j < n_args; ++j) {
            if (!tools[t].args[j].optional) { r0 = j; break; }
        }
        uint32_t max_first = (r0 < n_args) ? r0 : (n_args - 1u);
        for (uint32_t j = 0; j <= max_first; ++j) {
            int32_t key_end = dfa_add_literal_chain(dfa, open_brace_st, key_ptrs[j]);
            if (key_end < 0) goto cleanup;
            /* Merge key_end into val_entry[j] by copying val_entry[j]'s outgoing edges */
            const DfaState *vest = &dfa->states[val_entry[j]];
            for (uint32_t k = 0; k < vest->out_count; ++k) {
                uint8_t ch = vest->out_chars[k];
                if (dfa_add_edge(dfa, key_end, ch, vest->next[ch]) != 0) goto cleanup;
            }
        }
        if (r0 == n_args) {
            /* All arguments are optional: allow empty arguments "{}" */
            if (attach_exits_to_target(dfa, &open_exits, suffix, final_accept) != 0)
                goto cleanup;
        }

        /* Route from each argument a's value exits to legal next args [a+1 .. r_next] or closing suffix */
        for (uint32_t a = 0; a < n_args; ++a) {
            uint32_t r_next = n_args;
            for (uint32_t j = a + 1u; j < n_args; ++j) {
                if (!tools[t].args[j].optional) { r_next = j; break; }
            }
            if (a + 1u < n_args) {
                ExitList comma_exits = val_exits[a];
                if (attach_literal_exits(dfa, &comma_exits, ", ") != 0) goto cleanup;
                int32_t after_comma = comma_exits.states[0];
                uint32_t max_next = (r_next < n_args) ? r_next : (n_args - 1u);
                for (uint32_t j = a + 1u; j <= max_next; ++j) {
                    int32_t key_end = dfa_add_literal_chain(dfa, after_comma, key_ptrs[j]);
                    if (key_end < 0) goto cleanup;
                    const DfaState *vest = &dfa->states[val_entry[j]];
                    for (uint32_t k = 0; k < vest->out_count; ++k) {
                        uint8_t ch = vest->out_chars[k];
                        if (dfa_add_edge(dfa, key_end, ch, vest->next[ch]) != 0) goto cleanup;
                    }
                }
            }
            if (r_next == n_args) {
                if (attach_exits_to_target(dfa, &val_exits[a], suffix, final_accept) != 0)
                    goto cleanup;
            }
        }
    }

    rc = 0;
cleanup:
    free(quoted_names);
    free(qname_ptrs);
    free(name_ends);
    return rc;
}

/* ============================================================================
 * 3. Vocabulary Hash Map (for Greedy BPE Encoding) & First-Byte Buckets
 * ============================================================================ */

typedef struct {
    int32_t token_id; /* -1 = empty */
    uint64_t hash;
} VocabHashEntry;

typedef struct {
    int32_t *ids;
    uint32_t count;
    uint8_t  populated;
} LegalCacheEntry;

struct AsiGrammarEngine {
    const AsiGrammarVocab *vocab;
    CharDfa dfa;
    int32_t state;
    uint32_t sparse_limit;
    AsiGrammarTelemetry telemetry;

    /* First-byte vocabulary buckets (full 0..255 byte range for ASCII + UTF-8 tokens) */
    int32_t *bucket_data;
    uint32_t bucket_off[257];

    /* Hash map: token bytes -> token_id (for greedy longest-match forced prefixes) */
    VocabHashEntry *htab;
    uint32_t htab_mask;
    uint32_t max_tok_len;

    /* Per-DFA-state cached sorted legal token IDs */
    LegalCacheEntry *legal_cache;

    /* Scratch bitset for O(V) masking in Case 3 */
    uint8_t *legal_mask_scratch;
};

static inline uint64_t fnv1a_bytes(const char *s, uint32_t len) {
    uint64_t h = 14695981039346656037ULL;
    for (uint32_t i = 0; i < len; ++i) {
        h ^= (uint64_t)(uint8_t)s[i];
        h *= 1099511628211ULL;
    }
    return h ? h : 1ULL;
}

static int32_t vocab_htab_lookup(const VocabHashEntry *htab, uint32_t mask,
                                 const AsiGrammarVocab *vocab,
                                 const char *s, uint32_t len) {
    uint64_t h = fnv1a_bytes(s, len);
    uint32_t slot = (uint32_t)h & mask;
    for (;;) {
        int32_t tid = htab[slot].token_id;
        if (tid < 0) return -1;
        if (htab[slot].hash == h && vocab->token_lens[tid] == len &&
            memcmp(vocab->tokens[tid], s, len) == 0) {
            return tid;
        }
        slot = (slot + 1u) & mask;
    }
}

static void vocab_htab_insert(VocabHashEntry *htab, uint32_t mask,
                              const AsiGrammarVocab *vocab, int32_t tid) {
    const char *s = vocab->tokens[tid];
    uint32_t len = vocab->token_lens[tid];
    uint64_t h = fnv1a_bytes(s, len);
    uint32_t slot = (uint32_t)h & mask;
    for (;;) {
        int32_t cur = htab[slot].token_id;
        if (cur < 0) {
            htab[slot].token_id = tid;
            htab[slot].hash = h;
            return;
        }
        if (htab[slot].hash == h && vocab->token_lens[cur] == len &&
            memcmp(vocab->tokens[cur], s, len) == 0) {
            /* Prefer special <tool_call> / </tool_call> over split/duplicate, else keep lower ID */
            if (tid == vocab->tool_call_open_id || tid == vocab->tool_call_close_id) {
                htab[slot].token_id = tid;
            }
            return;
        }
        slot = (slot + 1u) & mask;
    }
}

int asi_grammar_vocab_greedy_encode(const AsiGrammarVocab *vocab,
                                    const char *text,
                                    size_t text_len,
                                    int32_t *out_ids,
                                    uint32_t max_ids) {
    if (!vocab || !text || !out_ids) return -1;
    uint32_t cap = 1024u;
    while (cap < vocab->vocab_size * 2u) cap <<= 1u;
    VocabHashEntry *htab = (VocabHashEntry *)malloc((size_t)cap * sizeof(VocabHashEntry));
    if (!htab) return -1;
    for (uint32_t i = 0; i < cap; ++i) { htab[i].token_id = -1; htab[i].hash = 0; }
    uint32_t mask = cap - 1u;
    uint32_t max_tlen = 1u;
    for (uint32_t i = 0; i < vocab->vocab_size; ++i) {
        if (!vocab->tokens[i] || vocab->token_lens[i] == 0u) continue;
        if (vocab->token_lens[i] > 64u) continue;
        if (vocab->token_lens[i] > max_tlen) max_tlen = vocab->token_lens[i];
        vocab_htab_insert(htab, mask, vocab, (int32_t)i);
    }

    uint32_t n_out = 0u;
    size_t pos = 0u;
    while (pos < text_len) {
        if (n_out >= max_ids) { free(htab); return -1; }
        size_t rem = text_len - pos;
        uint32_t try_len = (rem < (size_t)max_tlen) ? (uint32_t)rem : max_tlen;
        int32_t found = -1;
        for (uint32_t l = try_len; l >= 1u; --l) {
            found = vocab_htab_lookup(htab, mask, vocab, text + pos, l);
            if (found >= 0) {
                out_ids[n_out++] = found;
                pos += l;
                break;
            }
        }
        if (found < 0) { free(htab); return -1; }
    }
    free(htab);
    return (int)n_out;
}

static int cmp_i32_asc(const void *a, const void *b) {
    int32_t va = *(const int32_t *)a;
    int32_t vb = *(const int32_t *)b;
    return (va > vb) - (va < vb);
}

AsiGrammarEngine *asi_grammar_create(const AsiGrammarVocab *vocab,
                                     const AsiToolSchema *tools,
                                     uint32_t n_tools,
                                     const AsiGrammarConfig *cfg) {
    if (!vocab || !vocab->tokens || vocab->vocab_size == 0u || !tools || n_tools == 0u)
        return NULL;

    AsiGrammarEngine *g = (AsiGrammarEngine *)calloc(1, sizeof(AsiGrammarEngine));
    if (!g) return NULL;
    g->vocab = vocab;
    g->sparse_limit = (cfg && cfg->sparse_limit > 0u)
                          ? cfg->sparse_limit
                          : ASI_GRAMMAR_DEFAULT_SPARSE_LIMIT;

    if (compile_tool_schemas(&g->dfa, tools, n_tools, cfg) != 0) {
        asi_grammar_free(g);
        return NULL;
    }
    g->state = g->dfa.start_state;

    /* Build first-byte buckets for all non-empty tokens (ASCII + UTF-8) */
    uint32_t counts[256];
    memset(counts, 0, sizeof(counts));
    uint32_t tok_total = 0u;
    for (uint32_t i = 0; i < vocab->vocab_size; ++i) {
        const char *s = vocab->tokens[i];
        uint32_t len = vocab->token_lens[i];
        if (!s || len == 0u) continue;
        counts[(uint8_t)s[0]]++;
        tok_total++;
    }

    g->bucket_data = (int32_t *)malloc((size_t)(tok_total > 0u ? tok_total : 1u) * sizeof(int32_t));
    if (!g->bucket_data) {
        asi_grammar_free(g);
        return NULL;
    }
    uint32_t cur_off = 0u;
    for (int b = 0; b < 256; ++b) {
        g->bucket_off[b] = cur_off;
        cur_off += counts[b];
    }
    g->bucket_off[256] = cur_off;

    uint32_t write_pos[256];
    memcpy(write_pos, g->bucket_off, 256u * sizeof(uint32_t));
    for (uint32_t i = 0; i < vocab->vocab_size; ++i) {
        const char *s = vocab->tokens[i];
        uint32_t len = vocab->token_lens[i];
        if (!s || len == 0u) continue;
        uint8_t b0 = (uint8_t)s[0];
        g->bucket_data[write_pos[b0]++] = (int32_t)i;
    }

    /* Build hash map for fast greedy tokenization of forced prefixes */
    uint32_t hcap = 1024u;
    while (hcap < tok_total * 2u + 64u) hcap <<= 1u;
    g->htab = (VocabHashEntry *)malloc((size_t)hcap * sizeof(VocabHashEntry));
    if (!g->htab) {
        asi_grammar_free(g);
        return NULL;
    }
    for (uint32_t i = 0; i < hcap; ++i) { g->htab[i].token_id = -1; g->htab[i].hash = 0; }
    g->htab_mask = hcap - 1u;
    g->max_tok_len = 1u;
    for (uint32_t idx = 0; idx < tok_total; ++idx) {
        int32_t tid = g->bucket_data[idx];
        uint32_t len = vocab->token_lens[tid];
        if (len > 64u) continue;
        if (len > g->max_tok_len) g->max_tok_len = len;
        vocab_htab_insert(g->htab, g->htab_mask, vocab, tid);
    }

    g->legal_cache = (LegalCacheEntry *)calloc(g->dfa.n_states, sizeof(LegalCacheEntry));
    g->legal_mask_scratch = (uint8_t *)calloc(vocab->vocab_size, sizeof(uint8_t));
    if (!g->legal_cache || !g->legal_mask_scratch) {
        asi_grammar_free(g);
        return NULL;
    }

    return g;
}

void asi_grammar_free(AsiGrammarEngine *g) {
    if (!g) return;
    if (g->legal_cache) {
        for (uint32_t i = 0; i < g->dfa.n_states; ++i) {
            free(g->legal_cache[i].ids);
        }
        free(g->legal_cache);
    }
    free(g->legal_mask_scratch);
    free(g->htab);
    free(g->bucket_data);
    dfa_free(&g->dfa);
    free(g);
}

void asi_grammar_reset(AsiGrammarEngine *g) {
    if (!g) return;
    g->state = g->dfa.start_state;
    memset(&g->telemetry, 0, sizeof(g->telemetry));
}

int asi_grammar_is_complete(const AsiGrammarEngine *g) {
    if (!g || g->state < 0 || (uint32_t)g->state >= g->dfa.n_states) return 0;
    return (int)g->dfa.states[g->state].accepting;
}

int32_t asi_grammar_state(const AsiGrammarEngine *g) {
    return g ? g->state : -1;
}

uint32_t asi_grammar_num_states(const AsiGrammarEngine *g) {
    return g ? g->dfa.n_states : 0u;
}

uint32_t asi_grammar_sparse_limit(const AsiGrammarEngine *g) {
    return g ? g->sparse_limit : 0u;
}

void asi_grammar_get_telemetry(const AsiGrammarEngine *g, AsiGrammarTelemetry *out) {
    if (!g || !out) return;
    *out = g->telemetry;
}

/* ============================================================================
 * 4. Incremental Legal Token IDs & Forced-Token Fast-Forward
 * ============================================================================ */

int asi_grammar_get_legal_ids(AsiGrammarEngine *g,
                              const int32_t **out_legal_ids,
                              uint32_t *out_n_legal) {
    if (!g || !out_legal_ids || !out_n_legal) return -1;
    if (g->state < 0 || (uint32_t)g->state >= g->dfa.n_states) return -1;

    LegalCacheEntry *ce = &g->legal_cache[g->state];
    if (ce->populated) {
        *out_legal_ids = ce->ids;
        *out_n_legal = ce->count;
        return 0;
    }

    double t0 = gs_now_ms();
    const DfaState *st = &g->dfa.states[g->state];
    const AsiGrammarVocab *vocab = g->vocab;

    /* Upper bound on candidates from active first-byte buckets */
    uint32_t cand_bound = 0u;
    for (uint32_t k = 0; k < (uint32_t)st->out_count; ++k) {
        uint8_t b = st->out_chars[k];
        cand_bound += (g->bucket_off[b + 1u] - g->bucket_off[b]);
    }

    int32_t *tmp = NULL;
    uint32_t count = 0u;
    if (cand_bound > 0u) {
        tmp = (int32_t *)malloc((size_t)cand_bound * sizeof(int32_t));
        if (!tmp) return -1;
        for (uint32_t k = 0; k < (uint32_t)st->out_count; ++k) {
            uint8_t b = st->out_chars[k];
            uint32_t b_start = g->bucket_off[b];
            uint32_t b_end = g->bucket_off[b + 1u];
            for (uint32_t idx = b_start; idx < b_end; ++idx) {
                int32_t tid = g->bucket_data[idx];
                if (dfa_advance_bytes(&g->dfa, g->state,
                                      vocab->tokens[tid], vocab->token_lens[tid]) >= 0) {
                    tmp[count++] = tid;
                }
            }
        }
        /* Sort ascending for DRAM-sequential Q6_K row access */
        if (count > 1u) {
            qsort(tmp, count, sizeof(int32_t), cmp_i32_asc);
        }
    }

    if (count > 0u) {
        int32_t *shrunk = (int32_t *)realloc(tmp, (size_t)count * sizeof(int32_t));
        ce->ids = shrunk ? shrunk : tmp;
    } else {
        free(tmp);
        ce->ids = NULL;
    }
    ce->count = count;
    ce->populated = 1u;
    g->telemetry.mask_ms_sum += (gs_now_ms() - t0);

    *out_legal_ids = ce->ids;
    *out_n_legal = ce->count;
    return 0;
}

int asi_grammar_collect_forced_run(AsiGrammarEngine *g,
                                   int32_t *out_token_ids,
                                   uint32_t max_tokens,
                                   AsiForcedReason *out_reason) {
    if (!g || !out_token_ids || max_tokens == 0u) return -1;
    if (out_reason) *out_reason = ASI_FORCED_NONE;
    if (asi_grammar_is_complete(g)) {
        if (out_reason) *out_reason = ASI_FORCED_COMPLETE;
        return 0;
    }

    uint32_t collected = 0u;
    AsiForcedReason first_reason = ASI_FORCED_NONE;

    while (!asi_grammar_is_complete(g) && collected < max_tokens) {
        /* 1. Check deterministic forced literal prefix */
        char prefix[256];
        uint32_t plen = dfa_forced_prefix(&g->dfa, g->state, prefix, sizeof(prefix) - 1u);
        if (plen > 0u) {
            /* Greedily tokenize the forced prefix using longest-match in g->htab */
            uint32_t pos = 0u;
            int progress = 0;
            while (pos < plen && collected < max_tokens) {
                uint32_t rem = plen - pos;
                uint32_t try_len = (rem < g->max_tok_len) ? rem : g->max_tok_len;
                int32_t matched = -1;
                uint32_t matched_len = 0u;
                for (uint32_t l = try_len; l >= 1u; --l) {
                    int32_t tid = vocab_htab_lookup(g->htab, g->htab_mask,
                                                    g->vocab, prefix + pos, l);
                    if (tid >= 0) {
                        int32_t nxt = dfa_advance_bytes(&g->dfa, g->state,
                                                        g->vocab->tokens[tid],
                                                        g->vocab->token_lens[tid]);
                        if (nxt >= 0) {
                            matched = tid;
                            matched_len = l;
                            g->state = nxt;
                            break;
                        }
                    }
                }
                if (matched < 0) break;
                out_token_ids[collected++] = matched;
                g->telemetry.total_tokens++;
                g->telemetry.fast_forwarded_tokens++;
                g->telemetry.forced_prefix_tokens++;
                if (first_reason == ASI_FORCED_NONE) first_reason = ASI_FORCED_PREFIX;
                pos += matched_len;
                progress = 1;
            }
            if (progress) continue;
        }

        /* 2. Check if only 1 BPE token is legal at the current state */
        const int32_t *legal_ids = NULL;
        uint32_t n_legal = 0u;
        if (asi_grammar_get_legal_ids(g, &legal_ids, &n_legal) != 0) return -1;
        if (n_legal == 1u) {
            int32_t tid = legal_ids[0];
            int32_t nxt = dfa_advance_bytes(&g->dfa, g->state,
                                            g->vocab->tokens[tid],
                                            g->vocab->token_lens[tid]);
            if (nxt < 0) return -1;
            g->state = nxt;
            out_token_ids[collected++] = tid;
            g->telemetry.total_tokens++;
            g->telemetry.fast_forwarded_tokens++;
            g->telemetry.forced_single_tokens++;
            if (first_reason == ASI_FORCED_NONE) first_reason = ASI_FORCED_SINGLE_TOKEN;
            continue;
        }

        /* Reached a true branching choice point (n_legal > 1) */
        break;
    }

    if (collected > 0u) {
        g->telemetry.forced_runs++;
        if (out_reason) *out_reason = first_reason;
    }
    return (int)collected;
}

int asi_grammar_accept_token(AsiGrammarEngine *g, int32_t token_id) {
    if (!g || token_id < 0 || (uint32_t)token_id >= g->vocab->vocab_size) return -1;
    const char *s = g->vocab->tokens[token_id];
    uint32_t len = g->vocab->token_lens[token_id];
    if (!s || len == 0u) return -1;
    int32_t nxt = dfa_advance_bytes(&g->dfa, g->state, s, len);
    if (nxt < 0) return -1;
    g->state = nxt;
    g->telemetry.total_tokens++;
    g->telemetry.lm_head_calls++;
    return 0;
}

int asi_grammar_mask_logits(AsiGrammarEngine *g, float *logits, uint32_t vocab_size) {
    if (!g || !logits || vocab_size != g->vocab->vocab_size) return -1;
    const int32_t *legal_ids = NULL;
    uint32_t n_legal = 0u;
    if (asi_grammar_get_legal_ids(g, &legal_ids, &n_legal) != 0) return -1;
    if (n_legal == 0u) return -2;

    uint8_t *mask = g->legal_mask_scratch;
    memset(mask, 0, vocab_size);
    for (uint32_t i = 0; i < n_legal; ++i) {
        mask[legal_ids[i]] = 1u;
    }
    for (uint32_t i = 0; i < vocab_size; ++i) {
        if (!mask[i]) logits[i] = ASI_GRAMMAR_NEG_INF_FILL;
    }
    return 0;
}

/* ============================================================================
 * 5. Case 2: Grammar-Aware Sparse Q6_K lm_head & Case 3 Unified Dispatch
 * ============================================================================ */

static void gs_fill_neg_inf(float *dst, uint32_t n) {
    uint32_t i = 0u;
#if defined(__AVX2__)
    __m256 vneg = _mm256_set1_ps(ASI_GRAMMAR_NEG_INF_FILL);
    for (; i + 8u <= n; i += 8u) {
        _mm256_storeu_ps(dst + i, vneg);
    }
#endif
    for (; i < n; ++i) dst[i] = ASI_GRAMMAR_NEG_INF_FILL;
}

static inline void gs_prefetch_q6k_row(const uint8_t *rp, size_t row_bytes) {
    LILA_PREFETCH(rp);
    if (row_bytes > 256u) LILA_PREFETCH(rp + 256u);
    if (row_bytes > 512u) LILA_PREFETCH(rp + 512u);
    if (row_bytes > 768u) LILA_PREFETCH(rp + 768u);
}

typedef struct {
    const uint8_t *q6k_weights;
    const LilaQ8KBlock *qx;
    const float *x_norm;
    const int32_t *legal_ids;
    float *out_logits;
    uint32_t n_legal;
    int hidden_size;
    int bpr;
    size_t row_bytes;
    uint32_t quant_type;
    int is_repacked;
    int fp32_ref;
    _Alignas(64) atomic_uint next_idx;
} SparseQ6kCtx;

static void sparse_q6k_worker(int tid, int n_threads, void *vctx) {
    (void)tid;
    SparseQ6kCtx *ctx = (SparseQ6kCtx *)vctx;
    const uint8_t *w = ctx->q6k_weights;
    const int32_t *ids = ctx->legal_ids;
    float *out = ctx->out_logits;
    const size_t rb = ctx->row_bytes;
    const int bpr = ctx->bpr;
    const uint32_t cnt = ctx->n_legal;

    for (;;) {
        uint32_t start = (n_threads <= 1)
                             ? 0u
                             : atomic_fetch_add_explicit(&ctx->next_idx, 32u, memory_order_relaxed);
        if (start >= cnt) break;
        uint32_t end = (n_threads <= 1) ? cnt : (start + 32u);
        if (end > cnt) end = cnt;

        if (!ctx->fp32_ref && ctx->quant_type == 14u && !ctx->is_repacked) {
            uint32_t k = start;
#if defined(__AVX2__)
            for (; k + 1u < end; k += 2u) {
                int32_t id0 = ids[k];
                int32_t id1 = ids[k + 1u];
                const uint8_t *rp0 = w + (size_t)id0 * rb;
                const uint8_t *rp1 = w + (size_t)id1 * rb;
                if (k + 3u < end) {
                    gs_prefetch_q6k_row(w + (size_t)ids[k + 2u] * rb, rb);
                    gs_prefetch_q6k_row(w + (size_t)ids[k + 3u] * rb, rb);
                }
                float v0 = 0.0f, v1 = 0.0f;
                q6k_dot2_q8k_avx2(rp0, rp1, ctx->qx, bpr, &v0, &v1);
                out[id0] = v0;
                out[id1] = v1;
            }
#endif
            for (; k < end; ++k) {
                int32_t id = ids[k];
                if (k + 2u < end) {
                    gs_prefetch_q6k_row(w + (size_t)ids[k + 2u] * rb, rb);
                }
                out[id] = q6k_dot_q8k_simd(w + (size_t)id * rb, ctx->qx, bpr);
            }
        } else if (!ctx->fp32_ref && ctx->quant_type == 12u) {
            uint32_t k = start;
#if defined(__AVX2__)
            for (; k + 1u < end; k += 2u) {
                int32_t id0 = ids[k];
                int32_t id1 = ids[k + 1u];
                const uint8_t *rp0 = w + (size_t)id0 * rb;
                const uint8_t *rp1 = w + (size_t)id1 * rb;
                if (k + 3u < end) {
                    gs_prefetch_q6k_row(w + (size_t)ids[k + 2u] * rb, rb);
                    gs_prefetch_q6k_row(w + (size_t)ids[k + 3u] * rb, rb);
                }
                float v0 = 0.0f, v1 = 0.0f;
                q4k_dot2_q8k_avx2(rp0, rp1, ctx->qx, bpr, &v0, &v1);
                out[id0] = v0;
                out[id1] = v1;
            }
#endif
            for (; k < end; ++k) {
                int32_t id = ids[k];
                if (k + 2u < end) {
                    gs_prefetch_q6k_row(w + (size_t)ids[k + 2u] * rb, rb);
                }
                out[id] = q4k_dot_q8k_simd(w + (size_t)id * rb, ctx->qx, bpr);
            }
        } else {
            for (uint32_t k = start; k < end; ++k) {
                int32_t id = ids[k];
                if (k + 2u < end) {
                    gs_prefetch_q6k_row(w + (size_t)ids[k + 2u] * rb, rb);
                }
                out[id] = lila_quant_dot_fp32(w + (size_t)id * rb, ctx->x_norm,
                                              ctx->hidden_size, ctx->quant_type, ctx->is_repacked);
            }
        }
        if (n_threads <= 1) break;
    }
}

int asi_grammar_sparse_lm_head_quant(const uint8_t *weights,
                                     uint32_t quant_type,
                                     int is_repacked,
                                     uint32_t vocab_size,
                                     uint32_t hidden_size,
                                     const float *x_norm,
                                     const int32_t *legal_ids,
                                     uint32_t n_legal,
                                     float *out_logits,
                                     int fp32_ref,
                                     LilaLhcDispatchFn dispatch_fn,
                                     void *pool_opaque) {
    if (!weights || !x_norm || !out_logits || vocab_size == 0u || hidden_size == 0u)
        return -1;
    if (!lila_quant_is_supported(quant_type)) return -1;
    if (n_legal > 0u && !legal_ids) return -1;

    size_t rb = lila_quant_row_bytes(quant_type, (int)hidden_size, is_repacked ? 1 : 0);
    if (rb == 0u) return -1;

    gs_fill_neg_inf(out_logits, vocab_size);
    if (n_legal == 0u) return 0;

    int bpr = ((int)hidden_size + QK_K - 1) / QK_K;
    int need_q8k = (!fp32_ref && (quant_type == 12u || (quant_type == 14u && !is_repacked)));

    LilaQ8KBlock qx_stack[32];
    LilaQ8KBlock *qx = qx_stack;
    if (need_q8k) {
        if (bpr > 32) {
            qx = (LilaQ8KBlock *)malloc((size_t)bpr * sizeof(LilaQ8KBlock));
            if (!qx) return -1;
        }
        quantize_vec_q8k(qx, x_norm, (int)hidden_size);
    }

    SparseQ6kCtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.q6k_weights = weights;
    ctx.qx = need_q8k ? qx : NULL;
    ctx.x_norm = x_norm;
    ctx.legal_ids = legal_ids;
    ctx.out_logits = out_logits;
    ctx.n_legal = n_legal;
    ctx.hidden_size = (int)hidden_size;
    ctx.bpr = bpr;
    ctx.row_bytes = rb;
    ctx.quant_type = quant_type;
    ctx.is_repacked = is_repacked ? 1 : 0;
    ctx.fp32_ref = fp32_ref;
    atomic_init(&ctx.next_idx, 0u);

    if (dispatch_fn && n_legal >= 64u) {
        dispatch_fn(pool_opaque, sparse_q6k_worker, &ctx);
    } else {
        sparse_q6k_worker(0, 1, &ctx);
    }

    if (need_q8k && qx != qx_stack) free(qx);
    return 0;
}

int asi_grammar_sparse_lm_head_q6k(const uint8_t *q6k_weights,
                                   uint32_t vocab_size,
                                   uint32_t hidden_size,
                                   const float *x_norm,
                                   const int32_t *legal_ids,
                                   uint32_t n_legal,
                                   float *out_logits,
                                   int fp32_ref,
                                   LilaLhcDispatchFn dispatch_fn,
                                   void *pool_opaque) {
    return asi_grammar_sparse_lm_head_quant(q6k_weights, 14u, 0, vocab_size, hidden_size,
                                            x_norm, legal_ids, n_legal, out_logits,
                                            fp32_ref, dispatch_fn, pool_opaque);
}

typedef struct {
    const uint8_t *q6k_weights;
    const LilaQ8KBlock *qx;
    const float *x_norm;
    float *out_logits;
    uint32_t vocab_size;
    int hidden_size;
    int bpr;
    size_t row_bytes;
    uint32_t quant_type;
    int is_repacked;
    int fp32_ref;
    _Alignas(64) atomic_uint next_row;
} FullQ6kCtx;

static void full_q6k_worker(int tid, int n_threads, void *vctx) {
    (void)tid;
    FullQ6kCtx *ctx = (FullQ6kCtx *)vctx;
    const uint8_t *w = ctx->q6k_weights;
    float *out = ctx->out_logits;
    const size_t rb = ctx->row_bytes;
    const int bpr = ctx->bpr;
    const uint32_t V = ctx->vocab_size;

    for (;;) {
        uint32_t start = (n_threads <= 1)
                             ? 0u
                             : atomic_fetch_add_explicit(&ctx->next_row, 256u, memory_order_relaxed);
        if (start >= V) break;
        uint32_t end = (n_threads <= 1) ? V : (start + 256u);
        if (end > V) end = V;

        if (!ctx->fp32_ref && ctx->quant_type == 14u && !ctx->is_repacked) {
            uint32_t i = start;
#if defined(__AVX2__)
            for (; i + 1u < end; i += 2u) {
                const uint8_t *rp = w + (size_t)i * rb;
                if (i + 3u < end) {
                    gs_prefetch_q6k_row(rp + 2u * rb, rb);
                    gs_prefetch_q6k_row(rp + 3u * rb, rb);
                }
                q6k_dot2_q8k_avx2(rp, rp + rb, ctx->qx, bpr, &out[i], &out[i + 1u]);
            }
#endif
            for (; i < end; ++i) {
                out[i] = q6k_dot_q8k_simd(w + (size_t)i * rb, ctx->qx, bpr);
            }
        } else if (!ctx->fp32_ref && ctx->quant_type == 12u) {
            uint32_t i = start;
#if defined(__AVX2__)
            for (; i + 1u < end; i += 2u) {
                const uint8_t *rp = w + (size_t)i * rb;
                if (i + 3u < end) {
                    gs_prefetch_q6k_row(rp + 2u * rb, rb);
                    gs_prefetch_q6k_row(rp + 3u * rb, rb);
                }
                q4k_dot2_q8k_avx2(rp, rp + rb, ctx->qx, bpr, &out[i], &out[i + 1u]);
            }
#endif
            for (; i < end; ++i) {
                out[i] = q4k_dot_q8k_simd(w + (size_t)i * rb, ctx->qx, bpr);
            }
        } else {
            for (uint32_t i = start; i < end; ++i) {
                out[i] = lila_quant_dot_fp32(w + (size_t)i * rb, ctx->x_norm,
                                             ctx->hidden_size, ctx->quant_type, ctx->is_repacked);
            }
        }
        if (n_threads <= 1) break;
    }
}

int asi_grammar_full_lm_head_quant(const uint8_t *weights,
                                   uint32_t quant_type,
                                   int is_repacked,
                                   uint32_t vocab_size,
                                   uint32_t hidden_size,
                                   const float *x_norm,
                                   float *out_logits,
                                   int fp32_ref,
                                   LilaLhcDispatchFn dispatch_fn,
                                   void *pool_opaque) {
    if (!weights || !x_norm || !out_logits || vocab_size == 0u || hidden_size == 0u)
        return -1;
    if (!lila_quant_is_supported(quant_type)) return -1;
    size_t rb = lila_quant_row_bytes(quant_type, (int)hidden_size, is_repacked ? 1 : 0);
    if (rb == 0u) return -1;

    int bpr = ((int)hidden_size + QK_K - 1) / QK_K;
    int need_q8k = (!fp32_ref && (quant_type == 12u || (quant_type == 14u && !is_repacked)));

    LilaQ8KBlock qx_stack[32];
    LilaQ8KBlock *qx = qx_stack;
    if (need_q8k) {
        if (bpr > 32) {
            qx = (LilaQ8KBlock *)malloc((size_t)bpr * sizeof(LilaQ8KBlock));
            if (!qx) return -1;
        }
        quantize_vec_q8k(qx, x_norm, (int)hidden_size);
    }

    FullQ6kCtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.q6k_weights = weights;
    ctx.qx = need_q8k ? qx : NULL;
    ctx.x_norm = x_norm;
    ctx.out_logits = out_logits;
    ctx.vocab_size = vocab_size;
    ctx.hidden_size = (int)hidden_size;
    ctx.bpr = bpr;
    ctx.row_bytes = rb;
    ctx.quant_type = quant_type;
    ctx.is_repacked = is_repacked ? 1 : 0;
    ctx.fp32_ref = fp32_ref;
    atomic_init(&ctx.next_row, 0u);

    if (dispatch_fn && vocab_size >= 512u) {
        dispatch_fn(pool_opaque, full_q6k_worker, &ctx);
    } else {
        full_q6k_worker(0, 1, &ctx);
    }

    if (need_q8k && qx != qx_stack) free(qx);
    return 0;
}

int asi_grammar_full_lm_head_q6k(const uint8_t *q6k_weights,
                                 uint32_t vocab_size,
                                 uint32_t hidden_size,
                                 const float *x_norm,
                                 float *out_logits,
                                 int fp32_ref,
                                 LilaLhcDispatchFn dispatch_fn,
                                 void *pool_opaque) {
    return asi_grammar_full_lm_head_quant(q6k_weights, 14u, 0, vocab_size, hidden_size,
                                          x_norm, out_logits, fp32_ref,
                                          dispatch_fn, pool_opaque);
}

int asi_grammar_lm_head_forward_quant(AsiGrammarEngine *g,
                                      const uint8_t *weights,
                                      uint32_t quant_type,
                                      int is_repacked,
                                      uint32_t vocab_size,
                                      uint32_t hidden_size,
                                      const float *x_norm,
                                      float *out_logits,
                                      int fp32_ref,
                                      LilaLhcDispatchFn dispatch_fn,
                                      void *pool_opaque,
                                      AsiSieveForwardFn sieve_fn,
                                      void *sieve_ctx,
                                      AsiGrammarDispatchMode *out_mode) {
    if (!weights || !x_norm || !out_logits) return -1;

    /* Unconstrained path */
    if (!g) {
        if (sieve_fn) {
            if (out_mode) *out_mode = ASI_DISPATCH_SIEVE_MASK;
            return sieve_fn(sieve_ctx, weights, x_norm, out_logits, fp32_ref);
        }
        if (out_mode) *out_mode = ASI_DISPATCH_FULL_MASK;
        return asi_grammar_full_lm_head_quant(weights, quant_type, is_repacked,
                                              vocab_size, hidden_size,
                                              x_norm, out_logits, fp32_ref,
                                              dispatch_fn, pool_opaque);
    }

    const int32_t *legal_ids = NULL;
    uint32_t n_legal = 0u;
    if (asi_grammar_get_legal_ids(g, &legal_ids, &n_legal) != 0) return -1;
    if (n_legal == 0u) return -2;

    double t0 = gs_now_ms();
    int rc = 0;

    if (n_legal <= g->sparse_limit) {
        /* Case 2: Grammar-Aware Sparse lm_head (skip Sieve Stage-1 completely!) */
        if (out_mode) *out_mode = ASI_DISPATCH_SPARSE_Q6K;
        rc = asi_grammar_sparse_lm_head_quant(weights, quant_type, is_repacked,
                                              vocab_size, hidden_size,
                                              x_norm, legal_ids, n_legal, out_logits,
                                              fp32_ref, dispatch_fn, pool_opaque);
        g->telemetry.sparse_lm_head_calls++;
        g->telemetry.sparse_rows_eval_sum += n_legal;
    } else if (sieve_fn) {
        /* Case 3a: Sieve (1bit1024 K=2048) + grammar mask */
        if (out_mode) *out_mode = ASI_DISPATCH_SIEVE_MASK;
        rc = sieve_fn(sieve_ctx, weights, x_norm, out_logits, fp32_ref);
        if (rc == 0) {
            rc = asi_grammar_mask_logits(g, out_logits, vocab_size);
            if (rc == 0) {
                /* Safety check: if Sieve's top-K had 0 overlap with legal_ids,
                 * evaluate exact weights on the first min(n_legal, 256) legal tokens */
                float best = ASI_GRAMMAR_NEG_INF_FILL;
                for (uint32_t i = 0; i < n_legal; ++i) {
                    float v = out_logits[legal_ids[i]];
                    if (v > best) best = v;
                }
                if (best <= ASI_GRAMMAR_NEG_INF_FILL * 0.5f) {
                    uint32_t fb_n = (n_legal < 256u) ? n_legal : 256u;
                    rc = asi_grammar_sparse_lm_head_quant(weights, quant_type, is_repacked,
                                                          vocab_size, hidden_size,
                                                          x_norm, legal_ids, fb_n, out_logits,
                                                          fp32_ref, dispatch_fn, pool_opaque);
                }
            }
        }
        g->telemetry.sieve_lm_head_calls++;
    } else {
        /* Case 3b: Full exact weights + grammar mask */
        if (out_mode) *out_mode = ASI_DISPATCH_FULL_MASK;
        rc = asi_grammar_full_lm_head_quant(weights, quant_type, is_repacked,
                                            vocab_size, hidden_size,
                                            x_norm, out_logits, fp32_ref,
                                            dispatch_fn, pool_opaque);
        if (rc == 0) {
            rc = asi_grammar_mask_logits(g, out_logits, vocab_size);
        }
        g->telemetry.full_lm_head_calls++;
    }

    g->telemetry.lm_head_ms_sum += (gs_now_ms() - t0);
    return rc;
}

int asi_grammar_lm_head_forward_q6k(AsiGrammarEngine *g,
                                    const uint8_t *q6k_weights,
                                    uint32_t vocab_size,
                                    uint32_t hidden_size,
                                    const float *x_norm,
                                    float *out_logits,
                                    int fp32_ref,
                                    LilaLhcDispatchFn dispatch_fn,
                                    void *pool_opaque,
                                    AsiSieveForwardFn sieve_fn,
                                    void *sieve_ctx,
                                    AsiGrammarDispatchMode *out_mode) {
    return asi_grammar_lm_head_forward_quant(g, q6k_weights, 14u, 0,
                                             vocab_size, hidden_size,
                                             x_norm, out_logits, fp32_ref,
                                             dispatch_fn, pool_opaque,
                                             sieve_fn, sieve_ctx, out_mode);
}

/* ============================================================================
 * 6. Built-in C11 Schema-Compliant <tool_call> JSON Validator
 * ============================================================================ */

static void skip_ws(const char **p) {
    while (**p == ' ' || **p == '\t' || **p == '\r' || **p == '\n') (*p)++;
}

static int expect_char(const char **p, char c) {
    skip_ws(p);
    if (**p != c) return 0;
    (*p)++;
    return 1;
}

static int is_hex_digit_char(char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}

static int parse_strict_json_string(const char **p, char *out, size_t cap) {
    skip_ws(p);
    if (**p != '"') return 0;
    (*p)++;
    size_t len = 0u;
    while (**p && **p != '"') {
        unsigned char c = (unsigned char)**p;
        if (c < 0x20u) return 0; /* RFC 8259 disallows raw unescaped control chars 0x00..0x1F */
        if (c == '\\') {
            (*p)++;
            char esc = **p;
            if (!esc) return 0;
            char decoded = esc;
            switch (esc) {
                case '"':  decoded = '"';  (*p)++; break;
                case '\\': decoded = '\\'; (*p)++; break;
                case '/':  decoded = '/';  (*p)++; break;
                case 'b':  decoded = '\b'; (*p)++; break;
                case 'f':  decoded = '\f'; (*p)++; break;
                case 'n':  decoded = '\n'; (*p)++; break;
                case 'r':  decoded = '\r'; (*p)++; break;
                case 't':  decoded = '\t'; (*p)++; break;
                case 'u':
                    (*p)++;
                    for (int h = 0; h < 4; ++h) {
                        if (!is_hex_digit_char((*p)[h])) return 0;
                    }
                    (*p) += 4;
                    decoded = '?';
                    break;
                default:
                    return 0;
            }
            if (out && cap > 0u && len + 1u < cap) out[len++] = decoded;
            continue;
        }
        if (out && cap > 0u && len + 1u < cap) out[len++] = (char)c;
        (*p)++;
    }
    if (**p != '"') return 0;
    (*p)++;
    if (out && cap > 0u) out[len] = '\0';
    return 1;
}

static int parse_strict_int(const char **p) {
    skip_ws(p);
    const char *s = *p;
    if (*s == '-') {
        ++s;
        if (*s < '1' || *s > '9') return 0;
        ++s;
        while (*s >= '0' && *s <= '9') ++s;
    } else if (*s == '0') {
        ++s;
        if (*s >= '0' && *s <= '9') return 0; /* no leading zeros */
    } else if (*s >= '1' && *s <= '9') {
        ++s;
        while (*s >= '0' && *s <= '9') ++s;
    } else {
        return 0;
    }
    *p = s;
    return 1;
}

static int parse_strict_float(const char **p) {
    if (!parse_strict_int(p)) return 0;
    if (**p == '.') {
        (*p)++;
        if (**p < '0' || **p > '9') return 0;
        while (**p >= '0' && **p <= '9') (*p)++;
    }
    return 1;
}

static int parse_strict_bool(const char **p) {
    skip_ws(p);
    if (strncmp(*p, "true", 4) == 0 && !isalnum((unsigned char)(*p)[4])) {
        *p += 4;
        return 1;
    }
    if (strncmp(*p, "false", 5) == 0 && !isalnum((unsigned char)(*p)[5])) {
        *p += 5;
        return 1;
    }
    return 0;
}

int asi_grammar_validate_tool_call(const char *text,
                                   const AsiToolSchema *tools,
                                   uint32_t n_tools,
                                   int wrap_xml,
                                   char *err_buf,
                                   size_t err_cap) {
    if (!text || !tools || n_tools == 0u) {
        if (err_buf && err_cap) snprintf(err_buf, err_cap, "invalid arguments");
        return 0;
    }

    const char *p = text;
    if (wrap_xml) {
        skip_ws(&p);
        if (strncmp(p, "<tool_call>", 11) != 0) {
            if (err_buf && err_cap) snprintf(err_buf, err_cap, "missing <tool_call>");
            return 0;
        }
        p += 11;
    }

    if (!expect_char(&p, '{')) {
        if (err_buf && err_cap) snprintf(err_buf, err_cap, "expected '{'");
        return 0;
    }

    char key[128], tool_name[128];
    if (!parse_strict_json_string(&p, key, sizeof(key)) || strcmp(key, "name") != 0) {
        if (err_buf && err_cap) snprintf(err_buf, err_cap, "expected \"name\" key");
        return 0;
    }
    if (!expect_char(&p, ':') || !parse_strict_json_string(&p, tool_name, sizeof(tool_name))) {
        if (err_buf && err_cap) snprintf(err_buf, err_cap, "invalid tool name string");
        return 0;
    }

    const AsiToolSchema *tool = NULL;
    for (uint32_t i = 0; i < n_tools; ++i) {
        if (strcmp(tools[i].name, tool_name) == 0) { tool = &tools[i]; break; }
    }
    if (!tool) {
        if (err_buf && err_cap) snprintf(err_buf, err_cap, "unregistered tool '%s'", tool_name);
        return 0;
    }

    if (!expect_char(&p, ',') ||
        !parse_strict_json_string(&p, key, sizeof(key)) || strcmp(key, "arguments") != 0 ||
        !expect_char(&p, ':') ||
        !expect_char(&p, '{')) {
        if (err_buf && err_cap) snprintf(err_buf, err_cap, "expected \"arguments\": {");
        return 0;
    }

    uint32_t seen_mask = 0u;
    int last_found = -1;
    uint32_t parsed_args = 0u;

    skip_ws(&p);
    while (*p && *p != '}') {
        if (parsed_args > 0u && !expect_char(&p, ',')) {
            if (err_buf && err_cap) snprintf(err_buf, err_cap, "expected ',' between args");
            return 0;
        }
        char akey[128];
        if (!parse_strict_json_string(&p, akey, sizeof(akey)) || !expect_char(&p, ':')) {
            if (err_buf && err_cap) snprintf(err_buf, err_cap, "expected argument key");
            return 0;
        }
        int found = -1;
        for (uint32_t j = 0; j < tool->n_args; ++j) {
            if (strcmp(tool->args[j].name, akey) == 0) { found = (int)j; break; }
        }
        if (found < 0 || (seen_mask & (1u << found)) || found <= last_found) {
            if (err_buf && err_cap) snprintf(err_buf, err_cap, "unexpected/dup/out-of-order arg '%s'", akey);
            return 0;
        }
        seen_mask |= (1u << found);
        last_found = found;
        parsed_args++;

        const AsiToolArgSchema *arg = &tool->args[found];
        char sval[256];
        int ok = 0;
        switch (arg->type) {
            case ASI_ARG_STRING:
                ok = parse_strict_json_string(&p, sval, sizeof(sval));
                break;
            case ASI_ARG_INT:
                ok = parse_strict_int(&p);
                break;
            case ASI_ARG_FLOAT:
                ok = parse_strict_float(&p);
                break;
            case ASI_ARG_BOOL:
                ok = parse_strict_bool(&p);
                break;
            case ASI_ARG_ENUM:
                if (parse_strict_json_string(&p, sval, sizeof(sval))) {
                    for (uint32_t e = 0; e < arg->n_enum_values; ++e) {
                        if (strcmp(arg->enum_values[e], sval) == 0) { ok = 1; break; }
                    }
                }
                break;
        }
        if (!ok) {
            if (err_buf && err_cap) snprintf(err_buf, err_cap, "invalid value for arg '%s'", akey);
            return 0;
        }
        skip_ws(&p);
    }

    /* Verify every non-optional argument was present */
    for (uint32_t j = 0; j < tool->n_args; ++j) {
        if (!tool->args[j].optional && !(seen_mask & (1u << j))) {
            if (err_buf && err_cap) snprintf(err_buf, err_cap, "missing required arg '%s'", tool->args[j].name);
            return 0;
        }
    }

    if (!expect_char(&p, '}') || !expect_char(&p, '}')) {
        if (err_buf && err_cap) snprintf(err_buf, err_cap, "expected '}}'");
        return 0;
    }

    if (wrap_xml) {
        skip_ws(&p);
        if (strncmp(p, "</tool_call>", 12) != 0) {
            if (err_buf && err_cap) snprintf(err_buf, err_cap, "missing </tool_call>");
            return 0;
        }
        p += 12;
    }
    skip_ws(&p);
    if (*p != '\0') {
        if (err_buf && err_cap) snprintf(err_buf, err_cap, "trailing garbage after tool call");
        return 0;
    }
    return 1;
}
