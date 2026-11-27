#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "asi_v2.h"
#include "asi_ed25519.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef _WIN32
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

static void set_err(char *buf, size_t cap, const char *msg)
{
    if (!buf || cap == 0)
        return;
    snprintf(buf, cap, "%s", msg ? msg : "unknown error");
}

static uint64_t g_crc64_tables[8][256];
static int g_crc64_table_init = 0;

static void init_crc64_table(void)
{
    for (uint64_t i = 0; i < 256; i++)
    {
        uint64_t crc = i << 56;
        for (int b = 0; b < 8; b++)
        {
            if (crc & 0x8000000000000000ULL)
                crc = (crc << 1) ^ 0x42F0E1EBA9EA3693ULL;
            else
                crc <<= 1;
        }
        g_crc64_tables[0][i] = crc;
    }
    for (int t = 1; t < 8; t++)
    {
        for (uint64_t i = 0; i < 256; i++)
        {
            uint64_t prev = g_crc64_tables[t - 1][i];
            g_crc64_tables[t][i] = (prev << 8) ^ g_crc64_tables[0][prev >> 56];
        }
    }
    g_crc64_table_init = 1;
}

uint64_t asi_v2_crc64(const uint8_t *data, size_t len)
{
    if (!g_crc64_table_init)
        init_crc64_table();
    uint64_t crc = 0;
    size_t i = 0;
    while (i + 8u <= len)
    {
        crc = g_crc64_tables[7][((crc >> 56) ^ data[i + 0u]) & 0xFFu] ^
              g_crc64_tables[6][((crc >> 48) ^ data[i + 1u]) & 0xFFu] ^
              g_crc64_tables[5][((crc >> 40) ^ data[i + 2u]) & 0xFFu] ^
              g_crc64_tables[4][((crc >> 32) ^ data[i + 3u]) & 0xFFu] ^
              g_crc64_tables[3][((crc >> 24) ^ data[i + 4u]) & 0xFFu] ^
              g_crc64_tables[2][((crc >> 16) ^ data[i + 5u]) & 0xFFu] ^
              g_crc64_tables[1][((crc >>  8) ^ data[i + 6u]) & 0xFFu] ^
              g_crc64_tables[0][(crc          ^ data[i + 7u]) & 0xFFu];
        i += 8u;
    }
    for (; i < len; i++)
    {
        crc = (crc << 8) ^ g_crc64_tables[0][((crc >> 56) ^ data[i]) & 0xFFu];
    }
    return crc;
}

const char *asi_v2_arch_name(uint32_t arch_id)
{
    switch (arch_id)
    {
    case ASI_ARCH_DENSE_LLAMA:   return "dense_llama";
    case ASI_ARCH_HYBRID_QWEN35: return "hybrid_qwen35";
    case ASI_ARCH_DENSE_GEMMA:   return "dense_gemma";
    case ASI_ARCH_DENSE_QWEN2:   return "dense_qwen2";
    case ASI_ARCH_DENSE_QWEN3:   return "dense_qwen3";
    default:                     return "unknown";
    }
}

const AsiV2SectionEntry *asi_v2_find_section(const AsiV2Cartridge *cart, uint32_t type)
{
    if (!cart || !cart->sections)
        return NULL;
    for (uint32_t i = 0; i < cart->header.n_sections; i++)
    {
        if (cart->sections[i].type == type)
            return &cart->sections[i];
    }
    return NULL;
}

const AsiV2TensorEntry *asi_v2_find_tensor(const AsiV2Cartridge *cart, const char *name)
{
    if (!cart || !cart->tensors || !name)
        return NULL;
    for (uint32_t i = 0; i < cart->n_tensors; i++)
    {
        if (strncmp(cart->tensors[i].name, name, sizeof(cart->tensors[i].name)) == 0)
            return &cart->tensors[i];
    }
    return NULL;
}

int asi_v2_validate_memory(const uint8_t *data, size_t size, AsiV2Cartridge *out,
                           char *err_buf, size_t err_cap)
{
    if (!data || !out)
    {
        set_err(err_buf, err_cap, "null cartridge pointer");
        return -1;
    }
    memset(out, 0, sizeof(*out));
    out->base = data;
    out->file_size = size;

    if (size < sizeof(AsiV2Header))
    {
        set_err(err_buf, err_cap, "file smaller than ASI v2 header");
        return -1;
    }

    memcpy(&out->header, data, sizeof(AsiV2Header));
    if (out->header.magic != ASI_V2_MAGIC)
    {
        set_err(err_buf, err_cap, "invalid ASI magic");
        return -1;
    }
    if (out->header.version != ASI_V2_VERSION)
    {
        set_err(err_buf, err_cap, "unsupported ASI version (expected v2)");
        return -1;
    }
    if (out->header.total_size > size)
    {
        set_err(err_buf, err_cap, "truncated ASI v2 file");
        return -1;
    }
    if (out->header.n_sections == 0 || out->header.n_sections > 64)
    {
        set_err(err_buf, err_cap, "invalid section count");
        return -1;
    }

    uint64_t tbl_off = out->header.section_table_offset;
    uint64_t tbl_bytes = (uint64_t)out->header.n_sections * sizeof(AsiV2SectionEntry);
    if (tbl_off < sizeof(AsiV2Header) || tbl_off + tbl_bytes > size)
    {
        set_err(err_buf, err_cap, "section table out of bounds");
        return -1;
    }

    out->sections = (const AsiV2SectionEntry *)(data + tbl_off);

    for (uint32_t i = 0; i < out->header.n_sections; i++)
    {
        const AsiV2SectionEntry *s = &out->sections[i];
        if (s->offset > size || s->size > size - s->offset)
        {
            set_err(err_buf, err_cap, "section out of file bounds");
            return -1;
        }
        if ((s->offset % ASI_V2_PAGE_SIZE) != 0)
        {
            set_err(err_buf, err_cap, "section offset is not page-aligned");
            return -1;
        }
        int is_bulk_blob = (s->type == ASI_SEC_WEIGHTS || s->type == ASI_SEC_VISION ||
                            s->type == ASI_SEC_AUDIO || s->type == ASI_SEC_SIEVE_SKETCH);
        const char *verify_env = getenv("LILA_ASI_VERIFY_CRC");
        if (!is_bulk_blob || (verify_env && verify_env[0] == '1'))
        {
            uint64_t actual_crc = asi_v2_crc64(data + s->offset, (size_t)s->size);
            if (actual_crc != s->checksum)
            {
                set_err(err_buf, err_cap, "section CRC-64 checksum mismatch");
                return -1;
            }
        }
    }

    const AsiV2SectionEntry *cfg_sec = asi_v2_find_section(out, ASI_SEC_MODEL_CONFIG);
    if (!cfg_sec || cfg_sec->size < sizeof(AsiV2ModelConfig))
    {
        set_err(err_buf, err_cap, "missing or invalid MODEL_CONFIG section");
        return -1;
    }
    memcpy(&out->config, data + cfg_sec->offset, sizeof(AsiV2ModelConfig));
    if (out->config.n_layers == 0 || out->config.hidden_size == 0 || out->config.vocab_size == 0)
    {
        set_err(err_buf, err_cap, "MODEL_CONFIG has zero dimensions");
        return -1;
    }

    const AsiV2SectionEntry *w_sec = asi_v2_find_section(out, ASI_SEC_WEIGHTS);
    if (!w_sec || w_sec->size < sizeof(AsiV2WeightsHeader))
    {
        set_err(err_buf, err_cap, "missing or invalid WEIGHTS section");
        return -1;
    }
    AsiV2WeightsHeader wh;
    memcpy(&wh, data + w_sec->offset, sizeof(wh));
    uint64_t dir_bytes = sizeof(AsiV2WeightsHeader) + (uint64_t)wh.n_tensors * sizeof(AsiV2TensorEntry);
    if (wh.data_offset < dir_bytes || wh.data_offset > w_sec->size)
    {
        set_err(err_buf, err_cap, "WEIGHTS tensor directory out of bounds");
        return -1;
    }

    out->n_tensors = wh.n_tensors;
    out->tensors = (const AsiV2TensorEntry *)(data + w_sec->offset + sizeof(AsiV2WeightsHeader));
    out->weights_data = data + w_sec->offset + wh.data_offset;
    out->weights_data_size = (size_t)(w_sec->size - wh.data_offset);

    for (uint32_t i = 0; i < out->n_tensors; i++)
    {
        const AsiV2TensorEntry *t = &out->tensors[i];
        if (t->data_offset > out->weights_data_size ||
            t->data_size > out->weights_data_size - t->data_offset)
        {
            set_err(err_buf, err_cap, "tensor payload out of WEIGHTS bounds");
            return -1;
        }
    }

    if (!asi_v2_find_section(out, ASI_SEC_TOKENIZER))
    {
        set_err(err_buf, err_cap, "missing essential TOKENIZER section");
        return -1;
    }

    const char *req_sig = getenv("LILA_ASI_REQUIRE_SIGNATURE");
    if (req_sig && req_sig[0] == '1')
    {
        uint8_t pk_buf[32];
        const uint8_t *trusted_pk = NULL;
        const char *pk_hex = getenv("LILA_ASI_TRUSTED_PUBKEY");
        if (pk_hex && pk_hex[0] != '\0')
        {
            if (strlen(pk_hex) != 64)
            {
                set_err(err_buf, err_cap, "LILA_ASI_TRUSTED_PUBKEY must be 64 hex chars");
                return -1;
            }
            for (size_t i = 0; i < 32; i++)
            {
                unsigned int v = 0;
                if (!isxdigit((unsigned char)pk_hex[i * 2]) ||
                    !isxdigit((unsigned char)pk_hex[i * 2 + 1]) ||
                    sscanf(pk_hex + i * 2, "%2x", &v) != 1)
                {
                    set_err(err_buf, err_cap, "invalid hex in LILA_ASI_TRUSTED_PUBKEY");
                    return -1;
                }
                pk_buf[i] = (uint8_t)v;
            }
            trusted_pk = pk_buf;
        }
        if (asi_v2_verify_signature(out, trusted_pk, NULL, err_buf, err_cap) != 0)
            return -1;
    }

    return 0;
}

int asi_v2_open(const char *path, AsiV2Cartridge *out, char *err_buf, size_t err_cap)
{
    if (!path || !out)
    {
        set_err(err_buf, err_cap, "invalid arguments");
        return -1;
    }
    size_t file_size = 0;
    const uint8_t *mapped = NULL;

#ifdef _WIN32
    HANDLE hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE)
    {
        set_err(err_buf, err_cap, "cannot open ASI v2 file");
        return -1;
    }
    LARGE_INTEGER li;
    GetFileSizeEx(hFile, &li);
    file_size = (size_t)li.QuadPart;
    HANDLE hMap = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    CloseHandle(hFile);
    if (!hMap)
    {
        set_err(err_buf, err_cap, "CreateFileMappingA failed");
        return -1;
    }
    mapped = (const uint8_t *)MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(hMap);
#else
    int fd = open(path, O_RDONLY);
    if (fd < 0)
    {
        set_err(err_buf, err_cap, "cannot open ASI v2 file");
        return -1;
    }
    struct stat st;
    if (fstat(fd, &st) != 0)
    {
        close(fd);
        set_err(err_buf, err_cap, "fstat failed");
        return -1;
    }
    file_size = (size_t)st.st_size;
    void *m = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (m == MAP_FAILED)
    {
        set_err(err_buf, err_cap, "mmap failed");
        return -1;
    }
#ifdef MADV_HUGEPAGE
    {
        const char *no_thp = getenv("LILA_ASI_NO_HUGEPAGE");
        if (!no_thp || no_thp[0] != '1')
            madvise(m, file_size, MADV_HUGEPAGE);
    }
#endif
    mapped = (const uint8_t *)m;
#endif

    if (!mapped)
    {
        set_err(err_buf, err_cap, "mapping returned NULL");
        return -1;
    }
    int rc = asi_v2_validate_memory(mapped, file_size, out, err_buf, err_cap);
    out->owns_mmap = 1;
    if (rc != 0)
        asi_v2_close(out);
    return rc;
}

void asi_v2_close(AsiV2Cartridge *cart)
{
    if (!cart)
        return;
    if (cart->owns_mmap && cart->base)
    {
#ifdef _WIN32
        UnmapViewOfFile((LPCVOID)cart->base);
#else
        munmap((void *)cart->base, cart->file_size);
#endif
    }
    memset(cart, 0, sizeof(*cart));
}

static float repack_fp16_to_fp32(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t res = 0;
    if (exp == 0)
    {
        if (mant != 0)
        {
            while (!(mant & 0x400u)) { mant <<= 1; exp--; }
            exp++;
            mant &= ~0x400u;
            res = sign | ((exp + 112u) << 23) | (mant << 13);
        }
        else
        {
            res = sign;
        }
    }
    else if (exp == 31)
    {
        res = sign | 0x7F800000u | (mant << 13);
    }
    else
    {
        res = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &res, 4);
    return f;
}

/* Repack n_blocks of GGML Q6_K (210B/block) into 288B/block pre-unpacked AVX2 layout. */
void asi_v2_repack_q6k(const uint8_t *src, uint8_t *dst, size_t n_blocks)
{
    for (size_t b = 0; b < n_blocks; b++)
    {
        const uint8_t *block = src + b * 210;
        uint8_t *out = dst + b * 288;
        const uint8_t *ql = block;
        const uint8_t *qh = block + 128;
        const int8_t *sc = (const int8_t *)(block + 192);
        uint16_t d_raw;
        memcpy(&d_raw, block + 208, 2);
        float d = repack_fp16_to_fp32(d_raw);

        uint8_t *u6 = out;
        for (int half = 0; half < 2; half++)
        {
            for (int l = 0; l < 32; l++)
            {
                u6[l +  0] = (uint8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4));
                u6[l + 32] = (uint8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4));
                u6[l + 64] = (uint8_t)((ql[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4));
                u6[l + 96] = (uint8_t)((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4));
            }
            u6 += 128;
            ql += 64;
            qh += 32;
        }
        memcpy(out + 256, sc, 16);
        memcpy(out + 272, &d, 4);
        memset(out + 276, 0, 12);
    }
}

static void sha256_update_u32le(AsiSha256Ctx *ctx, uint32_t v)
{
    uint8_t b[4] = {
        (uint8_t)(v & 0xFFu),
        (uint8_t)((v >> 8) & 0xFFu),
        (uint8_t)((v >> 16) & 0xFFu),
        (uint8_t)((v >> 24) & 0xFFu),
    };
    asi_sha256_update(ctx, b, 4);
}

static void sha256_update_u64le(AsiSha256Ctx *ctx, uint64_t v)
{
    uint8_t b[8];
    for (int i = 0; i < 8; i++) b[i] = (uint8_t)((v >> (i * 8)) & 0xFFu);
    asi_sha256_update(ctx, b, 8);
}

int asi_v2_compute_signing_digest(const AsiV2Cartridge *cart,
                                  const AsiV2SignatureBlock *sig_blk,
                                  uint8_t out_digest[32])
{
    if (!cart || !cart->base || !cart->sections || !sig_blk || !out_digest)
        return -1;

    static const uint8_t kDomain[19] = "LILA-ASI-V2-SIG-V1";
    AsiSha256Ctx manifest;
    asi_sha256_init(&manifest);
    asi_sha256_update(&manifest, kDomain, sizeof(kDomain));

    uint32_t signed_count = 0;
    for (uint32_t i = 0; i < cart->header.n_sections; i++)
    {
        if (cart->sections[i].type != ASI_SEC_SIGNATURE)
            signed_count++;
    }

    sha256_update_u32le(&manifest, cart->header.magic);
    sha256_update_u32le(&manifest, cart->header.version);
    sha256_update_u32le(&manifest, cart->header.arch_id);
    sha256_update_u32le(&manifest, cart->header.flags & ~ASI_V2_FLAG_HAS_SIGNATURE);
    sha256_update_u32le(&manifest, signed_count);

    for (uint32_t i = 0; i < cart->header.n_sections; i++)
    {
        const AsiV2SectionEntry *s = &cart->sections[i];
        if (s->type == ASI_SEC_SIGNATURE)
            continue;
        if (s->offset > cart->file_size || s->size > cart->file_size - s->offset)
            return -1;

        AsiSha256Ctx sec_hash_ctx;
        uint8_t sec_hash[32];
        asi_sha256_init(&sec_hash_ctx);
        asi_sha256_update(&sec_hash_ctx, cart->base + s->offset, (size_t)s->size);
        asi_sha256_final(&sec_hash_ctx, sec_hash);

        sha256_update_u32le(&manifest, s->type);
        sha256_update_u32le(&manifest, s->flags);
        sha256_update_u64le(&manifest, s->size);
        sha256_update_u64le(&manifest, s->checksum);
        asi_sha256_update(&manifest, sec_hash, 32);
    }

    sha256_update_u32le(&manifest, sig_blk->version);
    sha256_update_u32le(&manifest, sig_blk->alg_id);
    sha256_update_u64le(&manifest, sig_blk->signed_at_unix);
    asi_sha256_update(&manifest, (const uint8_t *)sig_blk->publisher, sizeof(sig_blk->publisher));
    asi_sha256_update(&manifest, sig_blk->public_key, sizeof(sig_blk->public_key));
    asi_sha256_final(&manifest, out_digest);
    return 0;
}

int asi_v2_verify_signature(const AsiV2Cartridge *cart,
                            const uint8_t *trusted_pubkey_32,
                            AsiV2SignatureInfo *out_info,
                            char *err_buf,
                            size_t err_cap)
{
    if (out_info) memset(out_info, 0, sizeof(*out_info));
    if (!cart || !cart->base)
    {
        set_err(err_buf, err_cap, "null cartridge");
        return -1;
    }

    const AsiV2SectionEntry *sig_sec = asi_v2_find_section(cart, ASI_SEC_SIGNATURE);
    if (!sig_sec)
    {
        set_err(err_buf, err_cap, "cartridge has no SIGNATURE section");
        return -1;
    }
    if (sig_sec->size < sizeof(AsiV2SignatureBlock))
    {
        set_err(err_buf, err_cap, "SIGNATURE section is truncated");
        return -1;
    }

    AsiV2SignatureBlock blk;
    memcpy(&blk, cart->base + sig_sec->offset, sizeof(blk));
    if (memcmp(blk.magic, "ASISIG1\0", 8) != 0)
    {
        set_err(err_buf, err_cap, "invalid SIGNATURE block magic");
        return -1;
    }
    if (blk.version != ASI_SIG_VERSION || blk.alg_id != ASI_SIG_ALG_ED25519)
    {
        set_err(err_buf, err_cap, "unsupported SIGNATURE version or algorithm");
        return -1;
    }

    if (trusted_pubkey_32 != NULL && !asi_ct_memeq(blk.public_key, trusted_pubkey_32, 32))
    {
        set_err(err_buf, err_cap, "signer public key does not match trusted publisher key");
        return -1;
    }

    /* Re-verify section CRCs first so tampered bulk blobs are caught deterministically */
    for (uint32_t i = 0; i < cart->header.n_sections; i++)
    {
        const AsiV2SectionEntry *s = &cart->sections[i];
        uint64_t actual_crc = asi_v2_crc64(cart->base + s->offset, (size_t)s->size);
        if (actual_crc != s->checksum)
        {
            set_err(err_buf, err_cap, "section CRC-64 checksum mismatch during signature verify");
            return -1;
        }
    }

    uint8_t digest[32];
    if (asi_v2_compute_signing_digest(cart, &blk, digest) != 0)
    {
        set_err(err_buf, err_cap, "failed to compute canonical signing digest");
        return -1;
    }

    if (asi_ed25519_verify(blk.signature, digest, sizeof(digest), blk.public_key) != 0)
    {
        set_err(err_buf, err_cap, "Ed25519 signature verification failed (tampered cartridge)");
        return -1;
    }

    if (out_info)
    {
        out_info->valid = 1;
        out_info->alg_id = blk.alg_id;
        out_info->signed_at_unix = blk.signed_at_unix;
        memcpy(out_info->publisher, blk.publisher, sizeof(blk.publisher));
        out_info->publisher[sizeof(blk.publisher)] = '\0';
        memcpy(out_info->public_key, blk.public_key, sizeof(blk.public_key));
    }
    return 0;
}
