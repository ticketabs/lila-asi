#define _POSIX_C_SOURCE 200809L
#include "lila_lm_head_candidate.h"
#include "asi_platform.h"
#include "asi_v2.h"

#include <ctype.h>
#include <float.h>
#include <math.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define LHC_TILE_ROWS 64u
#define LHC_NEG_FILL  (-1e30f)

/* 2-bit uniform symmetric quantizer {-1.5, -0.5, +0.5, +1.5} with optimal Gaussian step */
static const float kLhc2BitStepMul = 0.9850f;

typedef struct {
    float score;
    int32_t id;
} LhcHeapNode;

typedef struct {
    int32_t id;
    uint8_t mask;
} LhcMergedRow;

typedef struct {
    _Alignas(16) int8_t lut[256][16];
    float b_scale[16];
    uint32_t n_blocks;
    uint32_t groups_per_block;
    /* 2BIT_1024 row-major 8-block maddubs query state */
    _Alignas(32) int8_t qx_planes[1024];
    float dx8[8];
    float bias_x;
} LhcQueryLut;

struct LilaLhcIndex {
    LilaLhcMode mode;
    uint32_t vocab_size;
    uint32_t hidden_size;
    uint32_t n_tiles;        /* 64-row tiles for 1-bit; 8-row groups for 2-bit */
    uint32_t tile_rows;      /* 64 for 1-bit, 8 for 2-bit */
    uint32_t n_groups;       /* 4-bit nibbles per row (128 or 256 for 1-bit) */
    uint32_t n_blocks;       /* 64-dim activation scale blocks: 8 or 16 */
    uint32_t groups_per_blk; /* 16 */
    size_t tile_bytes;       /* n_groups * 32 for 1-bit; 8 * 256 = 2048 for 2-bit */
    size_t sketch_bytes_per_row;
    size_t total_sketch_bytes;
    size_t exact_q6k_bytes;
    uint32_t quant_type;
    int is_repacked;
    int owns_buffers;
    size_t row_bytes;
    double build_ms;
    size_t extra_rss_bytes;
    uint8_t *sketch_data;    /* 64-byte aligned, n_tiles * tile_bytes */
    uint16_t *row_scales;    /* 64-byte aligned, n_tiles * tile_rows fp16 row scales */
};

struct LilaLhcWorkspace {
    uint32_t vocab_size;
    uint32_t top_k;
    uint32_t n_threads;
    uint32_t max_spec_b;
    size_t alloc_bytes;
    LhcHeapNode *heaps;        /* [max_spec_b * n_threads * top_k] */
    uint32_t *heap_counts;     /* [max_spec_b * n_threads] */
    int32_t *cands;            /* [max_spec_b * top_k] */
    uint32_t *cand_counts;     /* [max_spec_b] */
    LhcQueryLut *luts;         /* [max_spec_b] */
    LhcMergedRow *merged_rows; /* [max_spec_b * top_k] */
    LilaQ8KBlock *qx_lanes;    /* [max_spec_b * 4] */
    LilaLhcStats stats;
};

static double lhc_now_ms(void) {
    return (double)asi_monotonic_ns() * 1e-6;
}

static size_t lhc_current_rss_bytes(void) {
#if defined(__linux__)
    FILE *fp = fopen("/proc/self/status", "r");
    if (!fp) return 0;
    char line[256];
    size_t rss_kb = 0;
    while (fgets(line, sizeof(line), fp)) {
        if (strncmp(line, "VmRSS:", 6) == 0) {
            char *p = line + 6;
            rss_kb = (size_t)strtoull(p, NULL, 10);
            break;
        }
    }
    fclose(fp);
    return rss_kb * 1024u;
#else
    return 0;
#endif
}

static void *lhc_aligned_alloc(size_t alignment, size_t size) {
    if (size == 0) size = alignment;
    size_t padded = (size + alignment - 1u) & ~(alignment - 1u);
#if defined(_WIN32)
    return _aligned_malloc(padded, alignment);
#else
    return aligned_alloc(alignment, padded);
#endif
}

static void lhc_aligned_free(void *ptr) {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

uint32_t lila_lhc_abi_version(void) { return 1u; }

LilaLhcMode lila_lhc_parse_mode(const char *mode_str) {
    if (!mode_str || mode_str[0] == '\0') return LILA_LHC_MODE_OFF;
    char buf[32];
    size_t j = 0;
    for (size_t i = 0; mode_str[i] != '\0' && j + 1 < sizeof(buf); ++i) {
        unsigned char c = (unsigned char)mode_str[i];
        if (c == '_' || c == '-' || isspace(c)) continue;
        buf[j++] = (char)tolower(c);
    }
    buf[j] = '\0';
    if (strcmp(buf, "off") == 0 || strcmp(buf, "0") == 0 || strcmp(buf, "none") == 0) {
        return LILA_LHC_MODE_OFF;
    }
    if (strcmp(buf, "1bit512") == 0) return LILA_LHC_MODE_1BIT_512;
    if (strcmp(buf, "1bit1024") == 0) return LILA_LHC_MODE_1BIT_1024;
    if (strcmp(buf, "2bit1024") == 0) return LILA_LHC_MODE_2BIT_1024;
    return LILA_LHC_MODE_OFF;
}

const char *lila_lhc_mode_name(LilaLhcMode mode) {
    switch (mode) {
        case LILA_LHC_MODE_OFF:       return "off";
        case LILA_LHC_MODE_1BIT_512:  return "1bit512";
        case LILA_LHC_MODE_1BIT_1024: return "1bit1024";
        case LILA_LHC_MODE_2BIT_1024: return "2bit1024";
        default:                      return "unknown";
    }
}

/* ─────────────────────────── Parallel Index Builder ─────────────────────────── */

typedef struct {
    LilaLhcIndex *idx;
    const uint8_t *q6k_weights;
    const float *wt; /* output_norm[i]^2, length 1024 */
    size_t q6k_row_bytes;
    _Alignas(64) atomic_uint next_tile;
} LhcBuildCtx;

static void lhc_build_tile(const LhcBuildCtx *ctx, uint32_t tile) {
    LilaLhcIndex *idx = ctx->idx;
    const uint32_t V = idx->vocab_size;
    const float *wt = ctx->wt;
    const uint32_t row_base = tile * idx->tile_rows;
    const uint32_t valid = (row_base + idx->tile_rows <= V) ? idx->tile_rows : (V - row_base);
    uint8_t *tile_ptr = idx->sketch_data + (size_t)tile * idx->tile_bytes;
    uint16_t *scale_ptr = idx->row_scales + (size_t)tile * idx->tile_rows;

    if (idx->mode == LILA_LHC_MODE_1BIT_512) {
        for (uint32_t lr = 0; lr < valid; ++lr) {
            uint32_t r = row_base + lr;
            float w[1024];
            lila_dequant_row(w, ctx->q6k_weights + (size_t)r * ctx->q6k_row_bytes, 1024,
                             idx->quant_type, idx->is_repacked);

            float w512[512];
            float num = 0.0f, den = 0.0f;
            for (int i = 0; i < 512; ++i) {
                float wt0 = wt[2 * i];
                float wt1 = wt[2 * i + 1];
                float wsum = wt0 + wt1;
                w512[i] = (wsum > 0.0f)
                              ? ((w[2 * i] * wt0 + w[2 * i + 1] * wt1) / wsum)
                              : (0.5f * (w[2 * i] + w[2 * i + 1]));
                num += wsum * fabsf(w512[i]);
                den += wsum;
            }
            float alpha = (den > 0.0f) ? (num / den) : 0.0f;
            scale_ptr[lr] = fp32_to_fp16(alpha);

            uint32_t byte_off = lr & 31u;
            uint32_t shift = (lr >= 32u) ? 4u : 0u;
            for (uint32_t g = 0; g < 128u; ++g) {
                uint8_t nib = 0;
                if (w512[g * 4u + 0u] >= 0.0f) nib |= 1u;
                if (w512[g * 4u + 1u] >= 0.0f) nib |= 2u;
                if (w512[g * 4u + 2u] >= 0.0f) nib |= 4u;
                if (w512[g * 4u + 3u] >= 0.0f) nib |= 8u;
                tile_ptr[(size_t)g * 32u + byte_off] |= (uint8_t)(nib << shift);
            }
        }
    } else if (idx->mode == LILA_LHC_MODE_1BIT_1024) {
        float den = 0.0f;
        for (int i = 0; i < 1024; ++i) den += wt[i];
        float inv_den = (den > 0.0f) ? (1.0f / den) : 0.0f;

        for (uint32_t lr = 0; lr < valid; ++lr) {
            uint32_t r = row_base + lr;
            float w[1024];
            lila_dequant_row(w, ctx->q6k_weights + (size_t)r * ctx->q6k_row_bytes, 1024,
                             idx->quant_type, idx->is_repacked);

            float num = 0.0f;
            for (int i = 0; i < 1024; ++i) num += wt[i] * fabsf(w[i]);
            scale_ptr[lr] = fp32_to_fp16(num * inv_den);

            uint32_t byte_off = lr & 31u;
            uint32_t shift = (lr >= 32u) ? 4u : 0u;
            for (uint32_t g = 0; g < 256u; ++g) {
                uint8_t nib = 0;
                if (w[g * 4u + 0u] >= 0.0f) nib |= 1u;
                if (w[g * 4u + 1u] >= 0.0f) nib |= 2u;
                if (w[g * 4u + 2u] >= 0.0f) nib |= 4u;
                if (w[g * 4u + 3u] >= 0.0f) nib |= 8u;
                tile_ptr[(size_t)g * 32u + byte_off] |= (uint8_t)(nib << shift);
            }
        }
    } else if (idx->mode == LILA_LHC_MODE_2BIT_1024) {
        float den_w = 0.0f;
        for (int i = 0; i < 1024; ++i) den_w += wt[i];
        float inv_den_w = (den_w > 0.0f) ? (1.0f / den_w) : 0.0f;

        for (uint32_t lr = 0; lr < valid; ++lr) {
            uint32_t r = row_base + lr;
            float w[1024];
            lila_dequant_row(w, ctx->q6k_weights + (size_t)r * ctx->q6k_row_bytes, 1024,
                             idx->quant_type, idx->is_repacked);

            float ss = 0.0f;
            for (int i = 0; i < 1024; ++i) ss += wt[i] * (w[i] * w[i]);
            float std_w = sqrtf(ss * inv_den_w);
            float step = kLhc2BitStepMul * std_w + 1e-12f;
            float inv_step = 1.0f / step;

            uint8_t *dst = tile_ptr + (size_t)lr * 256u;
            float num = 0.0f, den = 0.0f;

            for (int b = 0; b < 256; ++b) {
                uint8_t packed = 0;
                for (int k = 0; k < 4; ++k) {
                    int idx_w = b * 4 + k;
                    float val = w[idx_w];
                    int q = (int)floorf(val * inv_step);
                    if (q < -2) q = -2;
                    else if (q > 1) q = 1;
                    uint8_t code = (uint8_t)(q + 2);
                    packed |= (uint8_t)(code << (2 * k));
                    float q_level = (float)code - 1.5f;
                    float wtv = wt[idx_w];
                    num += wtv * val * q_level;
                    den += wtv * q_level * q_level;
                }
                dst[b] = packed;
            }
            float alpha = (den > 0.0f) ? (num / den) : 0.0f;
            scale_ptr[lr] = fp32_to_fp16(alpha);
        }
    }
}

static void lhc_build_worker(int tid, int n_threads, void *vctx) {
    (void)tid;
    LhcBuildCtx *ctx = (LhcBuildCtx *)vctx;
    uint32_t n_tiles = ctx->idx->n_tiles;
    uint32_t chunk = (ctx->idx->tile_rows == 8u) ? 64u : 8u;
    if (n_threads <= 1) {
        for (uint32_t t = 0; t < n_tiles; ++t) lhc_build_tile(ctx, t);
        return;
    }
    for (;;) {
        uint32_t start = atomic_fetch_add_explicit(&ctx->next_tile, chunk, memory_order_relaxed);
        if (start >= n_tiles) break;
        uint32_t end = start + chunk;
        if (end > n_tiles) end = n_tiles;
        for (uint32_t t = start; t < end; ++t) lhc_build_tile(ctx, t);
    }
}

LilaLhcIndex *lila_lhc_build_quant_pool(LilaLhcMode mode,
                                        const uint8_t *weights,
                                        uint32_t quant_type,
                                        int is_repacked,
                                        uint32_t vocab_size,
                                        uint32_t hidden_size,
                                        const float *output_norm,
                                        LilaLhcDispatchFn dispatch_fn,
                                        void *pool_opaque) {
    if (!weights || vocab_size == 0 || hidden_size != 1024u) return NULL;
    if (!lila_quant_is_supported(quant_type)) return NULL;
    size_t row_bytes = lila_quant_row_bytes(quant_type, (int)hidden_size, is_repacked ? 1 : 0);
    if (row_bytes == 0u) return NULL;
    if (mode != LILA_LHC_MODE_1BIT_512 &&
        mode != LILA_LHC_MODE_1BIT_1024 &&
        mode != LILA_LHC_MODE_2BIT_1024) {
        return NULL;
    }

    double t0 = lhc_now_ms();
    size_t rss_before = lhc_current_rss_bytes();

    LilaLhcIndex *idx = (LilaLhcIndex *)calloc(1, sizeof(LilaLhcIndex));
    if (!idx) return NULL;

    idx->mode = mode;
    idx->vocab_size = vocab_size;
    idx->hidden_size = hidden_size;
    idx->quant_type = quant_type;
    idx->is_repacked = is_repacked ? 1 : 0;
    idx->owns_buffers = 1;
    idx->row_bytes = row_bytes;

    if (mode == LILA_LHC_MODE_1BIT_512) {
        idx->tile_rows = LHC_TILE_ROWS;
        idx->n_groups = 128u;
        idx->n_blocks = 8u;
        idx->groups_per_blk = 16u;
        idx->sketch_bytes_per_row = 66u;
        idx->tile_bytes = (size_t)idx->n_groups * 32u;
    } else if (mode == LILA_LHC_MODE_1BIT_1024) {
        idx->tile_rows = LHC_TILE_ROWS;
        idx->n_groups = 256u;
        idx->n_blocks = 16u;
        idx->groups_per_blk = 16u;
        idx->sketch_bytes_per_row = 130u;
        idx->tile_bytes = (size_t)idx->n_groups * 32u;
    } else {
        idx->tile_rows = 8u;
        idx->n_groups = 0u;
        idx->n_blocks = 8u;
        idx->groups_per_blk = 0u;
        idx->sketch_bytes_per_row = 258u;
        idx->tile_bytes = 8u * 256u;
    }
    idx->n_tiles = (vocab_size + idx->tile_rows - 1u) / idx->tile_rows;
    idx->total_sketch_bytes = (size_t)vocab_size * idx->sketch_bytes_per_row;
    idx->exact_q6k_bytes = (size_t)vocab_size * row_bytes;

    size_t data_bytes = (size_t)idx->n_tiles * idx->tile_bytes;
    size_t scale_bytes = (size_t)idx->n_tiles * idx->tile_rows * sizeof(uint16_t);
    idx->sketch_data = (uint8_t *)lhc_aligned_alloc(64u, data_bytes);
    idx->row_scales = (uint16_t *)lhc_aligned_alloc(64u, scale_bytes);
    if (!idx->sketch_data || !idx->row_scales) {
        lila_lhc_free(idx);
        return NULL;
    }
    memset(idx->sketch_data, 0, data_bytes);
    memset(idx->row_scales, 0, scale_bytes);

    float wt[1024];
    for (int i = 0; i < 1024; ++i) {
        float on_v = output_norm ? output_norm[i] : 1.0f;
        wt[i] = on_v * on_v;
    }

    LhcBuildCtx ctx;
    ctx.idx = idx;
    ctx.q6k_weights = weights;
    ctx.wt = wt;
    ctx.q6k_row_bytes = row_bytes;
    atomic_init(&ctx.next_tile, 0u);

    if (dispatch_fn && idx->n_tiles >= 4u) {
        dispatch_fn(pool_opaque, lhc_build_worker, &ctx);
    } else {
        lhc_build_worker(0, 1, &ctx);
    }

    idx->build_ms = lhc_now_ms() - t0;
    size_t rss_after = lhc_current_rss_bytes();
    idx->extra_rss_bytes = (rss_after > rss_before) ? (rss_after - rss_before) : 0u;
    return idx;
}

LilaLhcIndex *lila_lhc_build_quant(LilaLhcMode mode,
                                   const uint8_t *weights,
                                   uint32_t quant_type,
                                   int is_repacked,
                                   uint32_t vocab_size,
                                   uint32_t hidden_size,
                                   const float *output_norm) {
    return lila_lhc_build_quant_pool(mode, weights, quant_type, is_repacked,
                                     vocab_size, hidden_size, output_norm, NULL, NULL);
}

LilaLhcIndex *lila_lhc_build_q6k_pool(LilaLhcMode mode,
                                      const uint8_t *q6k_weights,
                                      uint32_t vocab_size,
                                      uint32_t hidden_size,
                                      const float *output_norm,
                                      LilaLhcDispatchFn dispatch_fn,
                                      void *pool_opaque) {
    return lila_lhc_build_quant_pool(mode, q6k_weights, 14u, 0,
                                     vocab_size, hidden_size, output_norm,
                                     dispatch_fn, pool_opaque);
}

LilaLhcIndex *lila_lhc_build_q6k(LilaLhcMode mode,
                                 const uint8_t *q6k_weights,
                                 uint32_t vocab_size,
                                 uint32_t hidden_size,
                                 const float *output_norm) {
    return lila_lhc_build_q6k_pool(mode, q6k_weights, vocab_size, hidden_size,
                                   output_norm, NULL, NULL);
}

uint32_t lila_lhc_quant_type(const LilaLhcIndex *index) {
    return index ? index->quant_type : 14u;
}

int lila_lhc_is_repacked(const LilaLhcIndex *index) {
    return index ? index->is_repacked : 0;
}

void lila_lhc_free(LilaLhcIndex *index) {
    if (!index) return;
    if (index->owns_buffers) {
        lhc_aligned_free(index->sketch_data);
        lhc_aligned_free(index->row_scales);
    }
    free(index);
}

LilaLhcIndex *lila_lhc_from_serialized(LilaLhcMode expected_mode,
                                       const uint8_t *sec_data,
                                       size_t sec_size,
                                       uint32_t quant_type,
                                       int is_repacked,
                                       uint32_t vocab_size,
                                       uint32_t hidden_size) {
    if (!sec_data || sec_size < sizeof(AsiSieveSketchHeader)) return NULL;
    if (((uintptr_t)sec_data & 63u) != 0u) return NULL;
    if (vocab_size == 0u || hidden_size != 1024u) return NULL;
    if (!lila_quant_is_supported(quant_type)) return NULL;

    AsiSieveSketchHeader hdr;
    memcpy(&hdr, sec_data, sizeof(hdr));
    if (memcmp(hdr.magic, ASI_SIEVE_MAGIC, 8) != 0) return NULL;

    LilaLhcMode mode = (LilaLhcMode)hdr.mode;
    if (expected_mode != LILA_LHC_MODE_OFF && mode != expected_mode) return NULL;
    if (mode != LILA_LHC_MODE_1BIT_512 &&
        mode != LILA_LHC_MODE_1BIT_1024 &&
        mode != LILA_LHC_MODE_2BIT_1024) {
        return NULL;
    }
    if (hdr.vocab_size != vocab_size || hdr.hidden_size != hidden_size) return NULL;
    if (hdr.quant_type != quant_type || hdr.is_repacked != (uint32_t)(is_repacked ? 1 : 0)) return NULL;

    size_t row_bytes = lila_quant_row_bytes(quant_type, (int)hidden_size, is_repacked ? 1 : 0);
    if (row_bytes == 0u) return NULL;

    double t0 = lhc_now_ms();
    LilaLhcIndex *idx = (LilaLhcIndex *)calloc(1, sizeof(LilaLhcIndex));
    if (!idx) return NULL;

    idx->mode = mode;
    idx->vocab_size = vocab_size;
    idx->hidden_size = hidden_size;
    idx->quant_type = quant_type;
    idx->is_repacked = is_repacked ? 1 : 0;
    idx->owns_buffers = 0;
    idx->row_bytes = row_bytes;

    if (mode == LILA_LHC_MODE_1BIT_512) {
        idx->tile_rows = LHC_TILE_ROWS;
        idx->n_groups = 128u;
        idx->n_blocks = 8u;
        idx->groups_per_blk = 16u;
        idx->sketch_bytes_per_row = 66u;
        idx->tile_bytes = (size_t)idx->n_groups * 32u;
    } else if (mode == LILA_LHC_MODE_1BIT_1024) {
        idx->tile_rows = LHC_TILE_ROWS;
        idx->n_groups = 256u;
        idx->n_blocks = 16u;
        idx->groups_per_blk = 16u;
        idx->sketch_bytes_per_row = 130u;
        idx->tile_bytes = (size_t)idx->n_groups * 32u;
    } else {
        idx->tile_rows = 8u;
        idx->n_groups = 0u;
        idx->n_blocks = 8u;
        idx->groups_per_blk = 0u;
        idx->sketch_bytes_per_row = 258u;
        idx->tile_bytes = 8u * 256u;
    }

    idx->n_tiles = (vocab_size + idx->tile_rows - 1u) / idx->tile_rows;
    idx->total_sketch_bytes = (size_t)vocab_size * idx->sketch_bytes_per_row;
    idx->exact_q6k_bytes = (size_t)vocab_size * row_bytes;

    size_t data_bytes = (size_t)idx->n_tiles * idx->tile_bytes;
    size_t scale_bytes = (size_t)idx->n_tiles * idx->tile_rows * sizeof(uint16_t);

    if (hdr.n_tiles != idx->n_tiles ||
        hdr.sketch_bytes != (uint64_t)data_bytes ||
        hdr.scales_bytes != (uint64_t)scale_bytes ||
        (hdr.sketch_offset & 63u) != 0u ||
        (hdr.scales_offset & 63u) != 0u ||
        hdr.sketch_offset < sizeof(AsiSieveSketchHeader) ||
        hdr.sketch_offset > (uint64_t)sec_size ||
        hdr.sketch_bytes > (uint64_t)sec_size - hdr.sketch_offset ||
        hdr.scales_offset > (uint64_t)sec_size ||
        hdr.scales_bytes > (uint64_t)sec_size - hdr.scales_offset) {
        free(idx);
        return NULL;
    }

    idx->sketch_data = (uint8_t *)(uintptr_t)(sec_data + hdr.sketch_offset);
    idx->row_scales = (uint16_t *)(uintptr_t)(sec_data + hdr.scales_offset);
    idx->build_ms = lhc_now_ms() - t0;
    idx->extra_rss_bytes = 0u;
    return idx;
}

int lila_lhc_serialize_section(const LilaLhcIndex *index,
                                 uint8_t *out_buf,
                                 size_t out_cap,
                                 size_t *out_written) {
    if (!index || !index->sketch_data || !index->row_scales || !out_written) return -1;
    size_t data_bytes = (size_t)index->n_tiles * index->tile_bytes;
    size_t scale_bytes = (size_t)index->n_tiles * index->tile_rows * sizeof(uint16_t);
    size_t sketch_off = 64u;
    size_t sketch_aligned = (data_bytes + 63u) & ~(size_t)63u;
    size_t scales_off = sketch_off + sketch_aligned;
    size_t scales_aligned = (scale_bytes + 63u) & ~(size_t)63u;
    size_t total = scales_off + scales_aligned;
    *out_written = total;
    if (!out_buf) return 0;
    if (out_cap < total) return -1;

    memset(out_buf, 0, total);
    AsiSieveSketchHeader hdr;
    memset(&hdr, 0, sizeof(hdr));
    memcpy(hdr.magic, ASI_SIEVE_MAGIC, 8);
    hdr.mode = (uint32_t)index->mode;
    hdr.vocab_size = index->vocab_size;
    hdr.hidden_size = index->hidden_size;
    hdr.n_tiles = index->n_tiles;
    hdr.quant_type = index->quant_type;
    hdr.is_repacked = (uint32_t)index->is_repacked;
    hdr.sketch_offset = (uint64_t)sketch_off;
    hdr.sketch_bytes = (uint64_t)data_bytes;
    hdr.scales_offset = (uint64_t)scales_off;
    hdr.scales_bytes = (uint64_t)scale_bytes;
    memcpy(out_buf, &hdr, sizeof(hdr));
    memcpy(out_buf + sketch_off, index->sketch_data, data_bytes);
    memcpy(out_buf + scales_off, index->row_scales, scale_bytes);
    return 0;
}

int lila_lhc_get_info(const LilaLhcIndex *index, LilaLhcInfo *out_info) {
    if (!index || !out_info) return -1;
    out_info->mode = index->mode;
    out_info->vocab_size = index->vocab_size;
    out_info->hidden_size = index->hidden_size;
    out_info->n_tiles = index->n_tiles;
    out_info->sketch_bytes_per_row = index->sketch_bytes_per_row;
    out_info->total_sketch_bytes = index->total_sketch_bytes;
    out_info->exact_q6k_bytes = index->exact_q6k_bytes;
    out_info->build_ms = index->build_ms;
    out_info->extra_rss_bytes = index->extra_rss_bytes;
    return 0;
}

/* ───────────────────────── Pre-Allocated Workspace ─────────────────────────── */

LilaLhcWorkspace *lila_lhc_workspace_create(uint32_t vocab_size,
                                            uint32_t top_k,
                                            uint32_t n_threads,
                                            uint32_t max_spec_b) {
    if (vocab_size == 0 || top_k == 0) return NULL;
    if (top_k > vocab_size) top_k = vocab_size;
    if (n_threads == 0) n_threads = 1;
    if (max_spec_b == 0) max_spec_b = 1;
    if (max_spec_b > 8) max_spec_b = 8;

    LilaLhcWorkspace *ws = (LilaLhcWorkspace *)calloc(1, sizeof(LilaLhcWorkspace));
    if (!ws) return NULL;
    ws->vocab_size = vocab_size;
    ws->top_k = top_k;
    ws->n_threads = n_threads;
    ws->max_spec_b = max_spec_b;

    size_t sz_heaps = (size_t)max_spec_b * n_threads * top_k * sizeof(LhcHeapNode);
    size_t sz_hcnts = (size_t)max_spec_b * n_threads * sizeof(uint32_t);
    size_t sz_cands = (size_t)max_spec_b * top_k * sizeof(int32_t);
    size_t sz_ccnts = (size_t)max_spec_b * sizeof(uint32_t);
    size_t sz_luts = (size_t)max_spec_b * sizeof(LhcQueryLut);
    size_t sz_merged = (size_t)max_spec_b * top_k * sizeof(LhcMergedRow);
    size_t sz_qx = (size_t)max_spec_b * 4u * sizeof(LilaQ8KBlock);

    ws->heaps = (LhcHeapNode *)lhc_aligned_alloc(64u, sz_heaps);
    ws->heap_counts = (uint32_t *)calloc(max_spec_b * n_threads, sizeof(uint32_t));
    ws->cands = (int32_t *)lhc_aligned_alloc(64u, sz_cands);
    ws->cand_counts = (uint32_t *)calloc(max_spec_b, sizeof(uint32_t));
    ws->luts = (LhcQueryLut *)lhc_aligned_alloc(64u, sz_luts);
    ws->merged_rows = (LhcMergedRow *)lhc_aligned_alloc(64u, sz_merged);
    ws->qx_lanes = (LilaQ8KBlock *)lhc_aligned_alloc(32u, sz_qx);

    if (!ws->heaps || !ws->heap_counts || !ws->cands || !ws->cand_counts ||
        !ws->luts || !ws->merged_rows || !ws->qx_lanes) {
        lila_lhc_workspace_free(ws);
        return NULL;
    }
    ws->alloc_bytes = sizeof(*ws) + sz_heaps + sz_hcnts + sz_cands + sz_ccnts +
                      sz_luts + sz_merged + sz_qx;
    return ws;
}

void lila_lhc_workspace_free(LilaLhcWorkspace *ws) {
    if (!ws) return;
    lhc_aligned_free(ws->heaps);
    free(ws->heap_counts);
    lhc_aligned_free(ws->cands);
    free(ws->cand_counts);
    lhc_aligned_free(ws->luts);
    lhc_aligned_free(ws->merged_rows);
    lhc_aligned_free(ws->qx_lanes);
    free(ws);
}

size_t lila_lhc_workspace_bytes(const LilaLhcWorkspace *ws) {
    return ws ? ws->alloc_bytes : 0u;
}

int lila_lhc_get_stats(const LilaLhcWorkspace *ws, LilaLhcStats *out_stats) {
    if (!ws || !out_stats) return -1;
    *out_stats = ws->stats;
    return 0;
}

/* ───────────────────────── Activation LUT Builder ──────────────────────────── */

static inline int8_t lhc_clamp_i8(float v) {
    int q = (int)lrintf(v);
    if (q > 127) return 127;
    if (q < -127) return -127;
    return (int8_t)q;
}

static void lhc_build_query_lut(const LilaLhcIndex *index, const float *x_norm, LhcQueryLut *qlut) {
    qlut->n_blocks = index->n_blocks;
    qlut->groups_per_block = index->groups_per_blk;

    if (index->mode == LILA_LHC_MODE_1BIT_512 || index->mode == LILA_LHC_MODE_1BIT_1024) {
        float x_in[1024];
        const float *xp = x_norm;
        if (index->mode == LILA_LHC_MODE_1BIT_512) {
            for (int i = 0; i < 512; ++i) x_in[i] = x_norm[2 * i] + x_norm[2 * i + 1];
            xp = x_in;
        }
        for (uint32_t b = 0; b < index->n_blocks; ++b) {
            uint32_t g0 = b * 16u;
            float smax = 0.0f;
            for (uint32_t g = 0; g < 16u; ++g) {
                const float *xg = xp + (size_t)(g0 + g) * 4u;
                float s = fabsf(xg[0]) + fabsf(xg[1]) + fabsf(xg[2]) + fabsf(xg[3]);
                if (s > smax) smax = s;
            }
            qlut->b_scale[b] = smax * (1.0f / 127.0f);
            float inv = (smax > 0.0f) ? (127.0f / smax) : 0.0f;
            for (uint32_t g = 0; g < 16u; ++g) {
                const float *xg = xp + (size_t)(g0 + g) * 4u;
                float x0 = xg[0], x1 = xg[1], x2 = xg[2], x3 = xg[3];
                int8_t *dst = qlut->lut[g0 + g];
                for (int mask = 0; mask < 16; ++mask) {
                    float dot = ((mask & 1) ? x0 : -x0)
                              + ((mask & 2) ? x1 : -x1)
                              + ((mask & 4) ? x2 : -x2)
                              + ((mask & 8) ? x3 : -x3);
                    dst[mask] = lhc_clamp_i8(dot * inv);
                }
            }
        }
    } else {
        /* LILA_LHC_MODE_2BIT_1024: 8 blocks x 128 elements (32 packed bytes/blk) */
        qlut->bias_x = 0.0f;
        for (int blk = 0; blk < 8; ++blk) {
            const float *xb = x_norm + blk * 128;
            float amax = 0.0f;
            for (int i = 0; i < 128; ++i) {
                float av = fabsf(xb[i]);
                if (av > amax) amax = av;
            }
            float dx = amax * (1.0f / 127.0f);
            float idx_s = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
            qlut->dx8[blk] = dx;
            int32_t sum_qx = 0;
            for (int l = 0; l < 32; ++l) {
                for (int p = 0; p < 4; ++p) {
                    int8_t q = lhc_clamp_i8(xb[l * 4 + p] * idx_s);
                    qlut->qx_planes[blk * 128 + p * 32 + l] = q;
                    sum_qx += (int32_t)q;
                }
            }
            qlut->bias_x += 1.5f * dx * (float)sum_qx;
        }
    }
}

/* ─────────────────────────── Min-Heap + Sort ────────────────────────────────── */

static inline int lhc_cand_better(float s1, int32_t id1, float s2, int32_t id2) {
    return (s1 > s2) || (s1 == s2 && id1 < id2);
}

static inline int lhc_heap_worse(LhcHeapNode a, LhcHeapNode b) {
    return (a.score < b.score) || (a.score == b.score && a.id > b.id);
}

static inline void lhc_heap_sift_down(LhcHeapNode *heap, uint32_t k, uint32_t idx) {
    for (;;) {
        uint32_t left = 2u * idx + 1u;
        uint32_t right = left + 1u;
        uint32_t smallest = idx;
        if (left < k && lhc_heap_worse(heap[left], heap[smallest])) smallest = left;
        if (right < k && lhc_heap_worse(heap[right], heap[smallest])) smallest = right;
        if (smallest == idx) break;
        LhcHeapNode tmp = heap[idx];
        heap[idx] = heap[smallest];
        heap[smallest] = tmp;
        idx = smallest;
    }
}

static inline void lhc_heap_push(LhcHeapNode *heap, uint32_t *count, uint32_t top_k,
                                 float score, int32_t id) {
    if (!isfinite(score)) return;
    uint32_t cnt = *count;
    if (cnt < top_k) {
        heap[cnt].score = score;
        heap[cnt].id = id;
        cnt++;
        *count = cnt;
        if (cnt == top_k) {
            for (int32_t i = (int32_t)(top_k / 2u) - 1; i >= 0; --i) {
                lhc_heap_sift_down(heap, top_k, (uint32_t)i);
            }
        }
    } else if (lhc_cand_better(score, id, heap[0].score, heap[0].id)) {
        heap[0].score = score;
        heap[0].id = id;
        lhc_heap_sift_down(heap, top_k, 0u);
    }
}

static void lhc_sort_int32_asc(int32_t *arr, uint32_t n) {
    if (n <= 1u) return;
    /* In-place max-heap sort: zero stack/heap allocations, O(n log n) worst case */
    for (int32_t start = (int32_t)(n / 2u) - 1; start >= 0; --start) {
        uint32_t root = (uint32_t)start;
        for (;;) {
            uint32_t child = 2u * root + 1u;
            if (child >= n) break;
            if (child + 1u < n && arr[child] < arr[child + 1u]) child++;
            if (arr[root] >= arr[child]) break;
            int32_t tmp = arr[root];
            arr[root] = arr[child];
            arr[child] = tmp;
            root = child;
        }
    }
    for (uint32_t end = n - 1u; end > 0u; --end) {
        int32_t tmp = arr[0];
        arr[0] = arr[end];
        arr[end] = tmp;
        uint32_t root = 0u;
        for (;;) {
            uint32_t child = 2u * root + 1u;
            if (child >= end) break;
            if (child + 1u < end && arr[child] < arr[child + 1u]) child++;
            if (arr[root] >= arr[child]) break;
            int32_t t = arr[root];
            arr[root] = arr[child];
            arr[child] = t;
            root = child;
        }
    }
}

static uint32_t lhc_merge_thread_heaps(LhcHeapNode *heaps_base,
                                       uint32_t *counts_base,
                                       uint32_t n_threads,
                                       uint32_t top_k,
                                       int32_t *out_cands) {
    uint32_t c0 = counts_base[0];
    if (n_threads <= 1u) {
        for (uint32_t i = 0; i < c0; ++i) out_cands[i] = heaps_base[i].id;
        lhc_sort_int32_asc(out_cands, c0);
        return c0;
    }

    LhcHeapNode *h0 = heaps_base;
    /* If thread 0 accumulated fewer than top_k elements, top up before heapifying */
    for (uint32_t t = 1u; t < n_threads; ++t) {
        uint32_t ct = counts_base[t];
        const LhcHeapNode *ht = heaps_base + (size_t)t * top_k;
        for (uint32_t i = 0; i < ct; ++i) {
            if (c0 < top_k) {
                h0[c0++] = ht[i];
                if (c0 == top_k) {
                    for (int32_t p = (int32_t)(top_k / 2u) - 1; p >= 0; --p) {
                        lhc_heap_sift_down(h0, top_k, (uint32_t)p);
                    }
                }
            } else if (lhc_cand_better(ht[i].score, ht[i].id, h0[0].score, h0[0].id)) {
                h0[0] = ht[i];
                lhc_heap_sift_down(h0, top_k, 0u);
            }
        }
    }

    for (uint32_t i = 0; i < c0; ++i) out_cands[i] = h0[i].id;
    lhc_sort_int32_asc(out_cands, c0);
    return c0;
}

/* ─────────────────────────── Stage-1 Tile Scoring ──────────────────────────── */

#if defined(__AVX2__) && defined(__F16C__)
static inline void lhc_score_tile_avx2(const LilaLhcIndex *index,
                                       const LhcQueryLut *qlut,
                                       uint32_t tile,
                                       __m256 *out_f) {
    const uint8_t *tp = index->sketch_data + (size_t)tile * index->tile_bytes;
    const __m256i mask_0f = _mm256_set1_epi8(0x0F);
    const uint32_t n_blocks = qlut->n_blocks;
    const uint32_t gpb = qlut->groups_per_block;

    __m256 f0 = _mm256_setzero_ps(), f1 = _mm256_setzero_ps();
    __m256 f2 = _mm256_setzero_ps(), f3 = _mm256_setzero_ps();
    __m256 f4 = _mm256_setzero_ps(), f5 = _mm256_setzero_ps();
    __m256 f6 = _mm256_setzero_ps(), f7 = _mm256_setzero_ps();

    for (uint32_t b = 0; b < n_blocks; ++b) {
        __m256i acc_lo0 = _mm256_setzero_si256();
        __m256i acc_lo1 = _mm256_setzero_si256();
        __m256i acc_hi0 = _mm256_setzero_si256();
        __m256i acc_hi1 = _mm256_setzero_si256();
        const uint32_t g_base = b * gpb;

        for (uint32_t g = 0; g < gpb; ++g) {
            __m256i packed = _mm256_load_si256((const __m256i *)tp);
            tp += 32;
            __m256i lo_idx = _mm256_and_si256(packed, mask_0f);
            __m256i hi_idx = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask_0f);
            __m256i v_lut = _mm256_broadcastsi128_si256(
                _mm_load_si128((const __m128i *)qlut->lut[g_base + g]));
            __m256i s_lo = _mm256_shuffle_epi8(v_lut, lo_idx);
            __m256i s_hi = _mm256_shuffle_epi8(v_lut, hi_idx);

            acc_lo0 = _mm256_add_epi16(acc_lo0, _mm256_cvtepi8_epi16(_mm256_castsi256_si128(s_lo)));
            acc_lo1 = _mm256_add_epi16(acc_lo1, _mm256_cvtepi8_epi16(_mm256_extracti128_si256(s_lo, 1)));
            acc_hi0 = _mm256_add_epi16(acc_hi0, _mm256_cvtepi8_epi16(_mm256_castsi256_si128(s_hi)));
            acc_hi1 = _mm256_add_epi16(acc_hi1, _mm256_cvtepi8_epi16(_mm256_extracti128_si256(s_hi, 1)));
        }

        __m256 vbs = _mm256_set1_ps(qlut->b_scale[b]);
        f0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(acc_lo0))), vbs, f0);
        f1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(acc_lo0, 1))), vbs, f1);
        f2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(acc_lo1))), vbs, f2);
        f3 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(acc_lo1, 1))), vbs, f3);
        f4 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(acc_hi0))), vbs, f4);
        f5 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(acc_hi0, 1))), vbs, f5);
        f6 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(acc_hi1))), vbs, f6);
        f7 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(acc_hi1, 1))), vbs, f7);
    }

    const uint16_t *sp = index->row_scales + (size_t)tile * LHC_TILE_ROWS;
    out_f[0] = _mm256_mul_ps(f0, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(sp + 0))));
    out_f[1] = _mm256_mul_ps(f1, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(sp + 8))));
    out_f[2] = _mm256_mul_ps(f2, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(sp + 16))));
    out_f[3] = _mm256_mul_ps(f3, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(sp + 24))));
    out_f[4] = _mm256_mul_ps(f4, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(sp + 32))));
    out_f[5] = _mm256_mul_ps(f5, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(sp + 40))));
    out_f[6] = _mm256_mul_ps(f6, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(sp + 48))));
    out_f[7] = _mm256_mul_ps(f7, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(sp + 56))));
}

static inline __m256 lhc_score_2bit_8rows_avx2(const LilaLhcIndex *index,
                                               const LhcQueryLut *qlut,
                                               uint32_t tile) {
    const uint8_t *tp = index->sketch_data + (size_t)tile * 2048u;
    const __m256i m03 = _mm256_set1_epi8(0x03);
    const __m256i ones16 = _mm256_set1_epi16(1);
    float raw[8];
    for (int pair = 0; pair < 4; ++pair) {
        const uint8_t *r0 = tp + (size_t)(pair * 2 + 0) * 256u;
        const uint8_t *r1 = r0 + 256u;
        __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
        for (int blk = 0; blk < 8; ++blk) {
            const int8_t *xp = qlut->qx_planes + blk * 128;
            __m256i x0 = _mm256_load_si256((const __m256i *)(xp + 0));
            __m256i x1 = _mm256_load_si256((const __m256i *)(xp + 32));
            __m256i x2 = _mm256_load_si256((const __m256i *)(xp + 64));
            __m256i x3 = _mm256_load_si256((const __m256i *)(xp + 96));

            __m256i p0 = _mm256_load_si256((const __m256i *)(r0 + blk * 32));
            __m256i p1 = _mm256_load_si256((const __m256i *)(r1 + blk * 32));

            __m256i a01 = _mm256_add_epi16(
                _mm256_maddubs_epi16(_mm256_and_si256(p0, m03), x0),
                _mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(p0, 2), m03), x1));
            __m256i a23 = _mm256_add_epi16(
                _mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(p0, 4), m03), x2),
                _mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(p0, 6), m03), x3));
            __m256i b01 = _mm256_add_epi16(
                _mm256_maddubs_epi16(_mm256_and_si256(p1, m03), x0),
                _mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(p1, 2), m03), x1));
            __m256i b23 = _mm256_add_epi16(
                _mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(p1, 4), m03), x2),
                _mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(p1, 6), m03), x3));

            __m256 vdx = _mm256_set1_ps(qlut->dx8[blk]);
            acc0 = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_add_epi16(a01, a23), ones16)), vdx, acc0);
            acc1 = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_add_epi16(b01, b23), ones16)), vdx, acc1);
        }
        raw[pair * 2 + 0] = lila_hsum_f32_avx2(acc0) - qlut->bias_x;
        raw[pair * 2 + 1] = lila_hsum_f32_avx2(acc1) - qlut->bias_x;
    }
    const uint16_t *sp = index->row_scales + (size_t)tile * 8u;
    __m256 vs = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)sp));
    return _mm256_mul_ps(_mm256_loadu_ps(raw), vs);
}
#else
static void lhc_score_tile_scalar(const LilaLhcIndex *index,
                                  const LhcQueryLut *qlut,
                                  uint32_t tile,
                                  uint32_t valid,
                                  float *out_scores) {
    const uint8_t *tp = index->sketch_data + (size_t)tile * index->tile_bytes;
    const uint16_t *sp = index->row_scales + (size_t)tile * index->tile_rows;
    if (index->mode == LILA_LHC_MODE_2BIT_1024) {
        for (uint32_t lr = 0; lr < valid; ++lr) {
            const uint8_t *rp = tp + (size_t)lr * 256u;
            float acc = 0.0f;
            for (int blk = 0; blk < 8; ++blk) {
                int32_t s = 0;
                for (int l = 0; l < 32; ++l) {
                    uint8_t p = rp[blk * 32 + l];
                    for (int k = 0; k < 4; ++k) {
                        uint8_t code = (uint8_t)((p >> (2 * k)) & 3u);
                        s += (int32_t)code * (int32_t)qlut->qx_planes[blk * 128 + k * 32 + l];
                    }
                }
                acc += (float)s * qlut->dx8[blk];
            }
            out_scores[lr] = fp16_to_fp32(sp[lr]) * (acc - qlut->bias_x);
        }
        return;
    }
    const uint32_t n_blocks = qlut->n_blocks;
    const uint32_t gpb = qlut->groups_per_block;

    for (uint32_t lr = 0; lr < valid; ++lr) out_scores[lr] = 0.0f;

    for (uint32_t b = 0; b < n_blocks; ++b) {
        int32_t acc[LHC_TILE_ROWS] = {0};
        const uint32_t g_base = b * gpb;
        for (uint32_t g = 0; g < gpb; ++g) {
            const uint8_t *gp = tp + (size_t)(g_base + g) * 32u;
            const int8_t *lut_g = qlut->lut[g_base + g];
            for (uint32_t lr = 0; lr < valid; ++lr) {
                uint8_t byte = gp[lr & 31u];
                uint8_t nib = (lr >= 32u) ? (uint8_t)(byte >> 4) : (uint8_t)(byte & 0x0Fu);
                acc[lr] += (int32_t)lut_g[nib];
            }
        }
        float bs = qlut->b_scale[b];
        for (uint32_t lr = 0; lr < valid; ++lr) {
            out_scores[lr] += (float)acc[lr] * bs;
        }
    }
    for (uint32_t lr = 0; lr < valid; ++lr) {
        out_scores[lr] *= fp16_to_fp32(sp[lr]);
    }
}
#endif

typedef struct {
    const LilaLhcIndex *index;
    const LhcQueryLut *qluts;
    uint32_t B;
    LilaLhcWorkspace *ws;      /* non-NULL for fused top-K; NULL for full-score out */
    float *out_coarse_scores;  /* non-NULL only when ws == NULL */
    _Alignas(64) atomic_uint next_tile;
} LhcStage1Ctx;

static void lhc_stage1_worker(int tid, int n_threads, void *vctx) {
    (void)n_threads;
    LhcStage1Ctx *ctx = (LhcStage1Ctx *)vctx;
    const LilaLhcIndex *index = ctx->index;
    const uint32_t V = index->vocab_size;
    const uint32_t n_tiles = index->n_tiles;
    const uint32_t tile_rows = index->tile_rows;
    const uint32_t B = ctx->B;
    const uint32_t chunk = (tile_rows == 8u) ? 64u : 8u;
    LilaLhcWorkspace *ws = ctx->ws;

    uint32_t safe_tid = (ws && (uint32_t)tid < ws->n_threads) ? (uint32_t)tid : 0u;
    const uint32_t top_k = ws ? ws->top_k : 0u;

    for (;;) {
        uint32_t start = atomic_fetch_add_explicit(&ctx->next_tile, chunk, memory_order_relaxed);
        if (start >= n_tiles) break;
        uint32_t end = start + chunk;
        if (end > n_tiles) end = n_tiles;

        for (uint32_t t = start; t < end; ++t) {
            if (t + 2u < n_tiles) {
                const uint8_t *pf = index->sketch_data + (size_t)(t + 2u) * index->tile_bytes;
                LILA_PREFETCH(pf);
                if (index->tile_bytes >= 4096u) LILA_PREFETCH(pf + 2048u);
            }
            uint32_t row_base = t * tile_rows;
            uint32_t valid = (row_base + tile_rows <= V) ? tile_rows : (V - row_base);

            for (uint32_t lane = 0; lane < B; ++lane) {
                const LhcQueryLut *qlut = &ctx->qluts[lane];
                if (!ws) {
                    _Alignas(32) float scores[LHC_TILE_ROWS];
#if defined(__AVX2__) && defined(__F16C__)
                    if (index->mode == LILA_LHC_MODE_2BIT_1024) {
                        __m256 fv = lhc_score_2bit_8rows_avx2(index, qlut, t);
                        _mm256_store_ps(scores, fv);
                    } else {
                        __m256 fv[8];
                        lhc_score_tile_avx2(index, qlut, t, fv);
                        for (int k = 0; k < 8; ++k) _mm256_store_ps(scores + k * 8, fv[k]);
                    }
#else
                    lhc_score_tile_scalar(index, qlut, t, valid, scores);
#endif
                    memcpy(ctx->out_coarse_scores + row_base, scores, (size_t)valid * sizeof(float));
                } else {
                    size_t slot = (size_t)lane * ws->n_threads + safe_tid;
                    LhcHeapNode *heap = ws->heaps + slot * top_k;
                    uint32_t *hcnt = &ws->heap_counts[slot];
                    _Alignas(32) float scores[LHC_TILE_ROWS];

#if defined(__AVX2__) && defined(__F16C__)
                    if (index->mode == LILA_LHC_MODE_2BIT_1024) {
                        __m256 fv = lhc_score_2bit_8rows_avx2(index, qlut, t);
                        if (*hcnt >= top_k && valid == 8u) {
                            __m256 vmin = _mm256_set1_ps(heap[0].score);
                            int mm = _mm256_movemask_ps(_mm256_cmp_ps(fv, vmin, _CMP_GE_OQ));
                            if (mm == 0) continue;
                            _mm256_store_ps(scores, fv);
                            while (mm) {
                                int bit = __builtin_ctz((unsigned)mm);
                                mm &= (mm - 1);
                                lhc_heap_push(heap, hcnt, top_k, scores[bit], (int32_t)(row_base + (uint32_t)bit));
                            }
                            continue;
                        }
                        _mm256_store_ps(scores, fv);
                    } else {
                        __m256 fv[8];
                        lhc_score_tile_avx2(index, qlut, t, fv);
                        if (*hcnt >= top_k && valid == LHC_TILE_ROWS) {
                            __m256 vmin = _mm256_set1_ps(heap[0].score);
                            int m0 = _mm256_movemask_ps(_mm256_cmp_ps(fv[0], vmin, _CMP_GE_OQ));
                            int m1 = _mm256_movemask_ps(_mm256_cmp_ps(fv[1], vmin, _CMP_GE_OQ));
                            int m2 = _mm256_movemask_ps(_mm256_cmp_ps(fv[2], vmin, _CMP_GE_OQ));
                            int m3 = _mm256_movemask_ps(_mm256_cmp_ps(fv[3], vmin, _CMP_GE_OQ));
                            int m4 = _mm256_movemask_ps(_mm256_cmp_ps(fv[4], vmin, _CMP_GE_OQ));
                            int m5 = _mm256_movemask_ps(_mm256_cmp_ps(fv[5], vmin, _CMP_GE_OQ));
                            int m6 = _mm256_movemask_ps(_mm256_cmp_ps(fv[6], vmin, _CMP_GE_OQ));
                            int m7 = _mm256_movemask_ps(_mm256_cmp_ps(fv[7], vmin, _CMP_GE_OQ));
                            int masks[8] = {m0, m1, m2, m3, m4, m5, m6, m7};
                            if ((m0 | m1 | m2 | m3 | m4 | m5 | m6 | m7) == 0) continue;
                            for (int k = 0; k < 8; ++k) {
                                int mm = masks[k];
                                if (mm == 0) continue;
                                _mm256_store_ps(scores + k * 8, fv[k]);
                                while (mm) {
                                    int bit = __builtin_ctz((unsigned)mm);
                                    mm &= (mm - 1);
                                    uint32_t lr = (uint32_t)(k * 8 + bit);
                                    lhc_heap_push(heap, hcnt, top_k, scores[lr], (int32_t)(row_base + lr));
                                }
                            }
                            continue;
                        }
                        for (int k = 0; k < 8; ++k) _mm256_store_ps(scores + k * 8, fv[k]);
                    }
#else
                    lhc_score_tile_scalar(index, qlut, t, valid, scores);
#endif
                    for (uint32_t lr = 0; lr < valid; ++lr) {
                        lhc_heap_push(heap, hcnt, top_k, scores[lr], (int32_t)(row_base + lr));
                    }
                }
            }
        }
    }
}

int lila_lhc_score_coarse_pool(const LilaLhcIndex *index,
                               const float *x_norm,
                               float *out_coarse_scores,
                               LilaLhcDispatchFn dispatch_fn,
                               void *pool_opaque) {
    if (!index || !x_norm || !out_coarse_scores) return -1;
    LhcQueryLut qlut;
    lhc_build_query_lut(index, x_norm, &qlut);

    LhcStage1Ctx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.index = index;
    ctx.qluts = &qlut;
    ctx.B = 1u;
    ctx.ws = NULL;
    ctx.out_coarse_scores = out_coarse_scores;
    atomic_init(&ctx.next_tile, 0u);

    if (dispatch_fn && index->n_tiles >= 4u) {
        dispatch_fn(pool_opaque, lhc_stage1_worker, &ctx);
    } else {
        lhc_stage1_worker(0, 1, &ctx);
    }
    return 0;
}

int lila_lhc_score_coarse(const LilaLhcIndex *index,
                          const float *x_norm,
                          float *out_coarse_scores) {
    return lila_lhc_score_coarse_pool(index, x_norm, out_coarse_scores, NULL, NULL);
}

int lila_lhc_select_topk(const float *coarse_scores,
                         uint32_t vocab_size,
                         uint32_t top_k,
                         int32_t *out_candidate_ids) {
    if (!coarse_scores || !out_candidate_ids || vocab_size == 0 || top_k == 0) return -1;
    if (top_k >= vocab_size) {
        for (uint32_t i = 0; i < vocab_size; ++i) out_candidate_ids[i] = (int32_t)i;
        return 0;
    }

    LhcHeapNode *heap = (LhcHeapNode *)malloc((size_t)top_k * sizeof(LhcHeapNode));
    if (!heap) return -1;
    uint32_t count = 0u;
    for (uint32_t i = 0; i < vocab_size; ++i) {
        lhc_heap_push(heap, &count, top_k, coarse_scores[i], (int32_t)i);
    }
    for (uint32_t i = 0; i < count; ++i) out_candidate_ids[i] = heap[i].id;
    free(heap);
    lhc_sort_int32_asc(out_candidate_ids, count);
    return (count == top_k) ? 0 : -1;
}

/* ─────────────────────────── Stage-2 Exact Q6_K ────────────────────────────── */

static inline void lhc_fill_neg_logits(float *dst, uint32_t vocab_size) {
    uint32_t i = 0u;
#if defined(__AVX2__)
    const __m256 vneg = _mm256_set1_ps(LHC_NEG_FILL);
    for (; i + 31u < vocab_size; i += 32u) {
        _mm256_storeu_ps(dst + i + 0u, vneg);
        _mm256_storeu_ps(dst + i + 8u, vneg);
        _mm256_storeu_ps(dst + i + 16u, vneg);
        _mm256_storeu_ps(dst + i + 24u, vneg);
    }
    for (; i + 7u < vocab_size; i += 8u) {
        _mm256_storeu_ps(dst + i, vneg);
    }
#endif
    for (; i < vocab_size; ++i) dst[i] = LHC_NEG_FILL;
}

static inline void lhc_prefetch_q6k_row(const uint8_t *rp, size_t row_bytes) {
    LILA_PREFETCH(rp);
    if (row_bytes > 256u) LILA_PREFETCH(rp + 256u);
    if (row_bytes > 512u) LILA_PREFETCH(rp + 512u);
    if (row_bytes > 768u) LILA_PREFETCH(rp + 768u);
}

typedef struct {
    const uint8_t *q6k_weights;
    const LilaQ8KBlock *qx;
    const float *x_norm;
    const int32_t *cand_ids;
    float *out_logits;
    uint32_t count;
    int hidden_size;
    int bpr;
    size_t row_bytes;
    uint32_t quant_type;
    int is_repacked;
    int fp32_ref;
    _Alignas(64) atomic_uint next_idx;
} LhcStage2Ctx;

static void lhc_stage2_worker(int tid, int n_threads, void *vctx) {
    (void)tid;
    LhcStage2Ctx *ctx = (LhcStage2Ctx *)vctx;
    const uint8_t *w = ctx->q6k_weights;
    const int32_t *cands = ctx->cand_ids;
    float *out = ctx->out_logits;
    const size_t rb = ctx->row_bytes;
    const int bpr = ctx->bpr;
    const uint32_t cnt = ctx->count;

    for (;;) {
        uint32_t start = (n_threads <= 1)
                             ? 0u
                             : atomic_fetch_add_explicit(&ctx->next_idx, 64u, memory_order_relaxed);
        if (start >= cnt) break;
        uint32_t end = (n_threads <= 1) ? cnt : (start + 64u);
        if (end > cnt) end = cnt;

        if (!ctx->fp32_ref && ctx->quant_type == 14u && !ctx->is_repacked) {
            uint32_t k = start;
#if defined(__AVX2__)
            for (; k + 1u < end; k += 2u) {
                int32_t id0 = cands[k];
                int32_t id1 = cands[k + 1u];
                const uint8_t *rp0 = w + (size_t)id0 * rb;
                const uint8_t *rp1 = w + (size_t)id1 * rb;
                if (k + 3u < end) {
                    lhc_prefetch_q6k_row(w + (size_t)cands[k + 2u] * rb, rb);
                    lhc_prefetch_q6k_row(w + (size_t)cands[k + 3u] * rb, rb);
                }
                float v0 = 0.0f, v1 = 0.0f;
                q6k_dot2_q8k_avx2(rp0, rp1, ctx->qx, bpr, &v0, &v1);
                out[id0] = v0;
                out[id1] = v1;
            }
#endif
            for (; k < end; ++k) {
                int32_t id = cands[k];
                if (k + 2u < end) {
                    lhc_prefetch_q6k_row(w + (size_t)cands[k + 2u] * rb, rb);
                }
                out[id] = q6k_dot_q8k_simd(w + (size_t)id * rb, ctx->qx, bpr);
            }
        } else if (!ctx->fp32_ref && ctx->quant_type == 12u) {
            uint32_t k = start;
#if defined(__AVX2__)
            for (; k + 1u < end; k += 2u) {
                int32_t id0 = cands[k];
                int32_t id1 = cands[k + 1u];
                const uint8_t *rp0 = w + (size_t)id0 * rb;
                const uint8_t *rp1 = w + (size_t)id1 * rb;
                if (k + 3u < end) {
                    lhc_prefetch_q6k_row(w + (size_t)cands[k + 2u] * rb, rb);
                    lhc_prefetch_q6k_row(w + (size_t)cands[k + 3u] * rb, rb);
                }
                float v0 = 0.0f, v1 = 0.0f;
                q4k_dot2_q8k_avx2(rp0, rp1, ctx->qx, bpr, &v0, &v1);
                out[id0] = v0;
                out[id1] = v1;
            }
#endif
            for (; k < end; ++k) {
                int32_t id = cands[k];
                if (k + 2u < end) {
                    lhc_prefetch_q6k_row(w + (size_t)cands[k + 2u] * rb, rb);
                }
                out[id] = q4k_dot_q8k_simd(w + (size_t)id * rb, ctx->qx, bpr);
            }
        } else {
            for (uint32_t k = start; k < end; ++k) {
                int32_t id = cands[k];
                if (k + 2u < end) {
                    lhc_prefetch_q6k_row(w + (size_t)cands[k + 2u] * rb, rb);
                }
                out[id] = lila_quant_dot_fp32(w + (size_t)id * rb, ctx->x_norm,
                                              ctx->hidden_size, ctx->quant_type, ctx->is_repacked);
            }
        }
        if (n_threads <= 1) break;
    }
}

extern int g_e05_hook_fast_flag;
extern int e05_lhc_forward_pool_intercept(const LilaLhcIndex *index,
                                          LilaLhcWorkspace *ws,
                                          LilaLhcDispatchFn dispatch_fn,
                                          void *pool_opaque,
                                          const uint8_t *q6k_weights,
                                          const float *x_norm,
                                          float *out_logits,
                                          int fp32_ref);

int lila_lhc_forward_pool_raw(const LilaLhcIndex *index,
                              LilaLhcWorkspace *ws,
                              LilaLhcDispatchFn dispatch_fn,
                              void *pool_opaque,
                              const uint8_t *q6k_weights,
                              const float *x_norm,
                              float *out_logits,
                              int fp32_ref) {
    if (!index || !ws || !q6k_weights || !x_norm || !out_logits) return -1;
    const uint32_t V = index->vocab_size;
    const uint32_t H = index->hidden_size;

    double t0 = lhc_now_ms();
    lhc_build_query_lut(index, x_norm, &ws->luts[0]);
    memset(ws->heap_counts, 0, (size_t)ws->n_threads * sizeof(uint32_t));

    LhcStage1Ctx s1ctx;
    memset(&s1ctx, 0, sizeof(s1ctx));
    s1ctx.index = index;
    s1ctx.qluts = &ws->luts[0];
    s1ctx.B = 1u;
    s1ctx.ws = ws;
    atomic_init(&s1ctx.next_tile, 0u);

    if (dispatch_fn && index->n_tiles >= 4u) {
        dispatch_fn(pool_opaque, lhc_stage1_worker, &s1ctx);
    } else {
        lhc_stage1_worker(0, 1, &s1ctx);
    }

    uint32_t cand_count = lhc_merge_thread_heaps(ws->heaps, ws->heap_counts,
                                                 ws->n_threads, ws->top_k, ws->cands);
    double t1 = lhc_now_ms();
    ws->stats.stage1_ms_sum += (t1 - t0);

    lhc_fill_neg_logits(out_logits, V);

    int bpr = (int)(H / QK_K);
    int need_q8k = (!fp32_ref && (index->quant_type == 12u || (index->quant_type == 14u && !index->is_repacked)));
    if (need_q8k) {
        quantize_vec_q8k(&ws->qx_lanes[0], x_norm, (int)H);
    }

    LhcStage2Ctx s2ctx;
    s2ctx.q6k_weights = q6k_weights;
    s2ctx.qx = need_q8k ? &ws->qx_lanes[0] : NULL;
    s2ctx.x_norm = x_norm;
    s2ctx.cand_ids = ws->cands;
    s2ctx.out_logits = out_logits;
    s2ctx.count = cand_count;
    s2ctx.hidden_size = (int)H;
    s2ctx.bpr = bpr;
    s2ctx.row_bytes = index->row_bytes;
    s2ctx.quant_type = index->quant_type;
    s2ctx.is_repacked = index->is_repacked;
    s2ctx.fp32_ref = fp32_ref;
    atomic_init(&s2ctx.next_idx, 0u);

    if (dispatch_fn && cand_count >= 128u) {
        dispatch_fn(pool_opaque, lhc_stage2_worker, &s2ctx);
    } else {
        lhc_stage2_worker(0, 1, &s2ctx);
    }

    double t2 = lhc_now_ms();
    ws->stats.stage2_ms_sum += (t2 - t1);
    ws->stats.eval_calls++;
    ws->stats.total_candidates_eval += cand_count;
    return 0;
}

int lila_lhc_forward_pool(const LilaLhcIndex *index,
                          LilaLhcWorkspace *ws,
                          LilaLhcDispatchFn dispatch_fn,
                          void *pool_opaque,
                          const uint8_t *q6k_weights,
                          const float *x_norm,
                          float *out_logits,
                          int fp32_ref) {
    if (__builtin_expect(g_e05_hook_fast_flag != 0, 0)) {
        return e05_lhc_forward_pool_intercept(index, ws, dispatch_fn, pool_opaque,
                                              q6k_weights, x_norm, out_logits, fp32_ref);
    }
    return lila_lhc_forward_pool_raw(index, ws, dispatch_fn, pool_opaque,
                                     q6k_weights, x_norm, out_logits, fp32_ref);
}

/* ───────────────────── Speculative Multi-Lane Batch Stage-2 ────────────────── */

static uint32_t lhc_merge_sorted_lanes(LilaLhcWorkspace *ws, uint32_t B) {
    uint32_t pos[8] = {0};
    uint32_t out_n = 0u;
    LhcMergedRow *out = ws->merged_rows;

    for (;;) {
        int32_t min_id = INT32_MAX;
        int found = 0;
        for (uint32_t b = 0; b < B; ++b) {
            if (pos[b] < ws->cand_counts[b]) {
                int32_t id = ws->cands[(size_t)b * ws->top_k + pos[b]];
                if (!found || id < min_id) {
                    min_id = id;
                    found = 1;
                }
            }
        }
        if (!found) break;
        uint8_t mask = 0u;
        for (uint32_t b = 0; b < B; ++b) {
            if (pos[b] < ws->cand_counts[b] &&
                ws->cands[(size_t)b * ws->top_k + pos[b]] == min_id) {
                mask |= (uint8_t)(1u << b);
                pos[b]++;
            }
        }
        out[out_n].id = min_id;
        out[out_n].mask = mask;
        out_n++;
    }
    return out_n;
}

typedef struct {
    const uint8_t *q6k_weights;
    const LilaQ8KBlock *qx_all;
    const LilaQ8KBlock *qx_lanes; /* [B * bpr] contiguous per lane */
    const float *xnb;
    const LhcMergedRow *merged_rows;
    float *out_ptrs[4];
    uint32_t u_count;
    uint32_t B;
    int hidden_size;
    int bpr;
    size_t row_bytes;
    uint32_t quant_type;
    int is_repacked;
    int fp32_ref;
    _Alignas(64) atomic_uint next_idx;
} LhcStage2SpecCtx;

static void lhc_stage2_spec_worker(int tid, int n_threads, void *vctx) {
    (void)tid;
    LhcStage2SpecCtx *ctx = (LhcStage2SpecCtx *)vctx;
    const uint8_t *w = ctx->q6k_weights;
    const LhcMergedRow *rows = ctx->merged_rows;
    const size_t rb = ctx->row_bytes;
    const int bpr = ctx->bpr;
    const uint32_t B = ctx->B;
    const uint32_t cnt = ctx->u_count;
    const uint8_t all_mask = (uint8_t)((1u << B) - 1u);

    for (;;) {
        uint32_t start = (n_threads <= 1)
                             ? 0u
                             : atomic_fetch_add_explicit(&ctx->next_idx, 32u, memory_order_relaxed);
        if (start >= cnt) break;
        uint32_t end = (n_threads <= 1) ? cnt : (start + 32u);
        if (end > cnt) end = cnt;

        for (uint32_t k = start; k < end; ++k) {
            int32_t id = rows[k].id;
            uint8_t mask = rows[k].mask;
            const uint8_t *rp = w + (size_t)id * rb;
            if (k + 2u < end) {
                lhc_prefetch_q6k_row(w + (size_t)rows[k + 2u].id * rb, rb);
            }

            if (!ctx->fp32_ref && ctx->quant_type == 14u && !ctx->is_repacked) {
#if defined(__AVX2__)
                if (mask == all_mask && ctx->qx_all && B >= 2u && B <= 4u) {
                    if (B == 4u) lila_q6k_1xK_reg_avx2(rp, ctx->qx_all, 4, 0, 4, bpr, ctx->out_ptrs, id);
                    else if (B == 3u) lila_q6k_1xK_reg_avx2(rp, ctx->qx_all, 3, 0, 3, bpr, ctx->out_ptrs, id);
                    else lila_q6k_1xK_reg_avx2(rp, ctx->qx_all, 2, 0, 2, bpr, ctx->out_ptrs, id);
                    continue;
                }
#endif
                for (uint32_t b = 0; b < B; ++b) {
                    if (mask & (uint8_t)(1u << b)) {
                        ctx->out_ptrs[b][id] = q6k_dot_q8k_simd(
                            rp, ctx->qx_lanes + (size_t)b * bpr, bpr);
                    }
                }
            } else if (!ctx->fp32_ref && ctx->quant_type == 12u) {
                for (uint32_t b = 0; b < B; ++b) {
                    if (mask & (uint8_t)(1u << b)) {
                        ctx->out_ptrs[b][id] = q4k_dot_q8k_simd(
                            rp, ctx->qx_lanes + (size_t)b * bpr, bpr);
                    }
                }
            } else {
                for (uint32_t b = 0; b < B; ++b) {
                    if (mask & (uint8_t)(1u << b)) {
                        ctx->out_ptrs[b][id] = lila_quant_dot_fp32(
                            rp, ctx->xnb + (size_t)b * ctx->hidden_size,
                            ctx->hidden_size, ctx->quant_type, ctx->is_repacked);
                    }
                }
            }
        }
        if (n_threads <= 1) break;
    }
}

int lila_lhc_forward_spec_pool(const LilaLhcIndex *index,
                               LilaLhcWorkspace *ws,
                               LilaLhcDispatchFn dispatch_fn,
                               void *pool_opaque,
                               const uint8_t *q6k_weights,
                               const float *xnb,
                               const LilaQ8KBlock *qx_all,
                               uint32_t B,
                               float *out0,
                               float *out_rest,
                               int fp32_ref) {
    if (!index || !ws || !q6k_weights || !xnb || !out0 || B == 0u || B > ws->max_spec_b || B > 4u) {
        return -1;
    }
    if (B == 1u) {
        return lila_lhc_forward_pool(index, ws, dispatch_fn, pool_opaque,
                                     q6k_weights, xnb, out0, fp32_ref);
    }
    if (!out_rest) return -1;

    const uint32_t V = index->vocab_size;
    const uint32_t H = index->hidden_size;
    const int bpr = (int)(H / QK_K);

    double t0 = lhc_now_ms();
    for (uint32_t b = 0; b < B; ++b) {
        lhc_build_query_lut(index, xnb + (size_t)b * H, &ws->luts[b]);
    }
    memset(ws->heap_counts, 0, (size_t)B * ws->n_threads * sizeof(uint32_t));

    LhcStage1Ctx s1ctx;
    memset(&s1ctx, 0, sizeof(s1ctx));
    s1ctx.index = index;
    s1ctx.qluts = ws->luts;
    s1ctx.B = B;
    s1ctx.ws = ws;
    atomic_init(&s1ctx.next_tile, 0u);

    if (dispatch_fn && index->n_tiles >= 4u) {
        dispatch_fn(pool_opaque, lhc_stage1_worker, &s1ctx);
    } else {
        lhc_stage1_worker(0, 1, &s1ctx);
    }

    for (uint32_t b = 0; b < B; ++b) {
        ws->cand_counts[b] = lhc_merge_thread_heaps(
            ws->heaps + (size_t)b * ws->n_threads * ws->top_k,
            ws->heap_counts + (size_t)b * ws->n_threads,
            ws->n_threads, ws->top_k, ws->cands + (size_t)b * ws->top_k);
    }
    uint32_t u_count = lhc_merge_sorted_lanes(ws, B);
    double t1 = lhc_now_ms();
    ws->stats.stage1_ms_sum += (t1 - t0);

    float *out_ptrs[4] = {
        out0,
        (B > 1u) ? out_rest : NULL,
        (B > 2u) ? (out_rest + (size_t)V) : NULL,
        (B > 3u) ? (out_rest + (size_t)2u * V) : NULL,
    };
    for (uint32_t b = 0; b < B; ++b) {
        lhc_fill_neg_logits(out_ptrs[b], V);
    }

    int need_q8k = (!fp32_ref && (index->quant_type == 12u || (index->quant_type == 14u && !index->is_repacked)));
    if (need_q8k) {
        for (uint32_t b = 0; b < B; ++b) {
            if (qx_all) {
                for (int blk = 0; blk < bpr; ++blk) {
                    ws->qx_lanes[(size_t)b * bpr + blk] = qx_all[(size_t)blk * B + b];
                }
            } else {
                quantize_vec_q8k(&ws->qx_lanes[(size_t)b * bpr], xnb + (size_t)b * H, (int)H);
            }
        }
    }

    LhcStage2SpecCtx s2ctx;
    memset(&s2ctx, 0, sizeof(s2ctx));
    s2ctx.q6k_weights = q6k_weights;
    s2ctx.qx_all = need_q8k ? qx_all : NULL;
    s2ctx.qx_lanes = need_q8k ? ws->qx_lanes : NULL;
    s2ctx.xnb = xnb;
    s2ctx.merged_rows = ws->merged_rows;
    for (uint32_t b = 0; b < 4u; ++b) s2ctx.out_ptrs[b] = out_ptrs[b];
    s2ctx.u_count = u_count;
    s2ctx.B = B;
    s2ctx.hidden_size = (int)H;
    s2ctx.bpr = bpr;
    s2ctx.row_bytes = index->row_bytes;
    s2ctx.quant_type = index->quant_type;
    s2ctx.is_repacked = index->is_repacked;
    s2ctx.fp32_ref = fp32_ref;
    atomic_init(&s2ctx.next_idx, 0u);

    if (dispatch_fn && u_count >= 128u) {
        dispatch_fn(pool_opaque, lhc_stage2_spec_worker, &s2ctx);
    } else {
        lhc_stage2_spec_worker(0, 1, &s2ctx);
    }

    double t2 = lhc_now_ms();
    ws->stats.stage2_ms_sum += (t2 - t1);
    ws->stats.spec_eval_calls++;
    ws->stats.total_candidates_eval += u_count;
    return 0;
}

/* ─────────────────────────── Quality Metrics ───────────────────────────────── */

static int cmp_int32_asc_q(const void *a, const void *b) {
    int32_t ia = *(const int32_t *)a;
    int32_t ib = *(const int32_t *)b;
    return (ia > ib) - (ia < ib);
}

static double lhc_kl_div(const float *exact_logits,
                         const int32_t *selected_ids,
                         uint32_t top_k,
                         uint32_t vocab_size,
                         float temperature) {
    float inv_t = 1.0f / temperature;
    float max_e = exact_logits[0];
    for (uint32_t i = 1; i < vocab_size; ++i) {
        if (exact_logits[i] > max_e) max_e = exact_logits[i];
    }
    double sum_e = 0.0;
    for (uint32_t i = 0; i < vocab_size; ++i) {
        sum_e += exp((double)(exact_logits[i] - max_e) * (double)inv_t);
    }
    double sum_keep = 0.0;
    for (uint32_t k = 0; k < top_k; ++k) {
        int32_t id = selected_ids[k];
        if (id >= 0 && (uint32_t)id < vocab_size) {
            sum_keep += exp((double)(exact_logits[id] - max_e) * (double)inv_t);
        }
    }
    double mass_ratio = (sum_e > 0.0) ? (sum_keep / sum_e) : 1.0;
    if (mass_ratio <= 1e-15) return 34.538776;
    if (mass_ratio >= 1.0) return 0.0;
    return -log(mass_ratio);
}

int lila_lhc_evaluate_metrics(const LilaLhcIndex *index,
                              const float *exact_logits,
                              const float *coarse_scores,
                              uint32_t top_k,
                              LilaLhcEvalMetrics *out_metrics) {
    if (!index || !exact_logits || !coarse_scores || !out_metrics || top_k == 0) return -1;
    const uint32_t V = index->vocab_size;
    if (top_k > V) top_k = V;

    memset(out_metrics, 0, sizeof(*out_metrics));
    out_metrics->top_k = top_k;

    int32_t exact_top1 = 0;
    float exact_best = exact_logits[0];
    for (uint32_t i = 1; i < V; ++i) {
        if (exact_logits[i] > exact_best) {
            exact_best = exact_logits[i];
            exact_top1 = (int32_t)i;
        }
    }
    out_metrics->exact_top1_id = exact_top1;
    out_metrics->exact_top1_logit = exact_best;

    float target_coarse = coarse_scores[exact_top1];
    uint32_t rank = 0;
    for (uint32_t i = 0; i < V; ++i) {
        if ((int32_t)i == exact_top1) continue;
        if (coarse_scores[i] > target_coarse ||
            (coarse_scores[i] == target_coarse && (int32_t)i < exact_top1)) {
            rank++;
        }
    }
    out_metrics->true_top1_coarse_rank = rank;

    int32_t *cands = (int32_t *)malloc((size_t)top_k * sizeof(int32_t));
    if (!cands) return -1;
    if (lila_lhc_select_topk(coarse_scores, V, top_k, cands) != 0) {
        free(cands);
        return -1;
    }

    int32_t filt_top1 = cands[0];
    float filt_best = exact_logits[filt_top1];
    for (uint32_t k = 1; k < top_k; ++k) {
        int32_t id = cands[k];
        float val = exact_logits[id];
        if (val > filt_best || (val == filt_best && id < filt_top1)) {
            filt_best = val;
            filt_top1 = id;
        }
    }
    out_metrics->filtered_top1_id = filt_top1;
    out_metrics->filtered_top1_logit = filt_best;
    out_metrics->top1_match = (filt_top1 == exact_top1) ? 1 : 0;

    uint32_t n_top = (V < 64u) ? V : 64u;
    int32_t exact_top64[64];
    if (lila_lhc_select_topk(exact_logits, V, n_top, exact_top64) == 0) {
        for (uint32_t i = 0; i < n_top; ++i) {
            for (uint32_t j = i + 1u; j < n_top; ++j) {
                if (exact_logits[exact_top64[j]] > exact_logits[exact_top64[i]]) {
                    int32_t t = exact_top64[i];
                    exact_top64[i] = exact_top64[j];
                    exact_top64[j] = t;
                }
            }
        }
        uint32_t r10 = 0, r64 = 0;
        for (uint32_t i = 0; i < n_top; ++i) {
            int32_t tid = exact_top64[i];
            if (bsearch(&tid, cands, (size_t)top_k, sizeof(int32_t), cmp_int32_asc_q) != NULL) {
                r64++;
                if (i < 10u) r10++;
            }
        }
        out_metrics->top10_recall_count = r10;
        out_metrics->top64_recall_count = r64;
    }

    out_metrics->kl_divergence = lhc_kl_div(exact_logits, cands, top_k, V, 1.0f);
    out_metrics->kl_divergence_t07 = lhc_kl_div(exact_logits, cands, top_k, V, 0.7f);

    size_t row_q6k_bytes = index->row_bytes;
    out_metrics->coarse_bytes_read = index->total_sketch_bytes;
    out_metrics->fine_bytes_read = (size_t)top_k * row_q6k_bytes;
    out_metrics->total_bytes_read = out_metrics->coarse_bytes_read + out_metrics->fine_bytes_read;
    out_metrics->bandwidth_reduction_ratio =
        (out_metrics->total_bytes_read > 0u)
            ? ((double)index->exact_q6k_bytes / (double)out_metrics->total_bytes_read)
            : 1.0;

    free(cands);
    return 0;
}
