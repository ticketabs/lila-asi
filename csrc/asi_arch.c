#include "asi_arch.h"
#include "asi_pool.h"
#include "lila_asi.h"
#include "q4k.h"
#include "lila_lm_head_candidate.h"
#include "asi_sc_internal.h"

static void lhc_asi_dispatch(void *pool_opaque, LilaLhcTaskFn fn, void *ctx)
{
    AsiThreadPool *pool = (AsiThreadPool *)pool_opaque;
    if (pool) asi_pool_run(pool, fn, ctx);
    else fn(0, 1, ctx);
}
#include <errno.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ASI_SPEC_MAX_K 3u
#define ASI_SPEC_FAIL_CAP 512u

typedef struct {
    int q6k_dot2;
    int q4k_dot2;
    int matvec_chunk;
    int lmhead_chunk;
    int fused_chunk;
    int fused_chunk_overridden;
    int batch_chunk_max;
    int blis_min_tokens;
    int profile;
    int speculative;
    int spec_draft_len;
    int spec_stats;
} AsiRuntimeConfig;

static AsiRuntimeConfig g_asi_rt_cfg;
static AsiOnce g_asi_rt_cfg_once = ASI_ONCE_INIT;

static int asi_env_int(const char *name, int def_val, int min_val, int *overridden)
{
    if (overridden) *overridden = 0;
    const char *v = getenv(name);
    if (!v || v[0] == '\0') return def_val;
    char *endp = NULL;
    long parsed = strtol(v, &endp, 10);
    if (endp == v || *endp != '\0' || parsed < (long)min_val || parsed > 1048576L)
    {
        return def_val;
    }
    if (overridden) *overridden = 1;
    return (int)parsed;
}

static void asi_init_runtime_config(void)
{
    g_asi_rt_cfg.q6k_dot2 = asi_env_int("LILA_ASI_Q6K_DOT2", 1, 0, NULL) ? 1 : 0;
    g_asi_rt_cfg.q4k_dot2 = asi_env_int("LILA_ASI_Q4K_DOT2", 1, 0, NULL) ? 1 : 0;
    g_asi_rt_cfg.matvec_chunk = asi_env_int("LILA_ASI_MATVEC_CHUNK", 64, 1, NULL);
    g_asi_rt_cfg.lmhead_chunk = asi_env_int("LILA_ASI_LMHEAD_CHUNK", 512, 1, NULL);
    g_asi_rt_cfg.fused_chunk = asi_env_int("LILA_ASI_FUSED_CHUNK", 32, 1, &g_asi_rt_cfg.fused_chunk_overridden);
    g_asi_rt_cfg.batch_chunk_max = asi_env_int("LILA_ASI_BATCH_CHUNK_MAX", 512, 1, NULL);
    g_asi_rt_cfg.blis_min_tokens = asi_env_int("LILA_ASI_BLIS_MIN_TOKENS", 96, 1, NULL);
    const char *prof = getenv("LILA_ASI_PROFILE");
    g_asi_rt_cfg.profile = (prof && prof[0] != '\0' && strcmp(prof, "0") != 0) ? 1 : 0;
    g_asi_rt_cfg.speculative = asi_env_int("LILA_ASI_SPECULATIVE", 1, 0, NULL) ? 1 : 0;
    int sdl = asi_env_int("LILA_ASI_SPEC_DRAFT_LEN", 3, 1, NULL);
    if (sdl > (int)ASI_SPEC_MAX_K) sdl = (int)ASI_SPEC_MAX_K;
    g_asi_rt_cfg.spec_draft_len = sdl;
    const char *ss = getenv("LILA_ASI_SPEC_STATS");
    g_asi_rt_cfg.spec_stats = (ss && ss[0] != '\0' && strcmp(ss, "0") != 0) ? 1 : 0;
}

static inline const AsiRuntimeConfig *asi_runtime_config(void)
{
    asi_call_once(&g_asi_rt_cfg_once, asi_init_runtime_config);
    return &g_asi_rt_cfg;
}

static inline double asi_now_ms(void)
{
    return (double)asi_monotonic_ns() * 1e-6;
}

typedef enum {
    QWEN35_LAYER_DELTANET = 0,
    QWEN35_LAYER_FULL_ATTN = 1
} Qwen35LayerKind;

typedef struct {
    Qwen35LayerKind kind;
    const AsiV2TensorEntry *attn_norm;
    const AsiV2TensorEntry *post_attn_norm;
    const AsiV2TensorEntry *ffn_gate;
    const AsiV2TensorEntry *ffn_up;
    const AsiV2TensorEntry *ffn_down;

    /* Gated DeltaNet (SSM) tensors */
    const AsiV2TensorEntry *attn_qkv;
    const AsiV2TensorEntry *attn_gate;
    const AsiV2TensorEntry *ssm_a;
    const AsiV2TensorEntry *ssm_alpha;
    const AsiV2TensorEntry *ssm_beta;
    const AsiV2TensorEntry *ssm_conv1d;
    const AsiV2TensorEntry *ssm_dt_bias;
    const AsiV2TensorEntry *ssm_norm;
    const AsiV2TensorEntry *ssm_out;

    /* Full Gated Attention tensors */
    const AsiV2TensorEntry *attn_q;
    const AsiV2TensorEntry *attn_k;
    const AsiV2TensorEntry *attn_v;
    const AsiV2TensorEntry *attn_q_norm;
    const AsiV2TensorEntry *attn_k_norm;
    const AsiV2TensorEntry *attn_output;
} Qwen35LayerWeights;

typedef struct {
    float *conv_state; /* [conv_kernel - 1, qkv_dim] */
    float *ssm_state;  /* [ssm_group_count, ssm_head_dim, ssm_state_size] */
} Qwen35DeltaNetState;

typedef struct {
    uint16_t *k_cache; /* [max_ctx, n_kv_heads * head_dim] FP16 */
    uint16_t *v_cache; /* [max_ctx, n_kv_heads * head_dim] FP16 */
} Qwen35AttnKVState;

typedef struct {
    float *x;
    float *x_norm;
    float *qkv;
    float *z;
    float *qg;
    float *k;
    float *v;
    float *attn_out;
    float *proj_out;
    float *ffn_gate;
    float *ffn_up;
    float *attn_scores;
    float *alpha;
    float *beta;
    float *v_pred;
    /* Persistent batch scratch buffers (grown up to chunk_max) */
    uint32_t batch_cap;
    float *xb;
    float *xnb;
    float *projb;
    float *qkvb;
    float *zb;
    float *ab;
    float *bb;
    float *qgb;
    float *kb;
    float *vb;
    float *aob;
    float *fgb;
    float *fub;
    LilaQ8KBlock *qx_h;
    float *thread_scores_blk;
    size_t thread_scores_cap;
} Qwen35Scratch;

typedef struct {
    int32_t prev_tok;
    int32_t cur_tok;
    int32_t bad_next;
    uint32_t gen;
} AsiSpecFailEntry;

struct AsiArchSession {
    const AsiV2Cartridge *cart;
    uint32_t max_ctx;
    uint32_t n_threads;
    const AsiV2TensorEntry *token_embd;
    const AsiV2TensorEntry *output_norm;
    const AsiV2TensorEntry *output_weight; /* NULL if tied to token_embd */
    Qwen35LayerWeights *layers;
    Qwen35DeltaNetState *ssm_states;
    Qwen35AttnKVState *kv_states;
    Qwen35Scratch scratch;
    AsiThreadPool *pool;
    uint32_t prefix_len;
    float *prefix_conv;
    float *prefix_ssm;
    uint16_t *prefix_k;
    uint16_t *prefix_v;
    int32_t *prefix_tokens;
    int prefix_has_tokens;
    float *rope_inv_freq;
    int fp32_ref;
    /* Speculative decoding state (lazy-allocated on first K > 0 batch) */
    int speculative;
    int spec_draft_len;
    int spec_stats;
    int32_t *token_history;
    uint32_t history_len;
    uint32_t history_valid_from;
    uint32_t prompt_len;
    uint32_t spec_n_dn;
    uint32_t spec_cap_k;
    uint32_t spec_count;
    uint32_t spec_cursor;
    uint32_t spec_start_pos;
    int32_t spec_tokens[ASI_SPEC_MAX_K];
    float *spec_logits;
    float *spec_saved_qkv;
    float *spec_saved_a;
    float *spec_saved_b;
    float *spec_tmp_logits;
    AsiSpecFailEntry *fail_hash;
    uint32_t fail_gen;
    uint64_t stat_eval_calls;
    uint64_t stat_cache_hits;
    uint64_t stat_batches;
    uint64_t stat_drafted;
    uint64_t stat_accepted;
    uint64_t stat_misses;
    LilaLhcMode lmhead_mode;
    uint32_t lmhead_k;
    int lmhead_prefill;
    int lmhead_stats;
    LilaLhcIndex *lmhead_idx;
    LilaLhcWorkspace *lmhead_ws;
    int spec_gate;
    uint32_t spec_consec_misses;
    uint32_t spec_cooldown;
    uint64_t stat_spec_gate_skips;
    /* Exp E06: State-Centric CoW dirty tracking */
    uint32_t kv_dirty_from;
    /* ASI v2.5: Multimodal M-RoPE coordinate offset (rope_pos - kv_pos) */
    int64_t mrope_pos_delta;
    int64_t prefix_mrope_delta;
    /* ASI v2.6 (Torrent): post-activation sparse ffn_down keep fraction + Q4_REPACK flag */
    float ffn_sparse_keep;
    int has_q4_repack;
    /* Optional activation calibration hook (NULL unless explicitly attached) */
    void (*calib_hook)(void *, uint32_t, int, const float *, uint32_t, uint32_t);
    void *calib_ctx;
};

#if defined(__AVX2__)
static inline __m256 lila_expf_avx2(__m256 x)
{
    x = _mm256_max_ps(_mm256_min_ps(x, _mm256_set1_ps(88.0f)), _mm256_set1_ps(-88.0f));
    const __m256 log2e = _mm256_set1_ps(1.44269504088896341f);
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 ln2_hi = _mm256_set1_ps(0.693359375f);
    const __m256 ln2_lo = _mm256_set1_ps(-2.12194440e-4f);
    __m256 n = _mm256_floor_ps(_mm256_fmadd_ps(x, log2e, half));
    x = _mm256_fnmadd_ps(n, ln2_hi, x);
    x = _mm256_fnmadd_ps(n, ln2_lo, x);
    __m256 y = _mm256_set1_ps(1.9875691500e-4f);
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.3981999507e-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(8.3334519073e-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(4.1665795894e-2f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.6666665459e-1f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(5.0000001201e-1f));
    y = _mm256_fmadd_ps(y, _mm256_mul_ps(x, x), x);
    y = _mm256_add_ps(y, _mm256_set1_ps(1.0f));
    __m256i emm0 = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvttps_epi32(n), _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(y, _mm256_castsi256_ps(emm0));
}
#endif

static inline float fast_silu(float x)
{
    return x / (1.0f + expf(-x));
}

static inline void fast_silu_mul_vec(float *gate, const float *up, uint32_t n)
{
    uint32_t i = 0;
#if defined(__AVX2__)
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 neg = _mm256_set1_ps(-0.0f);
    for (; i + 7 < n; i += 8)
    {
        __m256 g = _mm256_loadu_ps(gate + i);
        __m256 u = _mm256_loadu_ps(up + i);
        __m256 e = lila_expf_avx2(_mm256_xor_ps(g, neg));
        __m256 sig = _mm256_div_ps(g, _mm256_add_ps(one, e));
        _mm256_storeu_ps(gate + i, _mm256_mul_ps(sig, u));
    }
#endif
    for (; i < n; ++i)
    {
        gate[i] = fast_silu(gate[i]) * up[i];
    }
}

static inline void fast_silu_inplace_vec(float *x, uint32_t n)
{
    uint32_t i = 0;
#if defined(__AVX2__)
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 neg = _mm256_set1_ps(-0.0f);
    for (; i + 7 < n; i += 8)
    {
        __m256 v = _mm256_loadu_ps(x + i);
        __m256 e = lila_expf_avx2(_mm256_xor_ps(v, neg));
        _mm256_storeu_ps(x + i, _mm256_div_ps(v, _mm256_add_ps(one, e)));
    }
#endif
    for (; i < n; ++i)
    {
        x[i] = fast_silu(x[i]);
    }
}

static inline void fast_silu_mul_out_vec(float *out, const float *gate, uint32_t n)
{
    uint32_t i = 0;
#if defined(__AVX2__)
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 neg = _mm256_set1_ps(-0.0f);
    for (; i + 7 < n; i += 8)
    {
        __m256 g = _mm256_loadu_ps(gate + i);
        __m256 o = _mm256_loadu_ps(out + i);
        __m256 e = lila_expf_avx2(_mm256_xor_ps(g, neg));
        __m256 sig = _mm256_div_ps(g, _mm256_add_ps(one, e));
        _mm256_storeu_ps(out + i, _mm256_mul_ps(o, sig));
    }
#endif
    for (; i < n; ++i)
    {
        out[i] *= fast_silu(gate[i]);
    }
}

static inline float fast_sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

static inline float asi_softmax_f32_inplace(float *scores, uint32_t len, float max_score)
{
    float sum_exp = 0.0f;
    uint32_t p = 0;
#if defined(__AVX2__)
    __m256 vmax = _mm256_set1_ps(max_score);
    __m256 vsum = _mm256_setzero_ps();
    for (; p + 7 < len; p += 8)
    {
        __m256 e = lila_expf_avx2(_mm256_sub_ps(_mm256_loadu_ps(scores + p), vmax));
        _mm256_storeu_ps(scores + p, e);
        vsum = _mm256_add_ps(vsum, e);
    }
    sum_exp = lila_hsum_f32_avx2(vsum);
#endif
    for (; p < len; ++p)
    {
        float e = expf(scores[p] - max_score);
        scores[p] = e;
        sum_exp += e;
    }
    return 1.0f / sum_exp;
}

static inline void asi_mul_sigmoid_vec(float *oh, const float *gh, uint32_t n)
{
    uint32_t d = 0;
#if defined(__AVX2__)
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 neg = _mm256_set1_ps(-0.0f);
    for (; d + 7 < n; d += 8)
    {
        __m256 gv = _mm256_loadu_ps(gh + d);
        __m256 ov = _mm256_loadu_ps(oh + d);
        __m256 e = lila_expf_avx2(_mm256_xor_ps(gv, neg));
        __m256 sig = _mm256_div_ps(one, _mm256_add_ps(one, e));
        _mm256_storeu_ps(oh + d, _mm256_mul_ps(ov, sig));
    }
#endif
    for (; d < n; ++d)
    {
        oh[d] *= fast_sigmoid(gh[d]);
    }
}

static inline float fast_softplus(float x)
{
    return (x > 20.0f) ? x : log1pf(expf(x));
}

static void rmsnorm(float *out, const float *x, const float *w, uint32_t n, float eps)
{
    float ss = 0.0f;
#if defined(__AVX512F__)
    if ((n & 15u) == 0u)
    {
        __m512 vss = _mm512_setzero_ps();
        for (uint32_t i = 0; i < n; i += 16)
        {
            __m512 vx = _mm512_loadu_ps(x + i);
            vss = _mm512_fmadd_ps(vx, vx, vss);
        }
        ss = _mm512_reduce_add_ps(vss);
        float inv = 1.0f / sqrtf(ss / (float)n + eps);
        __m512 vinv = _mm512_set1_ps(inv);
        for (uint32_t i = 0; i < n; i += 16)
        {
            __m512 vx = _mm512_loadu_ps(x + i);
            __m512 vw = _mm512_loadu_ps(w + i);
            _mm512_storeu_ps(out + i, _mm512_mul_ps(_mm512_mul_ps(vx, vinv), vw));
        }
        return;
    }
#endif
#if defined(__AVX2__)
    if ((n & 7u) == 0u)
    {
        __m256 vss = _mm256_setzero_ps();
        for (uint32_t i = 0; i < n; i += 8)
        {
            __m256 vx = _mm256_loadu_ps(x + i);
            vss = _mm256_fmadd_ps(vx, vx, vss);
        }
        ss = lila_hsum_f32_avx2(vss);
        float inv = 1.0f / sqrtf(ss / (float)n + eps);
        __m256 vinv = _mm256_set1_ps(inv);
        for (uint32_t i = 0; i < n; i += 8)
        {
            __m256 vx = _mm256_loadu_ps(x + i);
            __m256 vw = _mm256_loadu_ps(w + i);
            _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_mul_ps(vx, vinv), vw));
        }
        return;
    }
#endif
#if defined(__aarch64__)
    if ((n & 3u) == 0u)
    {
        float32x4_t vss = vdupq_n_f32(0.0f);
        for (uint32_t i = 0; i < n; i += 4)
        {
            float32x4_t vx = vld1q_f32(x + i);
            vss = vfmaq_f32(vss, vx, vx);
        }
        ss = vaddvq_f32(vss);
        float inv = 1.0f / sqrtf(ss / (float)n + eps);
        float32x4_t vinv = vdupq_n_f32(inv);
        for (uint32_t i = 0; i < n; i += 4)
        {
            float32x4_t vx = vld1q_f32(x + i);
            float32x4_t vw = vld1q_f32(w + i);
            vst1q_f32(out + i, vmulq_f32(vmulq_f32(vx, vinv), vw));
        }
        return;
    }
#endif
    for (uint32_t i = 0; i < n; ++i)
    {
        ss += x[i] * x[i];
    }
    float inv = 1.0f / sqrtf(ss / (float)n + eps);
    for (uint32_t i = 0; i < n; ++i)
    {
        out[i] = (x[i] * inv) * w[i];
    }
}

static inline const uint8_t *tensor_data(const AsiV2Cartridge *cart, const AsiV2TensorEntry *t)
{
    return cart->weights_data + t->data_offset;
}

typedef struct {
    const uint8_t *w;
    const LilaQ8KBlock *qx;
    float *out;
    int rows;
    int blocks_per_row;
    uint32_t quant_type;
    atomic_int next_row;
    int chunk_rows;
} AsiMatvecCtx;

static inline void asi_prefetch_row(const uint8_t *rp, size_t rb)
{
    LILA_PREFETCH(rp);
    if (rb > 256) LILA_PREFETCH(rp + 256);
}

static void asi_matvec_range(const uint8_t *w, const LilaQ8KBlock *qx, float *out,
                             int start, int end, int bpr, uint32_t quant_type, int n_threads)
{
    (void)n_threads;
    if (quant_type == 12)
    {
        size_t rb = (size_t)bpr * Q4_K_BLOCK_SIZE;
        int i = start;
#if defined(__AVX2__)
        const AsiRuntimeConfig *cfg = asi_runtime_config();
        int use_dot2 = (cfg->q4k_dot2 < 0) ? (n_threads <= 1) : cfg->q4k_dot2;
        if (use_dot2)
        {
            for (; i + 1 < end; i += 2)
            {
                const uint8_t *rp = w + (size_t)i * rb;
                if (i + 3 < end)
                {
                    asi_prefetch_row(rp + 2 * rb, rb);
                    asi_prefetch_row(rp + 3 * rb, rb);
                }
                q4k_dot2_q8k_avx2(rp, rp + rb, qx, bpr, &out[i], &out[i + 1]);
            }
        }
#endif
        for (; i < end; ++i)
        {
            const uint8_t *rp = w + (size_t)i * rb;
            if (i + 2 < end) asi_prefetch_row(rp + 2 * rb, rb);
            out[i] = q4k_dot_q8k_simd(rp, qx, bpr);
        }
    }
    else if (quant_type == 14)
    {
        size_t rb = (size_t)bpr * Q6_K_BLOCK_SIZE;
        int i = start;
#if defined(__AVX2__)
        if (asi_runtime_config()->q6k_dot2)
        {
            for (; i + 1 < end; i += 2)
            {
                const uint8_t *rp = w + (size_t)i * rb;
                if (i + 3 < end)
                {
                    asi_prefetch_row(rp + 2 * rb, rb);
                    asi_prefetch_row(rp + 3 * rb, rb);
                }
                q6k_dot2_q8k_avx2(rp, rp + rb, qx, bpr, &out[i], &out[i + 1]);
            }
        }
#endif
        for (; i < end; ++i)
        {
            const uint8_t *rp = w + (size_t)i * rb;
            if (i + 2 < end) asi_prefetch_row(rp + 2 * rb, rb);
            out[i] = q6k_dot_q8k_simd(rp, qx, bpr);
        }
    }
}

static void asi_matvec_worker(int tid, int n_threads, void *vctx)
{
    (void)tid;
    AsiMatvecCtx *ctx = (AsiMatvecCtx *)vctx;
    int bpr = ctx->blocks_per_row;
    int rows = ctx->rows;
    int chunk = ctx->chunk_rows;

    if (n_threads <= 1)
    {
        asi_matvec_range(ctx->w, ctx->qx, ctx->out, 0, rows, bpr, ctx->quant_type, n_threads);
        return;
    }

    for (;;)
    {
        int start = atomic_fetch_add_explicit(&ctx->next_row, chunk, memory_order_relaxed);
        if (start >= rows) break;
        int end = start + chunk;
        if (end > rows) end = rows;
        asi_matvec_range(ctx->w, ctx->qx, ctx->out, start, end, bpr, ctx->quant_type, n_threads);
    }
}

static int matvec_q8k_pool(AsiThreadPool *pool,
                           const AsiV2Cartridge *cart,
                           const AsiV2TensorEntry *t,
                           float *out,
                           const LilaQ8KBlock *qx,
                           int bpr)
{
    const uint8_t *w = tensor_data(cart, t);
    int rows = (int)t->shape[1];
    const AsiRuntimeConfig *cfg = asi_runtime_config();
    int chunk = (rows >= 16384) ? cfg->lmhead_chunk : cfg->matvec_chunk;
    AsiMatvecCtx ctx;
    ctx.w = w;
    ctx.qx = qx;
    ctx.out = out;
    ctx.rows = rows;
    ctx.blocks_per_row = bpr;
    ctx.quant_type = t->quant_type;
    atomic_init(&ctx.next_row, 0);
    ctx.chunk_rows = chunk;
    if (pool && rows >= 64)
        asi_pool_run(pool, asi_matvec_worker, &ctx);
    else
        asi_matvec_worker(0, 1, &ctx);
    return 0;
}

typedef struct {
    const uint8_t *w[4];
    float *out[4];
    int rows[4];
    uint32_t quant_type[4];
    int n_mats;
    const LilaQ8KBlock *qx;
    int blocks_per_row;
    int apply_silu_mul;
    _Alignas(64) atomic_int next_row[4];
} AsiFusedMatvecCtx;

static void asi_fused_matvec_worker(int tid, int n_threads, void *vctx)
{
    (void)tid;
    AsiFusedMatvecCtx *ctx = (AsiFusedMatvecCtx *)vctx;
    if (n_threads <= 1)
    {
        for (int m = 0; m < ctx->n_mats; ++m)
        {
            asi_matvec_range(ctx->w[m], ctx->qx, ctx->out[m], 0, ctx->rows[m],
                             ctx->blocks_per_row, ctx->quant_type[m], n_threads);
        }
        if (ctx->apply_silu_mul && ctx->n_mats == 2 && ctx->rows[0] == ctx->rows[1])
        {
            fast_silu_mul_vec(ctx->out[0], ctx->out[1], (uint32_t)ctx->rows[0]);
        }
        return;
    }

    const AsiRuntimeConfig *cfg = asi_runtime_config();
    if (ctx->apply_silu_mul && ctx->n_mats == 2 && ctx->rows[0] == ctx->rows[1])
    {
        int total = ctx->rows[0];
        int chunk = cfg->fused_chunk_overridden ? cfg->fused_chunk : ((total >= 256) ? cfg->fused_chunk : 16);
        for (;;)
        {
            int start = atomic_fetch_add_explicit(&ctx->next_row[0], chunk, memory_order_relaxed);
            if (start >= total) break;
            int end = start + chunk;
            if (end > total) end = total;
            asi_matvec_range(ctx->w[0], ctx->qx, ctx->out[0], start, end, ctx->blocks_per_row, ctx->quant_type[0], n_threads);
            asi_matvec_range(ctx->w[1], ctx->qx, ctx->out[1], start, end, ctx->blocks_per_row, ctx->quant_type[1], n_threads);
            fast_silu_mul_vec(ctx->out[0] + start, ctx->out[1] + start, (uint32_t)(end - start));
        }
        return;
    }

    for (int m = 0; m < ctx->n_mats; ++m)
    {
        int total = ctx->rows[m];
        int chunk = cfg->fused_chunk_overridden ? cfg->fused_chunk : ((total >= 256) ? cfg->fused_chunk : 16);
        for (;;)
        {
            int start = atomic_fetch_add_explicit(&ctx->next_row[m], chunk, memory_order_relaxed);
            if (start >= total) break;
            int end = start + chunk;
            if (end > total) end = total;
            asi_matvec_range(ctx->w[m], ctx->qx, ctx->out[m], start, end, ctx->blocks_per_row, ctx->quant_type[m], n_threads);
        }
    }
}

typedef struct {
    const uint8_t *w[4];
    float *out[4];
    int rows[4];
    uint32_t quant_type[4];
    int is_repacked[4];
    int is_col_major[4];
    int n_mats;
    const float *in;
    int n_tokens;
    int cols;
    int apply_silu_mul;
    _Alignas(64) atomic_int next_row[4];
} AsiFp32BatchCtx;

static void asi_fp32_eval_mat_range(const AsiFp32BatchCtx *ctx, int m, int start, int end)
{
    if (ctx->n_tokens <= 1)
    {
        for (int i = start; i < end; ++i)
        {
            ctx->out[m][i] = lila_quant_dot_fp32_at(
                ctx->w[m], i, ctx->in, ctx->cols, ctx->rows[m],
                ctx->quant_type[m], ctx->is_repacked[m], ctx->is_col_major[m]);
        }
    }
    else
    {
        lila_quant_matmul_batch_fp32_range_ex(
            ctx->w[m], ctx->in, ctx->cols, ctx->out[m],
            ctx->n_tokens, ctx->rows[m], start, end,
            ctx->cols, ctx->quant_type[m], ctx->is_repacked[m], ctx->is_col_major[m]);
    }
}

static void asi_fp32_batch_worker(int tid, int n_threads, void *vctx)
{
    (void)tid;
    AsiFp32BatchCtx *ctx = (AsiFp32BatchCtx *)vctx;
    if (n_threads <= 1)
    {
        for (int m = 0; m < ctx->n_mats; ++m)
            asi_fp32_eval_mat_range(ctx, m, 0, ctx->rows[m]);
        if (ctx->apply_silu_mul && ctx->n_mats == 2 && ctx->rows[0] == ctx->rows[1])
            fast_silu_mul_vec(ctx->out[0], ctx->out[1], (uint32_t)((size_t)ctx->n_tokens * (size_t)ctx->rows[0]));
        return;
    }
    if (ctx->apply_silu_mul && ctx->n_mats == 2 && ctx->rows[0] == ctx->rows[1])
    {
        int total = ctx->rows[0];
        int chunk = (total >= 256) ? 32 : 16;
        for (;;)
        {
            int start = atomic_fetch_add_explicit(&ctx->next_row[0], chunk, memory_order_relaxed);
            if (start >= total) break;
            int end = start + chunk;
            if (end > total) end = total;
            asi_fp32_eval_mat_range(ctx, 0, start, end);
            asi_fp32_eval_mat_range(ctx, 1, start, end);
            for (int t = 0; t < ctx->n_tokens; ++t)
            {
                size_t off = (size_t)t * (size_t)total + (size_t)start;
                fast_silu_mul_vec(ctx->out[0] + off, ctx->out[1] + off, (uint32_t)(end - start));
            }
        }
        return;
    }
    for (int m = 0; m < ctx->n_mats; ++m)
    {
        int total = ctx->rows[m];
        int chunk = (total >= 16384) ? 256 : ((total >= 256) ? 32 : 16);
        for (;;)
        {
            int start = atomic_fetch_add_explicit(&ctx->next_row[m], chunk, memory_order_relaxed);
            if (start >= total) break;
            int end = start + chunk;
            if (end > total) end = total;
            asi_fp32_eval_mat_range(ctx, m, start, end);
        }
    }
}

static int matmul_fp32_pool(AsiThreadPool *pool,
                            const AsiV2Cartridge *cart,
                            const AsiV2TensorEntry **ts,
                            float **outs,
                            int n_mats,
                            const float *in,
                            int n_tokens,
                            int cols,
                            int apply_silu_mul)
{
    AsiFp32BatchCtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.n_mats = n_mats;
    ctx.in = in;
    ctx.n_tokens = (n_tokens > 0) ? n_tokens : 1;
    ctx.cols = cols;
    ctx.apply_silu_mul = apply_silu_mul;
    int max_rows = 0;
    for (int m = 0; m < n_mats; ++m)
    {
        if (!lila_quant_is_supported(ts[m]->quant_type)) return -1;
        ctx.w[m] = tensor_data(cart, ts[m]);
        ctx.out[m] = outs[m];
        ctx.rows[m] = (int)ts[m]->shape[1];
        ctx.quant_type[m] = ts[m]->quant_type;
        ctx.is_repacked[m] = (ts[m]->layout_flags & ASI_TENSOR_LAYOUT_AVX2_REPACK) ? 1 : 0;
        ctx.is_col_major[m] = (ts[m]->layout_flags & ASI_TENSOR_LAYOUT_COL_MAJOR) ? 1 : 0;
        atomic_init(&ctx.next_row[m], 0);
        if (ctx.rows[m] > max_rows) max_rows = ctx.rows[m];
    }
    if (pool && (max_rows >= 32 || (max_rows >= 16 && ctx.n_tokens > 1)))
        asi_pool_run(pool, asi_fp32_batch_worker, &ctx);
    else
        asi_fp32_batch_worker(0, 1, &ctx);
    return 0;
}

typedef struct {
    const uint8_t *w[4];
    float *out[4];
    int rows[4];
    int n_mats;
    const LilaQ8_64Block *qx64;
    const LilaQ3LutBlock *q3_luts;
    int n_blk64;
    int n_blk128;
    uint32_t quant_type[4];
    int apply_silu_mul;
    _Alignas(64) atomic_int next_row[4];
} AsiE07FusedMatvecCtx;

static void asi_e07_fused_matvec_worker(int tid, int n_threads, void *vctx)
{
    (void)tid;
    AsiE07FusedMatvecCtx *ctx = (AsiE07FusedMatvecCtx *)vctx;
    if (n_threads <= 1)
    {
        for (int m = 0; m < ctx->n_mats; ++m)
        {
            if (ctx->quant_type[m] == ASI_QUANT_Q4_REPACK)
                q4_repack_matvec_range(ctx->out[m], ctx->w[m], ctx->qx64, 0, ctx->rows[m], ctx->n_blk64);
            else if (ctx->quant_type[m] == ASI_QUANT_Q3_SPLITLUT)
                q3_splitlut_matvec_range(ctx->out[m], ctx->w[m], ctx->q3_luts, 0, ctx->rows[m], ctx->rows[m], ctx->n_blk128);
        }
        if (ctx->apply_silu_mul && ctx->n_mats == 2 && ctx->rows[0] == ctx->rows[1])
            fast_silu_mul_vec(ctx->out[0], ctx->out[1], (uint32_t)ctx->rows[0]);
        return;
    }

    if (ctx->apply_silu_mul && ctx->n_mats == 2 && ctx->rows[0] == ctx->rows[1] &&
        ctx->quant_type[0] == ctx->quant_type[1])
    {
        int total = ctx->rows[0];
        int chunk = (ctx->quant_type[0] == ASI_QUANT_Q3_SPLITLUT) ? Q3_SPLITLUT_TILE_ROWS : 64;
        for (;;)
        {
            int start = atomic_fetch_add_explicit(&ctx->next_row[0], chunk, memory_order_relaxed);
            if (start >= total) break;
            int end = start + chunk;
            if (end > total) end = total;
            for (int m = 0; m < 2; ++m)
            {
                if (ctx->quant_type[m] == ASI_QUANT_Q4_REPACK)
                    q4_repack_matvec_range(ctx->out[m], ctx->w[m], ctx->qx64, start, end, ctx->n_blk64);
                else
                    q3_splitlut_matvec_range(ctx->out[m], ctx->w[m], ctx->q3_luts, start, end, total, ctx->n_blk128);
            }
            fast_silu_mul_vec(ctx->out[0] + start, ctx->out[1] + start, (uint32_t)(end - start));
        }
        return;
    }

    for (int m = 0; m < ctx->n_mats; ++m)
    {
        int total = ctx->rows[m];
        int chunk = (ctx->quant_type[m] == ASI_QUANT_Q3_SPLITLUT) ? Q3_SPLITLUT_TILE_ROWS : 64;
        for (;;)
        {
            int start = atomic_fetch_add_explicit(&ctx->next_row[m], chunk, memory_order_relaxed);
            if (start >= total) break;
            int end = start + chunk;
            if (end > total) end = total;
            if (ctx->quant_type[m] == ASI_QUANT_Q4_REPACK)
                q4_repack_matvec_range(ctx->out[m], ctx->w[m], ctx->qx64, start, end, ctx->n_blk64);
            else if (ctx->quant_type[m] == ASI_QUANT_Q3_SPLITLUT)
                q3_splitlut_matvec_range(ctx->out[m], ctx->w[m], ctx->q3_luts, start, end, total, ctx->n_blk128);
        }
    }
}

typedef struct {
    const uint8_t *w;
    const LilaQ8_64Block *qx64;
    const LilaQ3LutBlock *q3_luts;
    float *out;
    int rows;
    int n_blk64;
    int n_blk128;
    uint32_t quant_type;
    _Alignas(64) atomic_int next_row;
    int chunk_rows;
} AsiE07MatvecCtx;

static void asi_e07_matvec_worker(int tid, int n_threads, void *vctx)
{
    (void)tid;
    AsiE07MatvecCtx *ctx = (AsiE07MatvecCtx *)vctx;
    int rows = ctx->rows;
    if (n_threads <= 1)
    {
        if (ctx->quant_type == ASI_QUANT_Q4_REPACK)
            q4_repack_matvec_range(ctx->out, ctx->w, ctx->qx64, 0, rows, ctx->n_blk64);
        else
            q3_splitlut_matvec_range(ctx->out, ctx->w, ctx->q3_luts, 0, rows, rows, ctx->n_blk128);
        return;
    }
    int chunk = ctx->chunk_rows;
    for (;;)
    {
        int start = atomic_fetch_add_explicit(&ctx->next_row, chunk, memory_order_relaxed);
        if (start >= rows) break;
        int end = start + chunk;
        if (end > rows) end = rows;
        if (ctx->quant_type == ASI_QUANT_Q4_REPACK)
            q4_repack_matvec_range(ctx->out, ctx->w, ctx->qx64, start, end, ctx->n_blk64);
        else
            q3_splitlut_matvec_range(ctx->out, ctx->w, ctx->q3_luts, start, end, rows, ctx->n_blk128);
    }
}

static inline int asi_tensor_needs_q8k(const AsiV2TensorEntry *t)
{
    if (!t) return 0;
    return (t->quant_type == 12u ||
            (t->quant_type == 14u && !(t->layout_flags & ASI_TENSOR_LAYOUT_AVX2_REPACK))) ? 1 : 0;
}

static int matvec_tensor_pool(AsiThreadPool *pool,
                              const AsiV2Cartridge *cart,
                              const AsiV2TensorEntry *t,
                              float *out,
                              const float *in,
                              int fp32_ref);

static void matvec_fused_q8k_pool(AsiThreadPool *pool,
                                  const AsiV2Cartridge *cart,
                                  const AsiV2TensorEntry **tensors,
                                  float **outs,
                                  int n_mats,
                                  const LilaQ8KBlock *qx,
                                  int bpr,
                                  const float *in,
                                  int cols,
                                  int apply_silu_mul,
                                  int fp32_ref)
{
    if (!fp32_ref)
    {
        int all_q8k = 1;
        int any_q4r = 0, any_q3 = 0, all_e07_rm = 1;
        for (int m = 0; m < n_mats; ++m)
        {
            uint32_t qt = tensors[m]->quant_type;
            uint32_t lf = tensors[m]->layout_flags;
            if ((qt != 12u && qt != 14u) || (lf & ASI_TENSOR_LAYOUT_AVX2_REPACK))
                all_q8k = 0;
            if (qt == ASI_QUANT_Q4_REPACK && !(lf & ASI_TENSOR_LAYOUT_COL_MAJOR))
                any_q4r = 1;
            else if (qt == ASI_QUANT_Q3_SPLITLUT)
                any_q3 = 1;
            else
                all_e07_rm = 0;
        }
        if (!all_q8k)
        {
            if (all_e07_rm && (any_q4r || any_q3))
            {
                int n_blk64 = (cols + Q4_REPACK_GROUP_SIZE - 1) / Q4_REPACK_GROUP_SIZE;
                int n_blk128 = (cols + Q3_SPLITLUT_BLK_COLS - 1) / Q3_SPLITLUT_BLK_COLS;
                LilaQ8_64Block stack_qx64[64];
                LilaQ8_64Block *qx64 = NULL;
                LilaQ3LutBlock stack_q3[32];
                LilaQ3LutBlock *q3_luts = NULL;
                if (any_q4r)
                {
                    qx64 = (n_blk64 <= 64) ? stack_qx64 : lila_q8_64_alloc((size_t)n_blk64);
                    if (!qx64) { matmul_fp32_pool(pool, cart, tensors, outs, n_mats, in, 1, cols, apply_silu_mul); return; }
                    quantize_vec_q8_64(qx64, in, cols);
                }
                if (any_q3)
                {
                    q3_luts = (n_blk128 <= 32) ? stack_q3 : (LilaQ3LutBlock *)asi_aligned_alloc(32, (size_t)n_blk128 * sizeof(LilaQ3LutBlock));
                    if (!q3_luts) { if (qx64 && qx64 != stack_qx64) lila_q8_64_free(qx64); matmul_fp32_pool(pool, cart, tensors, outs, n_mats, in, 1, cols, apply_silu_mul); return; }
                    q3_splitlut_build_lut(q3_luts, in, cols);
                }
                AsiE07FusedMatvecCtx ctx;
                memset(&ctx, 0, sizeof(ctx));
                ctx.n_mats = n_mats;
                ctx.qx64 = qx64;
                ctx.q3_luts = q3_luts;
                ctx.n_blk64 = n_blk64;
                ctx.n_blk128 = n_blk128;
                ctx.apply_silu_mul = apply_silu_mul;
                int max_rows = 0;
                for (int m = 0; m < n_mats; ++m)
                {
                    ctx.w[m] = tensor_data(cart, tensors[m]);
                    ctx.out[m] = outs[m];
                    ctx.rows[m] = (int)tensors[m]->shape[1];
                    ctx.quant_type[m] = tensors[m]->quant_type;
                    atomic_init(&ctx.next_row[m], 0);
                    if (ctx.rows[m] > max_rows) max_rows = ctx.rows[m];
                }
                if (pool && max_rows >= 64)
                    asi_pool_run(pool, asi_e07_fused_matvec_worker, &ctx);
                else
                    asi_e07_fused_matvec_worker(0, 1, &ctx);
                if (qx64 && qx64 != stack_qx64) lila_q8_64_free(qx64);
                if (q3_luts && q3_luts != stack_q3) asi_aligned_free(q3_luts);
                return;
            }
            if (!apply_silu_mul)
            {
                for (int m = 0; m < n_mats; ++m)
                    matvec_tensor_pool(pool, cart, tensors[m], outs[m], in, 0);
                return;
            }
            fp32_ref = 1;
        }
    }
    if (fp32_ref)
    {
        matmul_fp32_pool(pool, cart, tensors, outs, n_mats, in, 1, cols, apply_silu_mul);
        return;
    }
    AsiFusedMatvecCtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.n_mats = n_mats;
    ctx.qx = qx;
    ctx.blocks_per_row = bpr;
    ctx.apply_silu_mul = apply_silu_mul;
    int max_rows = 0;
    for (int m = 0; m < n_mats; ++m)
    {
        ctx.w[m] = tensor_data(cart, tensors[m]);
        ctx.out[m] = outs[m];
        ctx.rows[m] = (int)tensors[m]->shape[1];
        ctx.quant_type[m] = tensors[m]->quant_type;
        atomic_init(&ctx.next_row[m], 0);
        if (ctx.rows[m] > max_rows) max_rows = ctx.rows[m];
    }
    if (pool && max_rows >= 64)
        asi_pool_run(pool, asi_fused_matvec_worker, &ctx);
    else
        asi_fused_matvec_worker(0, 1, &ctx);
}

typedef struct {
    const uint8_t *w;
    const float *in;
    float *out;
    int total_rows;
    int cols;
    const uint32_t *active_cols;
    int n_active;
    _Alignas(64) atomic_int next_row;
} AsiQ4ColMajorCtx;

static void asi_q4_col_major_worker(int tid, int n_threads, void *vctx)
{
    (void)tid;
    AsiQ4ColMajorCtx *ctx = (AsiQ4ColMajorCtx *)vctx;
    if (n_threads <= 1)
    {
        q4_repack_col_major_matvec_range(ctx->out, ctx->w, ctx->in, 0, ctx->total_rows,
                                         ctx->total_rows, ctx->cols,
                                         ctx->active_cols, ctx->n_active);
        return;
    }
    int chunk = Q4_REPACK_GROUP_SIZE * 2;
    for (;;)
    {
        int start = atomic_fetch_add_explicit(&ctx->next_row, chunk, memory_order_relaxed);
        if (start >= ctx->total_rows) break;
        int end = start + chunk;
        if (end > ctx->total_rows) end = ctx->total_rows;
        q4_repack_col_major_matvec_range(ctx->out, ctx->w, ctx->in, start, end,
                                         ctx->total_rows, ctx->cols,
                                         ctx->active_cols, ctx->n_active);
    }
}

static int matvec_q4_col_major_pool(AsiThreadPool *pool,
                                    const AsiV2Cartridge *cart,
                                    const AsiV2TensorEntry *t,
                                    float *out,
                                    const float *in,
                                    float keep_frac)
{
    const uint8_t *w = tensor_data(cart, t);
    int cols = (int)t->shape[0];
    int rows = (int)t->shape[1];
    uint32_t stack_active[4096];
    uint32_t *active_cols = NULL;
    int n_active = cols;
    if (keep_frac > 0.0f && keep_frac < 0.999f && cols > 64)
    {
        uint32_t *buf = (cols <= 4096) ? stack_active : (uint32_t *)malloc((size_t)cols * sizeof(uint32_t));
        if (buf)
        {
            n_active = asi_select_topk_abs_cols(in, cols, keep_frac, buf, cols);
            active_cols = buf;
        }
    }
    AsiQ4ColMajorCtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.w = w;
    ctx.in = in;
    ctx.out = out;
    ctx.total_rows = rows;
    ctx.cols = cols;
    ctx.active_cols = active_cols;
    ctx.n_active = n_active;
    atomic_init(&ctx.next_row, 0);
    if (pool && rows >= Q4_REPACK_GROUP_SIZE * 2)
        asi_pool_run(pool, asi_q4_col_major_worker, &ctx);
    else
        asi_q4_col_major_worker(0, 1, &ctx);
    if (active_cols && active_cols != stack_active) free(active_cols);
    return 0;
}

static int matvec_tensor_pool(AsiThreadPool *pool,
                              const AsiV2Cartridge *cart,
                              const AsiV2TensorEntry *t,
                              float *out,
                              const float *in,
                              int fp32_ref)
{
    const uint8_t *w = tensor_data(cart, t);
    int cols = (int)t->shape[0];
    int rows = (int)t->shape[1];
    if (fp32_ref)
    {
        return matmul_fp32_pool(pool, cart, &t, &out, 1, in, 1, cols, 0);
    }
    if (t->quant_type == ASI_QUANT_Q4_REPACK && (t->layout_flags & ASI_TENSOR_LAYOUT_COL_MAJOR))
    {
        return matvec_q4_col_major_pool(pool, cart, t, out, in, 1.0f);
    }
    if (t->quant_type == ASI_QUANT_Q4_REPACK && !(t->layout_flags & ASI_TENSOR_LAYOUT_COL_MAJOR))
    {
        int n_blk64 = (cols + Q4_REPACK_GROUP_SIZE - 1) / Q4_REPACK_GROUP_SIZE;
        LilaQ8_64Block stack_qx[64];
        LilaQ8_64Block *qx = (n_blk64 <= 64) ? stack_qx : lila_q8_64_alloc((size_t)n_blk64);
        if (!qx) return -1;
        quantize_vec_q8_64(qx, in, cols);
        AsiE07MatvecCtx ctx;
        memset(&ctx, 0, sizeof(ctx));
        ctx.w = w;
        ctx.qx64 = qx;
        ctx.out = out;
        ctx.rows = rows;
        ctx.n_blk64 = n_blk64;
        ctx.quant_type = ASI_QUANT_Q4_REPACK;
        atomic_init(&ctx.next_row, 0);
        ctx.chunk_rows = (rows >= 16384) ? asi_runtime_config()->lmhead_chunk : asi_runtime_config()->matvec_chunk;
        if (pool && rows >= 64)
            asi_pool_run(pool, asi_e07_matvec_worker, &ctx);
        else
            asi_e07_matvec_worker(0, 1, &ctx);
        if (qx != stack_qx) lila_q8_64_free(qx);
        return 0;
    }
    if (t->quant_type == ASI_QUANT_Q3_SPLITLUT)
    {
        int n_blk128 = (cols + Q3_SPLITLUT_BLK_COLS - 1) / Q3_SPLITLUT_BLK_COLS;
        LilaQ3LutBlock stack_lut[32];
        LilaQ3LutBlock *luts = (n_blk128 <= 32) ? stack_lut : (LilaQ3LutBlock *)asi_aligned_alloc(32, (size_t)n_blk128 * sizeof(LilaQ3LutBlock));
        if (!luts) return -1;
        q3_splitlut_build_lut(luts, in, cols);
        AsiE07MatvecCtx ctx;
        memset(&ctx, 0, sizeof(ctx));
        ctx.w = w;
        ctx.q3_luts = luts;
        ctx.out = out;
        ctx.rows = rows;
        ctx.n_blk128 = n_blk128;
        ctx.quant_type = ASI_QUANT_Q3_SPLITLUT;
        atomic_init(&ctx.next_row, 0);
        ctx.chunk_rows = (rows >= 16384) ? 512 : 64;
        if (pool && rows >= 64)
            asi_pool_run(pool, asi_e07_matvec_worker, &ctx);
        else
            asi_e07_matvec_worker(0, 1, &ctx);
        if (luts != stack_lut) asi_aligned_free(luts);
        return 0;
    }
    if ((t->quant_type == 12 || t->quant_type == 14) && !(t->layout_flags & ASI_TENSOR_LAYOUT_AVX2_REPACK))
    {
        int bpr = (cols + QK_K - 1) / QK_K;
        if (bpr <= 0) return 0;
        LilaQ8KBlock stack_qx[64];
        LilaQ8KBlock *qx = (bpr <= 64) ? stack_qx : lila_q8k_alloc((size_t)bpr);
        if (qx)
        {
            quantize_vec_q8k(qx, in, cols);
            matvec_q8k_pool(pool, cart, t, out, qx, bpr);
            if (qx != stack_qx) free(qx);
            return 0;
        }
    }
    if (t->quant_type == 12) /* Q4_K */
    {
        q4k_matvec(out, w, in, rows, cols);
        return 0;
    }
    if (t->quant_type == 14) /* Q6_K */
    {
        if (t->layout_flags & ASI_TENSOR_LAYOUT_AVX2_REPACK)
            q6k_repacked_matvec(out, w, in, rows, cols);
        else
            q6k_matvec(out, w, in, rows, cols);
        return 0;
    }
    if (lila_quant_is_supported(t->quant_type))
    {
        return matmul_fp32_pool(pool, cart, &t, &out, 1, in, 1, cols, 0);
    }
    return -1;
}

static inline void asi_matmul_batch_range(float *out, const uint8_t *w, const LilaQ8KBlock *qx_all,
                                          int n_tokens, int start, int end, int rows, int bpr, uint32_t quant_type)
{
    if (quant_type == 12)
        q4k_matmul_batch_q8k_range(out, w, qx_all, n_tokens, start, end, rows, bpr);
    else if (quant_type == 14)
        q6k_matmul_batch_q8k_range(out, w, qx_all, n_tokens, start, end, rows, bpr);
}

typedef struct {
    const uint8_t *w;
    const LilaQ8KBlock *qx_all;
    float *out;
    float *out_rest;
    int n_tokens;
    int rows;
    int blocks_per_row;
    uint32_t quant_type;
    _Alignas(64) atomic_int next_row;
    int chunk_rows;
} AsiMatmulBatchCtx;

static void asi_matmul_batch_worker(int tid, int n_threads, void *vctx)
{
    (void)tid;
    AsiMatmulBatchCtx *ctx = (AsiMatmulBatchCtx *)vctx;
    if (n_threads <= 1)
    {
        if (ctx->out_rest)
            q6k_matmul_batch_q8k_split_range(ctx->out, ctx->out_rest, ctx->w, ctx->qx_all,
                                             ctx->n_tokens, 0, ctx->rows, ctx->rows, ctx->blocks_per_row);
        else
            asi_matmul_batch_range(ctx->out, ctx->w, ctx->qx_all,
                                   ctx->n_tokens, 0, ctx->rows, ctx->rows, ctx->blocks_per_row, ctx->quant_type);
        return;
    }
    int rows = ctx->rows;
    int chunk = ctx->chunk_rows;
    for (;;)
    {
        int start = atomic_fetch_add_explicit(&ctx->next_row, chunk, memory_order_relaxed);
        if (start >= rows) break;
        int end = start + chunk;
        if (end > rows) end = rows;
        if (ctx->out_rest)
            q6k_matmul_batch_q8k_split_range(ctx->out, ctx->out_rest, ctx->w, ctx->qx_all,
                                             ctx->n_tokens, start, end, rows, ctx->blocks_per_row);
        else
            asi_matmul_batch_range(ctx->out, ctx->w, ctx->qx_all,
                                   ctx->n_tokens, start, end, rows, ctx->blocks_per_row, ctx->quant_type);
    }
}

static int matmul_batch_q8k_pool(AsiThreadPool *pool,
                                 const AsiV2Cartridge *cart,
                                 const AsiV2TensorEntry *t,
                                 float *out,
                                 const LilaQ8KBlock *qx_all,
                                 int n_tokens,
                                 int bpr)
{
    const uint8_t *w = tensor_data(cart, t);
    int rows = (int)t->shape[1];
    int chunk = (rows >= 16384) ? 128 : ((rows >= 256) ? 32 : 16);
    AsiMatmulBatchCtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.w = w;
    ctx.qx_all = qx_all;
    ctx.out = out;
    ctx.out_rest = NULL;
    ctx.n_tokens = n_tokens;
    ctx.rows = rows;
    ctx.blocks_per_row = bpr;
    ctx.quant_type = t->quant_type;
    atomic_init(&ctx.next_row, 0);
    ctx.chunk_rows = chunk;
    if (pool && rows >= 16)
        asi_pool_run(pool, asi_matmul_batch_worker, &ctx);
    else
        asi_matmul_batch_worker(0, 1, &ctx);
    return 0;
}

static int matmul_batch_q8k_split_pool(AsiThreadPool *pool,
                                       const AsiV2Cartridge *cart,
                                       const AsiV2TensorEntry *t,
                                       float *out0,
                                       float *out_rest,
                                       const LilaQ8KBlock *qx_all,
                                       int n_tokens,
                                       int bpr)
{
    const uint8_t *w = tensor_data(cart, t);
    int rows = (int)t->shape[1];
    int chunk = (rows >= 16384) ? 128 : ((rows >= 256) ? 32 : 16);
    AsiMatmulBatchCtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.w = w;
    ctx.qx_all = qx_all;
    ctx.out = out0;
    ctx.out_rest = out_rest;
    ctx.n_tokens = n_tokens;
    ctx.rows = rows;
    ctx.blocks_per_row = bpr;
    ctx.quant_type = t->quant_type;
    atomic_init(&ctx.next_row, 0);
    ctx.chunk_rows = chunk;
    if (pool && rows >= 16)
        asi_pool_run(pool, asi_matmul_batch_worker, &ctx);
    else
        asi_matmul_batch_worker(0, 1, &ctx);
    return 0;
}

static inline int asi_tensor_blis_supported(const AsiV2TensorEntry *t)
{
    if (!t || (t->layout_flags & ASI_TENSOR_LAYOUT_COL_MAJOR)) return 0;
    if (t->quant_type == 14u && !(t->layout_flags & ASI_TENSOR_LAYOUT_AVX2_REPACK))
        return (((int)t->shape[0] % 256) == 0);
    if (t->quant_type == ASI_QUANT_Q4_REPACK || t->quant_type == ASI_QUANT_Q3_SPLITLUT)
        return 1;
    return 0;
}

static inline int asi_blis_prefill_enabled(void)
{
    const char *e = getenv("LILA_ASI_BLIS_PREFILL");
    return (!e || e[0] != '0') ? 1 : 0;
}

typedef struct {
    LilaQ8KBlock *qx_all;
    const float *in;
    int n_tokens;
    int cols;
} AsiQuantBatchCtx;

static void asi_quantize_q8k_worker(int tid, int n_threads, void *vctx)
{
    const AsiQuantBatchCtx *ctx = (const AsiQuantBatchCtx *)vctx;
    int start = (ctx->n_tokens * tid) / n_threads;
    int end = (ctx->n_tokens * (tid + 1)) / n_threads;
    for (int tok = start; tok < end; ++tok)
    {
        quantize_vec_q8k_strided(ctx->qx_all + tok, ctx->n_tokens,
                                 ctx->in + (size_t)tok * ctx->cols, ctx->cols);
    }
}

static void quantize_batch_q8k_pool(AsiThreadPool *pool,
                                    LilaQ8KBlock *qx_all,
                                    const float *in,
                                    int n_tokens,
                                    int cols)
{
    AsiQuantBatchCtx ctx = {qx_all, in, n_tokens, cols};
    if (pool && n_tokens >= 8)
        asi_pool_run(pool, asi_quantize_q8k_worker, &ctx);
    else
        asi_quantize_q8k_worker(0, 1, &ctx);
}

typedef struct {
    LilaQ8_64Block *qx_all;
    const float *in;
    int n_tokens;
    int cols;
} AsiQuantQ8_64BatchCtx;

static void asi_quantize_q8_64_worker(int tid, int n_threads, void *vctx)
{
    const AsiQuantQ8_64BatchCtx *ctx = (const AsiQuantQ8_64BatchCtx *)vctx;
    int start = (ctx->n_tokens * tid) / n_threads;
    int end = (ctx->n_tokens * (tid + 1)) / n_threads;
    for (int tok = start; tok < end; ++tok)
    {
        quantize_vec_q8_64_strided(ctx->qx_all + tok, ctx->n_tokens,
                                   ctx->in + (size_t)tok * (size_t)ctx->cols, ctx->cols);
    }
}

static void quantize_batch_q8_64_pool(AsiThreadPool *pool,
                                      LilaQ8_64Block *qx_all,
                                      const float *in,
                                      int n_tokens,
                                      int cols)
{
    AsiQuantQ8_64BatchCtx ctx = {qx_all, in, n_tokens, cols};
    if (pool && n_tokens >= 8)
        asi_pool_run(pool, asi_quantize_q8_64_worker, &ctx);
    else
        asi_quantize_q8_64_worker(0, 1, &ctx);
}

typedef struct {
    const uint8_t *w[4];
    float *out[4];
    int rows[4];
    uint32_t quant_type[4];
    int n_mats;
    const LilaQ8_64Block *qx_all;
    const LilaQ3LutBlock *q3_luts;
    int n_tokens;
    int n_blk64;
    int n_blk128;
    int apply_silu_mul;
    _Alignas(64) atomic_int next_row[4];
} AsiE07BatchCtx;

static void asi_e07_batch_eval_mat_range(const AsiE07BatchCtx *ctx, int m, int start, int end)
{
    if (ctx->quant_type[m] == ASI_QUANT_Q4_REPACK)
    {
        q4_repack_matmul_batch_q8_range(ctx->out[m], ctx->w[m], ctx->qx_all,
                                        ctx->n_tokens, start, end, ctx->rows[m], ctx->n_blk64);
    }
    else if (ctx->quant_type[m] == ASI_QUANT_Q3_SPLITLUT)
    {
        size_t tile_bytes = (size_t)ctx->n_blk128 * Q3_SPLITLUT_BLK_BYTES;
        int t0 = start / Q3_SPLITLUT_TILE_ROWS;
        int t1 = (end + Q3_SPLITLUT_TILE_ROWS - 1) / Q3_SPLITLUT_TILE_ROWS;
        for (int tile = t0; tile < t1; ++tile)
        {
            int r_base = tile * Q3_SPLITLUT_TILE_ROWS;
            int valid = ctx->rows[m] - r_base;
            if (valid > Q3_SPLITLUT_TILE_ROWS) valid = Q3_SPLITLUT_TILE_ROWS;
            const uint8_t *tp = ctx->w[m] + (size_t)tile * tile_bytes;
            for (int tok = 0; tok < ctx->n_tokens; ++tok)
            {
                const LilaQ3LutBlock *lb = ctx->q3_luts + (size_t)tok * (size_t)ctx->n_blk128;
                if (r_base >= start && r_base + valid <= end)
                {
                    q3_splitlut_tile_simd(tp, ctx->n_blk128, lb, valid,
                                          ctx->out[m] + (size_t)tok * (size_t)ctx->rows[m] + (size_t)r_base);
                }
                else
                {
                    float tmp[Q3_SPLITLUT_TILE_ROWS];
                    q3_splitlut_tile_simd(tp, ctx->n_blk128, lb, valid, tmp);
                    int lo = (start > r_base) ? (start - r_base) : 0;
                    int hi = (end < r_base + valid) ? (end - r_base) : valid;
                    float *dst = ctx->out[m] + (size_t)tok * (size_t)ctx->rows[m];
                    for (int lr = lo; lr < hi; ++lr) dst[r_base + lr] = tmp[lr];
                }
            }
        }
    }
}

static void asi_e07_batch_worker(int tid, int n_threads, void *vctx)
{
    (void)tid;
    AsiE07BatchCtx *ctx = (AsiE07BatchCtx *)vctx;
    if (n_threads <= 1)
    {
        for (int m = 0; m < ctx->n_mats; ++m)
            asi_e07_batch_eval_mat_range(ctx, m, 0, ctx->rows[m]);
        if (ctx->apply_silu_mul && ctx->n_mats >= 2 && ctx->rows[0] == ctx->rows[1])
        {
            fast_silu_mul_vec(ctx->out[0], ctx->out[1],
                              (uint32_t)((size_t)ctx->n_tokens * (size_t)ctx->rows[0]));
        }
        return;
    }

    if (ctx->apply_silu_mul && ctx->n_mats == 2 && ctx->rows[0] == ctx->rows[1] &&
        ctx->quant_type[0] == ctx->quant_type[1])
    {
        int rows = ctx->rows[0];
        int chunk = (ctx->quant_type[0] == ASI_QUANT_Q3_SPLITLUT) ? Q3_SPLITLUT_TILE_ROWS : 32;
        for (;;)
        {
            int start = atomic_fetch_add_explicit(&ctx->next_row[0], chunk, memory_order_relaxed);
            if (start >= rows) break;
            int end = start + chunk;
            if (end > rows) end = rows;
            asi_e07_batch_eval_mat_range(ctx, 0, start, end);
            asi_e07_batch_eval_mat_range(ctx, 1, start, end);
            for (int t = 0; t < ctx->n_tokens; ++t)
            {
                size_t off = (size_t)t * (size_t)rows + (size_t)start;
                fast_silu_mul_vec(ctx->out[0] + off, ctx->out[1] + off, (uint32_t)(end - start));
            }
        }
        return;
    }

    for (int m = 0; m < ctx->n_mats; ++m)
    {
        int rows = ctx->rows[m];
        int chunk = (ctx->quant_type[m] == ASI_QUANT_Q3_SPLITLUT) ? Q3_SPLITLUT_TILE_ROWS : 32;
        for (;;)
        {
            int start = atomic_fetch_add_explicit(&ctx->next_row[m], chunk, memory_order_relaxed);
            if (start >= rows) break;
            int end = start + chunk;
            if (end > rows) end = rows;
            asi_e07_batch_eval_mat_range(ctx, m, start, end);
        }
    }
}

static int matmul_batch_e07_pool(AsiThreadPool *pool,
                                 const AsiV2Cartridge *cart,
                                 const AsiV2TensorEntry **ts,
                                 float **outs,
                                 int n_mats,
                                 const float *in,
                                 int n_tokens,
                                 int cols,
                                 int apply_silu_mul)
{
    int any_q4r = 0, any_q3 = 0;
    for (int m = 0; m < n_mats; ++m)
    {
        if (ts[m]->quant_type == ASI_QUANT_Q4_REPACK) any_q4r = 1;
        else if (ts[m]->quant_type == ASI_QUANT_Q3_SPLITLUT) any_q3 = 1;
    }
    int n_blk64 = (cols + Q4_REPACK_GROUP_SIZE - 1) / Q4_REPACK_GROUP_SIZE;
    int n_blk128 = (cols + Q3_SPLITLUT_BLK_COLS - 1) / Q3_SPLITLUT_BLK_COLS;

    LilaQ8_64Block stack_qx[128];
    LilaQ8_64Block *qx_all = NULL;
    LilaQ3LutBlock *q3_luts = NULL;

    if (any_q4r)
    {
        size_t need = (size_t)n_tokens * (size_t)n_blk64;
        qx_all = (need <= 128u) ? stack_qx : lila_q8_64_alloc(need);
        if (!qx_all) return -1;
        quantize_batch_q8_64_pool(pool, qx_all, in, n_tokens, cols);
    }
    if (any_q3)
    {
        size_t need_l = (size_t)n_tokens * (size_t)n_blk128;
        q3_luts = (LilaQ3LutBlock *)asi_aligned_alloc(32, need_l * sizeof(LilaQ3LutBlock));
        if (!q3_luts) { if (qx_all && qx_all != stack_qx) lila_q8_64_free(qx_all); return -1; }
        for (int tok = 0; tok < n_tokens; ++tok)
        {
            q3_splitlut_build_lut(q3_luts + (size_t)tok * (size_t)n_blk128,
                                  in + (size_t)tok * (size_t)cols, cols);
        }
    }

    AsiE07BatchCtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.n_mats = n_mats;
    ctx.qx_all = qx_all;
    ctx.q3_luts = q3_luts;
    ctx.n_tokens = n_tokens;
    ctx.n_blk64 = n_blk64;
    ctx.n_blk128 = n_blk128;
    ctx.apply_silu_mul = apply_silu_mul;
    int max_rows = 0;
    for (int m = 0; m < n_mats; ++m)
    {
        ctx.w[m] = tensor_data(cart, ts[m]);
        ctx.out[m] = outs[m];
        ctx.rows[m] = (int)ts[m]->shape[1];
        ctx.quant_type[m] = ts[m]->quant_type;
        atomic_init(&ctx.next_row[m], 0);
        if (ctx.rows[m] > max_rows) max_rows = ctx.rows[m];
    }
    if (pool && (max_rows >= 16 || n_mats > 1))
        asi_pool_run(pool, asi_e07_batch_worker, &ctx);
    else
        asi_e07_batch_worker(0, 1, &ctx);

    if (qx_all && qx_all != stack_qx) lila_q8_64_free(qx_all);
    if (q3_luts) asi_aligned_free(q3_luts);
    return 0;
}

#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
/* BLIS L1 K-panel ceiling (256 x 16 x 4B = 16 KiB, fits cleanly in 32 KiB L1d) */
#define ASI_BLIS_KC_BLOCK 256

static inline void asi_dequant_q6k_16rows_blk(const uint8_t *w_q6k,
                                              size_t row_bytes,
                                              int n_base,
                                              int n_cnt,
                                              int blk_idx,
                                              float *Bp)
{
    if (n_cnt < 16)
        memset(Bp, 0, 256u * 16u * sizeof(float));
    for (int r = 0; r < n_cnt; ++r)
    {
        const uint8_t *blk = w_q6k + (size_t)(n_base + r) * row_bytes + (size_t)blk_idx * 210u;
        const uint8_t *ql = blk;
        const uint8_t *qh = blk + 128;
        const int8_t  *sc = (const int8_t *)(blk + 192);
        uint16_t d_raw;
        memcpy(&d_raw, blk + 208, 2);
        float d = _cvtsh_ss(d_raw);
        for (int half = 0; half < 2; ++half)
        {
            const uint8_t *qlh = ql + half * 64;
            const uint8_t *qhh = qh + half * 32;
            const int8_t  *sch = sc + half * 8;
            int base_k = half * 128;
            for (int l = 0; l < 32; ++l)
            {
                int is = l >> 4;
                float s0 = d * (float)sch[is + 0];
                float s2 = d * (float)sch[is + 2];
                float s4 = d * (float)sch[is + 4];
                float s6 = d * (float)sch[is + 6];
                Bp[(base_k + l +  0) * 16 + r] = s0 * (float)((int)((qlh[l +  0] & 0x0Fu) | (((qhh[l] >> 0) & 3u) << 4)) - 32);
                Bp[(base_k + l + 32) * 16 + r] = s2 * (float)((int)((qlh[l + 32] & 0x0Fu) | (((qhh[l] >> 2) & 3u) << 4)) - 32);
                Bp[(base_k + l + 64) * 16 + r] = s4 * (float)((int)((qlh[l +  0] >>   4)  | (((qhh[l] >> 4) & 3u) << 4)) - 32);
                Bp[(base_k + l + 96) * 16 + r] = s6 * (float)((int)((qlh[l + 32] >>   4)  | (((qhh[l] >> 6) & 3u) << 4)) - 32);
            }
        }
    }
}

static inline void asi_dequant_q4r_16rows_blk(const uint8_t *w_q4r,
                                              size_t row_bytes,
                                              int n_base,
                                              int n_cnt,
                                              int blk256_idx,
                                              int kc,
                                              float *Bp)
{
    if (n_cnt < 16)
        memset(Bp, 0, (size_t)kc * 16u * sizeof(float));
    int n_sub = (kc + Q4_REPACK_GROUP_SIZE - 1) / Q4_REPACK_GROUP_SIZE;
    for (int r = 0; r < n_cnt; ++r)
    {
        const uint8_t *row_blk0 = w_q4r + (size_t)(n_base + r) * row_bytes +
                                  (size_t)blk256_idx * 4u * Q4_REPACK_BLOCK_SIZE;
        for (int sb = 0; sb < n_sub; ++sb)
        {
            const uint8_t *blk = row_blk0 + (size_t)sb * Q4_REPACK_BLOCK_SIZE;
            uint16_t d_raw;
            memcpy(&d_raw, blk, 2);
            float d = _cvtsh_ss(d_raw);
            const uint8_t *qs = blk + 2;
            int base_k = sb * Q4_REPACK_GROUP_SIZE;
            int avail = kc - base_k;
            if (avail > Q4_REPACK_GROUP_SIZE) avail = Q4_REPACK_GROUP_SIZE;
            if (avail == Q4_REPACK_GROUP_SIZE)
            {
                for (int l = 0; l < 32; ++l)
                {
                    Bp[(base_k + l +  0) * 16 + r] = d * (float)((int)(qs[l] & 0x0Fu) - 8);
                    Bp[(base_k + l + 32) * 16 + r] = d * (float)((int)(qs[l] >> 4) - 8);
                }
            }
            else
            {
                for (int l = 0; l < 32; ++l)
                {
                    if (l < avail) Bp[(base_k + l) * 16 + r] = d * (float)((int)(qs[l] & 0x0Fu) - 8);
                    if (l + 32 < avail) Bp[(base_k + l + 32) * 16 + r] = d * (float)((int)(qs[l] >> 4) - 8);
                }
            }
        }
    }
}

static inline void asi_dequant_q3_16rows_blk(const uint8_t *w_q3,
                                             int n_blk128,
                                             int n_base,
                                             int n_cnt,
                                             int blk128_idx,
                                             int kc,
                                             float *Bp)
{
    if (n_cnt < 16)
        memset(Bp, 0, (size_t)kc * 16u * sizeof(float));
    int tile = n_base / Q3_SPLITLUT_TILE_ROWS;
    const uint8_t *bp_h = w_q3 + ((size_t)tile * (size_t)n_blk128 + (size_t)blk128_idx) * Q3_SPLITLUT_BLK_BYTES;
    const uint8_t *bp_l = bp_h + Q3_SPLITLUT_BLK_HI2_BYTES;
    const uint16_t *sp  = (const uint16_t *)(bp_l + Q3_SPLITLUT_BLK_LO1_BYTES);
    for (int r = 0; r < n_cnt; ++r)
    {
        int lr = (n_base & 63) + r;
        int bo = lr & 31;
        int sh = (lr >= 32) ? 4 : 0;
        uint16_t sc_raw;
        memcpy(&sc_raw, sp + lr, 2);
        float step = _cvtsh_ss(sc_raw);
        for (int k = 0; k < kc; ++k)
        {
            uint8_t nib_h = (uint8_t)((bp_h[(size_t)(k >> 1) * 32u + (size_t)bo] >> sh) & 0x0Fu);
            uint8_t hi2 = (k & 1) ? (uint8_t)((nib_h >> 2) & 0x03u) : (uint8_t)(nib_h & 0x03u);
            uint8_t nib_l = (uint8_t)((bp_l[(size_t)(k >> 2) * 32u + (size_t)bo] >> sh) & 0x0Fu);
            uint8_t lo1 = (uint8_t)((nib_l >> (k & 3)) & 0x01u);
            Bp[k * 16 + r] = step * ((float)(((int)hi2 << 1) | (int)lo1) - 3.5f);
        }
    }
}

static inline void asi_kernel_6x16_f32_avx2(int kc,
                                            const float *a0, const float *a1, const float *a2,
                                            const float *a3, const float *a4, const float *a5,
                                            const float *Bp,
                                            float *c0, float *c1, float *c2,
                                            float *c3, float *c4, float *c5,
                                            float beta, int n_cnt)
{
    __m256 r00, r01, r10, r11, r20, r21, r30, r31, r40, r41, r50, r51;
    if (beta == 0.0f)
    {
        r00 = _mm256_setzero_ps(); r01 = _mm256_setzero_ps();
        r10 = _mm256_setzero_ps(); r11 = _mm256_setzero_ps();
        r20 = _mm256_setzero_ps(); r21 = _mm256_setzero_ps();
        r30 = _mm256_setzero_ps(); r31 = _mm256_setzero_ps();
        r40 = _mm256_setzero_ps(); r41 = _mm256_setzero_ps();
        r50 = _mm256_setzero_ps(); r51 = _mm256_setzero_ps();
    }
    else if (n_cnt == 16)
    {
        r00 = _mm256_loadu_ps(c0); r01 = _mm256_loadu_ps(c0 + 8);
        r10 = _mm256_loadu_ps(c1); r11 = _mm256_loadu_ps(c1 + 8);
        r20 = _mm256_loadu_ps(c2); r21 = _mm256_loadu_ps(c2 + 8);
        r30 = _mm256_loadu_ps(c3); r31 = _mm256_loadu_ps(c3 + 8);
        r40 = _mm256_loadu_ps(c4); r41 = _mm256_loadu_ps(c4 + 8);
        r50 = _mm256_loadu_ps(c5); r51 = _mm256_loadu_ps(c5 + 8);
    }
    else
    {
        _Alignas(32) float tmp_in[16] = {0};
        float *cs[6] = {c0, c1, c2, c3, c4, c5};
        __m256 tlo[6], thi[6];
        for (int i = 0; i < 6; ++i)
        {
            memcpy(tmp_in, cs[i], (size_t)n_cnt * sizeof(float));
            tlo[i] = _mm256_load_ps(tmp_in);
            thi[i] = _mm256_load_ps(tmp_in + 8);
        }
        r00 = tlo[0]; r01 = thi[0]; r10 = tlo[1]; r11 = thi[1]; r20 = tlo[2]; r21 = thi[2];
        r30 = tlo[3]; r31 = thi[3]; r40 = tlo[4]; r41 = thi[4]; r50 = tlo[5]; r51 = thi[5];
    }

    for (int k = 0; k < kc; ++k)
    {
        __m256 b0 = _mm256_load_ps(Bp + k * 16);
        __m256 b1 = _mm256_load_ps(Bp + k * 16 + 8);
        __m256 va;
        va = _mm256_broadcast_ss(a0 + k); r00 = _mm256_fmadd_ps(va, b0, r00); r01 = _mm256_fmadd_ps(va, b1, r01);
        va = _mm256_broadcast_ss(a1 + k); r10 = _mm256_fmadd_ps(va, b0, r10); r11 = _mm256_fmadd_ps(va, b1, r11);
        va = _mm256_broadcast_ss(a2 + k); r20 = _mm256_fmadd_ps(va, b0, r20); r21 = _mm256_fmadd_ps(va, b1, r21);
        va = _mm256_broadcast_ss(a3 + k); r30 = _mm256_fmadd_ps(va, b0, r30); r31 = _mm256_fmadd_ps(va, b1, r31);
        va = _mm256_broadcast_ss(a4 + k); r40 = _mm256_fmadd_ps(va, b0, r40); r41 = _mm256_fmadd_ps(va, b1, r41);
        va = _mm256_broadcast_ss(a5 + k); r50 = _mm256_fmadd_ps(va, b0, r50); r51 = _mm256_fmadd_ps(va, b1, r51);
    }

    if (n_cnt == 16)
    {
        _mm256_storeu_ps(c0, r00); _mm256_storeu_ps(c0 + 8, r01);
        _mm256_storeu_ps(c1, r10); _mm256_storeu_ps(c1 + 8, r11);
        _mm256_storeu_ps(c2, r20); _mm256_storeu_ps(c2 + 8, r21);
        _mm256_storeu_ps(c3, r30); _mm256_storeu_ps(c3 + 8, r31);
        _mm256_storeu_ps(c4, r40); _mm256_storeu_ps(c4 + 8, r41);
        _mm256_storeu_ps(c5, r50); _mm256_storeu_ps(c5 + 8, r51);
    }
    else
    {
        _Alignas(32) float tmp_out[16];
        float *cs[6] = {c0, c1, c2, c3, c4, c5};
        __m256 slo[6] = {r00, r10, r20, r30, r40, r50};
        __m256 shi[6] = {r01, r11, r21, r31, r41, r51};
        for (int i = 0; i < 6; ++i)
        {
            _mm256_store_ps(tmp_out, slo[i]);
            _mm256_store_ps(tmp_out + 8, shi[i]);
            memcpy(cs[i], tmp_out, (size_t)n_cnt * sizeof(float));
        }
    }
}

static void asi_blis_6x16_range(float *out,
                                const uint8_t *w,
                                uint32_t quant_type,
                                const float *in,
                                int n_tokens,
                                int start,
                                int end,
                                int rows,
                                int cols)
{
    int k_step = (quant_type == ASI_QUANT_Q3_SPLITLUT) ? Q3_SPLITLUT_BLK_COLS : ASI_BLIS_KC_BLOCK;
    int bpr = (cols + k_step - 1) / k_step;
    size_t row_bytes = (quant_type == 14u)
                           ? ((size_t)bpr * 210u)
                           : lila_quant_row_bytes(quant_type, cols, 0);
    int n_blk128 = (cols + Q3_SPLITLUT_BLK_COLS - 1) / Q3_SPLITLUT_BLK_COLS;
    _Alignas(32) float Bp[ASI_BLIS_KC_BLOCK * 16];

    for (int r = start; r < end; r += 16)
    {
        int n_cnt = (r + 16 <= end) ? 16 : (end - r);
        for (int b = 0; b < bpr; ++b)
        {
            size_t k_off = (size_t)b * (size_t)k_step;
            int kc = cols - (int)k_off;
            if (kc > k_step) kc = k_step;

            if (quant_type == 14u)
                asi_dequant_q6k_16rows_blk(w, row_bytes, r, n_cnt, b, Bp);
            else if (quant_type == ASI_QUANT_Q4_REPACK)
                asi_dequant_q4r_16rows_blk(w, row_bytes, r, n_cnt, b, kc, Bp);
            else
                asi_dequant_q3_16rows_blk(w, n_blk128, r, n_cnt, b, kc, Bp);

            float beta = (b == 0) ? 0.0f : 1.0f;
            int t = 0;
            for (; t + 5 < n_tokens; t += 6)
            {
                asi_kernel_6x16_f32_avx2(
                    kc,
                    in + (size_t)(t + 0) * cols + k_off,
                    in + (size_t)(t + 1) * cols + k_off,
                    in + (size_t)(t + 2) * cols + k_off,
                    in + (size_t)(t + 3) * cols + k_off,
                    in + (size_t)(t + 4) * cols + k_off,
                    in + (size_t)(t + 5) * cols + k_off,
                    Bp,
                    out + (size_t)(t + 0) * rows + r,
                    out + (size_t)(t + 1) * rows + r,
                    out + (size_t)(t + 2) * rows + r,
                    out + (size_t)(t + 3) * rows + r,
                    out + (size_t)(t + 4) * rows + r,
                    out + (size_t)(t + 5) * rows + r,
                    beta, n_cnt);
            }
            for (; t < n_tokens; ++t)
            {
                float *ct = out + (size_t)t * rows + r;
                const float *at = in + (size_t)t * cols + k_off;
                __m256 r0, r1;
                if (beta == 0.0f)
                {
                    r0 = _mm256_setzero_ps();
                    r1 = _mm256_setzero_ps();
                }
                else if (n_cnt == 16)
                {
                    r0 = _mm256_loadu_ps(ct);
                    r1 = _mm256_loadu_ps(ct + 8);
                }
                else
                {
                    _Alignas(32) float tmp_in[16] = {0};
                    memcpy(tmp_in, ct, (size_t)n_cnt * sizeof(float));
                    r0 = _mm256_load_ps(tmp_in);
                    r1 = _mm256_load_ps(tmp_in + 8);
                }
                for (int k = 0; k < kc; ++k)
                {
                    __m256 va = _mm256_broadcast_ss(at + k);
                    r0 = _mm256_fmadd_ps(va, _mm256_load_ps(Bp + k * 16), r0);
                    r1 = _mm256_fmadd_ps(va, _mm256_load_ps(Bp + k * 16 + 8), r1);
                }
                if (n_cnt == 16)
                {
                    _mm256_storeu_ps(ct, r0);
                    _mm256_storeu_ps(ct + 8, r1);
                }
                else
                {
                    _Alignas(32) float tmp_out[16];
                    _mm256_store_ps(tmp_out, r0);
                    _mm256_store_ps(tmp_out + 8, r1);
                    memcpy(ct, tmp_out, (size_t)n_cnt * sizeof(float));
                }
            }
        }
    }
}

typedef struct {
    const uint8_t *w[4];
    float *out[4];
    int rows[4];
    uint32_t quant_type[4];
    int n_mats;
    const float *in;
    int n_tokens;
    int cols;
    int apply_silu_mul;
    _Alignas(64) atomic_int next_row[4];
} AsiBlisCtx;

static void asi_blis_worker(int tid, int n_threads, void *vctx)
{
    (void)tid;
    AsiBlisCtx *ctx = (AsiBlisCtx *)vctx;
    if (n_threads <= 1)
    {
        for (int m = 0; m < ctx->n_mats; ++m)
        {
            asi_blis_6x16_range(ctx->out[m], ctx->w[m], ctx->quant_type[m], ctx->in,
                                ctx->n_tokens, 0, ctx->rows[m], ctx->rows[m], ctx->cols);
        }
        if (ctx->apply_silu_mul && ctx->n_mats == 2)
        {
            fast_silu_mul_vec(ctx->out[0], ctx->out[1],
                              (uint32_t)((size_t)ctx->n_tokens * (size_t)ctx->rows[0]));
        }
        return;
    }

    if (ctx->apply_silu_mul && ctx->n_mats == 2 && ctx->rows[0] == ctx->rows[1])
    {
        int rows = ctx->rows[0];
        int chunk = (rows >= 256) ? 32 : 16;
        for (;;)
        {
            int start = atomic_fetch_add_explicit(&ctx->next_row[0], chunk, memory_order_relaxed);
            if (start >= rows) break;
            int end = start + chunk;
            if (end > rows) end = rows;
            asi_blis_6x16_range(ctx->out[0], ctx->w[0], ctx->quant_type[0], ctx->in,
                                ctx->n_tokens, start, end, rows, ctx->cols);
            asi_blis_6x16_range(ctx->out[1], ctx->w[1], ctx->quant_type[1], ctx->in,
                                ctx->n_tokens, start, end, rows, ctx->cols);
            for (int t = 0; t < ctx->n_tokens; ++t)
            {
                size_t off = (size_t)t * (size_t)rows + (size_t)start;
                fast_silu_mul_vec(ctx->out[0] + off, ctx->out[1] + off, (uint32_t)(end - start));
            }
        }
        return;
    }

    for (int m = 0; m < ctx->n_mats; ++m)
    {
        int rows = ctx->rows[m];
        int chunk = (rows >= 256) ? 32 : 16;
        for (;;)
        {
            int start = atomic_fetch_add_explicit(&ctx->next_row[m], chunk, memory_order_relaxed);
            if (start >= rows) break;
            int end = start + chunk;
            if (end > rows) end = rows;
            asi_blis_6x16_range(ctx->out[m], ctx->w[m], ctx->quant_type[m], ctx->in,
                                ctx->n_tokens, start, end, rows, ctx->cols);
        }
    }
}

#endif /* __AVX2__ && __FMA__ && __F16C__ */

static int matmul_batch_tensor(AsiThreadPool *pool,
                               const AsiV2Cartridge *cart,
                               const AsiV2TensorEntry *t,
                               float *out,
                               const float *in,
                               int n_tokens,
                               int fp32_ref)
{
    const uint8_t *w = tensor_data(cart, t);
    int cols = (int)t->shape[0];
    int rows = (int)t->shape[1];
    if (n_tokens <= 1)
    {
        return matvec_tensor_pool(pool, cart, t, out, in, fp32_ref);
    }
    if (fp32_ref)
    {
        return matmul_fp32_pool(pool, cart, &t, &out, 1, in, n_tokens, cols, 0);
    }
    if ((t->quant_type == ASI_QUANT_Q4_REPACK && !(t->layout_flags & ASI_TENSOR_LAYOUT_COL_MAJOR)) ||
        t->quant_type == ASI_QUANT_Q3_SPLITLUT)
    {
        return matmul_batch_e07_pool(pool, cart, &t, &out, 1, in, n_tokens, cols, 0);
    }
    if (t->quant_type == ASI_QUANT_Q4_REPACK && (t->layout_flags & ASI_TENSOR_LAYOUT_COL_MAJOR))
    {
        for (int tok = 0; tok < n_tokens; ++tok)
        {
            matvec_q4_col_major_pool(pool, cart, t, out + (size_t)tok * rows, in + (size_t)tok * cols, 1.0f);
        }
        return 0;
    }
    if ((t->quant_type == 12 || t->quant_type == 14) && !(t->layout_flags & ASI_TENSOR_LAYOUT_AVX2_REPACK))
    {
        int bpr = (cols + QK_K - 1) / QK_K;
        if (bpr <= 0) return 0;
        size_t need = (size_t)n_tokens * (size_t)bpr;
        LilaQ8KBlock stack_qx[256];
        LilaQ8KBlock *qx_all = (need <= 256) ? stack_qx : lila_q8k_alloc(need);
        if (qx_all)
        {
            if (qx_all == stack_qx) memset(stack_qx, 0, need * sizeof(LilaQ8KBlock));
            for (int tok = 0; tok < n_tokens; ++tok)
            {
                quantize_vec_q8k_strided(qx_all + tok, n_tokens, in + (size_t)tok * cols, cols);
            }
            matmul_batch_q8k_pool(pool, cart, t, out, qx_all, n_tokens, bpr);
            if (qx_all != stack_qx) free(qx_all);
            return 0;
        }
    }
    if (t->quant_type == 12) /* Q4_K */
    {
        q4k_matmul_batch(out, w, in, n_tokens, rows, cols);
        return 0;
    }
    if (t->quant_type == 14) /* Q6_K */
    {
        if (t->layout_flags & ASI_TENSOR_LAYOUT_AVX2_REPACK)
            q6k_repacked_matmul_batch(out, w, in, n_tokens, rows, cols);
        else
            q6k_matmul_batch(out, w, in, n_tokens, rows, cols);
        return 0;
    }
    if (lila_quant_is_supported(t->quant_type))
    {
        return matmul_fp32_pool(pool, cart, &t, &out, 1, in, n_tokens, cols, 0);
    }
    return -1;
}

static int matmul_batch_tensor_qx(AsiThreadPool *pool,
                                  const AsiV2Cartridge *cart,
                                  const AsiV2TensorEntry *t,
                                  float *out,
                                  const LilaQ8KBlock *qx_all,
                                  const float *in,
                                  int n_tokens,
                                  int bpr,
                                  int fp32_ref)
{
    if (fp32_ref)
    {
        return matmul_fp32_pool(pool, cart, &t, &out, 1, in, n_tokens, (int)t->shape[0], 0);
    }
#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
    if (in && n_tokens >= asi_runtime_config()->blis_min_tokens &&
        asi_blis_prefill_enabled() && asi_cpu_has_avx2_fma_f16c() &&
        asi_tensor_blis_supported(t))
    {
        AsiBlisCtx ctx;
        memset(&ctx, 0, sizeof(ctx));
        ctx.w[0] = tensor_data(cart, t);
        ctx.out[0] = out;
        ctx.rows[0] = (int)t->shape[1];
        ctx.quant_type[0] = t->quant_type;
        ctx.n_mats = 1;
        ctx.in = in;
        ctx.n_tokens = n_tokens;
        ctx.cols = (int)t->shape[0];
        atomic_init(&ctx.next_row[0], 0);
        if (pool && ctx.rows[0] >= 16)
            asi_pool_run(pool, asi_blis_worker, &ctx);
        else
            asi_blis_worker(0, 1, &ctx);
        return 0;
    }
#endif
    if (n_tokens > 1 &&
        ((t->quant_type == ASI_QUANT_Q4_REPACK && !(t->layout_flags & ASI_TENSOR_LAYOUT_COL_MAJOR)) ||
         t->quant_type == ASI_QUANT_Q3_SPLITLUT))
    {
        return matmul_batch_e07_pool(pool, cart, &t, &out, 1, in, n_tokens, (int)t->shape[0], 0);
    }
    if (qx_all && n_tokens > 1 && (t->quant_type == 12 || t->quant_type == 14) &&
        !(t->layout_flags & ASI_TENSOR_LAYOUT_AVX2_REPACK))
    {
        return matmul_batch_q8k_pool(pool, cart, t, out, qx_all, n_tokens, bpr);
    }
    return matmul_batch_tensor(pool, cart, t, out, in, n_tokens, 0);
}

typedef struct {
    const uint8_t *w[4];
    float *out[4];
    int rows[4];
    uint32_t quant_type[4];
    int n_mats;
    const LilaQ8KBlock *qx_all;
    int n_tokens;
    int blocks_per_row;
    int apply_silu_mul;
    _Alignas(64) atomic_int next_row[4];
} AsiFusedMatmulBatchCtx;

static void asi_fused_matmul_batch_worker(int tid, int n_threads, void *vctx)
{
    (void)tid;
    AsiFusedMatmulBatchCtx *ctx = (AsiFusedMatmulBatchCtx *)vctx;
    if (n_threads <= 1)
    {
        for (int m = 0; m < ctx->n_mats; ++m)
        {
            asi_matmul_batch_range(ctx->out[m], ctx->w[m], ctx->qx_all,
                                   ctx->n_tokens, 0, ctx->rows[m], ctx->rows[m],
                                   ctx->blocks_per_row, ctx->quant_type[m]);
        }
        if (ctx->apply_silu_mul && ctx->n_mats >= 2)
        {
            fast_silu_mul_vec(ctx->out[0], ctx->out[1],
                              (uint32_t)((size_t)ctx->n_tokens * (size_t)ctx->rows[0]));
        }
        return;
    }

    if (ctx->apply_silu_mul && ctx->n_mats == 2 && ctx->rows[0] == ctx->rows[1])
    {
        int rows = ctx->rows[0];
        int chunk = (rows >= 256) ? 32 : 16;
        for (;;)
        {
            int start = atomic_fetch_add_explicit(&ctx->next_row[0], chunk, memory_order_relaxed);
            if (start >= rows) break;
            int end = start + chunk;
            if (end > rows) end = rows;
            asi_matmul_batch_range(ctx->out[0], ctx->w[0], ctx->qx_all,
                                   ctx->n_tokens, start, end, rows,
                                   ctx->blocks_per_row, ctx->quant_type[0]);
            asi_matmul_batch_range(ctx->out[1], ctx->w[1], ctx->qx_all,
                                   ctx->n_tokens, start, end, rows,
                                   ctx->blocks_per_row, ctx->quant_type[1]);
            for (int t = 0; t < ctx->n_tokens; ++t)
            {
                size_t off = (size_t)t * (size_t)rows + (size_t)start;
                fast_silu_mul_vec(ctx->out[0] + off, ctx->out[1] + off, (uint32_t)(end - start));
            }
        }
        return;
    }

    for (int m = 0; m < ctx->n_mats; ++m)
    {
        int rows = ctx->rows[m];
        int chunk = (rows >= 256) ? 32 : 16;
        for (;;)
        {
            int start = atomic_fetch_add_explicit(&ctx->next_row[m], chunk, memory_order_relaxed);
            if (start >= rows) break;
            int end = start + chunk;
            if (end > rows) end = rows;
            asi_matmul_batch_range(ctx->out[m], ctx->w[m], ctx->qx_all,
                                   ctx->n_tokens, start, end, rows,
                                   ctx->blocks_per_row, ctx->quant_type[m]);
        }
    }
}

static int matmul_batch_fused_q8k_pool(AsiThreadPool *pool,
                                       const AsiV2Cartridge *cart,
                                       const AsiV2TensorEntry **ts,
                                       float **outs,
                                       int n_mats,
                                       const LilaQ8KBlock *qx_all,
                                       const float *in,
                                       int n_tokens,
                                       int bpr,
                                       int apply_silu_mul,
                                       int fp32_ref)
{
    if (fp32_ref)
    {
        return matmul_fp32_pool(pool, cart, ts, outs, n_mats, in, n_tokens, (int)ts[0]->shape[0], apply_silu_mul);
    }
#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
    if (in && n_tokens >= asi_runtime_config()->blis_min_tokens && n_mats >= 1 && n_mats <= 4 &&
        asi_blis_prefill_enabled() && asi_cpu_has_avx2_fma_f16c())
    {
        int all_blis = 1;
        for (int m = 0; m < n_mats; ++m)
        {
            if (!asi_tensor_blis_supported(ts[m]) ||
                (int)ts[m]->shape[0] != (int)ts[0]->shape[0])
            {
                all_blis = 0;
                break;
            }
        }
        if (all_blis)
        {
            AsiBlisCtx ctx;
            memset(&ctx, 0, sizeof(ctx));
            ctx.n_mats = n_mats;
            ctx.in = in;
            ctx.n_tokens = n_tokens;
            ctx.cols = (int)ts[0]->shape[0];
            ctx.apply_silu_mul = apply_silu_mul;
            for (int m = 0; m < n_mats; ++m)
            {
                ctx.w[m] = tensor_data(cart, ts[m]);
                ctx.out[m] = outs[m];
                ctx.rows[m] = (int)ts[m]->shape[1];
                ctx.quant_type[m] = ts[m]->quant_type;
                atomic_init(&ctx.next_row[m], 0);
            }
            if (pool)
                asi_pool_run(pool, asi_blis_worker, &ctx);
            else
                asi_blis_worker(0, 1, &ctx);
            return 0;
        }
    }
#endif
    if (n_tokens <= 1)
    {
        for (int m = 0; m < n_mats; ++m)
        {
            if (matmul_batch_tensor(pool, cart, ts[m], outs[m], in, n_tokens, 0) != 0)
                return -1;
        }
        if (apply_silu_mul && n_mats >= 2)
            fast_silu_mul_vec(outs[0], outs[1], (uint32_t)((size_t)n_tokens * (size_t)ts[0]->shape[1]));
        return 0;
    }
    {
        int all_e07_rm = 1;
        for (int m = 0; m < n_mats; ++m)
        {
            uint32_t qt = ts[m]->quant_type;
            uint32_t lf = ts[m]->layout_flags;
            if (!((qt == ASI_QUANT_Q4_REPACK && !(lf & ASI_TENSOR_LAYOUT_COL_MAJOR)) ||
                  qt == ASI_QUANT_Q3_SPLITLUT) ||
                (int)ts[m]->shape[0] != (int)ts[0]->shape[0])
            {
                all_e07_rm = 0;
                break;
            }
        }
        if (all_e07_rm)
        {
            return matmul_batch_e07_pool(pool, cart, ts, outs, n_mats, in, n_tokens,
                                         (int)ts[0]->shape[0], apply_silu_mul);
        }
    }
    if (!qx_all)
    {
        for (int m = 0; m < n_mats; ++m)
        {
            if (matmul_batch_tensor(pool, cart, ts[m], outs[m], in, n_tokens, 0) != 0)
                return -1;
        }
        if (apply_silu_mul && n_mats >= 2)
            fast_silu_mul_vec(outs[0], outs[1], (uint32_t)((size_t)n_tokens * (size_t)ts[0]->shape[1]));
        return 0;
    }
    for (int m = 0; m < n_mats; ++m)
    {
        if ((ts[m]->quant_type != 12 && ts[m]->quant_type != 14) ||
            (ts[m]->layout_flags & ASI_TENSOR_LAYOUT_AVX2_REPACK))
        {
            return matmul_fp32_pool(pool, cart, ts, outs, n_mats, in, n_tokens, (int)ts[0]->shape[0], apply_silu_mul);
        }
    }
    AsiFusedMatmulBatchCtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.n_mats = n_mats;
    ctx.qx_all = qx_all;
    ctx.n_tokens = n_tokens;
    ctx.blocks_per_row = bpr;
    ctx.apply_silu_mul = apply_silu_mul;
    for (int m = 0; m < n_mats; ++m)
    {
        ctx.w[m] = tensor_data(cart, ts[m]);
        ctx.out[m] = outs[m];
        ctx.rows[m] = (int)ts[m]->shape[1];
        ctx.quant_type[m] = ts[m]->quant_type;
        atomic_init(&ctx.next_row[m], 0);
    }
    if (pool)
        asi_pool_run(pool, asi_fused_matmul_batch_worker, &ctx);
    else
        asi_fused_matmul_batch_worker(0, 1, &ctx);
    return 0;
}

static inline float dot_f32_f16_avx2(const float *q, const uint16_t *k_fp16, uint32_t hd)
{
#if defined(__AVX2__) && defined(__F16C__)
    if ((hd & 7u) == 0u)
    {
        __m256 vacc = _mm256_setzero_ps();
        for (uint32_t d = 0; d < hd; d += 8)
        {
            __m128i kh = _mm_loadu_si128((const __m128i *)(k_fp16 + d));
            __m256 kf = _mm256_cvtph_ps(kh);
            __m256 qf = _mm256_loadu_ps(q + d);
            vacc = _mm256_fmadd_ps(qf, kf, vacc);
        }
        return lila_hsum_f32_avx2(vacc);
    }
#endif
    float dot = 0.0f;
    for (uint32_t d = 0; d < hd; ++d) dot += q[d] * fp16_to_fp32(k_fp16[d]);
    return dot;
}

static inline void accum_f16_f32_avx2(float *oh, const uint16_t *v_fp16, float prob, uint32_t hd)
{
#if defined(__AVX2__) && defined(__F16C__)
    if ((hd & 7u) == 0u)
    {
        __m256 vp = _mm256_set1_ps(prob);
        for (uint32_t d = 0; d < hd; d += 8)
        {
            __m128i vh = _mm_loadu_si128((const __m128i *)(v_fp16 + d));
            __m256 vf = _mm256_cvtph_ps(vh);
            __m256 ov = _mm256_loadu_ps(oh + d);
            _mm256_storeu_ps(oh + d, _mm256_fmadd_ps(vp, vf, ov));
        }
        return;
    }
#endif
    for (uint32_t d = 0; d < hd; ++d) oh[d] += prob * fp16_to_fp32(v_fp16[d]);
}

static void apply_rope_neox_partial(float *vec, uint32_t rope_dim, uint32_t pos, float theta, const float *inv_freq)
{
    uint32_t half = rope_dim / 2;
    for (uint32_t i = 0; i < half; ++i)
    {
        float freq = inv_freq ? inv_freq[i] : (1.0f / powf(theta, (float)(2 * i) / (float)rope_dim));
        float angle = (float)pos * freq;
        float c = cosf(angle);
        float s = sinf(angle);
        float x0 = vec[i];
        float x1 = vec[i + half];
        vec[i] = x0 * c - x1 * s;
        vec[i + half] = x0 * s + x1 * c;
    }
}

static inline uint32_t asi_effective_rope_pos(int64_t delta, uint32_t kv_pos)
{
    int64_t r = (int64_t)kv_pos + delta;
    return (r > 0) ? (uint32_t)r : 0u;
}

static void apply_rope_neox_3d(float *vec,
                               uint32_t rope_dim,
                               uint32_t pos_t,
                               uint32_t pos_h,
                               uint32_t pos_w,
                               const uint32_t sections[3],
                               uint32_t flags,
                               float theta,
                               const float *inv_freq)
{
    if (pos_t == pos_h && pos_h == pos_w)
    {
        apply_rope_neox_partial(vec, rope_dim, pos_t, theta, inv_freq);
        return;
    }
    uint32_t half = rope_dim / 2u;
    uint32_t sec[3];
    if (sections && (sections[0] + sections[1] + sections[2]) > 0u)
    {
        sec[0] = sections[0];
        sec[1] = sections[1];
        sec[2] = sections[2];
    }
    else if (half == 32u)
    {
        sec[0] = 11u;
        sec[1] = 11u;
        sec[2] = 10u;
    }
    else
    {
        sec[0] = half / 3u;
        sec[1] = half / 3u;
        sec[2] = half - sec[0] - sec[1];
    }
    for (uint32_t i = 0; i < half; ++i)
    {
        uint32_t p;
        if (flags & LILA_ASI_MROPE_INTERLEAVED)
        {
            p = pos_t;
            uint32_t mod3 = i % 3u;
            if (mod3 == 1u && i < sec[1] * 3u) p = pos_h;
            else if (mod3 == 2u && i < sec[2] * 3u) p = pos_w;
        }
        else
        {
            p = (i < sec[0]) ? pos_t : ((i < sec[0] + sec[1]) ? pos_h : pos_w);
        }
        float freq = inv_freq ? inv_freq[i] : (1.0f / powf(theta, (float)(2u * i) / (float)rope_dim));
        float angle = (float)p * freq;
        float c = cosf(angle);
        float s = sinf(angle);
        float x0 = vec[i];
        float x1 = vec[i + half];
        vec[i] = x0 * c - x1 * s;
        vec[i + half] = x0 * s + x1 * c;
    }
}

static const AsiV2TensorEntry *req_tensor(const AsiV2Cartridge *cart,
                                          const char *name,
                                          char *err_buf,
                                          size_t err_cap)
{
    const AsiV2TensorEntry *t = asi_v2_find_tensor(cart, name);
    if (!t)
    {
        if (err_buf && err_cap > 0)
        {
            snprintf(err_buf, err_cap, "missing required tensor '%s'", name);
        }
        return NULL;
    }
    return t;
}

static void qwen35_free_session(AsiArchSession *s)
{
    if (!s) return;
    if (s->spec_stats && s->stat_eval_calls > 0)
    {
        fprintf(stderr,
                "[ASI SPEC STATS] eval_calls=%llu cache_hits=%llu batches=%llu drafted=%llu accepted=%llu misses=%llu\n",
                (unsigned long long)s->stat_eval_calls,
                (unsigned long long)s->stat_cache_hits,
                (unsigned long long)s->stat_batches,
                (unsigned long long)s->stat_drafted,
                (unsigned long long)s->stat_accepted,
                (unsigned long long)s->stat_misses);
    }
    if (s->spec_stats && s->spec_gate && s->stat_eval_calls > 0)
    {
        fprintf(stderr, "[ASI SPEC GATE] gate=1 skips=%llu\n",
                (unsigned long long)s->stat_spec_gate_skips);
    }
    if (s->lmhead_stats && s->lmhead_idx && s->lmhead_ws)
    {
        LilaLhcInfo linfo;
        LilaLhcStats lstats;
        if (lila_lhc_get_info(s->lmhead_idx, &linfo) == 0 &&
            lila_lhc_get_stats(s->lmhead_ws, &lstats) == 0)
        {
            double calls = (double)(lstats.eval_calls + lstats.spec_eval_calls);
            double avg_s1 = (calls > 0.0) ? (lstats.stage1_ms_sum / calls) : 0.0;
            double avg_s2 = (calls > 0.0) ? (lstats.stage2_ms_sum / calls) : 0.0;
            double avg_cands = (calls > 0.0) ? ((double)lstats.total_candidates_eval / calls) : 0.0;
            fprintf(stderr,
                    "[ASI LMHEAD STATS] mode=%s K=%u build_ms=%.2f sketch_mb=%.2f ws_kb=%.1f extra_rss_mb=%.2f "
                    "calls=%llu spec_calls=%llu avg_s1_ms=%.2f avg_s2_ms=%.2f avg_cands=%.1f\n",
                    lila_lhc_mode_name(linfo.mode), s->lmhead_k, linfo.build_ms,
                    (double)linfo.total_sketch_bytes / (1024.0 * 1024.0),
                    (double)lila_lhc_workspace_bytes(s->lmhead_ws) / 1024.0,
                    (double)linfo.extra_rss_bytes / (1024.0 * 1024.0),
                    (unsigned long long)lstats.eval_calls,
                    (unsigned long long)lstats.spec_eval_calls,
                    avg_s1, avg_s2, avg_cands);
        }
    }
    lila_lhc_workspace_free(s->lmhead_ws);
    lila_lhc_free(s->lmhead_idx);
    asi_pool_destroy(s->pool);
    free(s->prefix_conv);
    free(s->prefix_ssm);
    free(s->prefix_k);
    free(s->prefix_v);
    free(s->prefix_tokens);
    free(s->rope_inv_freq);
    free(s->token_history);
    free(s->spec_logits);
    free(s->spec_saved_qkv);
    free(s->spec_saved_a);
    free(s->spec_saved_b);
    free(s->spec_tmp_logits);
    free(s->fail_hash);
    if (s->layers)
    {
        for (uint32_t i = 0; i < s->cart->config.n_layers; ++i)
        {
            if (s->ssm_states)
            {
                free(s->ssm_states[i].conv_state);
                free(s->ssm_states[i].ssm_state);
            }
            if (s->kv_states)
            {
                free(s->kv_states[i].k_cache);
                free(s->kv_states[i].v_cache);
            }
        }
    }
    free(s->ssm_states);
    free(s->kv_states);
    free(s->layers);
    free(s->scratch.x);
    free(s->scratch.x_norm);
    free(s->scratch.qkv);
    free(s->scratch.z);
    free(s->scratch.qg);
    free(s->scratch.k);
    free(s->scratch.v);
    free(s->scratch.attn_out);
    free(s->scratch.proj_out);
    free(s->scratch.ffn_gate);
    free(s->scratch.ffn_up);
    free(s->scratch.attn_scores);
    free(s->scratch.alpha);
    free(s->scratch.beta);
    free(s->scratch.v_pred);
    free(s->scratch.xb);
    free(s->scratch.xnb);
    free(s->scratch.projb);
    free(s->scratch.qkvb);
    free(s->scratch.zb);
    free(s->scratch.ab);
    free(s->scratch.bb);
    free(s->scratch.qgb);
    free(s->scratch.kb);
    free(s->scratch.vb);
    free(s->scratch.aob);
    free(s->scratch.fgb);
    free(s->scratch.fub);
    asi_aligned_free(s->scratch.qx_h);
    free(s->scratch.thread_scores_blk);
    free(s);
}

static void qwen35_reset_state(AsiArchSession *s)
{
    if (!s) return;
    const AsiV2ModelConfig *c = &s->cart->config;
    uint32_t qkv_dim = c->ssm_inner_size * 3;
    uint32_t conv_hist = (c->ssm_conv_kernel > 1) ? (c->ssm_conv_kernel - 1) : 0;
    uint32_t ssm_head_dim = (c->ssm_group_count > 0) ? (c->ssm_inner_size / c->ssm_group_count) : 0;
    uint32_t used_tokens = s->history_len;
    if (s->prompt_len > used_tokens) used_tokens = s->prompt_len;
    if (s->prefix_len > used_tokens) used_tokens = s->prefix_len;
    /* Add margin for any speculative draft tokens written past history_len */
    if (used_tokens > 0)
    {
        used_tokens += 16u;
        if (used_tokens > s->max_ctx) used_tokens = s->max_ctx;
    }
    else
    {
        used_tokens = s->max_ctx;
    }
    size_t kv_elems = (size_t)used_tokens * c->n_kv_heads * c->head_dim;

    for (uint32_t i = 0; i < c->n_layers; ++i)
    {
        if (s->layers[i].kind == QWEN35_LAYER_DELTANET)
        {
            if (s->ssm_states[i].conv_state && conv_hist > 0)
            {
                memset(s->ssm_states[i].conv_state, 0, (size_t)conv_hist * qkv_dim * sizeof(float));
            }
            if (s->ssm_states[i].ssm_state)
            {
                memset(s->ssm_states[i].ssm_state, 0,
                       (size_t)c->ssm_group_count * c->ssm_state_size * ssm_head_dim * sizeof(float));
            }
        }
        else
        {
            if (s->kv_states[i].k_cache)
            {
                memset(s->kv_states[i].k_cache, 0, kv_elems * sizeof(uint16_t));
            }
            if (s->kv_states[i].v_cache)
            {
                memset(s->kv_states[i].v_cache, 0, kv_elems * sizeof(uint16_t));
            }
        }
    }
    s->spec_count = 0;
    s->spec_cursor = 0;
    s->spec_start_pos = 0;
    s->history_len = 0;
    s->history_valid_from = 0;
    s->prompt_len = 0;
    s->kv_dirty_from = 0;
    s->mrope_pos_delta = 0;
    s->spec_consec_misses = 0;
    s->spec_cooldown = 0;
    s->mrope_pos_delta = 0;
    s->fail_gen++;
    if (s->fail_gen == 0)
    {
        if (s->fail_hash) memset(s->fail_hash, 0, ASI_SPEC_FAIL_CAP * sizeof(AsiSpecFailEntry));
        s->fail_gen = 1;
    }
}

static int qwen35_init_session(const AsiV2Cartridge *cart,
                               uint32_t max_ctx,
                               uint32_t n_threads,
                               AsiArchSession **out_session,
                               char *err_buf,
                               size_t err_cap)
{
    if (!cart || !out_session) return -1;
    const AsiV2ModelConfig *c = &cart->config;
    if (c->arch_id != ASI_ARCH_HYBRID_QWEN35 || c->n_layers == 0 || c->full_attention_interval == 0)
    {
        snprintf(err_buf, err_cap, "invalid hybrid_qwen35 config");
        return -1;
    }

    AsiArchSession *s = (AsiArchSession *)calloc(1, sizeof(AsiArchSession));
    if (!s) return -1;
    s->cart = cart;
    s->max_ctx = max_ctx > 0 ? max_ctx : 2048;
    s->n_threads = n_threads > 0 ? n_threads : 1;
    const char *fp32_env = getenv("LILA_ASI_FP32_REF");
    s->fp32_ref = (fp32_env && atoi(fp32_env) != 0);
    const AsiRuntimeConfig *cfg = asi_runtime_config();
    /* High-level plain-English preset: LILA_ASI_MODE=fast|balanced|exact
     *   fast     (default): Sieve 1bit1024 K=2048 + smart speculation ON
     *   balanced          : Sieve 2bit1024 K=1024 + smart speculation ON
     *   exact             : Sieve OFF + speculation OFF (100% exact full dictionary, lowest RAM)
     * Individual LILA_ASI_* engineer flags override the preset when explicitly set. */
    const char *preset_env = getenv("LILA_ASI_MODE");
    int preset_spec = cfg->speculative;
    int preset_spec_gate = 1;
    float preset_sparse_keep = 0.5f;
    LilaLhcMode preset_lhm = (!s->fp32_ref && cart->config.vocab_size > 4096u && cart->config.hidden_size == 1024u)
                                 ? LILA_LHC_MODE_1BIT_1024
                                 : LILA_LHC_MODE_OFF;
    int preset_k = 2048;
    if (preset_env && preset_env[0] != '\0')
    {
        if (strcasecmp(preset_env, "balanced") == 0)
        {
            preset_lhm = (!s->fp32_ref && cart->config.vocab_size > 4096u && cart->config.hidden_size == 1024u)
                             ? LILA_LHC_MODE_2BIT_1024
                             : LILA_LHC_MODE_OFF;
            preset_k = 1024;
            preset_spec = 1;
            preset_spec_gate = 1;
            preset_sparse_keep = 1.0f;
        }
        else if (strcasecmp(preset_env, "exact") == 0)
        {
            preset_lhm = LILA_LHC_MODE_OFF;
            preset_k = 2048;
            preset_spec = 0;
            preset_spec_gate = 0;
            preset_sparse_keep = 1.0f;
        }
    }
    const char *sk_env = getenv("LILA_ASI_FFN_SPARSE_KEEP");
    s->ffn_sparse_keep = (sk_env && sk_env[0] != '\0') ? (float)atof(sk_env) : preset_sparse_keep;
    if (s->ffn_sparse_keep <= 0.0f) s->ffn_sparse_keep = preset_sparse_keep;
    if (s->ffn_sparse_keep > 1.0f) s->ffn_sparse_keep = 1.0f;
    s->speculative = asi_env_int("LILA_ASI_SPECULATIVE", preset_spec, 0, NULL) ? 1 : 0;
    int sdl = asi_env_int("LILA_ASI_SPEC_DRAFT_LEN", cfg->spec_draft_len, 1, NULL);
    if (sdl > (int)ASI_SPEC_MAX_K) sdl = (int)ASI_SPEC_MAX_K;
    s->spec_draft_len = sdl;
    const char *ss = getenv("LILA_ASI_SPEC_STATS");
    s->spec_stats = (ss && ss[0] != '\0') ? (strcmp(ss, "0") != 0) : cfg->spec_stats;
    const char *lhm_env = getenv("LILA_ASI_LMHEAD_MODE");
    if (s->fp32_ref)
        s->lmhead_mode = LILA_LHC_MODE_OFF;
    else if (lhm_env && lhm_env[0] != '\0')
        s->lmhead_mode = lila_lhc_parse_mode(lhm_env);
    else
        s->lmhead_mode = preset_lhm;
    s->lmhead_k = (uint32_t)asi_env_int("LILA_ASI_LMHEAD_K", preset_k, 1, NULL);
    s->lmhead_prefill = asi_env_int("LILA_ASI_LMHEAD_PREFILL", 0, 0, NULL) ? 1 : 0;
    const char *lhs = getenv("LILA_ASI_LMHEAD_STATS");
    s->lmhead_stats = (lhs && lhs[0] != '\0' && strcmp(lhs, "0") != 0) ? 1 : 0;
    s->kv_dirty_from = 0;
    s->spec_gate = asi_env_int("LILA_ASI_SPEC_GATE", preset_spec_gate, 0, NULL) ? 1 : 0;
    s->fail_gen = 1;
    s->pool = asi_pool_create((int)s->n_threads);

    s->token_embd = req_tensor(cart, "token_embd.weight", err_buf, err_cap);
    s->output_norm = req_tensor(cart, "output_norm.weight", err_buf, err_cap);
    if (!s->token_embd || !s->output_norm)
    {
        qwen35_free_session(s);
        return -1;
    }
    s->output_weight = asi_v2_find_tensor(cart, "output.weight");

    s->layers = (Qwen35LayerWeights *)calloc(c->n_layers, sizeof(Qwen35LayerWeights));
    s->ssm_states = (Qwen35DeltaNetState *)calloc(c->n_layers, sizeof(Qwen35DeltaNetState));
    s->kv_states = (Qwen35AttnKVState *)calloc(c->n_layers, sizeof(Qwen35AttnKVState));
    s->token_history = (int32_t *)calloc(s->max_ctx, sizeof(int32_t));
    if (!s->layers || !s->ssm_states || !s->kv_states || !s->token_history)
    {
        qwen35_free_session(s);
        return -1;
    }

    uint32_t qkv_dim = c->ssm_inner_size * 3;
    s->scratch.x = (float *)calloc(c->hidden_size, sizeof(float));
    s->scratch.x_norm = (float *)calloc(c->hidden_size, sizeof(float));
    s->scratch.qkv = (float *)calloc(qkv_dim, sizeof(float));
    s->scratch.z = (float *)calloc(c->ssm_inner_size, sizeof(float));
    s->scratch.qg = (float *)calloc((size_t)c->n_heads * 2 * c->head_dim, sizeof(float));
    s->scratch.k = (float *)calloc((size_t)c->n_kv_heads * c->head_dim, sizeof(float));
    s->scratch.v = (float *)calloc((size_t)c->n_kv_heads * c->head_dim, sizeof(float));
    size_t attn_out_elems = (size_t)c->n_heads * c->head_dim;
    if (c->ssm_inner_size > attn_out_elems) attn_out_elems = c->ssm_inner_size;
    s->scratch.attn_out = (float *)calloc(attn_out_elems, sizeof(float));
    s->scratch.proj_out = (float *)calloc(c->hidden_size, sizeof(float));
    s->scratch.ffn_gate = (float *)calloc(c->intermediate_size, sizeof(float));
    s->scratch.ffn_up = (float *)calloc(c->intermediate_size, sizeof(float));
    s->scratch.attn_scores = (float *)calloc(s->max_ctx, sizeof(float));
    s->scratch.alpha = (float *)calloc(c->ssm_time_step_rank, sizeof(float));
    s->scratch.beta = (float *)calloc(c->ssm_time_step_rank, sizeof(float));
    s->scratch.v_pred = (float *)calloc(c->ssm_state_size, sizeof(float));
    {
        uint32_t n_rep_max = (c->n_kv_heads > 0) ? (c->n_heads / c->n_kv_heads) : 4u;
        if (n_rep_max < 4u) n_rep_max = 4u;
        s->scratch.thread_scores_cap = (size_t)4u * n_rep_max * s->max_ctx;
        s->scratch.thread_scores_blk = (float *)calloc(
            (size_t)s->n_threads * s->scratch.thread_scores_cap, sizeof(float));
    }
    if (c->rope_dim >= 2)
    {
        uint32_t half = c->rope_dim / 2;
        s->rope_inv_freq = (float *)malloc((size_t)half * sizeof(float));
        if (s->rope_inv_freq)
        {
            for (uint32_t i = 0; i < half; ++i)
                s->rope_inv_freq[i] = 1.0f / powf(c->rope_theta, (float)(2 * i) / (float)c->rope_dim);
        }
    }

    if (!s->scratch.x || !s->scratch.x_norm || !s->scratch.qkv || !s->scratch.z ||
        !s->scratch.qg || !s->scratch.k || !s->scratch.v || !s->scratch.attn_out ||
        !s->scratch.proj_out || !s->scratch.ffn_gate || !s->scratch.ffn_up ||
        !s->scratch.attn_scores || !s->scratch.alpha || !s->scratch.beta || !s->scratch.v_pred)
    {
        qwen35_free_session(s);
        return -1;
    }

    char name[64];
    for (uint32_t i = 0; i < c->n_layers; ++i)
    {
        Qwen35LayerWeights *lw = &s->layers[i];
        int is_full = ((i + 1) % c->full_attention_interval == 0);
        lw->kind = is_full ? QWEN35_LAYER_FULL_ATTN : QWEN35_LAYER_DELTANET;
        if (lw->kind == QWEN35_LAYER_DELTANET) s->spec_n_dn++;

        snprintf(name, sizeof(name), "blk.%u.attn_norm.weight", i);
        lw->attn_norm = req_tensor(cart, name, err_buf, err_cap);
        snprintf(name, sizeof(name), "blk.%u.post_attention_norm.weight", i);
        lw->post_attn_norm = req_tensor(cart, name, err_buf, err_cap);
        snprintf(name, sizeof(name), "blk.%u.ffn_gate.weight", i);
        lw->ffn_gate = req_tensor(cart, name, err_buf, err_cap);
        snprintf(name, sizeof(name), "blk.%u.ffn_up.weight", i);
        lw->ffn_up = req_tensor(cart, name, err_buf, err_cap);
        snprintf(name, sizeof(name), "blk.%u.ffn_down.weight", i);
        lw->ffn_down = req_tensor(cart, name, err_buf, err_cap);

        if (!lw->attn_norm || !lw->post_attn_norm || !lw->ffn_gate || !lw->ffn_up || !lw->ffn_down)
        {
            qwen35_free_session(s);
            return -1;
        }
        if (lw->ffn_gate->quant_type == ASI_QUANT_Q4_REPACK || lw->ffn_down->quant_type == ASI_QUANT_Q4_REPACK)
        {
            s->has_q4_repack = 1;
        }

        if (lw->kind == QWEN35_LAYER_DELTANET)
        {
            snprintf(name, sizeof(name), "blk.%u.attn_qkv.weight", i);
            lw->attn_qkv = req_tensor(cart, name, err_buf, err_cap);
            snprintf(name, sizeof(name), "blk.%u.attn_gate.weight", i);
            lw->attn_gate = req_tensor(cart, name, err_buf, err_cap);
            snprintf(name, sizeof(name), "blk.%u.ssm_a", i);
            lw->ssm_a = req_tensor(cart, name, err_buf, err_cap);
            snprintf(name, sizeof(name), "blk.%u.ssm_alpha.weight", i);
            lw->ssm_alpha = req_tensor(cart, name, err_buf, err_cap);
            snprintf(name, sizeof(name), "blk.%u.ssm_beta.weight", i);
            lw->ssm_beta = req_tensor(cart, name, err_buf, err_cap);
            snprintf(name, sizeof(name), "blk.%u.ssm_conv1d.weight", i);
            lw->ssm_conv1d = req_tensor(cart, name, err_buf, err_cap);
            snprintf(name, sizeof(name), "blk.%u.ssm_dt.bias", i);
            lw->ssm_dt_bias = req_tensor(cart, name, err_buf, err_cap);
            snprintf(name, sizeof(name), "blk.%u.ssm_norm.weight", i);
            lw->ssm_norm = req_tensor(cart, name, err_buf, err_cap);
            snprintf(name, sizeof(name), "blk.%u.ssm_out.weight", i);
            lw->ssm_out = req_tensor(cart, name, err_buf, err_cap);

            if (!lw->attn_qkv || !lw->attn_gate || !lw->ssm_a || !lw->ssm_alpha ||
                !lw->ssm_beta || !lw->ssm_conv1d || !lw->ssm_dt_bias || !lw->ssm_norm || !lw->ssm_out)
            {
                qwen35_free_session(s);
                return -1;
            }

            uint32_t conv_hist = (c->ssm_conv_kernel > 1) ? (c->ssm_conv_kernel - 1) : 0;
            uint32_t ssm_head_dim = (c->ssm_group_count > 0) ? (c->ssm_inner_size / c->ssm_group_count) : 0;
            if (conv_hist > 0)
            {
                s->ssm_states[i].conv_state = (float *)calloc((size_t)conv_hist * qkv_dim, sizeof(float));
            }
            s->ssm_states[i].ssm_state = (float *)calloc(
                (size_t)c->ssm_group_count * c->ssm_state_size * ssm_head_dim, sizeof(float));
            if ((conv_hist > 0 && !s->ssm_states[i].conv_state) || !s->ssm_states[i].ssm_state)
            {
                qwen35_free_session(s);
                return -1;
            }
        }
        else
        {
            snprintf(name, sizeof(name), "blk.%u.attn_q.weight", i);
            lw->attn_q = req_tensor(cart, name, err_buf, err_cap);
            snprintf(name, sizeof(name), "blk.%u.attn_k.weight", i);
            lw->attn_k = req_tensor(cart, name, err_buf, err_cap);
            snprintf(name, sizeof(name), "blk.%u.attn_v.weight", i);
            lw->attn_v = req_tensor(cart, name, err_buf, err_cap);
            snprintf(name, sizeof(name), "blk.%u.attn_q_norm.weight", i);
            lw->attn_q_norm = req_tensor(cart, name, err_buf, err_cap);
            snprintf(name, sizeof(name), "blk.%u.attn_k_norm.weight", i);
            lw->attn_k_norm = req_tensor(cart, name, err_buf, err_cap);
            snprintf(name, sizeof(name), "blk.%u.attn_output.weight", i);
            lw->attn_output = req_tensor(cart, name, err_buf, err_cap);

            if (!lw->attn_q || !lw->attn_k || !lw->attn_v ||
                !lw->attn_q_norm || !lw->attn_k_norm || !lw->attn_output)
            {
                qwen35_free_session(s);
                return -1;
            }

            size_t kv_elems = (size_t)s->max_ctx * c->n_kv_heads * c->head_dim;
            s->kv_states[i].k_cache = (uint16_t *)calloc(kv_elems, sizeof(uint16_t));
            s->kv_states[i].v_cache = (uint16_t *)calloc(kv_elems, sizeof(uint16_t));
            if (!s->kv_states[i].k_cache || !s->kv_states[i].v_cache)
            {
                qwen35_free_session(s);
                return -1;
            }
        }
    }

    if (s->lmhead_mode != LILA_LHC_MODE_OFF)
    {
        const AsiV2TensorEntry *lm_head = s->output_weight ? s->output_weight : s->token_embd;
        if (lm_head && lila_quant_is_supported(lm_head->quant_type) &&
            c->hidden_size == 1024u && c->vocab_size > 0u)
        {
            int lm_repacked = (lm_head->layout_flags & ASI_TENSOR_LAYOUT_AVX2_REPACK) ? 1 : 0;
            if (s->lmhead_k > c->vocab_size) s->lmhead_k = c->vocab_size;
            const AsiV2SectionEntry *sv = asi_v2_find_section(cart, ASI_SEC_SIEVE_SKETCH);
            if (sv && cart->base && sv->offset <= (uint64_t)cart->file_size &&
                sv->size <= (uint64_t)cart->file_size - sv->offset)
            {
                s->lmhead_idx = lila_lhc_from_serialized(
                    s->lmhead_mode, cart->base + sv->offset, (size_t)sv->size,
                    lm_head->quant_type, lm_repacked, c->vocab_size, c->hidden_size);
            }
            if (!s->lmhead_idx)
            {
                asi_pool_begin(s->pool);
                s->lmhead_idx = lila_lhc_build_quant_pool(
                    s->lmhead_mode, tensor_data(cart, lm_head), lm_head->quant_type, lm_repacked,
                    c->vocab_size, c->hidden_size,
                    (const float *)tensor_data(cart, s->output_norm), lhc_asi_dispatch, s->pool);
                asi_pool_end(s->pool);
            }
            s->lmhead_ws = lila_lhc_workspace_create(
                c->vocab_size, s->lmhead_k, s->n_threads, 1u + ASI_SPEC_MAX_K);
            if (!s->lmhead_idx || !s->lmhead_ws)
            {
                snprintf(err_buf, err_cap, "failed to build lm_head_candidate index");
                qwen35_free_session(s);
                return -1;
            }
            if (s->lmhead_stats || asi_runtime_config()->profile)
            {
                LilaLhcInfo linfo;
                if (lila_lhc_get_info(s->lmhead_idx, &linfo) == 0)
                {
                    fprintf(stderr,
                            "[ASI LMHEAD INIT] mode=%s K=%u build_ms=%.2f sketch_mb=%.2f ws_kb=%.1f extra_rss_mb=%.2f\n",
                            lila_lhc_mode_name(linfo.mode), s->lmhead_k, linfo.build_ms,
                            (double)linfo.total_sketch_bytes / (1024.0 * 1024.0),
                            (double)lila_lhc_workspace_bytes(s->lmhead_ws) / 1024.0,
                            (double)linfo.extra_rss_bytes / (1024.0 * 1024.0));
                }
            }
        }
    }

    *out_session = s;
    return 0;
}

static inline void deltanet_head_step_avx2(float *S,
                                           float *qh,
                                           float *kh,
                                           const float *vh,
                                           const float *zh,
                                           float *oh,
                                           const float *ssm_norm_w,
                                           float alpha_val,
                                           float dt_bias_val,
                                           float ssm_a_val,
                                           float beta_raw_val,
                                           float q_scale,
                                           uint32_t k_dim,
                                           uint32_t v_dim,
                                           float rms_eps)
{
    /* L2-normalize Q and K */
    float q_ss = 0.0f, k_ss = 0.0f;
#if defined(__AVX512F__)
    if ((k_dim & 15u) == 0u)
    {
        __m512 vq = _mm512_setzero_ps();
        __m512 vk = _mm512_setzero_ps();
        for (uint32_t i = 0; i < k_dim; i += 16)
        {
            __m512 qv = _mm512_loadu_ps(qh + i);
            __m512 kv = _mm512_loadu_ps(kh + i);
            vq = _mm512_fmadd_ps(qv, qv, vq);
            vk = _mm512_fmadd_ps(kv, kv, vk);
        }
        q_ss = _mm512_reduce_add_ps(vq);
        k_ss = _mm512_reduce_add_ps(vk);
        float q_inv = q_scale / sqrtf(q_ss + 1e-6f);
        float k_inv = 1.0f / sqrtf(k_ss + 1e-6f);
        __m512 vqi = _mm512_set1_ps(q_inv);
        __m512 vki = _mm512_set1_ps(k_inv);
        for (uint32_t i = 0; i < k_dim; i += 16)
        {
            _mm512_storeu_ps(qh + i, _mm512_mul_ps(_mm512_loadu_ps(qh + i), vqi));
            _mm512_storeu_ps(kh + i, _mm512_mul_ps(_mm512_loadu_ps(kh + i), vki));
        }
    }
    else
#endif
#if defined(__AVX2__)
    if ((k_dim & 7u) == 0u)
    {
        __m256 vq = _mm256_setzero_ps();
        __m256 vk = _mm256_setzero_ps();
        for (uint32_t i = 0; i < k_dim; i += 8)
        {
            __m256 qv = _mm256_loadu_ps(qh + i);
            __m256 kv = _mm256_loadu_ps(kh + i);
            vq = _mm256_fmadd_ps(qv, qv, vq);
            vk = _mm256_fmadd_ps(kv, kv, vk);
        }
        q_ss = lila_hsum_f32_avx2(vq);
        k_ss = lila_hsum_f32_avx2(vk);
        float q_inv = q_scale / sqrtf(q_ss + 1e-6f);
        float k_inv = 1.0f / sqrtf(k_ss + 1e-6f);
        __m256 vqi = _mm256_set1_ps(q_inv);
        __m256 vki = _mm256_set1_ps(k_inv);
        for (uint32_t i = 0; i < k_dim; i += 8)
        {
            _mm256_storeu_ps(qh + i, _mm256_mul_ps(_mm256_loadu_ps(qh + i), vqi));
            _mm256_storeu_ps(kh + i, _mm256_mul_ps(_mm256_loadu_ps(kh + i), vki));
        }
    }
    else
#endif
#if defined(__aarch64__)
    if ((k_dim & 3u) == 0u)
    {
        float32x4_t vq = vdupq_n_f32(0.0f);
        float32x4_t vk = vdupq_n_f32(0.0f);
        for (uint32_t i = 0; i < k_dim; i += 4)
        {
            float32x4_t qv = vld1q_f32(qh + i);
            float32x4_t kv = vld1q_f32(kh + i);
            vq = vfmaq_f32(vq, qv, qv);
            vk = vfmaq_f32(vk, kv, kv);
        }
        q_ss = vaddvq_f32(vq);
        k_ss = vaddvq_f32(vk);
        float q_inv = q_scale / sqrtf(q_ss + 1e-6f);
        float k_inv = 1.0f / sqrtf(k_ss + 1e-6f);
        float32x4_t vqi = vdupq_n_f32(q_inv);
        float32x4_t vki = vdupq_n_f32(k_inv);
        for (uint32_t i = 0; i < k_dim; i += 4)
        {
            vst1q_f32(qh + i, vmulq_f32(vld1q_f32(qh + i), vqi));
            vst1q_f32(kh + i, vmulq_f32(vld1q_f32(kh + i), vki));
        }
    }
    else
#endif
    {
        for (uint32_t i = 0; i < k_dim; ++i) { q_ss += qh[i] * qh[i]; k_ss += kh[i] * kh[i]; }
        float q_inv = q_scale / sqrtf(q_ss + 1e-6f);
        float k_inv = 1.0f / sqrtf(k_ss + 1e-6f);
        for (uint32_t i = 0; i < k_dim; ++i) { qh[i] *= q_inv; kh[i] *= k_inv; }
    }

    float dt = fast_softplus(alpha_val + dt_bias_val);
    float g = expf(ssm_a_val * dt);
    float beta = fast_sigmoid(beta_raw_val);

#if defined(__AVX512F__)
    if ((k_dim & 15u) == 0u)
    {
        __m512 vg = _mm512_set1_ps(g);
        for (uint32_t j = 0; j < v_dim; ++j)
        {
            float *Sj = S + (size_t)j * k_dim;
            __m512 vpred = _mm512_setzero_ps();
            for (uint32_t i = 0; i < k_dim; i += 16)
            {
                __m512 sd = _mm512_mul_ps(_mm512_loadu_ps(Sj + i), vg);
                _mm512_storeu_ps(Sj + i, sd);
                vpred = _mm512_fmadd_ps(sd, _mm512_loadu_ps(kh + i), vpred);
            }
            float pred = _mm512_reduce_add_ps(vpred);
            float delta = beta * (vh[j] - pred);
            __m512 vdelta = _mm512_set1_ps(delta);
            __m512 vov = _mm512_setzero_ps();
            for (uint32_t i = 0; i < k_dim; i += 16)
            {
                __m512 su = _mm512_fmadd_ps(vdelta, _mm512_loadu_ps(kh + i), _mm512_loadu_ps(Sj + i));
                _mm512_storeu_ps(Sj + i, su);
                vov = _mm512_fmadd_ps(su, _mm512_loadu_ps(qh + i), vov);
            }
            oh[j] = _mm512_reduce_add_ps(vov);
        }
    }
    else
#endif
#if defined(__AVX2__)
    if ((k_dim & 7u) == 0u)
    {
        __m256 vg = _mm256_set1_ps(g);
        for (uint32_t j = 0; j < v_dim; ++j)
        {
            float *Sj = S + (size_t)j * k_dim;
            __m256 vpred = _mm256_setzero_ps();
            for (uint32_t i = 0; i < k_dim; i += 8)
            {
                __m256 sd = _mm256_mul_ps(_mm256_loadu_ps(Sj + i), vg);
                _mm256_storeu_ps(Sj + i, sd);
                vpred = _mm256_fmadd_ps(sd, _mm256_loadu_ps(kh + i), vpred);
            }
            float pred = lila_hsum_f32_avx2(vpred);
            float delta = beta * (vh[j] - pred);
            __m256 vdelta = _mm256_set1_ps(delta);
            __m256 vov = _mm256_setzero_ps();
            for (uint32_t i = 0; i < k_dim; i += 8)
            {
                __m256 su = _mm256_fmadd_ps(vdelta, _mm256_loadu_ps(kh + i), _mm256_loadu_ps(Sj + i));
                _mm256_storeu_ps(Sj + i, su);
                vov = _mm256_fmadd_ps(su, _mm256_loadu_ps(qh + i), vov);
            }
            oh[j] = lila_hsum_f32_avx2(vov);
        }
    }
    else
#endif
#if defined(__aarch64__)
    if ((k_dim & 3u) == 0u)
    {
        float32x4_t vg = vdupq_n_f32(g);
        for (uint32_t j = 0; j < v_dim; ++j)
        {
            float *Sj = S + (size_t)j * k_dim;
            float32x4_t vpred = vdupq_n_f32(0.0f);
            for (uint32_t i = 0; i < k_dim; i += 4)
            {
                float32x4_t sd = vmulq_f32(vld1q_f32(Sj + i), vg);
                vst1q_f32(Sj + i, sd);
                vpred = vfmaq_f32(vpred, sd, vld1q_f32(kh + i));
            }
            float pred = vaddvq_f32(vpred);
            float delta = beta * (vh[j] - pred);
            float32x4_t vdelta = vdupq_n_f32(delta);
            float32x4_t vov = vdupq_n_f32(0.0f);
            for (uint32_t i = 0; i < k_dim; i += 4)
            {
                float32x4_t su = vfmaq_f32(vld1q_f32(Sj + i), vdelta, vld1q_f32(kh + i));
                vst1q_f32(Sj + i, su);
                vov = vfmaq_f32(vov, su, vld1q_f32(qh + i));
            }
            oh[j] = vaddvq_f32(vov);
        }
    }
    else
#endif
    {
        for (uint32_t j = 0; j < v_dim; ++j)
        {
            float *Sj = S + (size_t)j * k_dim;
            float pred = 0.0f;
            for (uint32_t i = 0; i < k_dim; ++i) { float d = Sj[i] * g; Sj[i] = d; pred += d * kh[i]; }
            float delta = beta * (vh[j] - pred);
            float ov = 0.0f;
            for (uint32_t i = 0; i < k_dim; ++i) { float u = Sj[i] + delta * kh[i]; Sj[i] = u; ov += u * qh[i]; }
            oh[j] = ov;
        }
    }
    rmsnorm(oh, oh, ssm_norm_w, v_dim, rms_eps);
    fast_silu_mul_out_vec(oh, zh, v_dim);
}

typedef struct {
    float *qkvb;
    float *conv_state;
    const float *conv_w;
    uint32_t B;
    uint32_t qkv_dim;
    uint32_t k_conv;
    uint32_t conv_hist;
    int spec_keep_state;
} AsiConv1dCtx;

static void asi_conv1d_worker(int tid, int n_threads, void *vctx)
{
    const AsiConv1dCtx *ctx = (const AsiConv1dCtx *)vctx;
    uint32_t g_total = (ctx->qkv_dim + 7u) / 8u;
    uint32_t ch_start = ((g_total * (uint32_t)tid) / (uint32_t)n_threads) * 8u;
    uint32_t ch_end = ((g_total * (uint32_t)(tid + 1)) / (uint32_t)n_threads) * 8u;
    if (ch_start > ctx->qkv_dim) ch_start = ctx->qkv_dim;
    if (ch_end > ctx->qkv_dim) ch_end = ctx->qkv_dim;
    uint32_t k_conv = ctx->k_conv;
    uint32_t conv_hist = ctx->conv_hist;
    uint32_t qkv_dim = ctx->qkv_dim;
    uint32_t B = ctx->B;

    if (conv_hist == 3 && k_conv == 4 && B >= 1)
    {
        float *cs0 = ctx->conv_state;
        float *cs1 = ctx->conv_state + qkv_dim;
        float *cs2 = ctx->conv_state + 2 * qkv_dim;
#if defined(__AVX2__)
        const __m256 one = _mm256_set1_ps(1.0f);
        const __m256 neg = _mm256_set1_ps(-0.0f);
        if (ctx->spec_keep_state)
        {
            uint32_t ch = ch_start;
            for (; ch + 7 < ch_end; ch += 8)
            {
                const float *w = ctx->conv_w + (size_t)ch * 4;
                __m128 r0 = _mm_loadu_ps(w + 0),  r1 = _mm_loadu_ps(w + 4);
                __m128 r2 = _mm_loadu_ps(w + 8),  r3 = _mm_loadu_ps(w + 12);
                _MM_TRANSPOSE4_PS(r0, r1, r2, r3);
                __m128 r4 = _mm_loadu_ps(w + 16), r5 = _mm_loadu_ps(w + 20);
                __m128 r6 = _mm_loadu_ps(w + 24), r7 = _mm_loadu_ps(w + 28);
                _MM_TRANSPOSE4_PS(r4, r5, r6, r7);
                __m256 vw0 = _mm256_set_m128(r4, r0);
                __m256 vw1 = _mm256_set_m128(r5, r1);
                __m256 vw2 = _mm256_set_m128(r6, r2);
                __m256 vw3 = _mm256_set_m128(r7, r3);

                __m256 vs0 = _mm256_loadu_ps(cs0 + ch);
                __m256 vs1 = _mm256_loadu_ps(cs1 + ch);
                __m256 vs2 = _mm256_loadu_ps(cs2 + ch);
                for (uint32_t t = 0; t < B; ++t)
                {
                    float *qkv = ctx->qkvb + (size_t)t * qkv_dim;
                    __m256 vx = _mm256_loadu_ps(qkv + ch);
                    __m256 vsum = _mm256_add_ps(_mm256_mul_ps(vw3, vx), _mm256_mul_ps(vw0, vs0));
                    vsum = _mm256_add_ps(vsum, _mm256_mul_ps(vw1, vs1));
                    vsum = _mm256_add_ps(vsum, _mm256_mul_ps(vw2, vs2));
                    __m256 ve = lila_expf_avx2(_mm256_xor_ps(vsum, neg));
                    _mm256_storeu_ps(qkv + ch, _mm256_div_ps(vsum, _mm256_add_ps(one, ve)));
                    vs0 = vs1;
                    vs1 = vs2;
                    vs2 = vx;
                    if (t == 0)
                    {
                        _mm256_storeu_ps(cs0 + ch, vs0);
                        _mm256_storeu_ps(cs1 + ch, vs1);
                        _mm256_storeu_ps(cs2 + ch, vs2);
                    }
                }
            }
            for (; ch < ch_end; ++ch)
            {
                const float *w = ctx->conv_w + (size_t)ch * 4;
                float s0 = cs0[ch], s1 = cs1[ch], s2 = cs2[ch];
                for (uint32_t t = 0; t < B; ++t)
                {
                    float *qkv = ctx->qkvb + (size_t)t * qkv_dim;
                    float x = qkv[ch];
                    qkv[ch] = fast_silu(w[3] * x + w[0] * s0 + w[1] * s1 + w[2] * s2);
                    s0 = s1; s1 = s2; s2 = x;
                    if (t == 0)
                    {
                        cs0[ch] = s0; cs1[ch] = s1; cs2[ch] = s2;
                    }
                }
            }
            return;
        }
        for (uint32_t t = 0; t < B; ++t)
        {
            float *qkv = ctx->qkvb + (size_t)t * qkv_dim;
            uint32_t ch = ch_start;
            for (; ch + 7 < ch_end; ch += 8)
            {
                const float *w = ctx->conv_w + (size_t)ch * 4;
                __m128 r0 = _mm_loadu_ps(w + 0),  r1 = _mm_loadu_ps(w + 4);
                __m128 r2 = _mm_loadu_ps(w + 8),  r3 = _mm_loadu_ps(w + 12);
                _MM_TRANSPOSE4_PS(r0, r1, r2, r3);
                __m128 r4 = _mm_loadu_ps(w + 16), r5 = _mm_loadu_ps(w + 20);
                __m128 r6 = _mm_loadu_ps(w + 24), r7 = _mm_loadu_ps(w + 28);
                _MM_TRANSPOSE4_PS(r4, r5, r6, r7);
                __m256 vw0 = _mm256_set_m128(r4, r0);
                __m256 vw1 = _mm256_set_m128(r5, r1);
                __m256 vw2 = _mm256_set_m128(r6, r2);
                __m256 vw3 = _mm256_set_m128(r7, r3);

                __m256 vx  = _mm256_loadu_ps(qkv + ch);
                __m256 vs0 = _mm256_loadu_ps(cs0 + ch);
                __m256 vs1 = _mm256_loadu_ps(cs1 + ch);
                __m256 vs2 = _mm256_loadu_ps(cs2 + ch);
                _mm256_storeu_ps(cs0 + ch, vs1);
                _mm256_storeu_ps(cs1 + ch, vs2);
                _mm256_storeu_ps(cs2 + ch, vx);

                __m256 vsum = _mm256_add_ps(_mm256_mul_ps(vw3, vx), _mm256_mul_ps(vw0, vs0));
                vsum = _mm256_add_ps(vsum, _mm256_mul_ps(vw1, vs1));
                vsum = _mm256_add_ps(vsum, _mm256_mul_ps(vw2, vs2));
                __m256 ve = lila_expf_avx2(_mm256_xor_ps(vsum, neg));
                _mm256_storeu_ps(qkv + ch, _mm256_div_ps(vsum, _mm256_add_ps(one, ve)));
            }
            for (; ch < ch_end; ++ch)
            {
                const float *w = ctx->conv_w + (size_t)ch * 4;
                float x = qkv[ch];
                float s0 = cs0[ch], s1 = cs1[ch], s2 = cs2[ch];
                cs0[ch] = s1; cs1[ch] = s2; cs2[ch] = x;
                qkv[ch] = fast_silu(w[3] * x + w[0] * s0 + w[1] * s1 + w[2] * s2);
            }
        }
#else
        if (ctx->spec_keep_state)
        {
            for (uint32_t ch = ch_start; ch < ch_end; ++ch)
            {
                const float *w = ctx->conv_w + (size_t)ch * 4;
                float s0 = cs0[ch], s1 = cs1[ch], s2 = cs2[ch];
                for (uint32_t t = 0; t < B; ++t)
                {
                    float *qp = ctx->qkvb + (size_t)t * qkv_dim + ch;
                    float x = *qp;
                    *qp = w[3] * x + w[0] * s0 + w[1] * s1 + w[2] * s2;
                    s0 = s1; s1 = s2; s2 = x;
                    if (t == 0)
                    {
                        cs0[ch] = s0; cs1[ch] = s1; cs2[ch] = s2;
                    }
                }
            }
            for (uint32_t t = 0; t < B; ++t)
            {
                fast_silu_inplace_vec(ctx->qkvb + (size_t)t * qkv_dim + ch_start, ch_end - ch_start);
            }
            return;
        }
        for (uint32_t t_tile = 0; t_tile < B; t_tile += 32)
        {
            uint32_t t_end = t_tile + 32;
            if (t_end > B) t_end = B;
            for (uint32_t ch = ch_start; ch < ch_end; ++ch)
            {
                const float *w = ctx->conv_w + (size_t)ch * 4;
                float w0 = w[0], w1 = w[1], w2 = w[2], w3 = w[3];
                float s0 = cs0[ch], s1 = cs1[ch], s2 = cs2[ch];
                float *qp = ctx->qkvb + (size_t)t_tile * qkv_dim + ch;
                for (uint32_t t = t_tile; t < t_end; ++t, qp += qkv_dim)
                {
                    float x = *qp;
                    float sum = w3 * x + w0 * s0 + w1 * s1 + w2 * s2;
                    s0 = s1;
                    s1 = s2;
                    s2 = x;
                    *qp = sum;
                }
                cs0[ch] = s0;
                cs1[ch] = s1;
                cs2[ch] = s2;
            }
            for (uint32_t t = t_tile; t < t_end; ++t)
            {
                float *qkv = ctx->qkvb + (size_t)t * qkv_dim;
                fast_silu_inplace_vec(qkv + ch_start, ch_end - ch_start);
            }
        }
#endif
        return;
    }

    if (ctx->spec_keep_state && conv_hist <= 32u)
    {
        float cs_tmp[32];
        for (uint32_t ch = ch_start; ch < ch_end; ++ch)
        {
            const float *w = ctx->conv_w + (size_t)ch * k_conv;
            for (uint32_t k = 0; k < conv_hist; ++k)
                cs_tmp[k] = ctx->conv_state[(size_t)k * qkv_dim + ch];
            for (uint32_t t = 0; t < B; ++t)
            {
                float *qkv = ctx->qkvb + (size_t)t * qkv_dim;
                float x = qkv[ch];
                float sum = w[k_conv - 1] * x;
                for (uint32_t k = 0; k < conv_hist; ++k) sum += w[k] * cs_tmp[k];
                for (uint32_t k = 0; k + 1 < conv_hist; ++k) cs_tmp[k] = cs_tmp[k + 1];
                if (conv_hist > 0) cs_tmp[conv_hist - 1] = x;
                if (t == 0)
                {
                    for (uint32_t k = 0; k < conv_hist; ++k)
                        ctx->conv_state[(size_t)k * qkv_dim + ch] = cs_tmp[k];
                }
                qkv[ch] = sum;
            }
        }
        for (uint32_t t = 0; t < B; ++t)
            fast_silu_inplace_vec(ctx->qkvb + (size_t)t * qkv_dim + ch_start, ch_end - ch_start);
        return;
    }

    for (uint32_t t = 0; t < B; ++t)
    {
        float *qkv = ctx->qkvb + (size_t)t * qkv_dim;
        for (uint32_t ch = ch_start; ch < ch_end; ++ch)
        {
            const float *w = ctx->conv_w + (size_t)ch * k_conv;
            float sum = w[k_conv - 1] * qkv[ch];
            for (uint32_t k = 0; k < conv_hist; ++k)
                sum += w[k] * ctx->conv_state[(size_t)k * qkv_dim + ch];
            for (uint32_t k = 0; k + 1 < conv_hist; ++k)
                ctx->conv_state[(size_t)k * qkv_dim + ch] = ctx->conv_state[(size_t)(k + 1) * qkv_dim + ch];
            if (conv_hist > 0)
                ctx->conv_state[(size_t)(conv_hist - 1) * qkv_dim + ch] = qkv[ch];
            qkv[ch] = sum;
        }
        fast_silu_inplace_vec(qkv + ch_start, ch_end - ch_start);
    }
}

typedef struct {
    float *ssm_state;
    float *qkvb;
    float *zb;
    float *ab;
    float *bb;
    float *aob;
    const float *ssm_a;
    const float *dt_bias;
    const float *ssm_norm_w;
    uint32_t B;
    uint32_t n_heads;
    uint32_t k_dim;
    uint32_t v_dim;
    uint32_t inner;
    uint32_t qkv_dim;
    uint32_t ssm_r;
    float q_scale;
    float rms_eps;
    int spec_keep_state;
    int state_only;
    _Alignas(64) atomic_int next_h;
} AsiDeltaNetCtx;

static inline int asi_chunkwy_enabled(void)
{
    const char *e = getenv("LILA_ASI_CHUNKWY");
    return (!e || e[0] != '0') ? 1 : 0;
}

static void asi_sgemm_rm(int M, int N, int K,
                         const float *A, int lda,
                         const float *B, int ldb,
                         float beta,
                         float *C, int ldc)
{
    if (beta != 0.0f && beta != 1.0f)
    {
        for (int i = 0; i < M; ++i)
        {
            float *ci = C + (size_t)i * ldc;
            for (int j = 0; j < N; ++j) ci[j] *= beta;
        }
    }
    const float k_beta = (beta == 0.0f) ? 0.0f : 1.0f;
#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
    _Alignas(32) float Bp[128 * 16];
    for (int j0 = 0; j0 < N; j0 += 16)
    {
        int n_cnt = N - j0;
        if (n_cnt > 16) n_cnt = 16;
        if (n_cnt == 16)
        {
            for (int k = 0; k < K; ++k)
                memcpy(Bp + (size_t)k * 16, B + (size_t)k * ldb + j0, 16 * sizeof(float));
        }
        else
        {
            for (int k = 0; k < K; ++k)
            {
                float *dst = Bp + (size_t)k * 16;
                memcpy(dst, B + (size_t)k * ldb + j0, (size_t)n_cnt * sizeof(float));
                for (int r = n_cnt; r < 16; ++r) dst[r] = 0.0f;
            }
        }
        int i = 0;
        for (; i + 5 < M; i += 6)
        {
            asi_kernel_6x16_f32_avx2(K,
                                     A + (size_t)(i + 0) * lda, A + (size_t)(i + 1) * lda,
                                     A + (size_t)(i + 2) * lda, A + (size_t)(i + 3) * lda,
                                     A + (size_t)(i + 4) * lda, A + (size_t)(i + 5) * lda,
                                     Bp,
                                     C + (size_t)(i + 0) * ldc + j0, C + (size_t)(i + 1) * ldc + j0,
                                     C + (size_t)(i + 2) * ldc + j0, C + (size_t)(i + 3) * ldc + j0,
                                     C + (size_t)(i + 4) * ldc + j0, C + (size_t)(i + 5) * ldc + j0,
                                     k_beta, n_cnt);
        }
        for (; i < M; ++i)
        {
            float tmp[16];
            memset(tmp, 0, sizeof(tmp));
            if (k_beta != 0.0f) memcpy(tmp, C + (size_t)i * ldc + j0, (size_t)n_cnt * sizeof(float));
            __m256 c0 = _mm256_loadu_ps(tmp);
            __m256 c1 = _mm256_loadu_ps(tmp + 8);
            const float *ai = A + (size_t)i * lda;
            for (int k = 0; k < K; ++k)
            {
                __m256 va = _mm256_broadcast_ss(ai + k);
                c0 = _mm256_fmadd_ps(va, _mm256_load_ps(Bp + (size_t)k * 16), c0);
                c1 = _mm256_fmadd_ps(va, _mm256_load_ps(Bp + (size_t)k * 16 + 8), c1);
            }
            _mm256_storeu_ps(tmp, c0);
            _mm256_storeu_ps(tmp + 8, c1);
            memcpy(C + (size_t)i * ldc + j0, tmp, (size_t)n_cnt * sizeof(float));
        }
    }
#else
    for (int i = 0; i < M; ++i)
    {
        const float *ai = A + (size_t)i * lda;
        float *ci = C + (size_t)i * ldc;
        if (k_beta == 0.0f) memset(ci, 0, (size_t)N * sizeof(float));
        for (int k = 0; k < K; ++k)
        {
            float a = ai[k];
            const float *bk = B + (size_t)k * ldb;
            for (int j = 0; j < N; ++j) ci[j] += a * bk[j];
        }
    }
#endif
}

typedef struct {
    float S_T[128 * 128];
    float KT[128 * 32];
    float Gkk[32 * 32];
    float Gqk[32 * 32];
    float P[32 * 128];
    float O_inter[32 * 128];
    float RHS[32 * 128];
    float U[32 * 128];
    float Linv[32 * 32];
    float A_mat[32 * 32];
} AsiDeltaNetWyBuf;

static int asi_deltanet_wy_chunk(AsiDeltaNetWyBuf *wb,
                                 const float *Q_tile,
                                 const float *K_tile,
                                 const float *V_tile,
                                 const float *log_g_arr,
                                 const float *beta_arr,
                                 float *O_tile,
                                 uint32_t nt,
                                 uint32_t k_dim,
                                 uint32_t v_dim)
{
    int C = (int)nt;
    int Kd = (int)k_dim;
    int Vd = (int)v_dim;
    float L[32], Gamma[32];
    float acc_log = 0.0f;
    for (int i = 0; i < C; ++i)
    {
        acc_log += log_g_arr[i];
        if (!isfinite(acc_log)) return -1;
        L[i] = acc_log;
        Gamma[i] = expf(acc_log);
    }
    float L_end = L[C - 1];
    float Gamma_end = Gamma[C - 1];

    for (int j = 0; j < C; ++j)
    {
        const float *kj = K_tile + (size_t)j * Kd;
        for (int k = 0; k < Kd; ++k) wb->KT[(size_t)k * C + j] = kj[k];
    }

    asi_sgemm_rm(C, C, Kd, K_tile, Kd, wb->KT, C, 0.0f, wb->Gkk, C);
    asi_sgemm_rm(C, C, Kd, Q_tile, Kd, wb->KT, C, 0.0f, wb->Gqk, C);
    asi_sgemm_rm(C, Vd, Kd, K_tile, Kd, wb->S_T, Vd, 0.0f, wb->P, Vd);
    asi_sgemm_rm(C, Vd, Kd, Q_tile, Kd, wb->S_T, Vd, 0.0f, wb->O_inter, Vd);

    for (int i = 0; i < C; ++i)
    {
        float bi = beta_arr[i];
        float gi = Gamma[i];
        const float *vi = V_tile + (size_t)i * Vd;
        const float *pi = wb->P + (size_t)i * Vd;
        float *ri = wb->RHS + (size_t)i * Vd;
        for (int d = 0; d < Vd; ++d) ri[d] = bi * (vi[d] - gi * pi[d]);
    }

    memset(wb->Linv, 0, (size_t)C * C * sizeof(float));
    for (int i = 0; i < C; ++i)
    {
        wb->Linv[i * C + i] = 1.0f;
        float bi = beta_arr[i];
        float li = L[i];
        for (int k = 0; k < i; ++k)
        {
            float mik = bi * expf(li - L[k]) * wb->Gkk[i * C + k];
            for (int j = 0; j <= k; ++j)
                wb->Linv[i * C + j] -= mik * wb->Linv[k * C + j];
        }
    }
    asi_sgemm_rm(C, Vd, C, wb->Linv, C, wb->RHS, Vd, 0.0f, wb->U, Vd);

    for (int i = 0; i < C; ++i)
    {
        float gi = Gamma[i];
        float li = L[i];
        const float *oi = wb->O_inter + (size_t)i * Vd;
        float *oo = O_tile + (size_t)i * Vd;
        for (int d = 0; d < Vd; ++d) oo[d] = gi * oi[d];
        for (int j = 0; j < C; ++j)
            wb->A_mat[i * C + j] = (j <= i) ? (expf(li - L[j]) * wb->Gqk[i * C + j]) : 0.0f;
    }
    asi_sgemm_rm(C, Vd, C, wb->A_mat, C, wb->U, Vd, 1.0f, O_tile, Vd);

    for (int j = 0; j < C; ++j)
    {
        float sc = expf(L_end - L[j]);
        for (int k = 0; k < Kd; ++k) wb->KT[(size_t)k * C + j] *= sc;
    }
    asi_sgemm_rm(Kd, Vd, C, wb->KT, C, wb->U, Vd, Gamma_end, wb->S_T, Vd);
    return 0;
}

static void asi_deltanet_worker(int tid, int n_threads, void *vctx)
{
    (void)tid;
    AsiDeltaNetCtx *ctx = (AsiDeltaNetCtx *)vctx;
    uint32_t n_heads = ctx->n_heads;
    uint32_t k_dim = ctx->k_dim;
    uint32_t v_dim = ctx->v_dim;

    if (!ctx->spec_keep_state && !ctx->state_only && ctx->B >= 32u &&
        (k_dim & 7u) == 0u && k_dim <= 128u && v_dim <= 128u && asi_chunkwy_enabled())
    {
        AsiDeltaNetWyBuf *wb = (AsiDeltaNetWyBuf *)asi_aligned_alloc(32, sizeof(AsiDeltaNetWyBuf));
        if (wb)
        {
            _Alignas(32) float Q_tile[4096];
            _Alignas(32) float K_tile[4096];
            _Alignas(32) float V_tile[4096];
            _Alignas(32) float O_tile[4096];
            float log_g_arr[32], g_arr[32], beta_arr[32];

            for (uint32_t h_idx = 0;; )
            {
                uint32_t h;
                if (n_threads <= 1)
                {
                    if (h_idx >= n_heads) break;
                    h = h_idx++;
                }
                else
                {
                    h = (uint32_t)atomic_fetch_add_explicit(&ctx->next_h, 1, memory_order_relaxed);
                    if (h >= n_heads) break;
                }
                float *S = ctx->ssm_state + (size_t)h * v_dim * k_dim;
                for (uint32_t j = 0; j < v_dim; ++j)
                    for (uint32_t k = 0; k < k_dim; ++k)
                        wb->S_T[(size_t)k * v_dim + j] = S[(size_t)j * k_dim + k];

                float dt_bias_val = ctx->dt_bias[h];
                float ssm_a_val = ctx->ssm_a[h];
                for (uint32_t t0 = 0; t0 < ctx->B; t0 += 32u)
                {
                    uint32_t nt = ctx->B - t0;
                    if (nt > 32u) nt = 32u;
                    for (uint32_t tc = 0; tc < nt; ++tc)
                    {
                        uint32_t t = t0 + tc;
                        const float *qkv = ctx->qkvb + (size_t)t * ctx->qkv_dim;
                        const float *qh = qkv + (size_t)h * k_dim;
                        const float *kh = qkv + ctx->inner + (size_t)h * k_dim;
                        const float *vh = qkv + 2 * ctx->inner + (size_t)h * v_dim;
                        float *qp = Q_tile + (size_t)tc * k_dim;
                        float *kp = K_tile + (size_t)tc * k_dim;
                        float *vp = V_tile + (size_t)tc * v_dim;
                        float q_ss = 0.0f, k_ss = 0.0f;
                        for (uint32_t i = 0; i < k_dim; ++i)
                        {
                            q_ss += qh[i] * qh[i];
                            k_ss += kh[i] * kh[i];
                        }
                        float q_inv = ctx->q_scale / sqrtf(q_ss + 1e-6f);
                        float k_inv = 1.0f / sqrtf(k_ss + 1e-6f);
                        for (uint32_t i = 0; i < k_dim; ++i)
                        {
                            qp[i] = qh[i] * q_inv;
                            kp[i] = kh[i] * k_inv;
                        }
                        memcpy(vp, vh, v_dim * sizeof(float));
                        float dt = fast_softplus(ctx->ab[(size_t)t * ctx->ssm_r + h] + dt_bias_val);
                        log_g_arr[tc] = ssm_a_val * dt;
                        g_arr[tc] = expf(log_g_arr[tc]);
                        beta_arr[tc] = fast_sigmoid(ctx->bb[(size_t)t * ctx->ssm_r + h]);
                    }
                    if (asi_deltanet_wy_chunk(wb, Q_tile, K_tile, V_tile, log_g_arr, beta_arr, O_tile, nt, k_dim, v_dim) != 0)
                    {
                        for (uint32_t j = 0; j < v_dim; ++j)
                            for (uint32_t k = 0; k < k_dim; ++k)
                                S[(size_t)j * k_dim + k] = wb->S_T[(size_t)k * v_dim + j];
                        for (uint32_t j = 0; j < v_dim; ++j)
                        {
                            float *Sj = S + (size_t)j * k_dim;
                            for (uint32_t tc = 0; tc < nt; ++tc)
                            {
                                const float *kp = K_tile + (size_t)tc * k_dim;
                                const float *qp = Q_tile + (size_t)tc * k_dim;
                                float g = g_arr[tc], beta = beta_arr[tc], pred = 0.0f;
                                for (uint32_t i = 0; i < k_dim; ++i) { Sj[i] *= g; pred += Sj[i] * kp[i]; }
                                float d = beta * (V_tile[(size_t)tc * v_dim + j] - pred), outv = 0.0f;
                                for (uint32_t i = 0; i < k_dim; ++i) { Sj[i] += d * kp[i]; outv += Sj[i] * qp[i]; }
                                O_tile[(size_t)tc * v_dim + j] = outv;
                            }
                        }
                        for (uint32_t j = 0; j < v_dim; ++j)
                            for (uint32_t k = 0; k < k_dim; ++k)
                                wb->S_T[(size_t)k * v_dim + j] = S[(size_t)j * k_dim + k];
                    }
                    for (uint32_t tc = 0; tc < nt; ++tc)
                    {
                        uint32_t t = t0 + tc;
                        float *oh = ctx->aob + (size_t)t * ctx->inner + (size_t)h * v_dim;
                        const float *zh = ctx->zb + (size_t)t * ctx->inner + (size_t)h * v_dim;
                        rmsnorm(oh, O_tile + (size_t)tc * v_dim, ctx->ssm_norm_w, v_dim, ctx->rms_eps);
                        fast_silu_mul_out_vec(oh, zh, v_dim);
                    }
                }
                for (uint32_t j = 0; j < v_dim; ++j)
                    for (uint32_t k = 0; k < k_dim; ++k)
                        S[(size_t)j * k_dim + k] = wb->S_T[(size_t)k * v_dim + j];
            }
            asi_aligned_free(wb);
            return;
        }
    }

#if defined(__AVX2__)
    if (ctx->B >= 1 && (k_dim & 7u) == 0u && (v_dim & 3u) == 0u && k_dim <= 256 && v_dim <= 256 &&
        (!ctx->spec_keep_state || ctx->B <= 16u))
    {
        uint32_t dt_t = (k_dim <= 128 && v_dim <= 128) ? 32u : 16u;
        _Alignas(32) float Q_tile[4096];
        _Alignas(32) float K_tile[4096];
        _Alignas(32) float V_tile[4096];
        _Alignas(32) float O_tile[4096];
        _Alignas(32) float S_spec[4][256];
        float g_arr[32], beta_arr[32];

        for (uint32_t h_idx = 0;; )
        {
            uint32_t h;
            if (n_threads <= 1)
            {
                if (h_idx >= n_heads) break;
                h = h_idx++;
            }
            else
            {
                h = (uint32_t)atomic_fetch_add_explicit(&ctx->next_h, 1, memory_order_relaxed);
                if (h >= n_heads) break;
            }

            float *S = ctx->ssm_state + (size_t)h * v_dim * k_dim;
            float dt_bias_val = ctx->dt_bias[h];
            float ssm_a_val = ctx->ssm_a[h];

            for (uint32_t t0 = 0; t0 < ctx->B; t0 += dt_t)
            {
                uint32_t nt = ctx->B - t0;
                if (nt > dt_t) nt = dt_t;

                for (uint32_t tc = 0; tc < nt; ++tc)
                {
                    uint32_t t = t0 + tc;
                    const float *qkv = ctx->qkvb + (size_t)t * ctx->qkv_dim;
                    const float *qh = qkv + (size_t)h * k_dim;
                    const float *kh = qkv + ctx->inner + (size_t)h * k_dim;
                    const float *vh = qkv + 2 * ctx->inner + (size_t)h * v_dim;
                    float *qp = Q_tile + (size_t)tc * k_dim;
                    float *kp = K_tile + (size_t)tc * k_dim;
                    float *vp = V_tile + (size_t)tc * v_dim;

                    if (!ctx->state_only)
                    {
                        __m256 vq = _mm256_setzero_ps();
                        __m256 vk = _mm256_setzero_ps();
                        for (uint32_t i = 0; i < k_dim; i += 8)
                        {
                            __m256 qv = _mm256_loadu_ps(qh + i);
                            __m256 kv = _mm256_loadu_ps(kh + i);
                            vq = _mm256_fmadd_ps(qv, qv, vq);
                            vk = _mm256_fmadd_ps(kv, kv, vk);
                        }
                        float q_ss = lila_hsum_f32_avx2(vq);
                        float k_ss = lila_hsum_f32_avx2(vk);
                        __m256 vqi = _mm256_set1_ps(ctx->q_scale / sqrtf(q_ss + 1e-6f));
                        __m256 vki = _mm256_set1_ps(1.0f / sqrtf(k_ss + 1e-6f));
                        for (uint32_t i = 0; i < k_dim; i += 8)
                        {
                            _mm256_store_ps(qp + i, _mm256_mul_ps(_mm256_loadu_ps(qh + i), vqi));
                            _mm256_store_ps(kp + i, _mm256_mul_ps(_mm256_loadu_ps(kh + i), vki));
                        }
                    }
                    else
                    {
                        __m256 vk = _mm256_setzero_ps();
                        for (uint32_t i = 0; i < k_dim; i += 8)
                        {
                            __m256 kv = _mm256_loadu_ps(kh + i);
                            vk = _mm256_fmadd_ps(kv, kv, vk);
                        }
                        float k_ss = lila_hsum_f32_avx2(vk);
                        __m256 vki = _mm256_set1_ps(1.0f / sqrtf(k_ss + 1e-6f));
                        for (uint32_t i = 0; i < k_dim; i += 8)
                        {
                            _mm256_store_ps(kp + i, _mm256_mul_ps(_mm256_loadu_ps(kh + i), vki));
                        }
                    }
                    memcpy(vp, vh, v_dim * sizeof(float));

                    float alpha_val = ctx->ab[(size_t)t * ctx->ssm_r + h];
                    float beta_raw = ctx->bb[(size_t)t * ctx->ssm_r + h];
                    float dt = fast_softplus(alpha_val + dt_bias_val);
                    g_arr[tc] = expf(ssm_a_val * dt);
                    beta_arr[tc] = fast_sigmoid(beta_raw);
                }

                for (uint32_t j = 0; j < v_dim; j += 4)
                {
                    float *S0 = S + (size_t)(j + 0) * k_dim;
                    float *S1 = S0 + k_dim;
                    float *S2 = S1 + k_dim;
                    float *S3 = S2 + k_dim;
                    for (uint32_t tc = 0; tc < nt; ++tc)
                    {
                        const float *kp = K_tile + (size_t)tc * k_dim;
                        const float *qp = Q_tile + (size_t)tc * k_dim;
                        const float *vp = V_tile + (size_t)tc * v_dim + j;
                        float *op = O_tile + (size_t)tc * v_dim + j;
                        __m256 vg = _mm256_set1_ps(g_arr[tc]);
                        float beta = beta_arr[tc];

                        const float *R0 = (ctx->spec_keep_state && tc > 0) ? S_spec[0] : S0;
                        const float *R1 = (ctx->spec_keep_state && tc > 0) ? S_spec[1] : S1;
                        const float *R2 = (ctx->spec_keep_state && tc > 0) ? S_spec[2] : S2;
                        const float *R3 = (ctx->spec_keep_state && tc > 0) ? S_spec[3] : S3;
                        float *W0 = ctx->spec_keep_state ? S_spec[0] : S0;
                        float *W1 = ctx->spec_keep_state ? S_spec[1] : S1;
                        float *W2 = ctx->spec_keep_state ? S_spec[2] : S2;
                        float *W3 = ctx->spec_keep_state ? S_spec[3] : S3;

                        __m256 vp0 = _mm256_setzero_ps(), vp1 = _mm256_setzero_ps();
                        __m256 vp2 = _mm256_setzero_ps(), vp3 = _mm256_setzero_ps();
                        for (uint32_t i = 0; i < k_dim; i += 8)
                        {
                            __m256 kv = _mm256_load_ps(kp + i);
                            __m256 s0 = _mm256_mul_ps(_mm256_loadu_ps(R0 + i), vg);
                            __m256 s1 = _mm256_mul_ps(_mm256_loadu_ps(R1 + i), vg);
                            __m256 s2 = _mm256_mul_ps(_mm256_loadu_ps(R2 + i), vg);
                            __m256 s3 = _mm256_mul_ps(_mm256_loadu_ps(R3 + i), vg);
                            _mm256_storeu_ps(W0 + i, s0);
                            _mm256_storeu_ps(W1 + i, s1);
                            _mm256_storeu_ps(W2 + i, s2);
                            _mm256_storeu_ps(W3 + i, s3);
                            vp0 = _mm256_fmadd_ps(s0, kv, vp0);
                            vp1 = _mm256_fmadd_ps(s1, kv, vp1);
                            vp2 = _mm256_fmadd_ps(s2, kv, vp2);
                            vp3 = _mm256_fmadd_ps(s3, kv, vp3);
                        }
                        __m256 vd0 = _mm256_set1_ps(beta * (vp[0] - lila_hsum_f32_avx2(vp0)));
                        __m256 vd1 = _mm256_set1_ps(beta * (vp[1] - lila_hsum_f32_avx2(vp1)));
                        __m256 vd2 = _mm256_set1_ps(beta * (vp[2] - lila_hsum_f32_avx2(vp2)));
                        __m256 vd3 = _mm256_set1_ps(beta * (vp[3] - lila_hsum_f32_avx2(vp3)));

                        if (!ctx->state_only)
                        {
                            __m256 vo0 = _mm256_setzero_ps(), vo1 = _mm256_setzero_ps();
                            __m256 vo2 = _mm256_setzero_ps(), vo3 = _mm256_setzero_ps();
                            for (uint32_t i = 0; i < k_dim; i += 8)
                            {
                                __m256 kv = _mm256_load_ps(kp + i);
                                __m256 qv = _mm256_load_ps(qp + i);
                                __m256 u0 = _mm256_fmadd_ps(vd0, kv, _mm256_loadu_ps(W0 + i));
                                __m256 u1 = _mm256_fmadd_ps(vd1, kv, _mm256_loadu_ps(W1 + i));
                                __m256 u2 = _mm256_fmadd_ps(vd2, kv, _mm256_loadu_ps(W2 + i));
                                __m256 u3 = _mm256_fmadd_ps(vd3, kv, _mm256_loadu_ps(W3 + i));
                                _mm256_storeu_ps(W0 + i, u0);
                                _mm256_storeu_ps(W1 + i, u1);
                                _mm256_storeu_ps(W2 + i, u2);
                                _mm256_storeu_ps(W3 + i, u3);
                                if (ctx->spec_keep_state && tc == 0)
                                {
                                    _mm256_storeu_ps(S0 + i, u0);
                                    _mm256_storeu_ps(S1 + i, u1);
                                    _mm256_storeu_ps(S2 + i, u2);
                                    _mm256_storeu_ps(S3 + i, u3);
                                }
                                vo0 = _mm256_fmadd_ps(u0, qv, vo0);
                                vo1 = _mm256_fmadd_ps(u1, qv, vo1);
                                vo2 = _mm256_fmadd_ps(u2, qv, vo2);
                                vo3 = _mm256_fmadd_ps(u3, qv, vo3);
                            }
                            op[0] = lila_hsum_f32_avx2(vo0);
                            op[1] = lila_hsum_f32_avx2(vo1);
                            op[2] = lila_hsum_f32_avx2(vo2);
                            op[3] = lila_hsum_f32_avx2(vo3);
                        }
                        else
                        {
                            for (uint32_t i = 0; i < k_dim; i += 8)
                            {
                                __m256 kv = _mm256_load_ps(kp + i);
                                __m256 u0 = _mm256_fmadd_ps(vd0, kv, _mm256_loadu_ps(W0 + i));
                                __m256 u1 = _mm256_fmadd_ps(vd1, kv, _mm256_loadu_ps(W1 + i));
                                __m256 u2 = _mm256_fmadd_ps(vd2, kv, _mm256_loadu_ps(W2 + i));
                                __m256 u3 = _mm256_fmadd_ps(vd3, kv, _mm256_loadu_ps(W3 + i));
                                _mm256_storeu_ps(W0 + i, u0);
                                _mm256_storeu_ps(W1 + i, u1);
                                _mm256_storeu_ps(W2 + i, u2);
                                _mm256_storeu_ps(W3 + i, u3);
                            }
                        }
                    }
                }

                if (!ctx->state_only)
                {
                    for (uint32_t tc = 0; tc < nt; ++tc)
                    {
                        uint32_t t = t0 + tc;
                        float *oh = ctx->aob + (size_t)t * ctx->inner + (size_t)h * v_dim;
                        const float *zh = ctx->zb + (size_t)t * ctx->inner + (size_t)h * v_dim;
                        rmsnorm(oh, O_tile + (size_t)tc * v_dim, ctx->ssm_norm_w, v_dim, ctx->rms_eps);
                        fast_silu_mul_out_vec(oh, zh, v_dim);
                    }
                }
            }
        }
        return;
    }
#endif

    size_t st_elems = (size_t)v_dim * k_dim;
    float *S_tmp = NULL;
    float *oh_tmp = NULL;
    float *zh_zero = NULL;
    if (ctx->spec_keep_state && ctx->B > 1)
    {
        S_tmp = (float *)malloc(st_elems * sizeof(float));
    }
    if (ctx->state_only)
    {
        oh_tmp = (float *)malloc((size_t)v_dim * sizeof(float));
        zh_zero = (float *)calloc(v_dim, sizeof(float));
    }

    for (uint32_t h_idx = 0;; )
    {
        uint32_t h;
        if (n_threads <= 1)
        {
            if (h_idx >= n_heads) break;
            h = h_idx++;
        }
        else
        {
            h = (uint32_t)atomic_fetch_add_explicit(&ctx->next_h, 1, memory_order_relaxed);
            if (h >= n_heads) break;
        }
        float *S_base = ctx->ssm_state + (size_t)h * st_elems;
        for (uint32_t t = 0; t < ctx->B; ++t)
        {
            float *qkv = ctx->qkvb + (size_t)t * ctx->qkv_dim;
            float *qh = qkv + (size_t)h * k_dim;
            float *kh = qkv + ctx->inner + (size_t)h * k_dim;
            float *vh = qkv + 2 * ctx->inner + (size_t)h * v_dim;
            float *zh = ctx->state_only ? zh_zero : (ctx->zb + (size_t)t * ctx->inner + (size_t)h * v_dim);
            float *oh = ctx->state_only ? oh_tmp : (ctx->aob + (size_t)t * ctx->inner + (size_t)h * v_dim);
            float a_val = ctx->ab[(size_t)t * ctx->ssm_r + h];
            float b_val = ctx->bb[(size_t)t * ctx->ssm_r + h];
            float *S_use = (ctx->spec_keep_state && t > 0 && S_tmp) ? S_tmp : S_base;

            deltanet_head_step_avx2(S_use, qh, kh, vh, zh, oh, ctx->ssm_norm_w,
                                    a_val, ctx->dt_bias[h], ctx->ssm_a[h], b_val,
                                    ctx->q_scale, k_dim, v_dim, ctx->rms_eps);
            if (ctx->spec_keep_state && t == 0 && ctx->B > 1 && S_tmp)
            {
                memcpy(S_tmp, S_base, st_elems * sizeof(float));
            }
        }
    }
    free(S_tmp);
    free(oh_tmp);
    free(zh_zero);
}

typedef struct {
    float *qgb;
    float *aob;
    const uint16_t *k_cache;
    const uint16_t *v_cache;
    const float *qn_w;
    const float *rope_inv_freq;
    float *thread_scores_blk;
    size_t thread_scores_cap;
    uint32_t B;
    uint32_t pos_start;
    uint32_t n_heads;
    uint32_t n_rep;
    uint32_t hd;
    uint32_t qg_dim;
    uint32_t kv_dim;
    uint32_t attn_dim;
    uint32_t rope_dim;
    float rope_theta;
    float rms_eps;
    float scale;
    const uint32_t *mrope_pos_3d;
    uint32_t mrope_sections[3];
    uint32_t mrope_flags;
    int64_t rope_pos_delta;
    _Alignas(64) atomic_int next_h;
} AsiFullAttnBatchCtx;

static void asi_full_attn_batch_worker(int tid, int n_threads, void *vctx)
{
    AsiFullAttnBatchCtx *ctx = (AsiFullAttnBatchCtx *)vctx;
    uint32_t max_pos = ctx->pos_start + ctx->B;
    uint32_t hd = ctx->hd;
    uint32_t kv_dim = ctx->kv_dim;
    uint32_t n_heads = ctx->n_heads;
    uint32_t n_rep = ctx->n_rep;

#if defined(__AVX2__) && defined(__F16C__)
    if (ctx->B >= 1 && hd <= 256 && (hd & 7u) == 0u && n_rep <= 8 &&
        (ctx->B >= 2 || (n_heads / n_rep) >= (uint32_t)n_threads))
    {
        uint32_t n_kv = n_heads / n_rep;
        uint32_t t_chunks = (ctx->B + 3u) / 4u;
        uint32_t n_tasks = n_kv * t_chunks;
        uint32_t nt_max = (ctx->B < 4u) ? ctx->B : 4u;
        size_t scores_cap = (size_t)nt_max * n_rep * max_pos;
        float stack_scores[4096];
        float *scores_blk = NULL;
        int heap_alloc = 0;
        if (scores_cap <= 4096)
        {
            scores_blk = stack_scores;
        }
        else if (ctx->thread_scores_blk && tid >= 0 && scores_cap <= ctx->thread_scores_cap)
        {
            scores_blk = ctx->thread_scores_blk + (size_t)tid * ctx->thread_scores_cap;
        }
        else
        {
            scores_blk = (float *)malloc(scores_cap * sizeof(float));
            heap_alloc = 1;
        }
        if (!scores_blk) return;

        _Alignas(32) float q_blk[4 * 8 * 256];
        _Alignas(32) float o_blk[4 * 8 * 256];
        _Alignas(32) float k_f32[256];
        _Alignas(32) float v_f32[256];

        for (uint32_t seq_task = 0;; )
        {
            uint32_t task;
            if (n_threads <= 1)
            {
                if (seq_task >= n_tasks) break;
                task = seq_task++;
            }
            else
            {
                task = (uint32_t)atomic_fetch_add_explicit(&ctx->next_h, 1, memory_order_relaxed);
                if (task >= n_tasks) break;
            }

            uint32_t kvh = task / t_chunks;
            uint32_t tc_idx = task - kvh * t_chunks;
            uint32_t t0 = tc_idx * 4u;
            uint32_t nt = ctx->B - t0;
            if (nt > 4u) nt = 4u;
            uint32_t h_base = kvh * n_rep;
            uint32_t max_t_pos = ctx->pos_start + t0 + nt - 1u;
            uint32_t stride_p = max_t_pos + 1u;

            float max_s[4 * 8];
            float inv_sum[4 * 8];
            for (uint32_t tc = 0; tc < nt; ++tc)
            {
                uint32_t tok_i = t0 + tc;
                uint32_t pos = ctx->pos_start + tok_i;
                float *qg = ctx->qgb + (size_t)tok_i * ctx->qg_dim;
                for (uint32_t r = 0; r < n_rep; ++r)
                {
                    uint32_t idx = tc * n_rep + r;
                    float *qh = qg + (size_t)(h_base + r) * 2 * hd;
                    float *q_dst = q_blk + (size_t)idx * hd;
                    rmsnorm(q_dst, qh, ctx->qn_w, hd, ctx->rms_eps);
                    if (ctx->mrope_pos_3d)
                    {
                        const uint32_t *p3 = ctx->mrope_pos_3d + (size_t)tok_i * 3u;
                        apply_rope_neox_3d(q_dst, ctx->rope_dim, p3[0], p3[1], p3[2],
                                           ctx->mrope_sections, ctx->mrope_flags,
                                           ctx->rope_theta, ctx->rope_inv_freq);
                    }
                    else
                    {
                        uint32_t rpos = asi_effective_rope_pos(ctx->rope_pos_delta, pos);
                        apply_rope_neox_partial(q_dst, ctx->rope_dim, rpos,
                                                ctx->rope_theta, ctx->rope_inv_freq);
                    }
                    max_s[idx] = -1e30f;
                }
            }
            memset(o_blk, 0, (size_t)nt * n_rep * hd * sizeof(float));

            for (uint32_t p = 0; p <= max_t_pos; ++p)
            {
                const uint16_t *kt = ctx->k_cache + (size_t)p * kv_dim + (size_t)kvh * hd;
                if (p + 2 <= max_t_pos)
                    LILA_PREFETCH(ctx->k_cache + (size_t)(p + 2) * kv_dim + (size_t)kvh * hd);
                for (uint32_t d = 0; d < hd; d += 8)
                    _mm256_store_ps(k_f32 + d, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(kt + d))));

                uint32_t tc_start = (p > ctx->pos_start + t0) ? (p - (ctx->pos_start + t0)) : 0u;
                for (uint32_t tc = tc_start; tc < nt; ++tc)
                {
                    if (n_rep == 4)
                    {
                        uint32_t base_idx = tc * 4u;
                        const float *q0 = q_blk + (size_t)(base_idx + 0) * hd;
                        const float *q1 = q_blk + (size_t)(base_idx + 1) * hd;
                        const float *q2 = q_blk + (size_t)(base_idx + 2) * hd;
                        const float *q3 = q_blk + (size_t)(base_idx + 3) * hd;
                        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
                        __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
                        for (uint32_t d = 0; d < hd; d += 8)
                        {
                            __m256 kv = _mm256_load_ps(k_f32 + d);
                            a0 = _mm256_fmadd_ps(_mm256_load_ps(q0 + d), kv, a0);
                            a1 = _mm256_fmadd_ps(_mm256_load_ps(q1 + d), kv, a1);
                            a2 = _mm256_fmadd_ps(_mm256_load_ps(q2 + d), kv, a2);
                            a3 = _mm256_fmadd_ps(_mm256_load_ps(q3 + d), kv, a3);
                        }
                        float s0 = lila_hsum_f32_avx2(a0) * ctx->scale;
                        float s1 = lila_hsum_f32_avx2(a1) * ctx->scale;
                        float s2 = lila_hsum_f32_avx2(a2) * ctx->scale;
                        float s3 = lila_hsum_f32_avx2(a3) * ctx->scale;
                        scores_blk[(size_t)(base_idx + 0) * stride_p + p] = s0;
                        scores_blk[(size_t)(base_idx + 1) * stride_p + p] = s1;
                        scores_blk[(size_t)(base_idx + 2) * stride_p + p] = s2;
                        scores_blk[(size_t)(base_idx + 3) * stride_p + p] = s3;
                        if (s0 > max_s[base_idx + 0]) max_s[base_idx + 0] = s0;
                        if (s1 > max_s[base_idx + 1]) max_s[base_idx + 1] = s1;
                        if (s2 > max_s[base_idx + 2]) max_s[base_idx + 2] = s2;
                        if (s3 > max_s[base_idx + 3]) max_s[base_idx + 3] = s3;
                    }
                    else
                    {
                        for (uint32_t r = 0; r < n_rep; ++r)
                        {
                            uint32_t idx = tc * n_rep + r;
                            const float *qr = q_blk + (size_t)idx * hd;
                            __m256 ar = _mm256_setzero_ps();
                            for (uint32_t d = 0; d < hd; d += 8)
                                ar = _mm256_fmadd_ps(_mm256_load_ps(qr + d), _mm256_load_ps(k_f32 + d), ar);
                            float sr = lila_hsum_f32_avx2(ar) * ctx->scale;
                            scores_blk[(size_t)idx * stride_p + p] = sr;
                            if (sr > max_s[idx]) max_s[idx] = sr;
                        }
                    }
                }
            }

            for (uint32_t tc = 0; tc < nt; ++tc)
            {
                uint32_t pos = ctx->pos_start + t0 + tc;
                for (uint32_t r = 0; r < n_rep; ++r)
                {
                    uint32_t idx = tc * n_rep + r;
                    inv_sum[idx] = asi_softmax_f32_inplace(scores_blk + (size_t)idx * stride_p, pos + 1, max_s[idx]);
                }
            }

            for (uint32_t p = 0; p <= max_t_pos; ++p)
            {
                const uint16_t *vt = ctx->v_cache + (size_t)p * kv_dim + (size_t)kvh * hd;
                if (p + 2 <= max_t_pos)
                    LILA_PREFETCH(ctx->v_cache + (size_t)(p + 2) * kv_dim + (size_t)kvh * hd);
                for (uint32_t d = 0; d < hd; d += 8)
                    _mm256_store_ps(v_f32 + d, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(vt + d))));

                uint32_t tc_start = (p > ctx->pos_start + t0) ? (p - (ctx->pos_start + t0)) : 0u;
                for (uint32_t tc = tc_start; tc < nt; ++tc)
                {
                    if (n_rep == 4)
                    {
                        uint32_t base_idx = tc * 4u;
                        __m256 vp0 = _mm256_set1_ps(scores_blk[(size_t)(base_idx + 0) * stride_p + p] * inv_sum[base_idx + 0]);
                        __m256 vp1 = _mm256_set1_ps(scores_blk[(size_t)(base_idx + 1) * stride_p + p] * inv_sum[base_idx + 1]);
                        __m256 vp2 = _mm256_set1_ps(scores_blk[(size_t)(base_idx + 2) * stride_p + p] * inv_sum[base_idx + 2]);
                        __m256 vp3 = _mm256_set1_ps(scores_blk[(size_t)(base_idx + 3) * stride_p + p] * inv_sum[base_idx + 3]);
                        float *o0 = o_blk + (size_t)(base_idx + 0) * hd;
                        float *o1 = o_blk + (size_t)(base_idx + 1) * hd;
                        float *o2 = o_blk + (size_t)(base_idx + 2) * hd;
                        float *o3 = o_blk + (size_t)(base_idx + 3) * hd;
                        for (uint32_t d = 0; d < hd; d += 8)
                        {
                            __m256 vv = _mm256_load_ps(v_f32 + d);
                            _mm256_store_ps(o0 + d, _mm256_fmadd_ps(vp0, vv, _mm256_load_ps(o0 + d)));
                            _mm256_store_ps(o1 + d, _mm256_fmadd_ps(vp1, vv, _mm256_load_ps(o1 + d)));
                            _mm256_store_ps(o2 + d, _mm256_fmadd_ps(vp2, vv, _mm256_load_ps(o2 + d)));
                            _mm256_store_ps(o3 + d, _mm256_fmadd_ps(vp3, vv, _mm256_load_ps(o3 + d)));
                        }
                    }
                    else
                    {
                        for (uint32_t r = 0; r < n_rep; ++r)
                        {
                            uint32_t idx = tc * n_rep + r;
                            __m256 vpr = _mm256_set1_ps(scores_blk[(size_t)idx * stride_p + p] * inv_sum[idx]);
                            float *orow = o_blk + (size_t)idx * hd;
                            for (uint32_t d = 0; d < hd; d += 8)
                                _mm256_store_ps(orow + d, _mm256_fmadd_ps(vpr, _mm256_load_ps(v_f32 + d), _mm256_load_ps(orow + d)));
                        }
                    }
                }
            }

            for (uint32_t tc = 0; tc < nt; ++tc)
            {
                float *qg = ctx->qgb + (size_t)(t0 + tc) * ctx->qg_dim;
                float *ao = ctx->aob + (size_t)(t0 + tc) * ctx->attn_dim;
                for (uint32_t r = 0; r < n_rep; ++r)
                {
                    uint32_t h = h_base + r;
                    uint32_t idx = tc * n_rep + r;
                    float *oh = ao + (size_t)h * hd;
                    const float *gh = qg + (size_t)h * 2 * hd + hd;
                    memcpy(oh, o_blk + (size_t)idx * hd, hd * sizeof(float));
                    asi_mul_sigmoid_vec(oh, gh, hd);
                }
            }
        }
        if (heap_alloc) free(scores_blk);
        return;
    }
#endif

    float stack_scores[2048];
    float *scores = NULL;
    int scalar_heap = 0;
    if (max_pos <= 2048)
    {
        scores = stack_scores;
    }
    else if (ctx->thread_scores_blk && tid >= 0 && (size_t)max_pos <= ctx->thread_scores_cap)
    {
        scores = ctx->thread_scores_blk + (size_t)tid * ctx->thread_scores_cap;
    }
    else
    {
        scores = (float *)malloc((size_t)max_pos * sizeof(float));
        scalar_heap = 1;
    }
    if (!scores) return;

    for (int h_idx = 0;; )
    {
        uint32_t h;
        if (n_threads <= 1)
        {
            if ((uint32_t)h_idx >= n_heads) break;
            h = (uint32_t)h_idx++;
        }
        else
        {
            h = (uint32_t)atomic_fetch_add_explicit(&ctx->next_h, 1, memory_order_relaxed);
            if (h >= n_heads) break;
        }
        uint32_t kvh = h / n_rep;
        for (uint32_t t = 0; t < ctx->B; ++t)
        {
            uint32_t pos = ctx->pos_start + t;
            float *qg = ctx->qgb + (size_t)t * ctx->qg_dim;
            float *ao = ctx->aob + (size_t)t * ctx->attn_dim;
            float *qh = qg + (size_t)h * 2 * hd;
            const float *gh = qh + hd;
            rmsnorm(qh, qh, ctx->qn_w, hd, ctx->rms_eps);
            if (ctx->mrope_pos_3d)
            {
                const uint32_t *p3 = ctx->mrope_pos_3d + (size_t)t * 3u;
                apply_rope_neox_3d(qh, ctx->rope_dim, p3[0], p3[1], p3[2],
                                   ctx->mrope_sections, ctx->mrope_flags,
                                   ctx->rope_theta, ctx->rope_inv_freq);
            }
            else
            {
                uint32_t rpos = asi_effective_rope_pos(ctx->rope_pos_delta, pos);
                apply_rope_neox_partial(qh, ctx->rope_dim, rpos,
                                        ctx->rope_theta, ctx->rope_inv_freq);
            }

            float max_score = -1e30f;
            for (uint32_t p = 0; p <= pos; ++p)
            {
                const uint16_t *kt = ctx->k_cache + (size_t)p * kv_dim + (size_t)kvh * hd;
                float sc_val = dot_f32_f16_avx2(qh, kt, hd) * ctx->scale;
                scores[p] = sc_val;
                if (sc_val > max_score) max_score = sc_val;
            }
            float inv_sum = asi_softmax_f32_inplace(scores, pos + 1, max_score);
            float *oh = ao + (size_t)h * hd;
            memset(oh, 0, hd * sizeof(float));
            for (uint32_t p = 0; p <= pos; ++p)
            {
                float prob = scores[p] * inv_sum;
                const uint16_t *vt = ctx->v_cache + (size_t)p * kv_dim + (size_t)kvh * hd;
                accum_f16_f32_avx2(oh, vt, prob, hd);
            }
            asi_mul_sigmoid_vec(oh, gh, hd);
        }
    }
    if (scalar_heap) free(scores);
}

typedef struct {
    double dn_proj;
    double ssm;
    double fa_proj;
    double fa_gqa;
    double ffn;
    double norm;
} Qwen35TokenProf;

static void qwen35_step_deltanet(AsiArchSession *s, uint32_t layer_idx, Qwen35TokenProf *prof)
{
    const AsiV2Cartridge *cart = s->cart;
    const AsiV2ModelConfig *c = &cart->config;
    const Qwen35LayerWeights *lw = &s->layers[layer_idx];
    Qwen35DeltaNetState *st = &s->ssm_states[layer_idx];
    Qwen35Scratch *sc = &s->scratch;

    uint32_t inner = c->ssm_inner_size;
    uint32_t qkv_dim = inner * 3;
    uint32_t n_heads = c->ssm_group_count;
    uint32_t k_dim = c->ssm_state_size;
    uint32_t v_dim = inner / n_heads;
    uint32_t k_conv = c->ssm_conv_kernel;
    uint32_t conv_hist = (k_conv > 1) ? (k_conv - 1) : 0;

    double tp = prof ? asi_now_ms() : 0.0;
    int bpr = ((int)c->hidden_size + QK_K - 1) / QK_K;
    const AsiV2TensorEntry *ts[4] = {lw->attn_qkv, lw->attn_gate, lw->ssm_alpha, lw->ssm_beta};
    float *outs[4] = {sc->qkv, sc->z, sc->alpha, sc->beta};
    if (s->fp32_ref)
    {
        matvec_fused_q8k_pool(s->pool, cart, ts, outs, 4, NULL, bpr, sc->x_norm, (int)c->hidden_size, 0, 1);
    }
    else if (!asi_tensor_needs_q8k(ts[0]) || !asi_tensor_needs_q8k(ts[1]) ||
             !asi_tensor_needs_q8k(ts[2]) || !asi_tensor_needs_q8k(ts[3]))
    {
        matvec_fused_q8k_pool(s->pool, cart, ts, outs, 4, NULL, bpr, sc->x_norm, (int)c->hidden_size, 0, 0);
    }
    else
    {
        LilaQ8KBlock stack_qx[64];
        LilaQ8KBlock *qx = (bpr <= 64) ? stack_qx : lila_q8k_alloc((size_t)bpr);
        if (qx)
        {
            quantize_vec_q8k(qx, sc->x_norm, (int)c->hidden_size);
            matvec_fused_q8k_pool(s->pool, cart, ts, outs, 4, qx, bpr, sc->x_norm, (int)c->hidden_size, 0, 0);
            if (qx != stack_qx) free(qx);
        }
        else
        {
            for (int m = 0; m < 4; ++m)
                matvec_tensor_pool(s->pool, cart, ts[m], outs[m], sc->x_norm, 0);
        }
    }
    if (prof)
    {
        double now = asi_now_ms();
        prof->dn_proj += now - tp;
        tp = now;
    }

    const float *conv_w = (const float *)tensor_data(cart, lw->ssm_conv1d);
    AsiConv1dCtx cctx = {sc->qkv, st->conv_state, conv_w, 1, qkv_dim, k_conv, conv_hist, 0};
    asi_pool_run(s->pool, asi_conv1d_worker, &cctx);

    const float *ssm_a = (const float *)tensor_data(cart, lw->ssm_a);
    const float *dt_bias = (const float *)tensor_data(cart, lw->ssm_dt_bias);
    const float *ssm_norm_w = (const float *)tensor_data(cart, lw->ssm_norm);

    float q_scale = 1.0f / sqrtf((float)k_dim);
    AsiDeltaNetCtx dctx = {
        st->ssm_state, sc->qkv, sc->z, sc->alpha, sc->beta, sc->attn_out,
        ssm_a, dt_bias, ssm_norm_w,
        1, n_heads, k_dim, v_dim, inner, qkv_dim, c->ssm_time_step_rank,
        q_scale, c->rms_norm_eps, 0, 0, ATOMIC_VAR_INIT(0)
    };
    asi_pool_run(s->pool, asi_deltanet_worker, &dctx);
    if (prof)
    {
        double now = asi_now_ms();
        prof->ssm += now - tp;
        tp = now;
    }

    matvec_tensor_pool(s->pool, cart, lw->ssm_out, sc->proj_out, sc->attn_out, s->fp32_ref);
    for (uint32_t i = 0; i < c->hidden_size; ++i)
    {
        sc->x[i] += sc->proj_out[i];
    }
    if (prof)
    {
        prof->dn_proj += asi_now_ms() - tp;
    }
}

static void qwen35_step_full_attn(AsiArchSession *s, uint32_t layer_idx, uint32_t pos, Qwen35TokenProf *prof)
{
    const AsiV2Cartridge *cart = s->cart;
    const AsiV2ModelConfig *c = &cart->config;
    const Qwen35LayerWeights *lw = &s->layers[layer_idx];
    Qwen35AttnKVState *kv = &s->kv_states[layer_idx];
    Qwen35Scratch *sc = &s->scratch;

    uint32_t n_heads = c->n_heads;
    uint32_t n_kv = c->n_kv_heads;
    uint32_t hd = c->head_dim;
    uint32_t kv_dim = n_kv * hd;
    uint32_t n_rep = n_heads / n_kv;

    double tp = prof ? asi_now_ms() : 0.0;
    int bpr = ((int)c->hidden_size + QK_K - 1) / QK_K;
    const AsiV2TensorEntry *ts[3] = {lw->attn_q, lw->attn_k, lw->attn_v};
    float *outs[3] = {sc->qg, sc->k, sc->v};
    if (s->fp32_ref)
    {
        matvec_fused_q8k_pool(s->pool, cart, ts, outs, 3, NULL, bpr, sc->x_norm, (int)c->hidden_size, 0, 1);
    }
    else if (!asi_tensor_needs_q8k(ts[0]) || !asi_tensor_needs_q8k(ts[1]) || !asi_tensor_needs_q8k(ts[2]))
    {
        matvec_fused_q8k_pool(s->pool, cart, ts, outs, 3, NULL, bpr, sc->x_norm, (int)c->hidden_size, 0, 0);
    }
    else
    {
        LilaQ8KBlock stack_qx[64];
        LilaQ8KBlock *qx = (bpr <= 64) ? stack_qx : lila_q8k_alloc((size_t)bpr);
        if (qx)
        {
            quantize_vec_q8k(qx, sc->x_norm, (int)c->hidden_size);
            matvec_fused_q8k_pool(s->pool, cart, ts, outs, 3, qx, bpr, sc->x_norm, (int)c->hidden_size, 0, 0);
            if (qx != stack_qx) free(qx);
        }
        else
        {
            for (int m = 0; m < 3; ++m)
                matvec_tensor_pool(s->pool, cart, ts[m], outs[m], sc->x_norm, 0);
        }
    }
    if (prof)
    {
        double now = asi_now_ms();
        prof->fa_proj += now - tp;
        tp = now;
    }

    const float *qn_w = (const float *)tensor_data(cart, lw->attn_q_norm);
    const float *kn_w = (const float *)tensor_data(cart, lw->attn_k_norm);

    uint32_t rpos = asi_effective_rope_pos(s->mrope_pos_delta, pos);
    for (uint32_t h = 0; h < n_kv; ++h)
    {
        float *kh = sc->k + (size_t)h * hd;
        rmsnorm(kh, kh, kn_w, hd, c->rms_norm_eps);
        apply_rope_neox_partial(kh, c->rope_dim, rpos, c->rope_theta, s->rope_inv_freq);
    }

    uint16_t *k_dst = kv->k_cache + (size_t)pos * kv_dim;
    uint16_t *v_dst = kv->v_cache + (size_t)pos * kv_dim;
    uint32_t kv_i = 0;
#if defined(__AVX2__) && defined(__F16C__)
    for (; kv_i + 7 < kv_dim; kv_i += 8)
    {
        __m128i hk = _mm256_cvtps_ph(_mm256_loadu_ps(sc->k + kv_i), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m128i hv = _mm256_cvtps_ph(_mm256_loadu_ps(sc->v + kv_i), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm_storeu_si128((__m128i *)(k_dst + kv_i), hk);
        _mm_storeu_si128((__m128i *)(v_dst + kv_i), hv);
    }
#endif
    for (; kv_i < kv_dim; ++kv_i)
    {
        k_dst[kv_i] = fp32_to_fp16(sc->k[kv_i]);
        v_dst[kv_i] = fp32_to_fp16(sc->v[kv_i]);
    }

    AsiFullAttnBatchCtx fctx;
    memset(&fctx, 0, sizeof(fctx));
    fctx.qgb = sc->qg;
    fctx.aob = sc->attn_out;
    fctx.k_cache = kv->k_cache;
    fctx.v_cache = kv->v_cache;
    fctx.qn_w = qn_w;
    fctx.rope_inv_freq = s->rope_inv_freq;
    fctx.thread_scores_blk = sc->thread_scores_blk;
    fctx.thread_scores_cap = sc->thread_scores_cap;
    fctx.B = 1u;
    fctx.pos_start = pos;
    fctx.n_heads = n_heads;
    fctx.n_rep = n_rep;
    fctx.hd = hd;
    fctx.qg_dim = n_heads * 2u * hd;
    fctx.kv_dim = kv_dim;
    fctx.attn_dim = n_heads * hd;
    fctx.rope_dim = c->rope_dim;
    fctx.rope_theta = c->rope_theta;
    fctx.rms_eps = c->rms_norm_eps;
    fctx.scale = 1.0f / sqrtf((float)hd);
    fctx.rope_pos_delta = s->mrope_pos_delta;
    atomic_init(&fctx.next_h, 0);
    asi_pool_run(s->pool, asi_full_attn_batch_worker, &fctx);

    if (prof)
    {
        double now = asi_now_ms();
        prof->fa_gqa += now - tp;
        tp = now;
    }

    matvec_tensor_pool(s->pool, cart, lw->attn_output, sc->proj_out, sc->attn_out, s->fp32_ref);
    for (uint32_t i = 0; i < c->hidden_size; ++i)
    {
        sc->x[i] += sc->proj_out[i];
    }
    if (prof)
    {
        prof->fa_proj += asi_now_ms() - tp;
    }
}

static int qwen35_forward_token(AsiArchSession *s, int32_t token_id, uint32_t pos, float *out_logits)
{
    if (!s || !out_logits || token_id < 0 || pos >= s->max_ctx) return -1;
    const AsiV2Cartridge *cart = s->cart;
    const AsiV2ModelConfig *c = &cart->config;
    if ((uint32_t)token_id >= c->vocab_size) return -1;

    s->spec_count = 0;
    s->spec_cursor = 0;
    if (pos == 0) s->mrope_pos_delta = 0;
    if (pos < s->kv_dirty_from) s->kv_dirty_from = pos;
    if (pos < s->history_valid_from) s->history_valid_from = pos;
    if (s->token_history)
    {
        s->token_history[pos] = token_id;
        if (pos + 1 > s->history_len) s->history_len = pos + 1;
    }

    Qwen35Scratch *sc = &s->scratch;
    const int do_prof = asi_runtime_config()->profile;
    Qwen35TokenProf prof = {0};
    double t_start = do_prof ? asi_now_ms() : 0.0;
    asi_pool_begin(s->pool);
    const uint8_t *embd_base = tensor_data(cart, s->token_embd);
    int is_embd_repacked = (s->token_embd->layout_flags & ASI_TENSOR_LAYOUT_AVX2_REPACK) ? 1 : 0;
    size_t embd_rb = lila_quant_row_bytes(s->token_embd->quant_type, (int)c->hidden_size, is_embd_repacked);
    if (embd_rb == 0)
    {
        asi_pool_end(s->pool);
        return -1;
    }
    lila_dequant_row(sc->x, embd_base + (size_t)token_id * embd_rb, (int)c->hidden_size,
                     s->token_embd->quant_type, is_embd_repacked);

    for (uint32_t l = 0; l < c->n_layers; ++l)
    {
        const Qwen35LayerWeights *lw = &s->layers[l];
        const float *an_w = (const float *)tensor_data(cart, lw->attn_norm);
        double tp = do_prof ? asi_now_ms() : 0.0;
        rmsnorm(sc->x_norm, sc->x, an_w, c->hidden_size, c->rms_norm_eps);
        if (do_prof) prof.norm += asi_now_ms() - tp;

        if (lw->kind == QWEN35_LAYER_DELTANET)
        {
            qwen35_step_deltanet(s, l, do_prof ? &prof : NULL);
        }
        else
        {
            qwen35_step_full_attn(s, l, pos, do_prof ? &prof : NULL);
        }

        const float *pn_w = (const float *)tensor_data(cart, lw->post_attn_norm);
        tp = do_prof ? asi_now_ms() : 0.0;
        rmsnorm(sc->x_norm, sc->x, pn_w, c->hidden_size, c->rms_norm_eps);
        if (do_prof)
        {
            double now = asi_now_ms();
            prof.norm += now - tp;
            tp = now;
        }

        int bpr = ((int)c->hidden_size + QK_K - 1) / QK_K;
        const AsiV2TensorEntry *ts[2] = {lw->ffn_gate, lw->ffn_up};
        float *outs[2] = {sc->ffn_gate, sc->ffn_up};
        if (s->fp32_ref)
        {
            matvec_fused_q8k_pool(s->pool, cart, ts, outs, 2, NULL, bpr, sc->x_norm, (int)c->hidden_size, 1, 1);
        }
        else if (!asi_tensor_needs_q8k(ts[0]) || !asi_tensor_needs_q8k(ts[1]))
        {
            matvec_fused_q8k_pool(s->pool, cart, ts, outs, 2, NULL, bpr, sc->x_norm, (int)c->hidden_size, 1, 0);
        }
        else
        {
            LilaQ8KBlock stack_qx[64];
            LilaQ8KBlock *qx = (bpr <= 64) ? stack_qx : lila_q8k_alloc((size_t)bpr);
            if (qx)
            {
                quantize_vec_q8k(qx, sc->x_norm, (int)c->hidden_size);
                matvec_fused_q8k_pool(s->pool, cart, ts, outs, 2, qx, bpr, sc->x_norm, (int)c->hidden_size, 1, 0);
                if (qx != stack_qx) free(qx);
            }
            else
            {
                matvec_tensor_pool(s->pool, cart, lw->ffn_gate, sc->ffn_gate, sc->x_norm, 0);
                matvec_tensor_pool(s->pool, cart, lw->ffn_up, sc->ffn_up, sc->x_norm, 0);
                fast_silu_mul_vec(sc->ffn_gate, sc->ffn_up, c->intermediate_size);
            }
        }
        if (!s->fp32_ref && lw->ffn_down->quant_type == ASI_QUANT_Q4_REPACK &&
            (lw->ffn_down->layout_flags & ASI_TENSOR_LAYOUT_COL_MAJOR))
        {
            matvec_q4_col_major_pool(s->pool, cart, lw->ffn_down, sc->proj_out, sc->ffn_gate, s->ffn_sparse_keep);
        }
        else
        {
            matvec_tensor_pool(s->pool, cart, lw->ffn_down, sc->proj_out, sc->ffn_gate, s->fp32_ref);
        }
        for (uint32_t i = 0; i < c->hidden_size; ++i)
        {
            sc->x[i] += sc->proj_out[i];
        }
        if (do_prof) prof.ffn += asi_now_ms() - tp;
    }

    double tp = do_prof ? asi_now_ms() : 0.0;
    const float *on_w = (const float *)tensor_data(cart, s->output_norm);
    rmsnorm(sc->x_norm, sc->x, on_w, c->hidden_size, c->rms_norm_eps);
    if (do_prof)
    {
        double now = asi_now_ms();
        prof.norm += now - tp;
        tp = now;
    }

    const AsiV2TensorEntry *lm_head = s->output_weight ? s->output_weight : s->token_embd;
    int rc = (s->lmhead_idx && s->lmhead_ws)
                 ? lila_lhc_forward_pool(s->lmhead_idx, s->lmhead_ws, lhc_asi_dispatch, s->pool,
                                         tensor_data(cart, lm_head), sc->x_norm, out_logits, s->fp32_ref)
                 : matvec_tensor_pool(s->pool, cart, lm_head, out_logits, sc->x_norm, s->fp32_ref);
    double t_lm = do_prof ? (asi_now_ms() - tp) : 0.0;
    asi_pool_end(s->pool);
    if (do_prof)
    {
        double t_end = asi_now_ms();
        fprintf(stderr,
                "[PROF token pos=%u] total=%.2f ms | dn_proj=%.2f ssm=%.2f fa_proj=%.2f fa_gqa=%.2f ffn=%.2f norm=%.2f lm_head=%.2f\n",
                pos, t_end - t_start,
                prof.dn_proj, prof.ssm, prof.fa_proj, prof.fa_gqa,
                prof.ffn, prof.norm, t_lm);
    }
    return rc;
}

static int ensure_batch_scratch(AsiArchSession *s, uint32_t B)
{
    Qwen35Scratch *sc = &s->scratch;
    if (B <= sc->batch_cap) return 0;

    const AsiV2ModelConfig *c = &s->cart->config;
    uint32_t H = c->hidden_size;
    uint32_t I = c->intermediate_size;
    uint32_t qkv_dim = c->ssm_inner_size * 3;
    uint32_t z_dim = c->ssm_inner_size;
    uint32_t qg_dim = c->n_heads * 2 * c->head_dim;
    uint32_t kv_dim = c->n_kv_heads * c->head_dim;
    uint32_t attn_dim = c->n_heads * c->head_dim;
    uint32_t ssm_r = c->ssm_time_step_rank;
    uint32_t max_in_dim = H;
    if (I > max_in_dim) max_in_dim = I;
    if (attn_dim > max_in_dim) max_in_dim = attn_dim;
    if (z_dim > max_in_dim) max_in_dim = z_dim;
    int bpr_max = ((int)max_in_dim + QK_K - 1) / QK_K;

    free(sc->xb); free(sc->xnb); free(sc->projb); free(sc->qkvb);
    free(sc->zb); free(sc->ab); free(sc->bb); free(sc->qgb);
    free(sc->kb); free(sc->vb); free(sc->aob); free(sc->fgb);
    free(sc->fub); asi_aligned_free(sc->qx_h);

    sc->xb    = (float *)malloc((size_t)B * H * sizeof(float));
    sc->xnb   = (float *)malloc((size_t)B * H * sizeof(float));
    sc->projb = (float *)malloc((size_t)B * H * sizeof(float));
    sc->qkvb  = (float *)malloc((size_t)B * qkv_dim * sizeof(float));
    sc->zb    = (float *)malloc((size_t)B * z_dim * sizeof(float));
    sc->ab    = (float *)malloc((size_t)B * ssm_r * sizeof(float));
    sc->bb    = (float *)malloc((size_t)B * ssm_r * sizeof(float));
    sc->qgb   = (float *)malloc((size_t)B * qg_dim * sizeof(float));
    sc->kb    = (float *)malloc((size_t)B * kv_dim * sizeof(float));
    sc->vb    = (float *)malloc((size_t)B * kv_dim * sizeof(float));
    sc->aob   = (float *)malloc((size_t)B * (attn_dim > z_dim ? attn_dim : z_dim) * sizeof(float));
    sc->fgb   = (float *)malloc((size_t)B * I * sizeof(float));
    sc->fub   = (float *)malloc((size_t)B * I * sizeof(float));
    size_t qx_h_sz = (size_t)B * (size_t)bpr_max * sizeof(LilaQ8KBlock);
    sc->qx_h  = qx_h_sz ? (LilaQ8KBlock *)asi_aligned_alloc(64, qx_h_sz) : NULL;

    if (!sc->xb || !sc->xnb || !sc->projb || !sc->qkvb || !sc->zb || !sc->ab || !sc->bb ||
        !sc->qgb || !sc->kb || !sc->vb || !sc->aob || !sc->fgb || !sc->fub || !sc->qx_h)
    {
        sc->batch_cap = 0;
        return -1;
    }
    sc->batch_cap = B;
    return 0;
}

typedef struct {
    const uint8_t *embd_base;
    const int32_t *token_ids;
    float *xb;
    uint32_t B;
    uint32_t H;
    size_t row_bytes;
    uint32_t quant_type;
    int is_repacked;
} AsiEmbdBatchCtx;

static void asi_embd_batch_worker(int tid, int n_threads, void *vctx)
{
    const AsiEmbdBatchCtx *ctx = (const AsiEmbdBatchCtx *)vctx;
    uint32_t t0 = (ctx->B * (uint32_t)tid) / (uint32_t)n_threads;
    uint32_t t1 = (ctx->B * (uint32_t)(tid + 1)) / (uint32_t)n_threads;
    for (uint32_t t = t0; t < t1; ++t)
    {
        float *dst = ctx->xb + (size_t)t * ctx->H;
        const uint8_t *src = ctx->embd_base + (size_t)ctx->token_ids[t] * ctx->row_bytes;
        lila_dequant_row(dst, src, (int)ctx->H, ctx->quant_type, ctx->is_repacked);
    }
}

typedef struct {
    float *xb;
    const float *projb;
    float *xnb;
    LilaQ8KBlock *qx_all;
    const float *w_norm;
    uint32_t B;
    uint32_t H;
    float eps;
    int do_add;
    int do_norm;
    int do_quant;
} AsiNormQuantCtx;

static void asi_norm_quant_worker(int tid, int n_threads, void *vctx)
{
    const AsiNormQuantCtx *ctx = (const AsiNormQuantCtx *)vctx;
    uint32_t t0 = (ctx->B * (uint32_t)tid) / (uint32_t)n_threads;
    uint32_t t1 = (ctx->B * (uint32_t)(tid + 1)) / (uint32_t)n_threads;
    uint32_t H = ctx->H;
    for (uint32_t t = t0; t < t1; ++t)
    {
        float *x = ctx->xb + (size_t)t * H;
        if (ctx->do_add)
        {
            const float *p = ctx->projb + (size_t)t * H;
            uint32_t i = 0;
#if defined(__AVX2__)
            for (; i + 7 < H; i += 8)
            {
                _mm256_storeu_ps(x + i, _mm256_add_ps(_mm256_loadu_ps(x + i), _mm256_loadu_ps(p + i)));
            }
#endif
            for (; i < H; ++i) x[i] += p[i];
        }
        if (ctx->do_norm)
        {
            float *xn = ctx->xnb + (size_t)t * H;
            rmsnorm(xn, x, ctx->w_norm, H, ctx->eps);
            if (ctx->do_quant)
            {
                quantize_vec_q8k_strided(ctx->qx_all + t, (int)ctx->B, xn, (int)H);
            }
        }
    }
}

static void asi_norm_quant_pool(AsiThreadPool *pool,
                                float *xb, const float *projb, float *xnb,
                                LilaQ8KBlock *qx_all, const float *w_norm,
                                uint32_t B, uint32_t H, float eps,
                                int do_add, int do_norm, int do_quant)
{
    AsiNormQuantCtx ctx = {xb, projb, xnb, qx_all, w_norm, B, H, eps, do_add, do_norm, do_quant};
    if (pool && B >= 8)
        asi_pool_run(pool, asi_norm_quant_worker, &ctx);
    else
        asi_norm_quant_worker(0, 1, &ctx);
}

typedef struct {
    float *kb;
    const float *vb;
    uint16_t *k_cache;
    uint16_t *v_cache;
    const float *kn_w;
    const float *rope_inv_freq;
    uint32_t B;
    uint32_t pos_start;
    uint32_t n_kv;
    uint32_t hd;
    uint32_t kv_dim;
    uint32_t rope_dim;
    float rope_theta;
    float rms_eps;
    const uint32_t *mrope_pos_3d;
    uint32_t mrope_sections[3];
    uint32_t mrope_flags;
    int64_t rope_pos_delta;
} AsiKvPrepBatchCtx;

static void asi_kv_prep_batch_worker(int tid, int n_threads, void *vctx)
{
    const AsiKvPrepBatchCtx *ctx = (const AsiKvPrepBatchCtx *)vctx;
    uint32_t t0 = (ctx->B * (uint32_t)tid) / (uint32_t)n_threads;
    uint32_t t1 = (ctx->B * (uint32_t)(tid + 1)) / (uint32_t)n_threads;
    uint32_t kv_dim = ctx->kv_dim;
    uint32_t hd = ctx->hd;
    for (uint32_t t = t0; t < t1; ++t)
    {
        uint32_t pos = ctx->pos_start + t;
        float *k = ctx->kb + (size_t)t * kv_dim;
        const float *v = ctx->vb + (size_t)t * kv_dim;

        for (uint32_t h = 0; h < ctx->n_kv; ++h)
        {
            float *kh = k + (size_t)h * hd;
            rmsnorm(kh, kh, ctx->kn_w, hd, ctx->rms_eps);
            if (ctx->mrope_pos_3d)
            {
                const uint32_t *p3 = ctx->mrope_pos_3d + (size_t)t * 3u;
                apply_rope_neox_3d(kh, ctx->rope_dim, p3[0], p3[1], p3[2],
                                   ctx->mrope_sections, ctx->mrope_flags,
                                   ctx->rope_theta, ctx->rope_inv_freq);
            }
            else
            {
                uint32_t rpos = asi_effective_rope_pos(ctx->rope_pos_delta, pos);
                apply_rope_neox_partial(kh, ctx->rope_dim, rpos,
                                        ctx->rope_theta, ctx->rope_inv_freq);
            }
        }
        uint16_t *k_dst = ctx->k_cache + (size_t)pos * kv_dim;
        uint16_t *v_dst = ctx->v_cache + (size_t)pos * kv_dim;
        uint32_t i = 0;
#if defined(__AVX2__) && defined(__F16C__)
        for (; i + 7 < kv_dim; i += 8)
        {
            __m128i hk = _mm256_cvtps_ph(_mm256_loadu_ps(k + i), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m128i hv = _mm256_cvtps_ph(_mm256_loadu_ps(v + i), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            _mm_storeu_si128((__m128i *)(k_dst + i), hk);
            _mm_storeu_si128((__m128i *)(v_dst + i), hv);
        }
#endif
        for (; i < kv_dim; ++i)
        {
            k_dst[i] = fp32_to_fp16(k[i]);
            v_dst[i] = fp32_to_fp16(v[i]);
        }
    }
}

static int qwen35_forward_chunk_ex(AsiArchSession *s,
                                   const int32_t *token_ids,
                                   uint32_t B,
                                   uint32_t pos_start,
                                   int compute_logits,
                                   int spec_mode,
                                   float *out_last_logits,
                                   float *out_spec_logits,
                                   const float *input_embds,
                                   const uint32_t *mrope_pos_3d,
                                   const uint32_t mrope_sections[3],
                                   uint32_t mrope_flags,
                                   const volatile int *cancel_flag)
{
    int do_prof = asi_runtime_config()->profile;
    double tc0 = do_prof ? asi_now_ms() : 0.0;
    double t_mm = 0.0, t_ssm = 0.0, t_attn = 0.0, t_conv = 0.0, t_norm = 0.0, t_lm = 0.0;
    const AsiV2Cartridge *cart = s->cart;
    const AsiV2ModelConfig *c = &cart->config;
    uint32_t H = c->hidden_size;
    uint32_t I = c->intermediate_size;
    uint32_t qkv_dim = c->ssm_inner_size * 3;
    uint32_t z_dim = c->ssm_inner_size;
    (void)z_dim;
    uint32_t qg_dim = c->n_heads * 2 * c->head_dim;
    uint32_t kv_dim = c->n_kv_heads * c->head_dim;
    uint32_t attn_dim = c->n_heads * c->head_dim;
    uint32_t ssm_r = c->ssm_time_step_rank;
    int bpr_h = ((int)H + QK_K - 1) / QK_K;

    if (ensure_batch_scratch(s, B) != 0) return -1;
    float *xb    = s->scratch.xb;
    float *xnb   = s->scratch.xnb;
    float *projb = s->scratch.projb;
    float *qkvb  = s->scratch.qkvb;
    float *zb    = s->scratch.zb;
    float *ab    = s->scratch.ab;
    float *bb    = s->scratch.bb;
    float *qgb   = s->scratch.qgb;
    float *kb    = s->scratch.kb;
    float *vb    = s->scratch.vb;
    float *aob   = s->scratch.aob;
    float *fgb   = s->scratch.fgb;
    float *fub   = s->scratch.fub;
    LilaQ8KBlock *qx_h = s->scratch.qx_h;

    if (input_embds != NULL)
    {
        memcpy(xb, input_embds, (size_t)B * H * sizeof(float));
    }
    else
    {
        const uint8_t *embd_base = tensor_data(cart, s->token_embd);
        int is_embd_repacked = (s->token_embd->layout_flags & ASI_TENSOR_LAYOUT_AVX2_REPACK) ? 1 : 0;
        size_t embd_rb = lila_quant_row_bytes(s->token_embd->quant_type, (int)H, is_embd_repacked);
        if (embd_rb == 0) return -1;
        AsiEmbdBatchCtx ectx = {embd_base, token_ids, xb, B, H, embd_rb, s->token_embd->quant_type, is_embd_repacked};
        if (s->pool && B >= 16)
            asi_pool_run(s->pool, asi_embd_batch_worker, &ectx);
        else
            asi_embd_batch_worker(0, 1, &ectx);
    }

    const int use_blis = (!s->fp32_ref &&
                          (int)B >= asi_runtime_config()->blis_min_tokens &&
                          asi_blis_prefill_enabled() &&
                          asi_cpu_has_avx2_fma_f16c());
    {
        double tp = do_prof ? asi_now_ms() : 0.0;
        const float *an0_w = (const float *)tensor_data(cart, s->layers[0].attn_norm);
        const AsiV2TensorEntry *l0_t = (s->layers[0].kind == QWEN35_LAYER_DELTANET)
                                           ? s->layers[0].attn_qkv
                                           : s->layers[0].attn_q;
        int need_q8 = (!s->fp32_ref && asi_tensor_needs_q8k(l0_t) &&
                       !(use_blis && asi_tensor_blis_supported(l0_t)));
        asi_norm_quant_pool(s->pool, xb, NULL, xnb, qx_h, an0_w, B, H, c->rms_norm_eps, 0, 1, need_q8);
        if (do_prof) t_norm += asi_now_ms() - tp;
    }

    uint32_t i_dn = 0;
    for (uint32_t l = 0; l < c->n_layers; ++l)
    {
        if (cancel_flag && *cancel_flag) return -2;
        const Qwen35LayerWeights *lw = &s->layers[l];
        double tp = 0.0;

        if (lw->kind == QWEN35_LAYER_DELTANET)
        {
            if (s->calib_hook) s->calib_hook(s->calib_ctx, l, 0, xnb, B, H);
            tp = do_prof ? asi_now_ms() : 0.0;
            const AsiV2TensorEntry *ts[4] = {lw->attn_qkv, lw->attn_gate, lw->ssm_alpha, lw->ssm_beta};
            float *outs[4] = {qkvb, zb, ab, bb};
            matmul_batch_fused_q8k_pool(s->pool, cart, ts, outs, 4,
                                        s->fp32_ref ? NULL : qx_h, xnb, (int)B, bpr_h, 0, s->fp32_ref);
            if (do_prof) t_mm += asi_now_ms() - tp;

            if (spec_mode && B > 1 && i_dn < s->spec_n_dn && (B - 1u) <= s->spec_cap_k &&
                s->spec_saved_qkv && s->spec_saved_a && s->spec_saved_b)
            {
                size_t cap_k = (size_t)s->spec_cap_k;
                size_t save_cnt = (size_t)(B - 1u);
                memcpy(s->spec_saved_qkv + (size_t)i_dn * cap_k * qkv_dim,
                       qkvb + qkv_dim, save_cnt * qkv_dim * sizeof(float));
                memcpy(s->spec_saved_a + (size_t)i_dn * cap_k * ssm_r,
                       ab + ssm_r, save_cnt * ssm_r * sizeof(float));
                memcpy(s->spec_saved_b + (size_t)i_dn * cap_k * ssm_r,
                       bb + ssm_r, save_cnt * ssm_r * sizeof(float));
            }

            Qwen35DeltaNetState *st = &s->ssm_states[l];
            const float *conv_w = (const float *)tensor_data(cart, lw->ssm_conv1d);
            const float *ssm_a = (const float *)tensor_data(cart, lw->ssm_a);
            const float *dt_bias = (const float *)tensor_data(cart, lw->ssm_dt_bias);
            const float *ssm_norm_w = (const float *)tensor_data(cart, lw->ssm_norm);
            uint32_t n_heads = c->ssm_group_count;
            uint32_t k_dim = c->ssm_state_size;
            uint32_t v_dim = c->ssm_inner_size / n_heads;
            uint32_t k_conv = c->ssm_conv_kernel;
            uint32_t conv_hist = (k_conv > 1) ? (k_conv - 1) : 0;
            float q_scale = 1.0f / sqrtf((float)k_dim);

            tp = do_prof ? asi_now_ms() : 0.0;
            AsiConv1dCtx cctx = {qkvb, st->conv_state, conv_w, B, qkv_dim, k_conv, conv_hist, spec_mode ? 1 : 0};
            asi_pool_run(s->pool, asi_conv1d_worker, &cctx);
            if (do_prof) t_conv += asi_now_ms() - tp;
            tp = do_prof ? asi_now_ms() : 0.0;
            AsiDeltaNetCtx dctx = {
                st->ssm_state, qkvb, zb, ab, bb, aob,
                ssm_a, dt_bias, ssm_norm_w,
                B, n_heads, k_dim, v_dim, c->ssm_inner_size, qkv_dim, ssm_r,
                q_scale, c->rms_norm_eps, spec_mode ? 1 : 0, 0, ATOMIC_VAR_INIT(0)
            };
            asi_pool_run(s->pool, asi_deltanet_worker, &dctx);
            if (do_prof) t_ssm += asi_now_ms() - tp;
            tp = do_prof ? asi_now_ms() : 0.0;
            int ssm_in_cols = (int)lw->ssm_out->shape[0];
            int ssm_in_bpr = (ssm_in_cols + QK_K - 1) / QK_K;
            if (s->calib_hook) s->calib_hook(s->calib_ctx, l, 1, aob, B, (uint32_t)ssm_in_cols);
            int need_q8_ssm_out = (!s->fp32_ref && asi_tensor_needs_q8k(lw->ssm_out) &&
                                   !(use_blis && asi_tensor_blis_supported(lw->ssm_out)));
            if (need_q8_ssm_out) quantize_batch_q8k_pool(s->pool, qx_h, aob, (int)B, ssm_in_cols);
            matmul_batch_tensor_qx(s->pool, cart, lw->ssm_out, projb,
                                   need_q8_ssm_out ? qx_h : NULL, aob, (int)B, ssm_in_bpr, s->fp32_ref);
            if (do_prof) t_mm += asi_now_ms() - tp;
            i_dn++;
        }
        else
        {
            if (s->calib_hook) s->calib_hook(s->calib_ctx, l, 0, xnb, B, H);
            tp = do_prof ? asi_now_ms() : 0.0;
            const AsiV2TensorEntry *ts[3] = {lw->attn_q, lw->attn_k, lw->attn_v};
            float *outs[3] = {qgb, kb, vb};
            matmul_batch_fused_q8k_pool(s->pool, cart, ts, outs, 3,
                                        s->fp32_ref ? NULL : qx_h, xnb, (int)B, bpr_h, 0, s->fp32_ref);
            if (do_prof) t_mm += asi_now_ms() - tp;

            Qwen35AttnKVState *kv = &s->kv_states[l];
            const float *qn_w = (const float *)tensor_data(cart, lw->attn_q_norm);
            const float *kn_w = (const float *)tensor_data(cart, lw->attn_k_norm);
            uint32_t n_heads = c->n_heads;
            uint32_t n_kv = c->n_kv_heads;
            uint32_t hd = c->head_dim;
            uint32_t n_rep = n_heads / n_kv;
            float scale = 1.0f / sqrtf((float)hd);

            tp = do_prof ? asi_now_ms() : 0.0;
            AsiKvPrepBatchCtx kctx;
            memset(&kctx, 0, sizeof(kctx));
            kctx.kb = kb;
            kctx.vb = vb;
            kctx.k_cache = kv->k_cache;
            kctx.v_cache = kv->v_cache;
            kctx.kn_w = kn_w;
            kctx.rope_inv_freq = s->rope_inv_freq;
            kctx.B = B;
            kctx.pos_start = pos_start;
            kctx.n_kv = n_kv;
            kctx.hd = hd;
            kctx.kv_dim = kv_dim;
            kctx.rope_dim = c->rope_dim;
            kctx.rope_theta = c->rope_theta;
            kctx.rms_eps = c->rms_norm_eps;
            kctx.mrope_pos_3d = mrope_pos_3d;
            kctx.mrope_flags = mrope_flags;
            kctx.rope_pos_delta = s->mrope_pos_delta;
            if (mrope_sections)
            {
                kctx.mrope_sections[0] = mrope_sections[0];
                kctx.mrope_sections[1] = mrope_sections[1];
                kctx.mrope_sections[2] = mrope_sections[2];
            }
            if (s->pool && B >= 8)
                asi_pool_run(s->pool, asi_kv_prep_batch_worker, &kctx);
            else
                asi_kv_prep_batch_worker(0, 1, &kctx);

            AsiFullAttnBatchCtx fctx;
            memset(&fctx, 0, sizeof(fctx));
            fctx.qgb = qgb;
            fctx.aob = aob;
            fctx.k_cache = kv->k_cache;
            fctx.v_cache = kv->v_cache;
            fctx.qn_w = qn_w;
            fctx.rope_inv_freq = s->rope_inv_freq;
            fctx.thread_scores_blk = s->scratch.thread_scores_blk;
            fctx.thread_scores_cap = s->scratch.thread_scores_cap;
            fctx.B = B;
            fctx.pos_start = pos_start;
            fctx.n_heads = n_heads;
            fctx.n_rep = n_rep;
            fctx.hd = hd;
            fctx.qg_dim = qg_dim;
            fctx.kv_dim = kv_dim;
            fctx.attn_dim = attn_dim;
            fctx.rope_dim = c->rope_dim;
            fctx.rope_theta = c->rope_theta;
            fctx.rms_eps = c->rms_norm_eps;
            fctx.scale = scale;
            fctx.mrope_pos_3d = mrope_pos_3d;
            fctx.mrope_flags = mrope_flags;
            fctx.rope_pos_delta = s->mrope_pos_delta;
            memcpy(fctx.mrope_sections, kctx.mrope_sections, sizeof(fctx.mrope_sections));
            atomic_init(&fctx.next_h, 0);
            asi_pool_run(s->pool, asi_full_attn_batch_worker, &fctx);
            if (do_prof) t_attn += asi_now_ms() - tp;
            tp = do_prof ? asi_now_ms() : 0.0;
            int attn_in_cols = (int)lw->attn_output->shape[0];
            int attn_in_bpr = (attn_in_cols + QK_K - 1) / QK_K;
            if (s->calib_hook) s->calib_hook(s->calib_ctx, l, 1, aob, B, (uint32_t)attn_in_cols);
            int need_q8_attn_out = (!s->fp32_ref && asi_tensor_needs_q8k(lw->attn_output) &&
                                    !(use_blis && asi_tensor_blis_supported(lw->attn_output)));
            if (need_q8_attn_out) quantize_batch_q8k_pool(s->pool, qx_h, aob, (int)B, attn_in_cols);
            matmul_batch_tensor_qx(s->pool, cart, lw->attn_output, projb,
                                   need_q8_attn_out ? qx_h : NULL, aob, (int)B, attn_in_bpr, s->fp32_ref);
            if (do_prof) t_mm += asi_now_ms() - tp;
        }

        tp = do_prof ? asi_now_ms() : 0.0;
        const float *pn_w = (const float *)tensor_data(cart, lw->post_attn_norm);
        int need_q8_ffn = (!s->fp32_ref && asi_tensor_needs_q8k(lw->ffn_gate) &&
                           !(use_blis && asi_tensor_blis_supported(lw->ffn_gate)));
        asi_norm_quant_pool(s->pool, xb, projb, xnb, qx_h, pn_w, B, H, c->rms_norm_eps, 1, 1, need_q8_ffn);
        if (do_prof) t_norm += asi_now_ms() - tp;
        tp = do_prof ? asi_now_ms() : 0.0;
        const AsiV2TensorEntry *ffn_ts[2] = {lw->ffn_gate, lw->ffn_up};
        float *ffn_outs[2] = {fgb, fub};
        if (s->calib_hook) s->calib_hook(s->calib_ctx, l, 2, xnb, B, H);
        matmul_batch_fused_q8k_pool(s->pool, cart, ffn_ts, ffn_outs, 2,
                                    need_q8_ffn ? qx_h : NULL, xnb, (int)B, bpr_h, 1, s->fp32_ref);
        if (do_prof) t_mm += asi_now_ms() - tp;
        tp = do_prof ? asi_now_ms() : 0.0;
        int ffn_in_bpr = ((int)I + QK_K - 1) / QK_K;
        if (s->calib_hook) s->calib_hook(s->calib_ctx, l, 3, fgb, B, I);
        int need_q8_down = (!s->fp32_ref && asi_tensor_needs_q8k(lw->ffn_down) &&
                            !(use_blis && asi_tensor_blis_supported(lw->ffn_down)));
        if (spec_mode && !s->fp32_ref && lw->ffn_down->quant_type == ASI_QUANT_Q4_REPACK &&
            (lw->ffn_down->layout_flags & ASI_TENSOR_LAYOUT_COL_MAJOR))
        {
            for (uint32_t tok = 0; tok < B; ++tok)
            {
                matvec_q4_col_major_pool(s->pool, cart, lw->ffn_down,
                                         projb + (size_t)tok * H,
                                         fgb + (size_t)tok * I,
                                         s->ffn_sparse_keep);
            }
        }
        else
        {
            if (need_q8_down) quantize_batch_q8k_pool(s->pool, qx_h, fgb, (int)B, (int)I);
            matmul_batch_tensor_qx(s->pool, cart, lw->ffn_down, projb,
                                   need_q8_down ? qx_h : NULL, fgb, (int)B, ffn_in_bpr, s->fp32_ref);
        }
        if (do_prof) t_mm += asi_now_ms() - tp;
        tp = do_prof ? asi_now_ms() : 0.0;
        const float *next_an = (l + 1 < c->n_layers) ? (const float *)tensor_data(cart, s->layers[l + 1].attn_norm) : NULL;
        int need_q8_next = 0;
        if (next_an && !s->fp32_ref)
        {
            const AsiV2TensorEntry *nl_t = (s->layers[l + 1].kind == QWEN35_LAYER_DELTANET)
                                               ? s->layers[l + 1].attn_qkv
                                               : s->layers[l + 1].attn_q;
            need_q8_next = asi_tensor_needs_q8k(nl_t) && !(use_blis && asi_tensor_blis_supported(nl_t));
        }
        asi_norm_quant_pool(s->pool, xb, projb, xnb, qx_h, next_an, B, H, c->rms_norm_eps,
                            1, (next_an != NULL), need_q8_next);
        if (do_prof) t_norm += asi_now_ms() - tp;
    }

    int rc = 0;
    if (compute_logits && out_last_logits)
    {
        double tp = do_prof ? asi_now_ms() : 0.0;
        const float *on_w = (const float *)tensor_data(cart, s->output_norm);
        const AsiV2TensorEntry *lm_head = s->output_weight ? s->output_weight : s->token_embd;
        if (!spec_mode)
        {
            rmsnorm(s->scratch.x_norm, xb + (size_t)(B - 1) * H, on_w, H, c->rms_norm_eps);
            rc = (s->lmhead_prefill && s->lmhead_idx && s->lmhead_ws)
                     ? lila_lhc_forward_pool(s->lmhead_idx, s->lmhead_ws, lhc_asi_dispatch, s->pool,
                                             tensor_data(cart, lm_head), s->scratch.x_norm, out_last_logits, s->fp32_ref)
                     : matvec_tensor_pool(s->pool, cart, lm_head, out_last_logits, s->scratch.x_norm, s->fp32_ref);
        }
        else if (s->lmhead_idx && s->lmhead_ws && B <= 4u)
        {
            int need_qx = (!s->fp32_ref &&
                           (lm_head->quant_type == 12u ||
                            (lm_head->quant_type == 14u &&
                             !(lm_head->layout_flags & ASI_TENSOR_LAYOUT_AVX2_REPACK))));
            asi_norm_quant_pool(s->pool, xb, NULL, xnb, need_qx ? qx_h : NULL, on_w, B, H,
                                c->rms_norm_eps, 0, 1, need_qx);
            rc = lila_lhc_forward_spec_pool(s->lmhead_idx, s->lmhead_ws, lhc_asi_dispatch, s->pool,
                                            tensor_data(cart, lm_head), xnb,
                                            need_qx ? qx_h : NULL, B,
                                            out_last_logits, out_spec_logits, s->fp32_ref);
        }
        else if (!s->fp32_ref && lm_head->quant_type == 14 &&
                 !(lm_head->layout_flags & ASI_TENSOR_LAYOUT_AVX2_REPACK))
        {
            asi_norm_quant_pool(s->pool, xb, NULL, xnb, qx_h, on_w, B, H, c->rms_norm_eps, 0, 1, 1);
            rc = matmul_batch_q8k_split_pool(s->pool, cart, lm_head, out_last_logits, out_spec_logits,
                                             qx_h, (int)B, bpr_h);
        }
        else
        {
            uint32_t V = c->vocab_size;
            asi_norm_quant_pool(s->pool, xb, NULL, xnb, NULL, on_w, B, H, c->rms_norm_eps, 0, 1, 0);
            if (!s->spec_tmp_logits && B <= (1u + s->spec_cap_k))
            {
                s->spec_tmp_logits = (float *)malloc((size_t)(1u + s->spec_cap_k) * V * sizeof(float));
            }
            if (!s->spec_tmp_logits)
            {
                rc = -1;
            }
            else
            {
                rc = matmul_batch_tensor(s->pool, cart, lm_head, s->spec_tmp_logits, xnb, (int)B, s->fp32_ref);
                if (rc == 0)
                {
                    memcpy(out_last_logits, s->spec_tmp_logits, (size_t)V * sizeof(float));
                    if (B > 1u && out_spec_logits)
                        memcpy(out_spec_logits, s->spec_tmp_logits + V, (size_t)(B - 1u) * V * sizeof(float));
                }
            }
        }
        if (do_prof) t_lm += asi_now_ms() - tp;
    }

    if (do_prof)
    {
        double tc1 = asi_now_ms();
        fprintf(stderr, "[PROF chunk B=%u pos=%u spec=%d] total=%.2f ms (mm=%.2f conv=%.2f ssm=%.2f attn=%.2f norm=%.2f lm=%.2f)\n",
                B, pos_start, spec_mode, tc1 - tc0, t_mm, t_conv, t_ssm, t_attn, t_norm, t_lm);
    }
    return rc;
}

static int qwen35_forward_chunk(AsiArchSession *s,
                                const int32_t *token_ids,
                                uint32_t B,
                                uint32_t pos_start,
                                int compute_logits,
                                float *out_last_logits)
{
    return qwen35_forward_chunk_ex(s, token_ids, B, pos_start, compute_logits, 0, out_last_logits, NULL,
                                   NULL, NULL, NULL, 0u, NULL);
}

static int32_t asi_argmax_f32(const float *v, uint32_t n)
{
    uint32_t best = 0;
    float best_val = v[0];
    for (uint32_t i = 1; i < n; ++i)
    {
        if (v[i] > best_val)
        {
            best_val = v[i];
            best = i;
        }
    }
    return (int32_t)best;
}

static int ensure_spec_buffers(AsiArchSession *s)
{
    if (!s) return -1;
    const AsiV2ModelConfig *c = &s->cart->config;
    uint32_t n_dn = 0;
    for (uint32_t l = 0; l < c->n_layers; ++l)
    {
        if (s->layers[l].kind == QWEN35_LAYER_DELTANET) n_dn++;
    }
    uint32_t cap_k = (uint32_t)s->spec_draft_len;
    if (cap_k == 0 || cap_k > ASI_SPEC_MAX_K) cap_k = ASI_SPEC_MAX_K;

    if (s->spec_logits && s->spec_cap_k == cap_k && s->spec_n_dn == n_dn && s->fail_hash)
        return 0;

    uint32_t qkv_dim = c->ssm_inner_size * 3;
    uint32_t ssm_r = c->ssm_time_step_rank;
    size_t l_elems = (size_t)cap_k * c->vocab_size;
    size_t qkv_elems = (size_t)cap_k * n_dn * qkv_dim;
    size_t ab_elems = (size_t)cap_k * n_dn * ssm_r;

    float *new_logits = (float *)malloc(l_elems * sizeof(float));
    float *new_qkv = qkv_elems ? (float *)malloc(qkv_elems * sizeof(float)) : NULL;
    float *new_a = ab_elems ? (float *)malloc(ab_elems * sizeof(float)) : NULL;
    float *new_b = ab_elems ? (float *)malloc(ab_elems * sizeof(float)) : NULL;
    AsiSpecFailEntry *new_fail = s->fail_hash ? s->fail_hash :
                                 (AsiSpecFailEntry *)calloc(ASI_SPEC_FAIL_CAP, sizeof(AsiSpecFailEntry));

    if (!new_logits || (qkv_elems && !new_qkv) || (ab_elems && (!new_a || !new_b)) || !new_fail)
    {
        free(new_logits);
        free(new_qkv);
        free(new_a);
        free(new_b);
        if (!s->fail_hash) free(new_fail);
        return -1;
    }

    free(s->spec_logits);
    free(s->spec_saved_qkv);
    free(s->spec_saved_a);
    free(s->spec_saved_b);
    s->spec_logits = new_logits;
    s->spec_saved_qkv = new_qkv;
    s->spec_saved_a = new_a;
    s->spec_saved_b = new_b;
    s->fail_hash = new_fail;
    s->spec_n_dn = n_dn;
    s->spec_cap_k = cap_k;
    return 0;
}

static inline uint32_t spec_fail_slot(int32_t prev_tok, int32_t cur_tok, int32_t cand_tok)
{
    uint32_t h = (uint32_t)prev_tok * 2654435761u;
    h ^= (uint32_t)cur_tok * 2246822519u;
    h ^= (uint32_t)cand_tok * 3266489917u;
    h ^= h >> 16;
    return h & (ASI_SPEC_FAIL_CAP - 1u);
}

static int spec_fail_has(const AsiArchSession *s, int32_t prev_tok, int32_t cur_tok, int32_t cand_tok)
{
    if (!s->fail_hash) return 0;
    uint32_t idx = spec_fail_slot(prev_tok, cur_tok, cand_tok);
    const AsiSpecFailEntry *e = &s->fail_hash[idx];
    return (e->gen == s->fail_gen && e->prev_tok == prev_tok && e->cur_tok == cur_tok && e->bad_next == cand_tok);
}

static void spec_fail_add(AsiArchSession *s, int32_t prev_tok, int32_t cur_tok, int32_t cand_tok)
{
    if (!s->fail_hash) return;
    uint32_t idx = spec_fail_slot(prev_tok, cur_tok, cand_tok);
    s->fail_hash[idx].prev_tok = prev_tok;
    s->fail_hash[idx].cur_tok = cur_tok;
    s->fail_hash[idx].bad_next = cand_tok;
    s->fail_hash[idx].gen = s->fail_gen;
}

static uint32_t qwen35_propose_draft(const AsiArchSession *s,
                                     int32_t token_id,
                                     uint32_t pos,
                                     int32_t *draft_out,
                                     uint32_t max_k)
{
    if (!s || !s->token_history || max_k == 0 || pos + 1u >= s->max_ctx) return 0;
    uint32_t avail_ctx = s->max_ctx - (pos + 1u);
    if (max_k > avail_ctx) max_k = avail_ctx;
    if (max_k == 0) return 0;

    uint32_t start = s->history_valid_from;
    if (pos <= start) return 0;
    const int32_t *hist = s->token_history;
    int32_t prev_tok = hist[pos - 1u];

    /* 1. Longest recent L >= 3 suffix match (L = 4, 3) with >= 1 following token */
    for (uint32_t L = 4u; L >= 3u; --L)
    {
        if (pos - start < L) continue;
        uint32_t m_min = start + L - 1u;
        for (uint32_t m = pos - 1u; m >= m_min; --m)
        {
            int match = 1;
            for (uint32_t d = 0; d < L; ++d)
            {
                if (hist[m - d] != hist[pos - d])
                {
                    match = 0;
                    break;
                }
            }
            if (match)
            {
                int32_t cand0 = hist[m + 1u];
                if (!spec_fail_has(s, prev_tok, token_id, cand0))
                {
                    uint32_t avail = pos - m;
                    uint32_t k = (avail < max_k) ? avail : max_k;
                    for (uint32_t j = 0; j < k; ++j)
                        draft_out[j] = hist[m + 1u + j];
                    return k;
                }
            }
            if (m == m_min) break;
        }
    }

    /* 2. Universal Qwen3.5 <think> (248068 -> \n\n </think> \n\n) */
    if (s->cart->config.vocab_size == 248320u && token_id == 248068 &&
        !spec_fail_has(s, prev_tok, token_id, 271))
    {
        uint32_t k = 0;
        if (k < max_k) draft_out[k++] = 271;
        if (k < max_k) draft_out[k++] = 248069;
        if (k < max_k) draft_out[k++] = 271;
        return k;
    }

    /* 3. High-precision L == 2 match: only if in prompt OR occ >= 2, and >= 2 following tokens */
    if (pos - start >= 2u)
    {
        uint32_t m_min = start + 1u;
        uint32_t occ = 0;
        uint32_t best_m = UINT32_MAX;
        for (uint32_t m = pos - 1u; m >= m_min; --m)
        {
            if (hist[m] == token_id && hist[m - 1u] == prev_tok)
            {
                occ++;
                if (best_m == UINT32_MAX && (pos - m) >= 2u &&
                    !spec_fail_has(s, prev_tok, token_id, hist[m + 1u]))
                {
                    best_m = m;
                }
                if (best_m != UINT32_MAX && (best_m < s->prompt_len || occ >= 2u))
                    break;
            }
            if (m == m_min) break;
        }
        if (best_m != UINT32_MAX && (best_m < s->prompt_len || occ >= 2u))
        {
            uint32_t avail = pos - best_m;
            uint32_t k = (avail < max_k) ? avail : max_k;
            for (uint32_t j = 0; j < k; ++j)
                draft_out[j] = hist[best_m + 1u + j];
            return k;
        }
    }

    return 0;
}

static void qwen35_advance_spec_deltanet(AsiArchSession *s, uint32_t spec_step)
{
    const AsiV2Cartridge *cart = s->cart;
    const AsiV2ModelConfig *c = &cart->config;
    uint32_t qkv_dim = c->ssm_inner_size * 3;
    uint32_t ssm_r = c->ssm_time_step_rank;
    uint32_t n_heads = c->ssm_group_count;
    uint32_t k_dim = c->ssm_state_size;
    uint32_t v_dim = c->ssm_inner_size / n_heads;
    uint32_t k_conv = c->ssm_conv_kernel;
    uint32_t conv_hist = (k_conv > 1) ? (k_conv - 1) : 0;
    float q_scale = 1.0f / sqrtf((float)k_dim);
    size_t cap_k = (size_t)s->spec_cap_k;

    asi_pool_begin(s->pool);
    uint32_t i_dn = 0;
    for (uint32_t l = 0; l < c->n_layers; ++l)
    {
        if (s->layers[l].kind != QWEN35_LAYER_DELTANET) continue;
        const Qwen35LayerWeights *lw = &s->layers[l];
        Qwen35DeltaNetState *st = &s->ssm_states[l];
        const float *conv_w = (const float *)tensor_data(cart, lw->ssm_conv1d);
        const float *ssm_a = (const float *)tensor_data(cart, lw->ssm_a);
        const float *dt_bias = (const float *)tensor_data(cart, lw->ssm_dt_bias);
        const float *ssm_norm_w = (const float *)tensor_data(cart, lw->ssm_norm);

        size_t qkv_off = ((size_t)i_dn * cap_k + spec_step) * qkv_dim;
        size_t ab_off  = ((size_t)i_dn * cap_k + spec_step) * ssm_r;
        memcpy(s->scratch.qkv, s->spec_saved_qkv + qkv_off, (size_t)qkv_dim * sizeof(float));

        AsiConv1dCtx cctx = {s->scratch.qkv, st->conv_state, conv_w, 1, qkv_dim, k_conv, conv_hist, 0};
        asi_pool_run(s->pool, asi_conv1d_worker, &cctx);

        AsiDeltaNetCtx dctx = {
            st->ssm_state, s->scratch.qkv, s->scratch.z,
            s->spec_saved_a + ab_off, s->spec_saved_b + ab_off, s->scratch.attn_out,
            ssm_a, dt_bias, ssm_norm_w,
            1, n_heads, k_dim, v_dim, c->ssm_inner_size, qkv_dim, ssm_r,
            q_scale, c->rms_norm_eps, 0, 1, ATOMIC_VAR_INIT(0)
        };
        asi_pool_run(s->pool, asi_deltanet_worker, &dctx);
        i_dn++;
    }
    asi_pool_end(s->pool);
}

static int qwen35_forward_speculative(AsiArchSession *s,
                                      int32_t token_id,
                                      uint32_t pos,
                                      float *out_logits)
{
    if (!s || !out_logits || token_id < 0 || pos >= s->max_ctx) return -1;
    const AsiV2ModelConfig *c = &s->cart->config;
    if ((uint32_t)token_id >= c->vocab_size) return -1;
    s->stat_eval_calls++;

    if (s->spec_cursor < s->spec_count &&
        pos == s->spec_start_pos + s->spec_cursor &&
        token_id == s->spec_tokens[s->spec_cursor])
    {
        extern int g_e05_hook_fast_flag;
        if (g_e05_hook_fast_flag)
        {
            /* Never serve cached unconstrained speculative logits while E05 grammar is active. */
            s->spec_count = 0;
            s->spec_cursor = 0;
            return qwen35_forward_token(s, token_id, pos, out_logits);
        }
        uint32_t step = s->spec_cursor;
        qwen35_advance_spec_deltanet(s, step);
        memcpy(out_logits, s->spec_logits + (size_t)step * c->vocab_size,
               (size_t)c->vocab_size * sizeof(float));
        s->spec_cursor++;
        if (s->spec_cursor >= s->spec_count)
        {
            s->spec_count = 0;
            s->spec_cursor = 0;
        }
        s->token_history[pos] = token_id;
        if (pos + 1u > s->history_len) s->history_len = pos + 1u;
        s->stat_cache_hits++;
        return 0;
    }

    /* Cache miss or mid-cache divergence: zero-rollback invalidation */
    s->spec_count = 0;
    s->spec_cursor = 0;

    if (pos < s->history_valid_from) s->history_valid_from = pos;
    s->token_history[pos] = token_id;
    if (pos + 1u > s->history_len) s->history_len = pos + 1u;

    extern int g_e05_hook_fast_flag;
    if (g_e05_hook_fast_flag || !s->speculative || s->spec_draft_len <= 0 ||
        ensure_batch_scratch(s, 1u + (uint32_t)s->spec_draft_len) != 0)
    {
        return qwen35_forward_token(s, token_id, pos, out_logits);
    }

    int32_t batch_toks[1 + ASI_SPEC_MAX_K];
    batch_toks[0] = token_id;
    uint32_t max_k = (uint32_t)s->spec_draft_len;
    if (max_k > ASI_SPEC_MAX_K) max_k = ASI_SPEC_MAX_K;
    if (s->has_q4_repack && max_k > 2u) max_k = 2u; /* E07 Exp 3C: Broadwell 16 YMM register cap (B <= 3) */
    if (ensure_spec_buffers(s) != 0)
    {
        return qwen35_forward_token(s, token_id, pos, out_logits);
    }

    int draft_kind = 0;
    uint32_t k = 0;
    if (s->spec_gate)
    {
        uint32_t start = s->history_valid_from;
        if (pos > start && pos + 1u < s->max_ctx)
        {
            uint32_t avail_ctx = s->max_ctx - (pos + 1u);
            uint32_t lim_k = (max_k < avail_ctx) ? max_k : avail_ctx;
            const int32_t *hist = s->token_history;
            int32_t prev_tok = hist[pos - 1u];
            if (s->cart->config.vocab_size == 248320u && token_id == 248068 &&
                !spec_fail_has(s, prev_tok, token_id, 271))
            {
                if (k < lim_k) batch_toks[1u + k++] = 271;
                if (k < lim_k) batch_toks[1u + k++] = 248069;
                if (k < lim_k) batch_toks[1u + k++] = 271;
                draft_kind = 1;
            }
            else if (s->spec_cooldown > 0u)
            {
                s->spec_cooldown--;
                s->stat_spec_gate_skips++;
            }
            else
            {
                for (uint32_t L = 4u; L >= 3u && k == 0u; --L)
                {
                    if (pos - start < L) continue;
                    uint32_t m_min = start + L - 1u;
                    for (uint32_t m = pos - 1u; m >= m_min; --m)
                    {
                        int match = 1;
                        for (uint32_t d = 0; d < L; ++d)
                        {
                            if (hist[m - d] != hist[pos - d]) { match = 0; break; }
                        }
                        if (match && (pos - m) >= 2u)
                        {
                            int32_t cand0 = hist[m + 1u];
                            if (!spec_fail_has(s, prev_tok, token_id, cand0))
                            {
                                uint32_t avail = pos - m;
                                uint32_t take = (avail < lim_k) ? avail : lim_k;
                                for (uint32_t j = 0; j < take; ++j)
                                    batch_toks[1u + j] = hist[m + 1u + j];
                                k = take;
                                draft_kind = 2;
                                break;
                            }
                        }
                        if (m == m_min) break;
                    }
                }
            }
        }
    }
    else
    {
        k = qwen35_propose_draft(s, token_id, pos, batch_toks + 1, max_k);
    }
    if (k == 0)
    {
        return qwen35_forward_token(s, token_id, pos, out_logits);
    }

    s->stat_batches++;
    s->stat_drafted += k;

    asi_pool_begin(s->pool);
    int rc = qwen35_forward_chunk_ex(s, batch_toks, 1u + k, pos, 1, 1, out_logits, s->spec_logits,
                                     NULL, NULL, NULL, 0u, NULL);
    asi_pool_end(s->pool);
    if (rc != 0) return rc;

    uint32_t V = c->vocab_size;
    int32_t prev_tok = (pos > s->history_valid_from) ? s->token_history[pos - 1u] : -1;
    uint32_t m = 0;
    for (uint32_t j = 0; j < k; ++j)
    {
        const float *lg = (j == 0) ? out_logits : (s->spec_logits + (size_t)(j - 1u) * V);
        int32_t pred = asi_argmax_f32(lg, V);
        if (pred == batch_toks[1u + j])
        {
            s->spec_tokens[j] = batch_toks[1u + j];
            m++;
        }
        else
        {
            int32_t ctx_prev = (j == 0) ? prev_tok : batch_toks[j - 1u];
            int32_t ctx_cur = batch_toks[j];
            spec_fail_add(s, ctx_prev, ctx_cur, batch_toks[1u + j]);
            s->stat_misses++;
            break;
        }
    }
    if (s->spec_gate && draft_kind == 2)
    {
        if (m == 0u)
        {
            s->spec_consec_misses++;
            s->spec_cooldown = (s->spec_consec_misses >= 4u) ? 16u : (1u << s->spec_consec_misses);
        }
        else if (m < 2u && k >= 2u)
        {
            s->spec_consec_misses++;
            s->spec_cooldown = (s->spec_consec_misses >= 4u) ? 8u : (1u << (s->spec_consec_misses - 1u));
        }
        else
        {
            s->spec_consec_misses = 0u;
            s->spec_cooldown = 0u;
        }
    }
    s->stat_accepted += m;
    s->spec_start_pos = pos + 1u;
    s->spec_count = m;
    s->spec_cursor = 0;
    return 0;
}

static int qwen35_forward_batch(AsiArchSession *s,
                                const int32_t *token_ids,
                                uint32_t n_tokens,
                                uint32_t pos_start,
                                float *out_last_logits)
{
    if (!s || !token_ids || n_tokens == 0 || !out_last_logits) return -1;
    if (pos_start + n_tokens > s->max_ctx) return -1;
    s->spec_count = 0;
    s->spec_cursor = 0;
    if (pos_start < s->kv_dirty_from) s->kv_dirty_from = pos_start;
    if (pos_start < s->history_valid_from) s->history_valid_from = pos_start;
    if (pos_start == 0)
    {
        s->mrope_pos_delta = 0;
        s->prompt_len = n_tokens;
        s->history_valid_from = 0;
    }
    else if (pos_start <= s->prompt_len && pos_start + n_tokens > s->prompt_len)
    {
        s->prompt_len = pos_start + n_tokens;
    }
    if (s->token_history)
    {
        memcpy(s->token_history + pos_start, token_ids, (size_t)n_tokens * sizeof(int32_t));
        if (pos_start + n_tokens > s->history_len) s->history_len = pos_start + n_tokens;
    }
    if (n_tokens == 1)
    {
        return qwen35_forward_token(s, token_ids[0], pos_start, out_last_logits);
    }

    const uint32_t chunk_max = (uint32_t)asi_runtime_config()->batch_chunk_max;
    uint32_t done = 0;
    asi_pool_begin(s->pool);
    while (done < n_tokens)
    {
        uint32_t B = n_tokens - done;
        if (B > chunk_max) B = chunk_max;
        int is_last = (done + B == n_tokens);
        if (qwen35_forward_chunk(s, token_ids + done, B, pos_start + done,
                                 is_last, out_last_logits) != 0)
        {
            asi_pool_end(s->pool);
            return -1;
        }
        done += B;
    }
    asi_pool_end(s->pool);
    return 0;
}

static int qwen35_save_prefix(AsiArchSession *s, uint32_t n_tokens)
{
    if (!s || n_tokens == 0 || n_tokens > s->max_ctx) return -1;
    const AsiV2ModelConfig *c = &s->cart->config;
    uint32_t qkv_dim = c->ssm_inner_size * 3;
    uint32_t conv_hist = (c->ssm_conv_kernel > 1) ? (c->ssm_conv_kernel - 1) : 0;
    size_t conv_per_layer = (size_t)conv_hist * qkv_dim;
    uint32_t ssm_head_dim = (c->ssm_group_count > 0) ? (c->ssm_inner_size / c->ssm_group_count) : 0;
    size_t ssm_per_layer = (size_t)c->ssm_group_count * c->ssm_state_size * ssm_head_dim;
    size_t kv_dim = (size_t)c->n_kv_heads * c->head_dim;
    size_t kv_per_layer = (size_t)n_tokens * kv_dim;

    uint32_t n_dn = 0, n_fa = 0;
    for (uint32_t l = 0; l < c->n_layers; ++l)
    {
        if (s->layers[l].kind == QWEN35_LAYER_DELTANET) n_dn++; else n_fa++;
    }

    free(s->prefix_conv); free(s->prefix_ssm); free(s->prefix_k); free(s->prefix_v);
    s->prefix_conv = (n_dn && conv_per_layer) ? (float *)malloc((size_t)n_dn * conv_per_layer * sizeof(float)) : NULL;
    s->prefix_ssm  = (n_dn && ssm_per_layer)  ? (float *)malloc((size_t)n_dn * ssm_per_layer * sizeof(float)) : NULL;
    s->prefix_k    = (n_fa && kv_per_layer)   ? (uint16_t *)malloc((size_t)n_fa * kv_per_layer * sizeof(uint16_t)) : NULL;
    s->prefix_v    = (n_fa && kv_per_layer)   ? (uint16_t *)malloc((size_t)n_fa * kv_per_layer * sizeof(uint16_t)) : NULL;
    if ((n_dn && conv_per_layer && !s->prefix_conv) || (n_dn && ssm_per_layer && !s->prefix_ssm) ||
        (n_fa && kv_per_layer && (!s->prefix_k || !s->prefix_v)))
    {
        free(s->prefix_conv); free(s->prefix_ssm); free(s->prefix_k); free(s->prefix_v);
        s->prefix_conv = s->prefix_ssm = NULL; s->prefix_k = s->prefix_v = NULL;
        s->prefix_len = 0;
        return -1;
    }

    uint32_t i_dn = 0, i_fa = 0;
    for (uint32_t l = 0; l < c->n_layers; ++l)
    {
        if (s->layers[l].kind == QWEN35_LAYER_DELTANET)
        {
            if (conv_per_layer)
                memcpy(s->prefix_conv + (size_t)i_dn * conv_per_layer, s->ssm_states[l].conv_state, conv_per_layer * sizeof(float));
            if (ssm_per_layer)
                memcpy(s->prefix_ssm + (size_t)i_dn * ssm_per_layer, s->ssm_states[l].ssm_state, ssm_per_layer * sizeof(float));
            i_dn++;
        }
        else
        {
            if (kv_per_layer)
            {
                memcpy(s->prefix_k + (size_t)i_fa * kv_per_layer, s->kv_states[l].k_cache, kv_per_layer * sizeof(uint16_t));
                memcpy(s->prefix_v + (size_t)i_fa * kv_per_layer, s->kv_states[l].v_cache, kv_per_layer * sizeof(uint16_t));
            }
            i_fa++;
        }
    }
    s->prefix_len = n_tokens;
    s->prefix_mrope_delta = s->mrope_pos_delta;
    free(s->prefix_tokens);
    s->prefix_tokens = NULL;
    s->prefix_has_tokens = 0;
    if (s->token_history && s->history_valid_from == 0 && s->history_len >= n_tokens)
    {
        s->prefix_tokens = (int32_t *)malloc((size_t)n_tokens * sizeof(int32_t));
        if (s->prefix_tokens)
        {
            memcpy(s->prefix_tokens, s->token_history, (size_t)n_tokens * sizeof(int32_t));
            s->prefix_has_tokens = 1;
        }
    }
    return 0;
}

static int qwen35_restore_prefix(AsiArchSession *s, uint32_t *out_n_tokens)
{
    if (!s || s->prefix_len == 0) return -1;
    const AsiV2ModelConfig *c = &s->cart->config;
    uint32_t qkv_dim = c->ssm_inner_size * 3;
    uint32_t conv_hist = (c->ssm_conv_kernel > 1) ? (c->ssm_conv_kernel - 1) : 0;
    size_t conv_per_layer = (size_t)conv_hist * qkv_dim;
    uint32_t ssm_head_dim = (c->ssm_group_count > 0) ? (c->ssm_inner_size / c->ssm_group_count) : 0;
    size_t ssm_per_layer = (size_t)c->ssm_group_count * c->ssm_state_size * ssm_head_dim;
    size_t kv_dim = (size_t)c->n_kv_heads * c->head_dim;
    size_t kv_per_layer = (size_t)s->prefix_len * kv_dim;

    uint32_t i_dn = 0, i_fa = 0;
    for (uint32_t l = 0; l < c->n_layers; ++l)
    {
        if (s->layers[l].kind == QWEN35_LAYER_DELTANET)
        {
            if (conv_per_layer && s->prefix_conv)
                memcpy(s->ssm_states[l].conv_state, s->prefix_conv + (size_t)i_dn * conv_per_layer, conv_per_layer * sizeof(float));
            if (ssm_per_layer && s->prefix_ssm)
                memcpy(s->ssm_states[l].ssm_state, s->prefix_ssm + (size_t)i_dn * ssm_per_layer, ssm_per_layer * sizeof(float));
            i_dn++;
        }
        else
        {
            if (kv_per_layer && s->prefix_k && s->prefix_v)
            {
                memcpy(s->kv_states[l].k_cache, s->prefix_k + (size_t)i_fa * kv_per_layer, kv_per_layer * sizeof(uint16_t));
                memcpy(s->kv_states[l].v_cache, s->prefix_v + (size_t)i_fa * kv_per_layer, kv_per_layer * sizeof(uint16_t));
            }
            i_fa++;
        }
    }
    s->spec_count = 0;
    s->spec_cursor = 0;
    s->spec_start_pos = 0;
    s->spec_consec_misses = 0;
    s->spec_cooldown = 0;
    s->mrope_pos_delta = s->prefix_mrope_delta;
    s->fail_gen++;
    if (s->fail_gen == 0)
    {
        if (s->fail_hash) memset(s->fail_hash, 0, ASI_SPEC_FAIL_CAP * sizeof(AsiSpecFailEntry));
        s->fail_gen = 1;
    }
    s->prompt_len = s->prefix_len;
    s->history_len = s->prefix_len;
    if (s->prefix_has_tokens && s->prefix_tokens && s->token_history)
    {
        memcpy(s->token_history, s->prefix_tokens, (size_t)s->prefix_len * sizeof(int32_t));
        s->history_valid_from = 0;
    }
    else
    {
        s->history_valid_from = s->prefix_len;
    }
    if (out_n_tokens) *out_n_tokens = s->prefix_len;
    return 0;
}

typedef struct {
    char magic[8];
    uint32_t numerics_version;
    uint32_t arch_id;
    uint8_t cart_identity_hash[24];
    uint8_t key_hash[32];
    uint32_t fp32_ref;
    uint32_t prefix_len;
    uint32_t n_dn;
    uint32_t n_fa;
    uint64_t conv_bytes;
    uint64_t ssm_bytes;
    uint64_t k_bytes;
    uint64_t v_bytes;
    uint64_t reserved;
    uint64_t payload_crc64;
    uint64_t header_crc64;
} AsiPrefixSnapshotHeader;

_Static_assert(sizeof(AsiPrefixSnapshotHeader) == 144, "AsiPrefixSnapshotHeader must be 144 bytes");
_Static_assert(offsetof(AsiPrefixSnapshotHeader, header_crc64) == 136, "header_crc64 offset mismatch");

static uint64_t g_asi_pfx_crc64_tab[256];
static AsiOnce g_asi_pfx_crc64_once = ASI_ONCE_INIT;

static void asi_pfx_crc64_init(void)
{
    for (uint64_t i = 0; i < 256; ++i)
    {
        uint64_t crc = i << 56;
        for (int b = 0; b < 8; ++b)
        {
            if (crc & 0x8000000000000000ULL)
                crc = (crc << 1) ^ 0x42F0E1EBA9EA3693ULL;
            else
                crc <<= 1;
        }
        g_asi_pfx_crc64_tab[i] = crc;
    }
}

static uint64_t asi_crc64_update(uint64_t crc, const uint8_t *data, size_t len)
{
    asi_call_once(&g_asi_pfx_crc64_once, asi_pfx_crc64_init);
    for (size_t i = 0; i < len; ++i)
    {
        crc = (crc << 8) ^ g_asi_pfx_crc64_tab[((crc >> 56) ^ data[i]) & 0xFFu];
    }
    return crc;
}

static void qwen35_prefix_geometry(const AsiArchSession *s,
                                   uint32_t plen,
                                   uint32_t *out_n_dn,
                                   uint32_t *out_n_fa,
                                   uint64_t *out_conv_bytes,
                                   uint64_t *out_ssm_bytes,
                                   uint64_t *out_k_bytes,
                                   uint64_t *out_v_bytes)
{
    const AsiV2ModelConfig *c = &s->cart->config;
    uint32_t qkv_dim = c->ssm_inner_size * 3u;
    uint32_t conv_hist = (c->ssm_conv_kernel > 1u) ? (c->ssm_conv_kernel - 1u) : 0u;
    size_t conv_per_layer = (size_t)conv_hist * qkv_dim;
    uint32_t ssm_head_dim = (c->ssm_group_count > 0u) ? (c->ssm_inner_size / c->ssm_group_count) : 0u;
    size_t ssm_per_layer = (size_t)c->ssm_group_count * c->ssm_state_size * ssm_head_dim;
    size_t kv_dim = (size_t)c->n_kv_heads * c->head_dim;
    size_t kv_per_layer = (size_t)plen * kv_dim;

    uint32_t n_dn = 0, n_fa = 0;
    for (uint32_t l = 0; l < c->n_layers; ++l)
    {
        if (s->layers[l].kind == QWEN35_LAYER_DELTANET) n_dn++; else n_fa++;
    }
    *out_n_dn = n_dn;
    *out_n_fa = n_fa;
    *out_conv_bytes = (uint64_t)n_dn * (uint64_t)conv_per_layer * sizeof(float);
    *out_ssm_bytes  = (uint64_t)n_dn * (uint64_t)ssm_per_layer * sizeof(float);
    *out_k_bytes    = (uint64_t)n_fa * (uint64_t)kv_per_layer * sizeof(uint16_t);
    *out_v_bytes    = (uint64_t)n_fa * (uint64_t)kv_per_layer * sizeof(uint16_t);
}

static int qwen35_save_prefix_file(const AsiArchSession *s,
                                   const uint8_t key_hash[32],
                                   const char *filepath)
{
    if (!s || s->prefix_len == 0 || !key_hash || !filepath || filepath[0] == '\0') return -1;
    uint32_t n_dn = 0, n_fa = 0;
    uint64_t conv_bytes = 0, ssm_bytes = 0, k_bytes = 0, v_bytes = 0;
    qwen35_prefix_geometry(s, s->prefix_len, &n_dn, &n_fa, &conv_bytes, &ssm_bytes, &k_bytes, &v_bytes);

    if ((conv_bytes && !s->prefix_conv) || (ssm_bytes && !s->prefix_ssm) ||
        (k_bytes && !s->prefix_k) || (v_bytes && !s->prefix_v))
    {
        return -1;
    }

    AsiPrefixSnapshotHeader hdr;
    memset(&hdr, 0, sizeof(hdr));
    memcpy(hdr.magic, "ASIPFX02", 8);
    hdr.numerics_version = LILA_ASI_NUMERICS_VERSION;
    hdr.arch_id = ASI_ARCH_HYBRID_QWEN35;
    memcpy(hdr.cart_identity_hash, s->cart->header.identity_hash, 24);
    memcpy(hdr.key_hash, key_hash, 32);
    hdr.fp32_ref = (uint32_t)(s->fp32_ref ? 1u : 0u);
    hdr.prefix_len = s->prefix_len;
    hdr.n_dn = n_dn;
    hdr.n_fa = n_fa;
    hdr.conv_bytes = conv_bytes;
    hdr.ssm_bytes = ssm_bytes;
    hdr.k_bytes = k_bytes;
    hdr.v_bytes = v_bytes;
    hdr.reserved = (uint64_t)s->prefix_mrope_delta;

    uint64_t pcrc = 0;
    if (conv_bytes) pcrc = asi_crc64_update(pcrc, (const uint8_t *)s->prefix_conv, (size_t)conv_bytes);
    if (ssm_bytes)  pcrc = asi_crc64_update(pcrc, (const uint8_t *)s->prefix_ssm,  (size_t)ssm_bytes);
    if (k_bytes)    pcrc = asi_crc64_update(pcrc, (const uint8_t *)s->prefix_k,    (size_t)k_bytes);
    if (v_bytes)    pcrc = asi_crc64_update(pcrc, (const uint8_t *)s->prefix_v,    (size_t)v_bytes);
    hdr.payload_crc64 = pcrc;
    hdr.header_crc64 = asi_v2_crc64((const uint8_t *)&hdr, offsetof(AsiPrefixSnapshotHeader, header_crc64));

    size_t flen = strlen(filepath);
    size_t tcap = flen + 64;
    char *tmp_path = (char *)malloc(tcap);
    if (!tmp_path) return -1;
    int tn = snprintf(tmp_path, tcap, "%s.tmp.%ld", filepath, asi_getpid());
    if (tn <= 0 || (size_t)tn >= tcap)
    {
        free(tmp_path);
        return -1;
    }

    FILE *fp = fopen(tmp_path, "wb");
    if (!fp)
    {
        free(tmp_path);
        return -1;
    }

    if (fwrite(&hdr, 1, sizeof(hdr), fp) != sizeof(hdr) ||
        (conv_bytes > 0 && fwrite(s->prefix_conv, 1, (size_t)conv_bytes, fp) != (size_t)conv_bytes) ||
        (ssm_bytes > 0  && fwrite(s->prefix_ssm,  1, (size_t)ssm_bytes,  fp) != (size_t)ssm_bytes) ||
        (k_bytes > 0    && fwrite(s->prefix_k,    1, (size_t)k_bytes,    fp) != (size_t)k_bytes) ||
        (v_bytes > 0    && fwrite(s->prefix_v,    1, (size_t)v_bytes,    fp) != (size_t)v_bytes))
    {
        fclose(fp);
        asi_unlink(tmp_path);
        free(tmp_path);
        return -1;
    }

    if (fflush(fp) != 0)
    {
        fclose(fp);
        asi_unlink(tmp_path);
        free(tmp_path);
        return -1;
    }
    if (asi_fsync_file(fp) != 0)
    {
        fclose(fp);
        asi_unlink(tmp_path);
        free(tmp_path);
        return -1;
    }
    if (fclose(fp) != 0)
    {
        asi_unlink(tmp_path);
        free(tmp_path);
        return -1;
    }
    if (asi_atomic_rename(tmp_path, filepath) != 0)
    {
        asi_unlink(tmp_path);
        free(tmp_path);
        return -1;
    }
    free(tmp_path);
    return 0;
}

static int qwen35_load_prefix_file(AsiArchSession *s,
                                   const uint8_t expected_key_hash[32],
                                   const char *filepath,
                                   uint32_t *out_n_tokens)
{
    if (!s || !expected_key_hash || !filepath || filepath[0] == '\0') return -1;

    FILE *fp = fopen(filepath, "rb");
    if (!fp) return -1;

    AsiPrefixSnapshotHeader hdr;
    float *new_conv = NULL;
    float *new_ssm = NULL;
    uint16_t *new_k = NULL;
    uint16_t *new_v = NULL;

    if (fread(&hdr, 1, sizeof(hdr), fp) != sizeof(hdr)) goto fail;
    if (memcmp(hdr.magic, "ASIPFX02", 8) != 0) goto fail;

    uint64_t expected_hcrc = asi_v2_crc64((const uint8_t *)&hdr, offsetof(AsiPrefixSnapshotHeader, header_crc64));
    if (hdr.header_crc64 != expected_hcrc) goto fail;
    if (hdr.numerics_version != LILA_ASI_NUMERICS_VERSION) goto fail;
    if (hdr.arch_id != ASI_ARCH_HYBRID_QWEN35) goto fail;
    if (memcmp(hdr.cart_identity_hash, s->cart->header.identity_hash, 24) != 0) goto fail;
    if (memcmp(hdr.key_hash, expected_key_hash, 32) != 0) goto fail;
    if (hdr.fp32_ref != (uint32_t)(s->fp32_ref ? 1u : 0u)) goto fail;
    if (hdr.prefix_len == 0 || hdr.prefix_len > s->max_ctx) goto fail;

    uint32_t exp_dn = 0, exp_fa = 0;
    uint64_t exp_conv = 0, exp_ssm = 0, exp_k = 0, exp_v = 0;
    qwen35_prefix_geometry(s, hdr.prefix_len, &exp_dn, &exp_fa, &exp_conv, &exp_ssm, &exp_k, &exp_v);
    if (hdr.n_dn != exp_dn || hdr.n_fa != exp_fa ||
        hdr.conv_bytes != exp_conv || hdr.ssm_bytes != exp_ssm ||
        hdr.k_bytes != exp_k || hdr.v_bytes != exp_v)
    {
        goto fail;
    }

    if (exp_conv > 0)
    {
        new_conv = (float *)malloc((size_t)exp_conv);
        if (!new_conv || fread(new_conv, 1, (size_t)exp_conv, fp) != (size_t)exp_conv) goto fail;
    }
    if (exp_ssm > 0)
    {
        new_ssm = (float *)malloc((size_t)exp_ssm);
        if (!new_ssm || fread(new_ssm, 1, (size_t)exp_ssm, fp) != (size_t)exp_ssm) goto fail;
    }
    if (exp_k > 0)
    {
        new_k = (uint16_t *)malloc((size_t)exp_k);
        if (!new_k || fread(new_k, 1, (size_t)exp_k, fp) != (size_t)exp_k) goto fail;
    }
    if (exp_v > 0)
    {
        new_v = (uint16_t *)malloc((size_t)exp_v);
        if (!new_v || fread(new_v, 1, (size_t)exp_v, fp) != (size_t)exp_v) goto fail;
    }

    uint8_t extra = 0;
    if (fread(&extra, 1, 1, fp) != 0 || ferror(fp)) goto fail;
    fclose(fp);
    fp = NULL;

    uint64_t pcrc = 0;
    if (exp_conv > 0) pcrc = asi_crc64_update(pcrc, (const uint8_t *)new_conv, (size_t)exp_conv);
    if (exp_ssm > 0)  pcrc = asi_crc64_update(pcrc, (const uint8_t *)new_ssm,  (size_t)exp_ssm);
    if (exp_k > 0)    pcrc = asi_crc64_update(pcrc, (const uint8_t *)new_k,    (size_t)exp_k);
    if (exp_v > 0)    pcrc = asi_crc64_update(pcrc, (const uint8_t *)new_v,    (size_t)exp_v);
    if (pcrc != hdr.payload_crc64) goto fail;

    free(s->prefix_conv);
    free(s->prefix_ssm);
    free(s->prefix_k);
    free(s->prefix_v);
    s->prefix_conv = new_conv;
    s->prefix_ssm  = new_ssm;
    s->prefix_k    = new_k;
    s->prefix_v    = new_v;
    s->prefix_len  = hdr.prefix_len;
    s->prefix_mrope_delta = (int64_t)hdr.reserved;
    free(s->prefix_tokens);
    s->prefix_tokens = NULL;
    s->prefix_has_tokens = 0;
    return qwen35_restore_prefix(s, out_n_tokens);

fail:
    if (fp) fclose(fp);
    free(new_conv);
    free(new_ssm);
    free(new_k);
    free(new_v);
    return -1;
}

static int64_t qwen35_get_rope_delta(const AsiArchSession *s)
{
    return s ? s->mrope_pos_delta : 0;
}

static void qwen35_set_rope_delta(AsiArchSession *s, int64_t delta)
{
    if (s) s->mrope_pos_delta = delta;
}

static void qwen35_set_calib_hook(AsiArchSession *s,
                                  void (*fn)(void *, uint32_t, int, const float *, uint32_t, uint32_t),
                                  void *user_ctx)
{
    if (s)
    {
        s->calib_hook = fn;
        s->calib_ctx = user_ctx;
    }
}

static int qwen35_forward_embd(AsiArchSession *s,
                               const float *embds,
                               uint32_t n_tokens,
                               uint32_t kv_pos_start,
                               const uint32_t *mrope_pos_3d,
                               const uint32_t mrope_sections[3],
                               uint32_t mrope_flags,
                               uint32_t next_rope_pos,
                               const volatile int *cancel_flag,
                               float *out_last_logits)
{
    if (!s || !embds || n_tokens == 0u) return -1;
    if (kv_pos_start + n_tokens > s->max_ctx) return -1;
    if (cancel_flag && *cancel_flag) return -2;

    const uint32_t H = s->cart->config.hidden_size;
    size_t total_floats = (size_t)n_tokens * (size_t)H;
    for (size_t i = 0; i < total_floats; ++i)
    {
        if (!isfinite(embds[i])) return -1;
    }

    s->spec_count = 0;
    s->spec_cursor = 0;
    if (kv_pos_start < s->kv_dirty_from) s->kv_dirty_from = kv_pos_start;
    if (kv_pos_start == 0u)
    {
        s->mrope_pos_delta = 0;
        s->prompt_len = n_tokens;
    }
    else if (kv_pos_start <= s->prompt_len && kv_pos_start + n_tokens > s->prompt_len)
    {
        s->prompt_len = kv_pos_start + n_tokens;
    }
    if (s->token_history)
    {
        for (uint32_t i = 0; i < n_tokens; ++i)
            s->token_history[kv_pos_start + i] = -1;
        if (kv_pos_start + n_tokens > s->history_len)
            s->history_len = kv_pos_start + n_tokens;
    }
    s->history_valid_from = kv_pos_start + n_tokens;

    const uint32_t chunk_max = (uint32_t)asi_runtime_config()->batch_chunk_max;
    uint32_t done = 0u;
    asi_pool_begin(s->pool);
    while (done < n_tokens)
    {
        if (cancel_flag && *cancel_flag)
        {
            asi_pool_end(s->pool);
            return -2;
        }
        uint32_t B = n_tokens - done;
        if (B > chunk_max) B = chunk_max;
        int is_last = (done + B == n_tokens);
        int rc = qwen35_forward_chunk_ex(
            s, NULL, B, kv_pos_start + done,
            (is_last && out_last_logits != NULL) ? 1 : 0, 0,
            out_last_logits, NULL,
            embds + (size_t)done * H,
            mrope_pos_3d ? (mrope_pos_3d + (size_t)done * 3u) : NULL,
            mrope_sections, mrope_flags, cancel_flag);
        if (rc != 0)
        {
            asi_pool_end(s->pool);
            return rc;
        }
        done += B;
    }
    asi_pool_end(s->pool);

    if (next_rope_pos > 0u)
    {
        s->mrope_pos_delta = (int64_t)next_rope_pos - (int64_t)(kv_pos_start + n_tokens);
    }
    else if (mrope_pos_3d != NULL)
    {
        uint32_t max_p = 0u;
        size_t n_coords = (size_t)n_tokens * 3u;
        for (size_t k = 0; k < n_coords; ++k)
        {
            if (mrope_pos_3d[k] > max_p) max_p = mrope_pos_3d[k];
        }
        s->mrope_pos_delta = (int64_t)(max_p + 1u) - (int64_t)(kv_pos_start + n_tokens);
    }
    return 0;
}

const AsiArchHandler g_asi_arch_hybrid_qwen35 = {
    .arch_id = ASI_ARCH_HYBRID_QWEN35,
    .name = "hybrid_qwen35",
    .init_session = qwen35_init_session,
    .free_session = qwen35_free_session,
    .forward_token = qwen35_forward_token,
    .forward_batch = qwen35_forward_batch,
    .forward_speculative = qwen35_forward_speculative,
    .reset_state = qwen35_reset_state,
    .save_prefix = qwen35_save_prefix,
    .restore_prefix = qwen35_restore_prefix,
    .save_prefix_file = qwen35_save_prefix_file,
    .load_prefix_file = qwen35_load_prefix_file,
    .forward_embd = qwen35_forward_embd,
    .get_rope_delta = qwen35_get_rope_delta,
    .set_rope_delta = qwen35_set_rope_delta,
    .set_calib_hook = qwen35_set_calib_hook,
};

/* g_asi_arch_dense_transformer provided by src/asi_dense.c */

/* =========================================================================
 * Exp E06: State-Centric Internal Bridge for Qwen35Session
 * ========================================================================= */

int qwen35_sc_get_geometry(const AsiArchSession *s, Qwen35StateGeom *out_geom)
{
    if (!s || !out_geom || !s->cart) return -1;
    const AsiV2ModelConfig *c = &s->cart->config;
    uint32_t qkv_dim = c->ssm_inner_size * 3u;
    uint32_t conv_hist = (c->ssm_conv_kernel > 1u) ? (c->ssm_conv_kernel - 1u) : 0u;
    size_t conv_per_layer = (size_t)conv_hist * qkv_dim;
    uint32_t ssm_head_dim = (c->ssm_group_count > 0u) ? (c->ssm_inner_size / c->ssm_group_count) : 0u;
    size_t ssm_per_layer = (size_t)c->ssm_group_count * c->ssm_state_size * ssm_head_dim;
    size_t kv_dim = (size_t)c->n_kv_heads * c->head_dim;

    uint32_t n_dn = 0, n_fa = 0;
    for (uint32_t l = 0; l < c->n_layers; ++l)
    {
        if (s->layers[l].kind == QWEN35_LAYER_DELTANET) n_dn++; else n_fa++;
    }

    memset(out_geom, 0, sizeof(*out_geom));
    out_geom->max_ctx = s->max_ctx;
    out_geom->n_dn = n_dn;
    out_geom->n_fa = n_fa;
    out_geom->fp32_ref = (uint32_t)(s->fp32_ref ? 1u : 0u);
    out_geom->conv_per_layer = conv_per_layer;
    out_geom->ssm_per_layer = ssm_per_layer;
    out_geom->kv_dim = kv_dim;
    out_geom->conv_bytes = (uint64_t)n_dn * (uint64_t)conv_per_layer * sizeof(float);
    out_geom->ssm_bytes = (uint64_t)n_dn * (uint64_t)ssm_per_layer * sizeof(float);
    memcpy(out_geom->cart_identity_hash, s->cart->header.identity_hash, 24);
    out_geom->arch_id = s->cart->header.arch_id;
    return 0;
}

int qwen35_sc_capture_deltanet(const AsiArchSession *s,
                               float *out_conv,
                               float *out_ssm)
{
    if (!s || !s->cart) return -1;
    Qwen35StateGeom g;
    if (qwen35_sc_get_geometry(s, &g) != 0) return -1;
    const AsiV2ModelConfig *c = &s->cart->config;

    uint32_t i_dn = 0;
    for (uint32_t l = 0; l < c->n_layers; ++l)
    {
        if (s->layers[l].kind == QWEN35_LAYER_DELTANET)
        {
            if (g.conv_per_layer > 0 && out_conv)
            {
                memcpy(out_conv + (size_t)i_dn * g.conv_per_layer,
                       s->ssm_states[l].conv_state,
                       g.conv_per_layer * sizeof(float));
            }
            if (g.ssm_per_layer > 0 && out_ssm)
            {
                memcpy(out_ssm + (size_t)i_dn * g.ssm_per_layer,
                       s->ssm_states[l].ssm_state,
                       g.ssm_per_layer * sizeof(float));
            }
            i_dn++;
        }
    }
    return 0;
}

int qwen35_sc_capture_kv_page(const AsiArchSession *s,
                              uint32_t fa_layer_idx,
                              uint32_t page_idx,
                              uint32_t page_tokens,
                              uint32_t take_tokens,
                              uint16_t *out_k,
                              uint16_t *out_v)
{
    if (!s || !s->cart || !out_k || !out_v || take_tokens == 0 || take_tokens > page_tokens)
        return -1;
    const AsiV2ModelConfig *c = &s->cart->config;
    size_t kv_dim = (size_t)c->n_kv_heads * c->head_dim;
    uint32_t tok_start = page_idx * page_tokens;
    if (tok_start + take_tokens > s->max_ctx) return -1;

    uint32_t i_fa = 0;
    for (uint32_t l = 0; l < c->n_layers; ++l)
    {
        if (s->layers[l].kind == QWEN35_LAYER_FULL_ATTN)
        {
            if (i_fa == fa_layer_idx)
            {
                size_t elems = (size_t)take_tokens * kv_dim;
                size_t offset = (size_t)tok_start * kv_dim;
                memcpy(out_k, s->kv_states[l].k_cache + offset, elems * sizeof(uint16_t));
                memcpy(out_v, s->kv_states[l].v_cache + offset, elems * sizeof(uint16_t));
                if (take_tokens < page_tokens)
                {
                    size_t rem_elems = (size_t)(page_tokens - take_tokens) * kv_dim;
                    memset(out_k + elems, 0, rem_elems * sizeof(uint16_t));
                    memset(out_v + elems, 0, rem_elems * sizeof(uint16_t));
                }
                return 0;
            }
            i_fa++;
        }
    }
    return -1;
}

int qwen35_sc_restore_state(AsiArchSession *s,
                            uint32_t n_tokens,
                            const float *conv,
                            const float *ssm,
                            const uint16_t *const *k_pages,
                            const uint16_t *const *v_pages,
                            uint32_t n_pages_per_layer,
                            uint32_t page_tokens,
                            const int32_t *tokens,
                            int has_tokens)
{
    if (!s || !s->cart || n_tokens > s->max_ctx) return -1;
    Qwen35StateGeom g;
    if (qwen35_sc_get_geometry(s, &g) != 0) return -1;
    const AsiV2ModelConfig *c = &s->cart->config;

    uint32_t i_dn = 0, i_fa = 0;
    for (uint32_t l = 0; l < c->n_layers; ++l)
    {
        if (s->layers[l].kind == QWEN35_LAYER_DELTANET)
        {
            if (g.conv_per_layer > 0 && conv)
            {
                memcpy(s->ssm_states[l].conv_state,
                       conv + (size_t)i_dn * g.conv_per_layer,
                       g.conv_per_layer * sizeof(float));
            }
            if (g.ssm_per_layer > 0 && ssm)
            {
                memcpy(s->ssm_states[l].ssm_state,
                       ssm + (size_t)i_dn * g.ssm_per_layer,
                       g.ssm_per_layer * sizeof(float));
            }
            i_dn++;
        }
        else
        {
            size_t kv_dim = g.kv_dim;
            for (uint32_t p = 0; p < n_pages_per_layer; ++p)
            {
                uint32_t tok_start = p * page_tokens;
                if (tok_start >= n_tokens) break;
                uint32_t take = n_tokens - tok_start;
                if (take > page_tokens) take = page_tokens;
                size_t idx = (size_t)i_fa * n_pages_per_layer + p;
                size_t offset = (size_t)tok_start * kv_dim;
                size_t elems = (size_t)take * kv_dim;
                memcpy(s->kv_states[l].k_cache + offset, k_pages[idx], elems * sizeof(uint16_t));
                memcpy(s->kv_states[l].v_cache + offset, v_pages[idx], elems * sizeof(uint16_t));
            }
            /* Zero tail KV cache beyond n_tokens for strict zero cross-branch pollution */
            if (n_tokens < s->max_ctx)
            {
                size_t offset = (size_t)n_tokens * kv_dim;
                size_t rem_elems = (size_t)(s->max_ctx - n_tokens) * kv_dim;
                memset(s->kv_states[l].k_cache + offset, 0, rem_elems * sizeof(uint16_t));
                memset(s->kv_states[l].v_cache + offset, 0, rem_elems * sizeof(uint16_t));
            }
            i_fa++;
        }
    }

    s->spec_count = 0;
    s->spec_cursor = 0;
    s->spec_start_pos = 0;
    s->spec_consec_misses = 0;
    s->spec_cooldown = 0;
    s->mrope_pos_delta = 0;
    s->fail_gen++;
    if (s->fail_gen == 0)
    {
        if (s->fail_hash) memset(s->fail_hash, 0, ASI_SPEC_FAIL_CAP * sizeof(AsiSpecFailEntry));
        s->fail_gen = 1;
    }
    s->prompt_len = n_tokens;
    s->history_len = n_tokens;
    if (has_tokens && tokens && s->token_history && n_tokens > 0)
    {
        memcpy(s->token_history, tokens, (size_t)n_tokens * sizeof(int32_t));
        s->history_valid_from = 0;
    }
    else
    {
        s->history_valid_from = n_tokens;
    }
    s->kv_dirty_from = n_tokens;
    return 0;
}

int qwen35_sc_get_history(const AsiArchSession *s,
                          const int32_t **out_tokens,
                          uint32_t *out_len,
                          uint32_t *out_valid_from)
{
    if (!s) return -1;
    if (out_tokens) *out_tokens = s->token_history;
    if (out_len) *out_len = s->history_len;
    if (out_valid_from) *out_valid_from = s->history_valid_from;
    return 0;
}

uint32_t qwen35_sc_get_kv_dirty_from(const AsiArchSession *s)
{
    return s ? s->kv_dirty_from : 0u;
}

void qwen35_sc_mark_kv_clean(AsiArchSession *s, uint32_t pos)
{
    if (s) s->kv_dirty_from = pos;
}

int64_t qwen35_sc_get_rope_delta(const AsiArchSession *s)
{
    return s ? s->mrope_pos_delta : 0;
}

void qwen35_sc_set_rope_delta(AsiArchSession *s, int64_t delta)
{
    if (s) s->mrope_pos_delta = delta;
}


const AsiArchHandler *asi_arch_find(uint32_t arch_id)
{
    switch (arch_id)
    {
        case ASI_ARCH_HYBRID_QWEN35:
            return &g_asi_arch_hybrid_qwen35;
        case ASI_ARCH_DENSE_LLAMA:
        case ASI_ARCH_DENSE_GEMMA:
        case ASI_ARCH_DENSE_QWEN2:
        case ASI_ARCH_DENSE_QWEN3:
            return &g_asi_arch_dense_transformer;
        default:
            return NULL;
    }
}
