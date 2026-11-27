#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "asi_kernels.h"
#include "q4k.h"

#include <math.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

int asi_kernel_ctx_init(AsiKernelCtx *ctx,
                        AsiThreadPool *existing_pool,
                        uint32_t n_threads,
                        const volatile int *cancel_flag)
{
    if (!ctx) return -1;
    memset(ctx, 0, sizeof(*ctx));
    if (n_threads == 0u) n_threads = 1u;
    ctx->n_threads = n_threads;
    ctx->cancel_flag = cancel_flag;
    if (existing_pool) {
        ctx->pool = existing_pool;
        ctx->owns_pool = 0;
    } else if (n_threads > 1u) {
        ctx->pool = asi_pool_create((int)n_threads);
        if (!ctx->pool) return -1;
        ctx->owns_pool = 1;
        asi_pool_begin(ctx->pool);
    }
    return 0;
}

void asi_kernel_ctx_destroy(AsiKernelCtx *ctx)
{
    if (!ctx) return;
    if (ctx->pool && ctx->owns_pool) {
        asi_pool_end(ctx->pool);
        asi_pool_destroy(ctx->pool);
    }
    ctx->pool = NULL;
    ctx->owns_pool = 0;
}

typedef struct {
    uint32_t            n_items;
    uint32_t            grain;
    AsiKernelRangeFn    fn;
    void               *user_data;
    const volatile int *cancel_flag;
    _Alignas(64) atomic_uint next_idx;
} KernelParForCtx;

static void kernel_par_for_worker(int tid, int n_threads, void *vctx)
{
    (void)n_threads;
    KernelParForCtx *p = (KernelParForCtx *)vctx;
    uint32_t grain = p->grain ? p->grain : 1u;
    for (;;) {
        if (p->cancel_flag && *p->cancel_flag) break;
        uint32_t start = atomic_fetch_add_explicit(&p->next_idx, grain, memory_order_relaxed);
        if (start >= p->n_items) break;
        uint32_t end = start + grain;
        if (end > p->n_items) end = p->n_items;
        p->fn(start, end, tid, p->user_data);
    }
}

void asi_kernel_parallel_for(AsiKernelCtx *ctx,
                             uint32_t n_items,
                             uint32_t grain,
                             AsiKernelRangeFn fn,
                             void *user_data)
{
    if (!fn || n_items == 0u) return;
    if (grain == 0u) grain = 1u;
    if (!ctx || !ctx->pool || ctx->n_threads <= 1u || n_items <= grain) {
        if (ctx && ctx->cancel_flag && *ctx->cancel_flag) return;
        fn(0u, n_items, 0, user_data);
        return;
    }
    KernelParForCtx pf;
    pf.n_items = n_items;
    pf.grain = grain;
    pf.fn = fn;
    pf.user_data = user_data;
    pf.cancel_flag = ctx->cancel_flag;
    atomic_init(&pf.next_idx, 0u);
    asi_pool_run(ctx->pool, kernel_par_for_worker, &pf);
}

typedef struct {
    const AsiKernelJob *jobs;
    uint32_t            n_jobs;
    const volatile int *cancel_flag;
    _Alignas(64) atomic_uint next_job;
} KernelJobsCtx;

static void kernel_jobs_worker(int tid, int n_threads, void *vctx)
{
    (void)tid;
    (void)n_threads;
    KernelJobsCtx *jc = (KernelJobsCtx *)vctx;
    for (;;) {
        if (jc->cancel_flag && *jc->cancel_flag) break;
        uint32_t idx = atomic_fetch_add_explicit(&jc->next_job, 1u, memory_order_relaxed);
        if (idx >= jc->n_jobs) break;
        if (jc->jobs[idx].fn) jc->jobs[idx].fn(jc->jobs[idx].arg);
    }
}

void asi_kernel_run_jobs(AsiKernelCtx *ctx,
                         const AsiKernelJob *jobs,
                         uint32_t n_jobs)
{
    if (!jobs || n_jobs == 0u) return;
    if (!ctx || !ctx->pool || ctx->n_threads <= 1u || n_jobs == 1u) {
        for (uint32_t i = 0u; i < n_jobs; i++) {
            if (ctx && ctx->cancel_flag && *ctx->cancel_flag) break;
            if (jobs[i].fn) jobs[i].fn(jobs[i].arg);
        }
        return;
    }
    KernelJobsCtx jc;
    jc.jobs = jobs;
    jc.n_jobs = n_jobs;
    jc.cancel_flag = ctx->cancel_flag;
    atomic_init(&jc.next_job, 0u);
    asi_pool_run(ctx->pool, kernel_jobs_worker, &jc);
}

static inline float apply_act_scalar(float v, uint32_t act_type)
{
    switch (act_type) {
        case ASI_ACT_GELU_TANH: {
            const float k = 0.7978845608028654f; /* sqrt(2 / pi) */
            float t = tanhf(k * (v + 0.044715f * v * v * v));
            return 0.5f * v * (1.0f + t);
        }
        case ASI_ACT_GELU_ERF:
            return 0.5f * v * (1.0f + erff(v * 0.7071067811865475f));
        case ASI_ACT_SILU:
            return v / (1.0f + expf(-v));
        case ASI_ACT_RELU:
            return (v > 0.0f) ? v : 0.0f;
        default:
            return v;
    }
}

#if defined(__AVX2__)
static inline float hsum_m256(__m256 v)
{
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(s);
    __m128 sums = _mm_add_ps(s, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

#if defined(__FMA__)
/* Accurate (< 1 ULP) 8-wide AVX2+FMA expf using Cody-Waite range reduction + minimax deg-5 poly */
static inline __m256 avx2_exp_ps(__m256 x)
{
    const __m256 exp_hi = _mm256_set1_ps(88.0f);
    const __m256 exp_lo = _mm256_set1_ps(-87.3365447505531f);
    const __m256 log2ef = _mm256_set1_ps(1.44269504088896341f);
    const __m256 ln2_hi = _mm256_set1_ps(0.693359375f);
    const __m256 ln2_lo = _mm256_set1_ps(-2.12194440e-4f);
    const __m256 p0 = _mm256_set1_ps(1.9875691500e-4f);
    const __m256 p1 = _mm256_set1_ps(1.3981999507e-3f);
    const __m256 p2 = _mm256_set1_ps(8.3334519073e-3f);
    const __m256 p3 = _mm256_set1_ps(4.1665795894e-2f);
    const __m256 p4 = _mm256_set1_ps(1.6666665459e-1f);
    const __m256 p5 = _mm256_set1_ps(5.0000001201e-1f);
    const __m256 one = _mm256_set1_ps(1.0f);

    __m256 underflow = _mm256_cmp_ps(x, exp_lo, _CMP_LT_OQ);
    x = _mm256_min_ps(x, exp_hi);
    x = _mm256_max_ps(x, exp_lo);

    __m256 fx = _mm256_round_ps(_mm256_mul_ps(x, log2ef),
                                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r = _mm256_fnmadd_ps(fx, ln2_hi, x);
    r = _mm256_fnmadd_ps(fx, ln2_lo, r);

    __m256 r2 = _mm256_mul_ps(r, r);
    __m256 y = _mm256_fmadd_ps(p0, r, p1);
    y = _mm256_fmadd_ps(y, r, p2);
    y = _mm256_fmadd_ps(y, r, p3);
    y = _mm256_fmadd_ps(y, r, p4);
    y = _mm256_fmadd_ps(y, r, p5);
    y = _mm256_fmadd_ps(y, r2, r);
    y = _mm256_add_ps(y, one);

    __m256i emm0 = _mm256_cvttps_epi32(fx);
    emm0 = _mm256_add_epi32(emm0, _mm256_set1_epi32(0x7f));
    emm0 = _mm256_slli_epi32(emm0, 23);
    __m256 res = _mm256_mul_ps(y, _mm256_castsi256_ps(emm0));
    return _mm256_andnot_ps(underflow, res);
}

static inline __m256 avx2_apply_act_ps(__m256 v, uint32_t act_type)
{
    if (act_type == ASI_ACT_GELU_TANH) {
        /* 0.5*v*(1 + tanh(z)) == v / (1 + exp(-2*z)), branchless & cancellation-free */
        const __m256 k = _mm256_set1_ps(0.7978845608028654f);
        const __m256 c = _mm256_set1_ps(0.044715f);
        const __m256 sign_mask = _mm256_set1_ps(-0.0f);
        const __m256 neg2 = _mm256_set1_ps(-2.0f);
        const __m256 one = _mm256_set1_ps(1.0f);
        __m256 v2 = _mm256_mul_ps(v, v);
        __m256 z = _mm256_mul_ps(k, _mm256_fmadd_ps(_mm256_mul_ps(c, v2), v, v));
        __m256 abs_z = _mm256_andnot_ps(sign_mask, z);
        __m256 t = avx2_exp_ps(_mm256_mul_ps(neg2, abs_z));
        __m256 num = _mm256_blendv_ps(one, t, z);
        return _mm256_div_ps(_mm256_mul_ps(v, num), _mm256_add_ps(one, t));
    }
    if (act_type == ASI_ACT_SILU) {
        const __m256 sign_mask = _mm256_set1_ps(-0.0f);
        const __m256 one = _mm256_set1_ps(1.0f);
        __m256 neg_abs = _mm256_or_ps(v, sign_mask);
        __m256 t = avx2_exp_ps(neg_abs);
        __m256 num = _mm256_blendv_ps(one, t, v);
        return _mm256_div_ps(_mm256_mul_ps(v, num), _mm256_add_ps(one, t));
    }
    if (act_type == ASI_ACT_RELU) {
        return _mm256_max_ps(v, _mm256_setzero_ps());
    }
    if (act_type == ASI_ACT_GELU_ERF) {
        _Alignas(32) float tmp[8];
        _mm256_store_ps(tmp, v);
        for (int i = 0; i < 8; i++) tmp[i] = apply_act_scalar(tmp[i], act_type);
        return _mm256_load_ps(tmp);
    }
    return v;
}
#endif
#endif

static inline int asi_fast_q8_enabled(void)
{
    const char *v = getenv("LILA_ASI_VISION_FAST_Q8");
    return (v && v[0] == '1' && v[1] == '\0') ? 1 : 0;
}

typedef struct {
    const uint8_t *w_data;
    size_t         row_bytes;
    uint32_t       quant_type;
    const float   *bias;
    const float   *x;
    uint32_t       M;
    uint32_t       N;
    uint32_t       K;
    uint32_t       act_type;
    const float   *residual;
    float         *out;
    uint32_t       m_tile_size;
    uint32_t       n_tile_size;
    uint32_t       n_ntiles;
    /* Optional quantized activation buffers for Q8_0 x Q8_0 fast path */
    const int8_t  *xq_qs;
    const float   *xq_d;
} LinearQuantCtx;

#if defined(__AVX2__) && defined(__FMA__)
/* L2/L1 cache-blocked Q8_0 x FP32 AVX2+FMA GEMM (MB=60, KB=256, NB=64/16/8).
 * - Packs X_tile[60][256] (60 KiB = 15 contiguous 4KB pages) once per (m_tile, n_tile, k_tile),
 *   eliminating L1 dTLB misses and 12 KiB-stride L1d set conflicts at K=3072 while staying
 *   100% resident in Broadwell's 256 KiB L2 cache across all 32 weight pairs.
 * - Dequantizes 2 Q8_0 weight rows once per (m_tile, k_tile) into 2 KiB L1d w_deq[2][256],
 *   amortizing weight dequantization & DRAM weight reads by 60x.
 * - Uses a 6x2 register-blocked microkernel (12 accumulators + 2 W + 1 X = 15 YMMs, zero spills).
 * - Disjoint (m_tile, n_tile) ownership guarantees bit-exact 1T == NT determinism. */
#define ASI_GEMM_MB 60u
#define ASI_GEMM_KB 256u
#define ASI_GEMM_NB_MAX 64u

static inline void epilogue_store_tile_avx2(const float c_tile[ASI_GEMM_MB][ASI_GEMM_NB_MAX],
                                            uint32_t m0, uint32_t m_cnt,
                                            uint32_t n0, uint32_t n_cnt,
                                            uint32_t N, uint32_t act,
                                            const float *bias,
                                            const float *res,
                                            float *out)
{
    for (uint32_t m = 0u; m < m_cnt; m++) {
        float *out_row = out + (size_t)(m0 + m) * N + n0;
        const float *res_row = res ? (res + (size_t)(m0 + m) * N + n0) : NULL;
        const float *src = c_tile[m];
        uint32_t nr = 0u;
        for (; nr + 7u < n_cnt; nr += 8u) {
            __m256 v = _mm256_load_ps(src + nr);
            if (bias) v = _mm256_add_ps(v, _mm256_loadu_ps(bias + n0 + nr));
            if (act != ASI_ACT_NONE) v = avx2_apply_act_ps(v, act);
            if (res_row) v = _mm256_add_ps(_mm256_loadu_ps(res_row + nr), v);
            _mm256_storeu_ps(out_row + nr, v);
        }
        for (; nr < n_cnt; nr++) {
            float v = src[nr] + (bias ? bias[n0 + nr] : 0.0f);
            if (act != ASI_ACT_NONE) v = apply_act_scalar(v, act);
            out_row[nr] = res_row ? (res_row[nr] + v) : v;
        }
    }
}

static void linear_q8_fp32_tile_range(uint32_t t0, uint32_t t1, int tid, void *vctx)
{
    (void)tid;
    const LinearQuantCtx *c = (const LinearQuantCtx *)vctx;
    const uint32_t M = c->M;
    const uint32_t N = c->N;
    const uint32_t K = c->K;
    const uint32_t n_tile_sz = c->n_tile_size;
    const uint32_t n_ntiles = c->n_ntiles;

    _Alignas(32) float c_tile[ASI_GEMM_MB][ASI_GEMM_NB_MAX];
    _Alignas(32) float w_deq[2][ASI_GEMM_KB];
    _Alignas(32) float x_pack[ASI_GEMM_MB][ASI_GEMM_KB];

    for (uint32_t tile_idx = t0; tile_idx < t1; tile_idx++) {
        uint32_t mt = tile_idx / n_ntiles;
        uint32_t nt = tile_idx % n_ntiles;
        uint32_t m0 = mt * ASI_GEMM_MB;
        uint32_t m_cnt = M - m0;
        if (m_cnt > ASI_GEMM_MB) m_cnt = ASI_GEMM_MB;
        uint32_t n0 = nt * n_tile_sz;
        uint32_t n_cnt = N - n0;
        if (n_cnt > n_tile_sz) n_cnt = n_tile_sz;

        for (uint32_t m = 0u; m < m_cnt; m++) {
            memset(c_tile[m], 0, (size_t)n_cnt * sizeof(float));
        }

        for (uint32_t k0 = 0u; k0 < K; k0 += ASI_GEMM_KB) {
            uint32_t kb = K - k0;
            if (kb > ASI_GEMM_KB) kb = ASI_GEMM_KB;
            uint32_t b0 = k0 / 32u;
            uint32_t nb_tile = kb / 32u;

            for (uint32_t m = 0u; m < m_cnt; m++) {
                memcpy(x_pack[m], c->x + (size_t)(m0 + m) * K + k0, (size_t)kb * sizeof(float));
            }

            uint32_t nr = 0u;
            for (; nr + 1u < n_cnt; nr += 2u) {
                const uint8_t *w_row0 = c->w_data + (size_t)(n0 + nr + 0u) * c->row_bytes + (size_t)b0 * Q8_0_BLOCK_SIZE;
                const uint8_t *w_row1 = w_row0 + c->row_bytes;

                if (nr + 3u < n_cnt) {
                    const uint8_t *pw0 = w_row0 + 2u * c->row_bytes;
                    const uint8_t *pw1 = pw0 + c->row_bytes;
                    _mm_prefetch((const char *)pw0, _MM_HINT_T0);
                    _mm_prefetch((const char *)pw1, _MM_HINT_T0);
                    _mm_prefetch((const char *)(pw0 + 128), _MM_HINT_T0);
                    _mm_prefetch((const char *)(pw1 + 128), _MM_HINT_T0);
                }

                /* Dequantize 2 weight rows for this K-tile ONCE (amortized across m_cnt rows) */
                for (uint32_t b = 0u; b < nb_tile; b++) {
                    const uint8_t *blk0 = w_row0 + (size_t)b * Q8_0_BLOCK_SIZE;
                    const uint8_t *blk1 = w_row1 + (size_t)b * Q8_0_BLOCK_SIZE;
                    uint16_t d0_raw, d1_raw;
                    memcpy(&d0_raw, blk0, sizeof(d0_raw));
                    memcpy(&d1_raw, blk1, sizeof(d1_raw));
                    __m256 vd0 = _mm256_set1_ps(fp16_to_fp32(d0_raw));
                    __m256 vd1 = _mm256_set1_ps(fp16_to_fp32(d1_raw));
                    const int8_t *qs0 = (const int8_t *)(blk0 + 2u);
                    const int8_t *qs1 = (const int8_t *)(blk1 + 2u);
                    float *dst0 = w_deq[0] + b * 32u;
                    float *dst1 = w_deq[1] + b * 32u;
                    for (uint32_t sub = 0u; sub < 32u; sub += 8u) {
                        __m256i q0 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(qs0 + sub)));
                        __m256i q1 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(qs1 + sub)));
                        _mm256_store_ps(dst0 + sub, _mm256_mul_ps(_mm256_cvtepi32_ps(q0), vd0));
                        _mm256_store_ps(dst1 + sub, _mm256_mul_ps(_mm256_cvtepi32_ps(q1), vd1));
                    }
                }

                /* 6x2 register-blocked AVX2+FMA microkernel (12 acc + 2 W + 1 X = 15 YMMs) */
                uint32_t m = 0u;
                for (; m + 5u < m_cnt; m += 6u) {
                    const float *x0 = x_pack[m + 0u];
                    const float *x1 = x_pack[m + 1u];
                    const float *x2 = x_pack[m + 2u];
                    const float *x3 = x_pack[m + 3u];
                    const float *x4 = x_pack[m + 4u];
                    const float *x5 = x_pack[m + 5u];
                    __m256 a00 = _mm256_setzero_ps(), a01 = _mm256_setzero_ps();
                    __m256 a10 = _mm256_setzero_ps(), a11 = _mm256_setzero_ps();
                    __m256 a20 = _mm256_setzero_ps(), a21 = _mm256_setzero_ps();
                    __m256 a30 = _mm256_setzero_ps(), a31 = _mm256_setzero_ps();
                    __m256 a40 = _mm256_setzero_ps(), a41 = _mm256_setzero_ps();
                    __m256 a50 = _mm256_setzero_ps(), a51 = _mm256_setzero_ps();
                    for (uint32_t k = 0u; k < kb; k += 8u) {
                        __m256 w0 = _mm256_load_ps(w_deq[0] + k);
                        __m256 w1 = _mm256_load_ps(w_deq[1] + k);
                        __m256 xv;
                        xv = _mm256_load_ps(x0 + k); a00 = _mm256_fmadd_ps(xv, w0, a00); a01 = _mm256_fmadd_ps(xv, w1, a01);
                        xv = _mm256_load_ps(x1 + k); a10 = _mm256_fmadd_ps(xv, w0, a10); a11 = _mm256_fmadd_ps(xv, w1, a11);
                        xv = _mm256_load_ps(x2 + k); a20 = _mm256_fmadd_ps(xv, w0, a20); a21 = _mm256_fmadd_ps(xv, w1, a21);
                        xv = _mm256_load_ps(x3 + k); a30 = _mm256_fmadd_ps(xv, w0, a30); a31 = _mm256_fmadd_ps(xv, w1, a31);
                        xv = _mm256_load_ps(x4 + k); a40 = _mm256_fmadd_ps(xv, w0, a40); a41 = _mm256_fmadd_ps(xv, w1, a41);
                        xv = _mm256_load_ps(x5 + k); a50 = _mm256_fmadd_ps(xv, w0, a50); a51 = _mm256_fmadd_ps(xv, w1, a51);
                    }
                    c_tile[m + 0u][nr + 0u] += hsum_m256(a00); c_tile[m + 0u][nr + 1u] += hsum_m256(a01);
                    c_tile[m + 1u][nr + 0u] += hsum_m256(a10); c_tile[m + 1u][nr + 1u] += hsum_m256(a11);
                    c_tile[m + 2u][nr + 0u] += hsum_m256(a20); c_tile[m + 2u][nr + 1u] += hsum_m256(a21);
                    c_tile[m + 3u][nr + 0u] += hsum_m256(a30); c_tile[m + 3u][nr + 1u] += hsum_m256(a31);
                    c_tile[m + 4u][nr + 0u] += hsum_m256(a40); c_tile[m + 4u][nr + 1u] += hsum_m256(a41);
                    c_tile[m + 5u][nr + 0u] += hsum_m256(a50); c_tile[m + 5u][nr + 1u] += hsum_m256(a51);
                }
                for (; m + 3u < m_cnt; m += 4u) {
                    const float *x0 = x_pack[m + 0u];
                    const float *x1 = x_pack[m + 1u];
                    const float *x2 = x_pack[m + 2u];
                    const float *x3 = x_pack[m + 3u];
                    __m256 a00 = _mm256_setzero_ps(), a01 = _mm256_setzero_ps();
                    __m256 a10 = _mm256_setzero_ps(), a11 = _mm256_setzero_ps();
                    __m256 a20 = _mm256_setzero_ps(), a21 = _mm256_setzero_ps();
                    __m256 a30 = _mm256_setzero_ps(), a31 = _mm256_setzero_ps();
                    for (uint32_t k = 0u; k < kb; k += 8u) {
                        __m256 w0 = _mm256_load_ps(w_deq[0] + k);
                        __m256 w1 = _mm256_load_ps(w_deq[1] + k);
                        __m256 xv;
                        xv = _mm256_load_ps(x0 + k); a00 = _mm256_fmadd_ps(xv, w0, a00); a01 = _mm256_fmadd_ps(xv, w1, a01);
                        xv = _mm256_load_ps(x1 + k); a10 = _mm256_fmadd_ps(xv, w0, a10); a11 = _mm256_fmadd_ps(xv, w1, a11);
                        xv = _mm256_load_ps(x2 + k); a20 = _mm256_fmadd_ps(xv, w0, a20); a21 = _mm256_fmadd_ps(xv, w1, a21);
                        xv = _mm256_load_ps(x3 + k); a30 = _mm256_fmadd_ps(xv, w0, a30); a31 = _mm256_fmadd_ps(xv, w1, a31);
                    }
                    c_tile[m + 0u][nr + 0u] += hsum_m256(a00); c_tile[m + 0u][nr + 1u] += hsum_m256(a01);
                    c_tile[m + 1u][nr + 0u] += hsum_m256(a10); c_tile[m + 1u][nr + 1u] += hsum_m256(a11);
                    c_tile[m + 2u][nr + 0u] += hsum_m256(a20); c_tile[m + 2u][nr + 1u] += hsum_m256(a21);
                    c_tile[m + 3u][nr + 0u] += hsum_m256(a30); c_tile[m + 3u][nr + 1u] += hsum_m256(a31);
                }
                for (; m < m_cnt; m++) {
                    const float *xm = x_pack[m];
                    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
                    for (uint32_t k = 0u; k < kb; k += 8u) {
                        __m256 xv = _mm256_load_ps(xm + k);
                        a0 = _mm256_fmadd_ps(xv, _mm256_load_ps(w_deq[0] + k), a0);
                        a1 = _mm256_fmadd_ps(xv, _mm256_load_ps(w_deq[1] + k), a1);
                    }
                    c_tile[m][nr + 0u] += hsum_m256(a0);
                    c_tile[m][nr + 1u] += hsum_m256(a1);
                }
            }

            if (nr < n_cnt) {
                const uint8_t *w_row0 = c->w_data + (size_t)(n0 + nr) * c->row_bytes + (size_t)b0 * Q8_0_BLOCK_SIZE;
                for (uint32_t b = 0u; b < nb_tile; b++) {
                    const uint8_t *blk0 = w_row0 + (size_t)b * Q8_0_BLOCK_SIZE;
                    uint16_t d0_raw;
                    memcpy(&d0_raw, blk0, sizeof(d0_raw));
                    __m256 vd0 = _mm256_set1_ps(fp16_to_fp32(d0_raw));
                    const int8_t *qs0 = (const int8_t *)(blk0 + 2u);
                    float *dst0 = w_deq[0] + b * 32u;
                    for (uint32_t sub = 0u; sub < 32u; sub += 8u) {
                        __m256i q0 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(qs0 + sub)));
                        _mm256_store_ps(dst0 + sub, _mm256_mul_ps(_mm256_cvtepi32_ps(q0), vd0));
                    }
                }
                uint32_t m = 0u;
                for (; m + 3u < m_cnt; m += 4u) {
                    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
                    __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
                    for (uint32_t k = 0u; k < kb; k += 8u) {
                        __m256 w0 = _mm256_load_ps(w_deq[0] + k);
                        a0 = _mm256_fmadd_ps(_mm256_load_ps(x_pack[m + 0u] + k), w0, a0);
                        a1 = _mm256_fmadd_ps(_mm256_load_ps(x_pack[m + 1u] + k), w0, a1);
                        a2 = _mm256_fmadd_ps(_mm256_load_ps(x_pack[m + 2u] + k), w0, a2);
                        a3 = _mm256_fmadd_ps(_mm256_load_ps(x_pack[m + 3u] + k), w0, a3);
                    }
                    c_tile[m + 0u][nr] += hsum_m256(a0);
                    c_tile[m + 1u][nr] += hsum_m256(a1);
                    c_tile[m + 2u][nr] += hsum_m256(a2);
                    c_tile[m + 3u][nr] += hsum_m256(a3);
                }
                for (; m < m_cnt; m++) {
                    __m256 a0 = _mm256_setzero_ps();
                    for (uint32_t k = 0u; k < kb; k += 8u) {
                        a0 = _mm256_fmadd_ps(_mm256_load_ps(x_pack[m] + k), _mm256_load_ps(w_deq[0] + k), a0);
                    }
                    c_tile[m][nr] += hsum_m256(a0);
                }
            }
        }

        epilogue_store_tile_avx2(c_tile, m0, m_cnt, n0, n_cnt, N, c->act_type, c->bias, c->residual, c->out);
    }
}

/* INT8 x INT8 (Q8_0 x Q8_0) AVX2+FMA fast path (LILA_ASI_VISION_FAST_Q8=1).
 * Uses symmetric [-127, 127] activation quantization so vpmaddubsw(|w|, sign(x, w))
 * is exact and never overflows int16 even when w == -128. */
typedef struct {
    const float *x;
    uint32_t     K;
    uint32_t     nb;
    int8_t      *xq_qs;
    float       *xq_d;
} QuantX8Ctx;

static void quantize_x_q8_range(uint32_t m0, uint32_t m1, int tid, void *vctx)
{
    (void)tid;
    const QuantX8Ctx *q = (const QuantX8Ctx *)vctx;
    const uint32_t K = q->K;
    const uint32_t nb = q->nb;
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);
    const __m256i min_q8 = _mm256_set1_epi8(-127);

    for (uint32_t m = m0; m < m1; m++) {
        const float *xm = q->x + (size_t)m * K;
        int8_t *dst_qs = q->xq_qs + (size_t)m * nb * 32u;
        float *dst_d = q->xq_d + (size_t)m * nb;
        for (uint32_t b = 0u; b < nb; b++) {
            const float *sp = xm + (size_t)b * 32u;
            __m256 v0 = _mm256_loadu_ps(sp + 0u);
            __m256 v1 = _mm256_loadu_ps(sp + 8u);
            __m256 v2 = _mm256_loadu_ps(sp + 16u);
            __m256 v3 = _mm256_loadu_ps(sp + 24u);
            __m256 m01 = _mm256_max_ps(_mm256_andnot_ps(sign_mask, v0), _mm256_andnot_ps(sign_mask, v1));
            __m256 m23 = _mm256_max_ps(_mm256_andnot_ps(sign_mask, v2), _mm256_andnot_ps(sign_mask, v3));
            __m256 vmax = _mm256_max_ps(m01, m23);
            __m128 m128 = _mm_max_ps(_mm256_castps256_ps128(vmax), _mm256_extractf128_ps(vmax, 1));
            m128 = _mm_max_ps(m128, _mm_movehl_ps(m128, m128));
            m128 = _mm_max_ss(m128, _mm_shuffle_ps(m128, m128, 1));
            float amax = _mm_cvtss_f32(m128);
            float d = amax / 127.0f;
            float id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
            dst_d[b] = d;

            __m256 vid = _mm256_set1_ps(id);
            __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(v0, vid));
            __m256i i1 = _mm256_cvtps_epi32(_mm256_mul_ps(v1, vid));
            __m256i i2 = _mm256_cvtps_epi32(_mm256_mul_ps(v2, vid));
            __m256i i3 = _mm256_cvtps_epi32(_mm256_mul_ps(v3, vid));
            __m256i p01 = _mm256_permute4x64_epi64(_mm256_packs_epi32(i0, i1), 0xD8);
            __m256i p23 = _mm256_permute4x64_epi64(_mm256_packs_epi32(i2, i3), 0xD8);
            __m256i q8 = _mm256_permute4x64_epi64(_mm256_packs_epi16(p01, p23), 0xD8);
            q8 = _mm256_max_epi8(q8, min_q8);
            _mm256_store_si256((__m256i *)(dst_qs + (size_t)b * 32u), q8);
        }
    }
}

#define ASI_Q8_KB_BLKS 16u

static void linear_q8_q8_tile_range(uint32_t t0, uint32_t t1, int tid, void *vctx)
{
    (void)tid;
    const LinearQuantCtx *c = (const LinearQuantCtx *)vctx;
    const uint32_t M = c->M;
    const uint32_t N = c->N;
    const uint32_t nb = c->K / 32u;
    const uint32_t n_tile_sz = c->n_tile_size;
    const uint32_t n_ntiles = c->n_ntiles;
    const __m256i ones16 = _mm256_set1_epi16(1);

    _Alignas(32) float c_tile[ASI_GEMM_MB][ASI_GEMM_NB_MAX];
    _Alignas(32) int8_t w_qs[2][ASI_Q8_KB_BLKS * 32u];
    _Alignas(32) int8_t w_aw[2][ASI_Q8_KB_BLKS * 32u];
    float w_d[2][ASI_Q8_KB_BLKS];

    for (uint32_t tile_idx = t0; tile_idx < t1; tile_idx++) {
        uint32_t mt = tile_idx / n_ntiles;
        uint32_t nt = tile_idx % n_ntiles;
        uint32_t m0 = mt * ASI_GEMM_MB;
        uint32_t m_cnt = M - m0;
        if (m_cnt > ASI_GEMM_MB) m_cnt = ASI_GEMM_MB;
        uint32_t n0 = nt * n_tile_sz;
        uint32_t n_cnt = N - n0;
        if (n_cnt > n_tile_sz) n_cnt = n_tile_sz;

        for (uint32_t m = 0u; m < m_cnt; m++) {
            memset(c_tile[m], 0, (size_t)n_cnt * sizeof(float));
        }

        for (uint32_t b0 = 0u; b0 < nb; b0 += ASI_Q8_KB_BLKS) {
            uint32_t nb_tile = nb - b0;
            if (nb_tile > ASI_Q8_KB_BLKS) nb_tile = ASI_Q8_KB_BLKS;

            uint32_t nr = 0u;
            for (; nr + 1u < n_cnt; nr += 2u) {
                const uint8_t *w_row0 = c->w_data + (size_t)(n0 + nr + 0u) * c->row_bytes + (size_t)b0 * Q8_0_BLOCK_SIZE;
                const uint8_t *w_row1 = w_row0 + c->row_bytes;

                if (nr + 3u < n_cnt) {
                    const uint8_t *pw0 = w_row0 + 2u * c->row_bytes;
                    const uint8_t *pw1 = pw0 + c->row_bytes;
                    _mm_prefetch((const char *)pw0, _MM_HINT_T0);
                    _mm_prefetch((const char *)pw1, _MM_HINT_T0);
                }

                for (uint32_t b = 0u; b < nb_tile; b++) {
                    const uint8_t *blk0 = w_row0 + (size_t)b * Q8_0_BLOCK_SIZE;
                    const uint8_t *blk1 = w_row1 + (size_t)b * Q8_0_BLOCK_SIZE;
                    uint16_t d0_raw, d1_raw;
                    memcpy(&d0_raw, blk0, sizeof(d0_raw));
                    memcpy(&d1_raw, blk1, sizeof(d1_raw));
                    w_d[0][b] = fp16_to_fp32(d0_raw);
                    w_d[1][b] = fp16_to_fp32(d1_raw);
                    __m256i q0 = _mm256_loadu_si256((const __m256i *)(blk0 + 2u));
                    __m256i q1 = _mm256_loadu_si256((const __m256i *)(blk1 + 2u));
                    _mm256_store_si256((__m256i *)(w_qs[0] + b * 32u), q0);
                    _mm256_store_si256((__m256i *)(w_qs[1] + b * 32u), q1);
                    _mm256_store_si256((__m256i *)(w_aw[0] + b * 32u), _mm256_sign_epi8(q0, q0));
                    _mm256_store_si256((__m256i *)(w_aw[1] + b * 32u), _mm256_sign_epi8(q1, q1));
                }

                uint32_t m = 0u;
                for (; m + 3u < m_cnt; m += 4u) {
                    const int8_t *x0 = c->xq_qs + ((size_t)(m0 + m + 0u) * nb + b0) * 32u;
                    const int8_t *x1 = c->xq_qs + ((size_t)(m0 + m + 1u) * nb + b0) * 32u;
                    const int8_t *x2 = c->xq_qs + ((size_t)(m0 + m + 2u) * nb + b0) * 32u;
                    const int8_t *x3 = c->xq_qs + ((size_t)(m0 + m + 3u) * nb + b0) * 32u;
                    const float *dx0 = c->xq_d + (size_t)(m0 + m + 0u) * nb + b0;
                    const float *dx1 = c->xq_d + (size_t)(m0 + m + 1u) * nb + b0;
                    const float *dx2 = c->xq_d + (size_t)(m0 + m + 2u) * nb + b0;
                    const float *dx3 = c->xq_d + (size_t)(m0 + m + 3u) * nb + b0;

                    __m256 a00 = _mm256_setzero_ps(), a01 = _mm256_setzero_ps();
                    __m256 a10 = _mm256_setzero_ps(), a11 = _mm256_setzero_ps();
                    __m256 a20 = _mm256_setzero_ps(), a21 = _mm256_setzero_ps();
                    __m256 a30 = _mm256_setzero_ps(), a31 = _mm256_setzero_ps();

                    for (uint32_t b = 0u; b < nb_tile; b++) {
                        __m256i w0 = _mm256_load_si256((const __m256i *)(w_qs[0] + b * 32u));
                        __m256i aw0 = _mm256_load_si256((const __m256i *)(w_aw[0] + b * 32u));
                        __m256i w1 = _mm256_load_si256((const __m256i *)(w_qs[1] + b * 32u));
                        __m256i aw1 = _mm256_load_si256((const __m256i *)(w_aw[1] + b * 32u));
                        float dw0 = w_d[0][b], dw1 = w_d[1][b];

                        __m256i xv, p0, p1;
                        xv = _mm256_load_si256((const __m256i *)(x0 + b * 32u));
                        p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(aw0, _mm256_sign_epi8(xv, w0)), ones16);
                        p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(aw1, _mm256_sign_epi8(xv, w1)), ones16);
                        a00 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p0), _mm256_set1_ps(dw0 * dx0[b]), a00);
                        a01 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p1), _mm256_set1_ps(dw1 * dx0[b]), a01);

                        xv = _mm256_load_si256((const __m256i *)(x1 + b * 32u));
                        p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(aw0, _mm256_sign_epi8(xv, w0)), ones16);
                        p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(aw1, _mm256_sign_epi8(xv, w1)), ones16);
                        a10 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p0), _mm256_set1_ps(dw0 * dx1[b]), a10);
                        a11 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p1), _mm256_set1_ps(dw1 * dx1[b]), a11);

                        xv = _mm256_load_si256((const __m256i *)(x2 + b * 32u));
                        p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(aw0, _mm256_sign_epi8(xv, w0)), ones16);
                        p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(aw1, _mm256_sign_epi8(xv, w1)), ones16);
                        a20 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p0), _mm256_set1_ps(dw0 * dx2[b]), a20);
                        a21 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p1), _mm256_set1_ps(dw1 * dx2[b]), a21);

                        xv = _mm256_load_si256((const __m256i *)(x3 + b * 32u));
                        p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(aw0, _mm256_sign_epi8(xv, w0)), ones16);
                        p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(aw1, _mm256_sign_epi8(xv, w1)), ones16);
                        a30 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p0), _mm256_set1_ps(dw0 * dx3[b]), a30);
                        a31 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p1), _mm256_set1_ps(dw1 * dx3[b]), a31);
                    }
                    c_tile[m + 0u][nr + 0u] += hsum_m256(a00); c_tile[m + 0u][nr + 1u] += hsum_m256(a01);
                    c_tile[m + 1u][nr + 0u] += hsum_m256(a10); c_tile[m + 1u][nr + 1u] += hsum_m256(a11);
                    c_tile[m + 2u][nr + 0u] += hsum_m256(a20); c_tile[m + 2u][nr + 1u] += hsum_m256(a21);
                    c_tile[m + 3u][nr + 0u] += hsum_m256(a30); c_tile[m + 3u][nr + 1u] += hsum_m256(a31);
                }
                for (; m < m_cnt; m++) {
                    const int8_t *xm = c->xq_qs + ((size_t)(m0 + m) * nb + b0) * 32u;
                    const float *dxm = c->xq_d + (size_t)(m0 + m) * nb + b0;
                    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
                    for (uint32_t b = 0u; b < nb_tile; b++) {
                        __m256i w0 = _mm256_load_si256((const __m256i *)(w_qs[0] + b * 32u));
                        __m256i aw0 = _mm256_load_si256((const __m256i *)(w_aw[0] + b * 32u));
                        __m256i w1 = _mm256_load_si256((const __m256i *)(w_qs[1] + b * 32u));
                        __m256i aw1 = _mm256_load_si256((const __m256i *)(w_aw[1] + b * 32u));
                        __m256i xv = _mm256_load_si256((const __m256i *)(xm + b * 32u));
                        __m256i p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(aw0, _mm256_sign_epi8(xv, w0)), ones16);
                        __m256i p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(aw1, _mm256_sign_epi8(xv, w1)), ones16);
                        a0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p0), _mm256_set1_ps(w_d[0][b] * dxm[b]), a0);
                        a1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p1), _mm256_set1_ps(w_d[1][b] * dxm[b]), a1);
                    }
                    c_tile[m][nr + 0u] += hsum_m256(a0);
                    c_tile[m][nr + 1u] += hsum_m256(a1);
                }
            }
            if (nr < n_cnt) {
                const uint8_t *w_row0 = c->w_data + (size_t)(n0 + nr) * c->row_bytes + (size_t)b0 * Q8_0_BLOCK_SIZE;
                for (uint32_t b = 0u; b < nb_tile; b++) {
                    const uint8_t *blk0 = w_row0 + (size_t)b * Q8_0_BLOCK_SIZE;
                    uint16_t d0_raw;
                    memcpy(&d0_raw, blk0, sizeof(d0_raw));
                    w_d[0][b] = fp16_to_fp32(d0_raw);
                    __m256i q0 = _mm256_loadu_si256((const __m256i *)(blk0 + 2u));
                    _mm256_store_si256((__m256i *)(w_qs[0] + b * 32u), q0);
                    _mm256_store_si256((__m256i *)(w_aw[0] + b * 32u), _mm256_sign_epi8(q0, q0));
                }
                for (uint32_t m = 0u; m < m_cnt; m++) {
                    const int8_t *xm = c->xq_qs + ((size_t)(m0 + m) * nb + b0) * 32u;
                    const float *dxm = c->xq_d + (size_t)(m0 + m) * nb + b0;
                    __m256 a0 = _mm256_setzero_ps();
                    for (uint32_t b = 0u; b < nb_tile; b++) {
                        __m256i w0 = _mm256_load_si256((const __m256i *)(w_qs[0] + b * 32u));
                        __m256i aw0 = _mm256_load_si256((const __m256i *)(w_aw[0] + b * 32u));
                        __m256i xv = _mm256_load_si256((const __m256i *)(xm + b * 32u));
                        __m256i p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(aw0, _mm256_sign_epi8(xv, w0)), ones16);
                        a0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p0), _mm256_set1_ps(w_d[0][b] * dxm[b]), a0);
                    }
                    c_tile[m][nr] += hsum_m256(a0);
                }
            }
        }

        epilogue_store_tile_avx2(c_tile, m0, m_cnt, n0, n_cnt, N, c->act_type, c->bias, c->residual, c->out);
    }
}
#endif /* __AVX2__ && __FMA__ */

static void linear_quant_range(uint32_t n0, uint32_t n1, int tid, void *vctx)
{
    (void)tid;
    const LinearQuantCtx *c = (const LinearQuantCtx *)vctx;
    uint32_t M = c->M;
    uint32_t N = c->N;
    uint32_t K = c->K;
    uint32_t act = c->act_type;
    const float *bias = c->bias;
    const float *res = c->residual;
    float *out = c->out;

    if (c->quant_type == 0u) {
#if defined(__AVX2__) && defined(__FMA__)
        uint32_t k8 = K & ~7u;
        for (uint32_t n = n0; n < n1; n++) {
            const float *w_row = (const float *)(c->w_data + (size_t)n * c->row_bytes);
            float b_val = bias ? bias[n] : 0.0f;
            uint32_t m = 0u;
            for (; m + 3u < M; m += 4u) {
                const float *x0 = c->x + (size_t)(m + 0u) * K;
                const float *x1 = c->x + (size_t)(m + 1u) * K;
                const float *x2 = c->x + (size_t)(m + 2u) * K;
                const float *x3 = c->x + (size_t)(m + 3u) * K;
                __m256 a0 = _mm256_setzero_ps();
                __m256 a1 = _mm256_setzero_ps();
                __m256 a2 = _mm256_setzero_ps();
                __m256 a3 = _mm256_setzero_ps();
                for (uint32_t k = 0u; k < k8; k += 8u) {
                    __m256 wf = _mm256_loadu_ps(w_row + k);
                    a0 = _mm256_fmadd_ps(wf, _mm256_loadu_ps(x0 + k), a0);
                    a1 = _mm256_fmadd_ps(wf, _mm256_loadu_ps(x1 + k), a1);
                    a2 = _mm256_fmadd_ps(wf, _mm256_loadu_ps(x2 + k), a2);
                    a3 = _mm256_fmadd_ps(wf, _mm256_loadu_ps(x3 + k), a3);
                }
                float s0 = hsum_m256(a0), s1 = hsum_m256(a1), s2 = hsum_m256(a2), s3 = hsum_m256(a3);
                for (uint32_t k = k8; k < K; k++) {
                    float wv = w_row[k];
                    s0 += wv * x0[k];
                    s1 += wv * x1[k];
                    s2 += wv * x2[k];
                    s3 += wv * x3[k];
                }
                float vals[4] = { s0 + b_val, s1 + b_val, s2 + b_val, s3 + b_val };
                for (uint32_t tm = 0u; tm < 4u; tm++) {
                    float v = (act == ASI_ACT_NONE) ? vals[tm] : apply_act_scalar(vals[tm], act);
                    size_t oi = (size_t)(m + tm) * N + n;
                    out[oi] = res ? (res[oi] + v) : v;
                }
            }
            for (; m < M; m++) {
                const float *xm = c->x + (size_t)m * K;
                float v = lila_fp32_dot_slice(w_row, xm, (int)K) + b_val;
                if (act != ASI_ACT_NONE) v = apply_act_scalar(v, act);
                size_t oi = (size_t)m * N + n;
                out[oi] = res ? (res[oi] + v) : v;
            }
        }
        return;
#endif
    }

    /* Generic quantized fallback */
    for (uint32_t n = n0; n < n1; n++) {
        const uint8_t *w_row = c->w_data + (size_t)n * c->row_bytes;
        float b_val = bias ? bias[n] : 0.0f;
        for (uint32_t m = 0u; m < M; m++) {
            const float *xm = c->x + (size_t)m * K;
            float v = lila_quant_dot_fp32(w_row, xm, (int)K, c->quant_type, 0) + b_val;
            if (act != ASI_ACT_NONE) v = apply_act_scalar(v, act);
            size_t oi = (size_t)m * N + n;
            out[oi] = res ? (res[oi] + v) : v;
        }
    }
}

int asi_k_linear_quant_bias(AsiKernelCtx *ctx,
                            const uint8_t *w_data,
                            uint32_t quant_type,
                            const float *bias,
                            const float *x,
                            uint32_t M,
                            uint32_t N,
                            uint32_t K,
                            uint32_t act_type,
                            const float *residual,
                            float *out)
{
    if (!w_data || !x || !out || M == 0u || N == 0u || K == 0u) return -1;
    size_t row_bytes = lila_quant_row_bytes(quant_type, (int)K, 0);
    if (row_bytes == 0u) return -1;

#if defined(__AVX2__) && defined(__FMA__)
    if (quant_type == 8u && (K & 31u) == 0u) {
        uint32_t n_tile_sz = (N >= 128u) ? 64u : ((N >= 32u) ? 16u : 8u);
        uint32_t n_mtiles = (M + ASI_GEMM_MB - 1u) / ASI_GEMM_MB;
        uint32_t n_ntiles = (N + n_tile_sz - 1u) / n_tile_sz;
        uint32_t total_tiles = n_mtiles * n_ntiles;

        if (asi_fast_q8_enabled()) {
            uint32_t nb = K / 32u;
            int8_t *xq_qs = (int8_t *)asi_aligned_alloc(32u, (size_t)M * nb * 32u);
            float *xq_d = (float *)malloc((size_t)M * nb * sizeof(float));
            if (xq_qs && xq_d) {
                QuantX8Ctx qctx = {x, K, nb, xq_qs, xq_d};
                asi_kernel_parallel_for(ctx, M, 16u, quantize_x_q8_range, &qctx);
                if (!(ctx && ctx->cancel_flag && *ctx->cancel_flag)) {
                    LinearQuantCtx lc = {
                        w_data, row_bytes, quant_type, bias, x, M, N, K, act_type, residual, out,
                        ASI_GEMM_MB, n_tile_sz, n_ntiles, xq_qs, xq_d
                    };
                    asi_kernel_parallel_for(ctx, total_tiles, 1u, linear_q8_q8_tile_range, &lc);
                }
                free(xq_d);
                asi_aligned_free(xq_qs);
                return (ctx && ctx->cancel_flag && *ctx->cancel_flag) ? -2 : 0;
            }
            free(xq_d);
            asi_aligned_free(xq_qs);
        }

        LinearQuantCtx lc = {
            w_data, row_bytes, quant_type, bias, x, M, N, K, act_type, residual, out,
            ASI_GEMM_MB, n_tile_sz, n_ntiles, NULL, NULL
        };
        asi_kernel_parallel_for(ctx, total_tiles, 1u, linear_q8_fp32_tile_range, &lc);
        return (ctx && ctx->cancel_flag && *ctx->cancel_flag) ? -2 : 0;
    }
#endif

    LinearQuantCtx lc = {
        w_data, row_bytes, quant_type, bias, x, M, N, K, act_type, residual, out,
        0u, 0u, 0u, NULL, NULL
    };
    uint32_t grain = (N >= 128u) ? 16u : 4u;
    asi_kernel_parallel_for(ctx, N, grain, linear_quant_range, &lc);
    return (ctx && ctx->cancel_flag && *ctx->cancel_flag) ? -2 : 0;
}

typedef struct {
    const float *x;
    const float *w;
    const float *b;
    uint32_t     D;
    float        eps;
    float       *out;
} NormCtx;

static void layernorm_range(uint32_t r0, uint32_t r1, int tid, void *vctx)
{
    (void)tid;
    const NormCtx *c = (const NormCtx *)vctx;
    uint32_t D = c->D;
    for (uint32_t r = r0; r < r1; r++) {
        const float *xr = c->x + (size_t)r * D;
        float *orow = c->out + (size_t)r * D;
        double sum = 0.0;
        uint32_t i = 0u;
#if defined(__AVX2__)
        __m256d vsum0 = _mm256_setzero_pd();
        __m256d vsum1 = _mm256_setzero_pd();
        for (; i + 7u < D; i += 8u) {
            vsum0 = _mm256_add_pd(vsum0, _mm256_cvtps_pd(_mm_loadu_ps(xr + i)));
            vsum1 = _mm256_add_pd(vsum1, _mm256_cvtps_pd(_mm_loadu_ps(xr + i + 4u)));
        }
        __m256d vs = _mm256_add_pd(vsum0, vsum1);
        __m128d vs2 = _mm_add_pd(_mm256_castpd256_pd128(vs), _mm256_extractf128_pd(vs, 1));
        sum = _mm_cvtsd_f64(_mm_add_sd(vs2, _mm_unpackhi_pd(vs2, vs2)));
#endif
        for (; i < D; i++) sum += (double)xr[i];
        double mean = sum / (double)D;
        double var_sum = 0.0;
        i = 0u;
#if defined(__AVX2__) && defined(__FMA__)
        __m256d vmean = _mm256_set1_pd(mean);
        __m256d vv0 = _mm256_setzero_pd();
        __m256d vv1 = _mm256_setzero_pd();
        for (; i + 7u < D; i += 8u) {
            __m256d d0 = _mm256_sub_pd(_mm256_cvtps_pd(_mm_loadu_ps(xr + i)), vmean);
            __m256d d1 = _mm256_sub_pd(_mm256_cvtps_pd(_mm_loadu_ps(xr + i + 4u)), vmean);
            vv0 = _mm256_fmadd_pd(d0, d0, vv0);
            vv1 = _mm256_fmadd_pd(d1, d1, vv1);
        }
        __m256d vv = _mm256_add_pd(vv0, vv1);
        __m128d vv2 = _mm_add_pd(_mm256_castpd256_pd128(vv), _mm256_extractf128_pd(vv, 1));
        var_sum = _mm_cvtsd_f64(_mm_add_sd(vv2, _mm_unpackhi_pd(vv2, vv2)));
#endif
        for (; i < D; i++) {
            double d = (double)xr[i] - mean;
            var_sum += d * d;
        }
        float inv_std = (float)(1.0 / sqrt(var_sum / (double)D + (double)c->eps));
        float m_f = (float)mean;
        i = 0u;
#if defined(__AVX2__) && defined(__FMA__)
        __m256 vm = _mm256_set1_ps(m_f);
        __m256 vi = _mm256_set1_ps(inv_std);
        if (c->b) {
            for (; i + 7u < D; i += 8u) {
                __m256 xv = _mm256_sub_ps(_mm256_loadu_ps(xr + i), vm);
                __m256 wv = _mm256_mul_ps(_mm256_mul_ps(xv, vi), _mm256_loadu_ps(c->w + i));
                _mm256_storeu_ps(orow + i, _mm256_add_ps(wv, _mm256_loadu_ps(c->b + i)));
            }
        } else {
            for (; i + 7u < D; i += 8u) {
                __m256 xv = _mm256_sub_ps(_mm256_loadu_ps(xr + i), vm);
                _mm256_storeu_ps(orow + i, _mm256_mul_ps(_mm256_mul_ps(xv, vi), _mm256_loadu_ps(c->w + i)));
            }
        }
#endif
        for (; i < D; i++) {
            float y = ((xr[i] - m_f) * inv_std) * c->w[i];
            orow[i] = c->b ? (y + c->b[i]) : y;
        }
    }
}

void asi_k_layernorm(AsiKernelCtx *ctx,
                     const float *x,
                     const float *w,
                     const float *b,
                     uint32_t M,
                     uint32_t D,
                     float eps,
                     float *out)
{
    if (!x || !w || !out || M == 0u || D == 0u) return;
    NormCtx nc = {x, w, b, D, eps, out};
    asi_kernel_parallel_for(ctx, M, 8u, layernorm_range, &nc);
}

static void rmsnorm_range(uint32_t r0, uint32_t r1, int tid, void *vctx)
{
    (void)tid;
    const NormCtx *c = (const NormCtx *)vctx;
    uint32_t D = c->D;
    for (uint32_t r = r0; r < r1; r++) {
        const float *xr = c->x + (size_t)r * D;
        float *orow = c->out + (size_t)r * D;
        double ss = 0.0;
        for (uint32_t i = 0u; i < D; i++) {
            double v = (double)xr[i];
            ss += v * v;
        }
        float inv_rms = (float)(1.0 / sqrt(ss / (double)D + (double)c->eps));
        for (uint32_t i = 0u; i < D; i++) {
            orow[i] = xr[i] * inv_rms * c->w[i];
        }
    }
}

void asi_k_rmsnorm(AsiKernelCtx *ctx,
                   const float *x,
                   const float *w,
                   uint32_t M,
                   uint32_t D,
                   float eps,
                   float *out)
{
    if (!x || !w || !out || M == 0u || D == 0u) return;
    NormCtx nc = {x, w, NULL, D, eps, out};
    asi_kernel_parallel_for(ctx, M, 8u, rmsnorm_range, &nc);
}

void asi_k_activation_inplace(AsiKernelCtx *ctx,
                              float *x,
                              size_t n,
                              uint32_t act_type)
{
    (void)ctx;
    if (!x || n == 0u || act_type == ASI_ACT_NONE) return;
    for (size_t i = 0u; i < n; i++) {
        x[i] = apply_act_scalar(x[i], act_type);
    }
}

void asi_k_add_inplace(AsiKernelCtx *ctx,
                       float *dst,
                       const float *src,
                       size_t n)
{
    (void)ctx;
    if (!dst || !src || n == 0u) return;
    size_t i = 0u;
#if defined(__AVX2__)
    for (; i + 7u < n; i += 8u) {
        __m256 d = _mm256_loadu_ps(dst + i);
        __m256 s = _mm256_loadu_ps(src + i);
        _mm256_storeu_ps(dst + i, _mm256_add_ps(d, s));
    }
#endif
    for (; i < n; i++) dst[i] += src[i];
}

void asi_k_resize_rgb_bilinear_chw(const uint8_t *rgb_u8,
                                   uint32_t src_w,
                                   uint32_t src_h,
                                   uint32_t stride_bytes,
                                   uint32_t dst_w,
                                   uint32_t dst_h,
                                   const float mean[3],
                                   const float std_v[3],
                                   float *out_chw)
{
    if (!rgb_u8 || !out_chw || src_w == 0u || src_h == 0u || dst_w == 0u || dst_h == 0u) return;
    uint32_t stride = stride_bytes ? stride_bytes : src_w * 3u;
    size_t plane = (size_t)dst_h * (size_t)dst_w;

    for (uint32_t y = 0u; y < dst_h; y++) {
        float yf = (dst_h > 1u && src_h > 1u)
            ? (float)((double)y * (double)(src_h - 1u) / (double)(dst_h - 1u))
            : 0.0f;
        uint32_t y0 = (uint32_t)floorf(yf);
        if (y0 >= src_h) y0 = src_h - 1u;
        uint32_t y1 = (y0 + 1u < src_h) ? (y0 + 1u) : (src_h - 1u);
        float wy = yf - (float)y0;
        float iwy = 1.0f - wy;

        const uint8_t *row0 = rgb_u8 + (size_t)y0 * stride;
        const uint8_t *row1 = rgb_u8 + (size_t)y1 * stride;

        for (uint32_t x = 0u; x < dst_w; x++) {
            float xf = (dst_w > 1u && src_w > 1u)
                ? (float)((double)x * (double)(src_w - 1u) / (double)(dst_w - 1u))
                : 0.0f;
            uint32_t x0 = (uint32_t)floorf(xf);
            if (x0 >= src_w) x0 = src_w - 1u;
            uint32_t x1 = (x0 + 1u < src_w) ? (x0 + 1u) : (src_w - 1u);
            float wx = xf - (float)x0;
            float iwx = 1.0f - wx;

            float w00 = iwy * iwx;
            float w01 = iwy * wx;
            float w10 = wy * iwx;
            float w11 = wy * wx;

            const uint8_t *p00 = row0 + (size_t)x0 * 3u;
            const uint8_t *p01 = row0 + (size_t)x1 * 3u;
            const uint8_t *p10 = row1 + (size_t)x0 * 3u;
            const uint8_t *p11 = row1 + (size_t)x1 * 3u;

            size_t dst_idx = (size_t)y * dst_w + x;
            for (int c = 0; c < 3; c++) {
                float v = (float)p00[c] * w00 + (float)p01[c] * w01
                        + (float)p10[c] * w10 + (float)p11[c] * w11;
                out_chw[(size_t)c * plane + dst_idx] = (v / 255.0f - mean[c]) / std_v[c];
            }
        }
    }
}

void asi_k_patch_extract_merged(const float *img_chw,
                                uint32_t new_h,
                                uint32_t new_w,
                                uint32_t patch_size,
                                uint32_t merge_size,
                                float *out_patches)
{
    if (!img_chw || !out_patches || patch_size == 0u || merge_size == 0u) return;
    uint32_t gh = new_h / patch_size;
    uint32_t gw = new_w / patch_size;
    uint32_t hm = gh / merge_size;
    uint32_t wm = gw / merge_size;
    uint32_t patch_vec = 3u * patch_size * patch_size;
    size_t plane = (size_t)new_h * (size_t)new_w;

    for (uint32_t yb = 0u; yb < hm; yb++) {
        for (uint32_t xb = 0u; xb < wm; xb++) {
            for (uint32_t dy = 0u; dy < merge_size; dy++) {
                for (uint32_t dx = 0u; dx < merge_size; dx++) {
                    uint32_t p_idx = ((yb * wm + xb) * merge_size + dy) * merge_size + dx;
                    float *dst = out_patches + (size_t)p_idx * patch_vec;
                    uint32_t y0 = (yb * merge_size + dy) * patch_size;
                    uint32_t x0 = (xb * merge_size + dx) * patch_size;
                    for (uint32_t c = 0u; c < 3u; c++) {
                        const float *cplane = img_chw + (size_t)c * plane;
                        for (uint32_t py = 0u; py < patch_size; py++) {
                            const float *src = cplane + (size_t)(y0 + py) * new_w + x0;
                            memcpy(dst + ((size_t)c * patch_size + py) * patch_size,
                                   src, patch_size * sizeof(float));
                        }
                    }
                }
            }
        }
    }
}

typedef struct {
    const float *patches;
    const float *w0;
    const float *w1;
    const float *bias;
    uint32_t     n_patches;
    uint32_t     dim;
    uint32_t     patch_vec;
    float       *out;
} Conv3dPatchCtx;

static void conv3d_patch_range(uint32_t d0, uint32_t d1, int tid, void *vctx)
{
    (void)tid;
    const Conv3dPatchCtx *c = (const Conv3dPatchCtx *)vctx;
    uint32_t np = c->n_patches;
    uint32_t dim = c->dim;
    uint32_t pv = c->patch_vec;

    for (uint32_t d_i = d0; d_i < d1; d_i++) {
        const float *r0 = c->w0 + (size_t)d_i * pv;
        const float *r1 = c->w1 ? (c->w1 + (size_t)d_i * pv) : NULL;
        float pb = c->bias ? c->bias[d_i] : 0.0f;
        for (uint32_t p_i = 0u; p_i < np; p_i++) {
            const float *pvec = c->patches + (size_t)p_i * pv;
            float acc = pb;
            uint32_t k = 0u;
#if defined(__AVX2__) && defined(__FMA__)
            __m256 va = _mm256_setzero_ps();
            if (r1) {
                for (; k + 7u < pv; k += 8u) {
                    __m256 ws = _mm256_add_ps(_mm256_loadu_ps(r0 + k), _mm256_loadu_ps(r1 + k));
                    va = _mm256_fmadd_ps(_mm256_loadu_ps(pvec + k), ws, va);
                }
            } else {
                for (; k + 7u < pv; k += 8u) {
                    va = _mm256_fmadd_ps(_mm256_loadu_ps(pvec + k), _mm256_loadu_ps(r0 + k), va);
                }
            }
            acc += hsum_m256(va);
#endif
            if (r1) {
                for (; k < pv; k++) acc += pvec[k] * (r0[k] + r1[k]);
            } else {
                for (; k < pv; k++) acc += pvec[k] * r0[k];
            }
            c->out[(size_t)p_i * dim + d_i] = acc;
        }
    }
}

void asi_k_conv3d_patch_pair(AsiKernelCtx *ctx,
                             const float *patches,
                             const float *w0,
                             const float *w1,
                             const float *bias,
                             uint32_t n_patches,
                             uint32_t dim,
                             uint32_t patch_vec,
                             float *out)
{
    if (!patches || !w0 || !out || n_patches == 0u || dim == 0u || patch_vec == 0u) return;
    Conv3dPatchCtx pc = {patches, w0, w1, bias, n_patches, dim, patch_vec, out};
    asi_kernel_parallel_for(ctx, dim, 32u, conv3d_patch_range, &pc);
}

int asi_k_pos_embd_2d_interp_add(const float *pos_table,
                                 uint32_t table_entries,
                                 uint32_t gh,
                                 uint32_t gw,
                                 uint32_t merge_size,
                                 uint32_t dim,
                                 float *x_inout)
{
    if (!pos_table || !x_inout || table_entries == 0u || merge_size == 0u) return -1;
    uint32_t base = (uint32_t)nearbyint(sqrt((double)table_entries));
    if (base == 0u || base * base != table_entries) return -1;

    uint32_t hm = gh / merge_size;
    uint32_t wm = gw / merge_size;
    for (uint32_t yb = 0u; yb < hm; yb++) {
        for (uint32_t xb = 0u; xb < wm; xb++) {
            for (uint32_t dy = 0u; dy < merge_size; dy++) {
                for (uint32_t dx = 0u; dx < merge_size; dx++) {
                    uint32_t gy = yb * merge_size + dy;
                    uint32_t gx = xb * merge_size + dx;
                    uint32_t p_idx = ((yb * wm + xb) * merge_size + dy) * merge_size + dx;

                    float yf = (gh > 1u && base > 1u)
                        ? (float)((double)gy * (double)(base - 1u) / (double)(gh - 1u))
                        : 0.0f;
                    float xf = (gw > 1u && base > 1u)
                        ? (float)((double)gx * (double)(base - 1u) / (double)(gw - 1u))
                        : 0.0f;
                    uint32_t y0 = (uint32_t)floorf(yf);
                    uint32_t x0 = (uint32_t)floorf(xf);
                    if (y0 >= base) y0 = base - 1u;
                    if (x0 >= base) x0 = base - 1u;
                    uint32_t y1 = (y0 + 1u < base) ? (y0 + 1u) : (base - 1u);
                    uint32_t x1 = (x0 + 1u < base) ? (x0 + 1u) : (base - 1u);
                    float wy = yf - (float)y0;
                    float wx = xf - (float)x0;
                    float w00 = (1.0f - wy) * (1.0f - wx);
                    float w01 = (1.0f - wy) * wx;
                    float w10 = wy * (1.0f - wx);
                    float w11 = wy * wx;

                    const float *p00 = pos_table + ((size_t)y0 * base + x0) * dim;
                    const float *p01 = pos_table + ((size_t)y0 * base + x1) * dim;
                    const float *p10 = pos_table + ((size_t)y1 * base + x0) * dim;
                    const float *p11 = pos_table + ((size_t)y1 * base + x1) * dim;
                    float *xdst = x_inout + (size_t)p_idx * dim;
                    for (uint32_t d = 0u; d < dim; d++) {
                        xdst[d] += p00[d] * w00 + p01[d] * w01 + p10[d] * w10 + p11[d] * w11;
                    }
                }
            }
        }
    }
    return 0;
}

void asi_k_rope_2d_vision_tables(uint32_t gh,
                                 uint32_t gw,
                                 uint32_t merge_size,
                                 uint32_t head_dim,
                                 float theta,
                                 float *cos_out,
                                 float *sin_out)
{
    if (!cos_out || !sin_out || merge_size == 0u || head_dim < 4u) return;
    uint32_t rope_half = head_dim / 2u;
    uint32_t quarter = rope_half / 2u;
    if (quarter > 128u) return;
    float inv_freq[128];
    for (uint32_t k = 0u; k < quarter; k++) {
        inv_freq[k] = 1.0f / powf(theta, (float)(2u * k) / (float)rope_half);
    }
    uint32_t hm = gh / merge_size;
    uint32_t wm = gw / merge_size;
    for (uint32_t yb = 0u; yb < hm; yb++) {
        for (uint32_t xb = 0u; xb < wm; xb++) {
            for (uint32_t dy = 0u; dy < merge_size; dy++) {
                for (uint32_t dx = 0u; dx < merge_size; dx++) {
                    float py = (float)(yb * merge_size + dy);
                    float px = (float)(xb * merge_size + dx);
                    uint32_t p_idx = ((yb * wm + xb) * merge_size + dy) * merge_size + dx;
                    float *cdst = cos_out + (size_t)p_idx * rope_half;
                    float *sdst = sin_out + (size_t)p_idx * rope_half;
                    for (uint32_t k = 0u; k < quarter; k++) {
                        float ay = py * inv_freq[k];
                        float ax = px * inv_freq[k];
                        cdst[k] = cosf(ay);
                        sdst[k] = sinf(ay);
                        cdst[quarter + k] = cosf(ax);
                        sdst[quarter + k] = sinf(ax);
                    }
                }
            }
        }
    }
}

typedef struct {
    float       *qkv;
    const float *cos_t;
    const float *sin_t;
    uint32_t     n_heads;
    uint32_t     head_dim;
} Rope2dQkvCtx;

static void rope_2d_qkv_range(uint32_t p0, uint32_t p1, int tid, void *vctx)
{
    (void)tid;
    const Rope2dQkvCtx *c = (const Rope2dQkvCtx *)vctx;
    uint32_t dim = c->n_heads * c->head_dim;
    uint32_t half = c->head_dim / 2u;
    for (uint32_t p_idx = p0; p_idx < p1; p_idx++) {
        float *qkv_p = c->qkv + (size_t)p_idx * 3u * dim;
        const float *cs = c->cos_t + (size_t)p_idx * half;
        const float *sn = c->sin_t + (size_t)p_idx * half;
        for (int qk = 0; qk < 2; qk++) {
            float *vec_base = qkv_p + (size_t)qk * dim;
            for (uint32_t h = 0u; h < c->n_heads; h++) {
                float *vh = vec_base + (size_t)h * c->head_dim;
                uint32_t i = 0u;
#if defined(__AVX2__) && defined(__FMA__)
                for (; i + 7u < half; i += 8u) {
                    __m256 v1 = _mm256_loadu_ps(vh + i);
                    __m256 v2 = _mm256_loadu_ps(vh + i + half);
                    __m256 vc = _mm256_loadu_ps(cs + i);
                    __m256 vs = _mm256_loadu_ps(sn + i);
                    _mm256_storeu_ps(vh + i, _mm256_fmsub_ps(v1, vc, _mm256_mul_ps(v2, vs)));
                    _mm256_storeu_ps(vh + i + half, _mm256_fmadd_ps(v1, vs, _mm256_mul_ps(v2, vc)));
                }
#endif
                for (; i < half; i++) {
                    float v1 = vh[i];
                    float v2 = vh[i + half];
                    vh[i] = v1 * cs[i] - v2 * sn[i];
                    vh[i + half] = v1 * sn[i] + v2 * cs[i];
                }
            }
        }
    }
}

void asi_k_rope_2d_qkv_apply(AsiKernelCtx *ctx,
                             float *qkv,
                             const float *cos_t,
                             const float *sin_t,
                             uint32_t n_patches,
                             uint32_t n_heads,
                             uint32_t head_dim)
{
    if (!qkv || !cos_t || !sin_t || n_patches == 0u) return;
    Rope2dQkvCtx rc = {qkv, cos_t, sin_t, n_heads, head_dim};
    asi_kernel_parallel_for(ctx, n_patches, 16u, rope_2d_qkv_range, &rc);
}

typedef struct {
    const float *qkv;
    uint32_t     n_patches;
    uint32_t     n_heads;
    uint32_t     head_dim;
    float        scale;
    float       *out;
    const volatile int *cancel_flag;
} SdpaBidirCtx;

static void sdpa_bidir_head_range(uint32_t h0, uint32_t h1, int tid, void *vctx)
{
    (void)tid;
    const SdpaBidirCtx *c = (const SdpaBidirCtx *)vctx;
    const uint32_t n_patches = c->n_patches;
    const uint32_t n_heads = c->n_heads;
    const uint32_t hd = c->head_dim;
    const uint32_t dim = n_heads * hd;
    const size_t qkv_stride = (size_t)3u * dim;

    /* Pack K_h and V_h contiguously per head so K/V stream cleanly out of L2 with zero TLB thrash */
    float *k_pack = (float *)malloc((size_t)n_patches * hd * sizeof(float));
    float *v_pack = (float *)malloc((size_t)n_patches * hd * sizeof(float));
    float *scores_all = (float *)malloc((size_t)4u * n_patches * sizeof(float));
    if (!k_pack || !v_pack || !scores_all) {
        free(scores_all);
        free(v_pack);
        free(k_pack);
        return;
    }

    float *s0 = scores_all + (size_t)0u * n_patches;
#if defined(__AVX2__) && defined(__FMA__)
    float *s1 = scores_all + (size_t)1u * n_patches;
    float *s2 = scores_all + (size_t)2u * n_patches;
    float *s3 = scores_all + (size_t)3u * n_patches;
#endif

    for (uint32_t hi = h0; hi < h1; hi++) {
        if (c->cancel_flag && *c->cancel_flag) break;

        for (uint32_t kj = 0u; kj < n_patches; kj++) {
            const float *base = c->qkv + (size_t)kj * qkv_stride + (size_t)hi * hd;
            memcpy(k_pack + (size_t)kj * hd, base + dim, (size_t)hd * sizeof(float));
            memcpy(v_pack + (size_t)kj * hd, base + 2u * dim, (size_t)hd * sizeof(float));
        }

        uint32_t qi = 0u;
#if defined(__AVX2__) && defined(__FMA__)
        const uint32_t hd8 = hd & ~7u;
        const uint32_t hd16 = hd & ~15u;
        _Alignas(32) float q_stack[4][128];
        for (; qi + 3u < n_patches; qi += 4u) {
            const float *q0 = c->qkv + (size_t)(qi + 0u) * qkv_stride + (size_t)hi * hd;
            const float *q1 = c->qkv + (size_t)(qi + 1u) * qkv_stride + (size_t)hi * hd;
            const float *q2 = c->qkv + (size_t)(qi + 2u) * qkv_stride + (size_t)hi * hd;
            const float *q3 = c->qkv + (size_t)(qi + 3u) * qkv_stride + (size_t)hi * hd;
            if (hd <= 128u) {
                memcpy(q_stack[0], q0, (size_t)hd * sizeof(float));
                memcpy(q_stack[1], q1, (size_t)hd * sizeof(float));
                memcpy(q_stack[2], q2, (size_t)hd * sizeof(float));
                memcpy(q_stack[3], q3, (size_t)hd * sizeof(float));
                q0 = q_stack[0]; q1 = q_stack[1]; q2 = q_stack[2]; q3 = q_stack[3];
            }

            float max0 = -1e30f, max1 = -1e30f, max2 = -1e30f, max3 = -1e30f;

            /* 4-query x 2-key QK^T: loads each K once for 4 queries, each Q once for 2 keys */
            uint32_t kj = 0u;
            for (; kj + 1u < n_patches; kj += 2u) {
                const float *k0 = k_pack + (size_t)(kj + 0u) * hd;
                const float *k1 = k_pack + (size_t)(kj + 1u) * hd;
                if (kj + 5u < n_patches) {
                    _mm_prefetch((const char *)(k_pack + (size_t)(kj + 4u) * hd), _MM_HINT_T0);
                    _mm_prefetch((const char *)(k_pack + (size_t)(kj + 5u) * hd), _MM_HINT_T0);
                }
                __m256 a00 = _mm256_setzero_ps(), a01 = _mm256_setzero_ps();
                __m256 a10 = _mm256_setzero_ps(), a11 = _mm256_setzero_ps();
                __m256 a20 = _mm256_setzero_ps(), a21 = _mm256_setzero_ps();
                __m256 a30 = _mm256_setzero_ps(), a31 = _mm256_setzero_ps();
                for (uint32_t d = 0u; d < hd8; d += 8u) {
                    __m256 vk0 = _mm256_loadu_ps(k0 + d);
                    __m256 vk1 = _mm256_loadu_ps(k1 + d);
                    __m256 vq;
                    vq = _mm256_loadu_ps(q0 + d); a00 = _mm256_fmadd_ps(vq, vk0, a00); a01 = _mm256_fmadd_ps(vq, vk1, a01);
                    vq = _mm256_loadu_ps(q1 + d); a10 = _mm256_fmadd_ps(vq, vk0, a10); a11 = _mm256_fmadd_ps(vq, vk1, a11);
                    vq = _mm256_loadu_ps(q2 + d); a20 = _mm256_fmadd_ps(vq, vk0, a20); a21 = _mm256_fmadd_ps(vq, vk1, a21);
                    vq = _mm256_loadu_ps(q3 + d); a30 = _mm256_fmadd_ps(vq, vk0, a30); a31 = _mm256_fmadd_ps(vq, vk1, a31);
                }
                float d00 = hsum_m256(a00), d01 = hsum_m256(a01);
                float d10 = hsum_m256(a10), d11 = hsum_m256(a11);
                float d20 = hsum_m256(a20), d21 = hsum_m256(a21);
                float d30 = hsum_m256(a30), d31 = hsum_m256(a31);
                for (uint32_t d = hd8; d < hd; d++) {
                    float fk0 = k0[d], fk1 = k1[d];
                    d00 += q0[d] * fk0; d01 += q0[d] * fk1;
                    d10 += q1[d] * fk0; d11 += q1[d] * fk1;
                    d20 += q2[d] * fk0; d21 += q2[d] * fk1;
                    d30 += q3[d] * fk0; d31 += q3[d] * fk1;
                }
                d00 *= c->scale; d01 *= c->scale;
                d10 *= c->scale; d11 *= c->scale;
                d20 *= c->scale; d21 *= c->scale;
                d30 *= c->scale; d31 *= c->scale;
                s0[kj] = d00; s0[kj + 1u] = d01;
                s1[kj] = d10; s1[kj + 1u] = d11;
                s2[kj] = d20; s2[kj + 1u] = d21;
                s3[kj] = d30; s3[kj + 1u] = d31;
                if (d00 > max0) max0 = d00;
                if (d01 > max0) max0 = d01;
                if (d10 > max1) max1 = d10;
                if (d11 > max1) max1 = d11;
                if (d20 > max2) max2 = d20;
                if (d21 > max2) max2 = d21;
                if (d30 > max3) max3 = d30;
                if (d31 > max3) max3 = d31;
            }
            for (; kj < n_patches; kj++) {
                const float *k0 = k_pack + (size_t)kj * hd;
                __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
                __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
                for (uint32_t d = 0u; d < hd8; d += 8u) {
                    __m256 vk0 = _mm256_loadu_ps(k0 + d);
                    a0 = _mm256_fmadd_ps(_mm256_loadu_ps(q0 + d), vk0, a0);
                    a1 = _mm256_fmadd_ps(_mm256_loadu_ps(q1 + d), vk0, a1);
                    a2 = _mm256_fmadd_ps(_mm256_loadu_ps(q2 + d), vk0, a2);
                    a3 = _mm256_fmadd_ps(_mm256_loadu_ps(q3 + d), vk0, a3);
                }
                float d0 = hsum_m256(a0), d1 = hsum_m256(a1), d2 = hsum_m256(a2), d3 = hsum_m256(a3);
                for (uint32_t d = hd8; d < hd; d++) {
                    float fk0 = k0[d];
                    d0 += q0[d] * fk0; d1 += q1[d] * fk0; d2 += q2[d] * fk0; d3 += q3[d] * fk0;
                }
                d0 *= c->scale; d1 *= c->scale; d2 *= c->scale; d3 *= c->scale;
                s0[kj] = d0; s1[kj] = d1; s2[kj] = d2; s3[kj] = d3;
                if (d0 > max0) max0 = d0;
                if (d1 > max1) max1 = d1;
                if (d2 > max2) max2 = d2;
                if (d3 > max3) max3 = d3;
            }

            /* Vectorized exact-polynomial AVX2 softmax for all 4 queries */
            float maxv[4] = {max0, max1, max2, max3};
            float *s_ptrs[4] = {s0, s1, s2, s3};
            for (int tq = 0; tq < 4; tq++) {
                float *sp = s_ptrs[tq];
                __m256 vm = _mm256_set1_ps(maxv[tq]);
                __m256 vsum = _mm256_setzero_ps();
                uint32_t j = 0u;
                for (; j + 7u < n_patches; j += 8u) {
                    __m256 e = avx2_exp_ps(_mm256_sub_ps(_mm256_loadu_ps(sp + j), vm));
                    _mm256_storeu_ps(sp + j, e);
                    vsum = _mm256_add_ps(vsum, e);
                }
                double sum_exp = (double)hsum_m256(vsum);
                for (; j < n_patches; j++) {
                    float e = expf(sp[j] - maxv[tq]);
                    sp[j] = e;
                    sum_exp += (double)e;
                }
                float inv_sum = (sum_exp > 0.0) ? (float)(1.0 / sum_exp) : 0.0f;
                __m256 vinv = _mm256_set1_ps(inv_sum);
                j = 0u;
                for (; j + 7u < n_patches; j += 8u) {
                    _mm256_storeu_ps(sp + j, _mm256_mul_ps(_mm256_loadu_ps(sp + j), vinv));
                }
                for (; j < n_patches; j++) sp[j] *= inv_sum;
            }

            /* 4-query x 16-dim register-blocked P*V: loads each V[kj] ONCE for 4 queries */
            float *o0 = c->out + (size_t)(qi + 0u) * dim + (size_t)hi * hd;
            float *o1 = c->out + (size_t)(qi + 1u) * dim + (size_t)hi * hd;
            float *o2 = c->out + (size_t)(qi + 2u) * dim + (size_t)hi * hd;
            float *o3 = c->out + (size_t)(qi + 3u) * dim + (size_t)hi * hd;

            uint32_t d = 0u;
            for (; d < hd16; d += 16u) {
                __m256 r0a = _mm256_setzero_ps(), r0b = _mm256_setzero_ps();
                __m256 r1a = _mm256_setzero_ps(), r1b = _mm256_setzero_ps();
                __m256 r2a = _mm256_setzero_ps(), r2b = _mm256_setzero_ps();
                __m256 r3a = _mm256_setzero_ps(), r3b = _mm256_setzero_ps();
                for (uint32_t j = 0u; j < n_patches; j++) {
                    const float *vk = v_pack + (size_t)j * hd + d;
                    __m256 va = _mm256_loadu_ps(vk);
                    __m256 vb = _mm256_loadu_ps(vk + 8u);
                    __m256 w;
                    w = _mm256_set1_ps(s0[j]); r0a = _mm256_fmadd_ps(w, va, r0a); r0b = _mm256_fmadd_ps(w, vb, r0b);
                    w = _mm256_set1_ps(s1[j]); r1a = _mm256_fmadd_ps(w, va, r1a); r1b = _mm256_fmadd_ps(w, vb, r1b);
                    w = _mm256_set1_ps(s2[j]); r2a = _mm256_fmadd_ps(w, va, r2a); r2b = _mm256_fmadd_ps(w, vb, r2b);
                    w = _mm256_set1_ps(s3[j]); r3a = _mm256_fmadd_ps(w, va, r3a); r3b = _mm256_fmadd_ps(w, vb, r3b);
                }
                _mm256_storeu_ps(o0 + d, r0a); _mm256_storeu_ps(o0 + d + 8u, r0b);
                _mm256_storeu_ps(o1 + d, r1a); _mm256_storeu_ps(o1 + d + 8u, r1b);
                _mm256_storeu_ps(o2 + d, r2a); _mm256_storeu_ps(o2 + d + 8u, r2b);
                _mm256_storeu_ps(o3 + d, r3a); _mm256_storeu_ps(o3 + d + 8u, r3b);
            }
            for (; d < hd8; d += 8u) {
                __m256 r0 = _mm256_setzero_ps(), r1 = _mm256_setzero_ps();
                __m256 r2 = _mm256_setzero_ps(), r3 = _mm256_setzero_ps();
                for (uint32_t j = 0u; j < n_patches; j++) {
                    __m256 va = _mm256_loadu_ps(v_pack + (size_t)j * hd + d);
                    r0 = _mm256_fmadd_ps(_mm256_set1_ps(s0[j]), va, r0);
                    r1 = _mm256_fmadd_ps(_mm256_set1_ps(s1[j]), va, r1);
                    r2 = _mm256_fmadd_ps(_mm256_set1_ps(s2[j]), va, r2);
                    r3 = _mm256_fmadd_ps(_mm256_set1_ps(s3[j]), va, r3);
                }
                _mm256_storeu_ps(o0 + d, r0);
                _mm256_storeu_ps(o1 + d, r1);
                _mm256_storeu_ps(o2 + d, r2);
                _mm256_storeu_ps(o3 + d, r3);
            }
            for (; d < hd; d++) {
                float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f, r3 = 0.0f;
                for (uint32_t j = 0u; j < n_patches; j++) {
                    float fv = v_pack[(size_t)j * hd + d];
                    r0 += s0[j] * fv; r1 += s1[j] * fv; r2 += s2[j] * fv; r3 += s3[j] * fv;
                }
                o0[d] = r0; o1[d] = r1; o2[d] = r2; o3[d] = r3;
            }
        }
#endif

        /* Remaining queries (n_patches % 4 or non-AVX2 builds) */
        for (; qi < n_patches; qi++) {
            const float *q_vec = c->qkv + (size_t)qi * qkv_stride + (size_t)hi * hd;
            float *out_vec = c->out + (size_t)qi * dim + (size_t)hi * hd;
            float max_s = -1e30f;
            for (uint32_t kj = 0u; kj < n_patches; kj++) {
                float dot = lila_fp32_dot_slice(q_vec, k_pack + (size_t)kj * hd, (int)hd) * c->scale;
                s0[kj] = dot;
                if (dot > max_s) max_s = dot;
            }
            double sum_exp = 0.0;
            for (uint32_t kj = 0u; kj < n_patches; kj++) {
                float e = expf(s0[kj] - max_s);
                s0[kj] = e;
                sum_exp += (double)e;
            }
            float inv_sum = (sum_exp > 0.0) ? (float)(1.0 / sum_exp) : 0.0f;
            memset(out_vec, 0, (size_t)hd * sizeof(float));
            for (uint32_t kj = 0u; kj < n_patches; kj++) {
                float w = s0[kj] * inv_sum;
                const float *v_vec = v_pack + (size_t)kj * hd;
                uint32_t d = 0u;
#if defined(__AVX2__) && defined(__FMA__)
                __m256 vw = _mm256_set1_ps(w);
                for (; d + 7u < hd; d += 8u) {
                    __m256 ov = _mm256_loadu_ps(out_vec + d);
                    __m256 vv = _mm256_loadu_ps(v_vec + d);
                    _mm256_storeu_ps(out_vec + d, _mm256_fmadd_ps(vw, vv, ov));
                }
#endif
                for (; d < hd; d++) out_vec[d] += w * v_vec[d];
            }
        }
    }

    free(scores_all);
    free(v_pack);
    free(k_pack);
}

int asi_k_sdpa_bidirectional_qkv(AsiKernelCtx *ctx,
                                 const float *qkv,
                                 uint32_t n_patches,
                                 uint32_t n_heads,
                                 uint32_t head_dim,
                                 float scale,
                                 float *out)
{
    if (!qkv || !out || n_patches == 0u || n_heads == 0u || head_dim == 0u) return -1;
    SdpaBidirCtx sc = {qkv, n_patches, n_heads, head_dim, scale, out, ctx ? ctx->cancel_flag : NULL};
    asi_kernel_parallel_for(ctx, n_heads, 1u, sdpa_bidir_head_range, &sc);
    return (ctx && ctx->cancel_flag && *ctx->cancel_flag) ? -2 : 0;
}

typedef struct {
    const float *in_nchw;
    const float *w_oihw;
    const float *bias;
    uint32_t     C_in, H_in, W_in;
    uint32_t     C_out, kH, kW;
    uint32_t     stride_h, stride_w;
    uint32_t     pad_h, pad_w;
    uint32_t     H_out, W_out;
    uint32_t     groups;
    float       *out_nchw;
} Conv2dF32Ctx;

static void conv2d_f32_range(uint32_t co0, uint32_t co1, int tid, void *vctx)
{
    (void)tid;
    const Conv2dF32Ctx *c = (const Conv2dF32Ctx *)vctx;
    uint32_t cg_in = c->C_in / c->groups;
    uint32_t cg_out = c->C_out / c->groups;
    size_t in_plane = (size_t)c->H_in * c->W_in;
    size_t out_plane = (size_t)c->H_out * c->W_out;

    for (uint32_t co = co0; co < co1; co++) {
        uint32_t g = co / cg_out;
        uint32_t ci_start = g * cg_in;
        const float *w_co = c->w_oihw + (size_t)co * cg_in * c->kH * c->kW;
        float b_val = c->bias ? c->bias[co] : 0.0f;
        float *out_plane_ptr = c->out_nchw + (size_t)co * out_plane;

        for (uint32_t oy = 0u; oy < c->H_out; oy++) {
            for (uint32_t ox = 0u; ox < c->W_out; ox++) {
                float acc = b_val;
                for (uint32_t ci = 0u; ci < cg_in; ci++) {
                    const float *in_p = c->in_nchw + (size_t)(ci_start + ci) * in_plane;
                    const float *w_ci = w_co + (size_t)ci * c->kH * c->kW;
                    for (uint32_t ky = 0u; ky < c->kH; ky++) {
                        int iy = (int)(oy * c->stride_h + ky) - (int)c->pad_h;
                        if (iy < 0 || (uint32_t)iy >= c->H_in) continue;
                        for (uint32_t kx = 0u; kx < c->kW; kx++) {
                            int ix = (int)(ox * c->stride_w + kx) - (int)c->pad_w;
                            if (ix < 0 || (uint32_t)ix >= c->W_in) continue;
                            acc += in_p[(size_t)iy * c->W_in + (uint32_t)ix] * w_ci[ky * c->kW + kx];
                        }
                    }
                }
                out_plane_ptr[(size_t)oy * c->W_out + ox] = acc;
            }
        }
    }
}

int asi_k_conv2d_f32(AsiKernelCtx *ctx,
                     const float *in_nchw,
                     const float *w_oihw,
                     const float *bias,
                     uint32_t C_in,
                     uint32_t H_in,
                     uint32_t W_in,
                     uint32_t C_out,
                     uint32_t kH,
                     uint32_t kW,
                     uint32_t stride_h,
                     uint32_t stride_w,
                     uint32_t pad_h,
                     uint32_t pad_w,
                     uint32_t groups,
                     float *out_nchw)
{
    if (!in_nchw || !w_oihw || !out_nchw || C_in == 0u || C_out == 0u ||
        kH == 0u || kW == 0u || stride_h == 0u || stride_w == 0u || groups == 0u) {
        return -1;
    }
    if ((C_in % groups) != 0u || (C_out % groups) != 0u) return -1;
    if (H_in + 2u * pad_h < kH || W_in + 2u * pad_w < kW) return -1;
    uint32_t H_out = (H_in + 2u * pad_h - kH) / stride_h + 1u;
    uint32_t W_out = (W_in + 2u * pad_w - kW) / stride_w + 1u;

    Conv2dF32Ctx cc = {
        in_nchw, w_oihw, bias, C_in, H_in, W_in, C_out, kH, kW,
        stride_h, stride_w, pad_h, pad_w, H_out, W_out, groups, out_nchw
    };
    asi_kernel_parallel_for(ctx, C_out, 4u, conv2d_f32_range, &cc);
    return (ctx && ctx->cancel_flag && *ctx->cancel_flag) ? -2 : 0;
}
