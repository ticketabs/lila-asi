#include "asi_dense.h"
#include "asi_platform.h"
#include "asi_pool.h"
#include "lila_asi.h"
#include "q4k.h"

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ASI_DENSE_SESSION_MAGIC 0x44454E53u /* "DENS" */
#define ASI_DENSE_DEFAULT_BATCH_CHUNK 512u
#define ASI_DENSE_MATVEC_CHUNK 64

typedef enum {
    DENSE_ROPE_NORM = 0, /* Adjacent-pair (2*i, 2*i+1) — GGUF llama */
    DENSE_ROPE_NEOX = 1, /* Half-split (i, i+rope_dim/2) — GGUF qwen2/qwen3/gemma */
} DenseRopeMode;

typedef enum {
    DENSE_ACT_SWIGLU = 0, /* silu(gate) * up — llama/qwen2/qwen3 */
    DENSE_ACT_GEGLU  = 1, /* gelu_tanh(gate) * up — gemma */
} DenseFfnAct;

typedef struct {
    const AsiV2TensorEntry *attn_norm;
    const AsiV2TensorEntry *ffn_norm;
    const AsiV2TensorEntry *attn_post_norm; /* Optional Gemma2 post-attention sandwich norm */
    const AsiV2TensorEntry *ffn_post_norm;  /* Optional Gemma2 post-FFN sandwich norm */

    const float *attn_norm_f32;
    const float *ffn_norm_f32;
    const float *attn_post_norm_f32;
    const float *ffn_post_norm_f32;

    const AsiV2TensorEntry *attn_q;
    const AsiV2TensorEntry *attn_k;
    const AsiV2TensorEntry *attn_v;
    const AsiV2TensorEntry *attn_qkv;
    const AsiV2TensorEntry *attn_output;

    const float *attn_q_bias;
    const float *attn_k_bias;
    const float *attn_v_bias;
    const float *attn_qkv_bias;
    const float *attn_output_bias;

    const float *attn_q_norm;
    const float *attn_k_norm;

    const AsiV2TensorEntry *ffn_gate;
    const AsiV2TensorEntry *ffn_up;
    const AsiV2TensorEntry *ffn_down;

    const float *ffn_gate_bias;
    const float *ffn_up_bias;
    const float *ffn_down_bias;
} DenseLayerWeights;

typedef struct {
    uint16_t *k_cache; /* [max_ctx, kv_dim] FP16 */
    uint16_t *v_cache; /* [max_ctx, kv_dim] FP16 */
} DenseKVState;

typedef struct {
    float *x;
    float *x_norm;
    float *q;
    float *k;
    float *v;
    float *qkv;
    float *attn_out;
    float *proj_out;
    float *ffn_gate;
    float *ffn_up;

    uint32_t batch_cap;
    float *xb;
    float *xnb;
    float *projb;
    float *qb;
    float *kb;
    float *vb;
    float *qkvb;
    float *aob;
    float *fgb;
    float *fub;
    LilaQ8KBlock *qx_h;
} DenseScratch;

typedef struct {
    uint32_t magic;
    const AsiV2Cartridge *cart;
    uint32_t max_ctx;
    uint32_t n_threads;
    uint32_t context_len;

    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    uint32_t q_dim;
    uint32_t kv_dim;
    uint32_t rope_dim;
    float rope_theta;
    float rms_norm_eps;
    float embed_scale;
    DenseRopeMode rope_mode;
    DenseFfnAct ffn_act;
    int fp32_ref;
    /* E08-E: opt-in per-layer streaming prefetch + interior-page MADV_DONTNEED eviction */
    int stream_evict;
    /* E08-D: YaRN attention temperature scale (1.0f for standard/linear/ntk/llama3) */
    float rope_attn_mscale;

    const AsiV2TensorEntry *token_embd;
    const AsiV2TensorEntry *output_norm;
    const AsiV2TensorEntry *output_weight; /* NULL if tied to token_embd */
    const float *output_norm_f32;
    const float *output_bias_f32;

    DenseLayerWeights *layers;
    DenseKVState *kv_states;
    DenseScratch scratch;
    AsiThreadPool *pool;
    float *rope_inv_freq;

    uint32_t prefix_len;
    uint16_t *prefix_k;
    uint16_t *prefix_v;

    AsiDenseLmHeadHook lm_head_hook;
    void *lm_head_hook_ctx;

    void **owned_allocs;
    size_t n_owned_allocs;
    size_t cap_owned_allocs;
} DenseSession;

static void dense_set_err(char *buf, size_t cap, const char *msg)
{
    if (!buf || cap == 0) return;
    snprintf(buf, cap, "%s", msg ? msg : "unknown error");
}

static void *dense_aligned_alloc(size_t alignment, size_t size)
{
    if (alignment < sizeof(void *)) alignment = sizeof(void *);
    if (size == 0) size = alignment;
    size_t padded = (size + alignment - 1u) & ~(alignment - 1u);
#if defined(_WIN32)
    return _aligned_malloc(padded, alignment);
#else
    return aligned_alloc(alignment, padded);
#endif
}

static void dense_aligned_free(void *ptr)
{
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

static int dense_track_alloc(DenseSession *s, void *ptr)
{
    if (!ptr) return -1;
    if (s->n_owned_allocs == s->cap_owned_allocs)
    {
        size_t new_cap = s->cap_owned_allocs ? s->cap_owned_allocs * 2u : 16u;
        void **next = (void **)realloc(s->owned_allocs, new_cap * sizeof(void *));
        if (!next) return -1;
        s->owned_allocs = next;
        s->cap_owned_allocs = new_cap;
    }
    s->owned_allocs[s->n_owned_allocs++] = ptr;
    return 0;
}

static inline const uint8_t *dense_tensor_data(const AsiV2Cartridge *cart, const AsiV2TensorEntry *t)
{
    return cart->weights_data + t->data_offset;
}

static int dense_ascii_ieq(const char *a, const char *b)
{
    if (!a || !b) return 0;
    while (*a && *b)
    {
        if (tolower((unsigned char)*a) != tolower((unsigned char)*b)) return 0;
        ++a;
        ++b;
    }
    return *a == '\0' && *b == '\0';
}

/* E08-E: Prefetch (is_evict=0) or interior-page MADV_DONTNEED evict (is_evict=1) dense layer tensors. */
static void dense_stream_advise_layer(const AsiV2Cartridge *cart, const DenseLayerWeights *lw, int is_evict)
{
    if (!cart || !cart->owns_mmap || !lw) return;
    const AsiV2TensorEntry *ts[12] = {
        lw->attn_norm, lw->ffn_norm, lw->attn_post_norm, lw->ffn_post_norm,
        lw->attn_qkv, lw->attn_q, lw->attn_k, lw->attn_v, lw->attn_output,
        lw->ffn_gate, lw->ffn_up, lw->ffn_down
    };
    for (size_t i = 0; i < sizeof(ts) / sizeof(ts[0]); ++i)
    {
        const AsiV2TensorEntry *t = ts[i];
        if (!t || t->data_size == 0u) continue;
        const void *ptr = dense_tensor_data(cart, t);
        if (is_evict)
            asi_stream_evict_pages(ptr, (size_t)t->data_size);
        else
            asi_stream_prefetch_pages(ptr, (size_t)t->data_size);
    }
}

/* E08-D: Unified RoPE frequency scaling (Standard / Linear / NTK / YaRN / Llama-3.1). */
static void dense_init_rope_inv_freq(DenseSession *s)
{
    s->rope_attn_mscale = 1.0f;
    if (s->rope_dim < 2u || !s->rope_inv_freq) return;

    /* LILA_ASI_ROPE_SCALE_MODE: Select long-context RoPE frequency scaling in dense models ("standard" [default]|"linear"|"ntk"|"yarn"|"llama3"). */
    const char *mode_env = getenv("LILA_ASI_ROPE_SCALE_MODE");
    /* LILA_ASI_ROPE_SCALE_FACTOR: Context extension multiplier applied by LILA_ASI_ROPE_SCALE_MODE (default: 1.0). */
    const char *fac_env = getenv("LILA_ASI_ROPE_SCALE_FACTOR");
    /* LILA_ASI_ROPE_ORIG_CTX: Original training context length for YaRN / Llama-3.1 frequency bands (default: model max_seq_len or 4096). */
    const char *orig_env = getenv("LILA_ASI_ROPE_ORIG_CTX");

    int mode = 0; /* 0=standard, 1=linear, 2=ntk, 3=yarn, 4=llama31 */
    if (mode_env && mode_env[0] != '\0')
    {
        if (dense_ascii_ieq(mode_env, "linear"))
            mode = 1;
        else if (dense_ascii_ieq(mode_env, "ntk"))
            mode = 2;
        else if (dense_ascii_ieq(mode_env, "yarn"))
            mode = 3;
        else if (dense_ascii_ieq(mode_env, "llama3") ||
                 dense_ascii_ieq(mode_env, "llama31") ||
                 dense_ascii_ieq(mode_env, "llama3.1"))
            mode = 4;
    }

    double factor = 1.0;
    if (fac_env && fac_env[0] != '\0')
    {
        double parsed = strtod(fac_env, NULL);
        if (isfinite(parsed) && parsed > 0.0) factor = parsed;
    }

    uint32_t orig_ctx = (s->cart->config.max_seq_len > 0u) ? s->cart->config.max_seq_len : 4096u;
    if (orig_env && orig_env[0] != '\0')
    {
        long parsed_ctx = strtol(orig_env, NULL, 10);
        if (parsed_ctx >= 1 && parsed_ctx <= 10485760L) orig_ctx = (uint32_t)parsed_ctx;
    }

    uint32_t half = s->rope_dim / 2u;
    if (mode == 0)
    {
        for (uint32_t i = 0; i < half; ++i)
            s->rope_inv_freq[i] = 1.0f / powf(s->rope_theta, (float)(2u * i) / (float)s->rope_dim);
        return;
    }

    double d_rope = (double)s->rope_dim;
    double theta = (double)s->rope_theta;
    const double pi = 3.14159265358979323846;
    const double beta_fast = 32.0;
    const double beta_slow = 1.0;
    const double low_freq_factor = 1.0;
    const double high_freq_factor = 4.0;

    if (mode == 3 && factor > 1.0)
        s->rope_attn_mscale = 0.1f * logf((float)factor) + 1.0f;

    for (uint32_t i = 0; i < half; ++i)
    {
        double exp_arg = (2.0 * (double)i) / d_rope;
        double base_inv = 1.0 / pow(theta, exp_arg);
        double scaled_inv = base_inv;

        if (mode == 1)
        {
            scaled_inv = base_inv / factor;
        }
        else if (mode == 2)
        {
            if (d_rope > 2.0)
            {
                double adj_theta = theta * pow(factor, d_rope / (d_rope - 2.0));
                scaled_inv = 1.0 / pow(adj_theta, exp_arg);
            }
            else
            {
                scaled_inv = base_inv / factor;
            }
        }
        else if (mode == 3)
        {
            double wavelen = (2.0 * pi) / base_inv;
            double ratio = (double)orig_ctx / wavelen;
            double gamma = (ratio - beta_slow) / (beta_fast - beta_slow);
            if (gamma < 0.0) gamma = 0.0;
            if (gamma > 1.0) gamma = 1.0;
            scaled_inv = (1.0 - gamma) * (base_inv / factor) + gamma * base_inv;
        }
        else if (mode == 4)
        {
            double wavelen = (2.0 * pi) / base_inv;
            double lo_wl = (double)orig_ctx / low_freq_factor;
            double hi_wl = (double)orig_ctx / high_freq_factor;
            if (wavelen < hi_wl)
                scaled_inv = base_inv;
            else if (wavelen > lo_wl)
                scaled_inv = base_inv / factor;
            else
            {
                double smooth = ((double)orig_ctx / wavelen - low_freq_factor) /
                                (high_freq_factor - low_freq_factor);
                scaled_inv = (1.0 - smooth) * (base_inv / factor) + smooth * base_inv;
            }
        }
        s->rope_inv_freq[i] = (float)scaled_inv;
    }
}

static size_t dense_expected_bytes(uint32_t quant_type, uint32_t layout_flags, uint64_t cols, uint64_t rows)
{
    if (cols == 0 || rows == 0) return 0;
    return lila_quant_matrix_bytes(
        quant_type,
        cols,
        rows,
        (layout_flags & ASI_TENSOR_LAYOUT_AVX2_REPACK) ? 1 : 0,
        (layout_flags & ASI_TENSOR_LAYOUT_COL_MAJOR) ? 1 : 0);
}

static const AsiV2TensorEntry *dense_req_2d(const AsiV2Cartridge *cart,
                                            const char *name,
                                            uint64_t exp_cols,
                                            uint64_t exp_rows,
                                            char *err_buf,
                                            size_t err_cap)
{
    const AsiV2TensorEntry *t = asi_v2_find_tensor(cart, name);
    if (!t)
    {
        if (err_buf && err_cap > 0) snprintf(err_buf, err_cap, "missing required tensor '%s'", name);
        return NULL;
    }
    if (t->shape[0] != exp_cols || t->shape[1] != exp_rows)
    {
        if (err_buf && err_cap > 0)
        {
            snprintf(err_buf, err_cap, "tensor '%s' shape [%llu, %llu] != expected [%llu, %llu]",
                     name, (unsigned long long)t->shape[0], (unsigned long long)t->shape[1],
                     (unsigned long long)exp_cols, (unsigned long long)exp_rows);
        }
        return NULL;
    }
    size_t need = dense_expected_bytes(t->quant_type, t->layout_flags, exp_cols, exp_rows);
    if (need == 0 || t->data_size < need)
    {
        if (err_buf && err_cap > 0)
        {
            snprintf(err_buf, err_cap, "tensor '%s' unsupported quant=%u or truncated (%llu < %zu)",
                     name, t->quant_type, (unsigned long long)t->data_size, need);
        }
        return NULL;
    }
    return t;
}

static const float *dense_bind_1d_f32(DenseSession *s,
                                      const char *name,
                                      uint64_t exp_len,
                                      int required,
                                      char *err_buf,
                                      size_t err_cap)
{
    const AsiV2TensorEntry *t = asi_v2_find_tensor(s->cart, name);
    if (!t)
    {
        if (required && err_buf && err_cap > 0)
            snprintf(err_buf, err_cap, "missing required vector '%s'", name);
        return NULL;
    }
    if (t->shape[0] != exp_len || t->shape[1] != 1u)
    {
        if (err_buf && err_cap > 0)
        {
            snprintf(err_buf, err_cap, "vector '%s' shape [%llu, %llu] != expected [%llu, 1]",
                     name, (unsigned long long)t->shape[0], (unsigned long long)t->shape[1],
                     (unsigned long long)exp_len);
        }
        return NULL;
    }
    size_t need = dense_expected_bytes(t->quant_type, t->layout_flags, exp_len, 1u);
    if (need == 0 || t->data_size < need)
    {
        if (err_buf && err_cap > 0)
            snprintf(err_buf, err_cap, "vector '%s' unsupported quant=%u or truncated", name, t->quant_type);
        return NULL;
    }
    const uint8_t *raw = dense_tensor_data(s->cart, t);
    if (t->quant_type == 0u)
    {
        /* Copy if unaligned to guarantee aligned float access under UBSan. */
        if (((uintptr_t)raw & 3u) == 0u)
            return (const float *)raw;
        float *buf = (float *)malloc((size_t)exp_len * sizeof(float));
        if (!buf || dense_track_alloc(s, buf) != 0) { free(buf); return NULL; }
        memcpy(buf, raw, (size_t)exp_len * sizeof(float));
        return buf;
    }
    if (t->quant_type == 1u || t->quant_type == 30u)
    {
        float *buf = (float *)malloc((size_t)exp_len * sizeof(float));
        if (!buf || dense_track_alloc(s, buf) != 0) { free(buf); return NULL; }
        for (uint64_t i = 0; i < exp_len; ++i)
        {
            uint16_t bits;
            memcpy(&bits, raw + i * 2u, 2u);
            buf[i] = (t->quant_type == 1u) ? fp16_to_fp32(bits) : bf16_to_fp32(bits);
        }
        return buf;
    }
    if (err_buf && err_cap > 0)
        snprintf(err_buf, err_cap, "vector '%s' has non-float quant=%u", name, t->quant_type);
    return NULL;
}

static inline float dense_silu(float x)
{
    return x / (1.0f + expf(-x));
}

static inline float dense_gelu_tanh(float x)
{
    const float sqrt_2_over_pi = 0.79788456080286535588f;
    const float coef_a = 0.044715f;
    float inner = sqrt_2_over_pi * x * (1.0f + coef_a * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

#if defined(__AVX2__)
static inline __m256 dense_expf_avx2(__m256 x)
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

static void dense_ffn_act_mul(float *gate, const float *up, uint32_t n, DenseFfnAct act, int fp32_ref)
{
    if (act == DENSE_ACT_GEGLU)
    {
        for (uint32_t i = 0; i < n; ++i)
            gate[i] = dense_gelu_tanh(gate[i]) * up[i];
        return;
    }
    uint32_t i = 0;
#if defined(__AVX2__)
    if (!fp32_ref)
    {
        const __m256 one = _mm256_set1_ps(1.0f);
        const __m256 neg = _mm256_set1_ps(-0.0f);
        for (; i + 7u < n; i += 8u)
        {
            __m256 g = _mm256_loadu_ps(gate + i);
            __m256 u = _mm256_loadu_ps(up + i);
            __m256 e = dense_expf_avx2(_mm256_xor_ps(g, neg));
            __m256 sig = _mm256_div_ps(g, _mm256_add_ps(one, e));
            _mm256_storeu_ps(gate + i, _mm256_mul_ps(sig, u));
        }
    }
#else
    (void)fp32_ref;
#endif
    for (; i < n; ++i)
        gate[i] = dense_silu(gate[i]) * up[i];
}

static void dense_rmsnorm(float *out, const float *x, const float *w, uint32_t n, float eps)
{
    float ss = 0.0f;
#if defined(__AVX2__)
    if ((n & 7u) == 0u)
    {
        __m256 vss = _mm256_setzero_ps();
        for (uint32_t i = 0; i < n; i += 8u)
        {
            __m256 vx = _mm256_loadu_ps(x + i);
            vss = _mm256_fmadd_ps(vx, vx, vss);
        }
        ss = lila_hsum_f32_avx2(vss);
        float inv = 1.0f / sqrtf(ss / (float)n + eps);
        __m256 vinv = _mm256_set1_ps(inv);
        for (uint32_t i = 0; i < n; i += 8u)
        {
            __m256 vx = _mm256_loadu_ps(x + i);
            __m256 vw = _mm256_loadu_ps(w + i);
            _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_mul_ps(vx, vinv), vw));
        }
        return;
    }
#endif
    for (uint32_t i = 0; i < n; ++i) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / (float)n + eps);
    for (uint32_t i = 0; i < n; ++i) out[i] = (x[i] * inv) * w[i];
}

static inline void dense_add_vec(float *dst, const float *src, uint32_t n)
{
    if (!src) return;
    uint32_t i = 0;
#if defined(__AVX2__)
    for (; i + 7u < n; i += 8u)
    {
        _mm256_storeu_ps(dst + i, _mm256_add_ps(_mm256_loadu_ps(dst + i), _mm256_loadu_ps(src + i)));
    }
#endif
    for (; i < n; ++i) dst[i] += src[i];
}

static void dense_apply_rope(float *vec, uint32_t rope_dim, uint32_t pos,
                             const float *inv_freq, DenseRopeMode mode)
{
    uint32_t half = rope_dim / 2u;
    if (mode == DENSE_ROPE_NORM)
    {
        for (uint32_t i = 0; i < half; ++i)
        {
            float angle = (float)pos * inv_freq[i];
            float c = cosf(angle);
            float s = sinf(angle);
            float x0 = vec[2u * i];
            float x1 = vec[2u * i + 1u];
            vec[2u * i]      = x0 * c - x1 * s;
            vec[2u * i + 1u] = x0 * s + x1 * c;
        }
        return;
    }
    for (uint32_t i = 0; i < half; ++i)
    {
        float angle = (float)pos * inv_freq[i];
        float c = cosf(angle);
        float s = sinf(angle);
        float x0 = vec[i];
        float x1 = vec[i + half];
        vec[i]        = x0 * c - x1 * s;
        vec[i + half] = x0 * s + x1 * c;
    }
}

static int dense_lookup_embed(const DenseSession *s, int32_t token_id, float *out)
{
    const AsiV2TensorEntry *emb = s->token_embd;
    const uint8_t *base = dense_tensor_data(s->cart, emb);
    uint32_t H = s->cart->config.hidden_size;
    int is_repacked = (emb->layout_flags & ASI_TENSOR_LAYOUT_AVX2_REPACK) ? 1 : 0;
    size_t row_bytes = dense_expected_bytes(emb->quant_type, emb->layout_flags, H, 1u);
    if (row_bytes == 0 || !lila_quant_is_supported(emb->quant_type)) return -1;
    const uint8_t *src = base + (size_t)token_id * row_bytes;
    lila_dequant_row(out, src, (int)H, emb->quant_type, is_repacked);

    if (s->embed_scale != 1.0f)
    {
        float es = s->embed_scale;
        uint32_t i = 0;
#if defined(__AVX2__)
        __m256 ves = _mm256_set1_ps(es);
        for (; i + 7u < H; i += 8u)
            _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_loadu_ps(out + i), ves));
#endif
        for (; i < H; ++i) out[i] *= es;
    }
    return 0;
}

static float dense_dot_row_fp32(const uint8_t *w_base, int row, const float *x,
                                int cols, int rows, uint32_t quant_type,
                                int is_repacked, int is_col_major)
{
    return lila_quant_dot_fp32_at(w_base, row, x, cols, rows, quant_type, is_repacked, is_col_major);
}

static float dense_q6k_repacked_dot_q8k(const uint8_t *row_ptr, const LilaQ8KBlock *qx, int bpr)
{
#if defined(__AVX2__)
    const __m256i ones16 = _mm256_set1_epi16(1);
    __m256 sum_vec = _mm256_setzero_ps();
    float bias = 0.0f;
    for (int b = 0; b < bpr; ++b)
    {
        const LilaQ6KRepackedBlock *blk = (const LilaQ6KRepackedBlock *)(row_ptr + (size_t)b * sizeof(LilaQ6KRepackedBlock));
        const LilaQ8KBlock *xb = &qx[b];
        float d = blk->d;
        for (int sb = 0; sb < 8; ++sb)
        {
            __m256i uv = _mm256_loadu_si256((const __m256i *)(blk->u6 + sb * 32));
            __m256i xv = _mm256_loadu_si256((const __m256i *)(xb->qs + sb * 32));
            __m256i prod32 = _mm256_madd_epi16(_mm256_maddubs_epi16(uv, xv), ones16);
            float xd = xb->d;
            float sc0 = d * (float)blk->scales[sb * 2 + 0] * xd;
            float sc1 = d * (float)blk->scales[sb * 2 + 1] * xd;
            sum_vec = _mm256_fmadd_ps(_mm256_cvtepi32_ps(prod32),
                                      _mm256_set_ps(sc1, sc1, sc1, sc1, sc0, sc0, sc0, sc0),
                                      sum_vec);
            bias += 32.0f * (sc0 * (float)xb->bsums[sb * 2 + 0] + sc1 * (float)xb->bsums[sb * 2 + 1]);
        }
    }
    return lila_hsum_f32_avx2(sum_vec) - bias;
#else
    float tmp[QK_K];
    float sum = 0.0f;
    for (int b = 0; b < bpr; ++b)
    {
        const LilaQ6KRepackedBlock *blk = (const LilaQ6KRepackedBlock *)(row_ptr + (size_t)b * sizeof(LilaQ6KRepackedBlock));
        dequant_q6k_repacked_block(tmp, blk);
        const LilaQ8KBlock *xb = &qx[b];
        for (int j = 0; j < QK_K; ++j)
            sum += tmp[j] * (xb->d * (float)xb->qs[j]);
    }
    return sum;
#endif
}

static void dense_matvec_q8k_range(const uint8_t *w, const LilaQ8KBlock *qx, float *out,
                                   int start, int end, int bpr, uint32_t quant_type, int is_repacked)
{
    if (quant_type == 12u)
    {
        size_t rb = (size_t)bpr * Q4_K_BLOCK_SIZE;
        int i = start;
#if defined(__AVX2__)
        for (; i + 1 < end; i += 2)
        {
            const uint8_t *rp = w + (size_t)i * rb;
            if (i + 3 < end)
            {
                LILA_PREFETCH(rp + 2 * rb);
                LILA_PREFETCH(rp + 3 * rb);
            }
            q4k_dot2_q8k_avx2(rp, rp + rb, qx, bpr, &out[i], &out[i + 1]);
        }
#endif
        for (; i < end; ++i)
        {
            const uint8_t *rp = w + (size_t)i * rb;
            if (i + 2 < end) LILA_PREFETCH(rp + 2 * rb);
            out[i] = q4k_dot_q8k_simd(rp, qx, bpr);
        }
        return;
    }
    if (quant_type == 14u && is_repacked)
    {
        size_t rb = (size_t)bpr * sizeof(LilaQ6KRepackedBlock);
        for (int i = start; i < end; ++i)
        {
            const uint8_t *rp = w + (size_t)i * rb;
            if (i + 2 < end) LILA_PREFETCH(rp + 2 * rb);
            out[i] = dense_q6k_repacked_dot_q8k(rp, qx, bpr);
        }
        return;
    }
    if (quant_type == 14u)
    {
        size_t rb = (size_t)bpr * Q6_K_BLOCK_SIZE;
        int i = start;
#if defined(__AVX2__)
        for (; i + 1 < end; i += 2)
        {
            const uint8_t *rp = w + (size_t)i * rb;
            if (i + 3 < end)
            {
                LILA_PREFETCH(rp + 2 * rb);
                LILA_PREFETCH(rp + 3 * rb);
            }
            q6k_dot2_q8k_avx2(rp, rp + rb, qx, bpr, &out[i], &out[i + 1]);
        }
#endif
        for (; i < end; ++i)
        {
            const uint8_t *rp = w + (size_t)i * rb;
            if (i + 2 < end) LILA_PREFETCH(rp + 2 * rb);
            out[i] = q6k_dot_q8k_simd(rp, qx, bpr);
        }
    }
}

typedef struct {
    const uint8_t *w[3];
    float *out[3];
    int rows[3];
    uint32_t quant_type[3];
    int is_repacked[3];
    int is_col_major[3];
    size_t row_bytes[3];
    int n_mats;

    const float *in;
    const LilaQ8KBlock *qx;
    const LilaQ8_64Block *qx64;
    const LilaQ3LutBlock *q3_luts;
    int n_tokens;
    int cols;
    int bpr;
    int n_blk64;
    int n_blk128;
    int fp32_ref;
    int apply_ffn_act;
    DenseFfnAct ffn_act;
    const float *bias[3];
    _Alignas(64) atomic_int next_row[3];
} DenseMatmulCtx;

static void dense_eval_mat_range(const DenseMatmulCtx *ctx, int m, int start, int end)
{
    uint32_t qt = ctx->quant_type[m];
    int rep = ctx->is_repacked[m];
    int cm = ctx->is_col_major[m];
    int n_tok = ctx->n_tokens;
    int rows = ctx->rows[m];
    int cols = ctx->cols;
    size_t rb = ctx->row_bytes[m];

    if (!ctx->fp32_ref && qt == ASI_QUANT_Q4_REPACK && !cm)
    {
        if (n_tok <= 1)
            q4_repack_matvec_range(ctx->out[m], ctx->w[m], ctx->qx64, start, end, ctx->n_blk64);
        else
            q4_repack_matmul_batch_q8_range(ctx->out[m], ctx->w[m], ctx->qx64, n_tok, start, end, rows, ctx->n_blk64);
    }
    else if (!ctx->fp32_ref && qt == ASI_QUANT_Q3_SPLITLUT)
    {
        if (n_tok <= 1)
        {
            q3_splitlut_matvec_range(ctx->out[m], ctx->w[m], ctx->q3_luts, start, end, rows, ctx->n_blk128);
        }
        else
        {
            size_t tile_bytes = (size_t)ctx->n_blk128 * Q3_SPLITLUT_BLK_BYTES;
            int t0 = start / Q3_SPLITLUT_TILE_ROWS;
            int t1 = (end + Q3_SPLITLUT_TILE_ROWS - 1) / Q3_SPLITLUT_TILE_ROWS;
            for (int tile = t0; tile < t1; ++tile)
            {
                int r_base = tile * Q3_SPLITLUT_TILE_ROWS;
                int valid = rows - r_base;
                if (valid > Q3_SPLITLUT_TILE_ROWS) valid = Q3_SPLITLUT_TILE_ROWS;
                const uint8_t *tp = ctx->w[m] + (size_t)tile * tile_bytes;
                for (int tok = 0; tok < n_tok; ++tok)
                {
                    const LilaQ3LutBlock *lb = ctx->q3_luts + (size_t)tok * (size_t)ctx->n_blk128;
                    if (r_base >= start && r_base + valid <= end)
                    {
                        q3_splitlut_tile_simd(tp, ctx->n_blk128, lb, valid,
                                              ctx->out[m] + (size_t)tok * (size_t)rows + (size_t)r_base);
                    }
                    else
                    {
                        float tmp[Q3_SPLITLUT_TILE_ROWS];
                        q3_splitlut_tile_simd(tp, ctx->n_blk128, lb, valid, tmp);
                        int lo = (start > r_base) ? (start - r_base) : 0;
                        int hi = (end < r_base + valid) ? (end - r_base) : valid;
                        float *dst = ctx->out[m] + (size_t)tok * (size_t)rows;
                        for (int lr = lo; lr < hi; ++lr) dst[r_base + lr] = tmp[lr];
                    }
                }
            }
        }
    }
    else if (!ctx->fp32_ref && (qt == 12u || qt == 14u))
    {
        if (n_tok <= 1)
        {
            dense_matvec_q8k_range(ctx->w[m], ctx->qx, ctx->out[m], start, end, ctx->bpr, qt, rep);
        }
        else if (!rep)
        {
            if (qt == 12u)
                q4k_matmul_batch_q8k_range(ctx->out[m], ctx->w[m], ctx->qx, n_tok, start, end, rows, ctx->bpr);
            else
                q6k_matmul_batch_q8k_range(ctx->out[m], ctx->w[m], ctx->qx, n_tok, start, end, rows, ctx->bpr);
        }
        else
        {
            for (int t = 0; t < n_tok; ++t)
            {
                for (int i = start; i < end; ++i)
                {
                    /* qx is strided by n_tok in batch mode */
                    float sum = 0.0f;
                    const uint8_t *rp = ctx->w[m] + (size_t)i * rb;
                    for (int b = 0; b < ctx->bpr; ++b)
                        sum += dense_q6k_repacked_dot_q8k(rp + (size_t)b * sizeof(LilaQ6KRepackedBlock),
                                                          ctx->qx + (size_t)b * (size_t)n_tok + (size_t)t, 1);
                    ctx->out[m][(size_t)t * (size_t)rows + (size_t)i] = sum;
                }
            }
        }
    }
    else if (ctx->fp32_ref && (qt == 12u || qt == 14u || qt == 100u || qt == 101u) && n_tok > 1)
    {
        lila_quant_matmul_batch_fp32_range_ex(ctx->w[m], ctx->in, cols, ctx->out[m],
                                               n_tok, rows, start, end, cols, qt, rep, cm);
    }
    else
    {
        for (int t = 0; t < n_tok; ++t)
        {
            const float *xt = ctx->in + (size_t)t * (size_t)cols;
            float *ot = ctx->out[m] + (size_t)t * (size_t)rows;
            for (int i = start; i < end; ++i)
            {
                ot[i] = dense_dot_row_fp32(ctx->w[m], i, xt, cols, rows, qt, rep, cm);
            }
        }
    }

    if (ctx->bias[m])
    {
        for (int t = 0; t < n_tok; ++t)
        {
            float *ot = ctx->out[m] + (size_t)t * (size_t)rows;
            for (int i = start; i < end; ++i)
                ot[i] += ctx->bias[m][i];
        }
    }
}

static void dense_matmul_worker(int tid, int n_threads, void *vctx)
{
    (void)tid;
    DenseMatmulCtx *ctx = (DenseMatmulCtx *)vctx;
    if (n_threads <= 1)
    {
        for (int m = 0; m < ctx->n_mats; ++m)
            dense_eval_mat_range(ctx, m, 0, ctx->rows[m]);
        if (ctx->apply_ffn_act && ctx->n_mats == 2 && ctx->rows[0] == ctx->rows[1])
        {
            dense_ffn_act_mul(ctx->out[0], ctx->out[1],
                              (uint32_t)((size_t)ctx->n_tokens * (size_t)ctx->rows[0]),
                              ctx->ffn_act, ctx->fp32_ref);
        }
        return;
    }

    if (ctx->apply_ffn_act && ctx->n_mats == 2 && ctx->rows[0] == ctx->rows[1])
    {
        int total = ctx->rows[0];
        int chunk = (total >= 256) ? 32 : 16;
        for (;;)
        {
            int start = atomic_fetch_add_explicit(&ctx->next_row[0], chunk, memory_order_relaxed);
            if (start >= total) break;
            int end = start + chunk;
            if (end > total) end = total;
            dense_eval_mat_range(ctx, 0, start, end);
            dense_eval_mat_range(ctx, 1, start, end);
            for (int t = 0; t < ctx->n_tokens; ++t)
            {
                size_t off = (size_t)t * (size_t)total + (size_t)start;
                dense_ffn_act_mul(ctx->out[0] + off, ctx->out[1] + off,
                                  (uint32_t)(end - start), ctx->ffn_act, ctx->fp32_ref);
            }
        }
        return;
    }

    for (int m = 0; m < ctx->n_mats; ++m)
    {
        int total = ctx->rows[m];
        int chunk = (total >= 16384) ? 256 : ((total >= 256) ? ASI_DENSE_MATVEC_CHUNK : 16);
        for (;;)
        {
            int start = atomic_fetch_add_explicit(&ctx->next_row[m], chunk, memory_order_relaxed);
            if (start >= total) break;
            int end = start + chunk;
            if (end > total) end = total;
            dense_eval_mat_range(ctx, m, start, end);
        }
    }
}

typedef struct {
    LilaQ8KBlock *qx_all;
    const float *in;
    int n_tokens;
    int cols;
} DenseQuantBatchCtx;

static void dense_quantize_q8k_worker(int tid, int n_threads, void *vctx)
{
    const DenseQuantBatchCtx *ctx = (const DenseQuantBatchCtx *)vctx;
    int start = (ctx->n_tokens * tid) / n_threads;
    int end = (ctx->n_tokens * (tid + 1)) / n_threads;
    for (int tok = start; tok < end; ++tok)
    {
        quantize_vec_q8k_strided(ctx->qx_all + tok, ctx->n_tokens,
                                 ctx->in + (size_t)tok * (size_t)ctx->cols, ctx->cols);
    }
}

static void dense_quantize_batch_q8k(AsiThreadPool *pool,
                                     LilaQ8KBlock *qx_all,
                                     const float *in,
                                     int n_tokens,
                                     int cols)
{
    if (n_tokens <= 1)
    {
        quantize_vec_q8k(qx_all, in, cols);
        return;
    }
    DenseQuantBatchCtx ctx = {qx_all, in, n_tokens, cols};
    if (pool && n_tokens >= 8)
        asi_pool_run(pool, dense_quantize_q8k_worker, &ctx);
    else
        dense_quantize_q8k_worker(0, 1, &ctx);
}

static int dense_matmul_multi(DenseSession *s,
                              const AsiV2TensorEntry * const *ts,
                              const float * const *biases,
                              float * const *outs,
                              int n_mats,
                              const LilaQ8KBlock *qx,
                              const float *in,
                              int n_tokens,
                              int cols,
                              int apply_ffn_act)
{
    DenseMatmulCtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.n_mats = n_mats;
    ctx.in = in;
    ctx.qx = qx;
    ctx.n_tokens = n_tokens;
    ctx.cols = cols;
    ctx.bpr = (cols + QK_K - 1) / QK_K;
    ctx.n_blk64 = (cols + Q4_REPACK_GROUP_SIZE - 1) / Q4_REPACK_GROUP_SIZE;
    ctx.n_blk128 = (cols + Q3_SPLITLUT_BLK_COLS - 1) / Q3_SPLITLUT_BLK_COLS;
    ctx.fp32_ref = s->fp32_ref;
    ctx.apply_ffn_act = apply_ffn_act;
    ctx.ffn_act = s->ffn_act;

    int needs_q8k = 0;
    int needs_q8_64 = 0;
    int needs_q3 = 0;
    int max_rows = 0;
    for (int m = 0; m < n_mats; ++m)
    {
        ctx.w[m] = dense_tensor_data(s->cart, ts[m]);
        ctx.out[m] = outs[m];
        ctx.rows[m] = (int)ts[m]->shape[1];
        ctx.quant_type[m] = ts[m]->quant_type;
        ctx.is_repacked[m] = (ts[m]->layout_flags & ASI_TENSOR_LAYOUT_AVX2_REPACK) ? 1 : 0;
        ctx.is_col_major[m] = (ts[m]->layout_flags & ASI_TENSOR_LAYOUT_COL_MAJOR) ? 1 : 0;
        ctx.row_bytes[m] = lila_quant_row_bytes(ts[m]->quant_type, cols, ctx.is_repacked[m]);
        ctx.bias[m] = biases ? biases[m] : NULL;
        atomic_init(&ctx.next_row[m], 0);
        if (ctx.rows[m] > max_rows) max_rows = ctx.rows[m];
        if (!s->fp32_ref && (ts[m]->quant_type == 12u || ts[m]->quant_type == 14u))
            needs_q8k = 1;
        if (!s->fp32_ref && ts[m]->quant_type == ASI_QUANT_Q4_REPACK && !ctx.is_col_major[m])
            needs_q8_64 = 1;
        if (!s->fp32_ref && ts[m]->quant_type == ASI_QUANT_Q3_SPLITLUT)
            needs_q3 = 1;
    }

    LilaQ8KBlock stack_qx[64];
    LilaQ8KBlock *tmp_qx = NULL;
    if (needs_q8k && !qx)
    {
        size_t need = (size_t)n_tokens * (size_t)ctx.bpr;
        tmp_qx = (need <= 64u) ? stack_qx : (LilaQ8KBlock *)dense_aligned_alloc(32u, need * sizeof(LilaQ8KBlock));
        if (!tmp_qx) return -1;
        dense_quantize_batch_q8k(s->pool, tmp_qx, in, n_tokens, cols);
        ctx.qx = tmp_qx;
    }
    LilaQ8_64Block stack_qx64[128];
    LilaQ8_64Block *tmp_qx64 = NULL;
    if (needs_q8_64)
    {
        size_t need = (size_t)n_tokens * (size_t)ctx.n_blk64;
        tmp_qx64 = (need <= 128u) ? stack_qx64 : (LilaQ8_64Block *)dense_aligned_alloc(32u, need * sizeof(LilaQ8_64Block));
        if (!tmp_qx64) { if (tmp_qx && tmp_qx != stack_qx) dense_aligned_free(tmp_qx); return -1; }
        for (int tok = 0; tok < n_tokens; ++tok)
            quantize_vec_q8_64_strided(tmp_qx64 + tok, n_tokens, in + (size_t)tok * (size_t)cols, cols);
        ctx.qx64 = tmp_qx64;
    }
    LilaQ3LutBlock stack_q3[32];
    LilaQ3LutBlock *tmp_q3 = NULL;
    if (needs_q3)
    {
        size_t need = (size_t)n_tokens * (size_t)ctx.n_blk128;
        tmp_q3 = (need <= 32u) ? stack_q3 : (LilaQ3LutBlock *)dense_aligned_alloc(32u, need * sizeof(LilaQ3LutBlock));
        if (!tmp_q3)
        {
            if (tmp_qx && tmp_qx != stack_qx) dense_aligned_free(tmp_qx);
            if (tmp_qx64 && tmp_qx64 != stack_qx64) dense_aligned_free(tmp_qx64);
            return -1;
        }
        for (int tok = 0; tok < n_tokens; ++tok)
            q3_splitlut_build_lut(tmp_q3 + (size_t)tok * (size_t)ctx.n_blk128, in + (size_t)tok * (size_t)cols, cols);
        ctx.q3_luts = tmp_q3;
    }

    if (s->pool && (max_rows >= 32 || (max_rows >= 16 && n_tokens > 1)))
        asi_pool_run(s->pool, dense_matmul_worker, &ctx);
    else
        dense_matmul_worker(0, 1, &ctx);

    if (tmp_qx && tmp_qx != stack_qx) dense_aligned_free(tmp_qx);
    if (tmp_qx64 && tmp_qx64 != stack_qx64) dense_aligned_free(tmp_qx64);
    if (tmp_q3 && tmp_q3 != stack_q3) dense_aligned_free(tmp_q3);
    return 0;
}

static int dense_eval_lm_head(DenseSession *s, const float *x_norm, float *out_logits)
{
    const AsiV2TensorEntry *lm_head = s->output_weight ? s->output_weight : s->token_embd;
    int rc;
    if (s->lm_head_hook)
    {
        rc = s->lm_head_hook(s->lm_head_hook_ctx, lm_head, x_norm, out_logits);
    }
    else
    {
        const float *bias = s->output_bias_f32;
        rc = dense_matmul_multi(s, &lm_head, &bias, &out_logits, 1, NULL,
                                x_norm, 1, (int)s->cart->config.hidden_size, 0);
    }
    return rc;
}

typedef struct {
    float *qb;
    float *aob;
    const uint16_t *k_cache;
    const uint16_t *v_cache;
    const float *qn_w;
    const float *rope_inv_freq;
    uint32_t B;
    uint32_t pos_start;
    uint32_t n_heads;
    uint32_t n_rep;
    uint32_t hd;
    uint32_t q_dim;
    uint32_t kv_dim;
    uint32_t rope_dim;
    DenseRopeMode rope_mode;
    float rms_eps;
    float scale;
    _Alignas(64) atomic_int next_task;
} DenseAttnCtx;

static inline void dense_cvt_f16_to_f32(float *dst, const uint16_t *src, uint32_t n)
{
    uint32_t d = 0;
#if defined(__AVX2__) && defined(__F16C__)
    for (; d + 7u < n; d += 8u)
    {
        _mm256_storeu_ps(dst + d, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(src + d))));
    }
#endif
    for (; d < n; ++d) dst[d] = fp16_to_fp32(src[d]);
}

static inline void dense_online_softmax_step(float *oh, float *m, float *l,
                                             const float *qh, const float *k_f32,
                                             const float *v_f32, uint32_t p,
                                             uint32_t hd, float scale)
{
    float s = lila_fp32_dot_slice(qh, k_f32, (int)hd) * scale;
    if (p == 0u)
    {
        *m = s;
        *l = 1.0f;
        memcpy(oh, v_f32, (size_t)hd * sizeof(float));
        return;
    }
    if (s > *m)
    {
        float alpha = expf(*m - s);
        *l = (*l) * alpha + 1.0f;
        *m = s;
        uint32_t d = 0;
#if defined(__AVX2__)
        __m256 va = _mm256_set1_ps(alpha);
        for (; d + 7u < hd; d += 8u)
        {
            __m256 ov = _mm256_loadu_ps(oh + d);
            __m256 vv = _mm256_loadu_ps(v_f32 + d);
            _mm256_storeu_ps(oh + d, _mm256_fmadd_ps(ov, va, vv));
        }
#endif
        for (; d < hd; ++d) oh[d] = oh[d] * alpha + v_f32[d];
    }
    else
    {
        float w = expf(s - *m);
        *l += w;
        uint32_t d = 0;
#if defined(__AVX2__)
        __m256 vw = _mm256_set1_ps(w);
        for (; d + 7u < hd; d += 8u)
        {
            __m256 ov = _mm256_loadu_ps(oh + d);
            __m256 vv = _mm256_loadu_ps(v_f32 + d);
            _mm256_storeu_ps(oh + d, _mm256_fmadd_ps(vw, vv, ov));
        }
#endif
        for (; d < hd; ++d) oh[d] += w * v_f32[d];
    }
}

static inline void dense_scale_vec(float *v, float s, uint32_t n)
{
    uint32_t d = 0;
#if defined(__AVX2__)
    __m256 vs = _mm256_set1_ps(s);
    for (; d + 7u < n; d += 8u)
        _mm256_storeu_ps(v + d, _mm256_mul_ps(_mm256_loadu_ps(v + d), vs));
#endif
    for (; d < n; ++d) v[d] *= s;
}

static void dense_gqa_attn_worker(int tid, int n_threads, void *vctx)
{
    (void)tid;
    DenseAttnCtx *ctx = (DenseAttnCtx *)vctx;
    uint32_t hd = ctx->hd;
    uint32_t kv_dim = ctx->kv_dim;
    uint32_t n_heads = ctx->n_heads;
    uint32_t n_rep = ctx->n_rep;
    uint32_t n_kv = n_heads / n_rep;

    /* Fast tiled GQA path when L1 buffers fit cleanly on stack; per-head fallback otherwise. */
    if (hd <= 256u && n_rep <= 8u && !(ctx->B == 1u && n_kv < (uint32_t)n_threads))
    {
        uint32_t t_chunks = (ctx->B + 3u) / 4u;
        uint32_t n_tasks = n_kv * t_chunks;
        _Alignas(32) float q_blk[4 * 8 * 256];
        _Alignas(32) float o_blk[4 * 8 * 256];
        _Alignas(32) float k_f32[256];
        _Alignas(32) float v_f32[256];

        for (uint32_t seq_task = 0;;)
        {
            uint32_t task;
            if (n_threads <= 1)
            {
                if (seq_task >= n_tasks) break;
                task = seq_task++;
            }
            else
            {
                task = (uint32_t)atomic_fetch_add_explicit(&ctx->next_task, 1, memory_order_relaxed);
                if (task >= n_tasks) break;
            }

            uint32_t kvh = task / t_chunks;
            uint32_t tc_idx = task - kvh * t_chunks;
            uint32_t t0 = tc_idx * 4u;
            uint32_t nt = ctx->B - t0;
            if (nt > 4u) nt = 4u;
            uint32_t h_base = kvh * n_rep;
            uint32_t max_t_pos = ctx->pos_start + t0 + nt - 1u;

            float m_arr[4 * 8];
            float l_arr[4 * 8];
            for (uint32_t tc = 0; tc < nt; ++tc)
            {
                uint32_t pos = ctx->pos_start + t0 + tc;
                const float *qt = ctx->qb + (size_t)(t0 + tc) * ctx->q_dim;
                for (uint32_t r = 0; r < n_rep; ++r)
                {
                    uint32_t idx = tc * n_rep + r;
                    float *q_dst = q_blk + (size_t)idx * hd;
                    const float *qh = qt + (size_t)(h_base + r) * hd;
                    if (ctx->qn_w)
                        dense_rmsnorm(q_dst, qh, ctx->qn_w, hd, ctx->rms_eps);
                    else
                        memcpy(q_dst, qh, (size_t)hd * sizeof(float));
                    if (ctx->rope_dim >= 2u)
                        dense_apply_rope(q_dst, ctx->rope_dim, pos, ctx->rope_inv_freq, ctx->rope_mode);
                    m_arr[idx] = 0.0f;
                    l_arr[idx] = 0.0f;
                }
            }

            for (uint32_t p = 0; p <= max_t_pos; ++p)
            {
                const uint16_t *kt = ctx->k_cache + (size_t)p * kv_dim + (size_t)kvh * hd;
                const uint16_t *vt = ctx->v_cache + (size_t)p * kv_dim + (size_t)kvh * hd;
                if (p + 2u <= max_t_pos)
                {
                    LILA_PREFETCH(ctx->k_cache + (size_t)(p + 2u) * kv_dim + (size_t)kvh * hd);
                    LILA_PREFETCH(ctx->v_cache + (size_t)(p + 2u) * kv_dim + (size_t)kvh * hd);
                }
                dense_cvt_f16_to_f32(k_f32, kt, hd);
                dense_cvt_f16_to_f32(v_f32, vt, hd);

                uint32_t tc_start = (p > ctx->pos_start + t0) ? (p - (ctx->pos_start + t0)) : 0u;
                for (uint32_t tc = tc_start; tc < nt; ++tc)
                {
                    for (uint32_t r = 0; r < n_rep; ++r)
                    {
                        uint32_t idx = tc * n_rep + r;
                        dense_online_softmax_step(o_blk + (size_t)idx * hd,
                                                  &m_arr[idx], &l_arr[idx],
                                                  q_blk + (size_t)idx * hd,
                                                  k_f32, v_f32, p, hd, ctx->scale);
                    }
                }
            }

            for (uint32_t tc = 0; tc < nt; ++tc)
            {
                float *ao = ctx->aob + (size_t)(t0 + tc) * ctx->q_dim;
                for (uint32_t r = 0; r < n_rep; ++r)
                {
                    uint32_t idx = tc * n_rep + r;
                    float inv_l = (l_arr[idx] > 0.0f) ? (1.0f / l_arr[idx]) : 0.0f;
                    float *oh = o_blk + (size_t)idx * hd;
                    dense_scale_vec(oh, inv_l, hd);
                    memcpy(ao + (size_t)(h_base + r) * hd, oh, (size_t)hd * sizeof(float));
                }
            }
        }
        return;
    }

    /* General fallback for large head_dim / high n_rep / MQA B=1 parallelism across heads. */
    float stack_k[512], stack_v[512];
    float *k_f32 = (hd <= 512u) ? stack_k : (float *)malloc((size_t)hd * sizeof(float));
    float *v_f32 = (hd <= 512u) ? stack_v : (float *)malloc((size_t)hd * sizeof(float));
    if (!k_f32 || !v_f32)
    {
        if (k_f32 != stack_k) free(k_f32);
        if (v_f32 != stack_v) free(v_f32);
        return;
    }

    for (uint32_t h_idx = 0;;)
    {
        uint32_t h;
        if (n_threads <= 1)
        {
            if (h_idx >= n_heads) break;
            h = h_idx++;
        }
        else
        {
            h = (uint32_t)atomic_fetch_add_explicit(&ctx->next_task, 1, memory_order_relaxed);
            if (h >= n_heads) break;
        }
        uint32_t kvh = h / n_rep;
        for (uint32_t t = 0; t < ctx->B; ++t)
        {
            uint32_t pos = ctx->pos_start + t;
            float *qh = ctx->qb + (size_t)t * ctx->q_dim + (size_t)h * hd;
            float *oh = ctx->aob + (size_t)t * ctx->q_dim + (size_t)h * hd;
            if (ctx->qn_w)
                dense_rmsnorm(qh, qh, ctx->qn_w, hd, ctx->rms_eps);
            if (ctx->rope_dim >= 2u)
                dense_apply_rope(qh, ctx->rope_dim, pos, ctx->rope_inv_freq, ctx->rope_mode);

            float m = 0.0f, l = 0.0f;
            for (uint32_t p = 0; p <= pos; ++p)
            {
                const uint16_t *kt = ctx->k_cache + (size_t)p * kv_dim + (size_t)kvh * hd;
                const uint16_t *vt = ctx->v_cache + (size_t)p * kv_dim + (size_t)kvh * hd;
                dense_cvt_f16_to_f32(k_f32, kt, hd);
                dense_cvt_f16_to_f32(v_f32, vt, hd);
                dense_online_softmax_step(oh, &m, &l, qh, k_f32, v_f32, p, hd, ctx->scale);
            }
            dense_scale_vec(oh, (l > 0.0f) ? (1.0f / l) : 0.0f, hd);
        }
    }

    if (k_f32 != stack_k) free(k_f32);
    if (v_f32 != stack_v) free(v_f32);
}

static inline void dense_store_kv_fp16(uint16_t *k_dst, uint16_t *v_dst,
                                       const float *k_src, const float *v_src, uint32_t kv_dim)
{
    uint32_t i = 0;
#if defined(__AVX2__) && defined(__F16C__)
    for (; i + 7u < kv_dim; i += 8u)
    {
        __m128i hk = _mm256_cvtps_ph(_mm256_loadu_ps(k_src + i), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m128i hv = _mm256_cvtps_ph(_mm256_loadu_ps(v_src + i), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm_storeu_si128((__m128i *)(k_dst + i), hk);
        _mm_storeu_si128((__m128i *)(v_dst + i), hv);
    }
#endif
    for (; i < kv_dim; ++i)
    {
        k_dst[i] = fp32_to_fp16(k_src[i]);
        v_dst[i] = fp32_to_fp16(v_src[i]);
    }
}

typedef struct {
    float *kb;
    const float *vb;
    uint16_t *k_cache;
    uint16_t *v_cache;
    const float *kn_w;
    const float *rope_inv_freq;
    DenseRopeMode rope_mode;
    uint32_t B;
    uint32_t pos_start;
    uint32_t n_kv;
    uint32_t hd;
    uint32_t kv_dim;
    uint32_t rope_dim;
    float rms_eps;
} DenseKvPrepCtx;

static void dense_kv_prep_worker(int tid, int n_threads, void *vctx)
{
    const DenseKvPrepCtx *ctx = (const DenseKvPrepCtx *)vctx;
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
            if (ctx->kn_w)
                dense_rmsnorm(kh, kh, ctx->kn_w, hd, ctx->rms_eps);
            if (ctx->rope_dim >= 2u)
                dense_apply_rope(kh, ctx->rope_dim, pos, ctx->rope_inv_freq, ctx->rope_mode);
        }
        dense_store_kv_fp16(ctx->k_cache + (size_t)pos * kv_dim,
                            ctx->v_cache + (size_t)pos * kv_dim,
                            k, v, kv_dim);
    }
}

static void dense_free_session(AsiArchSession *session)
{
    DenseSession *s = (DenseSession *)session;
    if (!s) return;
    asi_pool_destroy(s->pool);
    free(s->prefix_k);
    free(s->prefix_v);
    free(s->rope_inv_freq);
    if (s->kv_states && s->cart)
    {
        for (uint32_t i = 0; i < s->cart->config.n_layers; ++i)
        {
            free(s->kv_states[i].k_cache);
            free(s->kv_states[i].v_cache);
        }
    }
    free(s->kv_states);
    free(s->layers);
    for (size_t i = 0; i < s->n_owned_allocs; ++i)
        free(s->owned_allocs[i]);
    free(s->owned_allocs);

    free(s->scratch.x);
    free(s->scratch.x_norm);
    free(s->scratch.q);
    free(s->scratch.k);
    free(s->scratch.v);
    free(s->scratch.qkv);
    free(s->scratch.attn_out);
    free(s->scratch.proj_out);
    free(s->scratch.ffn_gate);
    free(s->scratch.ffn_up);

    free(s->scratch.xb);
    free(s->scratch.xnb);
    free(s->scratch.projb);
    free(s->scratch.qb);
    free(s->scratch.kb);
    free(s->scratch.vb);
    free(s->scratch.qkvb);
    free(s->scratch.aob);
    free(s->scratch.fgb);
    free(s->scratch.fub);
    dense_aligned_free(s->scratch.qx_h);
    free(s);
}

static void dense_reset_state(AsiArchSession *session)
{
    DenseSession *s = (DenseSession *)session;
    if (!s || !s->cart) return;
    size_t kv_elems = (size_t)s->max_ctx * s->kv_dim;
    for (uint32_t i = 0; i < s->cart->config.n_layers; ++i)
    {
        if (s->kv_states[i].k_cache)
            memset(s->kv_states[i].k_cache, 0, kv_elems * sizeof(uint16_t));
        if (s->kv_states[i].v_cache)
            memset(s->kv_states[i].v_cache, 0, kv_elems * sizeof(uint16_t));
    }
    s->context_len = 0;
}

static int dense_init_session(const AsiV2Cartridge *cart,
                              uint32_t max_ctx,
                              uint32_t n_threads,
                              AsiArchSession **out_session,
                              char *err_buf,
                              size_t err_cap)
{
    if (!cart || !out_session)
    {
        dense_set_err(err_buf, err_cap, "null cartridge or out_session");
        return -1;
    }
    *out_session = NULL;
    const AsiV2ModelConfig *c = &cart->config;
    if (c->arch_id != ASI_ARCH_DENSE_LLAMA &&
        c->arch_id != ASI_ARCH_DENSE_GEMMA &&
        c->arch_id != ASI_ARCH_DENSE_QWEN2 &&
        c->arch_id != ASI_ARCH_DENSE_QWEN3)
    {
        dense_set_err(err_buf, err_cap, "unsupported dense arch_id");
        return -1;
    }
    if (cart->header.arch_id != 0u && cart->header.arch_id != c->arch_id)
    {
        dense_set_err(err_buf, err_cap, "header.arch_id mismatches config.arch_id");
        return -1;
    }
    if (c->n_layers == 0 || c->hidden_size == 0 || c->intermediate_size == 0 ||
        c->n_heads == 0 || c->vocab_size == 0)
    {
        dense_set_err(err_buf, err_cap, "invalid dense model dimensions");
        return -1;
    }

    uint32_t n_heads = c->n_heads;
    uint32_t n_kv = (c->n_kv_heads > 0u) ? c->n_kv_heads : n_heads;
    if (n_kv == 0u || (n_heads % n_kv) != 0u)
    {
        dense_set_err(err_buf, err_cap, "n_heads must be divisible by n_kv_heads");
        return -1;
    }
    if (c->head_dim == 0u && (c->hidden_size % n_heads) != 0u)
    {
        dense_set_err(err_buf, err_cap, "hidden_size not divisible by n_heads when head_dim=0");
        return -1;
    }
    uint32_t hd = (c->head_dim > 0u) ? c->head_dim : (c->hidden_size / n_heads);
    if (hd == 0u)
    {
        dense_set_err(err_buf, err_cap, "invalid head_dim");
        return -1;
    }
    uint32_t rope_dim = (c->rope_dim > 0u) ? c->rope_dim : hd;
    if (rope_dim > hd || (rope_dim & 1u) != 0u)
    {
        dense_set_err(err_buf, err_cap, "invalid rope_dim");
        return -1;
    }

    DenseSession *s = (DenseSession *)calloc(1, sizeof(DenseSession));
    if (!s)
    {
        dense_set_err(err_buf, err_cap, "out of memory");
        return -1;
    }
    s->magic = ASI_DENSE_SESSION_MAGIC;
    s->cart = cart;
    s->max_ctx = (max_ctx > 0u) ? max_ctx : (c->max_seq_len > 0u ? c->max_seq_len : 2048u);
    if (n_threads == 0u)
    {
        int phys = asi_detect_physical_cores();
        n_threads = (phys > 0) ? (uint32_t)phys : 1u;
    }
    s->n_threads = n_threads;
    s->n_heads = n_heads;
    s->n_kv_heads = n_kv;
    s->head_dim = hd;
    s->q_dim = n_heads * hd;
    s->kv_dim = n_kv * hd;
    s->rope_dim = rope_dim;
    s->rope_theta = (c->rope_theta > 0.0f) ? c->rope_theta : 10000.0f;
    s->rms_norm_eps = (c->rms_norm_eps > 0.0f) ? c->rms_norm_eps : 1e-6f;
    if (c->embed_scale > 0.0f)
        s->embed_scale = c->embed_scale;
    else
        s->embed_scale = (c->arch_id == ASI_ARCH_DENSE_GEMMA) ? sqrtf((float)c->hidden_size) : 1.0f;

    s->rope_mode = (c->arch_id == ASI_ARCH_DENSE_LLAMA) ? DENSE_ROPE_NORM : DENSE_ROPE_NEOX;
    s->ffn_act   = (c->arch_id == ASI_ARCH_DENSE_GEMMA) ? DENSE_ACT_GEGLU : DENSE_ACT_SWIGLU;

    /* LILA_ASI_FP32_REF: Dequantize dense weights to FP32 before matvec for golden parity (1=on, 0=off [default]). */
    const char *fp32_env = getenv("LILA_ASI_FP32_REF");
    s->fp32_ref = (fp32_env && atoi(fp32_env) != 0) ? 1 : 0;
    /* LILA_ASI_STREAM_EVICT: E08-E per-layer MADV_WILLNEED prefetch + strict interior-page MADV_DONTNEED eviction for mmap weights (1=on, 0=off [default]). */
    const char *ev_env = getenv("LILA_ASI_STREAM_EVICT");
    s->stream_evict = (cart->owns_mmap && ev_env && atoi(ev_env) != 0) ? 1 : 0;
    s->rope_attn_mscale = 1.0f;

    s->pool = asi_pool_create((int)s->n_threads);
    if (!s->pool)
    {
        dense_set_err(err_buf, err_cap, "failed to create thread pool");
        dense_free_session((AsiArchSession *)s);
        return -1;
    }

    s->token_embd = dense_req_2d(cart, "token_embd.weight", c->hidden_size, c->vocab_size, err_buf, err_cap);
    s->output_norm = asi_v2_find_tensor(cart, "output_norm.weight");
    s->output_norm_f32 = dense_bind_1d_f32(s, "output_norm.weight", c->hidden_size, 1, err_buf, err_cap);
    if (!s->token_embd || !s->output_norm || !s->output_norm_f32)
    {
        dense_free_session((AsiArchSession *)s);
        return -1;
    }
    if (asi_v2_find_tensor(cart, "output.weight"))
    {
        s->output_weight = dense_req_2d(cart, "output.weight", c->hidden_size, c->vocab_size, err_buf, err_cap);
        if (!s->output_weight)
        {
            dense_free_session((AsiArchSession *)s);
            return -1;
        }
    }
    if (asi_v2_find_tensor(cart, "output.bias"))
    {
        s->output_bias_f32 = dense_bind_1d_f32(s, "output.bias", c->vocab_size, 1, err_buf, err_cap);
        if (!s->output_bias_f32)
        {
            dense_free_session((AsiArchSession *)s);
            return -1;
        }
    }

    s->layers = (DenseLayerWeights *)calloc(c->n_layers, sizeof(DenseLayerWeights));
    s->kv_states = (DenseKVState *)calloc(c->n_layers, sizeof(DenseKVState));
    if (!s->layers || !s->kv_states)
    {
        dense_set_err(err_buf, err_cap, "out of memory");
        dense_free_session((AsiArchSession *)s);
        return -1;
    }

    uint32_t H = c->hidden_size;
    uint32_t I = c->intermediate_size;
    uint32_t qkv_dim = s->q_dim + 2u * s->kv_dim;
    s->scratch.x        = (float *)calloc(H, sizeof(float));
    s->scratch.x_norm   = (float *)calloc(H, sizeof(float));
    s->scratch.q        = (float *)calloc(s->q_dim, sizeof(float));
    s->scratch.k        = (float *)calloc(s->kv_dim, sizeof(float));
    s->scratch.v        = (float *)calloc(s->kv_dim, sizeof(float));
    s->scratch.attn_out = (float *)calloc(s->q_dim, sizeof(float));
    s->scratch.proj_out = (float *)calloc(H, sizeof(float));
    s->scratch.ffn_gate = (float *)calloc(I, sizeof(float));
    s->scratch.ffn_up   = (float *)calloc(I, sizeof(float));
    if (!s->scratch.x || !s->scratch.x_norm || !s->scratch.q || !s->scratch.k ||
        !s->scratch.v || !s->scratch.attn_out || !s->scratch.proj_out ||
        !s->scratch.ffn_gate || !s->scratch.ffn_up)
    {
        dense_set_err(err_buf, err_cap, "out of memory allocating scratch");
        dense_free_session((AsiArchSession *)s);
        return -1;
    }

    if (s->rope_dim >= 2u)
    {
        uint32_t half = s->rope_dim / 2u;
        s->rope_inv_freq = (float *)malloc((size_t)half * sizeof(float));
        if (!s->rope_inv_freq)
        {
            dense_free_session((AsiArchSession *)s);
            return -1;
        }
        dense_init_rope_inv_freq(s);
    }

    int any_fused_qkv = 0;
    char name[64];
    for (uint32_t i = 0; i < c->n_layers; ++i)
    {
        DenseLayerWeights *lw = &s->layers[i];

        snprintf(name, sizeof(name), "blk.%u.attn_norm.weight", i);
        lw->attn_norm = asi_v2_find_tensor(cart, name);
        lw->attn_norm_f32 = dense_bind_1d_f32(s, name, H, 1, err_buf, err_cap);
        if (!lw->attn_norm || !lw->attn_norm_f32)
        {
            dense_free_session((AsiArchSession *)s);
            return -1;
        }

        snprintf(name, sizeof(name), "blk.%u.ffn_norm.weight", i);
        lw->ffn_norm = asi_v2_find_tensor(cart, name);
        if (lw->ffn_norm)
        {
            lw->ffn_norm_f32 = dense_bind_1d_f32(s, name, H, 1, err_buf, err_cap);
            if (!lw->ffn_norm_f32) { dense_free_session((AsiArchSession *)s); return -1; }

            snprintf(name, sizeof(name), "blk.%u.post_attention_norm.weight", i);
            if (asi_v2_find_tensor(cart, name))
            {
                lw->attn_post_norm = asi_v2_find_tensor(cart, name);
                lw->attn_post_norm_f32 = dense_bind_1d_f32(s, name, H, 1, err_buf, err_cap);
                if (!lw->attn_post_norm_f32) { dense_free_session((AsiArchSession *)s); return -1; }
            }
            snprintf(name, sizeof(name), "blk.%u.post_ffw_norm.weight", i);
            if (asi_v2_find_tensor(cart, name))
            {
                lw->ffn_post_norm = asi_v2_find_tensor(cart, name);
                lw->ffn_post_norm_f32 = dense_bind_1d_f32(s, name, H, 1, err_buf, err_cap);
                if (!lw->ffn_post_norm_f32) { dense_free_session((AsiArchSession *)s); return -1; }
            }
        }
        else
        {
            snprintf(name, sizeof(name), "blk.%u.post_attention_norm.weight", i);
            lw->ffn_norm = asi_v2_find_tensor(cart, name);
            lw->ffn_norm_f32 = dense_bind_1d_f32(s, name, H, 1, err_buf, err_cap);
            if (!lw->ffn_norm || !lw->ffn_norm_f32)
            {
                dense_set_err(err_buf, err_cap, "missing blk.%u.ffn_norm.weight");
                dense_free_session((AsiArchSession *)s);
                return -1;
            }
        }

        snprintf(name, sizeof(name), "blk.%u.attn_qkv.weight", i);
        if (asi_v2_find_tensor(cart, name))
        {
            lw->attn_qkv = dense_req_2d(cart, name, H, qkv_dim, err_buf, err_cap);
            if (!lw->attn_qkv) { dense_free_session((AsiArchSession *)s); return -1; }
            any_fused_qkv = 1;
            snprintf(name, sizeof(name), "blk.%u.attn_qkv.bias", i);
            if (asi_v2_find_tensor(cart, name))
            {
                lw->attn_qkv_bias = dense_bind_1d_f32(s, name, qkv_dim, 1, err_buf, err_cap);
                if (!lw->attn_qkv_bias) { dense_free_session((AsiArchSession *)s); return -1; }
            }
        }
        else
        {
            snprintf(name, sizeof(name), "blk.%u.attn_q.weight", i);
            lw->attn_q = dense_req_2d(cart, name, H, s->q_dim, err_buf, err_cap);
            snprintf(name, sizeof(name), "blk.%u.attn_k.weight", i);
            lw->attn_k = dense_req_2d(cart, name, H, s->kv_dim, err_buf, err_cap);
            snprintf(name, sizeof(name), "blk.%u.attn_v.weight", i);
            lw->attn_v = dense_req_2d(cart, name, H, s->kv_dim, err_buf, err_cap);
            if (!lw->attn_q || !lw->attn_k || !lw->attn_v)
            {
                dense_free_session((AsiArchSession *)s);
                return -1;
            }
        }

        snprintf(name, sizeof(name), "blk.%u.attn_q.bias", i);
        if (asi_v2_find_tensor(cart, name))
        {
            lw->attn_q_bias = dense_bind_1d_f32(s, name, s->q_dim, 1, err_buf, err_cap);
            if (!lw->attn_q_bias) { dense_free_session((AsiArchSession *)s); return -1; }
        }
        snprintf(name, sizeof(name), "blk.%u.attn_k.bias", i);
        if (asi_v2_find_tensor(cart, name))
        {
            lw->attn_k_bias = dense_bind_1d_f32(s, name, s->kv_dim, 1, err_buf, err_cap);
            if (!lw->attn_k_bias) { dense_free_session((AsiArchSession *)s); return -1; }
        }
        snprintf(name, sizeof(name), "blk.%u.attn_v.bias", i);
        if (asi_v2_find_tensor(cart, name))
        {
            lw->attn_v_bias = dense_bind_1d_f32(s, name, s->kv_dim, 1, err_buf, err_cap);
            if (!lw->attn_v_bias) { dense_free_session((AsiArchSession *)s); return -1; }
        }

        snprintf(name, sizeof(name), "blk.%u.attn_q_norm.weight", i);
        if (asi_v2_find_tensor(cart, name))
        {
            lw->attn_q_norm = dense_bind_1d_f32(s, name, hd, 1, err_buf, err_cap);
            if (!lw->attn_q_norm) { dense_free_session((AsiArchSession *)s); return -1; }
        }
        snprintf(name, sizeof(name), "blk.%u.attn_k_norm.weight", i);
        if (asi_v2_find_tensor(cart, name))
        {
            lw->attn_k_norm = dense_bind_1d_f32(s, name, hd, 1, err_buf, err_cap);
            if (!lw->attn_k_norm) { dense_free_session((AsiArchSession *)s); return -1; }
        }

        snprintf(name, sizeof(name), "blk.%u.attn_output.weight", i);
        lw->attn_output = dense_req_2d(cart, name, s->q_dim, H, err_buf, err_cap);
        if (!lw->attn_output) { dense_free_session((AsiArchSession *)s); return -1; }

        snprintf(name, sizeof(name), "blk.%u.attn_output.bias", i);
        if (asi_v2_find_tensor(cart, name))
        {
            lw->attn_output_bias = dense_bind_1d_f32(s, name, H, 1, err_buf, err_cap);
            if (!lw->attn_output_bias) { dense_free_session((AsiArchSession *)s); return -1; }
        }

        snprintf(name, sizeof(name), "blk.%u.ffn_gate.weight", i);
        lw->ffn_gate = dense_req_2d(cart, name, H, I, err_buf, err_cap);
        snprintf(name, sizeof(name), "blk.%u.ffn_up.weight", i);
        lw->ffn_up = dense_req_2d(cart, name, H, I, err_buf, err_cap);
        snprintf(name, sizeof(name), "blk.%u.ffn_down.weight", i);
        lw->ffn_down = dense_req_2d(cart, name, I, H, err_buf, err_cap);
        if (!lw->ffn_gate || !lw->ffn_up || !lw->ffn_down)
        {
            dense_free_session((AsiArchSession *)s);
            return -1;
        }

        snprintf(name, sizeof(name), "blk.%u.ffn_gate.bias", i);
        if (asi_v2_find_tensor(cart, name))
        {
            lw->ffn_gate_bias = dense_bind_1d_f32(s, name, I, 1, err_buf, err_cap);
            if (!lw->ffn_gate_bias) { dense_free_session((AsiArchSession *)s); return -1; }
        }
        snprintf(name, sizeof(name), "blk.%u.ffn_up.bias", i);
        if (asi_v2_find_tensor(cart, name))
        {
            lw->ffn_up_bias = dense_bind_1d_f32(s, name, I, 1, err_buf, err_cap);
            if (!lw->ffn_up_bias) { dense_free_session((AsiArchSession *)s); return -1; }
        }
        snprintf(name, sizeof(name), "blk.%u.ffn_down.bias", i);
        if (asi_v2_find_tensor(cart, name))
        {
            lw->ffn_down_bias = dense_bind_1d_f32(s, name, H, 1, err_buf, err_cap);
            if (!lw->ffn_down_bias) { dense_free_session((AsiArchSession *)s); return -1; }
        }

        size_t kv_elems = (size_t)s->max_ctx * s->kv_dim;
        s->kv_states[i].k_cache = (uint16_t *)calloc(kv_elems, sizeof(uint16_t));
        s->kv_states[i].v_cache = (uint16_t *)calloc(kv_elems, sizeof(uint16_t));
        if (!s->kv_states[i].k_cache || !s->kv_states[i].v_cache)
        {
            dense_set_err(err_buf, err_cap, "out of memory allocating KV cache");
            dense_free_session((AsiArchSession *)s);
            return -1;
        }
    }

    if (any_fused_qkv)
    {
        s->scratch.qkv = (float *)calloc(qkv_dim, sizeof(float));
        if (!s->scratch.qkv)
        {
            dense_set_err(err_buf, err_cap, "out of memory allocating qkv scratch");
            dense_free_session((AsiArchSession *)s);
            return -1;
        }
    }

    *out_session = (AsiArchSession *)s;
    return 0;
}

static void dense_split_qkv_row(const float *qkv, float *q, float *k, float *v,
                                const DenseLayerWeights *lw, uint32_t q_dim, uint32_t kv_dim)
{
    memcpy(q, qkv, (size_t)q_dim * sizeof(float));
    memcpy(k, qkv + q_dim, (size_t)kv_dim * sizeof(float));
    memcpy(v, qkv + q_dim + kv_dim, (size_t)kv_dim * sizeof(float));
    dense_add_vec(q, lw->attn_q_bias, q_dim);
    dense_add_vec(k, lw->attn_k_bias, kv_dim);
    dense_add_vec(v, lw->attn_v_bias, kv_dim);
}

static int dense_forward_token(AsiArchSession *session, int32_t token_id, uint32_t pos, float *out_logits)
{
    DenseSession *s = (DenseSession *)session;
    if (!s || s->magic != ASI_DENSE_SESSION_MAGIC || !out_logits || token_id < 0 || pos >= s->max_ctx)
        return -1;
    const AsiV2ModelConfig *c = &s->cart->config;
    if ((uint32_t)token_id >= c->vocab_size) return -1;

    uint32_t H = c->hidden_size;
    uint32_t I = c->intermediate_size;
    float attn_scale = (s->rope_attn_mscale * s->rope_attn_mscale) / sqrtf((float)s->head_dim);
    DenseScratch *sc = &s->scratch;

    asi_pool_begin(s->pool);
    if (dense_lookup_embed(s, token_id, sc->x) != 0)
    {
        asi_pool_end(s->pool);
        return -1;
    }

    if (s->stream_evict && c->n_layers > 0u)
        dense_stream_advise_layer(s->cart, &s->layers[0], 0);

    for (uint32_t l = 0; l < c->n_layers; ++l)
    {
        if (s->stream_evict && l + 1u < c->n_layers)
            dense_stream_advise_layer(s->cart, &s->layers[l + 1u], 0);
        const DenseLayerWeights *lw = &s->layers[l];
        DenseKVState *kv = &s->kv_states[l];

        dense_rmsnorm(sc->x_norm, sc->x, lw->attn_norm_f32, H, s->rms_norm_eps);

        if (lw->attn_qkv)
        {
            if (dense_matmul_multi(s, &lw->attn_qkv, &lw->attn_qkv_bias, &sc->qkv, 1,
                                     NULL, sc->x_norm, 1, (int)H, 0) != 0)
            {
                asi_pool_end(s->pool);
                return -1;
            }
            dense_split_qkv_row(sc->qkv, sc->q, sc->k, sc->v, lw, s->q_dim, s->kv_dim);
        }
        else
        {
            const AsiV2TensorEntry *qkv_ts[3] = {lw->attn_q, lw->attn_k, lw->attn_v};
            const float *qkv_bs[3] = {lw->attn_q_bias, lw->attn_k_bias, lw->attn_v_bias};
            float *qkv_outs[3] = {sc->q, sc->k, sc->v};
            if (dense_matmul_multi(s, qkv_ts, qkv_bs, qkv_outs, 3, NULL, sc->x_norm, 1, (int)H, 0) != 0)
            {
                asi_pool_end(s->pool);
                return -1;
            }
        }

        for (uint32_t h = 0; h < s->n_kv_heads; ++h)
        {
            float *kh = sc->k + (size_t)h * s->head_dim;
            if (lw->attn_k_norm)
                dense_rmsnorm(kh, kh, lw->attn_k_norm, s->head_dim, s->rms_norm_eps);
            if (s->rope_dim >= 2u)
                dense_apply_rope(kh, s->rope_dim, pos, s->rope_inv_freq, s->rope_mode);
        }
        dense_store_kv_fp16(kv->k_cache + (size_t)pos * s->kv_dim,
                            kv->v_cache + (size_t)pos * s->kv_dim,
                            sc->k, sc->v, s->kv_dim);

        DenseAttnCtx actx;
        memset(&actx, 0, sizeof(actx));
        actx.qb = sc->q;
        actx.aob = sc->attn_out;
        actx.k_cache = kv->k_cache;
        actx.v_cache = kv->v_cache;
        actx.qn_w = lw->attn_q_norm;
        actx.rope_inv_freq = s->rope_inv_freq;
        actx.B = 1u;
        actx.pos_start = pos;
        actx.n_heads = s->n_heads;
        actx.n_rep = s->n_heads / s->n_kv_heads;
        actx.hd = s->head_dim;
        actx.q_dim = s->q_dim;
        actx.kv_dim = s->kv_dim;
        actx.rope_dim = s->rope_dim;
        actx.rope_mode = s->rope_mode;
        actx.rms_eps = s->rms_norm_eps;
        actx.scale = attn_scale;
        atomic_init(&actx.next_task, 0);
        if (s->pool && s->n_threads > 1u)
            asi_pool_run(s->pool, dense_gqa_attn_worker, &actx);
        else
            dense_gqa_attn_worker(0, 1, &actx);

        if (dense_matmul_multi(s, &lw->attn_output, &lw->attn_output_bias, &sc->proj_out, 1,
                                 NULL, sc->attn_out, 1, (int)s->q_dim, 0) != 0)
        {
            asi_pool_end(s->pool);
            return -1;
        }
        if (lw->attn_post_norm_f32)
            dense_rmsnorm(sc->proj_out, sc->proj_out, lw->attn_post_norm_f32, H, s->rms_norm_eps);
        dense_add_vec(sc->x, sc->proj_out, H);

        dense_rmsnorm(sc->x_norm, sc->x, lw->ffn_norm_f32, H, s->rms_norm_eps);
        const AsiV2TensorEntry *ffn_ts[2] = {lw->ffn_gate, lw->ffn_up};
        const float *ffn_bs[2] = {lw->ffn_gate_bias, lw->ffn_up_bias};
        float *ffn_outs[2] = {sc->ffn_gate, sc->ffn_up};
        if (dense_matmul_multi(s, ffn_ts, ffn_bs, ffn_outs, 2, NULL, sc->x_norm, 1, (int)H, 1) != 0)
        {
            asi_pool_end(s->pool);
            return -1;
        }
        if (dense_matmul_multi(s, &lw->ffn_down, &lw->ffn_down_bias, &sc->proj_out, 1,
                                 NULL, sc->ffn_gate, 1, (int)I, 0) != 0)
        {
            asi_pool_end(s->pool);
            return -1;
        }
        if (lw->ffn_post_norm_f32)
            dense_rmsnorm(sc->proj_out, sc->proj_out, lw->ffn_post_norm_f32, H, s->rms_norm_eps);
        dense_add_vec(sc->x, sc->proj_out, H);
        if (s->stream_evict)
            dense_stream_advise_layer(s->cart, lw, 1);
    }

    dense_rmsnorm(sc->x_norm, sc->x, s->output_norm_f32, H, s->rms_norm_eps);
    int rc = dense_eval_lm_head(s, sc->x_norm, out_logits);
    asi_pool_end(s->pool);
    if (rc == 0) s->context_len = pos + 1u;
    return rc;
}

static int dense_ensure_batch_scratch(DenseSession *s, uint32_t B)
{
    DenseScratch *sc = &s->scratch;
    if (B <= sc->batch_cap) return 0;

    const AsiV2ModelConfig *c = &s->cart->config;
    uint32_t H = c->hidden_size;
    uint32_t I = c->intermediate_size;
    uint32_t q_dim = s->q_dim;
    uint32_t kv_dim = s->kv_dim;
    uint32_t qkv_dim = q_dim + 2u * kv_dim;
    uint32_t max_in = H;
    if (I > max_in) max_in = I;
    if (q_dim > max_in) max_in = q_dim;
    int bpr_max = ((int)max_in + QK_K - 1) / QK_K;

    free(sc->xb); free(sc->xnb); free(sc->projb);
    free(sc->qb); free(sc->kb); free(sc->vb); free(sc->qkvb);
    free(sc->aob); free(sc->fgb); free(sc->fub);
    dense_aligned_free(sc->qx_h);

    sc->xb    = (float *)malloc((size_t)B * H * sizeof(float));
    sc->xnb   = (float *)malloc((size_t)B * H * sizeof(float));
    sc->projb = (float *)malloc((size_t)B * H * sizeof(float));
    sc->qb    = (float *)malloc((size_t)B * q_dim * sizeof(float));
    sc->kb    = (float *)malloc((size_t)B * kv_dim * sizeof(float));
    sc->vb    = (float *)malloc((size_t)B * kv_dim * sizeof(float));
    sc->qkvb  = (float *)malloc((size_t)B * qkv_dim * sizeof(float));
    sc->aob   = (float *)malloc((size_t)B * q_dim * sizeof(float));
    sc->fgb   = (float *)malloc((size_t)B * I * sizeof(float));
    sc->fub   = (float *)malloc((size_t)B * I * sizeof(float));
    sc->qx_h  = (LilaQ8KBlock *)dense_aligned_alloc(64u, (size_t)B * (size_t)bpr_max * sizeof(LilaQ8KBlock));

    if (!sc->xb || !sc->xnb || !sc->projb || !sc->qb || !sc->kb || !sc->vb ||
        !sc->qkvb || !sc->aob || !sc->fgb || !sc->fub || !sc->qx_h)
    {
        sc->batch_cap = 0;
        return -1;
    }
    sc->batch_cap = B;
    return 0;
}

typedef struct {
    const DenseSession *s;
    const int32_t *token_ids;
    float *xb;
    uint32_t B;
} DenseEmbdBatchCtx;

static void dense_embd_batch_worker(int tid, int n_threads, void *vctx)
{
    const DenseEmbdBatchCtx *ctx = (const DenseEmbdBatchCtx *)vctx;
    uint32_t t0 = (ctx->B * (uint32_t)tid) / (uint32_t)n_threads;
    uint32_t t1 = (ctx->B * (uint32_t)(tid + 1)) / (uint32_t)n_threads;
    uint32_t H = ctx->s->cart->config.hidden_size;
    for (uint32_t t = t0; t < t1; ++t)
        dense_lookup_embed(ctx->s, ctx->token_ids[t], ctx->xb + (size_t)t * H);
}

typedef struct {
    float *xb;
    float *projb;
    float *xnb;
    LilaQ8KBlock *qx_all;
    const float *w_post_norm;
    const float *w_next_norm;
    uint32_t B;
    uint32_t H;
    float eps;
    int do_add;
    int do_norm;
    int do_quant;
} DenseNormQuantCtx;

static void dense_norm_quant_worker(int tid, int n_threads, void *vctx)
{
    const DenseNormQuantCtx *ctx = (const DenseNormQuantCtx *)vctx;
    uint32_t t0 = (ctx->B * (uint32_t)tid) / (uint32_t)n_threads;
    uint32_t t1 = (ctx->B * (uint32_t)(tid + 1)) / (uint32_t)n_threads;
    uint32_t H = ctx->H;
    for (uint32_t t = t0; t < t1; ++t)
    {
        float *x = ctx->xb + (size_t)t * H;
        if (ctx->do_add)
        {
            float *p = ctx->projb + (size_t)t * H;
            if (ctx->w_post_norm)
                dense_rmsnorm(p, p, ctx->w_post_norm, H, ctx->eps);
            dense_add_vec(x, p, H);
        }
        if (ctx->do_norm)
        {
            float *xn = ctx->xnb + (size_t)t * H;
            dense_rmsnorm(xn, x, ctx->w_next_norm, H, ctx->eps);
            if (ctx->do_quant)
                quantize_vec_q8k_strided(ctx->qx_all + t, (int)ctx->B, xn, (int)H);
        }
    }
}

static void dense_norm_quant_pool(AsiThreadPool *pool,
                                  float *xb, float *projb, float *xnb,
                                  LilaQ8KBlock *qx_all,
                                  const float *w_post_norm,
                                  const float *w_next_norm,
                                  uint32_t B, uint32_t H, float eps,
                                  int do_add, int do_norm, int do_quant)
{
    DenseNormQuantCtx ctx = {
        xb, projb, xnb, qx_all, w_post_norm, w_next_norm, B, H, eps, do_add, do_norm, do_quant
    };
    if (pool && B >= 8u)
        asi_pool_run(pool, dense_norm_quant_worker, &ctx);
    else
        dense_norm_quant_worker(0, 1, &ctx);
}

static int dense_forward_chunk(DenseSession *s,
                               const int32_t *token_ids,
                               uint32_t B,
                               uint32_t pos_start,
                               int compute_logits,
                               float *out_last_logits)
{
    if (dense_ensure_batch_scratch(s, B) != 0) return -1;
    const AsiV2ModelConfig *c = &s->cart->config;
    uint32_t H = c->hidden_size;
    uint32_t I = c->intermediate_size;
    float attn_scale = (s->rope_attn_mscale * s->rope_attn_mscale) / sqrtf((float)s->head_dim);
    DenseScratch *sc = &s->scratch;

    DenseEmbdBatchCtx ectx = {s, token_ids, sc->xb, B};
    if (s->pool && B >= 16u)
        asi_pool_run(s->pool, dense_embd_batch_worker, &ectx);
    else
        dense_embd_batch_worker(0, 1, &ectx);

    dense_norm_quant_pool(s->pool, sc->xb, NULL, sc->xnb, sc->qx_h,
                          NULL, s->layers[0].attn_norm_f32, B, H, s->rms_norm_eps,
                          0, 1, !s->fp32_ref);

    if (s->stream_evict && c->n_layers > 0u)
        dense_stream_advise_layer(s->cart, &s->layers[0], 0);

    for (uint32_t l = 0; l < c->n_layers; ++l)
    {
        if (s->stream_evict && l + 1u < c->n_layers)
            dense_stream_advise_layer(s->cart, &s->layers[l + 1u], 0);
        const DenseLayerWeights *lw = &s->layers[l];
        DenseKVState *kv = &s->kv_states[l];

        if (lw->attn_qkv)
        {
            if (dense_matmul_multi(s, &lw->attn_qkv, &lw->attn_qkv_bias, &sc->qkvb, 1,
                                     s->fp32_ref ? NULL : sc->qx_h, sc->xnb, (int)B, (int)H, 0) != 0)
                return -1;
            uint32_t qkv_dim = s->q_dim + 2u * s->kv_dim;
            for (uint32_t t = 0; t < B; ++t)
            {
                dense_split_qkv_row(sc->qkvb + (size_t)t * qkv_dim,
                                    sc->qb + (size_t)t * s->q_dim,
                                    sc->kb + (size_t)t * s->kv_dim,
                                    sc->vb + (size_t)t * s->kv_dim,
                                    lw, s->q_dim, s->kv_dim);
            }
        }
        else
        {
            const AsiV2TensorEntry *qkv_ts[3] = {lw->attn_q, lw->attn_k, lw->attn_v};
            const float *qkv_bs[3] = {lw->attn_q_bias, lw->attn_k_bias, lw->attn_v_bias};
            float *qkv_outs[3] = {sc->qb, sc->kb, sc->vb};
            if (dense_matmul_multi(s, qkv_ts, qkv_bs, qkv_outs, 3,
                                     s->fp32_ref ? NULL : sc->qx_h, sc->xnb, (int)B, (int)H, 0) != 0)
                return -1;
        }

        DenseKvPrepCtx kctx = {
            sc->kb, sc->vb, kv->k_cache, kv->v_cache,
            lw->attn_k_norm, s->rope_inv_freq, s->rope_mode,
            B, pos_start, s->n_kv_heads, s->head_dim, s->kv_dim, s->rope_dim, s->rms_norm_eps
        };
        if (s->pool && B >= 8u)
            asi_pool_run(s->pool, dense_kv_prep_worker, &kctx);
        else
            dense_kv_prep_worker(0, 1, &kctx);

        DenseAttnCtx actx;
        memset(&actx, 0, sizeof(actx));
        actx.qb = sc->qb;
        actx.aob = sc->aob;
        actx.k_cache = kv->k_cache;
        actx.v_cache = kv->v_cache;
        actx.qn_w = lw->attn_q_norm;
        actx.rope_inv_freq = s->rope_inv_freq;
        actx.B = B;
        actx.pos_start = pos_start;
        actx.n_heads = s->n_heads;
        actx.n_rep = s->n_heads / s->n_kv_heads;
        actx.hd = s->head_dim;
        actx.q_dim = s->q_dim;
        actx.kv_dim = s->kv_dim;
        actx.rope_dim = s->rope_dim;
        actx.rope_mode = s->rope_mode;
        actx.rms_eps = s->rms_norm_eps;
        actx.scale = attn_scale;
        atomic_init(&actx.next_task, 0);
        if (s->pool && s->n_threads > 1u)
            asi_pool_run(s->pool, dense_gqa_attn_worker, &actx);
        else
            dense_gqa_attn_worker(0, 1, &actx);

        if (!s->fp32_ref && (lw->attn_output->quant_type == 12u || lw->attn_output->quant_type == 14u))
            dense_quantize_batch_q8k(s->pool, sc->qx_h, sc->aob, (int)B, (int)s->q_dim);
        if (dense_matmul_multi(s, &lw->attn_output, &lw->attn_output_bias, &sc->projb, 1,
                                 s->fp32_ref ? NULL : sc->qx_h, sc->aob, (int)B, (int)s->q_dim, 0) != 0)
            return -1;

        dense_norm_quant_pool(s->pool, sc->xb, sc->projb, sc->xnb, sc->qx_h,
                              lw->attn_post_norm_f32, lw->ffn_norm_f32, B, H, s->rms_norm_eps,
                              1, 1, !s->fp32_ref);

        const AsiV2TensorEntry *ffn_ts[2] = {lw->ffn_gate, lw->ffn_up};
        const float *ffn_bs[2] = {lw->ffn_gate_bias, lw->ffn_up_bias};
        float *ffn_outs[2] = {sc->fgb, sc->fub};
        if (dense_matmul_multi(s, ffn_ts, ffn_bs, ffn_outs, 2,
                                 s->fp32_ref ? NULL : sc->qx_h, sc->xnb, (int)B, (int)H, 1) != 0)
            return -1;

        if (!s->fp32_ref && (lw->ffn_down->quant_type == 12u || lw->ffn_down->quant_type == 14u))
            dense_quantize_batch_q8k(s->pool, sc->qx_h, sc->fgb, (int)B, (int)I);
        if (dense_matmul_multi(s, &lw->ffn_down, &lw->ffn_down_bias, &sc->projb, 1,
                                 s->fp32_ref ? NULL : sc->qx_h, sc->fgb, (int)B, (int)I, 0) != 0)
            return -1;

        const float *next_an = (l + 1u < c->n_layers) ? s->layers[l + 1u].attn_norm_f32 : NULL;
        dense_norm_quant_pool(s->pool, sc->xb, sc->projb, sc->xnb, sc->qx_h,
                              lw->ffn_post_norm_f32, next_an, B, H, s->rms_norm_eps,
                              1, (next_an != NULL), (next_an != NULL && !s->fp32_ref));
        if (s->stream_evict)
            dense_stream_advise_layer(s->cart, lw, 1);
    }

    int rc = 0;
    if (compute_logits && out_last_logits)
    {
        dense_rmsnorm(sc->x_norm, sc->xb + (size_t)(B - 1u) * H, s->output_norm_f32, H, s->rms_norm_eps);
        rc = dense_eval_lm_head(s, sc->x_norm, out_last_logits);
    }
    return rc;
}

static int dense_forward_batch(AsiArchSession *session,
                               const int32_t *token_ids,
                               uint32_t n_tokens,
                               uint32_t pos_start,
                               float *out_last_logits)
{
    DenseSession *s = (DenseSession *)session;
    if (!s || s->magic != ASI_DENSE_SESSION_MAGIC || !token_ids || n_tokens == 0u || !out_last_logits)
        return -1;
    if (pos_start + n_tokens > s->max_ctx) return -1;
    uint32_t vocab = s->cart->config.vocab_size;
    for (uint32_t i = 0; i < n_tokens; ++i)
    {
        if (token_ids[i] < 0 || (uint32_t)token_ids[i] >= vocab) return -1;
    }
    if (n_tokens == 1u)
        return dense_forward_token(session, token_ids[0], pos_start, out_last_logits);

    /* LILA_ASI_BATCH_CHUNK_MAX: Maximum prompt tokens per dense prefill chunk batch (default: ASI_DENSE_DEFAULT_BATCH_CHUNK = 512). */
    uint32_t chunk_max = ASI_DENSE_DEFAULT_BATCH_CHUNK;
    const char *env_cm = getenv("LILA_ASI_BATCH_CHUNK_MAX");
    if (env_cm && env_cm[0] != '\0')
    {
        long parsed = strtol(env_cm, NULL, 10);
        if (parsed >= 1 && parsed <= 1048576L) chunk_max = (uint32_t)parsed;
    }

    asi_pool_begin(s->pool);
    uint32_t done = 0;
    while (done < n_tokens)
    {
        uint32_t B = n_tokens - done;
        if (B > chunk_max) B = chunk_max;
        int is_last = (done + B == n_tokens);
        if (dense_forward_chunk(s, token_ids + done, B, pos_start + done, is_last, out_last_logits) != 0)
        {
            asi_pool_end(s->pool);
            return -1;
        }
        done += B;
    }
    asi_pool_end(s->pool);
    s->context_len = pos_start + n_tokens;
    return 0;
}

/* ASIPFX02 prefix snapshot header (144 bytes, matches asi_arch.c layout). */
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
} DensePrefixSnapshotHeader;

_Static_assert(sizeof(DensePrefixSnapshotHeader) == 144, "DensePrefixSnapshotHeader must be 144 bytes");
_Static_assert(offsetof(DensePrefixSnapshotHeader, header_crc64) == 136, "header_crc64 offset mismatch");

static uint64_t g_dense_crc64_tab[256];
static AsiOnce g_dense_crc64_once = ASI_ONCE_INIT;

static void dense_crc64_init(void)
{
    for (uint64_t i = 0; i < 256; ++i)
    {
        uint64_t crc = i << 56;
        for (int b = 0; b < 8; ++b)
            crc = (crc & 0x8000000000000000ULL) ? ((crc << 1) ^ 0x42F0E1EBA9EA3693ULL) : (crc << 1);
        g_dense_crc64_tab[i] = crc;
    }
}

static uint64_t dense_crc64_update(uint64_t crc, const uint8_t *data, size_t len)
{
    asi_call_once(&g_dense_crc64_once, dense_crc64_init);
    for (size_t i = 0; i < len; ++i)
        crc = (crc << 8) ^ g_dense_crc64_tab[((crc >> 56) ^ data[i]) & 0xFFu];
    return crc;
}

static int dense_save_prefix(AsiArchSession *session, uint32_t n_tokens)
{
    DenseSession *s = (DenseSession *)session;
    if (!s || s->magic != ASI_DENSE_SESSION_MAGIC || n_tokens == 0u || n_tokens > s->max_ctx)
        return -1;
    uint32_t n_layers = s->cart->config.n_layers;
    size_t kv_per_layer = (size_t)n_tokens * s->kv_dim;
    size_t total_elems = (size_t)n_layers * kv_per_layer;

    uint16_t *new_k = (uint16_t *)malloc(total_elems * sizeof(uint16_t));
    uint16_t *new_v = (uint16_t *)malloc(total_elems * sizeof(uint16_t));
    if (!new_k || !new_v)
    {
        free(new_k);
        free(new_v);
        return -1;
    }
    for (uint32_t l = 0; l < n_layers; ++l)
    {
        memcpy(new_k + (size_t)l * kv_per_layer, s->kv_states[l].k_cache, kv_per_layer * sizeof(uint16_t));
        memcpy(new_v + (size_t)l * kv_per_layer, s->kv_states[l].v_cache, kv_per_layer * sizeof(uint16_t));
    }
    free(s->prefix_k);
    free(s->prefix_v);
    s->prefix_k = new_k;
    s->prefix_v = new_v;
    s->prefix_len = n_tokens;
    return 0;
}

static int dense_restore_prefix(AsiArchSession *session, uint32_t *out_n_tokens)
{
    DenseSession *s = (DenseSession *)session;
    if (!s || s->magic != ASI_DENSE_SESSION_MAGIC || s->prefix_len == 0u || !s->prefix_k || !s->prefix_v)
        return -1;
    uint32_t n_layers = s->cart->config.n_layers;
    size_t kv_per_layer = (size_t)s->prefix_len * s->kv_dim;
    size_t tail_elems = (size_t)(s->max_ctx - s->prefix_len) * s->kv_dim;
    for (uint32_t l = 0; l < n_layers; ++l)
    {
        memcpy(s->kv_states[l].k_cache, s->prefix_k + (size_t)l * kv_per_layer, kv_per_layer * sizeof(uint16_t));
        memcpy(s->kv_states[l].v_cache, s->prefix_v + (size_t)l * kv_per_layer, kv_per_layer * sizeof(uint16_t));
        if (tail_elems > 0)
        {
            memset(s->kv_states[l].k_cache + kv_per_layer, 0, tail_elems * sizeof(uint16_t));
            memset(s->kv_states[l].v_cache + kv_per_layer, 0, tail_elems * sizeof(uint16_t));
        }
    }
    s->context_len = s->prefix_len;
    if (out_n_tokens) *out_n_tokens = s->prefix_len;
    return 0;
}

static int dense_save_prefix_file(const AsiArchSession *session,
                                  const uint8_t key_hash[32],
                                  const char *filepath)
{
    const DenseSession *s = (const DenseSession *)session;
    if (!s || s->magic != ASI_DENSE_SESSION_MAGIC || s->prefix_len == 0u ||
        !s->prefix_k || !s->prefix_v || !key_hash || !filepath || filepath[0] == '\0')
        return -1;

    uint32_t n_layers = s->cart->config.n_layers;
    uint64_t kv_bytes = (uint64_t)n_layers * (uint64_t)s->prefix_len * (uint64_t)s->kv_dim * sizeof(uint16_t);

    DensePrefixSnapshotHeader hdr;
    memset(&hdr, 0, sizeof(hdr));
    memcpy(hdr.magic, "ASIPFX02", 8);
    hdr.numerics_version = LILA_ASI_NUMERICS_VERSION;
    hdr.arch_id = s->cart->config.arch_id;
    memcpy(hdr.cart_identity_hash, s->cart->header.identity_hash, 24);
    memcpy(hdr.key_hash, key_hash, 32);
    hdr.fp32_ref = (uint32_t)(s->fp32_ref ? 1u : 0u);
    hdr.prefix_len = s->prefix_len;
    hdr.n_dn = 0u;
    hdr.n_fa = n_layers;
    hdr.k_bytes = kv_bytes;
    hdr.v_bytes = kv_bytes;

    uint64_t pcrc = 0;
    pcrc = dense_crc64_update(pcrc, (const uint8_t *)s->prefix_k, (size_t)kv_bytes);
    pcrc = dense_crc64_update(pcrc, (const uint8_t *)s->prefix_v, (size_t)kv_bytes);
    hdr.payload_crc64 = pcrc;
    hdr.header_crc64 = dense_crc64_update(0, (const uint8_t *)&hdr, offsetof(DensePrefixSnapshotHeader, header_crc64));

    size_t flen = strlen(filepath);
    size_t tcap = flen + 64u;
    char *tmp_path = (char *)malloc(tcap);
    if (!tmp_path) return -1;
    int tn = snprintf(tmp_path, tcap, "%s.tmp.%ld", filepath, asi_getpid());
    if (tn <= 0 || (size_t)tn >= tcap) { free(tmp_path); return -1; }

    FILE *fp = fopen(tmp_path, "wb");
    if (!fp) { free(tmp_path); return -1; }

    if (fwrite(&hdr, 1, sizeof(hdr), fp) != sizeof(hdr) ||
        fwrite(s->prefix_k, 1, (size_t)kv_bytes, fp) != (size_t)kv_bytes ||
        fwrite(s->prefix_v, 1, (size_t)kv_bytes, fp) != (size_t)kv_bytes)
    {
        fclose(fp); asi_unlink(tmp_path); free(tmp_path);
        return -1;
    }
    if (fflush(fp) != 0) { fclose(fp); asi_unlink(tmp_path); free(tmp_path); return -1; }
    if (asi_fsync_file(fp) != 0)
    {
        fclose(fp); asi_unlink(tmp_path); free(tmp_path);
        return -1;
    }
    if (fclose(fp) != 0 || asi_atomic_rename(tmp_path, filepath) != 0)
    {
        asi_unlink(tmp_path); free(tmp_path);
        return -1;
    }
    free(tmp_path);
    return 0;
}

static int dense_load_prefix_file(AsiArchSession *session,
                                  const uint8_t expected_key_hash[32],
                                  const char *filepath,
                                  uint32_t *out_n_tokens)
{
    DenseSession *s = (DenseSession *)session;
    if (!s || s->magic != ASI_DENSE_SESSION_MAGIC || !expected_key_hash || !filepath || filepath[0] == '\0')
        return -1;

    FILE *fp = fopen(filepath, "rb");
    if (!fp) return -1;

    DensePrefixSnapshotHeader hdr;
    uint16_t *new_k = NULL;
    uint16_t *new_v = NULL;

    if (fread(&hdr, 1, sizeof(hdr), fp) != sizeof(hdr)) goto fail;
    if (memcmp(hdr.magic, "ASIPFX02", 8) != 0) goto fail;

    uint64_t expected_hcrc = dense_crc64_update(0, (const uint8_t *)&hdr,
                                                offsetof(DensePrefixSnapshotHeader, header_crc64));
    if (hdr.header_crc64 != expected_hcrc) goto fail;
    if (hdr.numerics_version != LILA_ASI_NUMERICS_VERSION) goto fail;
    if (hdr.arch_id != s->cart->config.arch_id) goto fail;
    if (memcmp(hdr.cart_identity_hash, s->cart->header.identity_hash, 24) != 0) goto fail;
    if (memcmp(hdr.key_hash, expected_key_hash, 32) != 0) goto fail;
    if (hdr.fp32_ref != (uint32_t)(s->fp32_ref ? 1u : 0u)) goto fail;
    if (hdr.prefix_len == 0u || hdr.prefix_len > s->max_ctx) goto fail;

    uint32_t n_layers = s->cart->config.n_layers;
    uint64_t exp_kv = (uint64_t)n_layers * (uint64_t)hdr.prefix_len * (uint64_t)s->kv_dim * sizeof(uint16_t);
    if (hdr.n_dn != 0u || hdr.n_fa != n_layers || hdr.conv_bytes != 0u ||
        hdr.ssm_bytes != 0u || hdr.k_bytes != exp_kv || hdr.v_bytes != exp_kv)
        goto fail;

    new_k = (uint16_t *)malloc((size_t)exp_kv);
    new_v = (uint16_t *)malloc((size_t)exp_kv);
    if (!new_k || !new_v) goto fail;
    if (fread(new_k, 1, (size_t)exp_kv, fp) != (size_t)exp_kv ||
        fread(new_v, 1, (size_t)exp_kv, fp) != (size_t)exp_kv)
        goto fail;

    uint8_t extra = 0;
    if (fread(&extra, 1, 1, fp) != 0 || ferror(fp)) goto fail;
    fclose(fp);
    fp = NULL;

    uint64_t pcrc = 0;
    pcrc = dense_crc64_update(pcrc, (const uint8_t *)new_k, (size_t)exp_kv);
    pcrc = dense_crc64_update(pcrc, (const uint8_t *)new_v, (size_t)exp_kv);
    if (pcrc != hdr.payload_crc64) goto fail;

    free(s->prefix_k);
    free(s->prefix_v);
    s->prefix_k = new_k;
    s->prefix_v = new_v;
    s->prefix_len = hdr.prefix_len;
    return dense_restore_prefix(session, out_n_tokens);

fail:
    if (fp) fclose(fp);
    free(new_k);
    free(new_v);
    return -1;
}

uint32_t asi_dense_session_context_len(const AsiArchSession *session)
{
    const DenseSession *s = (const DenseSession *)session;
    if (!s || s->magic != ASI_DENSE_SESSION_MAGIC) return 0u;
    return s->context_len;
}

int asi_dense_set_lm_head_hook(AsiArchSession *session, AsiDenseLmHeadHook hook, void *user_ctx)
{
    DenseSession *s = (DenseSession *)session;
    if (!s || s->magic != ASI_DENSE_SESSION_MAGIC) return -1;
    s->lm_head_hook = hook;
    s->lm_head_hook_ctx = hook ? user_ctx : NULL;
    return 0;
}

const AsiV2TensorEntry *asi_dense_get_lm_head_tensor(const AsiArchSession *session)
{
    const DenseSession *s = (const DenseSession *)session;
    if (!s || s->magic != ASI_DENSE_SESSION_MAGIC) return NULL;
    return s->output_weight ? s->output_weight : s->token_embd;
}

const float *asi_dense_get_output_norm_f32(const AsiArchSession *session)
{
    const DenseSession *s = (const DenseSession *)session;
    if (!s || s->magic != ASI_DENSE_SESSION_MAGIC) return NULL;
    return s->output_norm_f32;
}

const AsiArchHandler g_asi_arch_dense_transformer = {
    .arch_id = ASI_ARCH_DENSE_LLAMA,
    .name = "dense_transformer",
    .init_session = dense_init_session,
    .free_session = dense_free_session,
    .forward_token = dense_forward_token,
    .forward_batch = dense_forward_batch,
    .reset_state = dense_reset_state,
    .save_prefix = dense_save_prefix,
    .restore_prefix = dense_restore_prefix,
    .save_prefix_file = dense_save_prefix_file,
    .load_prefix_file = dense_load_prefix_file,
    .forward_speculative = NULL,
};
