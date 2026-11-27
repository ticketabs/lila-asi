#include "lila_asi.h"
#include "asi_arch.h"
#include "asi_dense.h"
#include "lila_lm_head_candidate.h"
#include "asi_sc_internal.h"
#include "q4k.h"

typedef struct {
    const AsiV2Cartridge *cart;
    LilaLhcIndex *idx;
    LilaLhcWorkspace *ws;
    int fp32_ref;
} DenseLhcHookCtx;

static int dense_lhc_hook_cb(void *user_ctx,
                             const AsiV2TensorEntry *lm_head,
                             const float *x_norm,
                             float *out_logits)
{
    DenseLhcHookCtx *ctx = (DenseLhcHookCtx *)user_ctx;
    if (!ctx || !ctx->cart || !ctx->idx || !ctx->ws || !lm_head) return -1;
    const uint8_t *w = ctx->cart->weights_data + lm_head->data_offset;
    return lila_lhc_forward_pool(ctx->idx, ctx->ws, NULL, NULL, w, x_norm, out_logits, ctx->fp32_ref);
}
#include "asi_cpu.h"
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct LilaAsiModel {
    AsiV2Cartridge cart;
    const AsiArchHandler *handler;
};

struct LilaAsiSession {
    LilaAsiModel *model;
    AsiArchSession *arch_session;
    uint32_t n_threads;
    DenseLhcHookCtx *dense_lhc;
    AsiStateTree *state_tree;
};

LilaAsiModel *lila_asi_open(const char *asi_path, char *err_buf, size_t err_cap)
{
    LilaAsiModel *m = (LilaAsiModel *)calloc(1, sizeof(LilaAsiModel));
    if (!m)
    {
        if (err_buf && err_cap > 0) snprintf(err_buf, err_cap, "out of memory");
        return NULL;
    }
    if (asi_v2_open(asi_path, &m->cart, err_buf, err_cap) != 0)
    {
        free(m);
        return NULL;
    }
    m->handler = asi_arch_find(m->cart.header.arch_id);
    if (!m->handler || !m->handler->init_session)
    {
        if (err_buf && err_cap > 0)
        {
            snprintf(err_buf, err_cap, "unsupported architecture '%s'",
                     asi_v2_arch_name(m->cart.header.arch_id));
        }
        asi_v2_close(&m->cart);
        free(m);
        return NULL;
    }
    return m;
}

void lila_asi_close(LilaAsiModel *model)
{
    if (!model) return;
    asi_v2_close(&model->cart);
    free(model);
}

int lila_asi_model_info(const LilaAsiModel *model, AsiV2ModelConfig *out_cfg)
{
    if (!model || !out_cfg) return -1;
    *out_cfg = model->cart.config;
    return 0;
}

int lila_asi_model_header(const LilaAsiModel *model, AsiV2Header *out_hdr)
{
    if (!model || !out_hdr) return -1;
    *out_hdr = model->cart.header;
    return 0;
}

int lila_asi_get_section(const LilaAsiModel *model,
                         uint32_t section_type,
                         const void **out_ptr,
                         size_t *out_size)
{
    if (!model || !out_ptr || !out_size) return -1;
    const AsiV2SectionEntry *sec = asi_v2_find_section(&model->cart, section_type);
    if (!sec) return -1;
    *out_ptr = model->cart.base + sec->offset;
    *out_size = (size_t)sec->size;
    return 0;
}

int lila_asi_export_sieve_section(const LilaAsiModel *model,
                                  const char *mode_str,
                                  uint8_t *out_buf,
                                  size_t out_cap,
                                  size_t *out_written)
{
    if (!model || !out_written) return -1;
    const AsiV2Cartridge *cart = &model->cart;
    const AsiV2ModelConfig *c = &cart->config;
    const AsiV2TensorEntry *lm_head = asi_v2_find_tensor(cart, "output.weight");
    if (!lm_head) lm_head = asi_v2_find_tensor(cart, "token_embd.weight");
    const AsiV2TensorEntry *out_norm = asi_v2_find_tensor(cart, "output_norm.weight");
    if (!lm_head || !out_norm || !cart->weights_data) return -1;

    LilaLhcMode mode = (mode_str && mode_str[0]) ? lila_lhc_parse_mode(mode_str) : LILA_LHC_MODE_1BIT_1024;
    if (mode == LILA_LHC_MODE_OFF) return -1;

    int is_repacked = (lm_head->layout_flags & ASI_TENSOR_LAYOUT_AVX2_REPACK) ? 1 : 0;
    const uint8_t *w_ptr = cart->weights_data + lm_head->data_offset;
    const float *norm_ptr = (out_norm->quant_type == 0)
        ? (const float *)(cart->weights_data + out_norm->data_offset)
        : NULL;

    LilaLhcIndex *idx = lila_lhc_build_quant_pool(
        mode, w_ptr, lm_head->quant_type, is_repacked,
        c->vocab_size, c->hidden_size, norm_ptr, NULL, NULL);
    if (!idx) return -1;

    int rc = lila_lhc_serialize_section(idx, out_buf, out_cap, out_written);
    lila_lhc_free(idx);
    return rc;
}

int lila_asi_verify_signature(const LilaAsiModel *model,
                              const uint8_t *trusted_pubkey_32,
                              AsiV2SignatureInfo *out_info,
                              char *err_buf,
                              size_t err_cap)
{
    if (!model)
    {
        if (err_buf && err_cap > 0) snprintf(err_buf, err_cap, "null model");
        return -1;
    }
    return asi_v2_verify_signature(&model->cart, trusted_pubkey_32, out_info, err_buf, err_cap);
}

void lila_asi_cpu_info(char *buf, size_t cap)
{
    asi_cpu_feature_string(buf, cap);
}

LilaAsiSession *lila_asi_session_create(LilaAsiModel *model,
                                         uint32_t max_ctx,
                                         uint32_t n_threads,
                                         char *err_buf,
                                         size_t err_cap)
{
    if (!model || !model->handler) return NULL;
    LilaAsiSession *s = (LilaAsiSession *)calloc(1, sizeof(LilaAsiSession));
    if (!s) return NULL;
    s->model = model;
    if (n_threads == 0)
    {
        int procs = asi_num_procs();
        n_threads = (procs > 2) ? (uint32_t)(procs / 2) : (uint32_t)procs;
        if (n_threads == 0) n_threads = 1;
    }
    s->n_threads = n_threads;
#ifdef _OPENMP
    omp_set_num_threads((int)n_threads);
#endif

    if (model->handler->init_session(&model->cart, max_ctx, n_threads,
                                     &s->arch_session, err_buf, err_cap) != 0)
    {
        free(s);
        return NULL;
    }
    if (model->handler == &g_asi_arch_dense_transformer)
    {
        const AsiV2ModelConfig *cfg = &model->cart.config;
        /* LILA_ASI_FP32_REF=1 bypasses quantized SIMD paths for unquantized FP32 parity checks. */
        const char *fp32_env = getenv("LILA_ASI_FP32_REF");
        int fp32_ref = (fp32_env && fp32_env[0] != '\0' && strcmp(fp32_env, "0") != 0) ? 1 : 0;
        /* LILA_ASI_MODE=fast|balanced|exact sets dense-transformer Sieve sketch & K preset. */
        const char *preset_env = getenv("LILA_ASI_MODE");
        LilaLhcMode preset_mode = (!fp32_ref && cfg->vocab_size > 4096u && cfg->hidden_size == 1024u)
                                      ? LILA_LHC_MODE_1BIT_1024
                                      : LILA_LHC_MODE_OFF;
        uint32_t preset_k = 2048u;
        if (preset_env && preset_env[0] != '\0')
        {
            if (asi_strcasecmp(preset_env, "balanced") == 0)
            {
                preset_mode = (!fp32_ref && cfg->vocab_size > 4096u && cfg->hidden_size == 1024u)
                                  ? LILA_LHC_MODE_2BIT_1024
                                  : LILA_LHC_MODE_OFF;
                preset_k = 1024u;
            }
            else if (asi_strcasecmp(preset_env, "exact") == 0)
            {
                preset_mode = LILA_LHC_MODE_OFF;
            }
        }
        /* LILA_ASI_LMHEAD_MODE / LILA_ASI_LMHEAD_K explicitly override dense-model Sieve settings. */
        const char *lhm_env = getenv("LILA_ASI_LMHEAD_MODE");
        LilaLhcMode mode = fp32_ref
                               ? LILA_LHC_MODE_OFF
                               : ((lhm_env && lhm_env[0] != '\0') ? lila_lhc_parse_mode(lhm_env) : preset_mode);
        if (mode != LILA_LHC_MODE_OFF)
        {
            const AsiV2TensorEntry *lm_head = asi_dense_get_lm_head_tensor(s->arch_session);
            const float *out_norm = asi_dense_get_output_norm_f32(s->arch_session);
            if (lm_head && lila_quant_is_supported(lm_head->quant_type) &&
                cfg->hidden_size == 1024u && cfg->vocab_size > 0u)
            {
                int lm_repacked = (lm_head->layout_flags & ASI_TENSOR_LAYOUT_AVX2_REPACK) ? 1 : 0;
                uint32_t k = preset_k;
                const char *k_env = getenv("LILA_ASI_LMHEAD_K");
                if (k_env && k_env[0] != '\0')
                {
                    long kv = strtol(k_env, NULL, 10);
                    if (kv >= 1) k = (uint32_t)kv;
                }
                if (k > cfg->vocab_size) k = cfg->vocab_size;
                DenseLhcHookCtx *dctx = (DenseLhcHookCtx *)calloc(1, sizeof(DenseLhcHookCtx));
                if (dctx)
                {
                    dctx->cart = &model->cart;
                    dctx->fp32_ref = fp32_ref;
                    dctx->idx = lila_lhc_build_quant(mode, model->cart.weights_data + lm_head->data_offset,
                                                     lm_head->quant_type, lm_repacked,
                                                     cfg->vocab_size, cfg->hidden_size, out_norm);
                    dctx->ws = lila_lhc_workspace_create(cfg->vocab_size, k, 1u, 1u);
                    if (dctx->idx && dctx->ws &&
                        asi_dense_set_lm_head_hook(s->arch_session, dense_lhc_hook_cb, dctx) == 0)
                    {
                        s->dense_lhc = dctx;
                    }
                    else
                    {
                        lila_lhc_workspace_free(dctx->ws);
                        lila_lhc_free(dctx->idx);
                        free(dctx);
                    }
                }
            }
        }
    }
    return s;
}

void lila_asi_session_free(LilaAsiSession *session)
{
    if (!session) return;
    if (session->state_tree)
    {
        lila_asi_sc_state_tree_free(session->state_tree);
        session->state_tree = NULL;
    }
    if (session->dense_lhc)
    {
        if (session->arch_session)
            asi_dense_set_lm_head_hook(session->arch_session, NULL, NULL);
        lila_lhc_workspace_free(session->dense_lhc->ws);
        lila_lhc_free(session->dense_lhc->idx);
        free(session->dense_lhc);
        session->dense_lhc = NULL;
    }
    if (session->model && session->model->handler && session->model->handler->free_session)
    {
        session->model->handler->free_session(session->arch_session);
    }
    free(session);
}

void lila_asi_session_reset(LilaAsiSession *session)
{
    if (!session || !session->model || !session->model->handler) return;
    if (session->model->handler->reset_state)
    {
        session->model->handler->reset_state(session->arch_session);
    }
}

int lila_asi_session_save_prefix(LilaAsiSession *session, uint32_t n_tokens)
{
    if (!session || !session->model || !session->model->handler) return -1;
    if (!session->model->handler->save_prefix) return -1;
    return session->model->handler->save_prefix(session->arch_session, n_tokens);
}

int lila_asi_session_restore_prefix(LilaAsiSession *session, uint32_t *out_n_tokens)
{
    if (!session || !session->model || !session->model->handler) return -1;
    if (!session->model->handler->restore_prefix) return -1;
    return session->model->handler->restore_prefix(session->arch_session, out_n_tokens);
}

int lila_asi_session_save_prefix_file(const LilaAsiSession *session,
                                      const uint8_t key_hash[32],
                                      const char *filepath)
{
    if (!session || !session->model || !session->model->handler) return -1;
    if (!session->model->handler->save_prefix_file) return -1;
    return session->model->handler->save_prefix_file(session->arch_session, key_hash, filepath);
}

int lila_asi_session_load_prefix_file(LilaAsiSession *session,
                                      const uint8_t expected_key_hash[32],
                                      const char *filepath,
                                      uint32_t *out_n_tokens)
{
    if (!session || !session->model || !session->model->handler) return -1;
    if (!session->model->handler->load_prefix_file) return -1;
    return session->model->handler->load_prefix_file(session->arch_session,
                                                     expected_key_hash,
                                                     filepath,
                                                     out_n_tokens);
}

int lila_asi_eval_token(LilaAsiSession *session,
                        int32_t token_id,
                        uint32_t pos,
                        float *out_logits)
{
    if (!session || !session->model || !session->model->handler) return -1;
#ifdef _OPENMP
    omp_set_num_threads((int)session->n_threads);
#endif
    if (session->model->handler->forward_speculative)
    {
        return session->model->handler->forward_speculative(
            session->arch_session, token_id, pos, out_logits);
    }
    return session->model->handler->forward_token(session->arch_session, token_id, pos, out_logits);
}

int lila_asi_eval_batch(LilaAsiSession *session,
                        const int32_t *token_ids,
                        uint32_t n_tokens,
                        uint32_t pos_start,
                        float *out_last_logits)
{
    if (!session || !session->model || !session->model->handler) return -1;
#ifdef _OPENMP
    omp_set_num_threads((int)session->n_threads);
#endif
    return session->model->handler->forward_batch(
        session->arch_session, token_ids, n_tokens, pos_start, out_last_logits);
}

int lila_asi_eval_embd(LilaAsiSession *session,
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
    if (!session || !session->model || !session->model->handler ||
        !session->model->handler->forward_embd)
    {
        return -1;
    }
#ifdef _OPENMP
    omp_set_num_threads((int)session->n_threads);
#endif
    return session->model->handler->forward_embd(
        session->arch_session, embds, n_tokens, kv_pos_start,
        mrope_pos_3d, mrope_sections, mrope_flags, next_rope_pos,
        cancel_flag, out_last_logits);
}

int64_t lila_asi_session_get_rope_delta(const LilaAsiSession *session)
{
    if (!session || !session->model || !session->model->handler ||
        !session->model->handler->get_rope_delta)
    {
        return 0;
    }
    return session->model->handler->get_rope_delta(session->arch_session);
}

void lila_asi_session_set_rope_delta(LilaAsiSession *session, int64_t delta)
{
    if (!session || !session->model || !session->model->handler ||
        !session->model->handler->set_rope_delta)
    {
        return;
    }
    session->model->handler->set_rope_delta(session->arch_session, delta);
}

uint32_t lila_asi_build_mrope_grid(uint32_t rope_pos_start,
                                   const uint32_t grid_thw[3],
                                   uint32_t *out_pos_3d,
                                   size_t out_cap_u32)
{
    if (!grid_thw || !out_pos_3d) return 0u;
    uint32_t T = grid_thw[0];
    uint32_t H = grid_thw[1];
    uint32_t W = grid_thw[2];
    if (T == 0u || H == 0u || W == 0u) return 0u;
    size_t needed = (size_t)T * (size_t)H * (size_t)W * 3u;
    if (needed > out_cap_u32) return 0u;

    size_t k = 0u;
    for (uint32_t t = 0; t < T; ++t)
    {
        for (uint32_t r = 0; r < H; ++r)
        {
            for (uint32_t c = 0; c < W; ++c)
            {
                out_pos_3d[k++] = rope_pos_start + t;
                out_pos_3d[k++] = rope_pos_start + r;
                out_pos_3d[k++] = rope_pos_start + c;
            }
        }
    }
    uint32_t max_dim = T;
    if (H > max_dim) max_dim = H;
    if (W > max_dim) max_dim = W;
    return rope_pos_start + max_dim;
}

typedef struct {
    int32_t id;
    float logit;
} CandLogit;

static int cmp_cand_desc(const void *a, const void *b)
{
    float la = ((const CandLogit *)a)->logit;
    float lb = ((const CandLogit *)b)->logit;
    return (lb > la) - (lb < la);
}

static float rng_uniform(uint64_t *state)
{
    uint64_t x = (*state) ? *state : 0x9E3779B97F4A7C15ULL;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return (float)((x * 0x2545F4914F6CDD1DULL) >> 40) / (float)(1ULL << 24);
}

int32_t lila_asi_sample(const float *logits,
                        uint32_t vocab_size,
                        float temperature,
                        uint32_t top_k,
                        float top_p,
                        uint64_t *rng_state)
{
    if (!logits || vocab_size == 0) return -1;
    if (temperature <= 0.0f)
    {
        int32_t best = 0;
        float best_v = logits[0];
        for (uint32_t i = 1; i < vocab_size; ++i)
        {
            if (logits[i] > best_v)
            {
                best_v = logits[i];
                best = (int32_t)i;
            }
        }
        return best;
    }

    uint32_t k = (top_k > 0 && top_k < vocab_size) ? top_k : 64;
    CandLogit *cands = (CandLogit *)malloc((size_t)k * sizeof(CandLogit));
    if (!cands) return 0;

    uint32_t count = 0;
    float min_v = -1e30f;
    uint32_t min_idx = 0;
    for (uint32_t i = 0; i < vocab_size; ++i)
    {
        float v = logits[i];
        if (count < k)
        {
            cands[count].id = (int32_t)i;
            cands[count].logit = v;
            if (count == 0 || v < min_v)
            {
                min_v = v;
                min_idx = count;
            }
            count++;
        }
        else if (v > min_v)
        {
            cands[min_idx].id = (int32_t)i;
            cands[min_idx].logit = v;
            min_v = cands[0].logit;
            min_idx = 0;
            for (uint32_t j = 1; j < k; ++j)
            {
                if (cands[j].logit < min_v)
                {
                    min_v = cands[j].logit;
                    min_idx = j;
                }
            }
        }
    }

    qsort(cands, count, sizeof(CandLogit), cmp_cand_desc);

    float max_l = cands[0].logit;
    float sum_p = 0.0f;
    for (uint32_t i = 0; i < count; ++i)
    {
        cands[i].logit = expf((cands[i].logit - max_l) / temperature);
        sum_p += cands[i].logit;
    }
    for (uint32_t i = 0; i < count; ++i)
    {
        cands[i].logit /= sum_p;
    }

    if (top_p > 0.0f && top_p < 1.0f)
    {
        float cum = 0.0f;
        uint32_t cut = count;
        for (uint32_t i = 0; i < count; ++i)
        {
            cum += cands[i].logit;
            if (cum >= top_p)
            {
                cut = i + 1;
                break;
            }
        }
        count = cut;
        float renorm = 0.0f;
        for (uint32_t i = 0; i < count; ++i) renorm += cands[i].logit;
        for (uint32_t i = 0; i < count; ++i) cands[i].logit /= renorm;
    }

    float r = rng_uniform(rng_state);
    float acc = 0.0f;
    int32_t chosen = cands[count - 1].id;
    for (uint32_t i = 0; i < count; ++i)
    {
        acc += cands[i].logit;
        if (r <= acc)
        {
            chosen = cands[i].id;
            break;
        }
    }
    free(cands);
    return chosen;
}

AsiArchSession *lila_asi_sc_get_arch_session(const LilaAsiSession *session)
{
    return session ? session->arch_session : NULL;
}

const AsiV2Cartridge *lila_asi_sc_get_cartridge(const LilaAsiSession *session)
{
    return (session && session->model) ? &session->model->cart : NULL;
}

AsiStateTree *lila_asi_sc_ensure_state_tree(LilaAsiSession *session)
{
    if (!session) return NULL;
    if (!session->state_tree)
    {
        session->state_tree = lila_asi_sc_state_tree_create(session);
    }
    return session->state_tree;
}

const AsiStateTree *lila_asi_sc_get_state_tree(const LilaAsiSession *session)
{
    return session ? session->state_tree : NULL;
}

void lila_asi_session_set_calib_hook(LilaAsiSession *session,
                                     LilaAsiCalibHookFn fn,
                                     void *user_ctx)
{
    if (!session || !session->model || !session->model->handler ||
        !session->model->handler->set_calib_hook)
    {
        return;
    }
    session->model->handler->set_calib_hook(session->arch_session, fn, user_ctx);
}

int lila_asi_dequant_f32(const uint8_t *src,
                         uint32_t quant_type,
                         uint32_t layout_flags,
                         int cols,
                         int rows,
                         float *dst_f32)
{
    if (!src || !dst_f32 || cols <= 0 || rows <= 0) return -1;
    if (!lila_quant_is_supported(quant_type)) return -1;
    int is_repacked = (layout_flags & ASI_TENSOR_LAYOUT_AVX2_REPACK) ? 1 : 0;
    int is_col_major = (layout_flags & ASI_TENSOR_LAYOUT_COL_MAJOR) ? 1 : 0;
    int r;
    #pragma omp parallel for schedule(static) if(rows >= 16)
    for (r = 0; r < rows; ++r)
    {
        lila_dequant_row_at(dst_f32 + (size_t)r * (size_t)cols, src, r, cols, rows,
                            quant_type, is_repacked, is_col_major);
    }
    return 0;
}

static void asi_quant_group64_q4r(const float *w, int avail, int search_steps,
                                  uint16_t *out_d, uint8_t *out_qs)
{
    float buf[Q4_REPACK_GROUP_SIZE];
    float amax = 0.0f;
    for (int i = 0; i < avail; ++i)
    {
        buf[i] = w[i];
        float a = fabsf(w[i]);
        if (a > amax) amax = a;
    }
    for (int i = avail; i < Q4_REPACK_GROUP_SIZE; ++i) buf[i] = 0.0f;

    if (amax == 0.0f)
    {
        *out_d = 0u;
        memset(out_qs, 0x88, 32);
        return;
    }

    int best_q[Q4_REPACK_GROUP_SIZE];
    uint16_t best_d = 0u;
    double best_err = 1e300;

    float s0 = amax / 7.5f;
    int steps = (search_steps > 1) ? search_steps : 1;
    for (int si = 0; si < steps; ++si)
    {
        float mul = (steps == 1) ? 1.0f : (0.72f + 0.32f * (float)si / (float)(steps - 1));
        float s_try = s0 * mul;
        float inv_s = (s_try > 0.0f) ? (1.0f / s_try) : 0.0f;
        int q_cur[Q4_REPACK_GROUP_SIZE];
        double num = 0.0, den = 0.0;
        for (int i = 0; i < avail; ++i)
        {
            int q = (int)lrintf(buf[i] * inv_s);
            if (q < -8) q = -8;
            else if (q > 7) q = 7;
            q_cur[i] = q;
            num += (double)buf[i] * (double)q;
            den += (double)q * (double)q;
        }
        for (int i = avail; i < Q4_REPACK_GROUP_SIZE; ++i) q_cur[i] = 0;

        float s_opt = (steps > 1 && den > 0.0) ? (float)(num / den) : s_try;
        if (s_opt <= 0.0f) s_opt = s_try;
        uint16_t d16 = fp32_to_fp16(s_opt);
        float d_val = fp16_to_fp32(d16);
        if (steps > 1 && d_val > 0.0f)
        {
            /* Re-snap codes to the fp16-quantized scale */
            float inv_dv = 1.0f / d_val;
            for (int i = 0; i < avail; ++i)
            {
                int q = (int)lrintf(buf[i] * inv_dv);
                if (q < -8) q = -8;
                else if (q > 7) q = 7;
                q_cur[i] = q;
            }
        }
        double err = 0.0;
        for (int i = 0; i < avail; ++i)
        {
            double diff = (double)buf[i] - (double)d_val * (double)q_cur[i];
            err += diff * diff;
        }
        if (err < best_err)
        {
            best_err = err;
            best_d = d16;
            memcpy(best_q, q_cur, sizeof(best_q));
        }
    }

    *out_d = best_d;
    for (int l = 0; l < 32; ++l)
    {
        uint8_t u0 = (uint8_t)(best_q[l] + 8);
        uint8_t u1 = (uint8_t)(best_q[l + 32] + 8);
        out_qs[l] = (uint8_t)((u0 & 0x0Fu) | ((u1 & 0x0Fu) << 4));
    }
}

int lila_asi_requant_q4_repack(const float *src_f32,
                               int rows,
                               int cols,
                               int col_major,
                               int search_steps,
                               uint8_t *dst)
{
    if (!src_f32 || !dst || rows <= 0 || cols <= 0) return -1;
    if (!col_major)
    {
        int n_blk = (cols + Q4_REPACK_GROUP_SIZE - 1) / Q4_REPACK_GROUP_SIZE;
        size_t row_bytes = (size_t)n_blk * Q4_REPACK_BLOCK_SIZE;
        int r;
        #pragma omp parallel for schedule(static) if(rows >= 16)
        for (r = 0; r < rows; ++r)
        {
            const float *row_in = src_f32 + (size_t)r * (size_t)cols;
            uint8_t *row_out = dst + (size_t)r * row_bytes;
            for (int b = 0; b < n_blk; ++b)
            {
                int base = b * Q4_REPACK_GROUP_SIZE;
                int avail = cols - base;
                if (avail > Q4_REPACK_GROUP_SIZE) avail = Q4_REPACK_GROUP_SIZE;
                uint16_t d16 = 0;
                uint8_t *blk_out = row_out + (size_t)b * Q4_REPACK_BLOCK_SIZE;
                asi_quant_group64_q4r(row_in + base, avail, search_steps, &d16, blk_out + 2);
                memcpy(blk_out, &d16, 2);
            }
        }
        return 0;
    }

    int n_blk = (rows + Q4_REPACK_GROUP_SIZE - 1) / Q4_REPACK_GROUP_SIZE;
    size_t col_bytes = (size_t)n_blk * Q4_REPACK_BLOCK_SIZE;
    int c;
    #pragma omp parallel for schedule(static) if(cols >= 16)
    for (c = 0; c < cols; ++c)
    {
        uint8_t *col_out = dst + (size_t)c * col_bytes;
        float grp[Q4_REPACK_GROUP_SIZE];
        for (int b = 0; b < n_blk; ++b)
        {
            int base = b * Q4_REPACK_GROUP_SIZE;
            int avail = rows - base;
            if (avail > Q4_REPACK_GROUP_SIZE) avail = Q4_REPACK_GROUP_SIZE;
            for (int i = 0; i < avail; ++i)
            {
                grp[i] = src_f32[(size_t)(base + i) * (size_t)cols + (size_t)c];
            }
            uint16_t d16 = 0;
            uint8_t *blk_out = col_out + (size_t)b * Q4_REPACK_BLOCK_SIZE;
            asi_quant_group64_q4r(grp, avail, search_steps, &d16, blk_out + 2);
            memcpy(blk_out, &d16, 2);
        }
    }
    return 0;
}

static void asi_quant_group128_q3(const float *w, int avail, int search_steps,
                                  uint16_t *out_step, int q3_out[Q3_SPLITLUT_BLK_COLS])
{
    float buf[Q3_SPLITLUT_BLK_COLS];
    double ss = 0.0;
    for (int i = 0; i < avail; ++i)
    {
        buf[i] = w[i];
        ss += (double)w[i] * (double)w[i];
    }
    for (int i = avail; i < Q3_SPLITLUT_BLK_COLS; ++i) buf[i] = 0.0f;

    if (ss == 0.0)
    {
        *out_step = 0u;
        for (int i = 0; i < Q3_SPLITLUT_BLK_COLS; ++i) q3_out[i] = 3;
        return;
    }

    float stdw = (float)sqrt(ss / (double)(avail > 0 ? avail : 1)) + 1e-12f;
    float step0 = 0.5860f * stdw;
    int steps = (search_steps > 1) ? search_steps : 1;
    uint16_t best_step = 0u;
    double best_err = 1e300;
    int best_q[Q3_SPLITLUT_BLK_COLS];

    for (int si = 0; si < steps; ++si)
    {
        float mul = (steps == 1) ? 1.0f : (0.72f + 0.52f * (float)si / (float)(steps - 1));
        float st_try = step0 * mul;
        float inv_st = (st_try > 0.0f) ? (1.0f / st_try) : 0.0f;
        int q_cur[Q3_SPLITLUT_BLK_COLS];
        double num = 0.0, den = 0.0;
        for (int i = 0; i < avail; ++i)
        {
            int q = (int)floorf(buf[i] * inv_st + 4.0f);
            if (q < 0) q = 0;
            else if (q > 7) q = 7;
            q_cur[i] = q;
            double vq = (double)q - 3.5;
            num += (double)buf[i] * vq;
            den += vq * vq;
        }
        for (int i = avail; i < Q3_SPLITLUT_BLK_COLS; ++i) q_cur[i] = 3;

        float st_opt = (steps > 1 && den > 0.0) ? (float)(num / den) : st_try;
        if (st_opt <= 0.0f) st_opt = st_try;
        uint16_t st16 = fp32_to_fp16(st_opt);
        float st_val = fp16_to_fp32(st16);
        if (steps > 1 && st_val > 0.0f)
        {
            float inv_sv = 1.0f / st_val;
            for (int i = 0; i < avail; ++i)
            {
                int q = (int)floorf(buf[i] * inv_sv + 4.0f);
                if (q < 0) q = 0;
                else if (q > 7) q = 7;
                q_cur[i] = q;
            }
        }
        double err = 0.0;
        for (int i = 0; i < avail; ++i)
        {
            double diff = (double)buf[i] - (double)st_val * ((double)q_cur[i] - 3.5);
            err += diff * diff;
        }
        if (err < best_err)
        {
            best_err = err;
            best_step = st16;
            memcpy(best_q, q_cur, sizeof(best_q));
        }
    }

    *out_step = best_step;
    memcpy(q3_out, best_q, sizeof(best_q));
}

int lila_asi_requant_q3_splitlut(const float *src_f32,
                                 int rows,
                                 int cols,
                                 int search_steps,
                                 uint8_t *dst)
{
    if (!src_f32 || !dst || rows <= 0 || cols <= 0) return -1;
    int n_tiles = (rows + Q3_SPLITLUT_TILE_ROWS - 1) / Q3_SPLITLUT_TILE_ROWS;
    int n_blk128 = (cols + Q3_SPLITLUT_BLK_COLS - 1) / Q3_SPLITLUT_BLK_COLS;
    size_t tile_bytes = (size_t)n_blk128 * Q3_SPLITLUT_BLK_BYTES;
    memset(dst, 0, (size_t)n_tiles * tile_bytes);

    int tile;
    #pragma omp parallel for schedule(static) if(n_tiles >= 2)
    for (tile = 0; tile < n_tiles; ++tile)
    {
        int r_base = tile * Q3_SPLITLUT_TILE_ROWS;
        int valid = rows - r_base;
        if (valid > Q3_SPLITLUT_TILE_ROWS) valid = Q3_SPLITLUT_TILE_ROWS;
        uint8_t *tp = dst + (size_t)tile * tile_bytes;

        for (int lr = 0; lr < valid; ++lr)
        {
            int r = r_base + lr;
            int bo = lr & 31;
            int sh = (lr >= 32) ? 4 : 0;
            const float *row_in = src_f32 + (size_t)r * (size_t)cols;

            for (int blk = 0; blk < n_blk128; ++blk)
            {
                int base = blk * Q3_SPLITLUT_BLK_COLS;
                int avail = cols - base;
                if (avail > Q3_SPLITLUT_BLK_COLS) avail = Q3_SPLITLUT_BLK_COLS;
                uint16_t sc16 = 0;
                int q3[Q3_SPLITLUT_BLK_COLS];
                asi_quant_group128_q3(row_in + base, avail, search_steps, &sc16, q3);

                uint8_t *bp_h = tp + (size_t)blk * Q3_SPLITLUT_BLK_BYTES;
                uint8_t *bp_l = bp_h + Q3_SPLITLUT_BLK_HI2_BYTES;
                uint8_t *bp_s = bp_l + Q3_SPLITLUT_BLK_LO1_BYTES;
                memcpy(bp_s + (size_t)lr * 2u, &sc16, 2);

                for (int g = 0; g < 64; ++g)
                {
                    uint8_t nib = (uint8_t)((q3[g * 2 + 0] >> 1) | ((q3[g * 2 + 1] >> 1) << 2));
                    bp_h[(size_t)g * 32u + (size_t)bo] |= (uint8_t)(nib << sh);
                }
                for (int g = 0; g < 32; ++g)
                {
                    uint8_t nib = (uint8_t)((q3[g * 4 + 0] & 1) |
                                            ((q3[g * 4 + 1] & 1) << 1) |
                                            ((q3[g * 4 + 2] & 1) << 2) |
                                            ((q3[g * 4 + 3] & 1) << 3));
                    bp_l[(size_t)g * 32u + (size_t)bo] |= (uint8_t)(nib << sh);
                }
            }
        }
    }
    return 0;
}
