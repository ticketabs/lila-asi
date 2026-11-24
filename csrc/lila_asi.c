#include "lila_asi.h"
#include "asi_arch.h"
#include "asi_dense.h"
#include "lila_lm_head_candidate.h"
#include "asi_sc_internal.h"

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
#include <math.h>
#include <omp.h>
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
        int procs = omp_get_num_procs();
        n_threads = (procs > 2) ? (uint32_t)(procs / 2) : (uint32_t)procs;
        if (n_threads == 0) n_threads = 1;
    }
    s->n_threads = n_threads;
    omp_set_num_threads((int)n_threads);

    if (model->handler->init_session(&model->cart, max_ctx, n_threads,
                                     &s->arch_session, err_buf, err_cap) != 0)
    {
        free(s);
        return NULL;
    }
    if (model->handler == &g_asi_arch_dense_transformer)
    {
        const AsiV2ModelConfig *cfg = &model->cart.config;
        const char *fp32_env = getenv("LILA_ASI_FP32_REF");
        int fp32_ref = (fp32_env && fp32_env[0] != '\0' && strcmp(fp32_env, "0") != 0) ? 1 : 0;
        const char *preset_env = getenv("LILA_ASI_MODE");
        LilaLhcMode preset_mode = (!fp32_ref && cfg->vocab_size > 4096u && cfg->hidden_size == 1024u)
                                      ? LILA_LHC_MODE_1BIT_1024
                                      : LILA_LHC_MODE_OFF;
        uint32_t preset_k = 2048u;
        if (preset_env && preset_env[0] != '\0')
        {
            if (strcasecmp(preset_env, "balanced") == 0)
            {
                preset_mode = (!fp32_ref && cfg->vocab_size > 4096u && cfg->hidden_size == 1024u)
                                  ? LILA_LHC_MODE_2BIT_1024
                                  : LILA_LHC_MODE_OFF;
                preset_k = 1024u;
            }
            else if (strcasecmp(preset_env, "exact") == 0)
            {
                preset_mode = LILA_LHC_MODE_OFF;
            }
        }
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
    omp_set_num_threads((int)session->n_threads);
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
    omp_set_num_threads((int)session->n_threads);
    return session->model->handler->forward_batch(
        session->arch_session, token_ids, n_tokens, pos_start, out_last_logits);
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
