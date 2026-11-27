#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "asi_grammar_sparse.h"
#include "asi_platform.h"
#include "asi_v2.h"
#include "lila_asi.h"
#include "lila_lm_head_candidate.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static inline double e05_now_ms(void) {
    return (double)asi_monotonic_ns() * 1e-6;
}

extern int lila_lhc_forward_pool_raw(const LilaLhcIndex *index,
                                     LilaLhcWorkspace *ws,
                                     LilaLhcDispatchFn dispatch_fn,
                                     void *pool_opaque,
                                     const uint8_t *q6k_weights,
                                     const float *x_norm,
                                     float *out_logits,
                                     int fp32_ref);

/* 0 during ordinary unconstrained decode/prefill so lila_lhc_forward_pool has zero hook overhead */
int g_e05_hook_fast_flag = 0;

typedef struct {
    int mode; /* 0 = unconstrained Sieve, 1 = E05 Grammar-Aware Sparse */
    int skip_lm_head;
    int track_stats;
    AsiGrammarEngine *grammar;
    uint32_t vocab_size;
    uint32_t hidden_size;

    uint64_t lm_head_calls;
    uint64_t sparse_calls;
    uint64_t sieve_calls;
    uint64_t sparse_rows_sum;
    double   lm_head_ms_sum;
    double   sparse_ms_sum;
    double   sieve_ms_sum;
} E05HookState;

static E05HookState g_e05_hook;

static inline int e05_env_disabled(void) {
    const char *env = getenv("LILA_ASI_DISABLE_E05");
    return (env != NULL && (env[0] == '1' || env[0] == 't' || env[0] == 'T' || env[0] == 'y' || env[0] == 'Y'));
}

static inline void e05_update_fast_flag(void) {
    if (e05_env_disabled()) {
        g_e05_hook_fast_flag = 0;
        return;
    }
    g_e05_hook_fast_flag = (g_e05_hook.skip_lm_head != 0 ||
                            (g_e05_hook.mode == 1 && g_e05_hook.grammar != NULL) ||
                            g_e05_hook.track_stats != 0) ? 1 : 0;
}

typedef struct {
    const LilaLhcIndex *index;
    LilaLhcWorkspace *ws;
    LilaLhcDispatchFn dispatch_fn;
    void *pool_opaque;
} E05SieveBridgeCtx;

static int e05_sieve_bridge_cb(void *sieve_ctx,
                               const uint8_t *q6k_weights,
                               const float *x_norm,
                               float *out_logits,
                               int fp32_ref) {
    E05SieveBridgeCtx *b = (E05SieveBridgeCtx *)sieve_ctx;
    return lila_lhc_forward_pool_raw(b->index, b->ws, b->dispatch_fn, b->pool_opaque,
                                     q6k_weights, x_norm, out_logits, fp32_ref);
}

int e05_lhc_forward_pool_intercept(const LilaLhcIndex *index,
                                   LilaLhcWorkspace *ws,
                                   LilaLhcDispatchFn dispatch_fn,
                                   void *pool_opaque,
                                   const uint8_t *q6k_weights,
                                   const float *x_norm,
                                   float *out_logits,
                                   int fp32_ref) {
    if (g_e05_hook.skip_lm_head) {
        return 0;
    }

    double t0 = e05_now_ms();
    int rc = 0;

    if (g_e05_hook.mode == 1 && g_e05_hook.grammar != NULL) {
        E05SieveBridgeCtx bctx = { index, ws, dispatch_fn, pool_opaque };
        AsiGrammarDispatchMode dmode = 0;
        const int32_t *legal_ids = NULL;
        uint32_t n_legal = 0u;
        asi_grammar_get_legal_ids(g_e05_hook.grammar, &legal_ids, &n_legal);

        rc = asi_grammar_lm_head_forward_quant(g_e05_hook.grammar, q6k_weights,
                                               lila_lhc_quant_type(index),
                                               lila_lhc_is_repacked(index),
                                               g_e05_hook.vocab_size, g_e05_hook.hidden_size,
                                               x_norm, out_logits, fp32_ref,
                                               dispatch_fn, pool_opaque,
                                               e05_sieve_bridge_cb, &bctx, &dmode);
        double dt = e05_now_ms() - t0;
        g_e05_hook.lm_head_calls++;
        g_e05_hook.lm_head_ms_sum += dt;
        if (dmode == ASI_DISPATCH_SPARSE_Q6K) {
            g_e05_hook.sparse_calls++;
            g_e05_hook.sparse_rows_sum += n_legal;
            g_e05_hook.sparse_ms_sum += dt;
        } else {
            g_e05_hook.sieve_calls++;
            g_e05_hook.sieve_ms_sum += dt;
        }
    } else {
        rc = lila_lhc_forward_pool_raw(index, ws, dispatch_fn, pool_opaque,
                                       q6k_weights, x_norm, out_logits, fp32_ref);
        double dt = e05_now_ms() - t0;
        g_e05_hook.lm_head_calls++;
        g_e05_hook.sieve_calls++;
        g_e05_hook.lm_head_ms_sum += dt;
        g_e05_hook.sieve_ms_sum += dt;
    }

    return rc;
}

void e05_hook_configure(int mode, AsiGrammarEngine *g, uint32_t vocab_size, uint32_t hidden_size) {
    g_e05_hook.mode = mode;
    g_e05_hook.grammar = g;
    if (vocab_size > 0u) g_e05_hook.vocab_size = vocab_size;
    if (hidden_size > 0u) g_e05_hook.hidden_size = hidden_size;
    e05_update_fast_flag();
}

void e05_hook_set_skip_lm_head(int skip) {
    g_e05_hook.skip_lm_head = skip;
    e05_update_fast_flag();
}

void e05_hook_reset_stats(void) {
    g_e05_hook.lm_head_calls = 0;
    g_e05_hook.sparse_calls = 0;
    g_e05_hook.sieve_calls = 0;
    g_e05_hook.sparse_rows_sum = 0;
    g_e05_hook.lm_head_ms_sum = 0.0;
    g_e05_hook.sparse_ms_sum = 0.0;
    g_e05_hook.sieve_ms_sum = 0.0;
    g_e05_hook.track_stats = 1;
    e05_update_fast_flag();
}

void e05_hook_get_stats(uint64_t *lm_head_calls,
                        uint64_t *sparse_calls,
                        uint64_t *sieve_calls,
                        uint64_t *sparse_rows_sum,
                        double *lm_head_ms_sum,
                        double *sparse_ms_sum,
                        double *sieve_ms_sum) {
    if (lm_head_calls) *lm_head_calls = g_e05_hook.lm_head_calls;
    if (sparse_calls) *sparse_calls = g_e05_hook.sparse_calls;
    if (sieve_calls) *sieve_calls = g_e05_hook.sieve_calls;
    if (sparse_rows_sum) *sparse_rows_sum = g_e05_hook.sparse_rows_sum;
    if (lm_head_ms_sum) *lm_head_ms_sum = g_e05_hook.lm_head_ms_sum;
    if (sparse_ms_sum) *sparse_ms_sum = g_e05_hook.sparse_ms_sum;
    if (sieve_ms_sum) *sieve_ms_sum = g_e05_hook.sieve_ms_sum;
}

int e05_load_vocab_from_model(const LilaAsiModel *model, AsiGrammarVocab *out_vocab) {
    const void *tok_sec = NULL;
    size_t tok_sec_sz = 0;
    if (lila_asi_get_section(model, ASI_SEC_TOKENIZER, &tok_sec, &tok_sec_sz) != 0) {
        return -1;
    }
    return asi_grammar_vocab_from_asi_json(out_vocab, tok_sec, tok_sec_sz);
}
