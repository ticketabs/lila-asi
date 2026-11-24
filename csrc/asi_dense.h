#ifndef LILA_ASI_DENSE_H
#define LILA_ASI_DENSE_H

#include "asi_arch.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Optional two-stage lm_head screening hook (Exp E03 readiness).
 *
 * When registered on a dense session, dense_eval_lm_head invokes this hook
 * instead of the default full-vocab matvec for single-token / final-token
 * logit computation:
 *   - lm_head: selected lm_head tensor (output.weight or tied token_embd.weight)
 *   - x_norm:  post-output_norm FP32 hidden state [hidden_size]
 *   - out_logits: caller-provided output buffer [vocab_size]
 * Return 0 on success (hook populated out_logits), or non-zero to fail.
 */
typedef int (*AsiDenseLmHeadHook)(void *user_ctx,
                                 const AsiV2TensorEntry *lm_head,
                                 const float *x_norm,
                                 float *out_logits);

/* Return current active KV context length (0 after init/reset, pos+1 after eval). */
uint32_t asi_dense_session_context_len(const AsiArchSession *session);

/* Register or clear (hook == NULL) an optional lm_head screening hook on session. */
int asi_dense_set_lm_head_hook(AsiArchSession *session,
                               AsiDenseLmHeadHook hook,
                               void *user_ctx);

/* Inspect the bound lm_head tensor entry (output.weight or tied token_embd.weight). */
const AsiV2TensorEntry *asi_dense_get_lm_head_tensor(const AsiArchSession *session);

/* Inspect the FP32 output_norm.weight vector [hidden_size] for sketch calibration. */
const float *asi_dense_get_output_norm_f32(const AsiArchSession *session);

#ifdef __cplusplus
}
#endif

#endif /* LILA_ASI_DENSE_H */
