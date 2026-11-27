#ifndef LILA_ASI_ABI_H
#define LILA_ASI_ABI_H

#include "asi_v2.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LILA_ASI_NUMERICS_VERSION 2U

typedef struct LilaAsiModel LilaAsiModel;
typedef struct LilaAsiSession LilaAsiSession;

/* Model lifecycle */
ASI_API LilaAsiModel *lila_asi_open(const char *asi_path, char *err_buf, size_t err_cap);
ASI_API void lila_asi_close(LilaAsiModel *model);
ASI_API int lila_asi_model_info(const LilaAsiModel *model, AsiV2ModelConfig *out_cfg);
ASI_API int lila_asi_model_header(const LilaAsiModel *model, AsiV2Header *out_hdr);
ASI_API int lila_asi_get_section(const LilaAsiModel *model,
                                 uint32_t section_type,
                                 const void **out_ptr,
                                 size_t *out_size);
ASI_API int lila_asi_export_sieve_section(const LilaAsiModel *model,
                                          const char *mode_str,
                                          uint8_t *out_buf,
                                          size_t out_cap,
                                          size_t *out_written);
ASI_API int lila_asi_verify_signature(const LilaAsiModel *model,
                                      const uint8_t *trusted_pubkey_32,
                                      AsiV2SignatureInfo *out_info,
                                      char *err_buf,
                                      size_t err_cap);
ASI_API void lila_asi_cpu_info(char *buf, size_t cap);

/* Session lifecycle */
ASI_API LilaAsiSession *lila_asi_session_create(LilaAsiModel *model,
                                                 uint32_t max_ctx,
                                                 uint32_t n_threads,
                                                 char *err_buf,
                                                 size_t err_cap);
ASI_API void lila_asi_session_free(LilaAsiSession *session);
ASI_API void lila_asi_session_reset(LilaAsiSession *session);
ASI_API int lila_asi_session_save_prefix(LilaAsiSession *session, uint32_t n_tokens);
ASI_API int lila_asi_session_restore_prefix(LilaAsiSession *session, uint32_t *out_n_tokens);
ASI_API int lila_asi_session_save_prefix_file(const LilaAsiSession *session,
                                              const uint8_t key_hash[32],
                                              const char *filepath);
ASI_API int lila_asi_session_load_prefix_file(LilaAsiSession *session,
                                              const uint8_t expected_key_hash[32],
                                              const char *filepath,
                                              uint32_t *out_n_tokens);

/* Inference */
ASI_API int lila_asi_eval_token(LilaAsiSession *session,
                                int32_t token_id,
                                uint32_t pos,
                                float *out_logits);
ASI_API int lila_asi_eval_batch(LilaAsiSession *session,
                                const int32_t *token_ids,
                                uint32_t n_tokens,
                                uint32_t pos_start,
                                float *out_last_logits);

/* Multimodal 3D M-RoPE layout flags */
#define LILA_ASI_MROPE_CHUNKED     0u
#define LILA_ASI_MROPE_INTERLEAVED 1u

/* Multimodal embedding forward pass with optional 3-axis M-RoPE.
 * - embds: contiguous [n_tokens, hidden_size] FP32 embeddings.
 * - kv_pos_start: physical KV / DeltaNet slot index to start writing at.
 * - mrope_pos_3d: optional [n_tokens * 3] array of (t, h, w) coordinates.
 *   When NULL, applies scalar (kv_pos + rope_pos_delta) across all 3 axes.
 * - mrope_sections: optional 3-element axis counts (e.g., {11, 11, 10}); NULL -> model default.
 * - mrope_flags: LILA_ASI_MROPE_INTERLEAVED (Qwen3.5 default) or LILA_ASI_MROPE_CHUNKED.
 * - next_rope_pos: if non-zero, sets session rope_pos_delta = (int64_t)next_rope_pos - (int64_t)(kv_pos_start + n_tokens).
 *   If 0 and mrope_pos_3d != NULL, automatically derives next_rope_pos = max(mrope_pos_3d) + 1.
 *   If 0 and mrope_pos_3d == NULL, preserves current rope_pos_delta.
 * - cancel_flag: optional cooperative cancel pointer (returns -2 when non-zero).
 * - out_last_logits: optional [vocab_size] output buffer (NULL skips final LM-head).
 */
ASI_API int lila_asi_eval_embd(LilaAsiSession *session,
                               const float *embds,
                               uint32_t n_tokens,
                               uint32_t kv_pos_start,
                               const uint32_t *mrope_pos_3d,
                               const uint32_t mrope_sections[3],
                               uint32_t mrope_flags,
                               uint32_t next_rope_pos,
                               const volatile int *cancel_flag,
                               float *out_last_logits);

ASI_API int64_t lila_asi_session_get_rope_delta(const LilaAsiSession *session);
ASI_API void    lila_asi_session_set_rope_delta(LilaAsiSession *session, int64_t delta);

/* Helper: build [T*H_m*W_m * 3] M-RoPE table {P+t, P+r, P+c} for grid_thw={T,H_m,W_m}.
 * Returns next_rope_pos = rope_pos_start + max(T, H_m, W_m), or 0 on invalid args. */
ASI_API uint32_t lila_asi_build_mrope_grid(uint32_t rope_pos_start,
                                           const uint32_t grid_thw[3],
                                           uint32_t *out_pos_3d,
                                           size_t out_cap_u32);

/* Sampling (temperature <= 0.0f selects greedy argmax) */
ASI_API int32_t lila_asi_sample(const float *logits,
                                uint32_t vocab_size,
                                float temperature,
                                uint32_t top_k,
                                float top_p,
                                uint64_t *rng_state);

/* Optional forward activation calibration hook (NULL in normal inference).
 * role: 0=attn_in, 1=attn_or_ssm_out_in, 2=ffn_in, 3=ffn_down_in */
typedef void (*LilaAsiCalibHookFn)(void *user_ctx,
                                   uint32_t layer_idx,
                                   int role,
                                   const float *act,
                                   uint32_t n_tokens,
                                   uint32_t dim);
ASI_API void lila_asi_session_set_calib_hook(LilaAsiSession *session,
                                             LilaAsiCalibHookFn fn,
                                             void *user_ctx);

/* Offline requantization helpers (E07/v2.6) */
ASI_API int lila_asi_dequant_f32(const uint8_t *src,
                                 uint32_t quant_type,
                                 uint32_t layout_flags,
                                 int cols,
                                 int rows,
                                 float *dst_f32);
ASI_API int lila_asi_requant_q4_repack(const float *src_f32,
                                       int rows,
                                       int cols,
                                       int col_major,
                                       int search_steps,
                                       uint8_t *dst);
ASI_API int lila_asi_requant_q3_splitlut(const float *src_f32,
                                         int rows,
                                         int cols,
                                         int search_steps,
                                         uint8_t *dst);

#ifdef __cplusplus
}
#endif

#endif /* LILA_ASI_ABI_H */
