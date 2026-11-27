#ifndef LILA_ASI_ARCH_H
#define LILA_ASI_ARCH_H

#include "asi_v2.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AsiArchSession AsiArchSession;

typedef struct {
    uint32_t arch_id;
    const char *name;

    /* Allocate model-specific runtime state (KV cache, DeltaNet recurrent state, scratch). */
    int (*init_session)(const AsiV2Cartridge *cart,
                        uint32_t max_ctx,
                        uint32_t n_threads,
                        AsiArchSession **out_session,
                        char *err_buf,
                        size_t err_cap);

    void (*free_session)(AsiArchSession *session);

    /* Run a single-token forward pass and write vocab_size logits into out_logits. */
    int (*forward_token)(AsiArchSession *session,
                         int32_t token_id,
                         uint32_t pos,
                         float *out_logits);

    /* Run a batched prefill forward pass for n_tokens starting at pos_start. */
    int (*forward_batch)(AsiArchSession *session,
                         const int32_t *token_ids,
                         uint32_t n_tokens,
                         uint32_t pos_start,
                         float *out_last_logits);

    /* Reset conversation / KV / SSM recurrent state for a new turn or cleared context. */
    void (*reset_state)(AsiArchSession *session);

    /* Optional in-session prefix state snapshot/restore for instant repeat-turn TTFT. */
    int (*save_prefix)(AsiArchSession *session, uint32_t n_tokens);
    int (*restore_prefix)(AsiArchSession *session, uint32_t *out_n_tokens);

    /* Optional disk-persisted prefix snapshot save/load (binary state cache). */
    int (*save_prefix_file)(const AsiArchSession *session,
                            const uint8_t key_hash[32],
                            const char *filepath);
    int (*load_prefix_file)(AsiArchSession *session,
                            const uint8_t expected_key_hash[32],
                            const char *filepath,
                            uint32_t *out_n_tokens);

    /* Optional speculative single-token forward pass (falls back to forward_token when NULL). */
    int (*forward_speculative)(AsiArchSession *session,
                               int32_t token_id,
                               uint32_t pos,
                               float *out_logits);

    /* Optional multimodal FP32 embedding forward pass with 3-axis M-RoPE. */
    int (*forward_embd)(AsiArchSession *session,
                        const float *embds,
                        uint32_t n_tokens,
                        uint32_t kv_pos_start,
                        const uint32_t *mrope_pos_3d,
                        const uint32_t mrope_sections[3],
                        uint32_t mrope_flags,
                        uint32_t next_rope_pos,
                        const volatile int *cancel_flag,
                        float *out_last_logits);
    int64_t (*get_rope_delta)(const AsiArchSession *session);
    void (*set_rope_delta)(AsiArchSession *session, int64_t delta);
    void (*set_calib_hook)(AsiArchSession *session,
                           void (*fn)(void *, uint32_t, int, const float *, uint32_t, uint32_t),
                           void *user_ctx);
} AsiArchHandler;

/* Lookup a registered architecture handler by AsiV2ArchId. Returns NULL if unsupported. */
const AsiArchHandler *asi_arch_find(uint32_t arch_id);

/* Built-in handlers */
extern const AsiArchHandler g_asi_arch_dense_transformer;
extern const AsiArchHandler g_asi_arch_hybrid_qwen35;

#ifdef __cplusplus
}
#endif

#endif /* LILA_ASI_ARCH_H */
