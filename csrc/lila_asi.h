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

/*
 * ============================================================================
 * Runtime Environment Switches Reference (all default cleanly when unset)
 * ============================================================================
 * Presets & LM-Head Screening (Sieve, E03):
 *   LILA_ASI_MODE=fast|balanced|exact  High-level preset: fast (default: 1bit1024 K=2048,
 *                                      spec ON, 50% sparse ffn_down), balanced (2bit1024 K=1024),
 *                                      exact (Sieve OFF, spec OFF, dense ffn_down).
 *   LILA_ASI_LMHEAD_MODE=off|1bit512|1bit1024|2bit1024  Override Sieve sketch mode (off = full vocab).
 *   LILA_ASI_LMHEAD_K=<int>            Top-K candidates refined by Stage 2 (default 2048/1024).
 *   LILA_ASI_LMHEAD_PREFILL=0|1        Apply Sieve on prefill logits (default 0: exact).
 *   LILA_ASI_LMHEAD_STATS=0|1          Print Sieve build/timing diagnostics to stderr (default 0).
 *   LILA_ASI_DISABLE_E05=0|1           Set 1/t/y to disable E05 <tool_call> DFA + sparse LM-head
 *                                      bypass/fast-forward for debugging or latency ablation (default 0).
 *
 * Torrent & Decode/Prefill Kernels (E07):
 *   LILA_ASI_CHUNKWY=0|1               Enable ChunkWY-32 parallel DeltaNet prefill (default 0).
 *                                      Math is exact under FP32_REF=1, but empirical Broadwell 2T
 *                                      prefill showed v2.3 AVX2 sequential DeltaNet beats ChunkWY-32
 *                                      by 23%-55% due to scratch alloc & multi-GEMM overhead; opt-in.
 *   LILA_ASI_BLIS_PREFILL=0|1          Enable L1-blocked BLIS 6x16 prefill GEMM (default 1).
 *   LILA_ASI_BLIS_MIN_TOKENS=<int>     Min batch tokens B to engage BLIS 6x16 (default 96).
 *   LILA_ASI_FFN_SPARSE_KEEP=<0..1>    Active channel keep fraction for col-major Q4 ffn_down
 *                                      (default 0.5 in fast preset, 1.0 in balanced/exact).
 *   LILA_ASI_SPECULATIVE=0|1           Enable self-speculative multi-token draft verify (default 1).
 *   LILA_ASI_SPEC_GATE=0|1             Share Sieve Stage-1 screening across draft lanes (default 1).
 *   LILA_ASI_SPEC_DRAFT_LEN=<1..3>     Draft horizon (capped at <=2 on Q4_REPACK to prevent YMM spill).
 *   LILA_ASI_SPEC_STATS=0|1            Print speculative acceptance stats to stderr (default 0).
 *   LILA_ASI_Q6K_DOT2=0|1 / Q4K_DOT2   Enable paired 2-row Q8K SIMD matvec kernels (default 1).
 *   LILA_ASI_MATVEC_CHUNK=<int>        Row work chunk size for pool workers (default 64).
 *   LILA_ASI_LMHEAD_CHUNK=<int>        Row chunk size for lm_head workers (default 512).
 *   LILA_ASI_FUSED_CHUNK=<int>         Row chunk size for fused projections (default 32).
 *   LILA_ASI_BATCH_CHUNK_MAX=<int>     Max tokens per prefill batch chunk (default 512).
 *
 * E08 Scale, Streaming & Multi-Arch Switches:
 *   LILA_ASI_STREAM_EVICT=0|1          Enable per-layer MADV_WILLNEED lookahead prefetch + strict
 *                                      interior MADV_DONTNEED page eviction for bounded-RAM streaming
 *                                      of large cartridges (default 0: keep resident pages cached).
 *   LILA_ASI_ROPE_SCALE_MODE=standard|linear|ntk|yarn|llama3
 *                                      Unified dense-transformer RoPE context-extension scaling
 *                                      (default standard).
 *   LILA_ASI_ROPE_SCALE_FACTOR=<float> RoPE context extension factor >= 1.0 (default 1.0).
 *   LILA_ASI_ROPE_ORIG_CTX=<int>       Original training context for YaRN/Llama-3.1 ramps.
 *
 * Correctness, Security, Vision, Threading & Telemetry:
 *   LILA_ASI_FP32_REF=0|1              Force slow unquantized FP32 reference matmuls for golden parity.
 *   LILA_ASI_VERIFY_CRC=0|1            Force eager startup CRC-64 check over bulk mmap weight blobs.
 *   LILA_ASI_NO_HUGEPAGE=0|1           Disable MADV_HUGEPAGE (THP) hint on weight mmap (default 0).
 *   LILA_ASI_REQUIRE_SIGNATURE=0|1     Fail closed in open() unless valid Ed25519 signature present.
 *   LILA_ASI_TRUSTED_PUBKEY=<64hex>    Expected Ed25519 publisher public key hex string.
 *   LILA_ASI_VISION_EXEC=auto|fast|graph  Select ViT fast kernel vs ASIG0001 universal graph DAG.
 *   LILA_ASI_VISION_FAST_Q8=0|1        Dynamic Q8xQ8 ViT linear path (default 0; fails 1e-5 tol check).
 *   LILA_ASI_SPIN_LIMIT=<int>          Thread-pool pause spin iterations before sleep (default 2048/64).
 *   LILA_ASI_PROFILE=0|1               Print per-op decode/prefill timing breakdown to stderr.
 *   LILA_ASI_DEBUG_TOKENS=0|1          Print CLI tokenizer load diagnostics.
 * Python SDK / Server Switches (src/core/{asi_backend,lilacore}.py):
 *   LILA_ASI_PATH=<path>, LILA_ASI_LIB=<path>, LILA_ASI_ARCH=avx2|scalar,
 *   LILA_ASI_DISABLE_DISK_PREFIX=0|1, LILA_ASI_PREFIX_CACHE_DIR=<dir>,
 *   LILA_ASI_PREFIX_CACHE_MAX_FILES=<int>, LILA_ASI_REPEAT_PENALTY=<float>
 * ============================================================================
 */
#ifdef __cplusplus
}
#endif

#endif /* LILA_ASI_ABI_H */
