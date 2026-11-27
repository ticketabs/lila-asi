#ifndef LILA_LM_HEAD_CANDIDATE_H
#define LILA_LM_HEAD_CANDIDATE_H

#include <stddef.h>
#include <stdint.h>
#include "q4k.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Exp E03: Multi-threaded zero-allocation two-stage full-vocabulary lm_head
 * screening for Qwen3.5 single-token decode (and optional spec/prefill).
 *
 * Preserves ALL vocabulary tokens across every language (no static token
 * range skipping). Packs all V rows into 64-row nibble tiles for cache-local
 * AVX2 vpshufb 4-bit LUT coarse scoring with 64-dim block activation scaling
 * (16 blocks across H = 1024), minimizing outlier quantization loss.
 *
 * Supported sketch modes (LILA_ASI_LMHEAD_MODE):
 *   - LILA_LHC_MODE_OFF       ("off")      : Exact full Q6_K baseline
 *   - LILA_LHC_MODE_1BIT_512  ("1bit512")  : 512-dim 2-elem pooled 1-bit sign (~16.4 MB @ 248k)
 *   - LILA_LHC_MODE_1BIT_1024 ("1bit1024") : 1024-dim full 1-bit sign (~32.3 MB @ 248k)
 *   - LILA_LHC_MODE_2BIT_1024 ("2bit1024") : 1024-dim full 2-bit Lloyd-Max (~64.1 MB @ 248k)
 */

typedef enum {
    LILA_LHC_MODE_OFF       = 0,
    LILA_LHC_MODE_1BIT_512  = 1,
    LILA_LHC_MODE_1BIT_1024 = 2,
    LILA_LHC_MODE_2BIT_1024 = 3,
} LilaLhcMode;

typedef struct LilaLhcIndex LilaLhcIndex;
typedef struct LilaLhcWorkspace LilaLhcWorkspace;

/* Matches AsiTaskFn signature in asi_pool.h without requiring asi_pool.h inclusion. */
typedef void (*LilaLhcTaskFn)(int tid, int n_threads, void *ctx);
typedef void (*LilaLhcDispatchFn)(void *pool_opaque, LilaLhcTaskFn fn, void *ctx);

typedef struct {
    LilaLhcMode mode;
    uint32_t vocab_size;
    uint32_t hidden_size;
    uint32_t n_tiles;
    size_t sketch_bytes_per_row;
    size_t total_sketch_bytes;
    size_t exact_q6k_bytes;
    double build_ms;
    size_t extra_rss_bytes;
} LilaLhcInfo;

typedef struct {
    uint64_t eval_calls;
    uint64_t spec_eval_calls;
    uint64_t total_candidates_eval;
    double stage1_ms_sum;
    double stage2_ms_sum;
} LilaLhcStats;

typedef struct {
    uint32_t top_k;
    int32_t exact_top1_id;
    float exact_top1_logit;
    int32_t filtered_top1_id;
    float filtered_top1_logit;
    uint32_t true_top1_coarse_rank;
    int top1_match;
    uint32_t top10_recall_count;
    uint32_t top64_recall_count;
    double kl_divergence;
    double kl_divergence_t07;
    size_t coarse_bytes_read;
    size_t fine_bytes_read;
    size_t total_bytes_read;
    double bandwidth_reduction_ratio;
} LilaLhcEvalMetrics;

uint32_t lila_lhc_abi_version(void);
LilaLhcMode lila_lhc_parse_mode(const char *mode_str);
const char *lila_lhc_mode_name(LilaLhcMode mode);

/* Build coarse screening index from any supported quant rows [vocab_size, hidden_size].
 * Uses dispatch_fn(pool_opaque, ...) if provided to build tiles in parallel.
 * Optional output_norm [hidden_size] weights the L2 row scale fit by activation variance. */
LilaLhcIndex *lila_lhc_build_quant_pool(LilaLhcMode mode,
                                        const uint8_t *weights,
                                        uint32_t quant_type,
                                        int is_repacked,
                                        uint32_t vocab_size,
                                        uint32_t hidden_size,
                                        const float *output_norm,
                                        LilaLhcDispatchFn dispatch_fn,
                                        void *pool_opaque);

LilaLhcIndex *lila_lhc_build_quant(LilaLhcMode mode,
                                   const uint8_t *weights,
                                   uint32_t quant_type,
                                   int is_repacked,
                                   uint32_t vocab_size,
                                   uint32_t hidden_size,
                                   const float *output_norm);

LilaLhcIndex *lila_lhc_build_q6k_pool(LilaLhcMode mode,
                                      const uint8_t *q6k_weights,
                                      uint32_t vocab_size,
                                      uint32_t hidden_size,
                                      const float *output_norm,
                                      LilaLhcDispatchFn dispatch_fn,
                                      void *pool_opaque);

LilaLhcIndex *lila_lhc_build_q6k(LilaLhcMode mode,
                                 const uint8_t *q6k_weights,
                                 uint32_t vocab_size,
                                 uint32_t hidden_size,
                                 const float *output_norm);

/* Zero-copy load of pre-serialized ASI_SEC_SIEVE_SKETCH (0x11) section.
 * Validates AsiSieveSketchHeader, alignment, and exact bounds; pointers alias sec_data directly.
 * Returns NULL cleanly on mismatch or corruption so caller can fall back to on-the-fly build. */
LilaLhcIndex *lila_lhc_from_serialized(LilaLhcMode expected_mode,
                                       const uint8_t *sec_data,
                                       size_t sec_size,
                                       uint32_t quant_type,
                                       int is_repacked,
                                       uint32_t vocab_size,
                                       uint32_t hidden_size);

/* Serialize a built LilaLhcIndex into ASI_SEC_SIEVE_SKETCH binary section layout.
 * Pass out_buf == NULL to query required byte count in *out_written. */
int lila_lhc_serialize_section(const LilaLhcIndex *index,
                                 uint8_t *out_buf,
                                 size_t out_cap,
                                 size_t *out_written);

uint32_t lila_lhc_quant_type(const LilaLhcIndex *index);
int lila_lhc_is_repacked(const LilaLhcIndex *index);

void lila_lhc_free(LilaLhcIndex *index);
int lila_lhc_get_info(const LilaLhcIndex *index, LilaLhcInfo *out_info);

/* Pre-allocate per-thread top-K heaps, candidate buffers, and LUT scratch.
 * After creation, decode calls perform ZERO malloc/free/calloc/realloc. */
LilaLhcWorkspace *lila_lhc_workspace_create(uint32_t vocab_size,
                                            uint32_t top_k,
                                            uint32_t n_threads,
                                            uint32_t max_spec_b);

void lila_lhc_workspace_free(LilaLhcWorkspace *ws);
size_t lila_lhc_workspace_bytes(const LilaLhcWorkspace *ws);
int lila_lhc_get_stats(const LilaLhcWorkspace *ws, LilaLhcStats *out_stats);

/* Score all vocab_size tokens into out_coarse_scores (for evaluation/tests). */
int lila_lhc_score_coarse_pool(const LilaLhcIndex *index,
                               const float *x_norm,
                               float *out_coarse_scores,
                               LilaLhcDispatchFn dispatch_fn,
                               void *pool_opaque);

int lila_lhc_score_coarse(const LilaLhcIndex *index,
                          const float *x_norm,
                          float *out_coarse_scores);

/* Select top_k token IDs from coarse_scores [vocab_size], returned sorted ascending. */
int lila_lhc_select_topk(const float *coarse_scores,
                         uint32_t vocab_size,
                         uint32_t top_k,
                         int32_t *out_candidate_ids);

/* Zero-allocation single-token two-stage forward pass:
 *   1. Stage-1 parallel AVX2 LUT scoring + per-thread top-K min-heaps (no V-float buffer!).
 *   2. Deterministic heap merge in O(n_threads * K log K) + ascending ID sort.
 *   3. Fill out_logits with -1e30f and run Stage-2 exact Q6_K (or FP32 ref) on top-K rows. */
int lila_lhc_forward_pool(const LilaLhcIndex *index,
                          LilaLhcWorkspace *ws,
                          LilaLhcDispatchFn dispatch_fn,
                          void *pool_opaque,
                          const uint8_t *q6k_weights,
                          const float *x_norm,
                          float *out_logits,
                          int fp32_ref);

/* Zero-allocation multi-lane (1 <= B <= max_spec_b) speculative verification forward pass:
 * Reads the coarse sketch index ONCE across all B vectors in Stage 1, merges unique
 * candidate rows across lanes, and evaluates exact Q6_K (or FP32 ref) once per unique row. */
int lila_lhc_forward_spec_pool(const LilaLhcIndex *index,
                               LilaLhcWorkspace *ws,
                               LilaLhcDispatchFn dispatch_fn,
                               void *pool_opaque,
                               const uint8_t *q6k_weights,
                               const float *xnb,
                               const LilaQ8KBlock *qx_all,
                               uint32_t B,
                               float *out0,
                               float *out_rest,
                               int fp32_ref);

/* Evaluate recall, top-1 agreement, coarse rank of true winner, KL divergence,
 * and DRAM byte traffic against precomputed exact_logits [vocab_size]. */
int lila_lhc_evaluate_metrics(const LilaLhcIndex *index,
                              const float *exact_logits,
                              const float *coarse_scores,
                              uint32_t top_k,
                              LilaLhcEvalMetrics *out_metrics);

#ifdef __cplusplus
}
#endif

#endif /* LILA_LM_HEAD_CANDIDATE_H */
