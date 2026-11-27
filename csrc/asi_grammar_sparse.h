#ifndef ASI_GRAMMAR_SPARSE_H
#define ASI_GRAMMAR_SPARSE_H

/*
 * Exp E05: Grammar-Aware Sparse lm_head + C11 <tool_call> Format Guardrails
 *
 * Pure C11, zero external runtime dependencies.
 * Built on top of ASI v2.4 with Sieve (lila_lm_head_candidate.h / q4k.h).
 *
 * Three-case decode dispatch:
 *   Case 1 (n_legal == 1 or deterministic forced literal prefix):
 *       Forced-Token Fast-Forward -> 0 lm_head computation!
 *       asi_grammar_collect_forced_run() collects consecutive forced tokens
 *       in one shot for batched prefill via lila_asi_eval_batch().
 *   Case 2 (1 < n_legal <= sparse_limit, e.g. <= 2048):
 *       Grammar-Aware Sparse lm_head -> skips Sieve Stage-1 completely,
 *       evaluates exact Q6_K ONLY on sorted ascending legal_ids[0..n_legal-1],
 *       fills all illegal tokens with -1e30f (100% exact math, <0.1 ms).
 *   Case 3 (n_legal > sparse_limit or unconstrained):
 *       Seamless fallback to Sieve (1bit1024 K=2048) or exact full Q6_K,
 *       masking any illegal tokens to -1e30f.
 */

#include <stddef.h>
#include <stdint.h>
#include "asi_platform.h"
#include "lila_lm_head_candidate.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ASI_GRAMMAR_MAX_TOOLS        512u
#define ASI_GRAMMAR_MAX_ARGS         16u
#define ASI_GRAMMAR_MAX_ENUM_VALS    16u
#define ASI_GRAMMAR_NEG_INF_FILL     (-1e30f)
#define ASI_GRAMMAR_DEFAULT_SPARSE_LIMIT 2048u

typedef enum {
    ASI_ARG_STRING = 0,
    ASI_ARG_INT    = 1,
    ASI_ARG_FLOAT  = 2,
    ASI_ARG_BOOL   = 3,
    ASI_ARG_ENUM   = 4,
} AsiArgType;

typedef struct {
    const char *name;
    AsiArgType type;
    uint32_t max_len;                                /* for ASI_ARG_STRING (0 -> unbounded RFC-8259 JSON string with escapes & UTF-8) */
    uint32_t max_digits;                             /* for ASI_ARG_INT/FLOAT (0 -> default 6) */
    const char *enum_values[ASI_GRAMMAR_MAX_ENUM_VALS]; /* for ASI_ARG_ENUM (unquoted strings) */
    uint32_t n_enum_values;
    int optional;                                    /* 0 -> required, 1 -> optional (may be omitted in JSON) */
} AsiToolArgSchema;

typedef struct {
    const char *name;
    AsiToolArgSchema args[ASI_GRAMMAR_MAX_ARGS];
    uint32_t n_args;
} AsiToolSchema;

typedef struct {
    int wrap_xml;               /* 1: <tool_call>\n{...}\n</tool_call>, 0: raw {...} */
    uint32_t default_max_str;   /* 0 -> unbounded looping RFC-8259 string; >0 -> bounded max_len */
    uint32_t default_max_digits;/* default 6 if 0 */
    uint32_t sparse_limit;      /* Case 2 threshold, default 2048 if 0 */
} AsiGrammarConfig;

typedef enum {
    ASI_FORCED_NONE         = 0,
    ASI_FORCED_COMPLETE     = 1,
    ASI_FORCED_PREFIX       = 2,
    ASI_FORCED_SINGLE_TOKEN = 3,
} AsiForcedReason;

typedef enum {
    ASI_DISPATCH_FORCED_SKIP = 1,
    ASI_DISPATCH_SPARSE_Q6K  = 2,
    ASI_DISPATCH_SIEVE_MASK  = 3,
    ASI_DISPATCH_FULL_MASK   = 4,
} AsiGrammarDispatchMode;

typedef struct {
    uint64_t total_tokens;
    uint64_t fast_forwarded_tokens;
    uint64_t forced_prefix_tokens;
    uint64_t forced_single_tokens;
    uint64_t forced_runs;
    uint64_t lm_head_calls;
    uint64_t sparse_lm_head_calls;
    uint64_t sieve_lm_head_calls;
    uint64_t full_lm_head_calls;
    uint64_t sparse_rows_eval_sum;
    double   lm_head_ms_sum;
    double   mask_ms_sum;
} AsiGrammarTelemetry;

/* Decoded byte-string vocabulary table */
typedef struct {
    uint32_t vocab_size;
    char   **tokens;      /* owned decoded byte strings (NULL if disabled/unused token) */
    uint32_t *token_lens; /* byte length of each token */
    int32_t  eos_token_id;
    int32_t  im_start_id;
    int32_t  im_end_id;
    int32_t  tool_call_open_id;
    int32_t  tool_call_close_id;
} AsiGrammarVocab;

typedef struct AsiGrammarEngine AsiGrammarEngine;

/* Optional Sieve callback matching lila_lhc_forward_pool semantics */
typedef int (*AsiSieveForwardFn)(void *sieve_ctx,
                                 const uint8_t *q6k_weights,
                                 const float *x_norm,
                                 float *out_logits,
                                 int fp32_ref);

/* ---- Vocabulary helpers ---- */

/* Build a vocabulary from caller-provided byte strings (copies all strings). */
ASI_API int asi_grammar_vocab_from_arrays(AsiGrammarVocab *out_vocab,
                                          uint32_t vocab_size,
                                          const char *const *tokens,
                                          const uint32_t *token_lens);

/* Parse ASI_SEC_TOKENIZER JSON directly from an ASI v2 cartridge in pure C11.
 * Decodes GPT-2 byte-level BPE (token_type == 1) and preserves literal
 * user-defined special tokens (token_type == 4, e.g. <tool_call>, </tool_call>). */
ASI_API int asi_grammar_vocab_from_asi_json(AsiGrammarVocab *out_vocab,
                                            const void *sec_ptr,
                                            size_t sec_size);

ASI_API void asi_grammar_vocab_free(AsiGrammarVocab *vocab);

/* Longest-match greedy tokenization of UTF-8/ASCII text using AsiGrammarVocab.
 * Returns number of tokens written, or -1 on error. */
ASI_API int asi_grammar_vocab_greedy_encode(const AsiGrammarVocab *vocab,
                                            const char *text,
                                            size_t text_len,
                                            int32_t *out_ids,
                                            uint32_t max_ids);

/* ---- Grammar Engine Lifecycle ---- */

ASI_API AsiGrammarEngine *asi_grammar_create(const AsiGrammarVocab *vocab,
                                             const AsiToolSchema *tools,
                                             uint32_t n_tools,
                                             const AsiGrammarConfig *cfg);

ASI_API void asi_grammar_free(AsiGrammarEngine *g);
ASI_API void asi_grammar_reset(AsiGrammarEngine *g);
ASI_API int  asi_grammar_is_complete(const AsiGrammarEngine *g);
ASI_API int32_t asi_grammar_state(const AsiGrammarEngine *g);
ASI_API uint32_t asi_grammar_num_states(const AsiGrammarEngine *g);
ASI_API uint32_t asi_grammar_sparse_limit(const AsiGrammarEngine *g);
ASI_API void asi_grammar_get_telemetry(const AsiGrammarEngine *g, AsiGrammarTelemetry *out);

/* ---- Incremental Masking & Forced-Token Fast-Forward ---- */

/* Compute (or fetch cached) sorted ascending legal token IDs for the current DFA state.
 * Returned pointer is owned by AsiGrammarEngine and valid until asi_grammar_free(). */
ASI_API int asi_grammar_get_legal_ids(AsiGrammarEngine *g,
                                      const int32_t **out_legal_ids,
                                      uint32_t *out_n_legal);

/* Case 1: Collect consecutive forced tokens (deterministic literal prefixes AND
 * single-legal-token steps) in one shot so the caller can feed them via
 * lila_asi_eval_batch() with 0 lm_head computation!
 * Advances the grammar state across all collected forced tokens and updates telemetry.
 * Returns number of forced tokens collected (0 if at a branching choice point or complete),
 * or -1 on error. */
ASI_API int asi_grammar_collect_forced_run(AsiGrammarEngine *g,
                                           int32_t *out_token_ids,
                                           uint32_t max_tokens,
                                           AsiForcedReason *out_reason);

/* Advance grammar state by a sampled token ID (records lm_head_calls++). */
ASI_API int asi_grammar_accept_token(AsiGrammarEngine *g, int32_t token_id);

/* Mask an existing full-vocab logits buffer in-place: illegal tokens -> -1e30f. */
ASI_API int asi_grammar_mask_logits(AsiGrammarEngine *g, float *logits, uint32_t vocab_size);

/* ---- Case 2 & Case 3: Grammar-Aware Sparse lm_head ---- */

ASI_API int asi_grammar_sparse_lm_head_quant(const uint8_t *weights,
                                             uint32_t quant_type,
                                             int is_repacked,
                                             uint32_t vocab_size,
                                             uint32_t hidden_size,
                                             const float *x_norm,
                                             const int32_t *legal_ids,
                                             uint32_t n_legal,
                                             float *out_logits,
                                             int fp32_ref,
                                             LilaLhcDispatchFn dispatch_fn,
                                             void *pool_opaque);

ASI_API int asi_grammar_full_lm_head_quant(const uint8_t *weights,
                                           uint32_t quant_type,
                                           int is_repacked,
                                           uint32_t vocab_size,
                                           uint32_t hidden_size,
                                           const float *x_norm,
                                           float *out_logits,
                                           int fp32_ref,
                                           LilaLhcDispatchFn dispatch_fn,
                                           void *pool_opaque);

ASI_API int asi_grammar_lm_head_forward_quant(AsiGrammarEngine *g,
                                              const uint8_t *weights,
                                              uint32_t quant_type,
                                              int is_repacked,
                                              uint32_t vocab_size,
                                              uint32_t hidden_size,
                                              const float *x_norm,
                                              float *out_logits,
                                              int fp32_ref,
                                              LilaLhcDispatchFn dispatch_fn,
                                              void *pool_opaque,
                                              AsiSieveForwardFn sieve_fn,
                                              void *sieve_ctx,
                                              AsiGrammarDispatchMode *out_mode);

/* Case 2 core kernel:
 * Fills out_logits[0..vocab_size-1] with -1e30f, quantizes x_norm once to Q8_K
 * (unless fp32_ref != 0), and evaluates exact Q6_K ONLY on legal_ids[0..n_legal-1]
 * (must be sorted ascending for optimal DRAM prefetching). */
ASI_API int asi_grammar_sparse_lm_head_q6k(const uint8_t *q6k_weights,
                                           uint32_t vocab_size,
                                           uint32_t hidden_size,
                                           const float *x_norm,
                                           const int32_t *legal_ids,
                                           uint32_t n_legal,
                                           float *out_logits,
                                           int fp32_ref,
                                           LilaLhcDispatchFn dispatch_fn,
                                           void *pool_opaque);

/* Exact full-vocab Q6_K reference matvec (for parity testing and Case 3 fallback). */
ASI_API int asi_grammar_full_lm_head_q6k(const uint8_t *q6k_weights,
                                         uint32_t vocab_size,
                                         uint32_t hidden_size,
                                         const float *x_norm,
                                         float *out_logits,
                                         int fp32_ref,
                                         LilaLhcDispatchFn dispatch_fn,
                                         void *pool_opaque);

/* Unified Grammar-Aware lm_head Forward:
 *   - If g == NULL: runs Sieve (if sieve_fn != NULL) else full Q6_K.
 *   - If 1 <= n_legal <= sparse_limit: Case 2 -> Grammar-Aware Sparse Q6_K (skips Sieve!).
 *   - If n_legal > sparse_limit: Case 3 -> Sieve (or full Q6_K) + grammar mask (-1e30f),
 *     with automatic sparse fallback if Sieve's top-K has 0 legal tokens. */
ASI_API int asi_grammar_lm_head_forward_q6k(AsiGrammarEngine *g,
                                            const uint8_t *q6k_weights,
                                            uint32_t vocab_size,
                                            uint32_t hidden_size,
                                            const float *x_norm,
                                            float *out_logits,
                                            int fp32_ref,
                                            LilaLhcDispatchFn dispatch_fn,
                                            void *pool_opaque,
                                            AsiSieveForwardFn sieve_fn,
                                            void *sieve_ctx,
                                            AsiGrammarDispatchMode *out_mode);

/* ---- Built-in C11 Schema-Compliant JSON Validator (for tests & benchmarks) ---- */

/* Validates that `text` (optionally wrapped in <tool_call>\n...\n</tool_call>)
 * is strictly valid JSON matching one of tools[0..n_tools-1] with exact tool name,
 * exact required arguments, and valid typed values. Returns 1 if valid, 0 if invalid. */
ASI_API int asi_grammar_validate_tool_call(const char *text,
                                           const AsiToolSchema *tools,
                                           uint32_t n_tools,
                                           int wrap_xml,
                                           char *err_buf,
                                           size_t err_cap);

#ifdef __cplusplus
}
#endif

#endif /* ASI_GRAMMAR_SPARSE_H */
