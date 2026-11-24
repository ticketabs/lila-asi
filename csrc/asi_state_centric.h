#ifndef LILA_ASI_STATE_CENTRIC_H
#define LILA_ASI_STATE_CENTRIC_H

/*
 * Exp E06: State-Centric C11 Runtime Extensions for ASI v2.4 (with Sieve)
 *
 * 1. Multi-Slot Copy-on-Write Forkable State Tree ("ASITREE1", up to 16 live slots)
 *    with Per-Branch Provenance tags (ACTIVE | COMMITTED | ABANDONED + SHA-256 trajectory).
 * 2. Crash-Recoverable Session WAL over atomic ASITREE1 checkpoints + uint64_t rng_state.
 * 3. C11 Confidence API (lila_asi_sample_ex) computing exact top1/top2/margin/logprob/entropy.
 * (The earlier margin-based Adaptive Gear-Shifter was dropped after audit.)
 */

#include "lila_asi.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * 1. Multi-Slot Forkable State Tree (ASITREE1) & Per-Branch Provenance
 * ========================================================================= */

#define ASI_STATE_TREE_MAGIC          "ASITREE1"
#define ASI_STATE_TREE_MAX_SLOTS      16U
#define ASI_STATE_TREE_KV_PAGE_TOKENS 16U
#define ASI_CP_INVALID                0U

typedef enum {
    ASI_BRANCH_ACTIVE    = 0,
    ASI_BRANCH_COMMITTED = 1,
    ASI_BRANCH_ABANDONED = 2,
} AsiBranchStatus;

/*
 * Per-Branch Provenance tag (ASI_STATE_CENTRIC_FEATURES.md Section 2.2).
 * Carried by every checkpoint slot AND the active session so abandoned or
 * rolled-back branches are explicitly tagged and can never silently leak
 * into committed state.
 */
typedef struct {
    uint32_t branch_id;
    uint32_t parent_branch_id;
    uint32_t state_version;
    uint32_t n_tokens;
    uint32_t status;                     /* AsiBranchStatus */
    uint32_t reserved;
    uint8_t  token_trajectory_sha256[32];
} AsiBranchProvenance;

_Static_assert(sizeof(AsiBranchProvenance) == 56,
               "AsiBranchProvenance must be 56 bytes");

typedef struct {
    uint32_t            cp_id;
    uint32_t            parent_cp_id;
    uint64_t            rng_state;
    AsiBranchProvenance prov;
    char                label[64];
} AsiCheckpointInfo;

/*
 * On-disk binary header for an ASITREE1 checkpoint file.
 * Protected by CRC-64 over header (excluding header_crc64) and CRC-64 over payload.
 */
typedef struct {
    char                magic[8];               /* "ASITREE1" */
    uint32_t            numerics_version;       /* LILA_ASI_NUMERICS_VERSION */
    uint32_t            arch_id;                /* AsiV2ArchId */
    uint8_t             cart_identity_hash[24];
    uint8_t             key_hash[32];
    uint32_t            fp32_ref;
    uint32_t            cp_id;
    uint32_t            parent_cp_id;
    uint32_t            kv_page_tokens;
    uint64_t            rng_state;
    AsiBranchProvenance prov;                   /* 56 bytes */
    uint32_t            n_dn;
    uint32_t            n_fa;
    uint64_t            conv_bytes;
    uint64_t            ssm_bytes;
    uint64_t            k_bytes;
    uint64_t            v_bytes;
    uint64_t            token_bytes;            /* Optional int32_t token_history payload */
    uint64_t            payload_crc64;
    uint64_t            header_crc64;           /* CRC-64 of all preceding bytes */
} AsiTreeFileHeader;

_Static_assert(sizeof(AsiTreeFileHeader) == 216,
               "AsiTreeFileHeader must be 216 bytes");
_Static_assert(offsetof(AsiTreeFileHeader, header_crc64) == 208,
               "AsiTreeFileHeader header_crc64 offset mismatch");

typedef struct {
    uint32_t live_slots;
    uint32_t max_slots;
    uint32_t physical_kv_pages;
    uint32_t logical_kv_pages;
    uint32_t shared_kv_pages;
    uint64_t cow_reuses;
    uint64_t cow_allocs;
} AsiCowStats;

/* Multi-slot checkpoint fork / restore / delete / save_file / load_file */
int lila_asi_cp_fork(LilaAsiSession *session,
                     uint32_t n_tokens,
                     uint64_t rng_state,
                     const char *label,
                     uint32_t *out_cp_id);

int lila_asi_cp_restore(LilaAsiSession *session,
                        uint32_t cp_id,
                        uint32_t *out_n_tokens,
                        uint64_t *out_rng_state);

int lila_asi_cp_delete(LilaAsiSession *session, uint32_t cp_id);

int lila_asi_cp_save_file(const LilaAsiSession *session,
                          uint32_t cp_id,
                          const uint8_t key_hash[32],
                          const char *filepath);

int lila_asi_cp_load_file(LilaAsiSession *session,
                          const uint8_t expected_key_hash[32],
                          const char *filepath,
                          const char *label,
                          uint32_t *out_cp_id);

/* Provenance inspection & explicit commit / abandon lifecycle */
int lila_asi_cp_get_info(const LilaAsiSession *session,
                         uint32_t cp_id,
                         AsiCheckpointInfo *out_info);

int lila_asi_session_get_provenance(const LilaAsiSession *session,
                                    AsiBranchProvenance *out_prov);

/*
 * Commit a checkpoint branch. Returns -1 if cp_id is ABANDONED (prevents
 * rolled-back branches from ever silently leaking into committed state).
 */
int lila_asi_cp_commit(LilaAsiSession *session, uint32_t cp_id);

/* Explicitly mark a checkpoint branch as ABANDONED. */
int lila_asi_cp_abandon(LilaAsiSession *session, uint32_t cp_id);

/* Query Copy-on-Write KV page sharing stats for the session's State Tree. */
int lila_asi_cp_cow_stats(const LilaAsiSession *session, AsiCowStats *out_stats);

/*
 * Compute a SHA-256 digest over the live session's raw DeltaNet (conv + ssm)
 * and FP16 KV cache bytes for tokens [0..n_tokens-1]. Used for exact
 * cross-branch zero-pollution verification.
 */
int lila_asi_session_state_sha256(const LilaAsiSession *session,
                                  uint32_t n_tokens,
                                  uint8_t out_sha256[32]);

/* =========================================================================
 * 2. Crash-Recoverable Session WAL (over atomic ASITREE1 checkpoints)
 * ========================================================================= */

typedef struct LilaAsiWal LilaAsiWal;

typedef struct {
    uint32_t restored_cp_id;
    uint32_t checkpoint_n_tokens;
    uint32_t replayed_tail_tokens;
    uint32_t total_n_tokens;
    uint64_t rng_state;
    uint32_t corrupt_tail_records_ignored;
} AsiWalResumeInfo;

LilaAsiWal *lila_asi_wal_open(const char *wal_dir,
                              const char *session_id,
                              const uint8_t key_hash[32]);

void lila_asi_wal_close(LilaAsiWal *wal);

/* Fork + atomically save ASITREE1 checkpoint file + durably fsync WAL record. */
int lila_asi_wal_checkpoint(LilaAsiWal *wal,
                            LilaAsiSession *session,
                            uint32_t n_tokens,
                            uint64_t rng_state,
                            const char *label,
                            uint32_t *out_cp_id);

/* Durably append tail tokens + exact post-step uint64_t rng_state to WAL. */
int lila_asi_wal_append_tokens(LilaAsiWal *wal,
                               uint32_t base_cp_id,
                               const int32_t *token_ids,
                               uint32_t n_tokens,
                               uint64_t rng_state);

/*
 * Recover from WAL into a fresh (or reset) session:
 *   1. Parse all intact CRC-verified WAL records (tolerating a torn trailing line).
 *   2. Load latest valid ASITREE1 checkpoint via lila_asi_cp_load_file + restore.
 *   3. Replay ONLY un-checkpointed tail tokens into session (writing last logits
 *      into out_last_logits if non-NULL and tail > 0).
 *   4. Return exact restored uint64_t rng_state and token counts.
 */
int lila_asi_wal_resume(LilaAsiWal *wal,
                        LilaAsiSession *session,
                        float *out_last_logits,
                        AsiWalResumeInfo *out_info);

/* =========================================================================
 * 3. C11 Confidence API (lila_asi_sample_ex)
 * ========================================================================= */

typedef struct {
    int32_t  token_id;
    float    top1_logit;
    float    top2_logit;
    float    margin;        /* top1_logit - top2_logit */
    float    top1_logprob;  /* log-softmax probability of top1 over top-64 */
    float    entropy;       /* Shannon entropy (nats) over top-64 softmax */
    uint32_t n_finite;      /* count of finite logits in vocab */
} AsiSampleStats;

/*
 * Extended sampler returning identical token_id as lila_asi_sample while
 * populating exact top1/top2/margin/top1_logprob/entropy in out_stats.
 */
int32_t lila_asi_sample_ex(const float *logits,
                           uint32_t vocab_size,
                           float temperature,
                           uint32_t top_k,
                           float top_p,
                           uint64_t *rng_state,
                           AsiSampleStats *out_stats);

/* Standalone SHA-256 helper exposed for verification tests */
void lila_asi_sha256(const void *data, size_t len, uint8_t out_hash[32]);

#ifdef __cplusplus
}
#endif

#endif /* LILA_ASI_STATE_CENTRIC_H */
