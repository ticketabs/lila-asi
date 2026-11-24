#ifndef LILA_ASI_SC_INTERNAL_H
#define LILA_ASI_SC_INTERNAL_H

#include "asi_arch.h"
#include "asi_state_centric.h"
#include <stddef.h>
#include <stdint.h>

typedef struct AsiStateTree AsiStateTree;

typedef struct {
    uint32_t max_ctx;
    uint32_t n_dn;
    uint32_t n_fa;
    uint32_t fp32_ref;
    size_t   conv_per_layer;   /* float elements per DeltaNet layer */
    size_t   ssm_per_layer;    /* float elements per DeltaNet layer */
    size_t   kv_dim;           /* uint16_t elements per token per FA layer */
    uint64_t conv_bytes;       /* total bytes across all n_dn layers */
    uint64_t ssm_bytes;        /* total bytes across all n_dn layers */
    uint8_t  cart_identity_hash[24];
    uint32_t arch_id;
} Qwen35StateGeom;

/* Bridges implemented in lila_asi_sc.c */
AsiArchSession *lila_asi_sc_get_arch_session(const LilaAsiSession *session);
const AsiV2Cartridge *lila_asi_sc_get_cartridge(const LilaAsiSession *session);
AsiStateTree *lila_asi_sc_ensure_state_tree(LilaAsiSession *session);
const AsiStateTree *lila_asi_sc_get_state_tree(const LilaAsiSession *session);
AsiStateTree *lila_asi_sc_state_tree_create(LilaAsiSession *session);
void lila_asi_sc_state_tree_free(AsiStateTree *tree);

/* Bridges implemented in asi_arch_sc.c */
int qwen35_sc_get_geometry(const AsiArchSession *s, Qwen35StateGeom *out_geom);
int qwen35_sc_capture_deltanet(const AsiArchSession *s,
                               float *out_conv,
                               float *out_ssm);
int qwen35_sc_capture_kv_page(const AsiArchSession *s,
                              uint32_t fa_layer_idx,
                              uint32_t page_idx,
                              uint32_t page_tokens,
                              uint32_t take_tokens,
                              uint16_t *out_k,
                              uint16_t *out_v);
int qwen35_sc_restore_state(AsiArchSession *s,
                            uint32_t n_tokens,
                            const float *conv,
                            const float *ssm,
                            const uint16_t *const *k_pages,
                            const uint16_t *const *v_pages,
                            uint32_t n_pages_per_layer,
                            uint32_t page_tokens,
                            const int32_t *tokens,
                            int has_tokens);
int qwen35_sc_get_history(const AsiArchSession *s,
                          const int32_t **out_tokens,
                          uint32_t *out_len,
                          uint32_t *out_valid_from);
uint32_t qwen35_sc_get_kv_dirty_from(const AsiArchSession *s);
void qwen35_sc_mark_kv_clean(AsiArchSession *s, uint32_t pos);

#endif /* LILA_ASI_SC_INTERNAL_H */
