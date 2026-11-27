#ifndef LILA_ASI_PAIRING_H
#define LILA_ASI_PAIRING_H

/*
 * asi_pairing — Neural Pairing Registry Contract & Bridge Executor (ASI v2.5 V8)
 *
 * Stored in dedicated cartridge section ASI_SEC_PAIRING (0x10). Keeps original
 * base weights (ASI_SEC_WEIGHTS / ASI_SEC_VISION) 100% immutable while allowing
 * validated multimodal translator bridges to be sealed inside one portable
 * cartridge and executed natively in C11.
 */

#include "asi_v2.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ASI_PAIRING_MAGIC          "ASIP0001"
#define ASI_PAIRING_FORMAT_VERSION 1u

typedef enum {
    ASI_PAIR_STATE_DISCOVERED             = 0,
    ASI_PAIR_STATE_COMPAT_CHECKED         = 1,
    ASI_PAIR_STATE_LEARNING_REQUIRED      = 2,
    ASI_PAIR_STATE_TRAINING               = 3,
    ASI_PAIR_STATE_VALIDATION             = 4,
    ASI_PAIR_STATE_APPROVED               = 5,
    ASI_PAIR_STATE_CONTINUOUS_IMPROVEMENT = 6,
    ASI_PAIR_STATE_REJECTED               = 7,
} AsiPairingState;

typedef enum {
    ASI_BRIDGE_LINEAR     = 1,
    ASI_BRIDGE_MLP_2LAYER = 2,
} AsiBridgeKind;

/* 192-byte self-describing binary header at the start of ASI_SEC_PAIRING (0x10). */
typedef struct {
    char     magic[8];                     /* "ASIP0001" */
    uint32_t format_version;               /* ASI_PAIRING_FORMAT_VERSION = 1 */
    uint32_t state;                        /* AsiPairingState */
    uint32_t bridge_kind;                  /* AsiBridgeKind */
    uint32_t act_type;                     /* AsiActivationType (for hidden/output activation) */
    uint32_t in_dim;                       /* Vision encoder projection_dim */
    uint32_t hidden_dim;                   /* MLP hidden width (0 or == out_dim for linear) */
    uint32_t out_dim;                      /* Target LLM hidden_size */
    uint32_t target_llm_arch_id;           /* Target AsiV2ArchId (0 = any matching dim) */
    uint32_t target_mrope_sections[4];
    uint32_t val_passed;                   /* Must be 1 for execution */
    uint32_t flags;
    float    val_cosine;
    float    val_loss;
    uint32_t train_steps;
    uint32_t n_tensors;                    /* 2 for LINEAR, 4 for MLP_2LAYER */
    uint64_t weights_offset;               /* Relative to start of ASI_SEC_PAIRING */
    uint8_t  source_vision_sha256[32];     /* Must match AsiV2VisionHeader.source_fingerprint */
    uint8_t  target_llm_sha256[32];        /* Optional target checkpoint binding (all 0 = unbound) */
    uint32_t reserved[10];
} AsiPairingHeader;

_Static_assert(sizeof(AsiPairingHeader) == 192, "AsiPairingHeader must be 192 bytes");

typedef struct {
    AsiPairingHeader        header;
    const AsiV2TensorEntry *tensors;
    const uint8_t          *weights_data;
    size_t                  weights_data_size;
} AsiPairingView;

int asi_pairing_is_executable(const AsiPairingHeader *hdr);

int asi_pairing_validate(const uint8_t *sec_data,
                         size_t sec_size,
                         const uint8_t expected_vision_sha256[32],
                         AsiPairingView *out_view,
                         char *err_buf,
                         size_t err_cap);

int asi_pairing_check_llm_compat(const AsiPairingView *pv,
                                 const AsiV2ModelConfig *llm_cfg,
                                 const uint8_t *target_llm_sha256_opt,
                                 char *err_buf,
                                 size_t err_cap);

int asi_pairing_apply(const AsiPairingView *pv,
                      const float *in_embds,
                      uint32_t n_tokens,
                      uint32_t n_threads,
                      const volatile int *cancel_flag,
                      float *out_embds,
                      size_t out_floats);

#ifdef __cplusplus
}
#endif

#endif /* LILA_ASI_PAIRING_H */
