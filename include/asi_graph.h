#ifndef LILA_ASI_GRAPH_H
#define LILA_ASI_GRAPH_H

/*
 * asi_graph — Declarative Computation Graph & Tier 1 Universal Execution Engine
 *
 * Stored in optional cartridge section ASI_SEC_GRAPH (0x0F). Allows any model
 * composed of supported asi_kernels operations to execute natively in C11
 * without writing a model-specific C adapter, while StageHook callbacks verify
 * numerical equivalence against optional Tier 2 native fast paths.
 */

#include "asi_vision.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ASI_GRAPH_MAGIC          "ASIG0001"
#define ASI_GRAPH_FORMAT_VERSION 1u

typedef enum {
    ASI_OP_NOP                     = 0,
    ASI_OP_RESIZE_RGB_NORM_CHW     = 1,
    ASI_OP_PATCH_EXTRACT_MERGED    = 2,
    ASI_OP_CONV3D_PATCH_PAIR       = 3,
    ASI_OP_POS_EMBD_2D_INTERP_ADD  = 4,
    ASI_OP_ROPE_2D_VISION_TABLES   = 5,
    ASI_OP_LAYERNORM               = 6,
    ASI_OP_RMSNORM                 = 7,
    ASI_OP_LINEAR_QUANT_BIAS       = 8,
    ASI_OP_ROPE_2D_QKV_APPLY       = 9,
    ASI_OP_SDPA_BIDIR_QKV          = 10,
    ASI_OP_ADD                     = 11,
    ASI_OP_ACTIVATION              = 12,
    ASI_OP_MERGE_2X2_VIEW          = 13,
    ASI_OP_CONV2D_F32              = 14,
    ASI_OP_MAX_SUPPORTED
} AsiOpCode;

/* 64-byte header at the start of ASI_SEC_GRAPH. */
typedef struct {
    char     magic[8];          /* "ASIG0001" */
    uint32_t format_version;    /* ASI_GRAPH_FORMAT_VERSION = 1 */
    uint32_t n_slots;           /* Total activation tensor slots */
    uint32_t n_nodes;           /* Total execution nodes */
    uint32_t input_slot;        /* Reserved primary input slot */
    uint32_t output_slot;       /* Slot copied to caller output buffer */
    uint32_t flags;
    uint32_t min_kernel_abi;
    uint32_t reserved[7];
} AsiGraphHeader;

_Static_assert(sizeof(AsiGraphHeader) == 64, "AsiGraphHeader must be 64 bytes");

/* 96-byte declarative graph operation node. */
typedef struct {
    uint32_t op_code;
    uint32_t flags;
    int32_t  in_slots[4];       /* -1 = unused */
    int32_t  out_slot;          /* -1 = unused (or aux_slot for 2-output ops like RoPE tables) */
    int32_t  weight_idx[4];     /* Indices into tensor table; -1 = unused */
    int32_t  i_params[5];       /* Op-specific integer parameters */
    float    f_params[4];       /* Op-specific float parameters */
    char     stage_name[16];    /* Optional stage hook name (empty = silent) */
} AsiGraphNode;

_Static_assert(sizeof(AsiGraphNode) == 96, "AsiGraphNode must be 96 bytes");

typedef struct {
    AsiGraphHeader      header;
    const AsiGraphNode *nodes;
} AsiGraphView;

int asi_graph_op_is_supported(uint32_t op_code);

int asi_graph_validate(const uint8_t *sec_data,
                       size_t sec_size,
                       uint32_t n_tensors,
                       AsiGraphView *out_view,
                       char *err_buf,
                       size_t err_cap);

int asi_graph_execute_vision(const AsiGraphView *graph,
                             const AsiV2VisionHeader *hdr,
                             const AsiV2TensorEntry *tensors,
                             const uint8_t *weights_data,
                             const AsiImageView *image,
                             uint32_t new_h,
                             uint32_t new_w,
                             uint32_t n_threads,
                             const volatile int *cancel_flag,
                             float *out_embds,
                             size_t out_embds_floats,
                             AsiVisionStageHook stage_hook,
                             void *stage_user_data);

#ifdef __cplusplus
}
#endif

#endif /* LILA_ASI_GRAPH_H */
