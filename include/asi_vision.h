#ifndef LILA_ASI_VISION_H
#define LILA_ASI_VISION_H

/*
 * ASI Universal Vision Runtime & Adapter Contract (v2.5)
 *
 * Self-describing `.asiv` sidecar format + architecture-independent
 * C11 vision adapter interface (`g_asi_vision_qwen35_vit` first).
 */

#include "asi_v2.h"
#include "asi_pairing.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ASIV_MAGIC          "ASIV0001"
#define ASIV_FORMAT_VERSION 1u

typedef enum {
    ASI_VISION_ADAPTER_UNKNOWN      = 0,
    ASI_VISION_ADAPTER_QWEN3VL_VIT  = 1,
    ASI_VISION_ADAPTER_DINOV3_DENSE = 2,
    ASI_VISION_ADAPTER_SAM_SEG      = 3,
    ASI_VISION_ADAPTER_DOC_OCR      = 4,
} AsiVisionAdapterId;

typedef enum {
    ASI_VISION_MODE_MULTIMODAL_EMBD        = 1,
    ASI_VISION_MODE_INDEPENDENT_PERCEPTION = 2,
} AsiVisionMode;

#define ASI_VISION_OUT_EMBD         (1u << 0)
#define ASI_VISION_OUT_FEATURE_MAP  (1u << 1)
#define ASI_VISION_OUT_BOXES        (1u << 2)
#define ASI_VISION_OUT_MASKS        (1u << 3)
#define ASI_VISION_OUT_TEXT_REGIONS (1u << 4)
#define ASI_VISION_OUT_KEYPOINTS    (1u << 5)
#define ASI_VISION_OUT_DEPTH        (1u << 6)

#define ASI_VISION_FLAG_USE_GELU_TANH   (1u << 0)
#define ASI_VISION_FLAG_HAS_DEEPSTACK     (1u << 1)
#define ASI_VISION_FLAG_TEMPORAL_CONV3D   (1u << 2)

/* 192-byte self-describing binary header inside ASI_SEC_VISION (0x0A). */
typedef struct {
    char     magic[8];                  /* "ASIV0001" */
    uint32_t format_version;            /* ASIV_FORMAT_VERSION = 1 */
    uint32_t adapter_id;                /* AsiVisionAdapterId */
    uint32_t mode;                      /* AsiVisionMode */
    uint32_t output_kinds;              /* ASI_VISION_OUT_* bitmask */
    uint32_t expected_llm_arch_id;      /* Compatible LLM AsiV2ArchId (0 = any) */
    uint32_t expected_llm_hidden_size;  /* Compatible LLM hidden_size (0 = independent) */
    uint32_t patch_size;
    uint32_t temporal_patch_size;
    uint32_t spatial_merge_size;
    uint32_t image_size;
    uint32_t hidden_size;
    uint32_t intermediate_size;
    uint32_t n_heads;
    uint32_t n_layers;
    uint32_t projection_dim;
    float    ln_eps;
    float    rope_theta;
    float    image_mean[3];
    float    image_std[3];
    uint32_t mrope_sections[4];
    uint32_t vision_start_id;
    uint32_t vision_end_id;
    uint32_t image_pad_id;
    uint32_t video_pad_id;
    uint32_t default_max_pixels;
    uint32_t flags;                     /* ASI_VISION_FLAG_* */
    uint32_t n_tensor_shapes;           /* Must equal AsiV2WeightsHeader.n_tensors */
    uint32_t reserved[4];
    uint8_t  source_fingerprint[32];    /* SHA-256 of source weights / checkpoint */
} AsiV2VisionHeader;

_Static_assert(sizeof(AsiV2VisionHeader) == 192, "AsiV2VisionHeader must be 192 bytes");

/* 40-byte full 4D tensor shape record paired 1:1 with AsiV2TensorEntry. */
typedef struct {
    uint32_t ndim;
    uint32_t reserved;
    uint64_t raw_dims[4];
} AsiV2VisionTensorShape;

_Static_assert(sizeof(AsiV2VisionTensorShape) == 40, "AsiV2VisionTensorShape must be 40 bytes");

/* 88-byte coordinate & temporal provenance record across resize/crop/rotation. */
typedef struct {
    uint32_t src_width;
    uint32_t src_height;
    uint32_t crop_x;
    uint32_t crop_y;
    uint32_t crop_w;
    uint32_t crop_h;
    uint32_t dst_width;
    uint32_t dst_height;
    float    scale_x;
    float    scale_y;
    float    rotation_deg;
    uint32_t frame_index;
    uint64_t timestamp_ms;
    uint8_t  source_sha256[32];
} AsiCoordTransform;

_Static_assert(sizeof(AsiCoordTransform) == 88, "AsiCoordTransform must be 88 bytes");

typedef struct {
    const uint8_t     *rgb_u8;
    uint32_t           width;
    uint32_t           height;
    uint32_t           stride_bytes; /* 0 -> tightly packed width * 3 */
    AsiCoordTransform  provenance;
} AsiImageView;

typedef struct {
    uint32_t resized_w;
    uint32_t resized_h;
    uint32_t grid_thw[3];     /* {T, H_merged, W_merged} */
    uint32_t n_patches;       /* T * H_patches * W_patches */
    uint32_t n_tokens;        /* T * H_merged * W_merged */
    size_t   est_scratch_bytes;
} AsiVisionPlan;

typedef void (*AsiVisionStageHook)(const char *stage_name,
                                   const float *data,
                                   size_t n_floats,
                                   void *user_data);

typedef struct LilaAsiVisionModel LilaAsiVisionModel;
typedef struct AsiVisionAdapterCtx AsiVisionAdapterCtx;

typedef struct {
    uint32_t adapter_id;
    const char *name;

    int (*validate_and_open)(const AsiV2VisionHeader *hdr,
                             const AsiV2TensorEntry *tensors,
                             const AsiV2VisionTensorShape *shapes,
                             uint32_t n_tensors,
                             const uint8_t *weights_data,
                             size_t weights_data_size,
                             AsiVisionAdapterCtx **out_ctx,
                             char *err_buf,
                             size_t err_cap);

    int (*plan_input)(const AsiVisionAdapterCtx *ctx,
                      const AsiV2VisionHeader *hdr,
                      uint32_t width,
                      uint32_t height,
                      uint32_t min_pixels,
                      uint32_t max_pixels,
                      AsiVisionPlan *out_plan);

    int (*encode_rgb)(AsiVisionAdapterCtx *ctx,
                      const AsiV2VisionHeader *hdr,
                      const AsiImageView *image,
                      uint32_t min_pixels,
                      uint32_t max_pixels,
                      uint32_t n_threads,
                      const volatile int *cancel_flag,
                      float *out_embds,
                      size_t out_embds_floats,
                      uint32_t out_grid_thw[3],
                      uint32_t *out_n_tokens,
                      AsiCoordTransform *out_transform,
                      AsiVisionStageHook stage_hook,
                      void *stage_user_data);

    void (*free_ctx)(AsiVisionAdapterCtx *ctx);
} AsiVisionAdapter;

/* Adapter lookup */
ASI_API const AsiVisionAdapter *asi_vision_find(uint32_t adapter_id);
extern const AsiVisionAdapter g_asi_vision_qwen35_vit;

/* Coordinate provenance helper */
ASI_API void asi_coord_map_to_source(const AsiCoordTransform *xf,
                                     float dst_x,
                                     float dst_y,
                                     float *out_src_x,
                                     float *out_src_y);

/* Public vision lifecycle and execution API */
typedef enum {
    ASI_VISION_EXEC_AUTO  = 0, /* Tier 2 native fast path if present, else Tier 1 Graph */
    ASI_VISION_EXEC_FAST  = 1, /* Require Tier 2 native fast path */
    ASI_VISION_EXEC_GRAPH = 2, /* Force Tier 1 Universal Graph Engine */
} AsiVisionExecMode;

typedef struct {
    uint64_t patch_ns;
    uint64_t layernorm_ns;
    uint64_t linear_q8_ns;
    uint64_t rope_ns;
    uint64_t sdpa_ns;
    uint64_t merger_ns;
    uint64_t total_ns;
} AsiVisionProfile;

ASI_API void lila_asi_vision_profile_enable(int enable);
ASI_API void lila_asi_vision_profile_reset(void);
ASI_API void lila_asi_vision_profile_get(AsiVisionProfile *out_prof);

ASI_API LilaAsiVisionModel *lila_asi_vision_open(const char *asiv_path,
                                                 char *err_buf,
                                                 size_t err_cap);
ASI_API void lila_asi_vision_close(LilaAsiVisionModel *vmodel);
ASI_API int  lila_asi_vision_evict(LilaAsiVisionModel *vmodel);
ASI_API int  lila_asi_vision_info(const LilaAsiVisionModel *vmodel, AsiV2VisionHeader *out_hdr);
ASI_API int  lila_asi_vision_has_graph(const LilaAsiVisionModel *vmodel);
ASI_API int  lila_asi_vision_set_exec_mode(LilaAsiVisionModel *vmodel, uint32_t exec_mode);
ASI_API int  lila_asi_vision_has_pairing(const LilaAsiVisionModel *vmodel);
ASI_API int  lila_asi_vision_pairing_info(const LilaAsiVisionModel *vmodel, AsiPairingHeader *out_hdr);
ASI_API int  lila_asi_vision_encode_paired_rgb(LilaAsiVisionModel *vmodel,
                                                const AsiImageView *image,
                                                uint32_t min_pixels,
                                                uint32_t max_pixels,
                                                uint32_t n_threads,
                                                const volatile int *cancel_flag,
                                                float *out_embds,
                                                size_t out_embds_floats,
                                                uint32_t out_grid_thw[3],
                                                uint32_t *out_n_tokens,
                                                AsiCoordTransform *out_transform);
ASI_API int  lila_asi_vision_check_compat(const LilaAsiVisionModel *vmodel,
                                          const AsiV2ModelConfig *llm_cfg,
                                          char *err_buf,
                                          size_t err_cap);
ASI_API int  lila_asi_vision_plan_grid(const LilaAsiVisionModel *vmodel,
                                       uint32_t width,
                                       uint32_t height,
                                       uint32_t min_pixels,
                                       uint32_t max_pixels,
                                       AsiVisionPlan *out_plan);
ASI_API int  lila_asi_vision_encode_rgb(LilaAsiVisionModel *vmodel,
                                        const AsiImageView *image,
                                        uint32_t min_pixels,
                                        uint32_t max_pixels,
                                        uint32_t n_threads,
                                        const volatile int *cancel_flag,
                                        float *out_embds,
                                        size_t out_embds_floats,
                                        uint32_t out_grid_thw[3],
                                        uint32_t *out_n_tokens,
                                        AsiCoordTransform *out_transform,
                                        AsiVisionStageHook stage_hook,
                                        void *stage_user_data);

#ifdef __cplusplus
}
#endif

#endif /* LILA_ASI_VISION_H */
