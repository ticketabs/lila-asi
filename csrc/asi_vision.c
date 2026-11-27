#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "asi_vision.h"
#include "asi_kernels.h"
#include "asi_graph.h"
#include "q4k.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

struct AsiVisionAdapterCtx {
    const AsiV2VisionHeader      *hdr;
    const AsiV2TensorEntry       *tensors;
    const AsiV2VisionTensorShape *shapes;
    uint32_t                      n_tensors;
    const uint8_t                *weights_data;
    size_t                        weights_data_size;

    /* Cached tensor pointers for fast Qwen3VL lookup */
    const AsiV2TensorEntry *patch_w0;
    const AsiV2TensorEntry *patch_w1;
    const AsiV2TensorEntry *patch_b;
    const AsiV2TensorEntry *pos_embd;
    const AsiV2TensorEntry *post_ln_w;
    const AsiV2TensorEntry *post_ln_b;
    const AsiV2TensorEntry *mm0_w;
    const AsiV2TensorEntry *mm0_b;
    const AsiV2TensorEntry *mm2_w;
    const AsiV2TensorEntry *mm2_b;
    struct {
        const AsiV2TensorEntry *ln1_w;
        const AsiV2TensorEntry *ln1_b;
        const AsiV2TensorEntry *qkv_w;
        const AsiV2TensorEntry *qkv_b;
        const AsiV2TensorEntry *out_w;
        const AsiV2TensorEntry *out_b;
        const AsiV2TensorEntry *ln2_w;
        const AsiV2TensorEntry *ln2_b;
        const AsiV2TensorEntry *up_w;
        const AsiV2TensorEntry *up_b;
        const AsiV2TensorEntry *down_w;
        const AsiV2TensorEntry *down_b;
    } *blk;
};

struct LilaAsiVisionModel {
    const uint8_t                *base;
    size_t                        file_size;
    int                           owns_mmap;
    AsiV2Header                   file_header;
    const AsiV2SectionEntry      *sections;
    AsiV2VisionHeader             vheader;
    const AsiV2TensorEntry       *tensors;
    const AsiV2VisionTensorShape *shapes;
    uint32_t                      n_tensors;
    const uint8_t                *weights_data;
    size_t                        weights_data_size;
    const AsiVisionAdapter       *adapter;
    AsiVisionAdapterCtx          *adapter_ctx;
    int                           has_graph;
    AsiGraphView                  graph;
    uint32_t                      exec_mode;
    int                           has_pairing;
    AsiPairingView                pairing;
};

static int              g_vision_profile_enabled = 0;
static AsiVisionProfile g_vision_profile = {0};

void lila_asi_vision_profile_enable(int enable)
{
    g_vision_profile_enabled = enable ? 1 : 0;
}

void lila_asi_vision_profile_reset(void)
{
    memset(&g_vision_profile, 0, sizeof(g_vision_profile));
}

void lila_asi_vision_profile_get(AsiVisionProfile *out_prof)
{
    if (!out_prof) return;
    *out_prof = g_vision_profile;
}

static void set_err(char *buf, size_t cap, const char *msg)
{
    if (!buf || cap == 0) return;
    snprintf(buf, cap, "%s", msg ? msg : "unknown vision error");
}

static const AsiV2TensorEntry *find_vtensor(const AsiV2TensorEntry *tensors,
                                            uint32_t n_tensors,
                                            const char *name)
{
    if (!tensors || !name) return NULL;
    for (uint32_t i = 0; i < n_tensors; i++) {
        if (strncmp(tensors[i].name, name, sizeof(tensors[i].name)) == 0)
            return &tensors[i];
    }
    return NULL;
}

static const uint8_t *tensor_bytes(const AsiVisionAdapterCtx *ctx,
                                   const AsiV2TensorEntry *t)
{
    return ctx->weights_data + t->data_offset;
}

/* ---------- Coordinate Provenance ---------- */

void asi_coord_map_to_source(const AsiCoordTransform *xf,
                             float dst_x,
                             float dst_y,
                             float *out_src_x,
                             float *out_src_y)
{
    if (!xf) {
        if (out_src_x) *out_src_x = dst_x;
        if (out_src_y) *out_src_y = dst_y;
        return;
    }
    uint32_t crop_w = xf->crop_w ? xf->crop_w : xf->src_width;
    uint32_t crop_h = xf->crop_h ? xf->crop_h : xf->src_height;
    uint32_t dst_w  = xf->dst_width ? xf->dst_width : crop_w;
    uint32_t dst_h  = xf->dst_height ? xf->dst_height : crop_h;

    double rad = (double)xf->rotation_deg * (3.14159265358979323846 / 180.0);
    double cr = cos(rad);
    double sr = sin(rad);
    float rot_w = (float)((double)crop_w * fabs(cr) + (double)crop_h * fabs(sr));
    float rot_h = (float)((double)crop_w * fabs(sr) + (double)crop_h * fabs(cr));

    float u = (dst_w > 0u) ? (dst_x * (rot_w / (float)dst_w)) : dst_x;
    float v = (dst_h > 0u) ? (dst_y * (rot_h / (float)dst_h)) : dst_y;

    double du = (double)u - (double)rot_w * 0.5;
    double dv = (double)v - (double)rot_h * 0.5;
    double dx = du * cr + dv * sr;
    double dy = -du * sr + dv * cr;

    float sx = (float)((double)xf->crop_x + (double)crop_w * 0.5 + dx);
    float sy = (float)((double)xf->crop_y + (double)crop_h * 0.5 + dy);
    if (out_src_x) *out_src_x = sx;
    if (out_src_y) *out_src_y = sy;
}

/* ---------- Math Kernels ---------- */

static void qwen3vl_smart_resize(uint32_t height,
                                 uint32_t width,
                                 uint32_t factor,
                                 uint32_t min_pixels,
                                 uint32_t max_pixels,
                                 uint32_t *out_h,
                                 uint32_t *out_w)
{
    if (factor == 0u) factor = 32u;
    if (min_pixels == 0u) min_pixels = 56u * 56u;
    if (max_pixels == 0u) max_pixels = 448u * 448u;

    double rh = nearbyint((double)height / (double)factor);
    double rw = nearbyint((double)width / (double)factor);
    uint64_t h_bar = (uint64_t)((rh < 1.0 ? 1.0 : rh) * (double)factor);
    uint64_t w_bar = (uint64_t)((rw < 1.0 ? 1.0 : rw) * (double)factor);

    if (h_bar * w_bar > (uint64_t)max_pixels) {
        double beta = sqrt(((double)height * (double)width) / (double)max_pixels);
        double fh = floor((double)height / beta / (double)factor);
        double fw = floor((double)width / beta / (double)factor);
        h_bar = (uint64_t)((fh < 1.0 ? 1.0 : fh) * (double)factor);
        w_bar = (uint64_t)((fw < 1.0 ? 1.0 : fw) * (double)factor);
    } else if (h_bar * w_bar < (uint64_t)min_pixels) {
        double prod = (double)height * (double)width;
        if (prod < 1.0) prod = 1.0;
        double beta = sqrt((double)min_pixels / prod);
        double ch = ceil((double)height * beta / (double)factor);
        double cw = ceil((double)width * beta / (double)factor);
        h_bar = (uint64_t)((ch < 1.0 ? 1.0 : ch) * (double)factor);
        w_bar = (uint64_t)((cw < 1.0 ? 1.0 : cw) * (double)factor);
    }
    *out_h = (uint32_t)h_bar;
    *out_w = (uint32_t)w_bar;
}

/* ---------- Qwen3VL ViT Adapter Implementation ---------- */

static void qwen3vl_free_ctx(AsiVisionAdapterCtx *ctx)
{
    if (!ctx) return;
    free(ctx->blk);
    free(ctx);
}

static int qwen3vl_validate_and_open(const AsiV2VisionHeader *hdr,
                                     const AsiV2TensorEntry *tensors,
                                     const AsiV2VisionTensorShape *shapes,
                                     uint32_t n_tensors,
                                     const uint8_t *weights_data,
                                     size_t weights_data_size,
                                     AsiVisionAdapterCtx **out_ctx,
                                     char *err_buf,
                                     size_t err_cap)
{
    (void)shapes;
    if (!hdr || !tensors || !weights_data || !out_ctx) {
        set_err(err_buf, err_cap, "null argument in qwen3vl_validate_and_open");
        return -1;
    }
    if (hdr->n_layers == 0u || hdr->n_layers > 64u ||
        hdr->n_heads == 0u || hdr->hidden_size == 0u ||
        (hdr->hidden_size % hdr->n_heads) != 0u ||
        hdr->spatial_merge_size == 0u || hdr->patch_size == 0u) {
        set_err(err_buf, err_cap, "invalid Qwen3VL ViT hyperparameters");
        return -1;
    }
    if (hdr->flags & ASI_VISION_FLAG_HAS_DEEPSTACK) {
        set_err(err_buf, err_cap, "deepstack vision layers not yet enabled");
        return -1;
    }

    AsiVisionAdapterCtx *ctx = (AsiVisionAdapterCtx *)calloc(1, sizeof(*ctx));
    if (!ctx) {
        set_err(err_buf, err_cap, "out of memory allocating vision adapter ctx");
        return -1;
    }
    ctx->hdr = hdr;
    ctx->tensors = tensors;
    ctx->shapes = shapes;
    ctx->n_tensors = n_tensors;
    ctx->weights_data = weights_data;
    ctx->weights_data_size = weights_data_size;

    ctx->blk = calloc(hdr->n_layers, sizeof(*ctx->blk));
    if (!ctx->blk) {
        qwen3vl_free_ctx(ctx);
        set_err(err_buf, err_cap, "out of memory allocating ViT blocks");
        return -1;
    }

    ctx->patch_w0 = find_vtensor(tensors, n_tensors, "v.patch_embd.weight");
    ctx->patch_w1 = find_vtensor(tensors, n_tensors, "v.patch_embd.weight.1");
    ctx->patch_b  = find_vtensor(tensors, n_tensors, "v.patch_embd.bias");
    ctx->pos_embd = find_vtensor(tensors, n_tensors, "v.position_embd.weight");
    ctx->post_ln_w = find_vtensor(tensors, n_tensors, "v.post_ln.weight");
    ctx->post_ln_b = find_vtensor(tensors, n_tensors, "v.post_ln.bias");
    ctx->mm0_w = find_vtensor(tensors, n_tensors, "mm.0.weight");
    ctx->mm0_b = find_vtensor(tensors, n_tensors, "mm.0.bias");
    ctx->mm2_w = find_vtensor(tensors, n_tensors, "mm.2.weight");
    ctx->mm2_b = find_vtensor(tensors, n_tensors, "mm.2.bias");

    if (!ctx->patch_w0 || !ctx->patch_b || !ctx->pos_embd ||
        !ctx->post_ln_w || !ctx->post_ln_b ||
        !ctx->mm0_w || !ctx->mm0_b || !ctx->mm2_w || !ctx->mm2_b) {
        qwen3vl_free_ctx(ctx);
        set_err(err_buf, err_cap, "missing required Qwen3VL base/merger tensors");
        return -1;
    }
    if (ctx->patch_w0->quant_type != 0u || (ctx->patch_w1 && ctx->patch_w1->quant_type != 0u) ||
        ctx->patch_b->quant_type != 0u || ctx->pos_embd->quant_type != 0u ||
        ctx->post_ln_w->quant_type != 0u || ctx->post_ln_b->quant_type != 0u) {
        qwen3vl_free_ctx(ctx);
        set_err(err_buf, err_cap, "patch/pos/norm weights must be F32");
        return -1;
    }

    for (uint32_t l = 0; l < hdr->n_layers; l++) {
        char name[48];
        #define FIND_BLK(field, suffix) do { \
            snprintf(name, sizeof(name), "v.blk.%u." suffix, l); \
            ctx->blk[l].field = find_vtensor(tensors, n_tensors, name); \
            if (!ctx->blk[l].field) { \
                qwen3vl_free_ctx(ctx); \
                set_err(err_buf, err_cap, "missing ViT block tensor"); \
                return -1; \
            } \
        } while (0)

        FIND_BLK(ln1_w, "ln1.weight");
        FIND_BLK(ln1_b, "ln1.bias");
        FIND_BLK(qkv_w, "attn_qkv.weight");
        FIND_BLK(qkv_b, "attn_qkv.bias");
        FIND_BLK(out_w, "attn_out.weight");
        FIND_BLK(out_b, "attn_out.bias");
        FIND_BLK(ln2_w, "ln2.weight");
        FIND_BLK(ln2_b, "ln2.bias");
        FIND_BLK(up_w,  "ffn_up.weight");
        FIND_BLK(up_b,  "ffn_up.bias");
        FIND_BLK(down_w, "ffn_down.weight");
        FIND_BLK(down_b, "ffn_down.bias");
        #undef FIND_BLK

        if (ctx->blk[l].ln1_w->quant_type != 0u || ctx->blk[l].ln1_b->quant_type != 0u ||
            ctx->blk[l].ln2_w->quant_type != 0u || ctx->blk[l].ln2_b->quant_type != 0u) {
            qwen3vl_free_ctx(ctx);
            set_err(err_buf, err_cap, "ViT LayerNorm weights must be F32");
            return -1;
        }
    }

    *out_ctx = ctx;
    return 0;
}

static int qwen3vl_plan_input(const AsiVisionAdapterCtx *ctx,
                              const AsiV2VisionHeader *hdr,
                              uint32_t width,
                              uint32_t height,
                              uint32_t min_pixels,
                              uint32_t max_pixels,
                              AsiVisionPlan *out_plan)
{
    (void)ctx;
    if (!hdr || !out_plan || width == 0u || height == 0u) return -1;
    uint32_t factor = hdr->patch_size * hdr->spatial_merge_size;
    uint32_t eff_max = max_pixels ? max_pixels : hdr->default_max_pixels;
    uint32_t new_h = 0u, new_w = 0u;
    qwen3vl_smart_resize(height, width, factor, min_pixels, eff_max, &new_h, &new_w);

    uint32_t gh = new_h / hdr->patch_size;
    uint32_t gw = new_w / hdr->patch_size;
    uint32_t hm = gh / hdr->spatial_merge_size;
    uint32_t wm = gw / hdr->spatial_merge_size;
    if (gh == 0u || gw == 0u || hm == 0u || wm == 0u) return -1;

    uint32_t n_patches = gh * gw;
    uint32_t n_tokens = hm * wm;
    out_plan->resized_w = new_w;
    out_plan->resized_h = new_h;
    out_plan->grid_thw[0] = 1u;
    out_plan->grid_thw[1] = hm;
    out_plan->grid_thw[2] = wm;
    out_plan->n_patches = n_patches;
    out_plan->n_tokens = n_tokens;

    size_t d = hdr->hidden_size;
    size_t inter = hdr->intermediate_size;
    size_t merge_dim = hdr->spatial_merge_size * hdr->spatial_merge_size * d;
    out_plan->est_scratch_bytes =
        ((size_t)3u * new_h * new_w +
         (size_t)n_patches * (d * 2u + 3u * d + inter) +
         (size_t)n_tokens * merge_dim) * sizeof(float);
    return 0;
}

static void populate_coord_transform(const AsiImageView *image,
                                     uint32_t new_w,
                                     uint32_t new_h,
                                     AsiCoordTransform *out_transform)
{
    if (!out_transform) return;
    *out_transform = image->provenance;
    if (out_transform->src_width == 0u) out_transform->src_width = image->width;
    if (out_transform->src_height == 0u) out_transform->src_height = image->height;
    if (out_transform->crop_w == 0u) out_transform->crop_w = image->width;
    if (out_transform->crop_h == 0u) out_transform->crop_h = image->height;
    out_transform->dst_width = new_w;
    out_transform->dst_height = new_h;
    out_transform->scale_x = (float)out_transform->crop_w / (float)new_w;
    out_transform->scale_y = (float)out_transform->crop_h / (float)new_h;
}

static int qwen3vl_linear_t(AsiKernelCtx *kctx,
                            const AsiVisionAdapterCtx *ctx,
                            const AsiV2TensorEntry *w_t,
                            const AsiV2TensorEntry *b_t,
                            const float *x,
                            uint32_t n_tokens,
                            uint32_t out_rows,
                            uint32_t in_cols,
                            uint32_t act_type,
                            const float *residual,
                            float *out)
{
    if (!w_t) return -1;
    const uint8_t *w_data = tensor_bytes(ctx, w_t);
    const float *bias = (b_t && b_t->quant_type == 0u) ? (const float *)tensor_bytes(ctx, b_t) : NULL;
    return asi_k_linear_quant_bias(kctx, w_data, w_t->quant_type, bias, x,
                                   n_tokens, out_rows, in_cols, act_type, residual, out);
}

static int qwen3vl_encode_rgb(AsiVisionAdapterCtx *ctx,
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
                              void *stage_user_data)
{
    if (!ctx || !hdr || !image || !image->rgb_u8 || !out_embds) return -1;
    if (cancel_flag && *cancel_flag) return -2;
    if (n_threads == 0u) n_threads = 1u;

    AsiVisionPlan plan;
    if (qwen3vl_plan_input(ctx, hdr, image->width, image->height, min_pixels, max_pixels, &plan) != 0)
        return -1;

    uint32_t new_w = plan.resized_w;
    uint32_t new_h = plan.resized_h;
    uint32_t P = hdr->patch_size;
    uint32_t M = hdr->spatial_merge_size;
    uint32_t gh = new_h / P;
    uint32_t gw = new_w / P;
    uint32_t hm = plan.grid_thw[1];
    uint32_t wm = plan.grid_thw[2];
    uint32_t n_patches = plan.n_patches;
    uint32_t n_tokens = plan.n_tokens;
    uint32_t dim = hdr->hidden_size;
    uint32_t inter = hdr->intermediate_size;
    uint32_t n_heads = hdr->n_heads;
    uint32_t head_dim = dim / n_heads;
    uint32_t patch_vec = 3u * P * P;
    uint32_t merge_dim = M * M * dim;
    uint32_t act_gelu = (hdr->flags & ASI_VISION_FLAG_USE_GELU_TANH) ? ASI_ACT_GELU_TANH : ASI_ACT_GELU_ERF;

    if (out_embds_floats < (size_t)n_tokens * (size_t)hdr->projection_dim) return -1;
    populate_coord_transform(image, new_w, new_h, out_transform);

    int rc = -1;
    size_t img_floats = (size_t)3u * new_h * new_w;
    float *img_chw = (float *)malloc(img_floats * sizeof(float));
    float *patches = (float *)malloc((size_t)n_patches * patch_vec * sizeof(float));
    float *x       = (float *)malloc((size_t)n_patches * dim * sizeof(float));
    float *xn      = (float *)malloc((size_t)n_patches * dim * sizeof(float));
    float *qkv     = (float *)malloc((size_t)n_patches * 3u * dim * sizeof(float));
    float *attn_in = (float *)malloc((size_t)n_patches * dim * sizeof(float));
    float *ffn_buf = (float *)malloc((size_t)n_patches * inter * sizeof(float));
    float *h1_buf  = (float *)malloc((size_t)n_tokens * merge_dim * sizeof(float));
    uint32_t rope_half = head_dim / 2u;
    float *rope_cos = (float *)malloc((size_t)n_patches * rope_half * sizeof(float));
    float *rope_sin = (float *)malloc((size_t)n_patches * rope_half * sizeof(float));

    AsiKernelCtx kctx;
    int kctx_ok = 0;
    if (!img_chw || !patches || !x || !xn || !qkv || !attn_in || !ffn_buf || !h1_buf || !rope_cos || !rope_sin)
        goto cleanup;
    if (asi_kernel_ctx_init(&kctx, NULL, n_threads, cancel_flag) != 0)
        goto cleanup;
    kctx_ok = 1;

    int prof = g_vision_profile_enabled;
    uint64_t t_total0 = prof ? asi_monotonic_ns() : 0u;
    uint64_t t_s = t_total0;

    /* 1. Bilinear RGB resize + normalization + merged-window patch extraction */
    asi_k_resize_rgb_bilinear_chw(image->rgb_u8, image->width, image->height,
                                  image->stride_bytes, new_w, new_h,
                                  hdr->image_mean, hdr->image_std, img_chw);
    if (cancel_flag && *cancel_flag) { rc = -2; goto cleanup; }
    asi_k_patch_extract_merged(img_chw, new_h, new_w, P, M, patches);

    /* 2. Conv3D patch embedding: dot(patch, w0 + w1) + bias */
    asi_k_conv3d_patch_pair(
        &kctx, patches,
        (const float *)tensor_bytes(ctx, ctx->patch_w0),
        ctx->patch_w1 ? (const float *)tensor_bytes(ctx, ctx->patch_w1) : NULL,
        (const float *)tensor_bytes(ctx, ctx->patch_b),
        n_patches, dim, patch_vec, x);
    if (stage_hook) stage_hook("patch_embd", x, (size_t)n_patches * dim, stage_user_data);

    /* 3. Bilinear 2D Position Embedding interpolation + 2D Vision RoPE tables */
    if (asi_k_pos_embd_2d_interp_add(
            (const float *)tensor_bytes(ctx, ctx->pos_embd),
            (uint32_t)ctx->pos_embd->shape[1], gh, gw, M, dim, x) != 0) {
        goto cleanup;
    }
    if (stage_hook) stage_hook("pos_added", x, (size_t)n_patches * dim, stage_user_data);

    asi_k_rope_2d_vision_tables(gh, gw, M, head_dim, hdr->rope_theta, rope_cos, rope_sin);
    if (prof) {
        uint64_t now = asi_monotonic_ns();
        g_vision_profile.patch_ns += now - t_s;
    }
    float attn_scale = 1.0f / sqrtf((float)head_dim);

    /* 4. 12 ViT Blocks via parallel asi_kernels (with fused GELU & fused residual) */
    size_t n_act = (size_t)n_patches * dim;
    for (uint32_t l = 0u; l < hdr->n_layers; l++) {
        if (cancel_flag && *cancel_flag) { rc = -2; goto cleanup; }

        if (prof) t_s = asi_monotonic_ns();
        asi_k_layernorm(&kctx, x,
                        (const float *)tensor_bytes(ctx, ctx->blk[l].ln1_w),
                        (const float *)tensor_bytes(ctx, ctx->blk[l].ln1_b),
                        n_patches, dim, hdr->ln_eps, xn);
        if (prof) {
            uint64_t now = asi_monotonic_ns();
            g_vision_profile.layernorm_ns += now - t_s;
            t_s = now;
        }

        if (qwen3vl_linear_t(&kctx, ctx, ctx->blk[l].qkv_w, ctx->blk[l].qkv_b,
                             xn, n_patches, 3u * dim, dim, ASI_ACT_NONE, NULL, qkv) != 0)
            goto cleanup;
        if (prof) {
            uint64_t now = asi_monotonic_ns();
            g_vision_profile.linear_q8_ns += now - t_s;
            t_s = now;
        }

        asi_k_rope_2d_qkv_apply(&kctx, qkv, rope_cos, rope_sin, n_patches, n_heads, head_dim);
        if (prof) {
            uint64_t now = asi_monotonic_ns();
            g_vision_profile.rope_ns += now - t_s;
            t_s = now;
        }

        if (asi_k_sdpa_bidirectional_qkv(&kctx, qkv, n_patches, n_heads, head_dim, attn_scale, attn_in) != 0) {
            rc = (cancel_flag && *cancel_flag) ? -2 : -1;
            goto cleanup;
        }
        if (prof) {
            uint64_t now = asi_monotonic_ns();
            g_vision_profile.sdpa_ns += now - t_s;
            t_s = now;
        }

        if (qwen3vl_linear_t(&kctx, ctx, ctx->blk[l].out_w, ctx->blk[l].out_b,
                             attn_in, n_patches, dim, dim, ASI_ACT_NONE, x, x) != 0)
            goto cleanup;
        if (prof) {
            uint64_t now = asi_monotonic_ns();
            g_vision_profile.linear_q8_ns += now - t_s;
            t_s = now;
        }

        asi_k_layernorm(&kctx, x,
                        (const float *)tensor_bytes(ctx, ctx->blk[l].ln2_w),
                        (const float *)tensor_bytes(ctx, ctx->blk[l].ln2_b),
                        n_patches, dim, hdr->ln_eps, xn);
        if (prof) {
            uint64_t now = asi_monotonic_ns();
            g_vision_profile.layernorm_ns += now - t_s;
            t_s = now;
        }

        if (qwen3vl_linear_t(&kctx, ctx, ctx->blk[l].up_w, ctx->blk[l].up_b,
                             xn, n_patches, inter, dim, act_gelu, NULL, ffn_buf) != 0)
            goto cleanup;

        if (qwen3vl_linear_t(&kctx, ctx, ctx->blk[l].down_w, ctx->blk[l].down_b,
                             ffn_buf, n_patches, dim, inter, ASI_ACT_NONE, x, x) != 0)
            goto cleanup;
        if (prof) {
            uint64_t now = asi_monotonic_ns();
            g_vision_profile.linear_q8_ns += now - t_s;
        }

        if (stage_hook) {
            char sname[24];
            snprintf(sname, sizeof(sname), "blk_%u", l);
            stage_hook(sname, x, n_act, stage_user_data);
        }
    }

    if (cancel_flag && *cancel_flag) { rc = -2; goto cleanup; }

    /* 5. Post-LN + 2x2 Spatial Merger MLP */
    if (prof) t_s = asi_monotonic_ns();
    asi_k_layernorm(&kctx, x,
                    (const float *)tensor_bytes(ctx, ctx->post_ln_w),
                    (const float *)tensor_bytes(ctx, ctx->post_ln_b),
                    n_patches, dim, hdr->ln_eps, xn);
    if (prof) {
        uint64_t now = asi_monotonic_ns();
        g_vision_profile.layernorm_ns += now - t_s;
        t_s = now;
    }
    if (stage_hook) stage_hook("post_ln", xn, n_act, stage_user_data);

    if (qwen3vl_linear_t(&kctx, ctx, ctx->mm0_w, ctx->mm0_b,
                         xn, n_tokens, merge_dim, merge_dim, act_gelu, NULL, h1_buf) != 0)
        goto cleanup;

    if (qwen3vl_linear_t(&kctx, ctx, ctx->mm2_w, ctx->mm2_b,
                         h1_buf, n_tokens, hdr->projection_dim, merge_dim, ASI_ACT_NONE, NULL, out_embds) != 0)
        goto cleanup;
    if (prof) {
        uint64_t now = asi_monotonic_ns();
        g_vision_profile.merger_ns += now - t_s;
        g_vision_profile.total_ns += now - t_total0;
    }

    if (stage_hook) {
        stage_hook("final_embds", out_embds, (size_t)n_tokens * hdr->projection_dim, stage_user_data);
    }

    if (out_grid_thw) {
        out_grid_thw[0] = 1u;
        out_grid_thw[1] = hm;
        out_grid_thw[2] = wm;
    }
    if (out_n_tokens) *out_n_tokens = n_tokens;
    rc = 0;

cleanup:
    if (kctx_ok) asi_kernel_ctx_destroy(&kctx);
    free(rope_sin);
    free(rope_cos);
    free(h1_buf);
    free(ffn_buf);
    free(attn_in);
    free(qkv);
    free(xn);
    free(x);
    free(patches);
    free(img_chw);
    return rc;
}

const AsiVisionAdapter g_asi_vision_qwen35_vit = {
    .adapter_id = ASI_VISION_ADAPTER_QWEN3VL_VIT,
    .name = "qwen3vl_vit",
    .validate_and_open = qwen3vl_validate_and_open,
    .plan_input = qwen3vl_plan_input,
    .encode_rgb = qwen3vl_encode_rgb,
    .free_ctx = qwen3vl_free_ctx,
};

const AsiVisionAdapter *asi_vision_find(uint32_t adapter_id)
{
    if (adapter_id == ASI_VISION_ADAPTER_QWEN3VL_VIT)
        return &g_asi_vision_qwen35_vit;
    return NULL;
}

/* ---------- Container Loader & Public Lifecycle API ---------- */

LilaAsiVisionModel *lila_asi_vision_open(const char *asiv_path, char *err_buf, size_t err_cap)
{
    if (!asiv_path) {
        set_err(err_buf, err_cap, "null asiv_path");
        return NULL;
    }

    size_t file_size = 0;
    const uint8_t *mapped = NULL;
    int owns_mmap = 0;

#ifdef _WIN32
    HANDLE hFile = CreateFileA(asiv_path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        set_err(err_buf, err_cap, "cannot open .asiv file");
        return NULL;
    }
    LARGE_INTEGER li;
    GetFileSizeEx(hFile, &li);
    file_size = (size_t)li.QuadPart;
    HANDLE hMap = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    CloseHandle(hFile);
    if (!hMap) {
        set_err(err_buf, err_cap, "CreateFileMappingA failed");
        return NULL;
    }
    mapped = (const uint8_t *)MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(hMap);
    owns_mmap = 1;
#else
    int fd = open(asiv_path, O_RDONLY);
    if (fd < 0) {
        set_err(err_buf, err_cap, "cannot open .asiv file");
        return NULL;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        close(fd);
        set_err(err_buf, err_cap, "fstat failed on .asiv file");
        return NULL;
    }
    file_size = (size_t)st.st_size;
    void *m = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (m == MAP_FAILED) {
        set_err(err_buf, err_cap, "mmap failed on .asiv file");
        return NULL;
    }
#if defined(__linux__) && defined(MADV_HUGEPAGE)
    (void)madvise(m, file_size, MADV_HUGEPAGE);
#endif
    mapped = (const uint8_t *)m;
    owns_mmap = 1;
#endif

    if (!mapped || file_size < sizeof(AsiV2Header)) {
        set_err(err_buf, err_cap, "invalid or truncated .asiv file");
        goto fail_map;
    }

    LilaAsiVisionModel *vm = (LilaAsiVisionModel *)calloc(1, sizeof(*vm));
    if (!vm) {
        set_err(err_buf, err_cap, "out of memory allocating LilaAsiVisionModel");
        goto fail_map;
    }
    vm->base = mapped;
    vm->file_size = file_size;
    vm->owns_mmap = owns_mmap;
    memcpy(&vm->file_header, mapped, sizeof(AsiV2Header));

    if (vm->file_header.magic != ASI_V2_MAGIC || vm->file_header.version != ASI_V2_VERSION) {
        set_err(err_buf, err_cap, "invalid .asiv outer ASI v2 magic/version");
        goto fail_vm;
    }
    if (vm->file_header.total_size > file_size ||
        vm->file_header.n_sections == 0u || vm->file_header.n_sections > 64u) {
        set_err(err_buf, err_cap, "truncated or invalid .asiv section header");
        goto fail_vm;
    }
    uint64_t tbl_off = vm->file_header.section_table_offset;
    uint64_t tbl_bytes = (uint64_t)vm->file_header.n_sections * sizeof(AsiV2SectionEntry);
    if (tbl_off < sizeof(AsiV2Header) || tbl_off + tbl_bytes > file_size) {
        set_err(err_buf, err_cap, ".asiv section table out of bounds");
        goto fail_vm;
    }
    vm->sections = (const AsiV2SectionEntry *)(mapped + tbl_off);

    const AsiV2SectionEntry *vsec = NULL;
    const AsiV2SectionEntry *gsec = NULL;
    const AsiV2SectionEntry *psec = NULL;
    /* LILA_ASI_VERIFY_CRC=1 forces eager CRC-64 checks over bulk .asiv vision weight sections. */
    const char *verify_env = getenv("LILA_ASI_VERIFY_CRC");
    int verify_bulk = (verify_env && verify_env[0] == '1');

    for (uint32_t i = 0; i < vm->file_header.n_sections; i++) {
        const AsiV2SectionEntry *s = &vm->sections[i];
        if (s->offset > file_size || s->size > file_size - s->offset ||
            (s->offset % ASI_V2_PAGE_SIZE) != 0u) {
            set_err(err_buf, err_cap, ".asiv section bounds/alignment error");
            goto fail_vm;
        }
        int is_bulk = (s->type == ASI_SEC_VISION || s->type == ASI_SEC_WEIGHTS || s->type == ASI_SEC_AUDIO);
        if (!is_bulk || verify_bulk) {
            uint64_t actual = asi_v2_crc64(mapped + s->offset, (size_t)s->size);
            if (actual != s->checksum) {
                set_err(err_buf, err_cap, ".asiv section CRC-64 checksum mismatch");
                goto fail_vm;
            }
        }
        if (s->type == ASI_SEC_VISION) vsec = s;
        else if (s->type == ASI_SEC_GRAPH) gsec = s;
        else if (s->type == ASI_SEC_PAIRING) psec = s;
    }

    if (!vsec || vsec->size < sizeof(AsiV2VisionHeader) + sizeof(AsiV2WeightsHeader)) {
        set_err(err_buf, err_cap, "missing or truncated ASI_SEC_VISION section");
        goto fail_vm;
    }

    const uint8_t *vp = mapped + vsec->offset;
    memcpy(&vm->vheader, vp, sizeof(AsiV2VisionHeader));
    if (memcmp(vm->vheader.magic, ASIV_MAGIC, 8) != 0) {
        set_err(err_buf, err_cap, "invalid inner ASIV0001 vision header magic");
        goto fail_vm;
    }
    if (vm->vheader.format_version != ASIV_FORMAT_VERSION) {
        set_err(err_buf, err_cap, "unsupported ASIV_FORMAT_VERSION");
        goto fail_vm;
    }

    const uint8_t *wp = vp + sizeof(AsiV2VisionHeader);
    size_t wsec_size = (size_t)(vsec->size - sizeof(AsiV2VisionHeader));
    AsiV2WeightsHeader wh;
    memcpy(&wh, wp, sizeof(wh));
    if (wh.n_tensors != vm->vheader.n_tensor_shapes || wh.n_tensors == 0u || wh.n_tensors > 4096u) {
        set_err(err_buf, err_cap, "invalid vision tensor count / shape mismatch");
        goto fail_vm;
    }
    uint64_t dir_bytes = sizeof(AsiV2WeightsHeader) +
        (uint64_t)wh.n_tensors * (sizeof(AsiV2TensorEntry) + sizeof(AsiV2VisionTensorShape));
    if (wh.data_offset < dir_bytes || wh.data_offset > wsec_size) {
        set_err(err_buf, err_cap, "vision tensor directory out of bounds");
        goto fail_vm;
    }

    vm->n_tensors = wh.n_tensors;
    vm->tensors = (const AsiV2TensorEntry *)(wp + sizeof(AsiV2WeightsHeader));
    vm->shapes = (const AsiV2VisionTensorShape *)(wp + sizeof(AsiV2WeightsHeader) +
                                                (size_t)wh.n_tensors * sizeof(AsiV2TensorEntry));
    vm->weights_data = wp + wh.data_offset;
    vm->weights_data_size = wsec_size - (size_t)wh.data_offset;

    for (uint32_t i = 0; i < vm->n_tensors; i++) {
        const AsiV2TensorEntry *t = &vm->tensors[i];
        if (t->data_offset > vm->weights_data_size ||
            t->data_size > vm->weights_data_size - t->data_offset) {
            set_err(err_buf, err_cap, "vision tensor payload out of bounds");
            goto fail_vm;
        }
        if (!lila_quant_is_supported(t->quant_type)) {
            set_err(err_buf, err_cap, "unsupported vision tensor quant_type");
            goto fail_vm;
        }
    }

    if (gsec) {
        if (asi_graph_validate(mapped + gsec->offset, (size_t)gsec->size,
                               vm->n_tensors, &vm->graph, err_buf, err_cap) != 0) {
            goto fail_vm;
        }
        vm->has_graph = 1;
    }

    if (psec) {
        if (asi_pairing_validate(mapped + psec->offset, (size_t)psec->size,
                                 vm->vheader.source_fingerprint, &vm->pairing,
                                 err_buf, err_cap) != 0) {
            goto fail_vm;
        }
        if (vm->pairing.header.in_dim != vm->vheader.projection_dim) {
            set_err(err_buf, err_cap, "pairing in_dim does not match vision projection_dim");
            goto fail_vm;
        }
        vm->has_pairing = 1;
    }

    /* LILA_ASI_VISION_EXEC=graph|fast overrides automatic fast-path vs ASIG0001 DAG selection. */
    const char *exec_env = getenv("LILA_ASI_VISION_EXEC");
    vm->exec_mode = ASI_VISION_EXEC_AUTO;
    if (exec_env && exec_env[0] != '\0') {
        if (strcmp(exec_env, "graph") == 0) vm->exec_mode = ASI_VISION_EXEC_GRAPH;
        else if (strcmp(exec_env, "fast") == 0) vm->exec_mode = ASI_VISION_EXEC_FAST;
    }

    vm->adapter = asi_vision_find(vm->vheader.adapter_id);
    if (vm->adapter) {
        if (vm->adapter->validate_and_open(&vm->vheader, vm->tensors, vm->shapes, vm->n_tensors,
                                           vm->weights_data, vm->weights_data_size,
                                           &vm->adapter_ctx, err_buf, err_cap) != 0) {
            goto fail_vm;
        }
    } else if (!vm->has_graph) {
        set_err(err_buf, err_cap, "unsupported vision adapter_id and no ASI_SEC_GRAPH present");
        goto fail_vm;
    }
    return vm;

fail_vm:
    free(vm);
fail_map:
    if (mapped && owns_mmap) {
#ifdef _WIN32
        UnmapViewOfFile(mapped);
#else
        munmap((void *)mapped, file_size);
#endif
    }
    return NULL;
}

void lila_asi_vision_close(LilaAsiVisionModel *vmodel)
{
    if (!vmodel) return;
    if (vmodel->adapter && vmodel->adapter_ctx && vmodel->adapter->free_ctx) {
        vmodel->adapter->free_ctx(vmodel->adapter_ctx);
        vmodel->adapter_ctx = NULL;
    }
    if (vmodel->base && vmodel->owns_mmap) {
#ifdef _WIN32
        UnmapViewOfFile(vmodel->base);
#else
        munmap((void *)vmodel->base, vmodel->file_size);
#endif
    }
    free(vmodel);
}

int lila_asi_vision_evict(LilaAsiVisionModel *vmodel)
{
    if (!vmodel || !vmodel->base) return -1;
#ifndef _WIN32
    if (vmodel->owns_mmap && vmodel->file_size > 0) {
        return madvise((void *)vmodel->base, vmodel->file_size, MADV_DONTNEED);
    }
#endif
    return 0;
}

int lila_asi_vision_info(const LilaAsiVisionModel *vmodel, AsiV2VisionHeader *out_hdr)
{
    if (!vmodel || !out_hdr) return -1;
    *out_hdr = vmodel->vheader;
    return 0;
}

int lila_asi_vision_check_compat(const LilaAsiVisionModel *vmodel,
                                 const AsiV2ModelConfig *llm_cfg,
                                 char *err_buf,
                                 size_t err_cap)
{
    if (!vmodel || !llm_cfg) {
        set_err(err_buf, err_cap, "null argument in lila_asi_vision_check_compat");
        return -1;
    }
    if (vmodel->vheader.mode != ASI_VISION_MODE_MULTIMODAL_EMBD) {
        /* Independent perception sidecars work alongside any LLM without shared embedding space */
        return 0;
    }
    int native_ok = 1;
    if (vmodel->vheader.expected_llm_arch_id != 0u &&
        llm_cfg->arch_id != vmodel->vheader.expected_llm_arch_id) {
        native_ok = 0;
        set_err(err_buf, err_cap, "vision sidecar expected_llm_arch_id mismatch");
    } else if (vmodel->vheader.expected_llm_hidden_size != 0u &&
               llm_cfg->hidden_size != vmodel->vheader.expected_llm_hidden_size) {
        native_ok = 0;
        set_err(err_buf, err_cap, "vision sidecar hidden_size mismatch with LLM");
    } else if (vmodel->vheader.projection_dim != llm_cfg->hidden_size) {
        native_ok = 0;
        set_err(err_buf, err_cap, "vision projection_dim does not match LLM hidden_size");
    }
    if (native_ok) return 0;
    if (vmodel->has_pairing) {
        return asi_pairing_check_llm_compat(&vmodel->pairing, llm_cfg, NULL, err_buf, err_cap);
    }
    return -1;
}

int lila_asi_vision_has_pairing(const LilaAsiVisionModel *vmodel)
{
    return (vmodel && vmodel->has_pairing) ? 1 : 0;
}

int lila_asi_vision_pairing_info(const LilaAsiVisionModel *vmodel, AsiPairingHeader *out_hdr)
{
    if (!vmodel || !vmodel->has_pairing || !out_hdr) return -1;
    *out_hdr = vmodel->pairing.header;
    return 0;
}

int lila_asi_vision_has_graph(const LilaAsiVisionModel *vmodel)
{
    return (vmodel && vmodel->has_graph) ? 1 : 0;
}

int lila_asi_vision_set_exec_mode(LilaAsiVisionModel *vmodel, uint32_t exec_mode)
{
    if (!vmodel) return -1;
    if (exec_mode > ASI_VISION_EXEC_GRAPH) return -1;
    if (exec_mode == ASI_VISION_EXEC_FAST && !vmodel->adapter) return -1;
    if (exec_mode == ASI_VISION_EXEC_GRAPH && !vmodel->has_graph) return -1;
    vmodel->exec_mode = exec_mode;
    return 0;
}

int lila_asi_vision_plan_grid(const LilaAsiVisionModel *vmodel,
                              uint32_t width,
                              uint32_t height,
                              uint32_t min_pixels,
                              uint32_t max_pixels,
                              AsiVisionPlan *out_plan)
{
    if (!vmodel) return -1;
    if (vmodel->adapter && vmodel->adapter->plan_input) {
        return vmodel->adapter->plan_input(vmodel->adapter_ctx, &vmodel->vheader,
                                           width, height, min_pixels, max_pixels, out_plan);
    }
    if (vmodel->has_graph) {
        return qwen3vl_plan_input(NULL, &vmodel->vheader, width, height, min_pixels, max_pixels, out_plan);
    }
    return -1;
}

int lila_asi_vision_encode_rgb(LilaAsiVisionModel *vmodel,
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
                               void *stage_user_data)
{
    if (!vmodel) return -1;
    int use_graph = 0;
    if (vmodel->exec_mode == ASI_VISION_EXEC_GRAPH) {
        if (!vmodel->has_graph) return -1;
        use_graph = 1;
    } else if (vmodel->exec_mode == ASI_VISION_EXEC_FAST) {
        if (!vmodel->adapter || !vmodel->adapter->encode_rgb) return -1;
        use_graph = 0;
    } else {
        if (vmodel->adapter && vmodel->adapter->encode_rgb) use_graph = 0;
        else if (vmodel->has_graph) use_graph = 1;
        else return -1;
    }

    if (!use_graph) {
        return vmodel->adapter->encode_rgb(vmodel->adapter_ctx, &vmodel->vheader,
                                           image, min_pixels, max_pixels, n_threads, cancel_flag,
                                           out_embds, out_embds_floats, out_grid_thw, out_n_tokens,
                                           out_transform, stage_hook, stage_user_data);
    }

    AsiVisionPlan plan;
    if (!image || !image->rgb_u8 || !out_embds) return -1;
    if (qwen3vl_plan_input(NULL, &vmodel->vheader, image->width, image->height,
                           min_pixels, max_pixels, &plan) != 0)
        return -1;
    if (out_embds_floats < (size_t)plan.n_tokens * (size_t)vmodel->vheader.projection_dim)
        return -1;
    populate_coord_transform(image, plan.resized_w, plan.resized_h, out_transform);

    int rc = asi_graph_execute_vision(&vmodel->graph, &vmodel->vheader,
                                      vmodel->tensors, vmodel->weights_data,
                                      image, plan.resized_h, plan.resized_w, n_threads,
                                      cancel_flag, out_embds, out_embds_floats,
                                      stage_hook, stage_user_data);
    if (rc != 0) return rc;

    if (out_grid_thw) {
        out_grid_thw[0] = plan.grid_thw[0];
        out_grid_thw[1] = plan.grid_thw[1];
        out_grid_thw[2] = plan.grid_thw[2];
    }
    if (out_n_tokens) *out_n_tokens = plan.n_tokens;
    return 0;
}

int lila_asi_vision_encode_paired_rgb(LilaAsiVisionModel *vmodel,
                                      const AsiImageView *image,
                                      uint32_t min_pixels,
                                      uint32_t max_pixels,
                                      uint32_t n_threads,
                                      const volatile int *cancel_flag,
                                      float *out_embds,
                                      size_t out_embds_floats,
                                      uint32_t out_grid_thw[3],
                                      uint32_t *out_n_tokens,
                                      AsiCoordTransform *out_transform)
{
    if (!vmodel || !vmodel->has_pairing || !image || !image->rgb_u8 || !out_embds) return -1;
    AsiVisionPlan plan;
    if (lila_asi_vision_plan_grid(vmodel, image->width, image->height, min_pixels, max_pixels, &plan) != 0)
        return -1;
    size_t bridge_out_floats = (size_t)plan.n_tokens * (size_t)vmodel->pairing.header.out_dim;
    if (out_embds_floats < bridge_out_floats) return -1;

    size_t raw_floats = (size_t)plan.n_tokens * (size_t)vmodel->vheader.projection_dim;
    float *raw_embds = (float *)malloc(raw_floats * sizeof(float));
    if (!raw_embds) return -1;

    int rc = lila_asi_vision_encode_rgb(vmodel, image, min_pixels, max_pixels, n_threads, cancel_flag,
                                        raw_embds, raw_floats, out_grid_thw, out_n_tokens, out_transform,
                                        NULL, NULL);
    if (rc == 0) {
        rc = asi_pairing_apply(&vmodel->pairing, raw_embds, plan.n_tokens, n_threads,
                               cancel_flag, out_embds, out_embds_floats);
    }
    free(raw_embds);
    return rc;
}
