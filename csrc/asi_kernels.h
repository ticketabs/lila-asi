#ifndef LILA_ASI_KERNELS_H
#define LILA_ASI_KERNELS_H

/*
 * asi_kernels — ASI Parallel Math & Multi-Job Execution Library
 *
 * Shared execution substrate for both:
 *   Tier 1: ASI Universal Graph Execution Engine (asi_graph.c)
 *   Tier 2: Native Architecture Fast Paths (asi_vision.c, asi_arch.c)
 *
 * Designed for cache-friendly AVX2/FMA execution on constrained CPUs (Broadwell 2C/4T)
 * while maintaining bitwise 1T vs NT determinism via disjoint output-domain work-stealing.
 */

#include "asi_pool.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    ASI_ACT_NONE      = 0,
    ASI_ACT_GELU_TANH = 1,
    ASI_ACT_GELU_ERF  = 2,
    ASI_ACT_SILU      = 3,
    ASI_ACT_RELU      = 4,
} AsiActivationType;

typedef struct {
    AsiThreadPool      *pool;
    uint32_t            n_threads;
    int                 owns_pool;
    const volatile int *cancel_flag;
} AsiKernelCtx;

typedef void (*AsiKernelRangeFn)(uint32_t start, uint32_t end, int tid, void *user_data);

typedef struct {
    void (*fn)(void *arg);
    void  *arg;
} AsiKernelJob;

/* Initialize kernel execution context. Reuses existing_pool if provided, otherwise
 * creates a persistent AsiThreadPool when n_threads > 1. */
int  asi_kernel_ctx_init(AsiKernelCtx *ctx,
                         AsiThreadPool *existing_pool,
                         uint32_t n_threads,
                         const volatile int *cancel_flag);
void asi_kernel_ctx_destroy(AsiKernelCtx *ctx);

/* Dynamic work-stealing parallel_for over [0, n_items) in chunks of size grain. */
void asi_kernel_parallel_for(AsiKernelCtx *ctx,
                             uint32_t n_items,
                             uint32_t grain,
                             AsiKernelRangeFn fn,
                             void *user_data);

/* Concurrent execution of independent heterogeneous jobs using the same persistent pool. */
void asi_kernel_run_jobs(AsiKernelCtx *ctx,
                         const AsiKernelJob *jobs,
                         uint32_t n_jobs);

/* Fused linear projection:
 *   out[m, n] = act(dot(x[m], W[n]) + bias[n]) + (residual ? residual[m, n] : 0)
 * Note: residual may alias out when act_type == ASI_ACT_NONE (in-place residual accumulation). */
int asi_k_linear_quant_bias(AsiKernelCtx *ctx,
                            const uint8_t *w_data,
                            uint32_t quant_type,
                            const float *bias,
                            const float *x,
                            uint32_t M,
                            uint32_t N,
                            uint32_t K,
                            uint32_t act_type,
                            const float *residual,
                            float *out);

void asi_k_layernorm(AsiKernelCtx *ctx,
                     const float *x,
                     const float *w,
                     const float *b,
                     uint32_t M,
                     uint32_t D,
                     float eps,
                     float *out);

void asi_k_rmsnorm(AsiKernelCtx *ctx,
                   const float *x,
                   const float *w,
                   uint32_t M,
                   uint32_t D,
                   float eps,
                   float *out);

void asi_k_activation_inplace(AsiKernelCtx *ctx,
                              float *x,
                              size_t n,
                              uint32_t act_type);

void asi_k_add_inplace(AsiKernelCtx *ctx,
                       float *dst,
                       const float *src,
                       size_t n);

void asi_k_resize_rgb_bilinear_chw(const uint8_t *rgb_u8,
                                   uint32_t src_w,
                                   uint32_t src_h,
                                   uint32_t stride_bytes,
                                   uint32_t dst_w,
                                   uint32_t dst_h,
                                   const float mean[3],
                                   const float std_v[3],
                                   float *out_chw);

void asi_k_patch_extract_merged(const float *img_chw,
                                uint32_t new_h,
                                uint32_t new_w,
                                uint32_t patch_size,
                                uint32_t merge_size,
                                float *out_patches);

void asi_k_conv3d_patch_pair(AsiKernelCtx *ctx,
                             const float *patches,
                             const float *w0,
                             const float *w1,
                             const float *bias,
                             uint32_t n_patches,
                             uint32_t dim,
                             uint32_t patch_vec,
                             float *out);

int asi_k_pos_embd_2d_interp_add(const float *pos_table,
                                 uint32_t table_entries,
                                 uint32_t gh,
                                 uint32_t gw,
                                 uint32_t merge_size,
                                 uint32_t dim,
                                 float *x_inout);

void asi_k_rope_2d_vision_tables(uint32_t gh,
                                 uint32_t gw,
                                 uint32_t merge_size,
                                 uint32_t head_dim,
                                 float theta,
                                 float *cos_out,
                                 float *sin_out);

void asi_k_rope_2d_qkv_apply(AsiKernelCtx *ctx,
                             float *qkv,
                             const float *cos_t,
                             const float *sin_t,
                             uint32_t n_patches,
                             uint32_t n_heads,
                             uint32_t head_dim);

int asi_k_sdpa_bidirectional_qkv(AsiKernelCtx *ctx,
                                 const float *qkv,
                                 uint32_t n_patches,
                                 uint32_t n_heads,
                                 uint32_t head_dim,
                                 float scale,
                                 float *out);

int asi_k_conv2d_f32(AsiKernelCtx *ctx,
                     const float *in_nchw,
                     const float *w_oihw,
                     const float *bias,
                     uint32_t C_in,
                     uint32_t H_in,
                     uint32_t W_in,
                     uint32_t C_out,
                     uint32_t kH,
                     uint32_t kW,
                     uint32_t stride_h,
                     uint32_t stride_w,
                     uint32_t pad_h,
                     uint32_t pad_w,
                     uint32_t groups,
                     float *out_nchw);

#ifdef __cplusplus
}
#endif

#endif /* LILA_ASI_KERNELS_H */
