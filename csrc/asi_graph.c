#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "asi_graph.h"
#include "asi_kernels.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void set_graph_err(char *buf, size_t cap, const char *msg)
{
    if (!buf || cap == 0u) return;
    snprintf(buf, cap, "%s", msg ? msg : "unknown graph error");
}

int asi_graph_op_is_supported(uint32_t op_code)
{
    return (op_code < (uint32_t)ASI_OP_MAX_SUPPORTED) ? 1 : 0;
}

int asi_graph_validate(const uint8_t *sec_data,
                       size_t sec_size,
                       uint32_t n_tensors,
                       AsiGraphView *out_view,
                       char *err_buf,
                       size_t err_cap)
{
    if (!sec_data || !out_view || sec_size < sizeof(AsiGraphHeader)) {
        set_graph_err(err_buf, err_cap, "missing or truncated ASI_SEC_GRAPH");
        return -1;
    }

    AsiGraphHeader hdr;
    memcpy(&hdr, sec_data, sizeof(hdr));
    if (memcmp(hdr.magic, ASI_GRAPH_MAGIC, 8) != 0) {
        set_graph_err(err_buf, err_cap, "invalid ASIG0001 graph magic");
        return -1;
    }
    if (hdr.format_version != ASI_GRAPH_FORMAT_VERSION) {
        set_graph_err(err_buf, err_cap, "unsupported ASIG format_version");
        return -1;
    }
    if (hdr.n_slots == 0u || hdr.n_slots > 256u ||
        hdr.n_nodes == 0u || hdr.n_nodes > 4096u ||
        hdr.output_slot >= hdr.n_slots || hdr.input_slot >= hdr.n_slots) {
        set_graph_err(err_buf, err_cap, "invalid graph slot/node bounds");
        return -1;
    }
    size_t needed = sizeof(AsiGraphHeader) + (size_t)hdr.n_nodes * sizeof(AsiGraphNode);
    if (sec_size < needed) {
        set_graph_err(err_buf, err_cap, "ASI_SEC_GRAPH smaller than declared nodes");
        return -1;
    }

    const AsiGraphNode *nodes = (const AsiGraphNode *)(sec_data + sizeof(AsiGraphHeader));
    for (uint32_t i = 0u; i < hdr.n_nodes; i++) {
        const AsiGraphNode *nd = &nodes[i];
        if (!asi_graph_op_is_supported(nd->op_code)) {
            set_graph_err(err_buf, err_cap, "graph contains unsupported op_code");
            return -1;
        }
        if (nd->out_slot < -1 || nd->out_slot >= (int32_t)hdr.n_slots) {
            set_graph_err(err_buf, err_cap, "graph node out_slot out of bounds");
            return -1;
        }
        for (int s = 0; s < 4; s++) {
            if (nd->in_slots[s] < -1 || nd->in_slots[s] >= (int32_t)hdr.n_slots) {
                set_graph_err(err_buf, err_cap, "graph node in_slot out of bounds");
                return -1;
            }
            if (nd->weight_idx[s] < -1 || nd->weight_idx[s] >= (int32_t)n_tensors) {
                set_graph_err(err_buf, err_cap, "graph node weight_idx out of bounds");
                return -1;
            }
        }
    }

    out_view->header = hdr;
    out_view->nodes = nodes;
    return 0;
}

typedef struct {
    float   *data;
    size_t   cap_floats;
    size_t   len_floats;
    uint32_t rows;
    uint32_t cols;
} AsiSlotBuf;

static float *ensure_slot(AsiSlotBuf *slots,
                          uint32_t n_slots,
                          int32_t slot_id,
                          uint32_t rows,
                          uint32_t cols)
{
    if (slot_id < 0 || (uint32_t)slot_id >= n_slots) return NULL;
    size_t need = (size_t)rows * (size_t)cols;
    if (need == 0u) return NULL;
    AsiSlotBuf *sb = &slots[slot_id];
    if (need > sb->cap_floats) {
        float *nb = (float *)realloc(sb->data, need * sizeof(float));
        if (!nb) return NULL;
        sb->data = nb;
        sb->cap_floats = need;
    }
    sb->len_floats = need;
    sb->rows = rows;
    sb->cols = cols;
    return sb->data;
}

static const float *get_f32_weight(const AsiV2TensorEntry *tensors,
                                   const uint8_t *weights_data,
                                   int32_t idx)
{
    if (idx < 0 || !tensors || !weights_data) return NULL;
    if (tensors[idx].quant_type != 0u) return NULL;
    return (const float *)(weights_data + tensors[idx].data_offset);
}

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
                             void *stage_user_data)
{
    if (!graph || !graph->nodes || !hdr || !tensors || !weights_data || !image || !image->rgb_u8 || !out_embds)
        return -1;
    if (cancel_flag && *cancel_flag) return -2;

    uint32_t P = hdr->patch_size;
    uint32_t M = hdr->spatial_merge_size;
    if (P == 0u || M == 0u || new_h == 0u || new_w == 0u) return -1;
    uint32_t gh = new_h / P;
    uint32_t gw = new_w / P;
    uint32_t n_patches = gh * gw;
    uint32_t n_tokens = (gh / M) * (gw / M);
    uint32_t dim = hdr->hidden_size;
    uint32_t n_heads = hdr->n_heads;
    uint32_t head_dim = (n_heads > 0u) ? (dim / n_heads) : 0u;
    if (n_patches == 0u || n_tokens == 0u || head_dim == 0u) return -1;

    uint32_t n_slots = graph->header.n_slots;
    AsiSlotBuf *slots = (AsiSlotBuf *)calloc(n_slots, sizeof(AsiSlotBuf));
    if (!slots) return -1;

    AsiKernelCtx kctx;
    if (asi_kernel_ctx_init(&kctx, NULL, n_threads, cancel_flag) != 0) {
        free(slots);
        return -1;
    }

    int rc = 0;
    for (uint32_t ni = 0u; ni < graph->header.n_nodes; ni++) {
        if (cancel_flag && *cancel_flag) { rc = -2; break; }
        const AsiGraphNode *nd = &graph->nodes[ni];
        float *hook_ptr = NULL;
        size_t hook_floats = 0u;

        switch (nd->op_code) {
            case ASI_OP_NOP:
                break;

            case ASI_OP_RESIZE_RGB_NORM_CHW: {
                float *dst = ensure_slot(slots, n_slots, nd->out_slot, 3u * new_h, new_w);
                if (!dst) { rc = -1; break; }
                asi_k_resize_rgb_bilinear_chw(image->rgb_u8, image->width, image->height,
                                              image->stride_bytes, new_w, new_h,
                                              hdr->image_mean, hdr->image_std, dst);
                hook_ptr = dst;
                hook_floats = (size_t)3u * new_h * new_w;
                break;
            }

            case ASI_OP_PATCH_EXTRACT_MERGED: {
                if (nd->in_slots[0] < 0 || !slots[nd->in_slots[0]].data) { rc = -1; break; }
                uint32_t patch_vec = 3u * P * P;
                float *dst = ensure_slot(slots, n_slots, nd->out_slot, n_patches, patch_vec);
                if (!dst) { rc = -1; break; }
                asi_k_patch_extract_merged(slots[nd->in_slots[0]].data, new_h, new_w, P, M, dst);
                hook_ptr = dst;
                hook_floats = (size_t)n_patches * patch_vec;
                break;
            }

            case ASI_OP_CONV3D_PATCH_PAIR: {
                if (nd->in_slots[0] < 0 || !slots[nd->in_slots[0]].data) { rc = -1; break; }
                const float *w0 = get_f32_weight(tensors, weights_data, nd->weight_idx[0]);
                const float *w1 = (nd->weight_idx[1] >= 0) ? get_f32_weight(tensors, weights_data, nd->weight_idx[1]) : NULL;
                const float *bias = (nd->weight_idx[2] >= 0) ? get_f32_weight(tensors, weights_data, nd->weight_idx[2]) : NULL;
                if (!w0) { rc = -1; break; }
                uint32_t patch_vec = slots[nd->in_slots[0]].cols;
                float *dst = ensure_slot(slots, n_slots, nd->out_slot, n_patches, dim);
                if (!dst) { rc = -1; break; }
                asi_k_conv3d_patch_pair(&kctx, slots[nd->in_slots[0]].data, w0, w1, bias,
                                        n_patches, dim, patch_vec, dst);
                hook_ptr = dst;
                hook_floats = (size_t)n_patches * dim;
                break;
            }

            case ASI_OP_POS_EMBD_2D_INTERP_ADD: {
                if (nd->in_slots[0] < 0 || !slots[nd->in_slots[0]].data) { rc = -1; break; }
                const float *ptbl = get_f32_weight(tensors, weights_data, nd->weight_idx[0]);
                if (!ptbl) { rc = -1; break; }
                uint32_t rows = slots[nd->in_slots[0]].rows;
                uint32_t cols = slots[nd->in_slots[0]].cols;
                float *dst = ensure_slot(slots, n_slots, nd->out_slot, rows, cols);
                if (!dst) { rc = -1; break; }
                if (nd->out_slot != nd->in_slots[0]) {
                    memcpy(dst, slots[nd->in_slots[0]].data, (size_t)rows * cols * sizeof(float));
                }
                uint32_t table_entries = (uint32_t)tensors[nd->weight_idx[0]].shape[1];
                if (asi_k_pos_embd_2d_interp_add(ptbl, table_entries, gh, gw, M, cols, dst) != 0) {
                    rc = -1;
                    break;
                }
                hook_ptr = dst;
                hook_floats = (size_t)rows * cols;
                break;
            }

            case ASI_OP_ROPE_2D_VISION_TABLES: {
                uint32_t rope_half = head_dim / 2u;
                float *cos_buf = ensure_slot(slots, n_slots, nd->out_slot, n_patches, rope_half);
                float *sin_buf = ensure_slot(slots, n_slots, nd->in_slots[0], n_patches, rope_half);
                if (!cos_buf || !sin_buf) { rc = -1; break; }
                float theta = (nd->f_params[0] > 0.0f) ? nd->f_params[0] : hdr->rope_theta;
                asi_k_rope_2d_vision_tables(gh, gw, M, head_dim, theta, cos_buf, sin_buf);
                break;
            }

            case ASI_OP_LAYERNORM: {
                if (nd->in_slots[0] < 0 || !slots[nd->in_slots[0]].data) { rc = -1; break; }
                const float *w = get_f32_weight(tensors, weights_data, nd->weight_idx[0]);
                const float *b = (nd->weight_idx[1] >= 0) ? get_f32_weight(tensors, weights_data, nd->weight_idx[1]) : NULL;
                if (!w) { rc = -1; break; }
                uint32_t rows = slots[nd->in_slots[0]].rows;
                uint32_t cols = slots[nd->in_slots[0]].cols;
                float *dst = ensure_slot(slots, n_slots, nd->out_slot, rows, cols);
                if (!dst) { rc = -1; break; }
                float eps = (nd->f_params[0] > 0.0f) ? nd->f_params[0] : hdr->ln_eps;
                asi_k_layernorm(&kctx, slots[nd->in_slots[0]].data, w, b, rows, cols, eps, dst);
                hook_ptr = dst;
                hook_floats = (size_t)rows * cols;
                break;
            }

            case ASI_OP_RMSNORM: {
                if (nd->in_slots[0] < 0 || !slots[nd->in_slots[0]].data) { rc = -1; break; }
                const float *w = get_f32_weight(tensors, weights_data, nd->weight_idx[0]);
                if (!w) { rc = -1; break; }
                uint32_t rows = slots[nd->in_slots[0]].rows;
                uint32_t cols = slots[nd->in_slots[0]].cols;
                float *dst = ensure_slot(slots, n_slots, nd->out_slot, rows, cols);
                if (!dst) { rc = -1; break; }
                float eps = (nd->f_params[0] > 0.0f) ? nd->f_params[0] : hdr->ln_eps;
                asi_k_rmsnorm(&kctx, slots[nd->in_slots[0]].data, w, rows, cols, eps, dst);
                hook_ptr = dst;
                hook_floats = (size_t)rows * cols;
                break;
            }

            case ASI_OP_LINEAR_QUANT_BIAS: {
                if (nd->in_slots[0] < 0 || !slots[nd->in_slots[0]].data || nd->weight_idx[0] < 0) {
                    rc = -1;
                    break;
                }
                const AsiV2TensorEntry *wt = &tensors[nd->weight_idx[0]];
                const uint8_t *w_data = weights_data + wt->data_offset;
                const float *bias = (nd->weight_idx[1] >= 0) ? get_f32_weight(tensors, weights_data, nd->weight_idx[1]) : NULL;
                uint32_t M_rows = slots[nd->in_slots[0]].rows;
                uint32_t K_cols = (uint32_t)wt->shape[0];
                uint32_t N_out = (uint32_t)wt->shape[1];
                if (slots[nd->in_slots[0]].cols != K_cols) { rc = -1; break; }

                const float *res_ptr = NULL;
                if (nd->i_params[1] != 0 && nd->in_slots[1] >= 0) {
                    res_ptr = slots[nd->in_slots[1]].data;
                    if (!res_ptr || slots[nd->in_slots[1]].rows != M_rows ||
                        slots[nd->in_slots[1]].cols != N_out) {
                        rc = -1;
                        break;
                    }
                }
                float *dst = ensure_slot(slots, n_slots, nd->out_slot, M_rows, N_out);
                if (!dst) { rc = -1; break; }
                if (res_ptr && nd->in_slots[1] == nd->out_slot) {
                    res_ptr = dst; /* Updated pointer in case realloc moved buffer */
                }
                uint32_t act = (uint32_t)nd->i_params[0];
                rc = asi_k_linear_quant_bias(&kctx, w_data, wt->quant_type, bias,
                                             slots[nd->in_slots[0]].data,
                                             M_rows, N_out, K_cols, act, res_ptr, dst);
                if (rc == 0) {
                    hook_ptr = dst;
                    hook_floats = (size_t)M_rows * N_out;
                }
                break;
            }

            case ASI_OP_ROPE_2D_QKV_APPLY: {
                if (nd->in_slots[0] < 0 || nd->in_slots[1] < 0 || nd->in_slots[2] < 0) { rc = -1; break; }
                const float *in_qkv = slots[nd->in_slots[0]].data;
                const float *cos_t = slots[nd->in_slots[1]].data;
                const float *sin_t = slots[nd->in_slots[2]].data;
                if (!in_qkv || !cos_t || !sin_t) { rc = -1; break; }
                uint32_t rows = slots[nd->in_slots[0]].rows;
                uint32_t cols = slots[nd->in_slots[0]].cols;
                float *dst = ensure_slot(slots, n_slots, nd->out_slot, rows, cols);
                if (!dst) { rc = -1; break; }
                if (nd->out_slot != nd->in_slots[0]) {
                    memcpy(dst, in_qkv, (size_t)rows * cols * sizeof(float));
                }
                asi_k_rope_2d_qkv_apply(&kctx, dst, cos_t, sin_t, rows, n_heads, head_dim);
                break;
            }

            case ASI_OP_SDPA_BIDIR_QKV: {
                if (nd->in_slots[0] < 0 || !slots[nd->in_slots[0]].data) { rc = -1; break; }
                uint32_t rows = slots[nd->in_slots[0]].rows;
                float *dst = ensure_slot(slots, n_slots, nd->out_slot, rows, dim);
                if (!dst) { rc = -1; break; }
                float scale = (nd->f_params[0] > 0.0f) ? nd->f_params[0] : (1.0f / sqrtf((float)head_dim));
                rc = asi_k_sdpa_bidirectional_qkv(&kctx, slots[nd->in_slots[0]].data,
                                                  rows, n_heads, head_dim, scale, dst);
                if (rc == 0) {
                    hook_ptr = dst;
                    hook_floats = (size_t)rows * dim;
                }
                break;
            }

            case ASI_OP_ADD: {
                if (nd->in_slots[0] < 0 || nd->in_slots[1] < 0) { rc = -1; break; }
                const float *a = slots[nd->in_slots[0]].data;
                const float *b = slots[nd->in_slots[1]].data;
                uint32_t rows = slots[nd->in_slots[0]].rows;
                uint32_t cols = slots[nd->in_slots[0]].cols;
                if (!a || !b || slots[nd->in_slots[1]].rows != rows || slots[nd->in_slots[1]].cols != cols) {
                    rc = -1;
                    break;
                }
                float *dst = ensure_slot(slots, n_slots, nd->out_slot, rows, cols);
                if (!dst) { rc = -1; break; }
                if (nd->out_slot == nd->in_slots[0]) {
                    asi_k_add_inplace(&kctx, dst, slots[nd->in_slots[1]].data, (size_t)rows * cols);
                } else {
                    size_t tot = (size_t)rows * cols;
                    for (size_t k = 0u; k < tot; k++) dst[k] = a[k] + b[k];
                }
                hook_ptr = dst;
                hook_floats = (size_t)rows * cols;
                break;
            }

            case ASI_OP_ACTIVATION: {
                if (nd->in_slots[0] < 0 || !slots[nd->in_slots[0]].data) { rc = -1; break; }
                uint32_t rows = slots[nd->in_slots[0]].rows;
                uint32_t cols = slots[nd->in_slots[0]].cols;
                float *dst = ensure_slot(slots, n_slots, nd->out_slot, rows, cols);
                if (!dst) { rc = -1; break; }
                if (nd->out_slot != nd->in_slots[0]) {
                    memcpy(dst, slots[nd->in_slots[0]].data, (size_t)rows * cols * sizeof(float));
                }
                asi_k_activation_inplace(&kctx, dst, (size_t)rows * cols, (uint32_t)nd->i_params[0]);
                hook_ptr = dst;
                hook_floats = (size_t)rows * cols;
                break;
            }

            case ASI_OP_MERGE_2X2_VIEW: {
                if (nd->in_slots[0] < 0 || !slots[nd->in_slots[0]].data) { rc = -1; break; }
                uint32_t merge_factor = (nd->i_params[0] > 0) ? (uint32_t)nd->i_params[0] : (M * M);
                uint32_t in_rows = slots[nd->in_slots[0]].rows;
                uint32_t in_cols = slots[nd->in_slots[0]].cols;
                if (merge_factor == 0u || (in_rows % merge_factor) != 0u) { rc = -1; break; }
                uint32_t out_rows = in_rows / merge_factor;
                uint32_t out_cols = in_cols * merge_factor;
                if (nd->out_slot == nd->in_slots[0]) {
                    slots[nd->out_slot].rows = out_rows;
                    slots[nd->out_slot].cols = out_cols;
                } else {
                    float *dst = ensure_slot(slots, n_slots, nd->out_slot, out_rows, out_cols);
                    if (!dst) { rc = -1; break; }
                    memcpy(dst, slots[nd->in_slots[0]].data, (size_t)out_rows * out_cols * sizeof(float));
                }
                break;
            }

            case ASI_OP_CONV2D_F32: {
                if (nd->in_slots[0] < 0 || !slots[nd->in_slots[0]].data) { rc = -1; break; }
                const float *w = get_f32_weight(tensors, weights_data, nd->weight_idx[0]);
                const float *b = (nd->weight_idx[1] >= 0) ? get_f32_weight(tensors, weights_data, nd->weight_idx[1]) : NULL;
                if (!w) { rc = -1; break; }
                uint32_t C_in = (uint32_t)nd->i_params[0];
                uint32_t H_in = (uint32_t)nd->i_params[1];
                uint32_t W_in = (uint32_t)nd->i_params[2];
                uint32_t C_out = (uint32_t)nd->i_params[3];
                uint32_t kH = (uint32_t)(nd->i_params[4] & 0xFFFF);
                uint32_t kW = (uint32_t)((nd->i_params[4] >> 16) & 0xFFFF);
                uint32_t stride_h = (nd->f_params[0] > 0.0f) ? (uint32_t)nd->f_params[0] : 1u;
                uint32_t stride_w = (nd->f_params[1] > 0.0f) ? (uint32_t)nd->f_params[1] : 1u;
                uint32_t pad_h = (uint32_t)nd->f_params[2];
                uint32_t pad_w = (uint32_t)nd->f_params[3];
                uint32_t groups = nd->flags ? nd->flags : 1u;
                if (kH == 0u || kW == 0u || H_in + 2u * pad_h < kH || W_in + 2u * pad_w < kW) {
                    rc = -1;
                    break;
                }
                uint32_t H_out = (H_in + 2u * pad_h - kH) / stride_h + 1u;
                uint32_t W_out = (W_in + 2u * pad_w - kW) / stride_w + 1u;
                float *dst = ensure_slot(slots, n_slots, nd->out_slot, C_out, H_out * W_out);
                if (!dst) { rc = -1; break; }
                rc = asi_k_conv2d_f32(&kctx, slots[nd->in_slots[0]].data, w, b,
                                      C_in, H_in, W_in, C_out, kH, kW,
                                      stride_h, stride_w, pad_h, pad_w, groups, dst);
                hook_ptr = dst;
                hook_floats = (size_t)C_out * H_out * W_out;
                break;
            }

            default:
                rc = -1;
                break;
        }

        if (rc != 0) break;
        if (stage_hook && hook_ptr && nd->stage_name[0] != '\0') {
            char sname[17];
            memcpy(sname, nd->stage_name, 16u);
            sname[16] = '\0';
            stage_hook(sname, hook_ptr, hook_floats, stage_user_data);
        }
    }

    if (rc == 0) {
        const AsiSlotBuf *out_sb = &slots[graph->header.output_slot];
        if (!out_sb->data || out_sb->len_floats == 0u || out_sb->len_floats > out_embds_floats) {
            rc = -1;
        } else {
            memcpy(out_embds, out_sb->data, out_sb->len_floats * sizeof(float));
        }
    }

    asi_kernel_ctx_destroy(&kctx);
    for (uint32_t s = 0u; s < n_slots; s++) free(slots[s].data);
    free(slots);
    return rc;
}
