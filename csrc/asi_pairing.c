#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "asi_pairing.h"
#include "asi_kernels.h"
#include "q4k.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void set_pair_err(char *buf, size_t cap, const char *msg)
{
    if (!buf || cap == 0u) return;
    snprintf(buf, cap, "%s", msg ? msg : "unknown pairing error");
}

static int sha256_is_nonzero(const uint8_t h[32])
{
    if (!h) return 0;
    for (int i = 0; i < 32; i++) {
        if (h[i] != 0u) return 1;
    }
    return 0;
}

static const AsiV2TensorEntry *find_ptensor(const AsiV2TensorEntry *tensors,
                                            uint32_t n_tensors,
                                            const char *name)
{
    for (uint32_t i = 0u; i < n_tensors; i++) {
        if (strncmp(tensors[i].name, name, sizeof(tensors[i].name)) == 0) {
            return &tensors[i];
        }
    }
    return NULL;
}

int asi_pairing_is_executable(const AsiPairingHeader *hdr)
{
    if (!hdr) return 0;
    if (hdr->val_passed != 1u) return 0;
    return (hdr->state == ASI_PAIR_STATE_APPROVED ||
            hdr->state == ASI_PAIR_STATE_CONTINUOUS_IMPROVEMENT) ? 1 : 0;
}

int asi_pairing_validate(const uint8_t *sec_data,
                         size_t sec_size,
                         const uint8_t expected_vision_sha256[32],
                         AsiPairingView *out_view,
                         char *err_buf,
                         size_t err_cap)
{
    if (!sec_data || !out_view || sec_size < sizeof(AsiPairingHeader)) {
        set_pair_err(err_buf, err_cap, "missing or truncated ASI_SEC_PAIRING");
        return -1;
    }

    AsiPairingHeader hdr;
    memcpy(&hdr, sec_data, sizeof(hdr));
    if (memcmp(hdr.magic, ASI_PAIRING_MAGIC, 8) != 0) {
        set_pair_err(err_buf, err_cap, "invalid ASIP0001 pairing magic");
        return -1;
    }
    if (hdr.format_version != ASI_PAIRING_FORMAT_VERSION) {
        set_pair_err(err_buf, err_cap, "unsupported ASI_PAIRING_FORMAT_VERSION");
        return -1;
    }
    if (!asi_pairing_is_executable(&hdr)) {
        set_pair_err(err_buf, err_cap, "pairing bridge is not validated/approved for execution");
        return -1;
    }
    if (hdr.in_dim == 0u || hdr.out_dim == 0u) {
        set_pair_err(err_buf, err_cap, "invalid pairing dimensions");
        return -1;
    }
    if (hdr.bridge_kind == ASI_BRIDGE_LINEAR) {
        if (hdr.n_tensors != 2u) {
            set_pair_err(err_buf, err_cap, "ASI_BRIDGE_LINEAR requires n_tensors == 2");
            return -1;
        }
    } else if (hdr.bridge_kind == ASI_BRIDGE_MLP_2LAYER) {
        if (hdr.n_tensors != 4u || hdr.hidden_dim == 0u) {
            set_pair_err(err_buf, err_cap, "ASI_BRIDGE_MLP_2LAYER requires n_tensors == 4 and hidden_dim > 0");
            return -1;
        }
    } else {
        set_pair_err(err_buf, err_cap, "unsupported pairing bridge_kind");
        return -1;
    }

    if (expected_vision_sha256 && sha256_is_nonzero(expected_vision_sha256)) {
        if (memcmp(hdr.source_vision_sha256, expected_vision_sha256, 32) != 0) {
            set_pair_err(err_buf, err_cap, "pairing source_vision_sha256 mismatch");
            return -1;
        }
    }

    uint64_t dir_end = sizeof(AsiPairingHeader) + (uint64_t)hdr.n_tensors * sizeof(AsiV2TensorEntry);
    if (hdr.weights_offset < dir_end || hdr.weights_offset > sec_size) {
        set_pair_err(err_buf, err_cap, "pairing weights_offset out of bounds");
        return -1;
    }

    const AsiV2TensorEntry *tensors = (const AsiV2TensorEntry *)(sec_data + sizeof(AsiPairingHeader));
    const uint8_t *wdata = sec_data + hdr.weights_offset;
    size_t wsize = sec_size - (size_t)hdr.weights_offset;

    for (uint32_t i = 0u; i < hdr.n_tensors; i++) {
        const AsiV2TensorEntry *t = &tensors[i];
        if (!lila_quant_is_supported(t->quant_type)) {
            set_pair_err(err_buf, err_cap, "unsupported pairing tensor quant_type");
            return -1;
        }
        if (t->data_offset > wsize || t->data_size > wsize - t->data_offset) {
            set_pair_err(err_buf, err_cap, "pairing tensor payload out of bounds");
            return -1;
        }
        if (t->quant_type == 0u) {
            size_t nf = (size_t)t->data_size / sizeof(float);
            const float *fp = (const float *)(wdata + t->data_offset);
            for (size_t k = 0u; k < nf; k++) {
                if (!isfinite(fp[k])) {
                    set_pair_err(err_buf, err_cap, "non-finite weight in pairing bridge");
                    return -1;
                }
            }
        }
    }

    const AsiV2TensorEntry *w0 = find_ptensor(tensors, hdr.n_tensors, "bridge.0.weight");
    const AsiV2TensorEntry *b0 = find_ptensor(tensors, hdr.n_tensors, "bridge.0.bias");
    if (!w0 || !b0 || b0->quant_type != 0u) {
        set_pair_err(err_buf, err_cap, "missing bridge.0 weight/bias");
        return -1;
    }
    uint32_t first_out = (hdr.bridge_kind == ASI_BRIDGE_LINEAR) ? hdr.out_dim : hdr.hidden_dim;
    if (w0->shape[0] != hdr.in_dim || w0->shape[1] != first_out || b0->shape[1] != first_out) {
        set_pair_err(err_buf, err_cap, "bridge.0 shape mismatch with pairing header");
        return -1;
    }

    if (hdr.bridge_kind == ASI_BRIDGE_MLP_2LAYER) {
        const AsiV2TensorEntry *w1 = find_ptensor(tensors, hdr.n_tensors, "bridge.1.weight");
        const AsiV2TensorEntry *b1 = find_ptensor(tensors, hdr.n_tensors, "bridge.1.bias");
        if (!w1 || !b1 || b1->quant_type != 0u) {
            set_pair_err(err_buf, err_cap, "missing bridge.1 weight/bias");
            return -1;
        }
        if (w1->shape[0] != hdr.hidden_dim || w1->shape[1] != hdr.out_dim || b1->shape[1] != hdr.out_dim) {
            set_pair_err(err_buf, err_cap, "bridge.1 shape mismatch with pairing header");
            return -1;
        }
    }

    out_view->header = hdr;
    out_view->tensors = tensors;
    out_view->weights_data = wdata;
    out_view->weights_data_size = wsize;
    return 0;
}

int asi_pairing_check_llm_compat(const AsiPairingView *pv,
                                 const AsiV2ModelConfig *llm_cfg,
                                 const uint8_t *target_llm_sha256_opt,
                                 char *err_buf,
                                 size_t err_cap)
{
    if (!pv || !llm_cfg) {
        set_pair_err(err_buf, err_cap, "null argument in asi_pairing_check_llm_compat");
        return -1;
    }
    if (!asi_pairing_is_executable(&pv->header)) {
        set_pair_err(err_buf, err_cap, "pairing is not approved/validated");
        return -1;
    }
    if (pv->header.target_llm_arch_id != 0u &&
        llm_cfg->arch_id != pv->header.target_llm_arch_id) {
        set_pair_err(err_buf, err_cap, "pairing target_llm_arch_id mismatch");
        return -1;
    }
    if (pv->header.out_dim != llm_cfg->hidden_size) {
        set_pair_err(err_buf, err_cap, "pairing out_dim does not match LLM hidden_size");
        return -1;
    }
    if (target_llm_sha256_opt &&
        sha256_is_nonzero(pv->header.target_llm_sha256) &&
        sha256_is_nonzero(target_llm_sha256_opt) &&
        memcmp(pv->header.target_llm_sha256, target_llm_sha256_opt, 32) != 0) {
        set_pair_err(err_buf, err_cap, "pairing target_llm_sha256 mismatch");
        return -1;
    }
    return 0;
}

int asi_pairing_apply(const AsiPairingView *pv,
                      const float *in_embds,
                      uint32_t n_tokens,
                      uint32_t n_threads,
                      const volatile int *cancel_flag,
                      float *out_embds,
                      size_t out_floats)
{
    if (!pv || !in_embds || !out_embds || n_tokens == 0u) return -1;
    if (!asi_pairing_is_executable(&pv->header)) return -1;
    if (cancel_flag && *cancel_flag) return -2;
    if (out_floats < (size_t)n_tokens * (size_t)pv->header.out_dim) return -1;

    const AsiV2TensorEntry *w0 = find_ptensor(pv->tensors, pv->header.n_tensors, "bridge.0.weight");
    const AsiV2TensorEntry *b0 = find_ptensor(pv->tensors, pv->header.n_tensors, "bridge.0.bias");
    if (!w0 || !b0) return -1;

    AsiKernelCtx kctx;
    if (asi_kernel_ctx_init(&kctx, NULL, n_threads, cancel_flag) != 0) return -1;

    int rc = -1;
    if (pv->header.bridge_kind == ASI_BRIDGE_LINEAR) {
        rc = asi_k_linear_quant_bias(&kctx,
                                     pv->weights_data + w0->data_offset,
                                     w0->quant_type,
                                     (const float *)(pv->weights_data + b0->data_offset),
                                     in_embds,
                                     n_tokens,
                                     pv->header.out_dim,
                                     pv->header.in_dim,
                                     pv->header.act_type,
                                     NULL,
                                     out_embds);
    } else if (pv->header.bridge_kind == ASI_BRIDGE_MLP_2LAYER) {
        const AsiV2TensorEntry *w1 = find_ptensor(pv->tensors, pv->header.n_tensors, "bridge.1.weight");
        const AsiV2TensorEntry *b1 = find_ptensor(pv->tensors, pv->header.n_tensors, "bridge.1.bias");
        if (!w1 || !b1) goto cleanup;

        float *hid = (float *)malloc((size_t)n_tokens * (size_t)pv->header.hidden_dim * sizeof(float));
        if (!hid) goto cleanup;

        rc = asi_k_linear_quant_bias(&kctx,
                                     pv->weights_data + w0->data_offset,
                                     w0->quant_type,
                                     (const float *)(pv->weights_data + b0->data_offset),
                                     in_embds,
                                     n_tokens,
                                     pv->header.hidden_dim,
                                     pv->header.in_dim,
                                     pv->header.act_type,
                                     NULL,
                                     hid);
        if (rc == 0) {
            rc = asi_k_linear_quant_bias(&kctx,
                                         pv->weights_data + w1->data_offset,
                                         w1->quant_type,
                                         (const float *)(pv->weights_data + b1->data_offset),
                                         hid,
                                         n_tokens,
                                         pv->header.out_dim,
                                         pv->header.hidden_dim,
                                         ASI_ACT_NONE,
                                         NULL,
                                         out_embds);
        }
        free(hid);
    }

cleanup:
    asi_kernel_ctx_destroy(&kctx);
    return rc;
}
