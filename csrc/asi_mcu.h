#ifndef LILA_ASI_MCU_H
#define LILA_ASI_MCU_H

/*
 * asi_mcu — Bare-Metal Freestanding C11 Static-Arena Inference Profile
 *
 * Designed for Microcontrollers (ESP32-S3, ESP32-C3/C6/P4 RISC-V, STM32 Cortex-M4F,
 * RP2040/RP2350) with strict RAM/Flash limits:
 *   - ZERO OS file mapping (`mmap`), ZERO dynamic heap (`malloc`/`free`), ZERO `stdio`/`math`/`pthread`
 *   - Fixed caller-supplied static RAM scratch (`AsiMcuArena`)
 *   - Zero-copy XIP Flash / ROM reading of standard `.asi` v2 cartridges (`F32`, `Q8_0`, `Q4_0`)
 *   - Audio Conv1D + Depthwise Conv2D + Quantized MLP wakeup / sensor classification
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ASI_MCU_V2_MAGIC          0x41534921u /* "ASI!" LE */
#define ASI_MCU_V2_VERSION        2u
#define ASI_MCU_MAX_SECTIONS      32u
#define ASI_MCU_MAX_TENSORS       128u

#define ASI_MCU_SEC_MODEL_CONFIG  0x01u
#define ASI_MCU_SEC_WEIGHTS       0x02u
#define ASI_MCU_SEC_METADATA      0x08u

#define ASI_MCU_QUANT_F32         0u
#define ASI_MCU_QUANT_F16         1u
#define ASI_MCU_QUANT_Q4_0        2u
#define ASI_MCU_QUANT_Q8_0        8u

#define ASI_MCU_ACT_NONE          0u
#define ASI_MCU_ACT_GELU_TANH     1u
#define ASI_MCU_ACT_SILU          2u
#define ASI_MCU_ACT_RELU          3u

typedef struct {
    uint8_t *buf;
    size_t   cap;
    size_t   used;
    size_t   peak;
} AsiMcuArena;

typedef struct {
    const uint8_t *rom;
    size_t         rom_size;
    uint32_t       arch_id;
    uint32_t       flags;
    uint32_t       n_sections;
    uint64_t       section_table_off;
} AsiMcuCart;

typedef struct {
    const uint8_t *entries_ptr;
    const uint8_t *weights_data;
    size_t         weights_data_size;
    uint32_t       n_tensors;
} AsiMcuWeights;

typedef struct {
    uint32_t quant_type;
    uint32_t layout_flags;
    uint32_t rows;
    uint32_t cols;
    uint64_t data_size;
    const uint8_t *data;
} AsiMcuTensor;

/* Arena & freestanding utilities */
void     asi_mcu_arena_init(AsiMcuArena *arena, void *static_mem, size_t cap_bytes);
void     asi_mcu_arena_reset(AsiMcuArena *arena);
void    *asi_mcu_arena_alloc(AsiMcuArena *arena, size_t n_bytes, size_t align);
size_t   asi_mcu_arena_peak(const AsiMcuArena *arena);

void    *asi_mcu_memcpy(void *dst, const void *src, size_t n);
void    *asi_mcu_memset(void *dst, int val, size_t n);
uint64_t asi_mcu_crc64(const void *data, size_t len);
float    asi_mcu_sqrtf(float x);
float    asi_mcu_expf(float x);
float    asi_mcu_tanhf(float x);
float    asi_mcu_fp16_to_fp32(uint16_t h);

/* Zero-copy ROM/Flash .asi v2 reader */
int asi_mcu_cart_open_mem(const uint8_t *rom,
                          size_t rom_size,
                          int verify_crc,
                          AsiMcuCart *out_cart);

int asi_mcu_find_section(const AsiMcuCart *cart,
                         uint32_t sec_type,
                         const uint8_t **out_ptr,
                         size_t *out_size);

int asi_mcu_weights_open(const AsiMcuCart *cart, AsiMcuWeights *out_w);

int asi_mcu_weights_find(const AsiMcuWeights *w,
                         const char *name,
                         AsiMcuTensor *out_t);

/* Freestanding neural kernels */
int asi_mcu_linear_f32(const float *w_f32,
                       const float *bias_f32,
                       const float *x,
                       uint32_t rows_out,
                       uint32_t cols_in,
                       uint32_t act_type,
                       float *out);

int asi_mcu_linear_q8_0(const uint8_t *w_q8_0,
                        const float *bias_f32,
                        const float *x,
                        uint32_t rows_out,
                        uint32_t cols_in,
                        uint32_t act_type,
                        float *out);

int asi_mcu_linear_q4_0(const uint8_t *w_q4_0,
                        const float *bias_f32,
                        const float *x,
                        uint32_t rows_out,
                        uint32_t cols_in,
                        uint32_t act_type,
                        float *out);

int asi_mcu_linear_tensor(const AsiMcuTensor *w,
                          const AsiMcuTensor *bias_opt,
                          const float *x,
                          uint32_t act_type,
                          float *out);

int asi_mcu_layernorm(const float *x,
                      const float *w,
                      const float *b,
                      uint32_t n,
                      float eps,
                      float *out);

int asi_mcu_rmsnorm(const float *x,
                    const float *w,
                    uint32_t n,
                    float eps,
                    float *out);

void asi_mcu_softmax(float *x, uint32_t n);

int asi_mcu_conv1d_f32(const float *in_ct,
                       const float *w_ckt,
                       const float *bias_opt,
                       uint32_t c_in,
                       uint32_t t_in,
                       uint32_t c_out,
                       uint32_t k_len,
                       uint32_t stride,
                       uint32_t pad,
                       uint32_t act_type,
                       float *out_ct);

int asi_mcu_depthwise_conv2d_f32(const float *in_chw,
                                 const float *w_ckk,
                                 const float *bias_opt,
                                 uint32_t c,
                                 uint32_t h_in,
                                 uint32_t w_in,
                                 uint32_t k_size,
                                 uint32_t stride,
                                 uint32_t pad,
                                 uint32_t act_type,
                                 float *out_chw);

/* End-to-end static-arena evaluation of an N-layer .asi MLP (fc0..fc{n-1}) */
int asi_mcu_eval_mlp_cart(const AsiMcuWeights *w,
                          AsiMcuArena *arena,
                          const float *in_vec,
                          uint32_t in_dim,
                          uint32_t n_layers,
                          uint32_t hidden_act,
                          uint32_t final_softmax,
                          float *out_vec,
                          uint32_t out_dim);

#ifdef __cplusplus
}
#endif

#endif /* LILA_ASI_MCU_H */
