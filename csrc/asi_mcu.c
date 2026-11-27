/*
 * asi_mcu.c — Strictly Freestanding C11 Bare-Metal Runtime for Microcontrollers
 *
 * Zero dependencies beyond <stdint.h> and <stddef.h>:
 *   compiles cleanly with `-ffreestanding -nostdlib` for RISC-V (ESP32-C3/C6/P4)
 *   and ARM Cortex-M4F (STM32 / nRF52 / RP2350), as well as standard host C11.
 */

#include "asi_mcu.h"

/* ---------------- Freestanding Memory, Hash, and Math Primitives ---------------- */

static inline uint16_t rd_u16(const uint8_t *p)
{
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static inline uint32_t rd_u32(const uint8_t *p)
{
    return (uint32_t)p[0] |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static inline uint64_t rd_u64(const uint8_t *p)
{
    return (uint64_t)rd_u32(p) | ((uint64_t)rd_u32(p + 4) << 32);
}

static inline float rd_f32(const uint8_t *p)
{
    union { uint32_t u; float f; } v;
    v.u = rd_u32(p);
    return v.f;
}

void *asi_mcu_memcpy(void *dst, const void *src, size_t n)
{
    uint8_t *d = (uint8_t *)dst;
    const uint8_t *s = (const uint8_t *)src;
    for (size_t i = 0u; i < n; i++) d[i] = s[i];
    return dst;
}

void *asi_mcu_memset(void *dst, int val, size_t n)
{
    uint8_t *d = (uint8_t *)dst;
    uint8_t b = (uint8_t)(val & 0xFF);
    for (size_t i = 0u; i < n; i++) d[i] = b;
    return dst;
}

uint64_t asi_mcu_crc64(const void *data, size_t len)
{
    /* Standard ECMA-182 polynomial matching engine/asi/asi_v2.c */
    const uint8_t *p = (const uint8_t *)data;
    uint64_t crc = 0xFFFFFFFFFFFFFFFFULL;
    const uint64_t poly = 0xC96C5795D7870F42ULL;
    for (size_t i = 0u; i < len; i++) {
        crc ^= (uint64_t)p[i];
        for (int b = 0; b < 8; b++) {
            uint64_t mask = (uint64_t)(-(int64_t)(crc & 1ULL));
            crc = (crc >> 1) ^ (poly & mask);
        }
    }
    return ~crc;
}

float asi_mcu_sqrtf(float x)
{
    if (x <= 0.0f) return 0.0f;
    union { float f; uint32_t u; } v;
    v.f = x;
    /* IEEE-754 initial guess + 4 Newton-Raphson steps */
    v.u = 0x1FBD1DF5u + (v.u >> 1);
    float y = v.f;
    y = 0.5f * (y + x / y);
    y = 0.5f * (y + x / y);
    y = 0.5f * (y + x / y);
    y = 0.5f * (y + x / y);
    return y;
}

float asi_mcu_expf(float x)
{
    if (x < -87.33f) return 0.0f;
    if (x > 88.0f) x = 88.0f;
    /* Range reduction x = k * ln(2) + r */
    const float log2ef = 1.44269504088896341f;
    const float ln2_hi = 0.693359375f;
    const float ln2_lo = -2.12194440e-4f;
    float fx = x * log2ef;
    int32_t k = (fx >= 0.0f) ? (int32_t)(fx + 0.5f) : (int32_t)(fx - 0.5f);
    float fk = (float)k;
    float r = (x - fk * ln2_hi) - fk * ln2_lo;
    float r2 = r * r;
    float y = 1.9875691500e-4f * r + 1.3981999507e-3f;
    y = y * r + 8.3334519073e-3f;
    y = y * r + 4.1665795894e-2f;
    y = y * r + 1.6666665459e-1f;
    y = y * r + 5.0000001201e-1f;
    y = y * r2 + r + 1.0f;
    union { float f; uint32_t u; } scale;
    int32_t exp_bits = (k + 127);
    if (exp_bits <= 0) return 0.0f;
    if (exp_bits >= 255) exp_bits = 254;
    scale.u = (uint32_t)exp_bits << 23;
    return y * scale.f;
}

float asi_mcu_tanhf(float x)
{
    if (x > 10.0f) return 1.0f;
    if (x < -10.0f) return -1.0f;
    float ax = (x < 0.0f) ? -x : x;
    float t = asi_mcu_expf(-2.0f * ax);
    float mag = (1.0f - t) / (1.0f + t);
    return (x < 0.0f) ? -mag : mag;
}

float asi_mcu_fp16_to_fp32(uint16_t h)
{
    uint32_t sign = ((uint32_t)(h & 0x8000u)) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x03FFu;
    union { uint32_t u; float f; } v;
    if (exp == 0u) {
        if (mant == 0u) {
            v.u = sign;
            return v.f;
        }
        while ((mant & 0x0400u) == 0u) {
            mant <<= 1;
            exp--;
        }
        exp++;
        mant &= ~0x0400u;
    } else if (exp == 31u) {
        v.u = sign | 0x7F800000u | (mant << 13);
        return v.f;
    }
    v.u = sign | ((exp + 112u) << 23) | (mant << 13);
    return v.f;
}

static inline float apply_act_mcu(float v, uint32_t act_type)
{
    switch (act_type) {
        case ASI_MCU_ACT_RELU:
            return (v > 0.0f) ? v : 0.0f;
        case ASI_MCU_ACT_SILU: {
            float e = asi_mcu_expf(-v);
            return v / (1.0f + e);
        }
        case ASI_MCU_ACT_GELU_TANH: {
            float v3 = v * v * v;
            float inner = 0.7978845608f * (v + 0.044715f * v3);
            return 0.5f * v * (1.0f + asi_mcu_tanhf(inner));
        }
        default:
            return v;
    }
}

/* ---------------- Fixed Static Arena Bump Allocator ---------------- */

void asi_mcu_arena_init(AsiMcuArena *arena, void *static_mem, size_t cap_bytes)
{
    if (!arena) return;
    arena->buf = (uint8_t *)static_mem;
    arena->cap = static_mem ? cap_bytes : 0u;
    arena->used = 0u;
    arena->peak = 0u;
}

void asi_mcu_arena_reset(AsiMcuArena *arena)
{
    if (!arena) return;
    arena->used = 0u;
}

void *asi_mcu_arena_alloc(AsiMcuArena *arena, size_t n_bytes, size_t align)
{
    if (!arena || !arena->buf || n_bytes == 0u) return NULL;
    if (align < 4u) align = 4u;
    uintptr_t base_addr = (uintptr_t)arena->buf;
    size_t misalign = (size_t)((base_addr + arena->used) & (align - 1u));
    size_t pad = misalign ? (align - misalign) : 0u;
    if (pad > arena->cap - arena->used) return NULL;
    size_t off = arena->used + pad;
    if (n_bytes > arena->cap - off) return NULL;
    arena->used = off + n_bytes;
    if (arena->used > arena->peak) arena->peak = arena->used;
    return arena->buf + off;
}

size_t asi_mcu_arena_peak(const AsiMcuArena *arena)
{
    return arena ? arena->peak : 0u;
}

/* ---------------- Zero-Copy ROM / Flash ASI v2 Reader ---------------- */

int asi_mcu_cart_open_mem(const uint8_t *rom,
                          size_t rom_size,
                          int verify_crc,
                          AsiMcuCart *out_cart)
{
    if (!rom || !out_cart || rom_size < 64u) return -1;
    uint32_t magic = rd_u32(rom + 0);
    uint32_t version = rd_u32(rom + 4);
    uint32_t arch_id = rd_u32(rom + 8);
    uint32_t flags = rd_u32(rom + 12);
    uint32_t n_sec = rd_u32(rom + 16);
    uint64_t total_sz = rd_u64(rom + 24);
    uint64_t tbl_off = rd_u64(rom + 32);

    if (magic != ASI_MCU_V2_MAGIC || version != ASI_MCU_V2_VERSION) return -1;
    if (n_sec == 0u || n_sec > ASI_MCU_MAX_SECTIONS || total_sz > rom_size) return -1;
    if (tbl_off < 64u || tbl_off + (uint64_t)n_sec * 32u > rom_size) return -1;

    for (uint32_t i = 0u; i < n_sec; i++) {
        const uint8_t *se = rom + tbl_off + (size_t)i * 32u;
        uint64_t off = rd_u64(se + 8);
        uint64_t sz = rd_u64(se + 16);
        uint64_t crc = rd_u64(se + 24);
        if (off > rom_size || sz > rom_size - off) return -1;
        if (verify_crc) {
            if (asi_mcu_crc64(rom + off, (size_t)sz) != crc) return -2;
        }
    }

    out_cart->rom = rom;
    out_cart->rom_size = rom_size;
    out_cart->arch_id = arch_id;
    out_cart->flags = flags;
    out_cart->n_sections = n_sec;
    out_cart->section_table_off = tbl_off;
    return 0;
}

int asi_mcu_find_section(const AsiMcuCart *cart,
                         uint32_t sec_type,
                         const uint8_t **out_ptr,
                         size_t *out_size)
{
    if (!cart || !cart->rom) return -1;
    for (uint32_t i = 0u; i < cart->n_sections; i++) {
        const uint8_t *se = cart->rom + cart->section_table_off + (size_t)i * 32u;
        if (rd_u32(se + 0) == sec_type) {
            uint64_t off = rd_u64(se + 8);
            uint64_t sz = rd_u64(se + 16);
            if (out_ptr) *out_ptr = cart->rom + off;
            if (out_size) *out_size = (size_t)sz;
            return 0;
        }
    }
    return -1;
}

int asi_mcu_weights_open(const AsiMcuCart *cart, AsiMcuWeights *out_w)
{
    const uint8_t *wsec = NULL;
    size_t wsize = 0u;
    if (!out_w || asi_mcu_find_section(cart, ASI_MCU_SEC_WEIGHTS, &wsec, &wsize) != 0) return -1;
    if (!wsec || wsize < 16u) return -1;
    uint32_t n_tensors = rd_u32(wsec + 0);
    uint64_t data_off = rd_u64(wsec + 8);
    uint64_t dir_end = 16u + (uint64_t)n_tensors * 80u;
    if (n_tensors == 0u || n_tensors > ASI_MCU_MAX_TENSORS || data_off < dir_end || data_off > wsize) return -1;

    const uint8_t *entries = wsec + 16u;
    size_t data_cap = wsize - (size_t)data_off;
    for (uint32_t i = 0u; i < n_tensors; i++) {
        const uint8_t *te = entries + (size_t)i * 80u;
        uint64_t toff = rd_u64(te + 64);
        uint64_t tsz = rd_u64(te + 72);
        if (toff > data_cap || tsz > data_cap - toff) return -1;
    }

    out_w->entries_ptr = entries;
    out_w->weights_data = wsec + data_off;
    out_w->weights_data_size = data_cap;
    out_w->n_tensors = n_tensors;
    return 0;
}

static int mcu_name_eq(const char *fixed40, const char *name)
{
    for (int i = 0; i < 40; i++) {
        char a = fixed40[i];
        char b = name[i];
        if (a != b) return 0;
        if (a == '\0') return 1;
    }
    return name[40] == '\0';
}

int asi_mcu_weights_find(const AsiMcuWeights *w,
                         const char *name,
                         AsiMcuTensor *out_t)
{
    if (!w || !w->entries_ptr || !name || !out_t) return -1;
    for (uint32_t i = 0u; i < w->n_tensors; i++) {
        const uint8_t *te = w->entries_ptr + (size_t)i * 80u;
        if (mcu_name_eq((const char *)te, name)) {
            uint64_t d0 = rd_u64(te + 48);
            uint64_t d1 = rd_u64(te + 56);
            uint64_t off = rd_u64(te + 64);
            uint64_t sz = rd_u64(te + 72);
            out_t->quant_type = rd_u32(te + 40);
            out_t->layout_flags = rd_u32(te + 44);
            out_t->cols = (uint32_t)(d0 ? d0 : 1u);
            out_t->rows = (uint32_t)(d1 ? d1 : 1u);
            out_t->data_size = sz;
            out_t->data = w->weights_data + off;
            return 0;
        }
    }
    return -1;
}

/* ---------------- Bare-Metal Quantized & Signal Kernels ---------------- */

int asi_mcu_linear_f32(const float *w_f32,
                       const float *bias_f32,
                       const float *x,
                       uint32_t rows_out,
                       uint32_t cols_in,
                       uint32_t act_type,
                       float *out)
{
    if (!w_f32 || !x || !out || rows_out == 0u || cols_in == 0u) return -1;
    for (uint32_t r = 0u; r < rows_out; r++) {
        const float *row = w_f32 + (size_t)r * cols_in;
        float acc = bias_f32 ? bias_f32[r] : 0.0f;
        for (uint32_t k = 0u; k < cols_in; k++) acc += row[k] * x[k];
        out[r] = apply_act_mcu(acc, act_type);
    }
    return 0;
}

int asi_mcu_linear_q8_0(const uint8_t *w_q8_0,
                        const float *bias_f32,
                        const float *x,
                        uint32_t rows_out,
                        uint32_t cols_in,
                        uint32_t act_type,
                        float *out)
{
    if (!w_q8_0 || !x || !out || rows_out == 0u || cols_in == 0u || (cols_in % 32u) != 0u) return -1;
    uint32_t n_blk = cols_in / 32u;
    size_t row_bytes = (size_t)n_blk * 34u;
    for (uint32_t r = 0u; r < rows_out; r++) {
        const uint8_t *row = w_q8_0 + (size_t)r * row_bytes;
        float acc = bias_f32 ? bias_f32[r] : 0.0f;
        for (uint32_t b = 0u; b < n_blk; b++) {
            const uint8_t *blk = row + (size_t)b * 34u;
            float d = asi_mcu_fp16_to_fp32(rd_u16(blk));
            const int8_t *qs = (const int8_t *)(blk + 2u);
            const float *xp = x + (size_t)b * 32u;
            float bsum = 0.0f;
            for (uint32_t k = 0u; k < 32u; k++) bsum += (float)qs[k] * xp[k];
            acc += d * bsum;
        }
        out[r] = apply_act_mcu(acc, act_type);
    }
    return 0;
}

int asi_mcu_linear_q4_0(const uint8_t *w_q4_0,
                        const float *bias_f32,
                        const float *x,
                        uint32_t rows_out,
                        uint32_t cols_in,
                        uint32_t act_type,
                        float *out)
{
    if (!w_q4_0 || !x || !out || rows_out == 0u || cols_in == 0u || (cols_in % 32u) != 0u) return -1;
    uint32_t n_blk = cols_in / 32u;
    size_t row_bytes = (size_t)n_blk * 18u;
    for (uint32_t r = 0u; r < rows_out; r++) {
        const uint8_t *row = w_q4_0 + (size_t)r * row_bytes;
        float acc = bias_f32 ? bias_f32[r] : 0.0f;
        for (uint32_t b = 0u; b < n_blk; b++) {
            const uint8_t *blk = row + (size_t)b * 18u;
            float d = asi_mcu_fp16_to_fp32(rd_u16(blk));
            const uint8_t *qs = blk + 2u;
            const float *xp = x + (size_t)b * 32u;
            float bsum = 0.0f;
            for (uint32_t k = 0u; k < 16u; k++) {
                int lo = (int)(qs[k] & 0x0Fu) - 8;
                int hi = (int)(qs[k] >> 4) - 8;
                bsum += (float)lo * xp[k] + (float)hi * xp[k + 16u];
            }
            acc += d * bsum;
        }
        out[r] = apply_act_mcu(acc, act_type);
    }
    return 0;
}

int asi_mcu_linear_tensor(const AsiMcuTensor *w,
                          const AsiMcuTensor *bias_opt,
                          const float *x,
                          uint32_t act_type,
                          float *out)
{
    if (!w || !x || !out) return -1;
    const float *bias = NULL;
    if (bias_opt) {
        if (bias_opt->quant_type != ASI_MCU_QUANT_F32) return -1;
        bias = (const float *)bias_opt->data;
    }
    if (w->quant_type == ASI_MCU_QUANT_F32) {
        /* Handle potentially unaligned ROM byte slices safely on strict MCUs */
        if (((uintptr_t)w->data & 3u) == 0u && (!bias || ((uintptr_t)bias & 3u) == 0u)) {
            return asi_mcu_linear_f32((const float *)w->data, bias, x, w->rows, w->cols, act_type, out);
        }
        for (uint32_t r = 0u; r < w->rows; r++) {
            const uint8_t *row_b = w->data + (size_t)r * w->cols * 4u;
            float acc = bias ? rd_f32((const uint8_t *)bias + (size_t)r * 4u) : 0.0f;
            for (uint32_t k = 0u; k < w->cols; k++) {
                acc += rd_f32(row_b + (size_t)k * 4u) * x[k];
            }
            out[r] = apply_act_mcu(acc, act_type);
        }
        return 0;
    }
    if (w->quant_type == ASI_MCU_QUANT_Q8_0) {
        return asi_mcu_linear_q8_0(w->data, bias, x, w->rows, w->cols, act_type, out);
    }
    if (w->quant_type == ASI_MCU_QUANT_Q4_0) {
        return asi_mcu_linear_q4_0(w->data, bias, x, w->rows, w->cols, act_type, out);
    }
    return -1;
}

int asi_mcu_layernorm(const float *x,
                      const float *w,
                      const float *b,
                      uint32_t n,
                      float eps,
                      float *out)
{
    if (!x || !w || !out || n == 0u) return -1;
    float sum = 0.0f;
    for (uint32_t i = 0u; i < n; i++) sum += x[i];
    float mean = sum / (float)n;
    float var = 0.0f;
    for (uint32_t i = 0u; i < n; i++) {
        float d = x[i] - mean;
        var += d * d;
    }
    float inv = 1.0f / asi_mcu_sqrtf(var / (float)n + eps);
    for (uint32_t i = 0u; i < n; i++) {
        float y = ((x[i] - mean) * inv) * w[i];
        out[i] = b ? (y + b[i]) : y;
    }
    return 0;
}

int asi_mcu_rmsnorm(const float *x,
                    const float *w,
                    uint32_t n,
                    float eps,
                    float *out)
{
    if (!x || !w || !out || n == 0u) return -1;
    float ss = 0.0f;
    for (uint32_t i = 0u; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / asi_mcu_sqrtf(ss / (float)n + eps);
    for (uint32_t i = 0u; i < n; i++) out[i] = x[i] * inv * w[i];
    return 0;
}

void asi_mcu_softmax(float *x, uint32_t n)
{
    if (!x || n == 0u) return;
    float mx = x[0];
    for (uint32_t i = 1u; i < n; i++) if (x[i] > mx) mx = x[i];
    float sum = 0.0f;
    for (uint32_t i = 0u; i < n; i++) {
        x[i] = asi_mcu_expf(x[i] - mx);
        sum += x[i];
    }
    float inv = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
    for (uint32_t i = 0u; i < n; i++) x[i] *= inv;
}

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
                       float *out_ct)
{
    if (!in_ct || !w_ckt || !out_ct || c_in == 0u || c_out == 0u || k_len == 0u || stride == 0u) return -1;
    if (t_in + 2u * pad < k_len) return -1;
    uint32_t t_out = (t_in + 2u * pad - k_len) / stride + 1u;
    for (uint32_t co = 0u; co < c_out; co++) {
        float bv = bias_opt ? bias_opt[co] : 0.0f;
        const float *wo = w_ckt + (size_t)co * c_in * k_len;
        float *oo = out_ct + (size_t)co * t_out;
        for (uint32_t to = 0u; to < t_out; to++) {
            int32_t ti0 = (int32_t)(to * stride) - (int32_t)pad;
            float acc = bv;
            for (uint32_t ci = 0u; ci < c_in; ci++) {
                const float *wi = wo + (size_t)ci * k_len;
                const float *ii = in_ct + (size_t)ci * t_in;
                for (uint32_t k = 0u; k < k_len; k++) {
                    int32_t ti = ti0 + (int32_t)k;
                    if (ti >= 0 && (uint32_t)ti < t_in) acc += ii[ti] * wi[k];
                }
            }
            oo[to] = apply_act_mcu(acc, act_type);
        }
    }
    return 0;
}

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
                                 float *out_chw)
{
    if (!in_chw || !w_ckk || !out_chw || c == 0u || k_size == 0u || stride == 0u) return -1;
    if (h_in + 2u * pad < k_size || w_in + 2u * pad < k_size) return -1;
    uint32_t h_out = (h_in + 2u * pad - k_size) / stride + 1u;
    uint32_t w_out = (w_in + 2u * pad - k_size) / stride + 1u;
    for (uint32_t ch = 0u; ch < c; ch++) {
        float bv = bias_opt ? bias_opt[ch] : 0.0f;
        const float *ich = in_chw + (size_t)ch * h_in * w_in;
        const float *wch = w_ckk + (size_t)ch * k_size * k_size;
        float *och = out_chw + (size_t)ch * h_out * w_out;
        for (uint32_t ho = 0u; ho < h_out; ho++) {
            int32_t hi0 = (int32_t)(ho * stride) - (int32_t)pad;
            for (uint32_t wo = 0u; wo < w_out; wo++) {
                int32_t wi0 = (int32_t)(wo * stride) - (int32_t)pad;
                float acc = bv;
                for (uint32_t kh = 0u; kh < k_size; kh++) {
                    int32_t hi = hi0 + (int32_t)kh;
                    if (hi < 0 || (uint32_t)hi >= h_in) continue;
                    for (uint32_t kw = 0u; kw < k_size; kw++) {
                        int32_t wi = wi0 + (int32_t)kw;
                        if (wi < 0 || (uint32_t)wi >= w_in) continue;
                        acc += ich[(size_t)hi * w_in + (uint32_t)wi] * wch[kh * k_size + kw];
                    }
                }
                och[(size_t)ho * w_out + wo] = apply_act_mcu(acc, act_type);
            }
        }
    }
    return 0;
}

static void fmt_fc_name(uint32_t layer, const char *suffix, char *out32)
{
    /* Write "fc<N>.weight" or "fc<N>.bias" without <stdio.h> */
    out32[0] = 'f';
    out32[1] = 'c';
    int p = 2;
    if (layer >= 10u) {
        out32[p++] = (char)('0' + ((layer / 10u) % 10u));
    }
    out32[p++] = (char)('0' + (layer % 10u));
    out32[p++] = '.';
    for (int i = 0; suffix[i] != '\0' && p < 31; i++) out32[p++] = suffix[i];
    out32[p] = '\0';
}

int asi_mcu_eval_mlp_cart(const AsiMcuWeights *w,
                          AsiMcuArena *arena,
                          const float *in_vec,
                          uint32_t in_dim,
                          uint32_t n_layers,
                          uint32_t hidden_act,
                          uint32_t final_softmax,
                          float *out_vec,
                          uint32_t out_dim)
{
    if (!w || !arena || !in_vec || !out_vec || n_layers == 0u || n_layers > 32u) return -1;
    asi_mcu_arena_reset(arena);

    const float *cur = in_vec;
    uint32_t cur_dim = in_dim;
    float *ping = NULL;
    float *pong = NULL;

    for (uint32_t l = 0u; l < n_layers; l++) {
        char wname[32], bname[32];
        AsiMcuTensor tw, tb;
        int has_b = 0;
        fmt_fc_name(l, "weight", wname);
        fmt_fc_name(l, "bias", bname);
        if (asi_mcu_weights_find(w, wname, &tw) != 0) return -1;
        if (tw.cols != cur_dim) return -1;
        if (asi_mcu_weights_find(w, bname, &tb) == 0) {
            if (tb.cols != tw.rows) return -1;
            has_b = 1;
        }
        int is_last = (l + 1u == n_layers);
        float *dst = NULL;
        if (is_last) {
            if (tw.rows != out_dim) return -1;
            dst = out_vec;
        } else {
            float **slot = ((l & 1u) == 0u) ? &ping : &pong;
            *slot = (float *)asi_mcu_arena_alloc(arena, (size_t)tw.rows * sizeof(float), 16u);
            if (!*slot) return -2;
            dst = *slot;
        }
        uint32_t act = is_last ? ASI_MCU_ACT_NONE : hidden_act;
        if (asi_mcu_linear_tensor(&tw, has_b ? &tb : NULL, cur, act, dst) != 0) return -1;
        cur = dst;
        cur_dim = tw.rows;
    }

    if (final_softmax) asi_mcu_softmax(out_vec, out_dim);
    return 0;
}
