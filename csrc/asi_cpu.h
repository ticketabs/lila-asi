#ifndef LILA_ASI_CPU_H
#define LILA_ASI_CPU_H

/*
 * Runtime CPU feature detection for portable Lila ASI distribution.
 */

#include <stdint.h>
#include <stdio.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#  if defined(_MSC_VER)
#    include <intrin.h>
#  elif defined(__GNUC__) || defined(__clang__)
#    include <cpuid.h>
#  endif
#endif

static inline int asi_cpu_is_compiled_avx2(void)
{
#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
    return 1;
#else
    return 0;
#endif
}

static inline void asi_cpu_cpuid(uint32_t leaf, uint32_t subleaf, uint32_t out[4])
{
    out[0] = out[1] = out[2] = out[3] = 0u;
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#  if defined(_MSC_VER)
    int info[4] = {0, 0, 0, 0};
    __cpuidex(info, (int)leaf, (int)subleaf);
    out[0] = (uint32_t)info[0];
    out[1] = (uint32_t)info[1];
    out[2] = (uint32_t)info[2];
    out[3] = (uint32_t)info[3];
#  elif defined(__GNUC__) || defined(__clang__)
    uint32_t max_leaf = __get_cpuid_max(0u, NULL);
    if (leaf <= max_leaf)
        __cpuid_count(leaf, subleaf, out[0], out[1], out[2], out[3]);
#  endif
#else
    (void)leaf;
    (void)subleaf;
#endif
}

static inline int asi_cpu_has_avx2_fma_f16c(void)
{
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    uint32_t r1[4], r7[4];
    asi_cpu_cpuid(1u, 0u, r1);
    /* Check FMA (ECX bit 12), OSXSAVE (ECX bit 27), F16C (ECX bit 29) */
    uint32_t needed_ecx = (1u << 12) | (1u << 27) | (1u << 29);
    if ((r1[2] & needed_ecx) != needed_ecx)
        return 0;

    /* Check OS-enabled XMM + YMM state via XGETBV(0) bits 1 and 2 */
    uint32_t xcr0_lo = 0u, xcr0_hi = 0u;
#  if defined(_MSC_VER)
    uint64_t xcr0 = _xgetbv(0u);
    xcr0_lo = (uint32_t)xcr0;
#  elif defined(__GNUC__) || defined(__clang__)
    __asm__ volatile(".byte 0x0f, 0x01, 0xd0" : "=a"(xcr0_lo), "=d"(xcr0_hi) : "c"(0u));
#  endif
    (void)xcr0_hi;
    if ((xcr0_lo & 0x6u) != 0x6u)
        return 0;

    asi_cpu_cpuid(7u, 0u, r7);
    /* Check AVX2 (EBX bit 5) */
    return (r7[1] & (1u << 5)) ? 1 : 0;
#else
    return 0;
#endif
}

static inline void asi_cpu_feature_string(char *buf, size_t cap)
{
    if (!buf || cap == 0) return;
#if defined(__x86_64__) || defined(_M_X64)
    const char *arch = "x86_64";
#elif defined(__aarch64__) || defined(_M_ARM64)
    const char *arch = "arm64";
#else
    const char *arch = "generic";
#endif
    int hw_avx2 = asi_cpu_has_avx2_fma_f16c();
    int compiled_avx2 = asi_cpu_is_compiled_avx2();
    snprintf(buf, cap, "%s compiled=%s host_avx2_fma_f16c=%s",
             arch,
             compiled_avx2 ? "avx2-fma-f16c" : "scalar",
             hw_avx2 ? "yes" : "no");
}

#endif /* LILA_ASI_CPU_H */
