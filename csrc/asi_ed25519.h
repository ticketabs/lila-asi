#ifndef LILA_ASI_ED25519_H
#define LILA_ASI_ED25519_H

/*
 * Self-contained FIPS 180-4 SHA-256, SHA-512, and RFC 8032 Ed25519 verification.
 * Zero external dependencies; portable C11 across Linux, macOS, and Windows.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t state[8];
    uint64_t bitlen;
    uint8_t  data[64];
    size_t   datalen;
} AsiSha256Ctx;

typedef struct {
    uint64_t state[8];
    uint64_t len_lo;
    uint64_t len_hi;
    uint8_t  data[128];
    size_t   datalen;
} AsiSha512Ctx;

void asi_sha256_init(AsiSha256Ctx *ctx);
void asi_sha256_update(AsiSha256Ctx *ctx, const uint8_t *data, size_t len);
void asi_sha256_final(AsiSha256Ctx *ctx, uint8_t hash[32]);

void asi_sha512_init(AsiSha512Ctx *ctx);
void asi_sha512_update(AsiSha512Ctx *ctx, const uint8_t *data, size_t len);
void asi_sha512_final(AsiSha512Ctx *ctx, uint8_t hash[64]);

int asi_ct_memeq(const uint8_t *a, const uint8_t *b, size_t n);

/*
 * Verify an RFC 8032 Ed25519 signature over `msg` (of length `msg_len`)
 * using 32-byte public key `pk` and 64-byte signature `sig`.
 * Enforces canonical S < L and valid curve encoding.
 * Returns 0 if valid, -1 if invalid.
 */
int asi_ed25519_verify(const uint8_t sig[64],
                       const uint8_t *msg,
                       size_t msg_len,
                       const uint8_t pk[32]);

#ifdef __cplusplus
}
#endif

#endif /* LILA_ASI_ED25519_H */
