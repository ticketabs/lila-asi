#include "asi_ed25519.h"
#include <string.h>

/* =========================================================================
 * Constant-Time Memory Compare
 * ========================================================================= */

int asi_ct_memeq(const uint8_t *a, const uint8_t *b, size_t n)
{
    uint8_t diff = 0;
    for (size_t i = 0; i < n; ++i)
        diff |= (uint8_t)(a[i] ^ b[i]);
    return diff == 0 ? 1 : 0;
}

/* =========================================================================
 * FIPS 180-4 SHA-256
 * ========================================================================= */

static const uint32_t K256[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
};

static inline uint32_t rotr32(uint32_t x, uint32_t n)
{
    return (x >> n) | (x << (32u - n));
}

static void sha256_transform(AsiSha256Ctx *ctx, const uint8_t blk[64])
{
    uint32_t m[64];
    for (int i = 0; i < 16; ++i)
    {
        m[i] = ((uint32_t)blk[i * 4] << 24) |
               ((uint32_t)blk[i * 4 + 1] << 16) |
               ((uint32_t)blk[i * 4 + 2] << 8) |
               ((uint32_t)blk[i * 4 + 3]);
    }
    for (int i = 16; i < 64; ++i)
    {
        uint32_t s0 = rotr32(m[i - 15], 7) ^ rotr32(m[i - 15], 18) ^ (m[i - 15] >> 3);
        uint32_t s1 = rotr32(m[i - 2], 17) ^ rotr32(m[i - 2], 19) ^ (m[i - 2] >> 10);
        m[i] = m[i - 16] + s0 + m[i - 7] + s1;
    }

    uint32_t a = ctx->state[0], b = ctx->state[1], c = ctx->state[2], d = ctx->state[3];
    uint32_t e = ctx->state[4], f = ctx->state[5], g = ctx->state[6], h = ctx->state[7];
    for (int i = 0; i < 64; ++i)
    {
        uint32_t S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t t1 = h + S1 + ch + K256[i] + m[i];
        uint32_t S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = S0 + maj;
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    ctx->state[0] += a; ctx->state[1] += b; ctx->state[2] += c; ctx->state[3] += d;
    ctx->state[4] += e; ctx->state[5] += f; ctx->state[6] += g; ctx->state[7] += h;
}

void asi_sha256_init(AsiSha256Ctx *ctx)
{
    ctx->datalen = 0;
    ctx->bitlen = 0;
    ctx->state[0] = 0x6a09e667u; ctx->state[1] = 0xbb67ae85u;
    ctx->state[2] = 0x3c6ef372u; ctx->state[3] = 0xa54ff53au;
    ctx->state[4] = 0x510e527fu; ctx->state[5] = 0x9b05688cu;
    ctx->state[6] = 0x1f83d9abu; ctx->state[7] = 0x5be0cd19u;
}

void asi_sha256_update(AsiSha256Ctx *ctx, const uint8_t *data, size_t len)
{
    for (size_t i = 0; i < len; ++i)
    {
        ctx->data[ctx->datalen++] = data[i];
        if (ctx->datalen == 64u)
        {
            sha256_transform(ctx, ctx->data);
            ctx->bitlen += 512ULL;
            ctx->datalen = 0;
        }
    }
}

void asi_sha256_final(AsiSha256Ctx *ctx, uint8_t hash[32])
{
    size_t i = ctx->datalen;
    if (ctx->datalen < 56u)
    {
        ctx->data[i++] = 0x80u;
        while (i < 56u) ctx->data[i++] = 0x00u;
    }
    else
    {
        ctx->data[i++] = 0x80u;
        while (i < 64u) ctx->data[i++] = 0x00u;
        sha256_transform(ctx, ctx->data);
        memset(ctx->data, 0, 56u);
    }
    ctx->bitlen += (uint64_t)ctx->datalen * 8ULL;
    for (int j = 0; j < 8; ++j)
        ctx->data[63 - j] = (uint8_t)((ctx->bitlen >> (j * 8)) & 0xFFu);
    sha256_transform(ctx, ctx->data);
    for (int j = 0; j < 8; ++j)
    {
        hash[j * 4 + 0] = (uint8_t)((ctx->state[j] >> 24) & 0xFFu);
        hash[j * 4 + 1] = (uint8_t)((ctx->state[j] >> 16) & 0xFFu);
        hash[j * 4 + 2] = (uint8_t)((ctx->state[j] >> 8) & 0xFFu);
        hash[j * 4 + 3] = (uint8_t)(ctx->state[j] & 0xFFu);
    }
}

/* =========================================================================
 * FIPS 180-4 SHA-512
 * ========================================================================= */

static const uint64_t K512[80] = {
    0x428a2f98d728ae22ULL, 0x7137449123ef65cdULL, 0xb5c0fbcfec4d3b2fULL, 0xe9b5dba58189dbbcULL,
    0x3956c25bf348b538ULL, 0x59f111f1b605d019ULL, 0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL,
    0xd807aa98a3030242ULL, 0x12835b0145706fbeULL, 0x243185be4ee4b28cULL, 0x550c7dc3d5ffb4e2ULL,
    0x72be5d74f27b896fULL, 0x80deb1fe3b1696b1ULL, 0x9bdc06a725c71235ULL, 0xc19bf174cf692694ULL,
    0xe49b69c19ef14ad2ULL, 0xefbe4786384f25e3ULL, 0x0fc19dc68b8cd5b5ULL, 0x240ca1cc77ac9c65ULL,
    0x2de92c6f592b0275ULL, 0x4a7484aa6ea6e483ULL, 0x5cb0a9dcbd41fbd4ULL, 0x76f988da831153b5ULL,
    0x983e5152ee66dfabULL, 0xa831c66d2db43210ULL, 0xb00327c898fb213fULL, 0xbf597fc7beef0ee4ULL,
    0xc6e00bf33da88fc2ULL, 0xd5a79147930aa725ULL, 0x06ca6351e003826fULL, 0x142929670a0e6e70ULL,
    0x27b70a8546d22ffcULL, 0x2e1b21385c26c926ULL, 0x4d2c6dfc5ac42aedULL, 0x53380d139d95b3dfULL,
    0x650a73548baf63deULL, 0x766a0abb3c77b2a8ULL, 0x81c2c92e47edaee6ULL, 0x92722c851482353bULL,
    0xa2bfe8a14cf10364ULL, 0xa81a664bbc423001ULL, 0xc24b8b70d0f89791ULL, 0xc76c51a30654be30ULL,
    0xd192e819d6ef5218ULL, 0xd69906245565a910ULL, 0xf40e35855771202aULL, 0x106aa07032bbd1b8ULL,
    0x19a4c116b8d2d0c8ULL, 0x1e376c085141ab53ULL, 0x2748774cdf8eeb99ULL, 0x34b0bcb5e19b48a8ULL,
    0x391c0cb3c5c95a63ULL, 0x4ed8aa4ae3418acbULL, 0x5b9cca4f7763e373ULL, 0x682e6ff3d6b2b8a3ULL,
    0x748f82ee5defb2fcULL, 0x78a5636f43172f60ULL, 0x84c87814a1f0ab72ULL, 0x8cc702081a6439ecULL,
    0x90befffa23631e28ULL, 0xa4506cebde82bde9ULL, 0xbef9a3f7b2c67915ULL, 0xc67178f2e372532bULL,
    0xca273eceea26619cULL, 0xd186b8c721c0c207ULL, 0xeada7dd6cde0eb1eULL, 0xf57d4f7fee6ed178ULL,
    0x06f067aa72176fbaULL, 0x0a637dc5a2c898a6ULL, 0x113f9804bef90daeULL, 0x1b710b35131c471bULL,
    0x28db77f523047d84ULL, 0x32caab7b40c72493ULL, 0x3c9ebe0a15c9bebcULL, 0x431d67c49c100d4cULL,
    0x4cc5d4becb3e42b6ULL, 0x597f299cfc657e2aULL, 0x5fcb6fab3ad6faecULL, 0x6c44198c4a475817ULL
};

static inline uint64_t rotr64(uint64_t x, uint32_t n)
{
    return (x >> n) | (x << (64u - n));
}

static void sha512_transform(AsiSha512Ctx *ctx, const uint8_t blk[128])
{
    uint64_t w[80];
    for (int i = 0; i < 16; ++i)
    {
        w[i] = ((uint64_t)blk[i * 8 + 0] << 56) |
               ((uint64_t)blk[i * 8 + 1] << 48) |
               ((uint64_t)blk[i * 8 + 2] << 40) |
               ((uint64_t)blk[i * 8 + 3] << 32) |
               ((uint64_t)blk[i * 8 + 4] << 24) |
               ((uint64_t)blk[i * 8 + 5] << 16) |
               ((uint64_t)blk[i * 8 + 6] << 8) |
               ((uint64_t)blk[i * 8 + 7]);
    }
    for (int i = 16; i < 80; ++i)
    {
        uint64_t s0 = rotr64(w[i - 15], 1) ^ rotr64(w[i - 15], 8) ^ (w[i - 15] >> 7);
        uint64_t s1 = rotr64(w[i - 2], 19) ^ rotr64(w[i - 2], 61) ^ (w[i - 2] >> 6);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    uint64_t a = ctx->state[0], b = ctx->state[1], c = ctx->state[2], d = ctx->state[3];
    uint64_t e = ctx->state[4], f = ctx->state[5], g = ctx->state[6], h = ctx->state[7];
    for (int i = 0; i < 80; ++i)
    {
        uint64_t S1 = rotr64(e, 14) ^ rotr64(e, 18) ^ rotr64(e, 41);
        uint64_t ch = (e & f) ^ ((~e) & g);
        uint64_t t1 = h + S1 + ch + K512[i] + w[i];
        uint64_t S0 = rotr64(a, 28) ^ rotr64(a, 34) ^ rotr64(a, 39);
        uint64_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint64_t t2 = S0 + maj;
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    ctx->state[0] += a; ctx->state[1] += b; ctx->state[2] += c; ctx->state[3] += d;
    ctx->state[4] += e; ctx->state[5] += f; ctx->state[6] += g; ctx->state[7] += h;
}

void asi_sha512_init(AsiSha512Ctx *ctx)
{
    ctx->datalen = 0;
    ctx->len_lo = 0;
    ctx->len_hi = 0;
    ctx->state[0] = 0x6a09e667f3bcc908ULL; ctx->state[1] = 0xbb67ae8584caa73bULL;
    ctx->state[2] = 0x3c6ef372fe94f82bULL; ctx->state[3] = 0xa54ff53a5f1d36f1ULL;
    ctx->state[4] = 0x510e527fade682d1ULL; ctx->state[5] = 0x9b05688c2b3e6c1fULL;
    ctx->state[6] = 0x1f83d9abfb41bd6bULL; ctx->state[7] = 0x5be0cd19137e2179ULL;
}

void asi_sha512_update(AsiSha512Ctx *ctx, const uint8_t *data, size_t len)
{
    for (size_t i = 0; i < len; ++i)
    {
        ctx->data[ctx->datalen++] = data[i];
        if (ctx->datalen == 128u)
        {
            sha512_transform(ctx, ctx->data);
            uint64_t prev = ctx->len_lo;
            ctx->len_lo += 1024ULL;
            if (ctx->len_lo < prev) ctx->len_hi++;
            ctx->datalen = 0;
        }
    }
}

void asi_sha512_final(AsiSha512Ctx *ctx, uint8_t hash[64])
{
    size_t i = ctx->datalen;
    uint64_t add_bits = (uint64_t)ctx->datalen * 8ULL;
    uint64_t prev = ctx->len_lo;
    ctx->len_lo += add_bits;
    if (ctx->len_lo < prev) ctx->len_hi++;

    if (ctx->datalen < 112u)
    {
        ctx->data[i++] = 0x80u;
        while (i < 112u) ctx->data[i++] = 0x00u;
    }
    else
    {
        ctx->data[i++] = 0x80u;
        while (i < 128u) ctx->data[i++] = 0x00u;
        sha512_transform(ctx, ctx->data);
        memset(ctx->data, 0, 112u);
    }
    for (int j = 0; j < 8; ++j)
    {
        ctx->data[112 + j] = (uint8_t)((ctx->len_hi >> (56 - j * 8)) & 0xFFu);
        ctx->data[120 + j] = (uint8_t)((ctx->len_lo >> (56 - j * 8)) & 0xFFu);
    }
    sha512_transform(ctx, ctx->data);
    for (int j = 0; j < 8; ++j)
    {
        for (int b = 0; b < 8; ++b)
            hash[j * 8 + b] = (uint8_t)((ctx->state[j] >> (56 - b * 8)) & 0xFFu);
    }
}

/* =========================================================================
 * RFC 8032 Ed25519 Verification (GF(2^255 - 19), 16x64-bit limbs)
 * ========================================================================= */

typedef int64_t gf[16];

static const gf GF_0 = {0};
static const gf GF_1 = {1};
static const gf ED_D = {
    0x78a3, 0x1359, 0x4dca, 0x75eb, 0xd8ab, 0x4141, 0x0a4d, 0x0070,
    0xe898, 0x7779, 0x4079, 0x8cc7, 0xfe73, 0x2b6f, 0x6cee, 0x5203
};
static const gf ED_D2 = {
    0xf159, 0x26b2, 0x9b94, 0xebd6, 0xb156, 0x8283, 0x149a, 0x00e0,
    0xd130, 0xeef3, 0x80f2, 0x198e, 0xfce7, 0x56df, 0xd9dc, 0x2406
};
static const gf ED_X = {
    0xd51a, 0x8f25, 0x2d60, 0xc956, 0xa7b2, 0x9525, 0xc760, 0x692c,
    0xdc5c, 0xfdd6, 0xe231, 0xc0a4, 0x53fe, 0xcd6e, 0x36d3, 0x2169
};
static const gf ED_Y = {
    0x6658, 0x6666, 0x6666, 0x6666, 0x6666, 0x6666, 0x6666, 0x6666,
    0x6666, 0x6666, 0x6666, 0x6666, 0x6666, 0x6666, 0x6666, 0x6666
};
static const gf ED_I = {
    0xa0b0, 0x4a0e, 0x1b27, 0xc4ee, 0xe478, 0xad2f, 0x1806, 0x2f43,
    0xd7a7, 0x3dfb, 0x0099, 0x2b4d, 0xdf0b, 0x4fc1, 0x2480, 0x2b83
};

static void gf_set(gf o, const gf a)
{
    for (int i = 0; i < 16; ++i) o[i] = a[i];
}

static void gf_car(gf o)
{
    int64_t c;
    for (int i = 0; i < 16; ++i)
    {
        o[i] += (1LL << 16);
        c = o[i] >> 16;
        o[(i + 1) * (i < 15)] += c - 1 + 37 * (c - 1) * (i == 15);
        o[i] -= c << 16;
    }
}

static void gf_sel(gf p, gf q, int b)
{
    int64_t t, c = ~(b - 1);
    for (int i = 0; i < 16; ++i)
    {
        t = c & (p[i] ^ q[i]);
        p[i] ^= t;
        q[i] ^= t;
    }
}

static void gf_add(gf o, const gf a, const gf b)
{
    for (int i = 0; i < 16; ++i) o[i] = a[i] + b[i];
}

static void gf_sub(gf o, const gf a, const gf b)
{
    for (int i = 0; i < 16; ++i) o[i] = a[i] - b[i];
}

static void gf_mul(gf o, const gf a, const gf b)
{
    int64_t t[31] = {0};
    for (int i = 0; i < 16; ++i)
        for (int j = 0; j < 16; ++j)
            t[i + j] += a[i] * b[j];
    for (int i = 0; i < 15; ++i)
        t[i] += 38 * t[i + 16];
    for (int i = 0; i < 16; ++i)
        o[i] = t[i];
    gf_car(o);
    gf_car(o);
}

static void gf_sq(gf o, const gf a)
{
    gf_mul(o, a, a);
}

static void gf_pack(uint8_t o[32], const gf n)
{
    gf m, t;
    gf_set(t, n);
    gf_car(t);
    gf_car(t);
    gf_car(t);
    for (int j = 0; j < 2; ++j)
    {
        m[0] = t[0] - 0xffed;
        for (int i = 1; i < 15; ++i)
        {
            m[i] = t[i] - 0xffff - ((m[i - 1] >> 16) & 1);
            m[i - 1] &= 0xffff;
        }
        m[15] = t[15] - 0x7fff - ((m[14] >> 16) & 1);
        int b = (int)((m[15] >> 16) & 1);
        m[14] &= 0xffff;
        gf_sel(t, m, 1 - b);
    }
    for (int i = 0; i < 16; ++i)
    {
        o[2 * i] = (uint8_t)(t[i] & 0xff);
        o[2 * i + 1] = (uint8_t)(t[i] >> 8);
    }
}

static void gf_unpack(gf o, const uint8_t n[32])
{
    for (int i = 0; i < 16; ++i)
        o[i] = (int64_t)n[2 * i] + ((int64_t)n[2 * i + 1] << 8);
    o[15] &= 0x7fff;
}

static int gf_neq(const gf a, const gf b)
{
    uint8_t c[32], d[32];
    gf_pack(c, a);
    gf_pack(d, b);
    return asi_ct_memeq(c, d, 32) ? 0 : 1;
}

static int gf_par(const gf a)
{
    uint8_t d[32];
    gf_pack(d, a);
    return d[0] & 1;
}

static void gf_pow2523(gf o, const gf i)
{
    gf c;
    gf_set(c, i);
    for (int a = 250; a >= 0; --a)
    {
        gf_sq(c, c);
        if (a != 1) gf_mul(c, c, i);
    }
    gf_set(o, c);
}

static void ed_add(gf p[4], gf q[4])
{
    gf a, b, c, d, t, e, f, g, h;
    gf_sub(a, p[1], p[0]);
    gf_sub(t, q[1], q[0]);
    gf_mul(a, a, t);
    gf_add(b, p[0], p[1]);
    gf_add(t, q[0], q[1]);
    gf_mul(b, b, t);
    gf_mul(c, p[3], q[3]);
    gf_mul(c, c, ED_D2);
    gf_mul(d, p[2], q[2]);
    gf_add(d, d, d);
    gf_sub(e, b, a);
    gf_sub(f, d, c);
    gf_add(g, d, c);
    gf_add(h, b, a);
    gf_mul(p[0], e, f);
    gf_mul(p[1], h, g);
    gf_mul(p[2], g, f);
    gf_mul(p[3], e, h);
}

static void ed_cswap(gf p[4], gf q[4], uint8_t b)
{
    for (int i = 0; i < 4; ++i) gf_sel(p[i], q[i], b);
}

static void ed_scalarmult(gf p[4], gf q[4], const uint8_t s[32])
{
    gf_set(p[0], GF_0);
    gf_set(p[1], GF_1);
    gf_set(p[2], GF_1);
    gf_set(p[3], GF_0);
    for (int i = 255; i >= 0; --i)
    {
        uint8_t b = (uint8_t)((s[i / 8] >> (i & 7)) & 1u);
        ed_cswap(p, q, b);
        ed_add(q, p);
        ed_add(p, p);
        ed_cswap(p, q, b);
    }
}

static int ed_unpackneg(gf r[4], const uint8_t p[32])
{
    gf t, chk, num, den, den2, den4, den6;
    gf_set(r[2], GF_1);
    gf_unpack(r[1], p);
    gf_sq(num, r[1]);
    gf_mul(den, num, ED_D);
    gf_sub(num, num, r[2]);
    gf_add(den, r[2], den);
    gf_sq(den2, den);
    gf_sq(den4, den2);
    gf_mul(den6, den4, den2);
    gf_mul(t, den6, num);
    gf_mul(t, t, den);
    gf_pow2523(t, t);
    gf_mul(t, t, num);
    gf_mul(t, t, den);
    gf_mul(t, t, den);
    gf_mul(r[0], t, den);

    gf_sq(chk, r[0]);
    gf_mul(chk, chk, den);
    if (gf_neq(chk, num))
        gf_mul(r[0], r[0], ED_I);
    gf_sq(chk, r[0]);
    gf_mul(chk, chk, den);
    if (gf_neq(chk, num))
        return -1;
    if (gf_par(r[0]) == (p[31] >> 7))
        gf_sub(r[0], GF_0, r[0]);
    gf_mul(r[3], r[0], r[1]);
    return 0;
}

static void ed_pack(uint8_t r[32], const gf p[4])
{
    /* Invert Z using Fermat's Little Theorem (Z^(p-2)) */
    gf zi, z2, z9, z11, z2_5_0, z2_10_0, z2_20_0, z2_50_0, z2_100_0, t, tx, ty;
    gf_sq(z2, p[2]);
    gf_sq(t, z2);
    gf_sq(t, t);
    gf_mul(z9, t, p[2]);
    gf_mul(z11, z9, z2);
    gf_sq(t, z11);
    gf_mul(z2_5_0, t, z9);

    gf_sq(t, z2_5_0);
    for (int i = 1; i < 5; ++i) gf_sq(t, t);
    gf_mul(z2_10_0, t, z2_5_0);

    gf_sq(t, z2_10_0);
    for (int i = 1; i < 10; ++i) gf_sq(t, t);
    gf_mul(z2_20_0, t, z2_10_0);

    gf_sq(t, z2_20_0);
    for (int i = 1; i < 20; ++i) gf_sq(t, t);
    gf_mul(t, t, z2_20_0);

    gf_sq(t, t);
    for (int i = 1; i < 10; ++i) gf_sq(t, t);
    gf_mul(z2_50_0, t, z2_10_0);

    gf_sq(t, z2_50_0);
    for (int i = 1; i < 50; ++i) gf_sq(t, t);
    gf_mul(z2_100_0, t, z2_50_0);

    gf_sq(t, z2_100_0);
    for (int i = 1; i < 100; ++i) gf_sq(t, t);
    gf_mul(t, t, z2_100_0);

    gf_sq(t, t);
    for (int i = 1; i < 50; ++i) gf_sq(t, t);
    gf_mul(t, t, z2_50_0);

    gf_sq(t, t);
    gf_sq(t, t);
    gf_sq(t, t);
    gf_sq(t, t);
    gf_sq(t, t);
    gf_mul(zi, t, z11);

    gf_mul(tx, p[0], zi);
    gf_mul(ty, p[1], zi);
    gf_pack(r, ty);
    r[31] ^= (uint8_t)(gf_par(tx) << 7);
}

/* Ed25519 group order L = 2^252 + 27742317777372353535851937790883648493 */
static const uint8_t ED25519_L[32] = {
    0xed, 0xd3, 0xf5, 0x5c, 0x1a, 0x63, 0x12, 0x58,
    0xd6, 0x9c, 0xf7, 0xa2, 0xde, 0xf9, 0xde, 0x14,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10
};

static int sc_is_canonical(const uint8_t s[32])
{
    for (int i = 31; i >= 0; --i)
    {
        if (s[i] < ED25519_L[i]) return 1;
        if (s[i] > ED25519_L[i]) return 0;
    }
    return 0; /* s == L is NOT canonical */
}

static void sc_reduce(uint8_t r[64])
{
    int64_t x[64];
    for (int i = 0; i < 64; ++i) x[i] = r[i];
    for (int i = 0; i < 64; ++i) r[i] = 0;

    static const int64_t L[32] = {
        0xed, 0xd3, 0xf5, 0x5c, 0x1a, 0x63, 0x12, 0x58,
        0xd6, 0x9c, 0xf7, 0xa2, 0xde, 0xf9, 0xde, 0x14,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10
    };
    int64_t carry;
    for (int i = 63; i >= 32; --i)
    {
        carry = 0;
        for (int j = i - 32, k = 0; j < i - 12; ++j, ++k)
        {
            x[j] += carry - 16 * x[i] * L[k];
            carry = (x[j] + 128) >> 8;
            x[j] -= carry * 256;
        }
        x[i - 12] += carry;
        x[i] = 0;
    }
    carry = 0;
    for (int j = 0; j < 32; ++j)
    {
        x[j] += carry - (x[31] >> 4) * L[j];
        carry = x[j] >> 8;
        x[j] &= 255;
    }
    for (int j = 0; j < 32; ++j)
        x[j] -= carry * L[j];
    for (int i = 0; i < 32; ++i)
    {
        x[i + 1] += x[i] >> 8;
        r[i] = (uint8_t)(x[i] & 255);
    }
}

int asi_ed25519_verify(const uint8_t sig[64],
                       const uint8_t *msg,
                       size_t msg_len,
                       const uint8_t pk[32])
{
    if (!sig || !pk || (!msg && msg_len > 0))
        return -1;
    if (!sc_is_canonical(sig + 32))
        return -1;

    gf p[4], q[4], b_pt[4];
    if (ed_unpackneg(q, pk) != 0)
        return -1;

    /* h = SHA512(R || A || M) mod L */
    AsiSha512Ctx sha;
    uint8_t h[64];
    asi_sha512_init(&sha);
    asi_sha512_update(&sha, sig, 32);
    asi_sha512_update(&sha, pk, 32);
    if (msg_len > 0)
        asi_sha512_update(&sha, msg, msg_len);
    asi_sha512_final(&sha, h);
    sc_reduce(h);

    /* p = h*(-A) */
    ed_scalarmult(p, q, h);

    /* q = S*B */
    gf_set(b_pt[0], ED_X);
    gf_set(b_pt[1], ED_Y);
    gf_set(b_pt[2], GF_1);
    gf_mul(b_pt[3], ED_X, ED_Y);
    ed_scalarmult(q, b_pt, sig + 32);

    /* Verify R == S*B - h*A (which equals p + q since q = -A was negated on unpack) */
    ed_add(p, q);
    uint8_t r_check[32];
    ed_pack(r_check, p);
    return asi_ct_memeq(sig, r_check, 32) ? 0 : -1;
}
