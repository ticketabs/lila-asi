#include "asi_state_centric.h"
#include "asi_sc_internal.h"

#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

/* =========================================================================
 * 1. Self-Contained FIPS 180-4 SHA-256 (Zero External Dependencies)
 * ========================================================================= */

typedef struct {
    uint32_t state[8];
    uint64_t bitlen;
    uint8_t  data[64];
    size_t   datalen;
} ScSha256Ctx;

static const uint32_t k_sha256[64] = {
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

static inline uint32_t sc_rotr32(uint32_t x, uint32_t n)
{
    return (x >> n) | (x << (32u - n));
}

static void sc_sha256_transform(ScSha256Ctx *ctx, const uint8_t block[64])
{
    uint32_t m[64];
    for (int i = 0; i < 16; ++i)
    {
        m[i] = ((uint32_t)block[i * 4] << 24) |
               ((uint32_t)block[i * 4 + 1] << 16) |
               ((uint32_t)block[i * 4 + 2] << 8) |
               ((uint32_t)block[i * 4 + 3]);
    }
    for (int i = 16; i < 64; ++i)
    {
        uint32_t s0 = sc_rotr32(m[i - 15], 7) ^ sc_rotr32(m[i - 15], 18) ^ (m[i - 15] >> 3);
        uint32_t s1 = sc_rotr32(m[i - 2], 17) ^ sc_rotr32(m[i - 2], 19) ^ (m[i - 2] >> 10);
        m[i] = m[i - 16] + s0 + m[i - 7] + s1;
    }

    uint32_t a = ctx->state[0], b = ctx->state[1], c = ctx->state[2], d = ctx->state[3];
    uint32_t e = ctx->state[4], f = ctx->state[5], g = ctx->state[6], h = ctx->state[7];

    for (int i = 0; i < 64; ++i)
    {
        uint32_t S1 = sc_rotr32(e, 6) ^ sc_rotr32(e, 11) ^ sc_rotr32(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t t1 = h + S1 + ch + k_sha256[i] + m[i];
        uint32_t S0 = sc_rotr32(a, 2) ^ sc_rotr32(a, 13) ^ sc_rotr32(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = S0 + maj;

        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    ctx->state[0] += a; ctx->state[1] += b; ctx->state[2] += c; ctx->state[3] += d;
    ctx->state[4] += e; ctx->state[5] += f; ctx->state[6] += g; ctx->state[7] += h;
}

static void sc_sha256_init(ScSha256Ctx *ctx)
{
    ctx->datalen = 0;
    ctx->bitlen = 0;
    ctx->state[0] = 0x6a09e667u; ctx->state[1] = 0xbb67ae85u;
    ctx->state[2] = 0x3c6ef372u; ctx->state[3] = 0xa54ff53au;
    ctx->state[4] = 0x510e527fu; ctx->state[5] = 0x9b05688cu;
    ctx->state[6] = 0x1f83d9abu; ctx->state[7] = 0x5be0cd19u;
}

static void sc_sha256_update(ScSha256Ctx *ctx, const uint8_t *data, size_t len)
{
    for (size_t i = 0; i < len; ++i)
    {
        ctx->data[ctx->datalen++] = data[i];
        if (ctx->datalen == 64)
        {
            sc_sha256_transform(ctx, ctx->data);
            ctx->bitlen += 512;
            ctx->datalen = 0;
        }
    }
}

static void sc_sha256_final(ScSha256Ctx *ctx, uint8_t hash[32])
{
    size_t i = ctx->datalen;
    ctx->bitlen += (uint64_t)ctx->datalen * 8ULL;

    ctx->data[i++] = 0x80;
    if (i > 56)
    {
        while (i < 64) ctx->data[i++] = 0x00;
        sc_sha256_transform(ctx, ctx->data);
        i = 0;
    }
    while (i < 56) ctx->data[i++] = 0x00;

    for (int j = 0; j < 8; ++j)
    {
        ctx->data[63 - j] = (uint8_t)(ctx->bitlen >> (j * 8));
    }
    sc_sha256_transform(ctx, ctx->data);

    for (int j = 0; j < 8; ++j)
    {
        hash[j * 4]     = (uint8_t)(ctx->state[j] >> 24);
        hash[j * 4 + 1] = (uint8_t)(ctx->state[j] >> 16);
        hash[j * 4 + 2] = (uint8_t)(ctx->state[j] >> 8);
        hash[j * 4 + 3] = (uint8_t)(ctx->state[j]);
    }
}

void lila_asi_sha256(const void *data, size_t len, uint8_t out_hash[32])
{
    ScSha256Ctx ctx;
    sc_sha256_init(&ctx);
    if (data && len > 0) sc_sha256_update(&ctx, (const uint8_t *)data, len);
    sc_sha256_final(&ctx, out_hash);
}

/* =========================================================================
 * 2. CRC-64 (ECMA-182, matches asi_arch.c) & CRC-32 (IEEE, for WAL lines)
 * ========================================================================= */

static uint64_t g_sc_crc64_tab[256];
static uint32_t g_sc_crc32_tab[256];
static pthread_once_t g_sc_crc_once = PTHREAD_ONCE_INIT;

static void sc_crc_init(void)
{
    for (uint64_t i = 0; i < 256; ++i)
    {
        uint64_t crc = i << 56;
        for (int b = 0; b < 8; ++b)
        {
            crc = (crc & 0x8000000000000000ULL) ? ((crc << 1) ^ 0x42F0E1EBA9EA3693ULL) : (crc << 1);
        }
        g_sc_crc64_tab[i] = crc;
    }
    for (uint32_t i = 0; i < 256; ++i)
    {
        uint32_t c = i;
        for (int b = 0; b < 8; ++b)
        {
            c = (c & 1u) ? ((c >> 1) ^ 0xEDB88320u) : (c >> 1);
        }
        g_sc_crc32_tab[i] = c;
    }
}

static uint64_t sc_crc64_update(uint64_t crc, const uint8_t *data, size_t len)
{
    pthread_once(&g_sc_crc_once, sc_crc_init);
    for (size_t i = 0; i < len; ++i)
    {
        crc = (crc << 8) ^ g_sc_crc64_tab[((crc >> 56) ^ data[i]) & 0xFFu];
    }
    return crc;
}

static uint32_t sc_crc32(const uint8_t *data, size_t len)
{
    pthread_once(&g_sc_crc_once, sc_crc_init);
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; ++i)
    {
        c = g_sc_crc32_tab[(c ^ data[i]) & 0xFFu] ^ (c >> 8);
    }
    return c ^ 0xFFFFFFFFu;
}

/* =========================================================================
 * 3. Copy-on-Write Multi-Slot State Tree (ASITREE1) & Provenance
 * ========================================================================= */

typedef struct {
    uint32_t  refcount;
    uint32_t  n_used;
    uint16_t *k_fp16;
    uint16_t *v_fp16;
} ScKvPage;

typedef struct {
    int                 in_use;
    uint32_t            cp_id;
    uint32_t            parent_cp_id;
    uint64_t            rng_state;
    AsiBranchProvenance prov;
    char                label[64];
    float              *conv_state;
    float              *ssm_state;
    uint32_t            n_fa;
    uint32_t            n_pages_per_layer;
    ScKvPage          **pages;     /* [n_fa * n_pages_per_layer] */
    int32_t            *tokens;    /* optional [prov.n_tokens] */
    int                 has_tokens;
} ScCheckpointSlot;

struct AsiStateTree {
    LilaAsiSession     *session;
    uint32_t            next_cp_id;
    uint32_t            next_branch_id;
    uint32_t            next_state_version;
    uint32_t            active_parent_cp_id;
    AsiBranchProvenance active_prov;
    uint64_t            cow_reuses;
    uint64_t            cow_allocs;
    ScCheckpointSlot    slots[ASI_STATE_TREE_MAX_SLOTS];
};

static ScKvPage *sc_kv_page_alloc(uint32_t page_tokens, size_t kv_dim, uint32_t n_used)
{
    ScKvPage *p = (ScKvPage *)calloc(1, sizeof(ScKvPage));
    if (!p) return NULL;
    size_t elems = (size_t)page_tokens * kv_dim;
    p->k_fp16 = (uint16_t *)calloc(elems, sizeof(uint16_t));
    p->v_fp16 = (uint16_t *)calloc(elems, sizeof(uint16_t));
    if (!p->k_fp16 || !p->v_fp16)
    {
        free(p->k_fp16);
        free(p->v_fp16);
        free(p);
        return NULL;
    }
    p->refcount = 1;
    p->n_used = n_used;
    return p;
}

static void sc_kv_page_release(ScKvPage *p)
{
    if (!p) return;
    if (p->refcount > 1)
    {
        p->refcount--;
        return;
    }
    free(p->k_fp16);
    free(p->v_fp16);
    free(p);
}

static void sc_slot_clear(ScCheckpointSlot *slot)
{
    if (!slot) return;
    free(slot->conv_state);
    free(slot->ssm_state);
    if (slot->pages)
    {
        size_t total = (size_t)slot->n_fa * slot->n_pages_per_layer;
        for (size_t i = 0; i < total; ++i)
        {
            sc_kv_page_release(slot->pages[i]);
        }
        free(slot->pages);
    }
    free(slot->tokens);
    memset(slot, 0, sizeof(*slot));
}

static ScCheckpointSlot *sc_find_slot(AsiStateTree *tree, uint32_t cp_id)
{
    if (!tree || cp_id == ASI_CP_INVALID) return NULL;
    for (uint32_t i = 0; i < ASI_STATE_TREE_MAX_SLOTS; ++i)
    {
        if (tree->slots[i].in_use && tree->slots[i].cp_id == cp_id)
            return &tree->slots[i];
    }
    return NULL;
}

static const ScCheckpointSlot *sc_find_slot_const(const AsiStateTree *tree, uint32_t cp_id)
{
    if (!tree || cp_id == ASI_CP_INVALID) return NULL;
    for (uint32_t i = 0; i < ASI_STATE_TREE_MAX_SLOTS; ++i)
    {
        if (tree->slots[i].in_use && tree->slots[i].cp_id == cp_id)
            return &tree->slots[i];
    }
    return NULL;
}

static ScCheckpointSlot *sc_alloc_free_slot(AsiStateTree *tree)
{
    if (!tree) return NULL;
    for (uint32_t i = 0; i < ASI_STATE_TREE_MAX_SLOTS; ++i)
    {
        if (!tree->slots[i].in_use)
            return &tree->slots[i];
    }
    return NULL;
}

static int sc_is_ancestor(const AsiStateTree *tree, uint32_t maybe_anc, uint32_t node_cp_id)
{
    uint32_t cur = node_cp_id;
    uint32_t hops = 0;
    while (cur != ASI_CP_INVALID && hops <= ASI_STATE_TREE_MAX_SLOTS)
    {
        if (cur == maybe_anc) return 1;
        const ScCheckpointSlot *s = sc_find_slot_const(tree, cur);
        if (!s) break;
        cur = s->parent_cp_id;
        hops++;
    }
    return 0;
}

AsiStateTree *lila_asi_sc_state_tree_create(LilaAsiSession *session)
{
    AsiStateTree *tree = (AsiStateTree *)calloc(1, sizeof(AsiStateTree));
    if (!tree) return NULL;
    tree->session = session;
    tree->next_cp_id = 1;
    tree->next_branch_id = 2;
    tree->next_state_version = 2;
    tree->active_parent_cp_id = ASI_CP_INVALID;
    tree->active_prov.branch_id = 1;
    tree->active_prov.parent_branch_id = 0;
    tree->active_prov.state_version = 1;
    tree->active_prov.n_tokens = 0;
    tree->active_prov.status = ASI_BRANCH_ACTIVE;
    lila_asi_sha256(NULL, 0, tree->active_prov.token_trajectory_sha256);
    return tree;
}

void lila_asi_sc_state_tree_free(AsiStateTree *tree)
{
    if (!tree) return;
    for (uint32_t i = 0; i < ASI_STATE_TREE_MAX_SLOTS; ++i)
    {
        if (tree->slots[i].in_use)
            sc_slot_clear(&tree->slots[i]);
    }
    free(tree);
}

int lila_asi_cp_fork(LilaAsiSession *session,
                     uint32_t n_tokens,
                     uint64_t rng_state,
                     const char *label,
                     uint32_t *out_cp_id)
{
    if (!session || n_tokens == 0) return -1;
    AsiArchSession *arch_s = lila_asi_sc_get_arch_session(session);
    if (!arch_s) return -1;

    Qwen35StateGeom g;
    if (qwen35_sc_get_geometry(arch_s, &g) != 0) return -1;
    if (n_tokens > g.max_ctx) return -1;

    AsiStateTree *tree = lila_asi_sc_ensure_state_tree(session);
    if (!tree) return -1;

    ScCheckpointSlot *slot = sc_alloc_free_slot(tree);
    if (!slot) return -1; /* All 16 slots occupied */

    const ScCheckpointSlot *parent_slot = sc_find_slot_const(tree, tree->active_parent_cp_id);
    uint32_t dirty_from = qwen35_sc_get_kv_dirty_from(arch_s);

    uint32_t page_tokens = ASI_STATE_TREE_KV_PAGE_TOKENS;
    uint32_t n_pages = (n_tokens + page_tokens - 1u) / page_tokens;

    float *conv_buf = (g.conv_bytes > 0) ? (float *)malloc((size_t)g.conv_bytes) : NULL;
    float *ssm_buf  = (g.ssm_bytes > 0)  ? (float *)malloc((size_t)g.ssm_bytes)  : NULL;
    ScKvPage **pages = (g.n_fa > 0 && n_pages > 0)
                           ? (ScKvPage **)calloc((size_t)g.n_fa * n_pages, sizeof(ScKvPage *))
                           : NULL;
    if ((g.conv_bytes > 0 && !conv_buf) ||
        (g.ssm_bytes > 0 && !ssm_buf) ||
        (g.n_fa > 0 && n_pages > 0 && !pages))
    {
        free(conv_buf);
        free(ssm_buf);
        free(pages);
        return -1;
    }

    if (qwen35_sc_capture_deltanet(arch_s, conv_buf, ssm_buf) != 0)
    {
        free(conv_buf);
        free(ssm_buf);
        free(pages);
        return -1;
    }

    /* Capture KV pages with Copy-on-Write sharing against parent_slot */
    for (uint32_t f = 0; f < g.n_fa; ++f)
    {
        for (uint32_t p = 0; p < n_pages; ++p)
        {
            uint32_t tok_start = p * page_tokens;
            uint32_t take = n_tokens - tok_start;
            if (take > page_tokens) take = page_tokens;
            uint32_t tok_end = tok_start + take;

            ScKvPage *page = NULL;
            if (parent_slot &&
                parent_slot->n_fa == g.n_fa &&
                p < parent_slot->n_pages_per_layer &&
                tok_end <= parent_slot->prov.n_tokens &&
                tok_end <= dirty_from)
            {
                ScKvPage *parent_p = parent_slot->pages[(size_t)f * parent_slot->n_pages_per_layer + p];
                if (parent_p && parent_p->n_used == take)
                {
                    parent_p->refcount++;
                    page = parent_p;
                    tree->cow_reuses++;
                }
            }

            if (!page)
            {
                page = sc_kv_page_alloc(page_tokens, g.kv_dim, take);
                if (!page ||
                    qwen35_sc_capture_kv_page(arch_s, f, p, page_tokens, take,
                                              page->k_fp16, page->v_fp16) != 0)
                {
                    sc_kv_page_release(page);
                    size_t total = (size_t)g.n_fa * n_pages;
                    for (size_t j = 0; j < total; ++j) sc_kv_page_release(pages[j]);
                    free(pages);
                    free(conv_buf);
                    free(ssm_buf);
                    return -1;
                }
                tree->cow_allocs++;
            }
            pages[(size_t)f * n_pages + p] = page;
        }
    }

    /* Capture token trajectory if history is valid from 0..n_tokens */
    const int32_t *hist = NULL;
    uint32_t hist_len = 0, hist_valid_from = 0;
    int32_t *tok_copy = NULL;
    int has_tokens = 0;
    uint8_t traj_sha[32];
    if (qwen35_sc_get_history(arch_s, &hist, &hist_len, &hist_valid_from) == 0 &&
        hist && hist_valid_from == 0 && hist_len >= n_tokens)
    {
        tok_copy = (int32_t *)malloc((size_t)n_tokens * sizeof(int32_t));
        if (tok_copy)
        {
            memcpy(tok_copy, hist, (size_t)n_tokens * sizeof(int32_t));
            has_tokens = 1;
        }
        lila_asi_sha256(hist, (size_t)n_tokens * sizeof(int32_t), traj_sha);
    }
    else
    {
        lila_asi_sha256(NULL, 0, traj_sha);
    }

    /* Update provenance: if extending past parent checkpoint on same branch_id, mint child branch_id */
    if (parent_slot && parent_slot->prov.branch_id == tree->active_prov.branch_id &&
        n_tokens > parent_slot->prov.n_tokens)
    {
        tree->active_prov.parent_branch_id = parent_slot->prov.branch_id;
        tree->active_prov.branch_id = tree->next_branch_id++;
    }
    tree->active_prov.state_version = tree->next_state_version++;
    tree->active_prov.n_tokens = n_tokens;
    tree->active_prov.status = ASI_BRANCH_ACTIVE;
    memcpy(tree->active_prov.token_trajectory_sha256, traj_sha, 32);

    uint32_t cp_id = tree->next_cp_id++;
    memset(slot, 0, sizeof(*slot));
    slot->in_use = 1;
    slot->cp_id = cp_id;
    slot->parent_cp_id = tree->active_parent_cp_id;
    slot->rng_state = rng_state;
    slot->prov = tree->active_prov;
    if (label && label[0] != '\0')
    {
        snprintf(slot->label, sizeof(slot->label), "%s", label);
    }
    else
    {
        snprintf(slot->label, sizeof(slot->label), "cp_%u", cp_id);
    }
    slot->conv_state = conv_buf;
    slot->ssm_state = ssm_buf;
    slot->n_fa = g.n_fa;
    slot->n_pages_per_layer = n_pages;
    slot->pages = pages;
    slot->tokens = tok_copy;
    slot->has_tokens = has_tokens;

    tree->active_parent_cp_id = cp_id;
    qwen35_sc_mark_kv_clean(arch_s, n_tokens);

    if (out_cp_id) *out_cp_id = cp_id;
    return 0;
}

int lila_asi_cp_restore(LilaAsiSession *session,
                        uint32_t cp_id,
                        uint32_t *out_n_tokens,
                        uint64_t *out_rng_state)
{
    if (!session || cp_id == ASI_CP_INVALID) return -1;
    AsiArchSession *arch_s = lila_asi_sc_get_arch_session(session);
    AsiStateTree *tree = lila_asi_sc_ensure_state_tree(session);
    if (!arch_s || !tree) return -1;

    ScCheckpointSlot *target = sc_find_slot(tree, cp_id);
    if (!target) return -1;

    /* Mark any active rolled-back branch checkpoints as ABANDONED */
    uint32_t cur = tree->active_parent_cp_id;
    uint32_t hops = 0;
    while (cur != ASI_CP_INVALID && cur != cp_id && hops <= ASI_STATE_TREE_MAX_SLOTS)
    {
        ScCheckpointSlot *cs = sc_find_slot(tree, cur);
        if (!cs) break;
        if (!sc_is_ancestor(tree, cur, cp_id) && cs->prov.status == ASI_BRANCH_ACTIVE)
        {
            cs->prov.status = ASI_BRANCH_ABANDONED;
        }
        cur = cs->parent_cp_id;
        hops++;
    }
    for (uint32_t i = 0; i < ASI_STATE_TREE_MAX_SLOTS; ++i)
    {
        if (tree->slots[i].in_use &&
            tree->slots[i].cp_id != cp_id &&
            tree->slots[i].prov.branch_id == tree->active_prov.branch_id &&
            tree->slots[i].prov.status == ASI_BRANCH_ACTIVE &&
            !sc_is_ancestor(tree, tree->slots[i].cp_id, cp_id))
        {
            tree->slots[i].prov.status = ASI_BRANCH_ABANDONED;
        }
    }

    size_t total_pages = (size_t)target->n_fa * target->n_pages_per_layer;
    const uint16_t **k_ptrs = NULL;
    const uint16_t **v_ptrs = NULL;
    if (total_pages > 0)
    {
        k_ptrs = (const uint16_t **)malloc(total_pages * sizeof(const uint16_t *));
        v_ptrs = (const uint16_t **)malloc(total_pages * sizeof(const uint16_t *));
        if (!k_ptrs || !v_ptrs)
        {
            free(k_ptrs);
            free(v_ptrs);
            return -1;
        }
        for (size_t i = 0; i < total_pages; ++i)
        {
            k_ptrs[i] = target->pages[i]->k_fp16;
            v_ptrs[i] = target->pages[i]->v_fp16;
        }
    }

    int rc = qwen35_sc_restore_state(arch_s,
                                     target->prov.n_tokens,
                                     target->conv_state,
                                     target->ssm_state,
                                     k_ptrs,
                                     v_ptrs,
                                     target->n_pages_per_layer,
                                     ASI_STATE_TREE_KV_PAGE_TOKENS,
                                     target->tokens,
                                     target->has_tokens);
    free(k_ptrs);
    free(v_ptrs);
    if (rc != 0) return -1;

    /* Mint fresh child branch_id for the newly restored active session */
    tree->active_parent_cp_id = target->cp_id;
    tree->active_prov.parent_branch_id = target->prov.branch_id;
    tree->active_prov.branch_id = tree->next_branch_id++;
    tree->active_prov.state_version = tree->next_state_version++;
    tree->active_prov.n_tokens = target->prov.n_tokens;
    tree->active_prov.status = ASI_BRANCH_ACTIVE;
    memcpy(tree->active_prov.token_trajectory_sha256, target->prov.token_trajectory_sha256, 32);

    if (out_n_tokens) *out_n_tokens = target->prov.n_tokens;
    if (out_rng_state) *out_rng_state = target->rng_state;
    return 0;
}

int lila_asi_cp_delete(LilaAsiSession *session, uint32_t cp_id)
{
    if (!session || cp_id == ASI_CP_INVALID) return -1;
    AsiStateTree *tree = lila_asi_sc_ensure_state_tree(session);
    if (!tree) return -1;
    ScCheckpointSlot *slot = sc_find_slot(tree, cp_id);
    if (!slot) return -1;

    uint32_t parent_id = slot->parent_cp_id;
    sc_slot_clear(slot);
    if (tree->active_parent_cp_id == cp_id)
    {
        tree->active_parent_cp_id = parent_id;
    }
    return 0;
}

int lila_asi_cp_get_info(const LilaAsiSession *session,
                         uint32_t cp_id,
                         AsiCheckpointInfo *out_info)
{
    if (!session || !out_info || cp_id == ASI_CP_INVALID) return -1;
    const AsiStateTree *tree = lila_asi_sc_get_state_tree(session);
    if (!tree) return -1;
    const ScCheckpointSlot *slot = sc_find_slot_const(tree, cp_id);
    if (!slot) return -1;

    memset(out_info, 0, sizeof(*out_info));
    out_info->cp_id = slot->cp_id;
    out_info->parent_cp_id = slot->parent_cp_id;
    out_info->rng_state = slot->rng_state;
    out_info->prov = slot->prov;
    memcpy(out_info->label, slot->label, sizeof(out_info->label));
    return 0;
}

int lila_asi_session_get_provenance(const LilaAsiSession *session,
                                    AsiBranchProvenance *out_prov)
{
    if (!session || !out_prov) return -1;
    const AsiStateTree *tree = lila_asi_sc_get_state_tree(session);
    if (!tree)
    {
        memset(out_prov, 0, sizeof(*out_prov));
        out_prov->branch_id = 1;
        out_prov->state_version = 1;
        out_prov->status = ASI_BRANCH_ACTIVE;
        lila_asi_sha256(NULL, 0, out_prov->token_trajectory_sha256);
        return 0;
    }
    *out_prov = tree->active_prov;
    return 0;
}

int lila_asi_cp_commit(LilaAsiSession *session, uint32_t cp_id)
{
    if (!session || cp_id == ASI_CP_INVALID) return -1;
    AsiStateTree *tree = lila_asi_sc_ensure_state_tree(session);
    if (!tree) return -1;
    ScCheckpointSlot *slot = sc_find_slot(tree, cp_id);
    if (!slot) return -1;

    /* Provenance invariant: ABANDONED branches can NEVER be committed! */
    if (slot->prov.status == ASI_BRANCH_ABANDONED)
    {
        return -1;
    }
    slot->prov.status = ASI_BRANCH_COMMITTED;
    if (tree->active_prov.branch_id == slot->prov.branch_id)
    {
        tree->active_prov.status = ASI_BRANCH_COMMITTED;
    }
    return 0;
}

int lila_asi_cp_abandon(LilaAsiSession *session, uint32_t cp_id)
{
    if (!session || cp_id == ASI_CP_INVALID) return -1;
    AsiStateTree *tree = lila_asi_sc_ensure_state_tree(session);
    if (!tree) return -1;
    ScCheckpointSlot *slot = sc_find_slot(tree, cp_id);
    if (!slot) return -1;
    if (slot->prov.status == ASI_BRANCH_COMMITTED) return -1;
    slot->prov.status = ASI_BRANCH_ABANDONED;
    return 0;
}

int lila_asi_cp_cow_stats(const LilaAsiSession *session, AsiCowStats *out_stats)
{
    if (!session || !out_stats) return -1;
    memset(out_stats, 0, sizeof(*out_stats));
    out_stats->max_slots = ASI_STATE_TREE_MAX_SLOTS;
    const AsiStateTree *tree = lila_asi_sc_get_state_tree(session);
    if (!tree) return 0;

    out_stats->cow_reuses = tree->cow_reuses;
    out_stats->cow_allocs = tree->cow_allocs;

    /* Collect unique physical page pointers across all live slots */
    size_t cap = 64;
    const ScKvPage **seen = (const ScKvPage **)malloc(cap * sizeof(const ScKvPage *));
    if (!seen) return -1;
    size_t n_seen = 0;

    for (uint32_t i = 0; i < ASI_STATE_TREE_MAX_SLOTS; ++i)
    {
        const ScCheckpointSlot *slot = &tree->slots[i];
        if (!slot->in_use) continue;
        out_stats->live_slots++;
        size_t total = (size_t)slot->n_fa * slot->n_pages_per_layer;
        out_stats->logical_kv_pages += (uint32_t)total;
        for (size_t j = 0; j < total; ++j)
        {
            const ScKvPage *p = slot->pages[j];
            if (!p) continue;
            int found = 0;
            for (size_t s = 0; s < n_seen; ++s)
            {
                if (seen[s] == p) { found = 1; break; }
            }
            if (!found)
            {
                if (n_seen == cap)
                {
                    cap *= 2;
                    const ScKvPage **ns = (const ScKvPage **)realloc(seen, cap * sizeof(const ScKvPage *));
                    if (!ns) { free(seen); return -1; }
                    seen = ns;
                }
                seen[n_seen++] = p;
                if (p->refcount > 1) out_stats->shared_kv_pages++;
            }
        }
    }
    out_stats->physical_kv_pages = (uint32_t)n_seen;
    free(seen);
    return 0;
}

int lila_asi_session_state_sha256(const LilaAsiSession *session,
                                  uint32_t n_tokens,
                                  uint8_t out_sha256[32])
{
    if (!session || !out_sha256 || n_tokens == 0) return -1;
    const AsiArchSession *arch_s = lila_asi_sc_get_arch_session(session);
    if (!arch_s) return -1;

    Qwen35StateGeom g;
    if (qwen35_sc_get_geometry(arch_s, &g) != 0) return -1;
    if (n_tokens > g.max_ctx) return -1;

    float *conv_buf = (g.conv_bytes > 0) ? (float *)malloc((size_t)g.conv_bytes) : NULL;
    float *ssm_buf  = (g.ssm_bytes > 0)  ? (float *)malloc((size_t)g.ssm_bytes)  : NULL;
    uint32_t page_tokens = ASI_STATE_TREE_KV_PAGE_TOKENS;
    uint16_t *k_tmp = (g.n_fa > 0) ? (uint16_t *)malloc((size_t)page_tokens * g.kv_dim * sizeof(uint16_t)) : NULL;
    uint16_t *v_tmp = (g.n_fa > 0) ? (uint16_t *)malloc((size_t)page_tokens * g.kv_dim * sizeof(uint16_t)) : NULL;

    if ((g.conv_bytes > 0 && !conv_buf) || (g.ssm_bytes > 0 && !ssm_buf) ||
        (g.n_fa > 0 && (!k_tmp || !v_tmp)))
    {
        free(conv_buf); free(ssm_buf); free(k_tmp); free(v_tmp);
        return -1;
    }

    int rc = qwen35_sc_capture_deltanet(arch_s, conv_buf, ssm_buf);
    if (rc == 0)
    {
        ScSha256Ctx sha;
        sc_sha256_init(&sha);
        if (g.conv_bytes > 0) sc_sha256_update(&sha, (const uint8_t *)conv_buf, (size_t)g.conv_bytes);
        if (g.ssm_bytes > 0)  sc_sha256_update(&sha, (const uint8_t *)ssm_buf,  (size_t)g.ssm_bytes);

        uint32_t n_pages = (n_tokens + page_tokens - 1u) / page_tokens;
        for (uint32_t f = 0; f < g.n_fa && rc == 0; ++f)
        {
            for (uint32_t p = 0; p < n_pages; ++p)
            {
                uint32_t tok_start = p * page_tokens;
                uint32_t take = n_tokens - tok_start;
                if (take > page_tokens) take = page_tokens;
                rc = qwen35_sc_capture_kv_page(arch_s, f, p, page_tokens, take, k_tmp, v_tmp);
                if (rc != 0) break;
                size_t bytes = (size_t)take * g.kv_dim * sizeof(uint16_t);
                sc_sha256_update(&sha, (const uint8_t *)k_tmp, bytes);
                sc_sha256_update(&sha, (const uint8_t *)v_tmp, bytes);
            }
        }
        if (rc == 0) sc_sha256_final(&sha, out_sha256);
    }

    free(conv_buf);
    free(ssm_buf);
    free(k_tmp);
    free(v_tmp);
    return rc;
}

/* =========================================================================
 * 4. Atomic ASITREE1 Persistence (save_file / load_file with CRC-64)
 * ========================================================================= */

int lila_asi_cp_save_file(const LilaAsiSession *session,
                          uint32_t cp_id,
                          const uint8_t key_hash[32],
                          const char *filepath)
{
    if (!session || cp_id == ASI_CP_INVALID || !key_hash || !filepath || filepath[0] == '\0')
        return -1;
    const AsiArchSession *arch_s = lila_asi_sc_get_arch_session(session);
    const AsiStateTree *tree = lila_asi_sc_get_state_tree(session);
    if (!arch_s || !tree) return -1;

    const ScCheckpointSlot *slot = sc_find_slot_const(tree, cp_id);
    if (!slot) return -1;

    Qwen35StateGeom g;
    if (qwen35_sc_get_geometry(arch_s, &g) != 0) return -1;

    uint32_t n_tokens = slot->prov.n_tokens;
    uint64_t k_bytes = (uint64_t)g.n_fa * (uint64_t)n_tokens * (uint64_t)g.kv_dim * sizeof(uint16_t);
    uint64_t v_bytes = k_bytes;
    uint64_t token_bytes = (slot->has_tokens && slot->tokens) ? (uint64_t)n_tokens * sizeof(int32_t) : 0ULL;

    AsiTreeFileHeader hdr;
    memset(&hdr, 0, sizeof(hdr));
    memcpy(hdr.magic, ASI_STATE_TREE_MAGIC, 8);
    hdr.numerics_version = LILA_ASI_NUMERICS_VERSION;
    hdr.arch_id = g.arch_id;
    memcpy(hdr.cart_identity_hash, g.cart_identity_hash, 24);
    memcpy(hdr.key_hash, key_hash, 32);
    hdr.fp32_ref = g.fp32_ref;
    hdr.cp_id = slot->cp_id;
    hdr.parent_cp_id = slot->parent_cp_id;
    hdr.kv_page_tokens = ASI_STATE_TREE_KV_PAGE_TOKENS;
    hdr.rng_state = slot->rng_state;
    hdr.prov = slot->prov;
    hdr.n_dn = g.n_dn;
    hdr.n_fa = g.n_fa;
    hdr.conv_bytes = g.conv_bytes;
    hdr.ssm_bytes = g.ssm_bytes;
    hdr.k_bytes = k_bytes;
    hdr.v_bytes = v_bytes;
    hdr.token_bytes = token_bytes;

    /* Compute payload CRC-64 in exact serialization order */
    uint64_t pcrc = 0;
    if (g.conv_bytes > 0) pcrc = sc_crc64_update(pcrc, (const uint8_t *)slot->conv_state, (size_t)g.conv_bytes);
    if (g.ssm_bytes > 0)  pcrc = sc_crc64_update(pcrc, (const uint8_t *)slot->ssm_state,  (size_t)g.ssm_bytes);

    uint32_t page_tokens = ASI_STATE_TREE_KV_PAGE_TOKENS;
    for (uint32_t f = 0; f < slot->n_fa; ++f)
    {
        for (uint32_t p = 0; p < slot->n_pages_per_layer; ++p)
        {
            uint32_t tok_start = p * page_tokens;
            if (tok_start >= n_tokens) break;
            uint32_t take = n_tokens - tok_start;
            if (take > page_tokens) take = page_tokens;
            const ScKvPage *pg = slot->pages[(size_t)f * slot->n_pages_per_layer + p];
            size_t bytes = (size_t)take * g.kv_dim * sizeof(uint16_t);
            pcrc = sc_crc64_update(pcrc, (const uint8_t *)pg->k_fp16, bytes);
        }
    }
    for (uint32_t f = 0; f < slot->n_fa; ++f)
    {
        for (uint32_t p = 0; p < slot->n_pages_per_layer; ++p)
        {
            uint32_t tok_start = p * page_tokens;
            if (tok_start >= n_tokens) break;
            uint32_t take = n_tokens - tok_start;
            if (take > page_tokens) take = page_tokens;
            const ScKvPage *pg = slot->pages[(size_t)f * slot->n_pages_per_layer + p];
            size_t bytes = (size_t)take * g.kv_dim * sizeof(uint16_t);
            pcrc = sc_crc64_update(pcrc, (const uint8_t *)pg->v_fp16, bytes);
        }
    }
    if (token_bytes > 0)
    {
        pcrc = sc_crc64_update(pcrc, (const uint8_t *)slot->tokens, (size_t)token_bytes);
    }

    hdr.payload_crc64 = pcrc;
    hdr.header_crc64 = asi_v2_crc64((const uint8_t *)&hdr, offsetof(AsiTreeFileHeader, header_crc64));

    size_t flen = strlen(filepath);
    size_t tcap = flen + 64;
    char *tmp_path = (char *)malloc(tcap);
    if (!tmp_path) return -1;
    int tn = snprintf(tmp_path, tcap, "%s.tmp.%ld", filepath, (long)getpid());
    if (tn <= 0 || (size_t)tn >= tcap) { free(tmp_path); return -1; }

    FILE *fp = fopen(tmp_path, "wb");
    if (!fp) { free(tmp_path); return -1; }

    int ok = 1;
    if (fwrite(&hdr, 1, sizeof(hdr), fp) != sizeof(hdr)) ok = 0;
    if (ok && g.conv_bytes > 0 && fwrite(slot->conv_state, 1, (size_t)g.conv_bytes, fp) != (size_t)g.conv_bytes) ok = 0;
    if (ok && g.ssm_bytes > 0  && fwrite(slot->ssm_state,  1, (size_t)g.ssm_bytes,  fp) != (size_t)g.ssm_bytes)  ok = 0;

    for (uint32_t f = 0; ok && f < slot->n_fa; ++f)
    {
        for (uint32_t p = 0; ok && p < slot->n_pages_per_layer; ++p)
        {
            uint32_t tok_start = p * page_tokens;
            if (tok_start >= n_tokens) break;
            uint32_t take = n_tokens - tok_start;
            if (take > page_tokens) take = page_tokens;
            const ScKvPage *pg = slot->pages[(size_t)f * slot->n_pages_per_layer + p];
            size_t bytes = (size_t)take * g.kv_dim * sizeof(uint16_t);
            if (fwrite(pg->k_fp16, 1, bytes, fp) != bytes) ok = 0;
        }
    }
    for (uint32_t f = 0; ok && f < slot->n_fa; ++f)
    {
        for (uint32_t p = 0; ok && p < slot->n_pages_per_layer; ++p)
        {
            uint32_t tok_start = p * page_tokens;
            if (tok_start >= n_tokens) break;
            uint32_t take = n_tokens - tok_start;
            if (take > page_tokens) take = page_tokens;
            const ScKvPage *pg = slot->pages[(size_t)f * slot->n_pages_per_layer + p];
            size_t bytes = (size_t)take * g.kv_dim * sizeof(uint16_t);
            if (fwrite(pg->v_fp16, 1, bytes, fp) != bytes) ok = 0;
        }
    }
    if (ok && token_bytes > 0 && fwrite(slot->tokens, 1, (size_t)token_bytes, fp) != (size_t)token_bytes) ok = 0;

    if (ok && fflush(fp) != 0) ok = 0;
    if (ok)
    {
        int fd = fileno(fp);
        if (fd >= 0 && fsync(fd) != 0 && errno != EINVAL && errno != EROFS) ok = 0;
    }
    if (fclose(fp) != 0) ok = 0;
    if (!ok || rename(tmp_path, filepath) != 0)
    {
        unlink(tmp_path);
        free(tmp_path);
        return -1;
    }
    free(tmp_path);
    return 0;
}

int lila_asi_cp_load_file(LilaAsiSession *session,
                          const uint8_t expected_key_hash[32],
                          const char *filepath,
                          const char *label,
                          uint32_t *out_cp_id)
{
    if (!session || !expected_key_hash || !filepath || filepath[0] == '\0') return -1;
    AsiArchSession *arch_s = lila_asi_sc_get_arch_session(session);
    AsiStateTree *tree = lila_asi_sc_ensure_state_tree(session);
    if (!arch_s || !tree) return -1;

    Qwen35StateGeom g;
    if (qwen35_sc_get_geometry(arch_s, &g) != 0) return -1;

    ScCheckpointSlot *slot = sc_alloc_free_slot(tree);
    if (!slot) return -1;

    FILE *fp = fopen(filepath, "rb");
    if (!fp) return -1;

    AsiTreeFileHeader hdr;
    float *new_conv = NULL;
    float *new_ssm = NULL;
    uint16_t *k_raw = NULL;
    uint16_t *v_raw = NULL;
    int32_t *new_tokens = NULL;
    ScKvPage **pages = NULL;
    uint32_t n_pages = 0;

    if (fread(&hdr, 1, sizeof(hdr), fp) != sizeof(hdr)) goto fail;
    if (memcmp(hdr.magic, ASI_STATE_TREE_MAGIC, 8) != 0) goto fail;

    uint64_t exp_hcrc = asi_v2_crc64((const uint8_t *)&hdr, offsetof(AsiTreeFileHeader, header_crc64));
    if (hdr.header_crc64 != exp_hcrc) goto fail;
    if (hdr.numerics_version != LILA_ASI_NUMERICS_VERSION) goto fail;
    if (hdr.arch_id != g.arch_id) goto fail;
    if (memcmp(hdr.cart_identity_hash, g.cart_identity_hash, 24) != 0) goto fail;
    if (memcmp(hdr.key_hash, expected_key_hash, 32) != 0) goto fail;
    if (hdr.fp32_ref != g.fp32_ref) goto fail;
    if (hdr.kv_page_tokens != ASI_STATE_TREE_KV_PAGE_TOKENS) goto fail;
    if (hdr.prov.n_tokens == 0 || hdr.prov.n_tokens > g.max_ctx) goto fail;

    uint32_t n_tokens = hdr.prov.n_tokens;
    uint64_t exp_kv = (uint64_t)g.n_fa * (uint64_t)n_tokens * (uint64_t)g.kv_dim * sizeof(uint16_t);
    if (hdr.n_dn != g.n_dn || hdr.n_fa != g.n_fa ||
        hdr.conv_bytes != g.conv_bytes || hdr.ssm_bytes != g.ssm_bytes ||
        hdr.k_bytes != exp_kv || hdr.v_bytes != exp_kv)
    {
        goto fail;
    }
    if (hdr.token_bytes != 0 && hdr.token_bytes != (uint64_t)n_tokens * sizeof(int32_t))
    {
        goto fail;
    }

    if (hdr.conv_bytes > 0)
    {
        new_conv = (float *)malloc((size_t)hdr.conv_bytes);
        if (!new_conv || fread(new_conv, 1, (size_t)hdr.conv_bytes, fp) != (size_t)hdr.conv_bytes) goto fail;
    }
    if (hdr.ssm_bytes > 0)
    {
        new_ssm = (float *)malloc((size_t)hdr.ssm_bytes);
        if (!new_ssm || fread(new_ssm, 1, (size_t)hdr.ssm_bytes, fp) != (size_t)hdr.ssm_bytes) goto fail;
    }
    if (hdr.k_bytes > 0)
    {
        k_raw = (uint16_t *)malloc((size_t)hdr.k_bytes);
        if (!k_raw || fread(k_raw, 1, (size_t)hdr.k_bytes, fp) != (size_t)hdr.k_bytes) goto fail;
    }
    if (hdr.v_bytes > 0)
    {
        v_raw = (uint16_t *)malloc((size_t)hdr.v_bytes);
        if (!v_raw || fread(v_raw, 1, (size_t)hdr.v_bytes, fp) != (size_t)hdr.v_bytes) goto fail;
    }
    if (hdr.token_bytes > 0)
    {
        new_tokens = (int32_t *)malloc((size_t)hdr.token_bytes);
        if (!new_tokens || fread(new_tokens, 1, (size_t)hdr.token_bytes, fp) != (size_t)hdr.token_bytes) goto fail;
    }

    uint8_t extra = 0;
    if (fread(&extra, 1, 1, fp) != 0 || ferror(fp)) goto fail;
    fclose(fp);
    fp = NULL;

    uint64_t pcrc = 0;
    if (hdr.conv_bytes > 0)  pcrc = sc_crc64_update(pcrc, (const uint8_t *)new_conv,   (size_t)hdr.conv_bytes);
    if (hdr.ssm_bytes > 0)   pcrc = sc_crc64_update(pcrc, (const uint8_t *)new_ssm,    (size_t)hdr.ssm_bytes);
    if (hdr.k_bytes > 0)     pcrc = sc_crc64_update(pcrc, (const uint8_t *)k_raw,      (size_t)hdr.k_bytes);
    if (hdr.v_bytes > 0)     pcrc = sc_crc64_update(pcrc, (const uint8_t *)v_raw,      (size_t)hdr.v_bytes);
    if (hdr.token_bytes > 0) pcrc = sc_crc64_update(pcrc, (const uint8_t *)new_tokens, (size_t)hdr.token_bytes);
    if (pcrc != hdr.payload_crc64) goto fail;

    /* Reconstruct CoW KV pages from contiguous k_raw / v_raw */
    uint32_t page_tokens = ASI_STATE_TREE_KV_PAGE_TOKENS;
    n_pages = (n_tokens + page_tokens - 1u) / page_tokens;
    if (g.n_fa > 0 && n_pages > 0)
    {
        pages = (ScKvPage **)calloc((size_t)g.n_fa * n_pages, sizeof(ScKvPage *));
        if (!pages) goto fail;
        size_t per_layer_elems = (size_t)n_tokens * g.kv_dim;
        for (uint32_t f = 0; f < g.n_fa; ++f)
        {
            for (uint32_t p = 0; p < n_pages; ++p)
            {
                uint32_t tok_start = p * page_tokens;
                uint32_t take = n_tokens - tok_start;
                if (take > page_tokens) take = page_tokens;
                ScKvPage *pg = sc_kv_page_alloc(page_tokens, g.kv_dim, take);
                if (!pg) goto fail;
                size_t src_off = (size_t)f * per_layer_elems + (size_t)tok_start * g.kv_dim;
                size_t elems = (size_t)take * g.kv_dim;
                memcpy(pg->k_fp16, k_raw + src_off, elems * sizeof(uint16_t));
                memcpy(pg->v_fp16, v_raw + src_off, elems * sizeof(uint16_t));
                pages[(size_t)f * n_pages + p] = pg;
                tree->cow_allocs++;
            }
        }
    }
    free(k_raw);
    free(v_raw);
    k_raw = NULL;
    v_raw = NULL;

    uint32_t cp_id = tree->next_cp_id++;
    if (hdr.prov.branch_id >= tree->next_branch_id)
        tree->next_branch_id = hdr.prov.branch_id + 1;
    if (hdr.prov.state_version >= tree->next_state_version)
        tree->next_state_version = hdr.prov.state_version + 1;

    memset(slot, 0, sizeof(*slot));
    slot->in_use = 1;
    slot->cp_id = cp_id;
    slot->parent_cp_id = hdr.parent_cp_id;
    slot->rng_state = hdr.rng_state;
    slot->prov = hdr.prov;
    if (label && label[0] != '\0')
        snprintf(slot->label, sizeof(slot->label), "%s", label);
    else
        snprintf(slot->label, sizeof(slot->label), "loaded_%u", cp_id);
    slot->conv_state = new_conv;
    slot->ssm_state = new_ssm;
    slot->n_fa = g.n_fa;
    slot->n_pages_per_layer = n_pages;
    slot->pages = pages;
    slot->tokens = new_tokens;
    slot->has_tokens = (new_tokens != NULL) ? 1 : 0;

    if (out_cp_id) *out_cp_id = cp_id;
    return 0;

fail:
    if (fp) fclose(fp);
    free(new_conv);
    free(new_ssm);
    free(k_raw);
    free(v_raw);
    free(new_tokens);
    if (pages)
    {
        size_t total = (size_t)g.n_fa * n_pages;
        for (size_t i = 0; i < total; ++i) sc_kv_page_release(pages[i]);
        free(pages);
    }
    return -1;
}

/* =========================================================================
 * 5. Crash-Recoverable Session WAL over Atomic ASITREE1 Checkpoints
 * ========================================================================= */

struct LilaAsiWal {
    char     wal_dir[512];
    char     session_id[128];
    char     wal_path[700];
    uint8_t  key_hash[32];
    uint64_t next_seq;
    uint32_t next_ckpt_file_id;
};

static int sc_wal_append_line(LilaAsiWal *wal, const char *body)
{
    uint32_t crc = sc_crc32((const uint8_t *)body, strlen(body));
    FILE *fp = fopen(wal->wal_path, "a");
    if (!fp) return -1;
    int ok = (fprintf(fp, "%08x|%s\n", crc, body) > 0);
    if (ok && fflush(fp) != 0) ok = 0;
    if (ok)
    {
        int fd = fileno(fp);
        if (fd >= 0 && fsync(fd) != 0 && errno != EINVAL && errno != EROFS) ok = 0;
    }
    if (fclose(fp) != 0) ok = 0;
    return ok ? 0 : -1;
}

LilaAsiWal *lila_asi_wal_open(const char *wal_dir,
                              const char *session_id,
                              const uint8_t key_hash[32])
{
    if (!wal_dir || !session_id || !key_hash) return NULL;
    if (mkdir(wal_dir, 0755) != 0 && errno != EEXIST) return NULL;

    LilaAsiWal *wal = (LilaAsiWal *)calloc(1, sizeof(LilaAsiWal));
    if (!wal) return NULL;
    snprintf(wal->wal_dir, sizeof(wal->wal_dir), "%s", wal_dir);
    snprintf(wal->session_id, sizeof(wal->session_id), "%s", session_id);
    snprintf(wal->wal_path, sizeof(wal->wal_path), "%s/%s.wal", wal_dir, session_id);
    memcpy(wal->key_hash, key_hash, 32);
    wal->next_seq = 1;
    wal->next_ckpt_file_id = 1;
    return wal;
}

void lila_asi_wal_close(LilaAsiWal *wal)
{
    free(wal);
}

int lila_asi_wal_checkpoint(LilaAsiWal *wal,
                            LilaAsiSession *session,
                            uint32_t n_tokens,
                            uint64_t rng_state,
                            const char *label,
                            uint32_t *out_cp_id)
{
    if (!wal || !session || n_tokens == 0) return -1;
    uint32_t cp_id = ASI_CP_INVALID;
    if (lila_asi_cp_fork(session, n_tokens, rng_state, label ? label : "wal_ckpt", &cp_id) != 0)
        return -1;

    uint32_t fid = wal->next_ckpt_file_id++;
    char fname[256];
    char fpath[800];
    snprintf(fname, sizeof(fname), "%s.ckpt_%04u.asitree1", wal->session_id, fid);
    snprintf(fpath, sizeof(fpath), "%s/%s", wal->wal_dir, fname);

    if (lila_asi_cp_save_file(session, cp_id, wal->key_hash, fpath) != 0)
        return -1;

    char body[512];
    snprintf(body, sizeof(body), "C|%llu|%u|%u|%016llx|%s",
             (unsigned long long)wal->next_seq++,
             cp_id,
             n_tokens,
             (unsigned long long)rng_state,
             fname);
    if (sc_wal_append_line(wal, body) != 0) return -1;
    if (out_cp_id) *out_cp_id = cp_id;
    return 0;
}

int lila_asi_wal_append_tokens(LilaAsiWal *wal,
                               uint32_t base_cp_id,
                               const int32_t *token_ids,
                               uint32_t n_tokens,
                               uint64_t rng_state)
{
    if (!wal || !token_ids || n_tokens == 0) return -1;
    size_t cap = 128 + (size_t)n_tokens * 12;
    char *body = (char *)malloc(cap);
    if (!body) return -1;

    int off = snprintf(body, cap, "T|%llu|%u|%016llx|%u|",
                       (unsigned long long)wal->next_seq++,
                       base_cp_id,
                       (unsigned long long)rng_state,
                       n_tokens);
    if (off < 0 || (size_t)off >= cap) { free(body); return -1; }
    for (uint32_t i = 0; i < n_tokens; ++i)
    {
        int n = snprintf(body + off, cap - (size_t)off, "%s%d", (i == 0) ? "" : ",", token_ids[i]);
        if (n < 0 || (size_t)(off + n) >= cap) { free(body); return -1; }
        off += n;
    }

    int rc = sc_wal_append_line(wal, body);
    free(body);
    return rc;
}

typedef struct {
    char     kind; /* 'C' or 'T' */
    uint64_t seq;
    uint32_t cp_id;
    uint32_t n_tokens;
    uint64_t rng_state;
    char     fname[256];
    int32_t *tokens;
} ScWalParsedRec;

static void sc_wal_free_recs(ScWalParsedRec *recs, size_t n)
{
    for (size_t i = 0; i < n; ++i) free(recs[i].tokens);
    free(recs);
}

int lila_asi_wal_resume(LilaAsiWal *wal,
                        LilaAsiSession *session,
                        float *out_last_logits,
                        AsiWalResumeInfo *out_info)
{
    if (!wal || !session) return -1;
    FILE *fp = fopen(wal->wal_path, "r");
    if (!fp) return -1;

    size_t cap = 32, n_recs = 0;
    ScWalParsedRec *recs = (ScWalParsedRec *)calloc(cap, sizeof(ScWalParsedRec));
    if (!recs) { fclose(fp); return -1; }

    uint32_t corrupt_tail = 0;
    char line[8192];
    while (fgets(line, sizeof(line), fp))
    {
        size_t len = strlen(line);
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r'))
            line[--len] = '\0';
        if (len == 0) continue;

        char *bar = strchr(line, '|');
        int valid = 0;
        if (bar && (bar - line) == 8)
        {
            char hex[9];
            memcpy(hex, line, 8);
            hex[8] = '\0';
            char *endp = NULL;
            unsigned long exp_crc = strtoul(hex, &endp, 16);
            if (endp == hex + 8)
            {
                const char *body = bar + 1;
                uint32_t act_crc = sc_crc32((const uint8_t *)body, strlen(body));
                if (act_crc == (uint32_t)exp_crc)
                {
                    if (n_recs == cap)
                    {
                        cap *= 2;
                        ScWalParsedRec *nr = (ScWalParsedRec *)realloc(recs, cap * sizeof(ScWalParsedRec));
                        if (!nr) { sc_wal_free_recs(recs, n_recs); fclose(fp); return -1; }
                        recs = nr;
                    }
                    ScWalParsedRec *r = &recs[n_recs];
                    memset(r, 0, sizeof(*r));
                    if (body[0] == 'C' && body[1] == '|')
                    {
                        unsigned long long seq = 0, rng = 0;
                        unsigned int cid = 0, ntok = 0;
                        char fname[256] = {0};
                        if (sscanf(body + 2, "%llu|%u|%u|%llx|%255s", &seq, &cid, &ntok, &rng, fname) == 5)
                        {
                            r->kind = 'C';
                            r->seq = (uint64_t)seq;
                            r->cp_id = (uint32_t)cid;
                            r->n_tokens = (uint32_t)ntok;
                            r->rng_state = (uint64_t)rng;
                            snprintf(r->fname, sizeof(r->fname), "%s", fname);
                            n_recs++;
                            valid = 1;
                        }
                    }
                    else if (body[0] == 'T' && body[1] == '|')
                    {
                        unsigned long long seq = 0, rng = 0;
                        unsigned int cid = 0, ntok = 0;
                        int consumed = 0;
                        if (sscanf(body + 2, "%llu|%u|%llx|%u|%n", &seq, &cid, &rng, &ntok, &consumed) == 4 &&
                            consumed > 0 && ntok > 0)
                        {
                            r->tokens = (int32_t *)malloc((size_t)ntok * sizeof(int32_t));
                            if (r->tokens)
                            {
                                const char *p = body + 2 + consumed;
                                uint32_t parsed = 0;
                                while (parsed < ntok && *p)
                                {
                                    char *ep = NULL;
                                    long tv = strtol(p, &ep, 10);
                                    if (ep == p) break;
                                    r->tokens[parsed++] = (int32_t)tv;
                                    p = (*ep == ',') ? (ep + 1) : ep;
                                }
                                if (parsed == ntok)
                                {
                                    r->kind = 'T';
                                    r->seq = (uint64_t)seq;
                                    r->cp_id = (uint32_t)cid;
                                    r->n_tokens = (uint32_t)ntok;
                                    r->rng_state = (uint64_t)rng;
                                    n_recs++;
                                    valid = 1;
                                }
                                else
                                {
                                    free(r->tokens);
                                    r->tokens = NULL;
                                }
                            }
                        }
                    }
                }
            }
        }
        if (!valid)
        {
            /* Tolerate torn trailing line from simulated mid-write crash */
            if (feof(fp) || fgetc(fp) == EOF)
            {
                corrupt_tail++;
                break;
            }
            sc_wal_free_recs(recs, n_recs);
            fclose(fp);
            return -1;
        }
    }
    fclose(fp);

    /* Find latest valid checkpoint record whose file loads cleanly */
    size_t ckpt_idx = n_recs;
    uint32_t loaded_cp_id = ASI_CP_INVALID;
    for (size_t i = n_recs; i > 0; --i)
    {
        size_t idx = i - 1;
        if (recs[idx].kind == 'C')
        {
            char fpath[800];
            snprintf(fpath, sizeof(fpath), "%s/%s", wal->wal_dir, recs[idx].fname);
            if (lila_asi_cp_load_file(session, wal->key_hash, fpath, "wal_recovered", &loaded_cp_id) == 0)
            {
                ckpt_idx = idx;
                break;
            }
        }
    }
    if (ckpt_idx == n_recs || loaded_cp_id == ASI_CP_INVALID)
    {
        sc_wal_free_recs(recs, n_recs);
        return -1;
    }

    uint32_t ckpt_pos = 0;
    uint64_t cur_rng = 0;
    if (lila_asi_cp_restore(session, loaded_cp_id, &ckpt_pos, &cur_rng) != 0)
    {
        sc_wal_free_recs(recs, n_recs);
        return -1;
    }

    /* Replay ONLY un-checkpointed tail token records after ckpt_idx */
    const AsiV2Cartridge *cart = lila_asi_sc_get_cartridge(session);
    uint32_t V = cart ? cart->config.vocab_size : 0;
    float *tmp_logits = (!out_last_logits && V > 0) ? (float *)malloc((size_t)V * sizeof(float)) : NULL;
    float *logit_dst = out_last_logits ? out_last_logits : tmp_logits;

    uint32_t pos = ckpt_pos;
    uint32_t replayed = 0;
    for (size_t i = ckpt_idx + 1; i < n_recs; ++i)
    {
        if (recs[i].kind == 'T')
        {
            for (uint32_t t = 0; t < recs[i].n_tokens; ++t)
            {
                if (!logit_dst || lila_asi_eval_token(session, recs[i].tokens[t], pos, logit_dst) != 0)
                {
                    free(tmp_logits);
                    sc_wal_free_recs(recs, n_recs);
                    return -1;
                }
                pos++;
                replayed++;
            }
            cur_rng = recs[i].rng_state;
        }
    }
    free(tmp_logits);

    if (n_recs > 0)
    {
        wal->next_seq = recs[n_recs - 1].seq + 1;
    }
    sc_wal_free_recs(recs, n_recs);

    if (out_info)
    {
        out_info->restored_cp_id = loaded_cp_id;
        out_info->checkpoint_n_tokens = ckpt_pos;
        out_info->replayed_tail_tokens = replayed;
        out_info->total_n_tokens = pos;
        out_info->rng_state = cur_rng;
        out_info->corrupt_tail_records_ignored = corrupt_tail;
    }
    return 0;
}

/* =========================================================================
 * 6. C11 Confidence API (lila_asi_sample_ex)
 * ========================================================================= */

static int sc_cmp_float_desc(const void *a, const void *b)
{
    float fa = *(const float *)a;
    float fb = *(const float *)b;
    return (fb > fa) - (fb < fa);
}

int32_t lila_asi_sample_ex(const float *logits,
                           uint32_t vocab_size,
                           float temperature,
                           uint32_t top_k,
                           float top_p,
                           uint64_t *rng_state,
                           AsiSampleStats *out_stats)
{
    if (!logits || vocab_size == 0) return -1;

    if (out_stats)
    {
        memset(out_stats, 0, sizeof(*out_stats));
        float top64[64];
        uint32_t count = 0;
        float min_v = 0.0f;
        uint32_t min_idx = 0;
        uint32_t n_finite = 0;

        for (uint32_t i = 0; i < vocab_size; ++i)
        {
            float v = logits[i];
            if (!isfinite(v)) continue;
            n_finite++;
            if (count < 64u)
            {
                top64[count] = v;
                if (count == 0 || v < min_v)
                {
                    min_v = v;
                    min_idx = count;
                }
                count++;
            }
            else if (v > min_v)
            {
                top64[min_idx] = v;
                min_v = top64[0];
                min_idx = 0;
                for (uint32_t j = 1; j < 64u; ++j)
                {
                    if (top64[j] < min_v)
                    {
                        min_v = top64[j];
                        min_idx = j;
                    }
                }
            }
        }

        out_stats->n_finite = n_finite;
        if (count > 0)
        {
            qsort(top64, count, sizeof(float), sc_cmp_float_desc);
            out_stats->top1_logit = top64[0];
            out_stats->top2_logit = (count > 1) ? top64[1] : -INFINITY;
            out_stats->margin = (count > 1) ? (top64[0] - top64[1]) : INFINITY;

            double max_l = (double)top64[0];
            double sum_exp = 0.0;
            for (uint32_t i = 0; i < count; ++i)
            {
                sum_exp += exp((double)top64[i] - max_l);
            }
            double lse = max_l + log(sum_exp);
            out_stats->top1_logprob = (float)(max_l - lse);

            double ent = 0.0;
            for (uint32_t i = 0; i < count; ++i)
            {
                double diff = (double)top64[i] - lse;
                double p = exp(diff);
                if (p > 0.0) ent -= p * diff;
            }
            out_stats->entropy = (float)ent;
        }
        else
        {
            out_stats->top1_logit = -INFINITY;
            out_stats->top2_logit = -INFINITY;
            out_stats->margin = 0.0f;
            out_stats->top1_logprob = -INFINITY;
            out_stats->entropy = 0.0f;
        }
    }

    int32_t chosen = lila_asi_sample(logits, vocab_size, temperature, top_k, top_p, rng_state);
    if (out_stats) out_stats->token_id = chosen;
    return chosen;
}
