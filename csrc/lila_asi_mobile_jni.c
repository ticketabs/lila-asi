/*
 * lila_asi_mobile_jni.c — High-Level C11 & Android JNI Mobile Bridge for Mode 1 (On-Device Offline)
 *
 * Exposes:
 *   1. Simple C11 text-to-text API (`lila_asi_mobile_generate_utf8`) for iOS Swift, Flutter FFI,
 *      React Native JSI, and Desktop C/C++ apps (using embedded ASI_SEC_TOKENIZER BPE).
 *   2. Android JNI bindings (`Java_com_ticketlabs_lilaasi_LilaAsiEngine_*`) when compiled under
 *      Android NDK (`__ANDROID__` / `<jni.h>`).
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "lila_asi.h"
#include "asi_grammar_sparse.h"
#include "asi_state_centric.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

typedef struct {
    LilaAsiModel      *model;
    LilaAsiSession    *session;
    AsiV2ModelConfig   cfg;
    AsiGrammarVocab    vocab;
    int                vocab_ok;
    float             *logits;
    uint64_t           rng;
} LilaAsiMobileHandle;

LilaAsiMobileHandle *lila_asi_mobile_open(const char *asi_path,
                                          uint32_t max_ctx,
                                          uint32_t n_threads,
                                          char *err_buf,
                                          size_t err_cap)
{
    LilaAsiMobileHandle *h = (LilaAsiMobileHandle *)calloc(1, sizeof(LilaAsiMobileHandle));
    if (!h) return NULL;
    h->rng = 0x9E3779B97F4A7C15ULL;
    h->model = lila_asi_open(asi_path, err_buf, err_cap);
    if (!h->model) { free(h); return NULL; }
    lila_asi_model_info(h->model, &h->cfg);
    h->session = lila_asi_session_create(h->model, max_ctx ? max_ctx : 2048u, n_threads ? n_threads : 2u, err_buf, err_cap);
    if (!h->session) { lila_asi_close(h->model); free(h); return NULL; }
    h->logits = (float *)calloc(h->cfg.vocab_size, sizeof(float));
    const void *sec_ptr = NULL;
    size_t sec_sz = 0;
    if (lila_asi_get_section(h->model, ASI_SEC_TOKENIZER, &sec_ptr, &sec_sz) == 0 && sec_ptr && sec_sz > 0) {
        if (asi_grammar_vocab_from_asi_json(&h->vocab, sec_ptr, sec_sz) == 0) {
            h->vocab_ok = 1;
        }
    }
    return h;
}

void lila_asi_mobile_close(LilaAsiMobileHandle *h)
{
    if (!h) return;
    if (h->vocab_ok) asi_grammar_vocab_free(&h->vocab);
    free(h->logits);
    if (h->session) lila_asi_session_free(h->session);
    if (h->model) lila_asi_close(h->model);
    free(h);
}

int lila_asi_mobile_generate_utf8(LilaAsiMobileHandle *h,
                                  const char *prompt_utf8,
                                  uint32_t max_new_tokens,
                                  float temperature,
                                  char *out_utf8,
                                  size_t out_cap)
{
    if (!h || !h->vocab_ok || !prompt_utf8 || !out_utf8 || out_cap == 0) return -1;
    out_utf8[0] = '\0';
    int32_t prompt_ids[2048];
    int n_prompt = asi_grammar_vocab_greedy_encode(&h->vocab, prompt_utf8, strlen(prompt_utf8), prompt_ids, 2048u);
    if (n_prompt <= 0) return -1;

    lila_asi_session_reset(h->session);
    if (lila_asi_eval_batch(h->session, prompt_ids, (uint32_t)n_prompt, 0u, h->logits) != 0) return -1;

    uint32_t pos = (uint32_t)n_prompt;
    size_t written = 0;
    for (uint32_t step = 0; step < max_new_tokens; ++step) {
        AsiSampleStats stats;
        int32_t tok = lila_asi_sample_ex(h->logits, h->cfg.vocab_size, temperature, 20u, 0.95f, &h->rng, &stats);
        if (tok < 0 || (uint32_t)tok >= h->vocab.vocab_size) break;
        const char *piece = h->vocab.tokens[tok];
        uint32_t plen = h->vocab.token_lens[tok];
        if (piece && plen > 0) {
            if (plen == 10 && memcmp(piece, "<|im_end|>", 10) == 0) break;
            if (written + plen + 1 < out_cap) {
                memcpy(out_utf8 + written, piece, plen);
                written += plen;
                out_utf8[written] = '\0';
            }
        }
        if (step + 1u < max_new_tokens) {
            if (lila_asi_eval_token(h->session, tok, pos++, h->logits) != 0) break;
        }
    }
    return (int)written;
}

#if defined(__ANDROID__)
#include <jni.h>

JNIEXPORT jlong JNICALL
Java_com_ticketlabs_lilaasi_LilaAsiEngine_nativeOpen(JNIEnv *env, jclass clazz, jstring jpath, jint maxCtx, jint nThreads)
{
    (void)clazz;
    const char *path = (*env)->GetStringUTFChars(env, jpath, NULL);
    char err[256] = {0};
    LilaAsiMobileHandle *h = lila_asi_mobile_open(path, (uint32_t)maxCtx, (uint32_t)nThreads, err, sizeof(err));
    (*env)->ReleaseStringUTFChars(env, jpath, path);
    return (jlong)(intptr_t)h;
}

JNIEXPORT jstring JNICALL
Java_com_ticketlabs_lilaasi_LilaAsiEngine_nativeGenerate(JNIEnv *env, jclass clazz, jlong handle, jstring jprompt, jint maxTokens, jfloat temp)
{
    (void)clazz;
    LilaAsiMobileHandle *h = (LilaAsiMobileHandle *)(intptr_t)handle;
    if (!h) return (*env)->NewStringUTF(env, "");
    const char *prompt = (*env)->GetStringUTFChars(env, jprompt, NULL);
    char *buf = (char *)calloc(65536, 1);
    if (buf) {
        lila_asi_mobile_generate_utf8(h, prompt, (uint32_t)maxTokens, (float)temp, buf, 65536);
    }
    (*env)->ReleaseStringUTFChars(env, jprompt, prompt);
    jstring res = (*env)->NewStringUTF(env, buf ? buf : "");
    free(buf);
    return res;
}

JNIEXPORT void JNICALL
Java_com_ticketlabs_lilaasi_LilaAsiEngine_nativeClose(JNIEnv *env, jclass clazz, jlong handle)
{
    (void)env;
    (void)clazz;
    lila_asi_mobile_close((LilaAsiMobileHandle *)(intptr_t)handle);
}
#endif
