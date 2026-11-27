#ifndef LILA_ASI_PLATFORM_H
#define LILA_ASI_PLATFORM_H

/*
 * Cross-platform OS primitives for Lila ASI C11 runtime (Linux, macOS, Windows).
 */

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#  include <direct.h>
#  include <io.h>
#  include <malloc.h>
#  include <process.h>
#  include <time.h>
#else
#  include <pthread.h>
#  include <strings.h>
#  include <sys/stat.h>
#  include <time.h>
#  include <unistd.h>
#endif

#if defined(_WIN32) && defined(LILA_ASI_BUILD_DLL)
#  define ASI_API __declspec(dllexport)
#else
#  define ASI_API
#endif

static inline void *asi_aligned_alloc(size_t alignment, size_t size)
{
    if (alignment < sizeof(void *)) alignment = sizeof(void *);
    if (size == 0) size = alignment;
#if defined(_WIN32)
    return _aligned_malloc(size, alignment);
#else
    size_t padded = (size + alignment - 1u) & ~(alignment - 1u);
    return aligned_alloc(alignment, padded);
#endif
}

static inline void asi_aligned_free(void *ptr)
{
    if (!ptr) return;
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

static inline uint64_t asi_monotonic_ns(void)
{
#if defined(_WIN32)
    static LARGE_INTEGER freq = {0};
    LARGE_INTEGER counter;
    if (freq.QuadPart == 0)
        QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    if (freq.QuadPart <= 0)
        return (uint64_t)GetTickCount64() * 1000000ULL;
    /* Split to avoid uint64 overflow while preserving nanosecond resolution */
    uint64_t q = (uint64_t)counter.QuadPart;
    uint64_t f = (uint64_t)freq.QuadPart;
    return (q / f) * 1000000000ULL + ((q % f) * 1000000000ULL) / f;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
#endif
}

static inline int asi_fsync_file(FILE *f)
{
    if (!f) return -1;
#if defined(_WIN32)
    int fd = _fileno(f);
    if (fd < 0) return -1;
    return (_commit(fd) == 0) ? 0 : -1;
#else
    int fd = fileno(f);
    if (fd < 0) return -1;
    if (fsync(fd) != 0 && errno != EINVAL && errno != EROFS)
        return -1;
    return 0;
#endif
}

static inline int asi_mkdir(const char *path)
{
    if (!path || !path[0]) return -1;
#if defined(_WIN32)
    if (_mkdir(path) == 0 || errno == EEXIST)
        return 0;
    return -1;
#else
    if (mkdir(path, 0755) == 0 || errno == EEXIST)
        return 0;
    return -1;
#endif
}

static inline int asi_setenv(const char *key, const char *val)
{
    if (!key || !val) return -1;
#if defined(_WIN32)
    return (_putenv_s(key, val) == 0) ? 0 : -1;
#else
    return setenv(key, val, 1);
#endif
}

static inline char *asi_strtok_r(char *str, const char *delim, char **saveptr)
{
#if defined(_WIN32)
    return strtok_s(str, delim, saveptr);
#else
    return strtok_r(str, delim, saveptr);
#endif
}

static inline int asi_num_procs(void)
{
#if defined(_WIN32)
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (si.dwNumberOfProcessors > 0) ? (int)si.dwNumberOfProcessors : 1;
#else
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return (n > 0) ? (int)n : 1;
#endif
}

static inline long asi_getpid(void)
{
#if defined(_WIN32)
    return (long)_getpid();
#else
    return (long)getpid();
#endif
}

static inline int asi_unlink(const char *path)
{
#if defined(_WIN32)
    return _unlink(path);
#else
    return unlink(path);
#endif
}

static inline int asi_atomic_rename(const char *src, const char *dst)
{
#if defined(_WIN32)
    return MoveFileExA(src, dst, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) ? 0 : -1;
#else
    return rename(src, dst);
#endif
}

static inline int asi_strcasecmp(const char *a, const char *b)
{
#if defined(_WIN32)
    return _stricmp(a, b);
#else
    return strcasecmp(a, b);
#endif
}

static inline const void *asi_memmem(const void *haystack, size_t hlen,
                                     const void *needle, size_t nlen)
{
    if (nlen == 0) return haystack;
    if (!haystack || !needle || hlen < nlen) return NULL;
    const uint8_t *h = (const uint8_t *)haystack;
    const uint8_t *nd = (const uint8_t *)needle;
    size_t limit = hlen - nlen;
    for (size_t i = 0; i <= limit; ++i)
    {
        if (h[i] == nd[0] && memcmp(h + i, nd, nlen) == 0)
            return (const void *)(h + i);
    }
    return NULL;
}

#if defined(_WIN32)
typedef INIT_ONCE AsiOnce;
#define ASI_ONCE_INIT INIT_ONCE_STATIC_INIT

static inline BOOL CALLBACK asi_once_trampoline_(PINIT_ONCE once, PVOID param, PVOID *ctx)
{
    (void)once;
    (void)ctx;
    void (*fn)(void) = (void (*)(void))param;
    if (fn) fn();
    return TRUE;
}

static inline void asi_call_once(AsiOnce *once, void (*fn)(void))
{
    InitOnceExecuteOnce(once, asi_once_trampoline_, (PVOID)fn, NULL);
}
#else
typedef pthread_once_t AsiOnce;
#define ASI_ONCE_INIT PTHREAD_ONCE_INIT

static inline void asi_call_once(AsiOnce *once, void (*fn)(void))
{
    pthread_once(once, fn);
}
#endif

#endif /* LILA_ASI_PLATFORM_H */
