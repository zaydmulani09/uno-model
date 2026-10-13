// Memory read bandwidth: T threads each sum their slice of a 512 MiB buffer (far past L2).
// Prints best-of-REPS GB/s (1e9 B/s) for T = 1..4. Build: cc -O2 -pthread membw.c
#include <arm_neon.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BYTES (512ull << 20)
#define N (BYTES / sizeof(uint64_t))
#define REPS 7

static uint64_t *buf;
typedef struct { size_t lo, hi; uint64_t sum; } job_t;

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void *work(void *p) {
    job_t *j = p;
    uint64x2_t a0 = vdupq_n_u64(0), a1 = a0, a2 = a0, a3 = a0;
    for (size_t i = j->lo; i < j->hi; i += 8) {
        a0 = vaddq_u64(a0, vld1q_u64(buf + i));
        a1 = vaddq_u64(a1, vld1q_u64(buf + i + 2));
        a2 = vaddq_u64(a2, vld1q_u64(buf + i + 4));
        a3 = vaddq_u64(a3, vld1q_u64(buf + i + 6));
    }
    uint64x2_t s = vaddq_u64(vaddq_u64(a0, a1), vaddq_u64(a2, a3));
    j->sum = vgetq_lane_u64(s, 0) + vgetq_lane_u64(s, 1);
    return NULL;
}

int main(void) {
    buf = aligned_alloc(64, BYTES);
    for (size_t i = 0; i < N; i++) buf[i] = i;  // touch every page
    int ok = 1;
    for (int t = 1; t <= 4; t++) {
        double best = 1e30;
        for (int r = 0; r < REPS; r++) {
            pthread_t th[4];
            job_t jb[4];
            double t0 = now();
            for (int k = 0; k < t; k++) {
                size_t chunk = N / t & ~(size_t)7;  // 8-aligned slices; last one takes the rest
                jb[k].lo = chunk * k;
                jb[k].hi = k == t - 1 ? N : chunk * (k + 1);
                pthread_create(&th[k], NULL, work, &jb[k]);
            }
            uint64_t total = 0;
            for (int k = 0; k < t; k++) { pthread_join(th[k], NULL); total += jb[k].sum; }
            double dt = now() - t0;
            ok &= total == (uint64_t)N * (N - 1) / 2;  // keeps loads alive + checks coverage
            if (dt < best) best = dt;
        }
        printf("threads=%d read_GBps=%.2f\n", t, BYTES / best / 1e9);
    }
    printf("checksum_ok=%d\n", ok);
    return !ok;
}
