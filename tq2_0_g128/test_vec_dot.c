// TQ2_0_G128 x Q8_K: NEON vec_dot vs the generic C kernel vs an fp32 reference on dequantized data.
// Random ternary weights with independent fp16 scales per 128 (the case TQ2_0 can't represent).
// Errors are relative to sum|w*a| (the condition scale). Exit code 1 if the native kernel (NEON on
// ARM, AVX2 on x86) and the generic one disagree beyond float rounding.
#include "ggml.h"
#include "ggml-cpu.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

void ggml_vec_dot_tq2_0_g128_q8_K_generic(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);

static float frand(void) { return (float) rand() / RAND_MAX * 2.0f - 1.0f; }

// Q8_K has no to_float in ggml; its block is {float d; int8 qs[256]; int16 bsums[16]}
typedef struct { float d; int8_t qs[256]; int16_t bsums[16]; } q8k_t;
static void dequant_q8_K(const void * vx, float * y, int n) {
    const q8k_t * x = vx;
    for (int i = 0; i < n / 256; ++i)
        for (int j = 0; j < 256; ++j) y[i*256 + j] = x[i].d * x[i].qs[j];
}

int main(void) {
    ggml_cpu_init();
    srand(0);
    const int n = 6144;  // largest Bonsai row (ffn_down)
    const struct ggml_type_traits     * tt = ggml_get_type_traits(GGML_TYPE_TQ2_0_G128);
    const struct ggml_type_traits_cpu * tc = ggml_get_type_traits_cpu(GGML_TYPE_TQ2_0_G128);
    const struct ggml_type_traits_cpu * t8 = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);

    float * w = malloc(n * sizeof(float)), * a = malloc(n * sizeof(float));
    float * wd = malloc(n * sizeof(float)), * ad = malloc(n * sizeof(float));
    void * qw = malloc(ggml_row_size(GGML_TYPE_TQ2_0_G128, n));
    void * qa = malloc(ggml_row_size(GGML_TYPE_Q8_K, n));

    double max_rel_gen = 0, max_rel_ref = 0;
    int bad = 0;
    for (int trial = 0; trial < 2000; ++trial) {
        for (int g = 0; g < n / 128; ++g) {
            const float s = ggml_fp16_to_fp32(ggml_fp32_to_fp16(0.001f + 0.05f * (float) rand() / RAND_MAX));
            for (int j = 0; j < 128; ++j) w[g*128 + j] = s * (float) (rand() % 3 - 1);
        }
        const float amp = trial % 2 ? 1.0f : 30.0f;
        for (int j = 0; j < n; ++j) a[j] = amp * frand();

        tt->from_float_ref(w, qw, n);
        t8->from_float(a, qa, n);
        tt->to_float(qw, wd, n);
        dequant_q8_K(qa, ad, n);

        float s_neon, s_gen;
        tc->vec_dot(n, &s_neon, 0, qw, 0, qa, 0, 1);
        ggml_vec_dot_tq2_0_g128_q8_K_generic(n, &s_gen, 0, qw, 0, qa, 0, 1);
        double ref = 0, cond = 0;  // cond = sum |w*a|: the scale float rounding errors live on
        for (int j = 0; j < n; ++j) { ref += (double) wd[j] * ad[j]; cond += fabs((double) wd[j] * ad[j]); }
        for (int j = 0; j < n; ++j) if (wd[j] != w[j]) { printf("dequant mismatch at %d\n", j); return 1; }

        const double scale = cond;
        const double rg = fabs(s_neon - s_gen) / scale, rr = fabs(s_neon - ref) / scale;
        if (rg > max_rel_gen) max_rel_gen = rg;
        if (rr > max_rel_ref) max_rel_ref = rr;
        if (rg > 1e-6) bad++;  // float32 accumulation over 6144 terms stays far below this
    }
    printf("trials=2000 n=%d  max_err/sum|w*a|: native vs generic=%.3g  native vs fp64 ref=%.3g  bad=%d\n",
           n, max_rel_gen, max_rel_ref, bad);
    printf("quantize/dequantize round trip: exact\n");
    return bad != 0;
}
