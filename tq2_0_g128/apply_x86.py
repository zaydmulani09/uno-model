"""TQ2_0_G128, part 3: AVX2 vec_dot for the x86 PC that runs the evals (the generic C kernel is ~10x
too slow for 1800 questions). Run from the llama.cpp root after apply_ggml.py and apply_rest.py."""


def sub(p, old, new, count=1):
    s = open(p, encoding='utf-8').read()
    n = s.count(old)
    assert n == count, (p, old[:70], n)
    open(p, 'w', encoding='utf-8', newline='').write(s.replace(old, new))


sub('ggml/src/ggml-cpu/arch-fallback.h',
    '#if defined(GGML_CPU_GENERIC) || !(defined(__aarch64__) || defined(__arm__) || defined(_M_ARM) || defined(_M_ARM64) || defined(_M_ARM64EC))',
    '#if defined(GGML_CPU_GENERIC) || !(defined(__aarch64__) || defined(__arm__) || defined(_M_ARM) || defined(_M_ARM64) || defined(_M_ARM64EC) || \\\n'
    '                                   defined(__x86_64__) || defined(__i386__) || defined(_M_IX86) || defined(_M_X64))')
sub('ggml/src/ggml-cpu/arch-fallback.h', '// uno-model TQ2_0_G128: NEON kernel on ARM only, generic everywhere else',
    '// uno-model TQ2_0_G128: NEON kernel on ARM, AVX2 on x86, generic everywhere else')

sub('ggml/src/ggml-cpu/arch/x86/quants.c',
    'void ggml_vec_dot_q2_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {',
'''// TQ2_0_G128: two 128-weight blocks per Q8_K block, one fp16 scale each (see the ARM kernel)
void ggml_vec_dot_tq2_0_g128_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    assert(n % QK_K == 0);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_tq2_0_g128 * GGML_RESTRICT x = vx;
    const block_q8_K       * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

#if defined(__AVX2__)
    __m256 sumf = _mm256_setzero_ps();
    const __m256i m3 = _mm256_set1_epi8(3);

    for (int i = 0; i < nb; ++i) {
        for (int h = 0; h < 2; ++h) {
            const int8_t * GGML_RESTRICT q8 = y[i].qs + 128*h;

            __m256i qx0 = _mm256_loadu_si256((const __m256i *) x[2*i + h].qs);
            __m256i qx1 = _mm256_and_si256(_mm256_srli_epi16(qx0, 2), m3);
            __m256i qx2 = _mm256_and_si256(_mm256_srli_epi16(qx0, 4), m3);
            __m256i qx3 = _mm256_and_si256(_mm256_srli_epi16(qx0, 6), m3);
            qx0 = _mm256_and_si256(qx0, m3);

            // codes 0..2 times int8: 16-bit pair sums, at most 2*2*128 per lane per product pair
            qx0 = _mm256_maddubs_epi16(qx0, _mm256_loadu_si256((const __m256i *) (q8 +  0)));
            qx1 = _mm256_maddubs_epi16(qx1, _mm256_loadu_si256((const __m256i *) (q8 + 32)));
            qx2 = _mm256_maddubs_epi16(qx2, _mm256_loadu_si256((const __m256i *) (q8 + 64)));
            qx3 = _mm256_maddubs_epi16(qx3, _mm256_loadu_si256((const __m256i *) (q8 + 96)));

            __m256i sumi = _mm256_add_epi16(_mm256_add_epi16(qx0, qx1), _mm256_add_epi16(qx2, qx3));
            sumi = _mm256_madd_epi16(sumi, _mm256_set1_epi16(1));
            // minus sum(y) of this half: its 8 bsums, widened to int32
            sumi = _mm256_sub_epi32(sumi, _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i *) (y[i].bsums + 8*h))));

            const __m256 d = _mm256_set1_ps(y[i].d * GGML_CPU_FP16_TO_FP32(x[2*i + h].d));
            sumf = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(sumi), d), sumf);
        }
    }

    *s = hsum_float_8(sumf);
#else
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    ggml_vec_dot_tq2_0_g128_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

void ggml_vec_dot_q2_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {''')
print('x86: ok')
