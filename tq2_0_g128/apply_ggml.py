"""Add GGML_TYPE_TQ2_0_G128 to a llama.cpp b11594 tree (ggml core, CPU backend, generic kernel).
Run from the llama.cpp root. The NEON kernel, gguf-py and llama plumbing are separate steps."""


def sub(p, old, new, count=1):
    s = open(p, encoding='utf-8').read()
    n = s.count(old)
    assert n == count, (p, old[:70], n)
    open(p, 'w', encoding='utf-8', newline='').write(s.replace(old, new))


# --- ggml.h
sub('ggml/include/ggml.h', '        GGML_TYPE_COUNT   = 43,',
    '        // uno-model: TQ2_0 layout with one fp16 scale per 128 weights (lossless for g128 ternary)\n'
    '        GGML_TYPE_TQ2_0_G128 = 160,\n        GGML_TYPE_COUNT   = 161,')
sub('ggml/include/ggml.h', '        GGML_FTYPE_MOSTLY_Q2_0    = 28, // except 1d tensors\n',
    '        GGML_FTYPE_MOSTLY_Q2_0    = 28, // except 1d tensors\n'
    '        GGML_FTYPE_MOSTLY_TQ2_0_G128 = 160, // except 1d tensors\n')

# --- ggml-common.h
sub('ggml/src/ggml-common.h',
    'static_assert(sizeof(block_tq2_0) == sizeof(ggml_half) + QK_K / 4, "wrong tq2_0 block size/padding");\n',
    'static_assert(sizeof(block_tq2_0) == sizeof(ggml_half) + QK_K / 4, "wrong tq2_0 block size/padding");\n\n'
    '// 2.125 bpw: TQ2_0 layout, one scale per 128 weights\n'
    '#define QK_TQ2_0_G128 128\n'
    'typedef struct {\n'
    '    uint8_t qs[QK_TQ2_0_G128/4]; // byte m holds weights m, m+32, m+64, m+96 (2 bits each)\n'
    '    ggml_half d;\n'
    '} block_tq2_0_g128;\n'
    'static_assert(sizeof(block_tq2_0_g128) == sizeof(ggml_half) + QK_TQ2_0_G128 / 4, "wrong tq2_0_g128 block size/padding");\n')

# --- ggml-quants.h
H = 'ggml/src/ggml-quants.h'
sub(H, 'GGML_API void quantize_row_tq2_0_ref(const float * GGML_RESTRICT x, block_tq2_0 * GGML_RESTRICT y, int64_t k);\n',
    'GGML_API void quantize_row_tq2_0_ref(const float * GGML_RESTRICT x, block_tq2_0 * GGML_RESTRICT y, int64_t k);\n'
    'GGML_API void quantize_row_tq2_0_g128_ref(const float * GGML_RESTRICT x, block_tq2_0_g128 * GGML_RESTRICT y, int64_t k);\n')
sub(H, 'GGML_API void dequantize_row_tq2_0(const block_tq2_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);\n',
    'GGML_API void dequantize_row_tq2_0(const block_tq2_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);\n'
    'GGML_API void dequantize_row_tq2_0_g128(const block_tq2_0_g128 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);\n')
sub(H, 'GGML_API size_t quantize_tq2_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);\n',
    'GGML_API size_t quantize_tq2_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);\n'
    'GGML_API size_t quantize_tq2_0_g128(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);\n')

# --- ggml-quants.c
Q = 'ggml/src/ggml-quants.c'
sub(Q, 'size_t quantize_tq1_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {',
'''void quantize_row_tq2_0_g128_ref(const float * GGML_RESTRICT x, block_tq2_0_g128 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TQ2_0_G128 == 0);
    const int64_t nb = k / QK_TQ2_0_G128;

    for (int64_t i = 0; i < nb; i++) {
        float amax = 0.0f;
        for (int j = 0; j < QK_TQ2_0_G128; j++) {
            amax = MAX(amax, fabsf(x[j]));
        }

        const float d  = amax;
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        for (size_t m = 0; m < 32; ++m) {
            uint8_t q = 0;
            for (size_t n = 0; n < 4; ++n) {
                // -1, 0, 1 -> 0, 1, 2
                int xi = lroundf(x[m + n*32] * id) + 1;
                q += (xi & 3) << (2*n);
            }
            y[i].qs[m] = q;
        }
        x += QK_TQ2_0_G128;
    }
}

size_t quantize_tq1_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {''')
sub(Q, 'void dequantize_row_tq1_0(const block_tq1_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {',
'''size_t quantize_tq2_0_g128(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    (void)quant_weights; // not used
    const size_t row_size = ggml_row_size(GGML_TYPE_TQ2_0_G128, n_per_row);
    quantize_row_tq2_0_g128_ref(src, dst, (int64_t)nrow*n_per_row);
    return nrow * row_size;
}

void dequantize_row_tq2_0_g128(const block_tq2_0_g128 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TQ2_0_G128 == 0);
    const int64_t nb = k / QK_TQ2_0_G128;

    for (int64_t i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(x[i].d);
        for (size_t l = 0; l < 4; ++l) {
            for (size_t m = 0; m < 32; ++m) {
                int8_t q = (x[i].qs[m] >> (l*2)) & 3;
                *y++ = (float) (q - 1) * d;
            }
        }
    }
}

void dequantize_row_tq1_0(const block_tq1_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {''')
sub(Q, '''        case GGML_TYPE_TQ2_0:
            {
                VALIDATE_ROW_DATA_D_F16_IMPL(block_tq2_0, data, nb);
            } break;''', '''        case GGML_TYPE_TQ2_0:
            {
                VALIDATE_ROW_DATA_D_F16_IMPL(block_tq2_0, data, nb);
            } break;
        case GGML_TYPE_TQ2_0_G128:
            {
                VALIDATE_ROW_DATA_D_F16_IMPL(block_tq2_0_g128, data, nb);
            } break;''')

# --- ggml.c
G = 'ggml/src/ggml.c'
sub(G, '''        .to_float                 = (ggml_to_float_t) dequantize_row_tq2_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_tq2_0_ref,
    },''', '''        .to_float                 = (ggml_to_float_t) dequantize_row_tq2_0,
        .from_float_ref           = (ggml_from_float_t) quantize_row_tq2_0_ref,
    },
    [GGML_TYPE_TQ2_0_G128] = {
        .type_name                = "tq2_0_g128",
        .blck_size                = QK_TQ2_0_G128,
        .type_size                = sizeof(block_tq2_0_g128),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_tq2_0_g128,
        .from_float_ref           = (ggml_from_float_t) quantize_row_tq2_0_g128_ref,
    },''')
sub(G, '        case GGML_FTYPE_MOSTLY_Q2_0:          wtype = GGML_TYPE_Q2_0;  break;\n',
    '        case GGML_FTYPE_MOSTLY_Q2_0:          wtype = GGML_TYPE_Q2_0;  break;\n'
    '        case GGML_FTYPE_MOSTLY_TQ2_0_G128:    wtype = GGML_TYPE_TQ2_0_G128; break;\n')
sub(G, '        case GGML_TYPE_TQ2_0:   result = quantize_tq2_0  (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;\n',
    '        case GGML_TYPE_TQ2_0:   result = quantize_tq2_0  (src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;\n'
    '        case GGML_TYPE_TQ2_0_G128: result = quantize_tq2_0_g128(src + start, (char *) dst + start_row * row_size, nrows, n_per_row, imatrix); break;\n')

# --- ggml-cpu
sub('ggml/src/ggml-cpu/ggml-cpu.c', '''        .vec_dot                  = ggml_vec_dot_tq2_0_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },''', '''        .vec_dot                  = ggml_vec_dot_tq2_0_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_TQ2_0_G128] = {
        .from_float               = quantize_row_tq2_0_g128,
        .vec_dot                  = ggml_vec_dot_tq2_0_g128_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },''')
CH = 'ggml/src/ggml-cpu/quants.h'
VD = '(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);\n'
sub(CH, 'void quantize_row_tq2_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);\n',
    'void quantize_row_tq2_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);\n'
    'void quantize_row_tq2_0_g128(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);\n')
sub(CH, 'void ggml_vec_dot_tq2_0_q8_K' + VD, 'void ggml_vec_dot_tq2_0_q8_K' + VD + 'void ggml_vec_dot_tq2_0_g128_q8_K' + VD)
sub(CH, 'void ggml_vec_dot_tq2_0_q8_K_generic' + VD, 'void ggml_vec_dot_tq2_0_q8_K_generic' + VD + 'void ggml_vec_dot_tq2_0_g128_q8_K_generic' + VD)
CQ = 'ggml/src/ggml-cpu/quants.c'
sub(CQ, '''    block_tq2_0 * GGML_RESTRICT y = vy;
    quantize_row_tq2_0_ref(x, y, k);
}
''', '''    block_tq2_0 * GGML_RESTRICT y = vy;
    quantize_row_tq2_0_ref(x, y, k);
}

void quantize_row_tq2_0_g128(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(k % QK_TQ2_0_G128 == 0);
    block_tq2_0_g128 * GGML_RESTRICT y = vy;
    quantize_row_tq2_0_g128_ref(x, y, k);
}
''')
sub(CQ, 'void ggml_vec_dot_q2_K_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {',
'''// two G128 blocks per Q8_K block, each half with its own scale
void ggml_vec_dot_tq2_0_g128_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    assert(n % QK_K == 0);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_tq2_0_g128 * GGML_RESTRICT x = vx;
    const block_q8_K       * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;
    float sumf = 0.0f;

    for (int i = 0; i < nb; ++i) {
        float sumh = 0.0f;
        for (int h = 0; h < 2; ++h) {
            const block_tq2_0_g128 * xb = &x[2*i + h];
            const int8_t           * q8 = y[i].qs + 128*h;
            int32_t sumi = 0;
            for (int l = 0; l < 4; ++l) {
                for (int k = 0; k < 32; ++k) {
                    sumi += q8[l*32 + k] * (((xb->qs[k] >> (l*2)) & 3) - 1);
                }
            }
            sumh += (float) sumi * GGML_CPU_FP16_TO_FP32(xb->d);
        }
        sumf += sumh * y[i].d;
    }

    *s = sumf;
}

void ggml_vec_dot_q2_K_q8_K_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {''')

# every TQ2_0 case label in ops.cpp also covers the new type
sub('ggml/src/ggml-cpu/ops.cpp', '        case GGML_TYPE_TQ2_0:\n',
    '        case GGML_TYPE_TQ2_0:\n        case GGML_TYPE_TQ2_0_G128:\n', count=7)

# only ARM gets a native kernel
F = 'ggml/src/ggml-cpu/arch-fallback.h'
s = open(F, encoding='utf-8').read().rstrip('\n') + '''

// uno-model TQ2_0_G128: NEON kernel on ARM only, generic everywhere else
#if defined(GGML_CPU_GENERIC) || !(defined(__aarch64__) || defined(__arm__) || defined(_M_ARM) || defined(_M_ARM64) || defined(_M_ARM64EC))
#define ggml_vec_dot_tq2_0_g128_q8_K_generic ggml_vec_dot_tq2_0_g128_q8_K
#endif
'''
open(F, 'w', encoding='utf-8', newline='').write(s)
print('ggml: ok')
