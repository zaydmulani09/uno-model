"""TQ2_0_G128, part 2: ARM NEON vec_dot, llama ftype plumbing, gguf-py. Run from the llama.cpp root
after apply_ggml.py."""


def sub(p, old, new, count=1):
    s = open(p, encoding='utf-8').read()
    n = s.count(old)
    assert n == count, (p, old[:70], n)
    open(p, 'w', encoding='utf-8', newline='').write(s.replace(old, new))


# --- NEON kernel (A53: no dotprod, int16 vmlal accumulators like TQ2_0's fallback path)
sub('ggml/src/ggml-cpu/arch/arm/quants.c',
    'void ggml_vec_dot_q2_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {',
'''// TQ2_0_G128: two 128-weight blocks per 256-wide Q8_K block, each half with its own fp16 scale.
// Codes are 0..2 (weight + 1), so the integer dot is sum(q*y) - sum(y); sum(y) for a half is
// that half's 8 bsums.
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

#if defined(__ARM_NEON)
    float sumf = 0.0f;

    const uint8x16_t m3 = vdupq_n_u8(3);

    for (int i = 0; i < nb; ++i) {
        float sumh[2];

        for (int h = 0; h < 2; ++h) {
            const uint8_t * GGML_RESTRICT qs = x[2*i + h].qs;
            const int8_t  * GGML_RESTRICT q8 = y[i].qs + 128*h;

            const uint8x16_t qx0 = vld1q_u8(qs);
            const uint8x16_t qx1 = vld1q_u8(qs + 16);

            const int8x16_t sqx0 = vreinterpretq_s8_u8(vandq_u8(qx0, m3));
            const int8x16_t sqx1 = vreinterpretq_s8_u8(vandq_u8(qx1, m3));
            const int8x16_t sqx2 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(qx0, 2), m3));
            const int8x16_t sqx3 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(qx1, 2), m3));
            const int8x16_t sqx4 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(qx0, 4), m3));
            const int8x16_t sqx5 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(qx1, 4), m3));
            const int8x16_t sqx6 = vreinterpretq_s8_u8(vshrq_n_u8(qx0, 6));
            const int8x16_t sqx7 = vreinterpretq_s8_u8(vshrq_n_u8(qx1, 6));

            const int8x16_t qy0 = vld1q_s8(q8 +   0);
            const int8x16_t qy1 = vld1q_s8(q8 +  16);
            const int8x16_t qy2 = vld1q_s8(q8 +  32);
            const int8x16_t qy3 = vld1q_s8(q8 +  48);
            const int8x16_t qy4 = vld1q_s8(q8 +  64);
            const int8x16_t qy5 = vld1q_s8(q8 +  80);
            const int8x16_t qy6 = vld1q_s8(q8 +  96);
            const int8x16_t qy7 = vld1q_s8(q8 + 112);

            const int16x8_t ysum = vld1q_s16(y[i].bsums + 8*h);

#if defined(__ARM_FEATURE_DOTPROD)
            int32x4_t sumi = vdotq_s32(vdupq_n_s32(0), sqx0, qy0);
            sumi = vdotq_s32(sumi, sqx1, qy1);
            sumi = vdotq_s32(sumi, sqx2, qy2);
            sumi = vdotq_s32(sumi, sqx3, qy3);
            sumi = vdotq_s32(sumi, sqx4, qy4);
            sumi = vdotq_s32(sumi, sqx5, qy5);
            sumi = vdotq_s32(sumi, sqx6, qy6);
            sumi = vdotq_s32(sumi, sqx7, qy7);
            sumi = vsubq_s32(sumi, vpaddlq_s16(ysum));
            sumh[h] = (float) vaddvq_s32(sumi);
#else
            // 8 products of |code*y| <= 256 per lane per accumulator: no int16 overflow
            int16x8_t sumi0 = vmull_s8(vget_low_s8(sqx0), vget_low_s8(qy0));
            int16x8_t sumi1 = vmull_s8(vget_high_s8(sqx0), vget_high_s8(qy0));
            sumi0 = vmlal_s8(sumi0, vget_low_s8(sqx1),  vget_low_s8(qy1));
            sumi1 = vmlal_s8(sumi1, vget_high_s8(sqx1), vget_high_s8(qy1));
            sumi0 = vmlal_s8(sumi0, vget_low_s8(sqx2),  vget_low_s8(qy2));
            sumi1 = vmlal_s8(sumi1, vget_high_s8(sqx2), vget_high_s8(qy2));
            sumi0 = vmlal_s8(sumi0, vget_low_s8(sqx3),  vget_low_s8(qy3));
            sumi1 = vmlal_s8(sumi1, vget_high_s8(sqx3), vget_high_s8(qy3));
            sumi0 = vmlal_s8(sumi0, vget_low_s8(sqx4),  vget_low_s8(qy4));
            sumi1 = vmlal_s8(sumi1, vget_high_s8(sqx4), vget_high_s8(qy4));
            sumi0 = vmlal_s8(sumi0, vget_low_s8(sqx5),  vget_low_s8(qy5));
            sumi1 = vmlal_s8(sumi1, vget_high_s8(sqx5), vget_high_s8(qy5));
            sumi0 = vmlal_s8(sumi0, vget_low_s8(sqx6),  vget_low_s8(qy6));
            sumi1 = vmlal_s8(sumi1, vget_high_s8(sqx6), vget_high_s8(qy6));
            sumi0 = vmlal_s8(sumi0, vget_low_s8(sqx7),  vget_low_s8(qy7));
            sumi1 = vmlal_s8(sumi1, vget_high_s8(sqx7), vget_high_s8(qy7));
            sumi0 = vsubq_s16(vaddq_s16(sumi0, sumi1), ysum);
            sumh[h] = (float) vaddlvq_s16(sumi0);
#endif
        }

        sumf += y[i].d * (GGML_CPU_FP16_TO_FP32(x[2*i].d) * sumh[0] + GGML_CPU_FP16_TO_FP32(x[2*i + 1].d) * sumh[1]);
    }

    *s = sumf;
#else
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    ggml_vec_dot_tq2_0_g128_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

void ggml_vec_dot_q2_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {''')

# --- llama ftype plumbing
sub('include/llama.h', '        LLAMA_FTYPE_MOSTLY_Q2_0          = 41, // except 1d tensors\n',
    '        LLAMA_FTYPE_MOSTLY_Q2_0          = 41, // except 1d tensors\n'
    '        LLAMA_FTYPE_MOSTLY_TQ2_0_G128    = 160, // except 1d tensors (uno-model)\n')
L = 'src/llama-model-loader.cpp'
sub(L, '        case LLAMA_FTYPE_MOSTLY_TQ2_0:     name = LLAMA_FTYPE_PREFIX "TQ2_0 - 2.06 bpw ternary"; break;\n',
    '        case LLAMA_FTYPE_MOSTLY_TQ2_0:     name = LLAMA_FTYPE_PREFIX "TQ2_0 - 2.06 bpw ternary"; break;\n'
    '        case LLAMA_FTYPE_MOSTLY_TQ2_0_G128: name = LLAMA_FTYPE_PREFIX "TQ2_0_G128 - 2.13 bpw ternary"; break;\n')
sub(L, '            case GGML_TYPE_TQ2_0:   ftype = LLAMA_FTYPE_MOSTLY_TQ2_0;   break;\n',
    '            case GGML_TYPE_TQ2_0:   ftype = LLAMA_FTYPE_MOSTLY_TQ2_0;   break;\n'
    '            case GGML_TYPE_TQ2_0_G128: ftype = LLAMA_FTYPE_MOSTLY_TQ2_0_G128; break;\n')
Q = 'src/llama-quant.cpp'
# the CPU kernel pairs two blocks per 256-wide Q8_K block, so rows must be a multiple of 256
sub(Q, '    const int64_t qk_k = ggml_blck_size(target_type);\n',
    '    const int64_t qk_k = target_type == GGML_TYPE_TQ2_0_G128 ? 256 : ggml_blck_size(target_type);\n')
sub(Q, '            case GGML_TYPE_TQ2_0:   return_type = GGML_TYPE_Q4_0;   break;\n',
    '            case GGML_TYPE_TQ2_0:\n            case GGML_TYPE_TQ2_0_G128: return_type = GGML_TYPE_Q4_0; break;\n')
sub(Q, '            else if (ftype == LLAMA_FTYPE_MOSTLY_TQ1_0 || ftype == LLAMA_FTYPE_MOSTLY_TQ2_0 || ftype == LLAMA_FTYPE_MOSTLY_Q2_0) {',
    '            else if (ftype == LLAMA_FTYPE_MOSTLY_TQ1_0 || ftype == LLAMA_FTYPE_MOSTLY_TQ2_0 || ftype == LLAMA_FTYPE_MOSTLY_Q2_0 || ftype == LLAMA_FTYPE_MOSTLY_TQ2_0_G128) {')
sub(Q, '        case LLAMA_FTYPE_MOSTLY_TQ2_0:   return GGML_TYPE_TQ2_0;\n',
    '        case LLAMA_FTYPE_MOSTLY_TQ2_0:   return GGML_TYPE_TQ2_0;\n'
    '        case LLAMA_FTYPE_MOSTLY_TQ2_0_G128: return GGML_TYPE_TQ2_0_G128;\n')
sub('tools/quantize/quantize.cpp', '    { "TQ2_0",    LLAMA_FTYPE_MOSTLY_TQ2_0,    " 2.06 bpw ternarization",           },\n',
    '    { "TQ2_0",    LLAMA_FTYPE_MOSTLY_TQ2_0,    " 2.06 bpw ternarization",           },\n'
    '    { "TQ2_0_G128", LLAMA_FTYPE_MOSTLY_TQ2_0_G128, " 2.13 bpw ternarization, scale per 128 (uno-model)", },\n')

# --- gguf-py
C = 'gguf-py/gguf/constants.py'
sub(C, '    TQ2_0   = 35\n', '    TQ2_0   = 35\n    TQ2_0_G128 = 160  # uno-model\n')
sub(C, '    MOSTLY_TQ2_0         = 37  # except 1d tensors\n',
    '    MOSTLY_TQ2_0         = 37  # except 1d tensors\n    MOSTLY_TQ2_0_G128    = 160  # except 1d tensors (uno-model)\n')
sub(C, '    GGMLQuantizationType.TQ2_0:   (256, 2 + 64),\n',
    '    GGMLQuantizationType.TQ2_0:   (256, 2 + 64),\n    GGMLQuantizationType.TQ2_0_G128: (128, 2 + 32),\n')
sub('gguf-py/gguf/quants.py', 'class MXFP4(__Quant, qtype=GGMLQuantizationType.MXFP4):',
'''class TQ2_0_G128(__Quant, qtype=GGMLQuantizationType.TQ2_0_G128):
    # TQ2_0 layout over 128 weights: 32 code bytes then one fp16 scale
    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d = abs(blocks).max(axis=-1, keepdims=True)
        with np.errstate(divide="ignore"):
            id = np.where(d == 0, 0, 1 / d)
        qs = np_roundf(blocks * id)
        qs = (qs.astype(np.int8) + np.int8(1)).astype(np.uint8)

        qs = qs.reshape((n_blocks, 4, 32)) << np.array([0, 2, 4, 6], dtype=np.uint8).reshape((1, 4, 1))
        qs = qs[:, 0, :] | qs[:, 1, :] | qs[:, 2, :] | qs[:, 3, :]

        d = d.astype(np.float16).view(np.uint8)

        return np.concatenate([qs, d], axis=-1)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        qs, d = np.hsplit(blocks, [32])

        d = d.view(np.float16).astype(np.float32)

        qs = qs.reshape((n_blocks, 1, 32)) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape((1, 4, 1))
        qs = (qs & 0x03).reshape((n_blocks, -1)).astype(np.int8) - np.int8(1)

        return (d * qs.astype(np.float32))


class MXFP4(__Quant, qtype=GGMLQuantizationType.MXFP4):''')
print('rest: ok')
