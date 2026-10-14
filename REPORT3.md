# uno-model, stage 1b: a lossless fast ternary type for Bonsai on the UNO Q

The pre-registration is in [PREREG3.md](PREREG3.md), pushed before any stage 1b result. The reference is Bonsai PQ2_0, which is lossless, at 3.94 tok/s.

| gate | result | numbers |
|---|---|---|
| **G4** (TQ2_0_G128 ≥ 6.24 tok/s and every eval no more than 2 points below PQ2_0) | **PASS** | **6.68 tok/s (1.70× PQ2_0)**. MMLU −0.1, GSM8K ±0.0, TriviaQA −0.4. |
| **G5** (vocab v2 ≥ 1.10× and every eval no more than 3 points below unrestricted) | **FAIL** | Speed passes at **1.135×** (6.68 → 7.58). MMLU −0.3 and GSM8K −0.3 pass, but **TriviaQA −4.2** fails. |
| **T** (best config ≥ 6.24 tok/s and MMLU ≥ 51) | **FAIL** | The best config, TQ2_0_G128, reaches 6.68 tok/s but **MMLU is 46.8** on eval v3. |

Because T failed, no Hugging Face repo was created; the pre-registration made the release conditional on T.

**Why T can't pass:** the PQ2_0 reference itself scores MMLU **46.9** [43.8, 50.0] on the 1000-question set. The stage 0/1 figure of 54.0 came from 200 questions, and its 95% interval [47.1, 60.8] already reached down to the true value. No lossless Bonsai configuration can reach MMLU 51 on this eval, so T's MMLU bar was set from a noisy small-sample estimate. The speed half of T is met.

## TQ2_0_G128

Patches: [tq2_0_g128/](tq2_0_g128/), and the full diff against b11594 is [llama.cpp-b11594-uno-model.patch](tq2_0_g128/llama.cpp-b11594-uno-model.patch), 22 files.

- **Block:** 128 weights = 32 bytes of 2-bit codes in TQ2_0's layout (byte m holds weights m, m+32, m+64, m+96) + one fp16 scale. That is 34 bytes, 2.125 bpw. `GGML_TYPE_TQ2_0_G128 = 160`, deliberately far from upstream ids.
- **Code paths:**
  - C reference quantize and dequantize, row validation, `llama-quantize` ftype `TQ2_0_G128`;
  - gguf-py quantize and dequantize;
  - CPU vec_dot against Q8_K.
- **The vec_dot:**
  - Each 256-wide Q8_K block covers two G128 blocks. Each half gets its own integer sum, minus that half's 8 `bsums`, times its own fp16 scale.
  - **NEON:** no dotprod (A53), `vmull_s8`/`vmlal_s8` into int16 accumulators.
  - **AVX2:** `maddubs`, added for the PC evals.
  - **Generic C:** everything else.
- **Kernel tests** ([tq2_0_g128/test_vec_dot.c](tq2_0_g128/test_vec_dot.c)): 2000 random ternary rows of 6144 weights, with independent fp16 scales per 128.

| kernel | vs generic C | vs fp64 reference (error / Σ\|w·a\|) |
|---|---|---|
| NEON, on the board | **identical** (0) | 1.2e-8 |
| AVX2, on the PC | 2.4e-8 | 2.4e-8 |

Quantize and dequantize round-trip exactly in both C and gguf-py.

### Lossless conversion
`prism-ml/Ternary-Bonsai-1.7B-unpacked` FP16 → `convert_hf_to_gguf.py --outtype f16` → `llama-quantize TQ2_0_G128`, with every ternary 2-D tensor (197, including the tied embedding/LM head) in TQ2_0_G128. The file is 463,286,240 bytes; PQ2_0's is 463,290,464.

Each tensor was dequantized to fp32 and hashed per tensor ([tq2_0_g128/dequant_hash.py](tq2_0_g128/dequant_hash.py), [logs/pc3/](logs/pc3/)):
- **vs PQ2_0:** **197/197 tensors bit-identical** (raw fp32 bytes).
- **vs the FP16 source:**
  - 42,458 of 1.72B weights (0.0025%) differ, in 740 of 13.4M groups ([logs/pc3/compare_f16.json](logs/pc3/compare_f16.json));
  - every one of those groups holds **two different magnitudes** in PrismML's unpacked FP16, an artifact of the unpacking;
  - both PQ2_0 and TQ2_0_G128 map such a group to its max magnitude. In other words, PrismML's own packed release does the same thing.

## Throughput on the board (`llama-bench -p 64 -n 64 -r 3`, tok/s)

| model | pp64 1T | pp64 4T | tg64 1T | tg64 2T | tg64 3T | tg64 4T | vs PQ2_0 3.94 |
|---|---:|---:|---:|---:|---:|---:|---:|
| PQ2_0 (stage 1, fork) | 1.34 | 5.29 | 1.04 | 2.01 | 2.93 | 3.94 | 1.00x |
| **TQ2_0_G128** | 2.35 | 9.20 | 1.82 | 3.38 | 4.93 | **6.68** | **1.70x** |
| TQ2_0_G128 + vocab v2 | 2.36 | 9.27 | 2.08 | 3.88 | 5.59 | 7.58 | 1.93x |

Four threads is fastest for every config. For comparison, stage 1's lossy TQ2_0 (one scale per 256) ran at 7.23 tok/s. Keeping per-128 scales costs about 8% of that speed and removes the 30% weight corruption.

## Vocabulary restriction v2

The ranker is [vocab/rank_v2.py](vocab/rank_v2.py) and the kept set is [vocab/top32768_v2.json](vocab/top32768_v2.json). Each source has equal weight:

| source | docs | dropped (8-gram overlap with an eval question) | tokens |
|---|---:|---:|---:|
| FineWeb-edu sample-10BT shard 0 | 20,000 | 22 | 20.6M |
| UltraChat-200k train_sft, rendered with Bonsai's chat template | 10,000 | 2 | 11.9M |
| codeparrot-clean-valid shard 0 | 10,000 | 2 | 23.0M |
| GSM8K **train** | 7,473 | 16 | 1.4M |

- **Forced:** 286 tokens: the 256 single bytes, the 26 added tokens and the chat-template scaffold (`user`, `assistant`, `system`, `\n`, `\n\n`, `<|im_start|>`, `<|im_end|>`, `<think>`, `</think>`).
- **Weighted coverage:** 97.1%.
- **Fixed since stage 1:** `\n`, `:`, `<<`, `>>`, `####` and every chat token are now in the set.

## Quality, eval v3 (accuracy %, [95% Wilson])

Eval set: [eval/evalset_v3.jsonl](eval/evalset_v3.jsonl), with the same prompt builders and scoring as stage 0 and seed 0.

| model | MMLU (1000) | GSM8K (300) | TriviaQA (500) |
|---|---:|---:|---:|
| PQ2_0 (reference) | 46.9 [43.8, 50.0] | 62.7 [57.1, 67.9] | 21.2 [17.8, 25.0] |
| TQ2_0_G128 | 46.8 [43.7, 49.9] | 62.7 [57.1, 67.9] | 20.8 [17.5, 24.6] |
| TQ2_0_G128 + vocab v2 | 46.5 [43.4, 49.6] | 62.3 [56.7, 67.6] | 16.6 [13.6, 20.1] |

### G4: TQ2_0_G128 vs PQ2_0 (paired bootstrap 95% CI)

| task | PQ2_0 | G128 | Δ | 95% CI | right→wrong | wrong→right |
|---|---:|---:|---:|---|---:|---:|
| MMLU | 46.9 | 46.8 | −0.1 | [−0.7, +0.5] | 5 | 4 |
| GSM8K | 62.7 | 62.7 | +0.0 | [−2.3, +2.3] | 6 | 6 |
| TriviaQA | 21.2 | 20.8 | −0.4 | [−1.2, +0.4] | 3 | 1 |

The weights are bit-identical, so the flips come only from kernel arithmetic: PQ2_0 runs on the fork's x86 kernel, and G128 on the new AVX2 kernel with different float summation order. Every interval contains 0.

### G5: vocab v2 vs unrestricted TQ2_0_G128 (paired bootstrap 95% CI)

| task | unrestricted | v2 | Δ | 95% CI | right→wrong | wrong→right |
|---|---:|---:|---:|---|---:|---:|
| MMLU | 46.8 | 46.5 | −0.3 | [−0.8, +0.2] | 5 | 2 |
| GSM8K | 62.7 | 62.3 | −0.3 | [−2.3, +2.0] | 6 | 5 |
| TriviaQA | 20.8 | **16.6** | **−4.2** | [−6.4, −2.0] | 28 | 7 |

The v2 fix worked for formatting: GSM8K, which collapsed in stage 1 (68 → 23), is now flat. The new failure is **rare proper nouns**:
- **All 28 TriviaQA right→wrong flips** are answers that use a token outside the kept set.
- The model can still spell such a word from smaller pieces, but its preferred single token is gone, and greedy decoding switches answer. Examples: "Mozart" → "Johann Sebastian Bach", "Kazakhstan" → "Kyiv", "Paul von Hindenburg" → "Paul von Schleim", "SpongeBob SquarePants" → "Pinocchio".
- The demo shows the same thing: under v2 the sky prompt says "Ray optics" instead of "Rayleigh scattering".
- **Lesson:** a frequency-ranked head is safe for grammar-constrained or formatted tasks, but it costs open-domain factual recall, which lives in the long tail of the vocabulary.

## Target T
- **Eligible configs:** PQ2_0 (3.94 tok/s) and TQ2_0_G128 (6.68; G4 quality passed). V2 is not eligible because it failed G5 quality.
- **Best config:** **TQ2_0_G128**, at 6.68 tok/s (≥ 6.24, ok) with MMLU 46.8 (< 51.0, **fail**). **T: FAIL.**

## Demo (board, best config TQ2_0_G128, thinking disabled, greedy, 128 new tokens, 4 threads)

Full output: [logs/board3/demo_G128.log](logs/board3/demo_G128.log). The v2 run is in [logs/board3/demo_G128-V2.log](logs/board3/demo_G128-V2.log).

**1. "Explain why the sky is blue in two sentences."** (prompt 23 tok at 9.07 tok/s; 85 tok generated at **6.55 tok/s**)
> The sky appears blue because of a phenomenon called Rayleigh scattering. Sunlight, which is white light, consists of different colors with varying wavelengths. When sunlight passes through the Earth's atmosphere, shorter wavelengths of light (like blue and violet) are scattered more efficiently in all directions, while longer wavelengths (like red and yellow) are scattered less. This scattering makes blue light more visible, giving the sky its characteristic blue color.

It is still not two sentences.

**2. "A shop sells pencils at 3 for $1. How much do 12 pencils cost? Answer step by step."** (prompt 37 tok at 9.22 tok/s; 127 tok at **6.35 tok/s**)
> To find out how much 12 pencils cost, we can break the problem down step by step.
> ### **Step 1: Understand the pricing** The shop sells pencils at **3 for $1**. This means for every 3 pencils, the cost is **$1**.
> ### **Step 2: Find the cost per pencil** We can divide the cost of 3 pencils by 3 to find the cost per pencil: $\text{Cost per pencil} = \frac{\$1}{3} = \$\frac{1}{3}$ So, each pencil costs **

It hits the 128-token cap before the answer. With thinking disabled it now answers directly (stage 1's run spent its tokens inside `<think>`), but its markdown is verbose.

**3. "Write a haiku about a small computer."** (prompt 21 tok at 9.17 tok/s; 16 tok at **6.52 tok/s**)
> Silent pulse,
> data flows like a river,
> small mind processes.

## Notes and deviations
- **AVX2 kernel (not pre-registered).** PREREG3 said x86 would use the generic C kernel. The first G128 eval ran at about 8 prompt tok/s per slot, which would have taken many hours, so I added an AVX2 kernel and verified it against generic C (2.4e-8 of Σ|w·a|). The partial generic-kernel run was discarded.
- **Eval re-run.** One queued job and one manual job briefly ran the G128 and V2 evals at the same time on the same port, so their answers could have mixed. Both runs were killed, their partial outputs deleted, and both re-run one after the other with a check that only one server was alive. The PQ2_0 eval ran alone the whole time.
- **Builds:** board binaries cross-built with zig (`-mcpu=cortex_a53`); PC binaries with zig `x86_64-windows-gnu -mcpu=native`. PQ2_0 was evaluated on the unmodified-kernel fork build from stage 1.
- **Board footprint:** all work stayed in `/home/arduino/uno-model`, with no system changes. Before: `/` 6.4G, `/home/arduino` 598M. After cleanup: `/` 6.4G, `/home/arduino` **598M**, an exact match ([logs/board3/board_before.log](logs/board3/board_before.log), [logs/board3/board_after_cleanup.log](logs/board3/board_after_cleanup.log)).

Raw logs:
- board benches, kernel test, demos and cleanup: `logs/board3/`
- PC hashes, F16 comparison, x86 kernel test, quantize log, eval summary: `logs/pc3/`
- per-question outputs: `eval/results/*_v3.jsonl`
- tables: [analyze3.py](analyze3.py) → [logs/analysis3.md](logs/analysis3.md)
