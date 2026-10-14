---
license: apache-2.0
base_model:
- prism-ml/Ternary-Bonsai-1.7B-unpacked
library_name: gguf
pipeline_tag: text-generation
tags:
- gguf
- ternary
- 1.58-bit
- llama.cpp
- cortex-a53
- arduino
- edge
---

# Ternary-Bonsai-1.7B TQ2_0_G128

PrismML's Ternary-Bonsai-1.7B (a ternary Qwen3-1.7B), repacked losslessly into **TQ2_0_G128**: a new llama.cpp tensor type that combines TQ2_0's fast 2-bit layout with Bonsai's native scale of one fp16 per 128 weights. It runs **6.66 tok/s** on an Arduino UNO Q (4x Cortex-A53, no dot-product instructions). That is faster than Gemma 3 1B Q4_0 on the same board on the same day, and it scores well above Gemma on MMLU and GSM8K.

> **This file needs a patched llama.cpp.** TQ2_0_G128 is not in upstream llama.cpp; it uses ggml type id 160. Stock llama.cpp, Ollama, LM Studio and similar tools will refuse to load it. Build from the patch below.

Full methodology, pre-registrations and raw logs: [github.com/zaydmulani09/uno-model](https://github.com/zaydmulani09/uno-model) (stages 1b and 1c).

## Files

| file | what |
|---|---|
| `Ternary-Bonsai-1.7B-TQ2_0_G128.gguf` | the model (463 MB, 2.125 bits per weight; all 197 ternary tensors including the tied embedding and LM head) |
| `tq2_0_g128-b11594.patch` | the llama.cpp patch against tag `b11594`: the type, quantize and dequantize, gguf-py support, and CPU kernels (ARM NEON without dotprod, x86 AVX2, generic C) |
| `test_vec_dot.c` | kernel check: native vs generic vs an fp64 reference |

The same patch is a single commit on branch [`tq2_0-g128` of zaydmulani09/llama.cpp](https://github.com/zaydmulani09/llama.cpp/tree/tq2_0-g128). It has not been proposed upstream.

## Build and run

```bash
git clone https://github.com/zaydmulani09/llama.cpp -b tq2_0-g128
cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target llama-cli llama-server llama-bench
./build/bin/llama-server -m Ternary-Bonsai-1.7B-TQ2_0_G128.gguf -t 4
```

The patch also applies to a plain `b11594` checkout with `git apply tq2_0_g128-b11594.patch`. On a Cortex-A53 board, build with `-mcpu=cortex-a53` (or natively on the board).

### Thinking off
Bonsai inherits Qwen3's hybrid thinking. For short direct answers, turn thinking off:
- with the chat template, use `enable_thinking=false`;
- with a raw prompt, prefill an empty think block after the assistant tag:

```
<|im_start|>user
Write a haiku about a small computer.<|im_end|>
<|im_start|>assistant
<think>

</think>

```

With thinking on, the model often spends a 128-token budget inside `<think>` before answering.

## Format

| | |
|---|---|
| block | 128 weights: 32 bytes of 2-bit codes + one fp16 scale (34 bytes) |
| code layout | as TQ2_0: byte m holds weights m, m+32, m+64, m+96 at bits 0, 2, 4, 6 |
| value | (code - 1) * scale, code in {0, 1, 2} |
| matmul | against Q8_K activations; each 256-wide Q8_K block covers two G128 blocks, each half with its own scale |

**Lossless:** every tensor, dequantized, is bit-identical to PrismML's own packed PQ2_0 release (197 of 197 tensors). Against the unpacked FP16 checkpoint, 0.0025% of weights differ. Those sit in 740 groups where the FP16 file holds two magnitudes, and PQ2_0 rounds them the same way.

**Kernels** were checked over 2000 random rows of 6144 weights. The NEON kernel matches the generic C kernel exactly. AVX2 differs from it only by float rounding, 2.4e-8 relative to sum |w*a|.

## Measured

### Speed: Arduino UNO Q (Qualcomm QRB2210, 4x Cortex-A53 @ 2.0 GHz, no asimddp), same day
`llama-bench -t 4 -p 64 -n 64 -r 3`, two alternating rounds.

| model | tg64 tok/s, round 1 | round 2 | mean |
|---|---:|---:|---:|
| **Ternary-Bonsai-1.7B TQ2_0_G128** | 6.65 | 6.67 | **6.66** |
| Gemma 3 1B Q4_0 (QAT) | 6.18 | 6.26 | 6.22 |
| Qwen3.5-0.8B Q4_0 | 6.12 | 6.07 | 6.10 |

For reference, the same weights in PrismML's PQ2_0 format run 3.94 tok/s on this board, and mainline's Q2_0 g64 runs 3.53.

### Quality
Same prompts for every model: raw completion, no chat template, greedy decoding, run on an x86 PC with the same GGUF files. Accuracy % with 95% Wilson intervals.
- MMLU: 1000 questions, 0-shot, answer letter constrained by a grammar.
- GSM8K: 300 questions, 4-shot.
- TriviaQA: 500 questions, rc.nocontext, 5-shot exact match.

| model | MMLU (1000) | GSM8K (300) | TriviaQA (500) |
|---|---:|---:|---:|
| **Ternary-Bonsai-1.7B TQ2_0_G128** | **46.8** [43.7, 49.9] | **62.7** [57.1, 67.9] | **20.8** [17.5, 24.6] |
| Ternary-Bonsai-1.7B PQ2_0 (PrismML format, same weights) | 46.9 [43.8, 50.0] | 62.7 [57.1, 67.9] | 21.2 [17.8, 25.0] |
| Gemma 3 1B Q4_0 (QAT) | 38.4 [35.4, 41.5] | 12.0 [8.8, 16.2] | 19.8 [16.5, 23.5] |
| Qwen3.5-0.8B Q4_0 | 38.5 [35.5, 41.6] | 26.3 [21.7, 31.6] | 10.4 [8.0, 13.4] |
| Qwen3-0.6B Q4_0 | 29.5 [26.8, 32.4] | 30.0 [25.1, 35.4] | 8.8 [6.6, 11.6] |

The release rule was pre-registered: be at least as fast as same-day Gemma 3 1B Q4_0, and score higher on both MMLU and GSM8K with non-overlapping 95% intervals. It passed on all three counts. TriviaQA overlaps with Gemma.

## Honest limits
- **Needs the patch.** No upstream llama.cpp, Ollama or LM Studio build can load this file.
- **Speed is from one board**, the Arduino UNO Q's Cortex-A53, at 4 threads with greedy decoding and short contexts (64 tokens). Long contexts will be slower. Other CPUs have not been benchmarked. On CPUs with dotprod (A55, A76 and later), the NEON kernel still works but doesn't use those instructions.
- **Quality was measured on a PC, not the board.** The weights are identical; on the board only kernel float rounding differs (matched against the generic kernel at 0 difference).
- **The scores are modest in absolute terms.** MMLU is about 47%, not the 54% an earlier 200-question run suggested. TriviaQA is about 21%: open-domain recall is weak at this size.
- **The eval is raw completion** with fixed few-shot prompts, not the chat template, so numbers are comparable across these models but not to official leaderboard figures. This format can understate instruction-tuned baselines; Gemma's 12% on GSM8K in particular likely understates it (its chat-format score was not measured here).
- **Answers are verbose.** With thinking off, the model still uses heavy markdown and can run long on short questions.
- **Inherited behavior.** This is PrismML's model, unchanged; any biases or failure modes of Bonsai and Qwen3 carry over.

## Credits and licenses
- **Model:** [Ternary-Bonsai-1.7B](https://huggingface.co/prism-ml/Ternary-Bonsai-1.7B-unpacked) by PrismML, Apache-2.0. This repo only repacks its weights into a different container, with no retraining.
- **Base architecture:** [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) by the Qwen team, Apache-2.0.
- **Runtime:** [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp). The patch derives from llama.cpp and keeps llama.cpp's MIT license. The model file is Apache-2.0.
- TQ2_0's layout comes from llama.cpp's existing TQ2_0 type, and Q8_K from ggml.
