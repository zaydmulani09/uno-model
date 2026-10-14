# uno-model, stage 1: making Bonsai-1.7B fast on the UNO Q

The pre-registration is in [PREREG2.md](PREREG2.md), pushed before any stage 1 result. The baseline is Ternary-Bonsai-1.7B Q2_0_g64 at 3.53 tok/s with MMLU 54 / GSM8K 66 / TriviaQA 22. A same-day re-run of the baseline gave 3.56 tok/s.

| gate | result | why |
|---|---|---|
| **G2** (≥ 1.25× speed and every eval within 3 points) | **FAIL** | TQ2_0 is **2.05×** faster but MMLU is 50.5, which misses ≥ 51.0 by one question. PQ2_0 keeps quality but is only 1.12×. |
| **G3** (vocab restriction ≥ 1.10× and within 3 points) | **FAIL** | Speed passes at **1.16×**. GSM8K drops 68 → 23 and TriviaQA drops 22 → 0, because the pre-registered token set has no newline token. |
| **T** (≥ 6.24 tok/s and MMLU ≥ 50 on the best final configuration) | **FAIL** | The best final configuration is PQ2_0 without restriction: 3.94 tok/s. |

Step 3 (the NEON kernel for Q2_0 g64) **was not built**. Its pre-registered trigger is speed only, and TQ2_0 reached 7.23 tok/s, past the 4.41 tok/s trigger.

## Candidates (board, `llama-bench -p 64 -n 64 -r 3`, tok/s)

| model | pp64 1T | pp64 4T | tg64 1T | tg64 2T | tg64 3T | tg64 4T | vs baseline 3.53 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q2_0_g64 (baseline, same-day re-run) | 1.21 | 4.78 | 0.94 | 1.83 | 2.68 | 3.56 | 1.01x |
| **(a) TQ2_0** (mainline b11594) | 2.58 | 10.16 | 1.98 | 3.68 | 5.36 | **7.23** | **2.05x** |
| (b) PQ2_0 g128 (PrismML fork `9f24523`, unmodified) | 1.34 | 5.29 | 1.04 | 2.01 | 2.93 | 3.94 | 1.12x |
| (b) PQ2_0, vocab-patched fork build | 1.34 | 5.28 | 1.04 | 2.00 | 2.93 | 3.88 | 1.10x |
| PQ2_0 + vocab restriction (V32k) | 1.34 | 5.30 | 1.20 | 2.32 | 3.37 | 4.50 | 1.27x |
| TQ2_0 + V32k *(extra, not gated)* | 2.60 | 10.11 | 2.27 | 4.25 | 6.07 | 8.19 | 2.32x |
| Q2_0_g64 + V32k *(extra, not gated)* | 1.21 | 4.77 | 1.09 | 2.10 | 3.08 | 4.08 | 1.16x |

As in stage 0, the fastest thread count is 4 for everything.

**(b) Q2_0 g128 file:** `Ternary-Bonsai-1.7B-Q2_0.gguf` from prism-ml does **not load** in the current `prism` branch. Its tensors carry type id 42, which the fork now uses for upstream's g64 Q2_0, while the data is g128 (same byte size as the PQ2_0 file). PQ2_0 (type id 142) is the loadable g128 format, so it is the (b) candidate. The fork cross-built for A53 with zig without changes.

**(a) TQ2_0 is lossy for Bonsai** ([tools/tq2_loss.py](tools/tq2_loss.py), [logs/pc2/tq2_loss.json](logs/pc2/tq2_loss.json)):
- Bonsai's native scales are per 128 weights, but TQ2_0 has one scale per 256.
- 97.8% of 256-blocks have two different native scales, so **30.2% of all weights change value**: the smaller-scale half gets rounded at the larger scale.
- Almost nothing is zeroed (0.02%).

## Quality (PC, frozen stage 0 eval, identical prompts)

| model | MMLU | GSM8K | TriviaQA | G2 quality (≥ 51 / 63 / 19) |
|---|---:|---:|---:|---|
| baseline Q2_0_g64 (stage 0) | 54.0 | 66.0 | 22.0 | |
| (a) TQ2_0 | **50.5** | 73.0 | 19.5 | no (MMLU) |
| (b) PQ2_0 | 53.5 | 67.0 | 22.0 | yes |

### G2

| candidate | tg64 4T | speedup | speed ≥ 4.41 | quality | G2 |
|---|---:|---:|---|---|---|
| TQ2_0 | 7.23 | 2.05x | yes | no (MMLU 50.5 < 51.0) | **FAIL** |
| PQ2_0 | 3.94 | 1.12x | no | yes | **FAIL** |

**G2: FAIL.** The TQ2_0 MMLU miss is one question out of 200, and it is inside run-to-run noise. Running PQ2_0 twice with different builds of identical code gave 107/67/44 and 106/68/44, because the 4-slot server batches differently. The gate was pre-registered as a hard line and stays as written.

## Vocabulary restriction (G3)

**Mechanism** ([vocab/llama.cpp-vocab-restrict.patch](vocab/llama.cpp-vocab-restrict.patch), [vocab/llama.cpp-prism-vocab-restrict.patch](vocab/llama.cpp-prism-vocab-restrict.patch), [vocab/restrict.py](vocab/restrict.py)):
- The GGUF gets three extra tensors: `output_sub.weight` (the 32768 selected LM-head rows plus one zero row, in the model's own quant type), `output_sub.bias` (0, with −inf for the zero row) and `output_sub_map.weight` (I32 token → row).
- When those tensors are present, the patched qwen3 graph:
  1. multiplies only the 32769-row head;
  2. adds the bias;
  3. gathers the logits back to full vocabulary order with `get_rows` on the transposed logits.
- Tokens outside the set get −inf. Sampler, grammar and tokenizer see the full vocabulary.
- A model without those tensors runs exactly as before.

**Token set** ([vocab/rank.py](vocab/rank.py), [vocab/top32768.json](vocab/top32768.json)):
- the top 32768 tokens by frequency in wikitext-103-raw-v1 train (124.2M tokens), with all 26 added tokens forced in;
- the set covers 99.35% of corpus tokens. Bonsai's tokenizer is byte-identical to Qwen3-0.6B's.

**Best candidate** by the pre-registered rule (fastest that passes the quality bar): **PQ2_0**, so the patch was ported to the fork. Both sides of the pair below use the same patched fork build, on the board and on the PC.

tg64 at 4T: 3.88 → 4.50 tok/s = **1.159×** (≥ 1.10, ok)

| task | unrestricted | restricted | Δ | right→wrong | wrong→right |
|---|---:|---:|---:|---:|---:|
| MMLU | 53.0 | 52.5 | −0.5 | 2 | 1 |
| GSM8K | 68.0 | **23.0** | −45.0 | 48 | 3 |
| TriviaQA | 22.0 | **0.0** | −22.0 | 44 | 0 |

**G3: FAIL.** The cause is the corpus, not the mechanism. MMLU is grammar-constrained to one letter and is unaffected. Free generation broke for two reasons:
- **Formatting:** wikitext-103 is space-tokenized (`in 1914 , the`) and each line ends in `" \n"`. The bare `"\n"` token (id 198) and the unspaced `":"` (id 25) therefore never make the top 32768. Every TriviaQA answer has to stop on a newline, so none could stop.
- **GSM8K markup:** the few-shot answers use `<<a-b=c>>` calculator markup and `####`. Those tokens are also absent, so the model loops (`$140$140$140…`) or keeps "thinking".

### Post-hoc probe (not a gate; designed after seeing the G3 failure)
Same ranking, but the 256 single-byte tokens are always kept, displacing the 121 least-frequent entries ([vocab/top32768_bytes_posthoc.json](vocab/top32768_bytes_posthoc.json), [logs/analysis2_posthoc_bytes.md](logs/analysis2_posthoc_bytes.md)).

| task | unrestricted | restricted, bytes forced | right→wrong | wrong→right |
|---|---:|---:|---:|---:|
| MMLU | 53.0 | 53.5 | 0 | 1 |
| GSM8K | 68.0 | 24.0 | 47 | 3 |
| TriviaQA | 22.0 | 19.5 | 5 | 0 |

- **TriviaQA recovers** to within 3 points. The newline was the whole TriviaQA problem.
- **GSM8K does not recover.** The model still can't write `<<`/`>>`/`####` as single tokens. This head has the same tensor shapes and types as V32k, so it was not benched separately; its speed is the V32k 4.50 tok/s.
- **Lesson:** a frequency-ranked vocabulary must be built from text in the same format as what the model will produce, or the needed tokens must be pinned.

## Target T
The best final configuration is PQ2_0 without restriction (G3 failed): **3.94 tok/s, MMLU 53.5 → T: FAIL** on speed.

For the record, not as a pass: **TQ2_0 alone reaches 7.23 tok/s with MMLU 50.5**, which clears both of T's numbers. It is not the best final configuration under the pre-registered rules, because it failed the G2 quality bar on MMLU by one question.

## Demo (board, PQ2_0, unrestricted, 4 threads, greedy, 128 new tokens)

Full output: [logs/board2/demo_PQ2_0.log](logs/board2/demo_PQ2_0.log).

**1. "Explain why the sky is blue in two sentences."** (prompt 19 tok at 5.05 tok/s; 88 tok generated at **3.85 tok/s**; 26.7 s total)
> The sky appears blue because of a phenomenon called Rayleigh scattering. Sunlight, which is white light, consists of different colors with varying wavelengths. When sunlight enters Earth's atmosphere, it interacts with gas molecules and small particles, and shorter wavelengths (like blue and violet) scatter more efficiently than longer wavelengths. This scattering causes the blue light to be spread out in all directions, making the sky appear blue to our eyes.

It says "two sentences" and writes four.

**2. "A shop sells pencils at 3 for $1. How much do 12 pencils cost? Answer step by step."** (prompt 33 tok at 5.24 tok/s; 127 tok at **3.82 tok/s**; 39.7 s)
> \<think\> Okay, let's see. The shop sells pencils at 3 for $1. So, I need to figure out how much 12 pencils cost. Hmm, first, maybe I should figure out the price per pencil. If 3 pencils cost $1, then each pencil would cost $1 divided by 3. Let me do that calculation. $1 divided by 3 is approximately $0.333... But maybe they want the exact price. Wait, if 3 pencils cost $1, …

It enters thinking mode and hits the 128-token cap before answering.

**3. "Write a haiku about a small computer."** (prompt 17 tok at 5.21 tok/s; 20 tok at **3.87 tok/s**; 8.4 s)
> Silent pulse,
> data flows like a river,
> small heart beats.

## Notes and deviations
- **Builds:** all board binaries were cross-built with zig (`-mcpu=cortex_a53`, `GGML_OPENMP=OFF`) and pushed over USB ADB. PC eval binaries:
  - mainline release b11594 for TQ2_0, matching stage 0;
  - the PrismML fork built for Windows with zig (`x86_64-windows-gnu -mcpu=native`; the fork's UI embed needs `-DHOST_CXX_COMPILER`) for PQ2_0;
  - the patched fork Windows build for both sides of the G3 pair.
- **Missing vocab entries:** the 151669-entry GGUF vocab (not 151936; the padding ids aren't real tokens) means "all special tokens" = the 26 added tokens, ids 151643–151668.
- **Eval noise:** 4 parallel slots cause batch-dependent float rounding, about ±1 question per task between runs.
- **Board footprint:** all work stayed in `/home/arduino/uno-model`, with no system changes. Before: `/` 6.4G used, `/home/arduino` 598M. After cleanup: `/` 6.4G, `/home/arduino` **598M**, an exact match ([logs/board2/board_before.log](logs/board2/board_before.log), [logs/board2/board_after_cleanup.log](logs/board2/board_after_cleanup.log)).

Raw logs:
- board benches, demo and cleanup: `logs/board2/`
- PC convert, quantize and build logs, TQ2_0 loss, eval summary: `logs/pc2/`
- per-question outputs: `eval/results/*.jsonl`
- tables: `analyze2.py` → [logs/analysis2.md](logs/analysis2.md)
