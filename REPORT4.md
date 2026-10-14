# uno-model, stage 1c: baselines and release rule R

The pre-registration is in [PREREG4.md](PREREG4.md), pushed before any stage 1c result. PREREG3's target **T stays recorded as FAIL**. R is a new release rule: it was declared because T's MMLU bar came from a noisy 200-question number.

**R: PASS.** TQ2_0_G128 is faster than the same-day Gemma 3 1B Q4_0 control and beats it on MMLU and GSM8K with non-overlapping 95% intervals.

**Published:**
- **Model:** [huggingface.co/zaydmulani/Ternary-Bonsai-1.7B-TQ2_0_G128](https://huggingface.co/zaydmulani/Ternary-Bonsai-1.7B-TQ2_0_G128). It contains the GGUF, the kernel patch, the kernel test and the model card.
- **llama.cpp patch branch:** [zaydmulani09/llama.cpp @ tq2_0-g128](https://github.com/zaydmulani09/llama.cpp/tree/tq2_0-g128), a single commit on `b11594`. **No upstream PR was opened.**

## Rule R

| condition | TQ2_0_G128 | Gemma 3 1B Q4_0 | ok |
|---|---|---|---|
| speed: mean tg64 at 4 threads, same day | **6.66** | 6.22 | yes |
| MMLU higher, 95% intervals disjoint | **46.8** [43.7, 49.9] | 38.4 [35.4, 41.5] | yes |
| GSM8K higher, 95% intervals disjoint | **62.7** [57.1, 67.9] | 12.0 [8.8, 16.2] | yes |

## Eval v3 (accuracy % [95% Wilson])

MMLU 1000, GSM8K 300, TriviaQA 500, with the same prompts as stages 0–1b, one server at a time, on the mainline b11594 Windows build. The Bonsai rows are reused from stage 1b.

| model | MMLU (1000) | GSM8K (300) | TriviaQA (500) |
|---|---:|---:|---:|
| **Ternary-Bonsai-1.7B TQ2_0_G128** | 46.8 [43.7, 49.9] | 62.7 [57.1, 67.9] | 20.8 [17.5, 24.6] |
| Ternary-Bonsai-1.7B PQ2_0 | 46.9 [43.8, 50.0] | 62.7 [57.1, 67.9] | 21.2 [17.8, 25.0] |
| Gemma 3 1B Q4_0 (QAT) | 38.4 [35.4, 41.5] | 12.0 [8.8, 16.2] | 19.8 [16.5, 23.5] |
| Qwen3.5-0.8B Q4_0 | 38.5 [35.5, 41.6] | 26.3 [21.7, 31.6] | 10.4 [8.0, 13.4] |
| Qwen3-0.6B Q4_0 | 29.5 [26.8, 32.4] | 30.0 [25.1, 35.4] | 8.8 [6.6, 11.6] |

Qwen3.5-0.8B Q4_0 was quantized here from `Qwen/Qwen3.5-0.8B` (text model) with mainline b11594 `convert_hf_to_gguf.py` + `llama-quantize Q4_0`. The pre-registered unsloth fallback was not needed.

**Raw completion understates instruction-tuned baselines.** Every model gets raw-completion few-shot prompts with no chat template, which keeps them identical across models but disadvantages chat-tuned ones. Gemma's 12.0 on GSM8K is the clearest case; its chat-format GSM8K was not measured. The model card says so.

## Board, same day, tg64 at 4 threads (tok/s)

`llama-bench -t 4 -p 64 -n 64 -r 3`, two alternating rounds in the pre-registered order. Temperatures rose from 38.7 to 63.5 °C over the six runs.

| model | round 1 | round 2 | mean |
|---|---:|---:|---:|
| Ternary-Bonsai-1.7B TQ2_0_G128 | 6.65 | 6.67 | **6.66** |
| Gemma 3 1B Q4_0 | 6.18 | 6.26 | 6.22 |
| Qwen3.5-0.8B Q4_0 | 6.12 | 6.07 | 6.10 |

TQ2_0_G128 matches its stage 1b figure (6.68); Gemma matches its stage 0 figure (6.24).

## Release artifacts
- **HF model card:** credits PrismML Bonsai (Apache-2.0), Qwen3 (Apache-2.0) and llama.cpp. It contains the measured tables, the thinking-off note and an honest-limits section, and has no em dashes. A copy is at [release/MODEL_CARD.md](release/MODEL_CARD.md).
- **License note:** llama.cpp is MIT-licensed, not Apache-2.0, so the card states the patch keeps llama.cpp's MIT license. The model file is Apache-2.0.
- **Patch:** the TQ2_0_G128 type only, without the stage 1 vocab-restriction patch. It is [tq2_0_g128/tq2_0_g128-b11594.patch](tq2_0_g128/tq2_0_g128-b11594.patch): 18 files, A53 NEON without dotprod, AVX2 and generic C. That same patch is the one commit on the fork branch. A clean worktree with only this patch cross-built `llama-bench` and `llama-completion` for A53 successfully.
- **GGUF:** sha256 `0ed8fd77828545420b5f7a0136c72bbffa836aadb005fbb367fd549773bf3cc9`, the same file benched on the board.

## Safety
All work stayed in `/home/arduino/uno-model`, with no system changes. Before: `/` 6.4G, `/home/arduino` 598M. After cleanup: `/` 6.4G, `/home/arduino` **598M**, an exact match ([logs/board4/board_before.log](logs/board4/board_before.log), [logs/board4/board_after_cleanup.log](logs/board4/board_after_cleanup.log)).

Raw logs:
- board rounds and cleanup: `logs/board4/`
- Qwen3.5 convert and quantize logs, eval summary: `logs/pc4/`
- per-question outputs: `eval/results/*_v3.jsonl`
- tables: [analyze4.py](analyze4.py) → [logs/analysis4.md](logs/analysis4.md)
