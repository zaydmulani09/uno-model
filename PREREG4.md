# Stage 1c pre-registration

Committed and pushed before any stage 1c result existed. The thresholds below do not change after results.

## Status of earlier gates
- **PREREG3 target T stays recorded as FAIL** (TQ2_0_G128: MMLU 46.8 < 51.0 on eval v3). Nothing in this stage changes that record.
- **Rule R below is a new rule.** It is declared because T's MMLU bar of 51 was derived from a noisy 200-question estimate: Bonsai's 54.0 on 200 questions had a 95% interval of [47.1, 60.8], and the 1000-question value is 46.9. R does not re-score T. It replaces the release criterion from here on, by comparing against same-protocol baselines instead of a fixed number.

## Baselines on eval v3
The eval set is `eval/evalset_v3.jsonl` (MMLU 1000, GSM8K 300, TriviaQA 500), with the same prompts, scoring and greedy decoding as stages 0–1b. Each model runs alone, with one llama-server at a time, on the mainline b11594 Windows release build.

| model | source |
|---|---|
| Gemma 3 1B Q4_0 | the stage 0 file (`bartowski/google_gemma-3-1b-it-qat-GGUF`, QAT Q4_0) |
| Qwen3-0.6B Q4_0 | the stage 0 file (quantized here from `Qwen/Qwen3-0.6B`) |
| Qwen3.5-0.8B Q4_0 | quantized here from `Qwen/Qwen3.5-0.8B` HF weights (mainline b11594 `convert_hf_to_gguf.py --outtype f16`, text model, then `llama-quantize Q4_0`). If that conversion fails, `unsloth/Qwen3.5-0.8B-GGUF` `Qwen3.5-0.8B-Q4_0.gguf` is used, and the substitution is reported. |

Reused unchanged from stage 1b: Bonsai PQ2_0 and TQ2_0_G128 on eval v3.

Every accuracy is reported with a 95% Wilson interval.

## Board, same-day speed control
- `llama-bench -t 4 -p 64 -n 64 -r 3` with the stage 1b patched mainline A53 build.
- Two alternating rounds, in this order: TQ2_0_G128 → Gemma 3 1B Q4_0 → Qwen3.5-0.8B Q4_0 → TQ2_0_G128 → Gemma 3 1B Q4_0 → Qwen3.5-0.8B Q4_0.
- Each model's speed is the mean tg64 tok/s over its two rounds.

## Rule R (release)
R **passes** iff all three of the following hold:
1. **Speed:** TQ2_0_G128 mean tg64 at 4 threads ≥ the same-day Gemma 3 1B Q4_0 mean tg64 at 4 threads;
2. **MMLU:** TQ2_0_G128 MMLU accuracy > Gemma's, **and** the 95% Wilson intervals don't overlap (TQ2_0_G128's lower bound > Gemma's upper bound);
3. **GSM8K:** the same as for MMLU.

TriviaQA and the Qwen baselines are reported only; they don't enter R.

## If R passes
- **Public Hugging Face model repo** `zaydmulani/Ternary-Bonsai-1.7B-TQ2_0_G128`, containing:
  - the TQ2_0_G128 GGUF;
  - the TQ2_0_G128 kernel patch for llama.cpp b11594 (A53 NEON without dotprod, AVX2, generic);
  - a model card crediting PrismML Bonsai, Qwen3 and llama.cpp, with the measured table, the thinking-off note and honest limits, and no em dashes.
- **llama.cpp patch branch:** prepared on a fork, with **no upstream PR opened**.

## If R fails
Nothing is published, and the report names which comparison failed.

## Safety
- Work only in `/home/arduino/uno-model` on the board, with no system changes.
- Record `df -h` and `free -m` before and after.
- Delete everything at the end and confirm `df -h` matches the start.
