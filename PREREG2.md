# Stage 1 pre-registration

Committed and pushed before any stage 1 result existed. The thresholds below do not change after results.

## Baseline (from stage 0, fixed)
Ternary-Bonsai-1.7B Q2_0_g64, mainline llama.cpp b11594, run on the board.

| | value |
|---|---|
| tg64 at 4 threads | **3.53 tok/s** |
| MMLU | **54.0** |
| GSM8K | **66.0** |
| TriviaQA | **22.0** |

## Protocols (unchanged from stage 0)
- **Throughput:** `llama-bench -t 1,2,3,4 -p 64 -n 64 -r 3` on the board. The gates use **tg64 tok/s at 4 threads**.
- **Builds:** cross-compiled with zig (`-target aarch64-linux-gnu.2.41 -mcpu=cortex_a53`, `GGML_OPENMP=OFF`), pushed over ADB.
- **Quality:** the same frozen `eval/evalset.jsonl` (200 MMLU / 100 GSM8K / 200 TriviaQA), the same prompts, the same scoring and the same greedy decoding (`eval/run_eval.py`). It runs on the PC, using a PC build of the same source tree as each candidate.

## Candidates
- **(a) Bonsai-TQ2_0.** Built from `prism-ml/Ternary-Bonsai-1.7B-unpacked` (FP16):
  - mainline b11594 `convert_hf_to_gguf.py --outtype f16`;
  - then `llama-quantize TQ2_0 --token-embedding-type tq2_0 --output-tensor-type tq2_0`, so every ternary 2-D weight, including the tied embedding/LM head, is TQ2_0.
  - Bench with the mainline b11594 board build.
  - Note: TQ2_0 has one scale per 256 weights, while Bonsai's native scales are per 128. Wherever the two 128-groups of a block differ, the conversion is lossy. That loss is part of what is being tested; the fraction of changed weights will be reported.
- **(b) Bonsai on the PrismML fork.** `PrismML-Eng/llama.cpp`, `prism` branch, commit `9f24523`. Files `Ternary-Bonsai-1.7B-PQ2_0.gguf` and `Ternary-Bonsai-1.7B-Q2_0.gguf` from `prism-ml/Ternary-Bonsai-1.7B-gguf` (g128). Each one that cross-builds and loads is benched. If it fails to build for A53, it is reported as a failure, not worked around.
- **(c) A53 NEON kernel for mainline Q2_0 g64 (conditional).**
  - **Trigger:** built only if neither (a) nor (b) reaches ≥ 1.25 × 3.53 = **4.41 tok/s** tg64 at 4 threads. The trigger is speed only, as worded in the brief.
  - **Quality:** the kernel computes the same function on the same weights as the baseline. It is not re-evaluated on the PC, where it wouldn't run. Instead, the board checks equivalence:
    1. a vec_dot unit test against the scalar reference on random data, with max relative error reported;
    2. the first 50 MMLU questions run on the board with the old and new kernel, which must give identical answers.

    If both checks pass, the candidate inherits the baseline quality numbers.

## Best candidate
Among candidates that pass the quality part of G2 (each eval no more than 3 points below baseline), the fastest tg64 at 4 threads is the best candidate. If none passes, the best candidate is the baseline itself (mainline Q2_0_g64).

## Vocabulary restriction (applied to the best candidate)
- **Token set.** The 32768 most frequent Qwen3 tokens, counted by tokenizing a fixed corpus that is not the eval questions: **wikitext-103-raw-v1 train split, all of it** (`Salesforce/wikitext`), with the model's own tokenizer.
  - All special/added tokens (id ≥ 151643) are always included and count toward the 32768.
  - Remaining slots are filled by count; ties go to the lower id.
  - The set is saved to the repo as `vocab/top32768.json`.
- **Mechanism.** A llama.cpp patch:
  - the LM head multiplies only the 32768 selected rows;
  - the resulting logits are scattered back to the full 151936-entry vocabulary, with every other token set to −inf.

  Sampling, grammar and tokenizer see the full vocabulary.

## Gates
- **G2: PASS** iff some candidate has tg64 at 4 threads ≥ **4.41 tok/s** (1.25 × 3.53) **and** each of MMLU ≥ **51.0**, GSM8K ≥ **63.0**, TriviaQA ≥ **19.0** (no more than 3 points below baseline; higher is fine).
- **G3: PASS** iff vocabulary restriction on the best candidate gives tg64 at 4 threads ≥ **1.10×** the same candidate without restriction, **and** each eval is no more than 3 points below the same candidate without restriction.
  - Reported regardless: a paired flip table per task (right→wrong, wrong→right).
- **Target T: PASS** iff the best final configuration (best candidate, with restriction if G3 passed, without otherwise) reaches tg64 at 4 threads ≥ **6.24 tok/s** **and** MMLU ≥ **50.0**.

## Demo
Three fixed prompts, run end to end on the board with the best final configuration (`llama-cli`/`llama-simple`-style generation, greedy, 128 new tokens). Outputs and timings are printed verbatim.
1. "Explain why the sky is blue in two sentences."
2. "A shop sells pencils at 3 for $1. How much do 12 pencils cost? Answer step by step."
3. "Write a haiku about a small computer."

## Safety
- Work only inside `/home/arduino/uno-model` on the board.
- Record `df -h` and `free -m` before and after.
- Delete everything at the end and confirm `df -h` matches the start.
