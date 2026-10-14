# Stage 1b pre-registration

Committed and pushed before any stage 1b result existed. The thresholds below do not change after results.

**Reference:** Ternary-Bonsai-1.7B PQ2_0 (PrismML fork `9f24523`, lossless g128) at **3.94 tok/s** tg64 at 4 threads, from stage 1.

## New type: TQ2_0_G128 (patched mainline b11594)
- **Block:** 128 weights = 32 bytes of 2-bit codes in TQ2_0's layout (byte m holds weights m, m+32, m+64, m+96 at bits 0/2/4/6) + one fp16 scale `d`. That is 34 bytes, 2.125 bpw.
- **Type id:** `GGML_TYPE_TQ2_0_G128 = 160`. It is deliberately far from upstream ids, so files can't be mistaken for a future upstream type.
- **Quantize:** `d = max|w|` per 128, `q = round(w/d) + 1`.
- **Dequantize:** `w = (q - 1) * d`.
- **Coverage:** C reference and gguf-py implementations; `llama-quantize` ftype `TQ2_0_G128`.
- **CPU vec_dot against Q8_K:**
  - Each 256-wide Q8_K block covers two G128 blocks. Each half gets its own integer sum, minus that half's 8 `bsums`, times its own fp16 scale.
  - NEON path: no `dotprod`, using `vmlal_s8` into int16 accumulators, as TQ2_0's non-dotprod path does.
  - Generic C fallback for x86 (the PC eval).
- **Bonsai conversion:** `prism-ml/Ternary-Bonsai-1.7B-unpacked` FP16 → mainline `convert_hf_to_gguf.py --outtype f16` → `llama-quantize TQ2_0_G128`, with token embedding and output also TQ2_0_G128.
- **Lossless check:** every tensor of the TQ2_0_G128 GGUF is dequantized to fp32 and compared bit-for-bit with:
  - the dequantized PQ2_0 GGUF (fork gguf-py);
  - the F16 source.

  Per-tensor sha256 over the fp32 bytes is reported. A mismatch is reported as "not lossless"; it is not fixed silently.

## Vocabulary restriction v2
- **K = 32768** token ids.
- **Always kept:**
  - the 256 single-byte tokens (ids 0–255);
  - all added/special tokens (ids 151643–151668, which includes `<|im_start|>`, `<|im_end|>`, `<think>`, `</think>`);
  - every token the model's chat template emits around a turn. Those are covered by the added tokens plus the template text, which appears in the chat corpus.
- **Ranking corpus.** Each of the four sources gets equal weight: a token's score is the mean, over sources, of its count divided by that source's total token count. The rest of the 32768 are filled by score, with ties going to the lower id. The sources:
  1. FineWeb-edu `sample-10BT` shard 0, first 20,000 documents;
  2. UltraChat-200k `train_sft`, first 10,000 conversations, rendered with Bonsai's own chat template (`apply_chat_template`, thinking disabled);
  3. codeparrot-clean-valid shard 0, first 10,000 files;
  4. GSM8K **train** split, all 7,473 problems, in the eval's `Question: …\nAnswer: …` format, including `<<a=b>>` and `####`.
- **Contamination guard.** Any document sharing an 8-word shingle with any eval question in the stage 1b eval set is dropped. The dropped count is reported.
- **Mechanism.** The stage 1 patch: `output_sub` / `output_sub.bias` / `output_sub_map` tensors and a `get_rows` scatter to the full vocabulary with −inf. Saved as `vocab/top32768_v2.json`.

## Eval set v3
Built by the same prompt builders and scoring as stage 0, with seed 0, in `eval/evalset_v3.jsonl`:
- **MMLU:** 1000 questions (test);
- **GSM8K:** 300 (test, 4-shot from the first 4 train problems);
- **TriviaQA:** 500 (rc.nocontext validation, 5-shot).

Same prompts for every candidate, greedy decoding.

**Reporting:**
- every accuracy gets a 95% Wilson interval;
- every paired comparison gets a 95% paired-bootstrap interval on the difference (10,000 resamples, seed 0), plus flip counts.

Gates use point estimates.

## Models evaluated
- **PQ2_0:** the reference, on the fork build.
- **TQ2_0_G128:** on the patched mainline build.
- **TQ2_0_G128 + vocab v2:** on the patched mainline build.

## Gates
"No more than X points below" is the reading of "within X points" used in stages 1 and 1b. |Δ| is reported too.
- **G4: PASS** iff TQ2_0_G128 has tg64 at 4 threads ≥ **6.24 tok/s** on the board **and**, for each of MMLU, GSM8K and TriviaQA, its accuracy is ≥ PQ2_0's accuracy − **2.0** points (eval v3).
- **G5: PASS** iff, on TQ2_0_G128, vocab v2 gives tg64 at 4 threads ≥ **1.10×** the same model without restriction (same binary, same session), **and** each eval is ≥ the unrestricted accuracy − **3.0** points. A paired-flip report is always given.
- **Best config:** the fastest tg64 at 4 threads among the following eligible configs:
  - PQ2_0 (always eligible);
  - TQ2_0_G128, if it passes G4's quality part;
  - TQ2_0_G128 + v2, if it passes G4's quality part and G5's quality part.
- **Target T: PASS** iff the best config has tg64 at 4 threads ≥ **6.24 tok/s** and MMLU ≥ **51.0** (eval v3).

## Demo
Three prompts on the board with the best config, run end to end:
1. "Explain why the sky is blue in two sentences."
2. "A shop sells pencils at 3 for $1. How much do 12 pencils cost? Answer step by step."
3. "Write a haiku about a small computer."

Settings: thinking disabled by prefilling an empty `<think>\n\n</think>\n\n` assistant block (Qwen3 non-thinking format), greedy, 128 new tokens, 4 threads. Outputs and timings are printed verbatim.

## Release
If T passes, a public Hugging Face repo `zaydmulani/Ternary-Bonsai-1.7B-TQ2_0_G128-GGUF` gets:
- the best-config GGUF;
- the patches;
- a model card crediting PrismML Bonsai and Qwen3 (Apache-2.0).

## Safety
- Work only in `/home/arduino/uno-model` on the board, with no system changes.
- Record `df -h` and `free -m` before and after.
- Delete everything at the end and confirm `df -h` matches the start.
