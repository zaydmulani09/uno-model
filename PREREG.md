# Stage 0 pre-registration

Committed before any board throughput result for Qwen3-0.6B Q4_0 existed. Thresholds below do not change after results.

## Question
On an Arduino UNO Q (4x Cortex-A53, no `asimddp`), does a ternary quant beat 4-bit for small LLMs?

## Models
| name | source |
|---|---|
| Gemma 3 1B Q4_0 | `bartowski/google_gemma-3-1b-it-qat-GGUF`, file `google_gemma-3-1b-it-qat-Q4_0.gguf` (Google QAT weights) |
| Qwen3-0.6B Q4_0 | quantized here from `Qwen/Qwen3-0.6B` HF weights (convert → F16 → `llama-quantize Q4_0`) |
| Qwen3-0.6B TQ2_0 | quantized here from the same F16 (`llama-quantize TQ2_0`), post-training ternarization, no QAT |
| Ternary-Bonsai-1.7B Q2_0_g64 | `prism-ml/Ternary-Bonsai-1.7B-gguf`, file `Ternary-Bonsai-1.7B-Q2_0_g64.gguf` |

## Throughput protocol (board)
`llama-bench -t 1,2,3,4 -p 64 -n 64 -r 3`, llama.cpp tag b11594, one model at a time, nothing else running.

## Rule G1
TQ2_0 passes only if its **token-generation speed (tg64 tok/s)** is **at least 20% higher** than Q4_0 of the same model (Qwen3-0.6B) on the board.
Each quant is taken at its own fastest thread count. Per-thread-count ratios are reported too, but they don't decide G1.

## Quality protocol (PC)
Frozen set `eval/evalset.jsonl`, seed 0: 200 MMLU (0-shot, answer letter constrained by grammar), 100 GSM8K (4-shot CoT, last number / `####`), 200 TriviaQA rc.nocontext validation (5-shot, normalized exact match against aliases).
Raw completion with no chat template, greedy decoding, the same prompts for all four models. Quality is reported only; it does not affect G1.
