### Eval v3 (accuracy % [95% Wilson])

| model | MMLU (1000) | GSM8K (300) | TriviaQA (500) |
|---|---:|---:|---:|
| Ternary-Bonsai-1.7B-TQ2_0_G128 | 46.8 [43.7, 49.9] | 62.7 [57.1, 67.9] | 20.8 [17.5, 24.6] |
| Ternary-Bonsai-1.7B-PQ2_0 | 46.9 [43.8, 50.0] | 62.7 [57.1, 67.9] | 21.2 [17.8, 25.0] |
| gemma-3-1b-it-qat-Q4_0 | 38.4 [35.4, 41.5] | 12.0 [8.8, 16.2] | 19.8 [16.5, 23.5] |
| Qwen3-0.6B-Q4_0 | 29.5 [26.8, 32.4] | 30.0 [25.1, 35.4] | 8.8 [6.6, 11.6] |
| Qwen3.5-0.8B-Q4_0 | 38.5 [35.5, 41.6] | 26.3 [21.7, 31.6] | 10.4 [8.0, 13.4] |

### Board, same day, tg64 at 4 threads (tok/s)

| model | round 1 | round 2 | mean |
|---|---:|---:|---:|
| Ternary-Bonsai-1.7B-TQ2_0_G128 | 6.65 | 6.67 | 6.66 |
| gemma-3-1b-it-qat-Q4_0 | 6.18 | 6.26 | 6.22 |
| Qwen3.5-0.8B-Q4_0 | 6.12 | 6.07 | 6.10 |

### Rule R

| condition | TQ2_0_G128 | Gemma 3 1B Q4_0 | ok |
|---|---|---|---|
| speed (mean tg64 4T) | 6.66 | 6.22 | yes |
| mmlu higher, intervals disjoint | 46.8 [43.7, 49.9] | 38.4 [35.4, 41.5] | yes |
| gsm8k higher, intervals disjoint | 62.7 [57.1, 67.9] | 12.0 [8.8, 16.2] | yes |

**R: PASS**
