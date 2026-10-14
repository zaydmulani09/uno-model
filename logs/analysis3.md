### Board throughput (tok/s, llama-bench -p 64 -n 64 -r 3)

| model | pp64 1T | pp64 4T | tg64 1T | tg64 2T | tg64 3T | tg64 4T | vs PQ2_0 3.94 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ternary-Bonsai-1.7B-TQ2_0_G128 | 2.35 | 9.20 | 1.82 | 3.38 | 4.93 | 6.68 | 1.70x |
| Ternary-Bonsai-1.7B-TQ2_0_G128-V2 | 2.36 | 9.27 | 2.08 | 3.88 | 5.59 | 7.58 | 1.93x |

### Quality, eval v3 (accuracy % [95% Wilson])

| model | MMLU (1000) | GSM8K (300) | TriviaQA (500) |
|---|---:|---:|---:|
| Ternary-Bonsai-1.7B-PQ2_0 | 46.9 [43.8, 50.0] | 62.7 [57.1, 67.9] | 21.2 [17.8, 25.0] |
| Ternary-Bonsai-1.7B-TQ2_0_G128 | 46.8 [43.7, 49.9] | 62.7 [57.1, 67.9] | 20.8 [17.5, 24.6] |
| Ternary-Bonsai-1.7B-TQ2_0_G128-V2 | 46.5 [43.4, 49.6] | 62.3 [56.7, 67.6] | 16.6 [13.6, 20.1] |

### G4 quality: Ternary-Bonsai-1.7B-TQ2_0_G128 vs Ternary-Bonsai-1.7B-PQ2_0 (difference, paired bootstrap 95% CI, flips)

| task | base | new | delta | 95% CI | right->wrong | wrong->right | ok (delta >= -2) |
|---|---:|---:|---:|---|---:|---:|---|
| mmlu | 46.9 | 46.8 | -0.1 | [-0.7, +0.5] | 5 | 4 | yes |
| gsm8k | 62.7 | 62.7 | +0.0 | [-2.3, +2.3] | 6 | 6 | yes |
| triviaqa | 21.2 | 20.8 | -0.4 | [-1.2, +0.4] | 3 | 1 | yes |

G4 speed: 6.68 tok/s >= 6.24: yes

**G4: PASS**

### G5 quality: Ternary-Bonsai-1.7B-TQ2_0_G128-V2 vs Ternary-Bonsai-1.7B-TQ2_0_G128 (difference, paired bootstrap 95% CI, flips)

| task | base | new | delta | 95% CI | right->wrong | wrong->right | ok (delta >= -3) |
|---|---:|---:|---:|---|---:|---:|---|
| mmlu | 46.8 | 46.5 | -0.3 | [-0.8, +0.2] | 5 | 2 | yes |
| gsm8k | 62.7 | 62.3 | -0.3 | [-2.3, +2.0] | 6 | 5 | yes |
| triviaqa | 20.8 | 16.6 | -4.2 | [-6.4, -2.0] | 28 | 7 | no |

G5 speed: 6.68 -> 7.58 tok/s = 1.135x (>= 1.10: yes)

**G5: FAIL**

### Target T

Eligible: Ternary-Bonsai-1.7B-PQ2_0 (3.94), Ternary-Bonsai-1.7B-TQ2_0_G128 (6.68)
Best config: **Ternary-Bonsai-1.7B-TQ2_0_G128**, tg64 4T 6.68 tok/s, MMLU 46.8

**T: FAIL**
