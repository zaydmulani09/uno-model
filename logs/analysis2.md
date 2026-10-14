### Board throughput (tok/s; same protocol as stage 0)

| model | pp64 1T | pp64 4T | tg64 1T | tg64 2T | tg64 3T | tg64 4T | tg64 4T vs baseline 3.53 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ternary-Bonsai-1.7B-Q2_0_g64 | 1.21 | 4.78 | 0.94 | 1.83 | 2.68 | 3.56 | 1.01x |
| Ternary-Bonsai-1.7B-TQ2_0 | 2.58 | 10.16 | 1.98 | 3.68 | 5.36 | 7.23 | 2.05x |
| Ternary-Bonsai-1.7B-PQ2_0.unpatched | 1.34 | 5.29 | 1.04 | 2.01 | 2.93 | 3.94 | 1.12x |
| Ternary-Bonsai-1.7B-PQ2_0 | 1.34 | 5.28 | 1.04 | 2.00 | 2.93 | 3.88 | 1.10x |
| Ternary-Bonsai-1.7B-PQ2_0-V32k | 1.34 | 5.30 | 1.20 | 2.32 | 3.37 | 4.50 | 1.27x |
| Ternary-Bonsai-1.7B-TQ2_0-V32k | 2.60 | 10.11 | 2.27 | 4.25 | 6.07 | 8.19 | 2.32x |
| Ternary-Bonsai-1.7B-Q2_0_g64-V32k | 1.21 | 4.77 | 1.09 | 2.10 | 3.08 | 4.08 | 1.16x |

### G2 (candidate >= 4.41 tok/s at 4T and each eval >= baseline - 3)

| candidate | tg64 4T | speedup | MMLU | GSM8K | TriviaQA | speed ok | quality ok | G2 |
|---|---:|---:|---:|---:|---:|---|---|---|
| Ternary-Bonsai-1.7B-TQ2_0 | 7.23 | 2.05x | 50.5 | 73.0 | 19.5 | yes | no | **FAIL** |
| Ternary-Bonsai-1.7B-PQ2_0 | 3.94 | 1.12x | 53.5 | 67.0 | 22.0 | no | yes | **FAIL** |
| baseline Q2_0_g64 | 3.53 | 1.00x | 54.0 | 66.0 | 22.0 | | | |

**G2: FAIL**

### G3 (vocab restriction on the best candidate: Ternary-Bonsai-1.7B-PQ2_0)

tg64 4T: 3.88 -> 4.50 tok/s = **1.159x** (need >= 1.10x)

| task | unrestricted | restricted | delta | right->wrong | wrong->right |
|---|---:|---:|---:|---:|---:|
| mmlu | 53.0 | 52.5 | -0.5 | 2 | 1 |
| gsm8k | 68.0 | 23.0 | -45.0 | 48 | 3 |
| triviaqa | 22.0 | 0.0 | -22.0 | 44 | 0 |

**G3: FAIL** (speed ok, quality no)
