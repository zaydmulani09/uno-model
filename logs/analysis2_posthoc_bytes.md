### G3 (vocab restriction on the best candidate: Ternary-Bonsai-1.7B-PQ2_0)

tg64 4T: 3.88 -> 4.50 tok/s = **1.159x** (need >= 1.10x)

| task | unrestricted | restricted | delta | right->wrong | wrong->right |
|---|---:|---:|---:|---:|---:|
| mmlu | 53.0 | 53.5 | +0.5 | 0 | 1 |
| gsm8k | 68.0 | 24.0 | -44.0 | 47 | 3 |
| triviaqa | 22.0 | 19.5 | -2.5 | 5 | 0 |

**G3: FAIL** (speed ok, quality no)
