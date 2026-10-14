| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen3 1.7B TQ2_0 - 2.06 bpw ternary | 423.34 MiB |     1.72 B | CPU        |       1 |            pp64 |          2.58 ± 0.01 |
| qwen3 1.7B TQ2_0 - 2.06 bpw ternary | 423.34 MiB |     1.72 B | CPU        |       1 |            tg64 |          1.98 ± 0.00 |
| qwen3 1.7B TQ2_0 - 2.06 bpw ternary | 423.34 MiB |     1.72 B | CPU        |       2 |            pp64 |          5.16 ± 0.00 |
| qwen3 1.7B TQ2_0 - 2.06 bpw ternary | 423.34 MiB |     1.72 B | CPU        |       2 |            tg64 |          3.68 ± 0.02 |
| qwen3 1.7B TQ2_0 - 2.06 bpw ternary | 423.34 MiB |     1.72 B | CPU        |       3 |            pp64 |          7.60 ± 0.02 |
| qwen3 1.7B TQ2_0 - 2.06 bpw ternary | 423.34 MiB |     1.72 B | CPU        |       3 |            tg64 |          5.36 ± 0.03 |
| qwen3 1.7B TQ2_0 - 2.06 bpw ternary | 423.34 MiB |     1.72 B | CPU        |       4 |            pp64 |         10.16 ± 0.03 |
| qwen3 1.7B TQ2_0 - 2.06 bpw ternary | 423.34 MiB |     1.72 B | CPU        |       4 |            tg64 |          7.23 ± 0.03 |

build: e0bcacc (1)
