| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen3 1.7B TQ2_0 - 2.06 bpw ternary | 440.55 MiB |     1.79 B | CPU        |       1 |            pp64 |          2.60 ± 0.00 |
| qwen3 1.7B TQ2_0 - 2.06 bpw ternary | 440.55 MiB |     1.79 B | CPU        |       1 |            tg64 |          2.27 ± 0.00 |
| qwen3 1.7B TQ2_0 - 2.06 bpw ternary | 440.55 MiB |     1.79 B | CPU        |       2 |            pp64 |          5.15 ± 0.00 |
| qwen3 1.7B TQ2_0 - 2.06 bpw ternary | 440.55 MiB |     1.79 B | CPU        |       2 |            tg64 |          4.25 ± 0.02 |
| qwen3 1.7B TQ2_0 - 2.06 bpw ternary | 440.55 MiB |     1.79 B | CPU        |       3 |            pp64 |          7.60 ± 0.02 |
| qwen3 1.7B TQ2_0 - 2.06 bpw ternary | 440.55 MiB |     1.79 B | CPU        |       3 |            tg64 |          6.07 ± 0.04 |
| qwen3 1.7B TQ2_0 - 2.06 bpw ternary | 440.55 MiB |     1.79 B | CPU        |       4 |            pp64 |         10.11 ± 0.02 |
| qwen3 1.7B TQ2_0 - 2.06 bpw ternary | 440.55 MiB |     1.79 B | CPU        |       4 |            tg64 |          8.19 ± 0.02 |

build: e0bcacc (1)
