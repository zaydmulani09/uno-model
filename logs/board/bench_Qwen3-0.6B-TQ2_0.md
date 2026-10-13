| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen3 0.6B TQ2_0 - 2.06 bpw ternary | 313.71 MiB |   751.63 M | CPU        |       1 |            pp64 |          7.56 ± 0.01 |
| qwen3 0.6B TQ2_0 - 2.06 bpw ternary | 313.71 MiB |   751.63 M | CPU        |       1 |            tg64 |          3.79 ± 0.02 |
| qwen3 0.6B TQ2_0 - 2.06 bpw ternary | 313.71 MiB |   751.63 M | CPU        |       2 |            pp64 |         15.17 ± 0.07 |
| qwen3 0.6B TQ2_0 - 2.06 bpw ternary | 313.71 MiB |   751.63 M | CPU        |       2 |            tg64 |          6.99 ± 0.00 |
| qwen3 0.6B TQ2_0 - 2.06 bpw ternary | 313.71 MiB |   751.63 M | CPU        |       3 |            pp64 |         21.41 ± 0.00 |
| qwen3 0.6B TQ2_0 - 2.06 bpw ternary | 313.71 MiB |   751.63 M | CPU        |       3 |            tg64 |          9.94 ± 0.09 |
| qwen3 0.6B TQ2_0 - 2.06 bpw ternary | 313.71 MiB |   751.63 M | CPU        |       4 |            pp64 |         29.52 ± 0.01 |
| qwen3 0.6B TQ2_0 - 2.06 bpw ternary | 313.71 MiB |   751.63 M | CPU        |       4 |            tg64 |         14.02 ± 0.02 |

build: e0bcacc (1)
