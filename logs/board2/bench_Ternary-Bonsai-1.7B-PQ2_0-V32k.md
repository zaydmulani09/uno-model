| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen3 1.7B PQ2_0 - 2.13 bpw (group 128) | 453.86 MiB |     1.79 B | CPU        |       1 |            pp64 |          1.34 ± 0.00 |
| qwen3 1.7B PQ2_0 - 2.13 bpw (group 128) | 453.86 MiB |     1.79 B | CPU        |       1 |            tg64 |          1.20 ± 0.00 |
| qwen3 1.7B PQ2_0 - 2.13 bpw (group 128) | 453.86 MiB |     1.79 B | CPU        |       2 |            pp64 |          2.68 ± 0.00 |
| qwen3 1.7B PQ2_0 - 2.13 bpw (group 128) | 453.86 MiB |     1.79 B | CPU        |       2 |            tg64 |          2.32 ± 0.02 |
| qwen3 1.7B PQ2_0 - 2.13 bpw (group 128) | 453.86 MiB |     1.79 B | CPU        |       3 |            pp64 |          3.97 ± 0.00 |
| qwen3 1.7B PQ2_0 - 2.13 bpw (group 128) | 453.86 MiB |     1.79 B | CPU        |       3 |            tg64 |          3.37 ± 0.01 |
| qwen3 1.7B PQ2_0 - 2.13 bpw (group 128) | 453.86 MiB |     1.79 B | CPU        |       4 |            pp64 |          5.30 ± 0.01 |
| qwen3 1.7B PQ2_0 - 2.13 bpw (group 128) | 453.86 MiB |     1.79 B | CPU        |       4 |            tg64 |          4.50 ± 0.02 |

build: 9f24523 (1)
