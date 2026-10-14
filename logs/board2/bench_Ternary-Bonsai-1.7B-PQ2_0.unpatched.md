| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen3 1.7B PQ2_0 - 2.13 bpw (group 128) | 436.16 MiB |     1.72 B | CPU        |       1 |            pp64 |          1.34 ± 0.00 |
| qwen3 1.7B PQ2_0 - 2.13 bpw (group 128) | 436.16 MiB |     1.72 B | CPU        |       1 |            tg64 |          1.04 ± 0.00 |
| qwen3 1.7B PQ2_0 - 2.13 bpw (group 128) | 436.16 MiB |     1.72 B | CPU        |       2 |            pp64 |          2.67 ± 0.00 |
| qwen3 1.7B PQ2_0 - 2.13 bpw (group 128) | 436.16 MiB |     1.72 B | CPU        |       2 |            tg64 |          2.01 ± 0.00 |
| qwen3 1.7B PQ2_0 - 2.13 bpw (group 128) | 436.16 MiB |     1.72 B | CPU        |       3 |            pp64 |          3.96 ± 0.00 |
| qwen3 1.7B PQ2_0 - 2.13 bpw (group 128) | 436.16 MiB |     1.72 B | CPU        |       3 |            tg64 |          2.93 ± 0.01 |
| qwen3 1.7B PQ2_0 - 2.13 bpw (group 128) | 436.16 MiB |     1.72 B | CPU        |       4 |            pp64 |          5.29 ± 0.00 |
| qwen3 1.7B PQ2_0 - 2.13 bpw (group 128) | 436.16 MiB |     1.72 B | CPU        |       4 |            tg64 |          3.94 ± 0.02 |

build: 9f24523 (1)
