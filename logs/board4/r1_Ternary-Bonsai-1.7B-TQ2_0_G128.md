| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen3 1.7B TQ2_0_G128 - 2.13 bpw ternary | 436.16 MiB |     1.72 B | CPU        |       4 |            pp64 |          9.22 ± 0.02 |
| qwen3 1.7B TQ2_0_G128 - 2.13 bpw ternary | 436.16 MiB |     1.72 B | CPU        |       4 |            tg64 |          6.65 ± 0.04 |

build: e0bcacc (1)
