| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen3 1.7B TQ2_0_G128 - 2.13 bpw ternary | 436.16 MiB |     1.72 B | CPU        |       4 |            pp64 |          9.22 ± 0.01 |
| qwen3 1.7B TQ2_0_G128 - 2.13 bpw ternary | 436.16 MiB |     1.72 B | CPU        |       4 |            tg64 |          6.67 ± 0.02 |

build: e0bcacc (1)
