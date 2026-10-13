# uno-model, stage 0: ternary vs 4-bit on an Arduino UNO Q

**G1: PASS.** On the board, Qwen3-0.6B TQ2_0 generates **14.02 tok/s** against **9.76 tok/s** for Q4_0, **+43.7%**. The threshold was +20%. The gap holds at every thread count (+42% to +44%).

**The fastest thread count is 4** for every model, for both generation and prompt processing.

**Caveat on that pass:** G1 measures speed only, as pre-registered in [PREREG.md](PREREG.md). Ternarizing Qwen3-0.6B after training destroys it: chance on MMLU and 0% on GSM8K and TriviaQA. The ternary model that actually works is Ternary-Bonsai-1.7B, which was trained ternary. It scores best on quality but is the slowest on this CPU.

## Board

| | |
|---|---|
| board | Arduino UNO Q, Qualcomm QRB2210, 4x Cortex-A53 ("Kryo-V2") @ 2.016 GHz, Debian 13 |
| RAM | 3669 MiB usable (MemTotal 3758016 kB) + 1834 MiB swap |
| CPU flags | `fp asimd evtstrm aes pmull sha1 sha2 crc32 cpuid` |
| **asimddp** | **not present**: no SDOT/UDOT, so every int8 dot product runs as plain NEON multiply-adds |
| governor | schedutil; all 4 cores held 2016000 kHz during the sweep (sampled), max zone temp 60.7 °C, no throttling |

Memory read bandwidth (`tools/membw.c`: NEON sum over a 512 MiB buffer, best of 7, three runs, [logs/board/board_membw.log](logs/board/board_membw.log)):

| threads | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|
| read GB/s (median of 3 runs) | 2.39 | 5.23 | 7.20 | 8.94 |

## Throughput on the board (llama-bench, pp64 / tg64, 3 reps)

| model | size MiB | matmul params M | threads | pp64 tok/s | tg64 tok/s | tg implied GB/s | pp implied GFLOP/s | tg implied GFLOP/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gemma-3-1b-it-qat-Q4_0 | 682 | 1000 | 1 | 2.73 ± 0.00 | 1.74 ± 0.00 | 1.24 | 5.5 | 3.5 |
| gemma-3-1b-it-qat-Q4_0 | 682 | 1000 | 2 | 5.38 ± 0.00 | 3.30 ± 0.02 | 2.36 | 10.8 | 6.6 |
| gemma-3-1b-it-qat-Q4_0 | 682 | 1000 | 3 | 7.82 ± 0.01 | 4.70 ± 0.02 | 3.36 | 15.6 | 9.4 |
| gemma-3-1b-it-qat-Q4_0 | 682 | 1000 | 4 | 10.28 ± 0.04 | 6.24 ± 0.02 | 4.47 | 20.6 | 12.5 |
| Qwen3-0.6B-Q4_0 | 442 | 596 | 1 | 4.23 ± 0.00 | 2.63 ± 0.01 | 0.99 | 5.0 | 3.1 |
| Qwen3-0.6B-Q4_0 | 442 | 596 | 2 | 8.40 ± 0.00 | 4.90 ± 0.00 | 1.84 | 10.0 | 5.8 |
| Qwen3-0.6B-Q4_0 | 442 | 596 | 3 | 12.18 ± 0.00 | 7.00 ± 0.12 | 2.63 | 14.5 | 8.3 |
| Qwen3-0.6B-Q4_0 | 442 | 596 | 4 | 16.44 ± 0.16 | 9.76 ± 0.05 | 3.67 | 19.6 | 11.6 |
| Qwen3-0.6B-TQ2_0 | 314 | 596 | 1 | 7.56 ± 0.01 | 3.79 ± 0.02 | 0.92 | 9.0 | 4.5 |
| Qwen3-0.6B-TQ2_0 | 314 | 596 | 2 | 15.17 ± 0.07 | 6.99 ± 0.00 | 1.69 | 18.1 | 8.3 |
| Qwen3-0.6B-TQ2_0 | 314 | 596 | 3 | 21.41 ± 0.00 | 9.94 ± 0.09 | 2.40 | 25.5 | 11.9 |
| Qwen3-0.6B-TQ2_0 | 314 | 596 | 4 | 29.52 ± 0.01 | **14.02 ± 0.02** | 3.39 | 35.2 | 16.7 |
| Ternary-Bonsai-1.7B-Q2_0_g64 | 462 | 1720 | 1 | 1.21 ± 0.00 | 0.94 ± 0.00 | 0.46 | 4.2 | 3.2 |
| Ternary-Bonsai-1.7B-Q2_0_g64 | 462 | 1720 | 2 | 2.42 ± 0.00 | 1.83 ± 0.01 | 0.89 | 8.3 | 6.3 |
| Ternary-Bonsai-1.7B-Q2_0_g64 | 462 | 1720 | 3 | 3.58 ± 0.00 | 2.65 ± 0.00 | 1.29 | 12.3 | 9.1 |
| Ternary-Bonsai-1.7B-Q2_0_g64 | 462 | 1720 | 4 | 4.77 ± 0.01 | 3.53 ± 0.01 | 1.71 | 16.4 | 12.1 |

How the implied numbers are computed ([analyze.py](analyze.py), [logs/weight_bytes.json](logs/weight_bytes.json)):
- **GB/s** = tok/s × weight bytes streamed per token. That counts every tensor except the input embedding, which is a one-row lookup when the model has a separate output head. Qwen3 GGUFs have one; Gemma 3 and Bonsai tie it, so their whole embedding is read through the head every token.
- **GFLOP/s** = 2 × matmul weights × tok/s. Attention-over-KV FLOPs are ignored; at 64 tokens of context they are negligible.

### Fastest thread count

| model | best tg threads | tg tok/s | best pp threads | pp tok/s |
|---|---:|---:|---:|---:|
| gemma-3-1b-it-qat-Q4_0 | 4 | 6.24 | 4 | 10.28 |
| Qwen3-0.6B-Q4_0 | 4 | 9.76 | 4 | 16.44 |
| Qwen3-0.6B-TQ2_0 | 4 | 14.02 | 4 | 29.52 |
| Ternary-Bonsai-1.7B-Q2_0_g64 | 4 | 3.53 | 4 | 4.77 |

## Quality (PC, same GGUFs, identical prompts)

| model | MMLU (200) | GSM8K (100) | TriviaQA (200) |
|---|---:|---:|---:|
| gemma-3-1b-it-qat-Q4_0 | 79/200 (39.5%) | 14/100 (14.0%) | 41/200 (20.5%) |
| Qwen3-0.6B-Q4_0 | 65/200 (32.5%) | 30/100 (30.0%) | 21/200 (10.5%) |
| Qwen3-0.6B-TQ2_0 | 41/200 (20.5%) | 0/100 (0.0%) | 0/200 (0.0%) |
| Ternary-Bonsai-1.7B-Q2_0_g64 | **108/200 (54.0%)** | **66/100 (66.0%)** | **44/200 (22.0%)** |

The frozen set is [eval/evalset.jsonl](eval/evalset.jsonl), seed 0. Every prompt and output is in `eval/results/*.jsonl`.
- **MMLU:** 0-shot; the answer letter is constrained by a grammar.
- **GSM8K:** 4-shot CoT; the answer is the number after `####`, or the last number.
- **TriviaQA:** rc.nocontext validation, 5-shot; normalized exact match against the aliases.

All runs use raw completion with no chat template and greedy decoding, on a PC (Windows x64) running the same llama.cpp tag, b11594. With 200 items, one MMLU standard error is about ±3.4 points; with 100 items, one GSM8K standard error is about ±4.6 points.

## Rule G1

> TQ2_0 passes only if at least 20% faster than Q4_0 of the same model on the board.

Defined in PREREG.md before the Q4_0 board numbers existed: tg64 tok/s, each quant at its own fastest thread count.

**Qwen3-0.6B: TQ2_0 14.02 tok/s vs Q4_0 9.76 tok/s = +43.7% → PASS.**
Per thread count: 1T +44.2%, 2T +42.7%, 3T +42.0%, 4T +43.7%. Prompt processing gains even more: +79.6% at 4T.

## What the numbers say

1. **Generation here is compute-bound, not memory-bound.** At 4 threads the best model streams only 3.4–4.5 GB/s of weights, about half the 8.9 GB/s the cores can read. Throughput also scales almost linearly with cores (×3.6–3.8 from 1T to 4T). Without `asimddp`, each int8 dot product runs as plain NEON multiply-adds, and that arithmetic is the ceiling.
2. **That is why TQ2_0 wins on speed.** The win is not that it is smaller. Its kernel does less arithmetic per weight: 16.7 GFLOP/s-equivalent against 11.6 for Q4_0, at almost the same bandwidth. The headroom is capped by the parts that stay higher-precision. `llama-quantize TQ2_0` keeps the 151936×1024 output head at Q6_K, and that is 128 MB, more than half of the 241 MB streamed per token.
3. **Ternarizing after training is not usable.** Qwen3-0.6B at TQ2_0 is at chance on MMLU (20.5%) and outputs garbage on generation tasks (0%). G1 passing means the kernel is fast. It does not mean this model is usable.
4. **Ternary-Bonsai-1.7B, trained ternary, is the best model on quality.** It scores 54% MMLU and 66% GSM8K against 32.5% and 30% for Qwen3-0.6B Q4_0. It is also the slowest, at 3.53 tok/s. Its Q2_0_g64 kernel reaches only about 12 GFLOP/s-equivalent on A53, the same as Q4_0. It shows none of TQ2_0's per-weight advantage, so its 2.9× larger parameter count costs about 2.8× the time.
   - **Open question for stage 1:** would Bonsai repacked to TQ2_0 run at about 16–17 GFLOP/s-equivalent (≈5 tok/s)?

## Deviations from the brief (all forced, none chosen after seeing results)

1. **Not on the network.** The board had no Wi-Fi link (only `lo` and `docker0`). Everything went over **USB ADB** (`adb` from the local Android SDK). No address or login was needed.
2. **llama.cpp was cross-compiled, not built on the board.** The board has no gcc or cmake, no internet route, and `sudo` asks for a password, so installing build tools on it was impossible. Instead, mainline llama.cpp **tag b11594 (ggml e0bcacc)** was cross-compiled on the PC:
   - toolchain: zig 0.16 (clang) with `-target aarch64-linux-gnu.2.41 -mcpu=cortex_a53`, matching the board's glibc 2.41;
   - build: `GGML_NATIVE=OFF`, `GGML_OPENMP=OFF` (ggml's own threadpool; zig ships no libgomp), static ggml/llama.
   - Full configure and build logs are in `logs/pc/cross_*.log`.

   Docker was the first choice for this; Docker Desktop failed to start on the PC, so it wasn't used.
3. **Gemma 3 1B Q4_0** is Google's QAT Q4_0 via the `bartowski/google_gemma-3-1b-it-qat-GGUF` mirror, because Google's own repo is gated.
4. **MMLU 0-shot letter bias.** Qwen3-0.6B Q4_0 answers "A" for 169/200 questions. Prompts are kept exactly as pre-registered; the bias is reported, not tuned away.
5. **USB incident.** Midway through setup, the board was unplugged by accident, and the laptop's USB port then stopped working. After that was fixed the board rebooted, and all benchmarks ran after the reboot on a clean boot.
   - The first `membw` run in `board_baseline_before.log` has `checksum_ok=0` because of a slicing bug (fixed in `tools/membw.c`). The bandwidth numbers above come from the fixed binary, in `board_membw.log`.

## Board footprint (safety)

All work was confined to `/home/arduino/uno-model`. No packages were installed and no system settings, Docker, partitions or bootloader were touched. Before and after logs: `logs/board/board_baseline_before.log` and `logs/board/board_after_cleanup.log`.

| | start | after cleanup |
|---|---|---|
| `/home/arduino` (mmcblk0p69) | 598M used, 17G free | **598M used, 17G free** (exact match) |
| `/` (mmcblk0p68) | 6.2G used | 6.4G used |
| RAM available | 3216 MiB | 3176 MiB |

The `/home/arduino/uno-model` directory has been deleted (`rm -rf`).

The 0.2G rise on `/` was already present at uptime 0 after the reboot, before anything of mine ran. Nothing in this project writes outside `/home/arduino/uno-model`. The likely source is system journal or App Lab state from the reboot.

## Reproduce

```
eval/build_evalset.py                 # freeze the 500 questions (already committed)
tools/weight_bytes.py *.gguf          # per-token bytes / matmul params
tools/run_bench.sh                    # on the board
eval/run_eval.py llama-server.exe <gguf> <name>   # on the PC
analyze.py                            # tables above
```

Model hashes (sha256, verified identical on PC and board):

```
6364f1139062ff56ee03faf69c8feba6633f740305f8f8b0fe922b25de9438e3  gemma-3-1b-it-qat-Q4_0.gguf
34a119e136d15e4121b94b95e4f44e129e21737ebef97f4bfe4f2b1da29c5d5e  Qwen3-0.6B-Q4_0.gguf
1ad7ad5e3ebde59542c856f5213edb1c376f61b2b1abc8560730b60813c7b76e  Qwen3-0.6B-TQ2_0.gguf
6d0ecb3d9055969b5cde332b6fdb60e67ed3599e9f73e56b977731e1467e5c91  Ternary-Bonsai-1.7B-Q2_0_g64.gguf
```
