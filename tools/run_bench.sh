#!/bin/sh
# Stage-0 sweep: each model at 1,2,3,4 threads; pp64 + tg64, 3 reps.
cd /home/arduino/uno-model
for m in gemma-3-1b-it-qat-Q4_0 Qwen3-0.6B-Q4_0 Qwen3-0.6B-TQ2_0 Ternary-Bonsai-1.7B-Q2_0_g64; do
  echo "== $m $(date -u +%T) temps: $(cat /sys/class/thermal/thermal_zone0/temp)"
  bin/llama-bench -m models/$m.gguf -t 1,2,3,4 -p 64 -n 64 -r 3 -o md -oe jsonl > logs/bench_$m.md 2> logs/bench_$m.jsonl
  cat logs/bench_$m.md
done
echo "== done $(date -u +%T) temps: $(cat /sys/class/thermal/thermal_zone0/temp)"
