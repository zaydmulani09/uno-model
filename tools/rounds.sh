#!/bin/sh
# Stage 1c same-day control: two alternating rounds at 4 threads (PREREG4 order)
cd /home/arduino/uno-model
for r in 1 2; do
  for m in Ternary-Bonsai-1.7B-TQ2_0_G128 gemma-3-1b-it-qat-Q4_0 Qwen3.5-0.8B-Q4_0; do
    echo "== round $r $m $(date -u +%T) temp $(cat /sys/class/thermal/thermal_zone0/temp)" >> logs/rounds.log
    bin/llama-bench -m models/$m.gguf -t 4 -p 64 -n 64 -r 3 -o md -oe jsonl > logs/r${r}_$m.md 2> logs/r${r}_$m.jsonl
    cat logs/r${r}_$m.md >> logs/rounds.log
  done
done
echo "== done $(date -u +%T)" >> logs/rounds.log
