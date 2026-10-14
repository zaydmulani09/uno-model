#!/bin/sh
# usage: bench.sh <bench-binary> <model-name>; same protocol as stage 0
cd /home/arduino/uno-model
echo "== $2 $(date -u +%T) temp $(cat /sys/class/thermal/thermal_zone0/temp)" >> logs/bench_all.log
bin/$1 -m models/$2.gguf -t 1,2,3,4 -p 64 -n 64 -r 3 -o md -oe jsonl > logs/bench_$2.md 2> logs/bench_$2.jsonl
cat logs/bench_$2.md >> logs/bench_all.log
echo "== end $2 $(date -u +%T) temp $(cat /sys/class/thermal/thermal_zone0/temp)" >> logs/bench_all.log
