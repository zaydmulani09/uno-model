#!/bin/sh
# Stage 1 demo: 3 fixed prompts end to end on the board, greedy, 128 new tokens, 4 threads.
# usage: demo.sh <llama-completion binary> <model name>
cd /home/arduino/uno-model
for p in "Explain why the sky is blue in two sentences." \
         "A shop sells pencils at 3 for \$1. How much do 12 pencils cost? Answer step by step." \
         "Write a haiku about a small computer."; do
  echo "=================== PROMPT: $p"
  bin/$1 -m models/$2.gguf -t 4 -n 128 --temp 0 --seed 0 -no-cnv --no-warmup \
    -p "<|im_start|>user
$p<|im_end|>
<|im_start|>assistant
" 2>&1
done
