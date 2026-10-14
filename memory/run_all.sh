#!/usr/bin/env bash
# Stage 2a, one launcher (PREREG5). Run on a Linux box with one 80 GB GPU (H100 or A100), from a clone of
# github.com/zaydmulani09/uno-model:   bash memory/run_all.sh
# Needs no credentials: models and the tokenized data are public on Hugging Face.
# Resumable: every step skips itself if its output already exists.
set -euo pipefail
cd "$(dirname "$0")/.."
R=$PWD; M=$R/memory; RES=$R/eval/results_mem; RUNS=$R/runs; DATA=${DATA:-$R/data}
mkdir -p "$RES" "$RUNS" "$DATA" "$R/logs/gpu"
exec > >(tee -a "$R/logs/gpu/run_all.log") 2>&1

nvidia-smi
pip install -q "transformers==5.19.0" numpy pandas tokenizers huggingface_hub safetensors accelerate
python - <<'EOF'
from huggingface_hub import snapshot_download
import os
snapshot_download("zaydmulani/uno-model-stage2a-tokens", repo_type="dataset", local_dir=os.environ.get("DATA", "data"))
snapshot_download("prism-ml/Ternary-Bonsai-1.7B-unpacked")
snapshot_download("Qwen/Qwen3-8B", allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])
EOF
sha256sum "$DATA"/train.bin "$DATA"/val.bin | tee "$R/logs/gpu/data_sha256.txt"

ev() { local name=$1; shift; [ -f "$RES/$name.jsonl" ] || python "$M/eval_hf.py" --name "$name" "$@"; }
tr() { local d=$1 stop=$2 tag=$3; [ -f "$RUNS/$d/mem_${tag}.pt" ] || \
       python "$M/train.py" --design "$d" --data "$DATA" --out "$RUNS/$d" --stop-tokens "$stop"; }

ev base --variant base

run_design() {
  local d=$1 final
  ev "${d}_off_init" --variant off --design "$d"
  python "$M/analyze5.py" sanity "$d" || { echo "SANITY FAIL ($d): stop, nothing else is scored"; exit 3; }
  tr "$d" 100e6 100M; ev "${d}_on_100M" --variant on --design "$d" --ckpt "$RUNS/$d/mem_100M.pt"
  tr "$d" 300e6 300M; ev "${d}_on_300M" --variant on --design "$d" --ckpt "$RUNS/$d/mem_300M.pt"
  if python "$M/analyze5.py" continue "$d"; then
    tr "$d" 1e9 1000M; ev "${d}_on_1000M" --variant on --design "$d" --ckpt "$RUNS/$d/mem_1000M.pt"; final=1000M
  else
    final=300M
  fi
  ev "${d}_zero_${final}" --variant zero --design "$d" --ckpt "$RUNS/$d/mem_${final}.pt"
  ev "${d}_shuffle_${final}" --variant shuffle --design "$d" --ckpt "$RUNS/$d/mem_${final}.pt"
  echo "$final" > "$RUNS/$d/FINAL"
  python "$M/export.py" "$d" "$RUNS/$d/mem_${final}.pt" "$RUNS/$d/memory_${d}_${final}.safetensors" \
    | tee "$RUNS/$d/export.json"
  if python "$M/analyze5.py" g6 "$d" "$final"; then echo PASS > "$RUNS/$d/G6"; else echo FAIL > "$RUNS/$d/G6"; fi
}

# a crash anywhere aborts the launcher (set -e); only a measured G6 FAIL moves on to design B
run_design A
echo "design A: G6 $(cat "$RUNS/A/G6")"
if [ "$(cat "$RUNS/A/G6")" = FAIL ]; then
  run_design B
  echo "design B: G6 $(cat "$RUNS/B/G6")"
fi
python "$M/analyze5.py" report | tee "$R/logs/gpu/report_tables.md"
tar czf "$R/stage2a_results.tgz" eval/results_mem logs/gpu runs/*/train_log.jsonl runs/*/FINAL runs/*/G6 runs/*/export.json \
  runs/*/*.safetensors 2>/dev/null || true
echo "DONE: $R/stage2a_results.tgz"
