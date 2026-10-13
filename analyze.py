"""Build the results tables from raw logs: logs/board/bench_*.jsonl + eval/results/*.jsonl.
Prints markdown; REPORT.md pastes it. Implied numbers:
  GB/s    = tok/s * bytes_per_token (all weight tensors streamed once per token; the input
            embedding is a one-row lookup unless it is tied to the output head) - logs/weight_bytes.json
  GFLOP/s = 2 * matmul_params * tok/s (one multiply-add per matmul weight per token)"""
import json
from pathlib import Path

ROOT = Path(__file__).parent
W = json.load(open(ROOT / "logs" / "weight_bytes.json"))
MODELS = ["gemma-3-1b-it-qat-Q4_0", "Qwen3-0.6B-Q4_0", "Qwen3-0.6B-TQ2_0", "Ternary-Bonsai-1.7B-Q2_0_g64"]


def bench(m):
    rows = [json.loads(l) for l in (ROOT / "logs" / "board" / f"bench_{m}.jsonl").open() if l.startswith("{")]
    out = {}
    for r in rows:
        kind = "pp" if r["n_prompt"] else "tg"
        out[(kind, r["n_threads"])] = r
    return out


def evals(m):
    p = ROOT / "eval" / "results" / f"{m}.jsonl"
    if not p.exists():
        return {}
    rs = [json.loads(l) for l in p.open(encoding="utf-8")]
    return {t: (sum(r["correct"] for r in rs if r["task"] == t), sum(r["task"] == t for r in rs))
            for t in ("mmlu", "gsm8k", "triviaqa")}


B = {m: bench(m) for m in MODELS}
print("### Board throughput (Arduino UNO Q, 4x Cortex-A53 @ 2.0 GHz, llama.cpp b11594)\n")
print("| model | size MiB | matmul params M | threads | pp64 tok/s | tg64 tok/s | tg implied GB/s | pp implied GFLOP/s | tg implied GFLOP/s |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for m in MODELS:
    for t in (1, 2, 3, 4):
        pp, tg = B[m][("pp", t)], B[m][("tg", t)]
        size, params = tg["model_size"], W[m]["matmul_params"]
        rd = W[m]["bytes_per_token"]
        print(f"| {m} | {size / 2**20:.0f} | {params / 1e6:.0f} | {t} | "
              f"{pp['avg_ts']:.2f} ± {pp['stddev_ts']:.2f} | {tg['avg_ts']:.2f} ± {tg['stddev_ts']:.2f} | "
              f"{tg['avg_ts'] * rd / 1e9:.2f} | {2 * params * pp['avg_ts'] / 1e9:.1f} | "
              f"{2 * params * tg['avg_ts'] / 1e9:.1f} |")

print("\n### Fastest thread count\n")
print("| model | best tg threads | tg tok/s | best pp threads | pp tok/s |")
print("|---|---:|---:|---:|---:|")
best = {}
for m in MODELS:
    bt = max((1, 2, 3, 4), key=lambda t: B[m][("tg", t)]["avg_ts"])
    bp = max((1, 2, 3, 4), key=lambda t: B[m][("pp", t)]["avg_ts"])
    best[m] = B[m][("tg", bt)]["avg_ts"]
    print(f"| {m} | {bt} | {best[m]:.2f} | {bp} | {B[m][('pp', bp)]['avg_ts']:.2f} |")

print("\n### Quality (PC, identical prompts, greedy)\n")
print("| model | MMLU (200) | GSM8K (100) | TriviaQA (200) |")
print("|---|---:|---:|---:|")
for m in MODELS:
    e = evals(m)
    cell = lambda t: f"{e[t][0]}/{e[t][1]} ({100 * e[t][0] / e[t][1]:.1f}%)" if t in e else "n/a"
    print(f"| {m} | {cell('mmlu')} | {cell('gsm8k')} | {cell('triviaqa')} |")

q4, tq = best["Qwen3-0.6B-Q4_0"], best["Qwen3-0.6B-TQ2_0"]
speedup = tq / q4 - 1
print("\n### Rule G1\n")
print(f"Qwen3-0.6B TQ2_0 best tg = {tq:.2f} tok/s, Q4_0 best tg = {q4:.2f} tok/s, "
      f"speedup = {100 * speedup:+.1f}% (threshold +20.0%) -> **{'PASS' if speedup >= 0.20 else 'FAIL'}**")
print("\nPer thread count:", ", ".join(
    f"{t}T {100 * (B['Qwen3-0.6B-TQ2_0'][('tg', t)]['avg_ts'] / B['Qwen3-0.6B-Q4_0'][('tg', t)]['avg_ts'] - 1):+.1f}%"
    for t in (1, 2, 3, 4)))
