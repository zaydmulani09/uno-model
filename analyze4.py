"""Stage 1c tables and rule R (PREREG4.md): eval v3 baselines with 95% Wilson intervals, and the
same-day 4-thread speed rounds from logs/board4/r{1,2}_<model>.jsonl."""
import json, math
from pathlib import Path

ROOT = Path(__file__).parent
TASKS = ("mmlu", "gsm8k", "triviaqa")
G, GEM = "Ternary-Bonsai-1.7B-TQ2_0_G128", "gemma-3-1b-it-qat-Q4_0"
EVAL = [G, "Ternary-Bonsai-1.7B-PQ2_0", GEM, "Qwen3-0.6B-Q4_0", "Qwen3.5-0.8B-Q4_0"]
BOARD = [G, GEM, "Qwen3.5-0.8B-Q4_0"]


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * p, 100 * (c - h), 100 * (c + h)


def acc(m):
    rs = [json.loads(l) for l in (ROOT / "eval" / "results" / f"{m}_v3.jsonl").open(encoding="utf-8")]
    return {t: wilson(sum(r["correct"] for r in rs if r["task"] == t), sum(r["task"] == t for r in rs)) for t in TASKS}


def tg4(m, r):
    rows = [json.loads(l) for l in (ROOT / "logs" / "board4" / f"r{r}_{m}.jsonl").open() if l.startswith("{")]
    return next(x["avg_ts"] for x in rows if x["n_gen"] and x["n_threads"] == 4)


A = {m: acc(m) for m in EVAL}
print("### Eval v3 (accuracy % [95% Wilson])\n")
print("| model | MMLU (1000) | GSM8K (300) | TriviaQA (500) |")
print("|---|---:|---:|---:|")
for m in EVAL:
    print(f"| {m} | " + " | ".join(f"{A[m][t][0]:.1f} [{A[m][t][1]:.1f}, {A[m][t][2]:.1f}]" for t in TASKS) + " |")

S = {m: (tg4(m, 1), tg4(m, 2)) for m in BOARD}
print("\n### Board, same day, tg64 at 4 threads (tok/s)\n")
print("| model | round 1 | round 2 | mean |")
print("|---|---:|---:|---:|")
for m in BOARD:
    print(f"| {m} | {S[m][0]:.2f} | {S[m][1]:.2f} | {sum(S[m]) / 2:.2f} |")

speed = sum(S[G]) / 2 >= sum(S[GEM]) / 2
q = {t: A[G][t][0] > A[GEM][t][0] and A[G][t][1] > A[GEM][t][2] for t in ("mmlu", "gsm8k")}
print("\n### Rule R\n")
print(f"| condition | TQ2_0_G128 | Gemma 3 1B Q4_0 | ok |")
print("|---|---|---|---|")
print(f"| speed (mean tg64 4T) | {sum(S[G]) / 2:.2f} | {sum(S[GEM]) / 2:.2f} | {'yes' if speed else 'no'} |")
for t in ("mmlu", "gsm8k"):
    print(f"| {t} higher, intervals disjoint | {A[G][t][0]:.1f} [{A[G][t][1]:.1f}, {A[G][t][2]:.1f}] | "
          f"{A[GEM][t][0]:.1f} [{A[GEM][t][1]:.1f}, {A[GEM][t][2]:.1f}] | {'yes' if q[t] else 'no'} |")
print(f"\n**R: {'PASS' if speed and all(q.values()) else 'FAIL'}**")
