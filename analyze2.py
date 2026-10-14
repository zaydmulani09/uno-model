"""Stage 1 tables and gates (PREREG2.md) from raw logs: logs/board2/bench_*.jsonl, eval/results/*.jsonl.
Usage: python analyze2.py <best-unrestricted-eval-name> <best-restricted-eval-name> <best-bench-name> <best-restricted-bench-name>"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).parent
BASE = {"tg4": 3.53, "mmlu": 54.0, "gsm8k": 66.0, "triviaqa": 22.0}
TASKS = ("mmlu", "gsm8k", "triviaqa")
CANDS = ["Ternary-Bonsai-1.7B-TQ2_0", "Ternary-Bonsai-1.7B-PQ2_0"]
# G2 benches each candidate as built (PQ2_0: the unmodified fork); ".unpatched" is that run, the plain
# PQ2_0 file is the re-run with the vocab-patched fork binary used for the G3 pair
G2_BENCH = {"Ternary-Bonsai-1.7B-TQ2_0": "Ternary-Bonsai-1.7B-TQ2_0",
            "Ternary-Bonsai-1.7B-PQ2_0": "Ternary-Bonsai-1.7B-PQ2_0.unpatched"}
BENCHES = ["Ternary-Bonsai-1.7B-Q2_0_g64", "Ternary-Bonsai-1.7B-TQ2_0", "Ternary-Bonsai-1.7B-PQ2_0.unpatched",
           "Ternary-Bonsai-1.7B-PQ2_0", "Ternary-Bonsai-1.7B-PQ2_0-V32k",
           "Ternary-Bonsai-1.7B-TQ2_0-V32k", "Ternary-Bonsai-1.7B-Q2_0_g64-V32k"]


def bench(m):
    p = ROOT / "logs" / "board2" / f"bench_{m}.jsonl"
    rows = [json.loads(l) for l in p.open() if l.startswith("{")]
    return {("pp" if r["n_prompt"] else "tg", r["n_threads"]): r["avg_ts"] for r in rows}


def results(name):
    return {r["id"]: r for r in map(json.loads, (ROOT / "eval" / "results" / f"{name}.jsonl").open(encoding="utf-8"))}


def acc(res):
    return {t: 100 * sum(r["correct"] for r in res.values() if r["task"] == t) / sum(r["task"] == t for r in res.values())
            for t in TASKS}


B = {m: bench(m) for m in BENCHES}
print("### Board throughput (tok/s; same protocol as stage 0)\n")
print("| model | pp64 1T | pp64 4T | tg64 1T | tg64 2T | tg64 3T | tg64 4T | tg64 4T vs baseline 3.53 |")
print("|---|---:|---:|---:|---:|---:|---:|---:|")
for m in BENCHES:
    b = B[m]
    print(f"| {m} | {b[('pp', 1)]:.2f} | {b[('pp', 4)]:.2f} | " + " | ".join(f"{b[('tg', t)]:.2f}" for t in (1, 2, 3, 4))
          + f" | {b[('tg', 4)] / BASE['tg4']:.2f}x |")

print("\n### G2 (candidate >= 4.41 tok/s at 4T and each eval >= baseline - 3)\n")
print("| candidate | tg64 4T | speedup | MMLU | GSM8K | TriviaQA | speed ok | quality ok | G2 |")
print("|---|---:|---:|---:|---:|---:|---|---|---|")
g2_any = False
for m in CANDS:
    a = acc(results(m))
    tg = B[G2_BENCH[m]][("tg", 4)]
    sp = tg >= 1.25 * BASE["tg4"]
    q = all(a[t] >= BASE[t] - 3 for t in TASKS)
    g2_any |= sp and q
    print(f"| {m} | {tg:.2f} | {tg / BASE['tg4']:.2f}x | " + " | ".join(f"{a[t]:.1f}" for t in TASKS)
          + f" | {'yes' if sp else 'no'} | {'yes' if q else 'no'} | **{'PASS' if sp and q else 'FAIL'}** |")
print(f"| baseline Q2_0_g64 | 3.53 | 1.00x | 54.0 | 66.0 | 22.0 | | | |")
print(f"\n**G2: {'PASS' if g2_any else 'FAIL'}**")

if len(sys.argv) == 5:
    un, re_, bun, bre = sys.argv[1:5]
    U, R = results(un), results(re_)
    au, ar = acc(U), acc(R)
    ratio = B[bre][("tg", 4)] / B[bun][("tg", 4)]
    q = all(ar[t] >= au[t] - 3 for t in TASKS)
    print(f"\n### G3 (vocab restriction on the best candidate: {bun})\n")
    print(f"tg64 4T: {B[bun][('tg', 4)]:.2f} -> {B[bre][('tg', 4)]:.2f} tok/s = **{ratio:.3f}x** (need >= 1.10x)\n")
    print("| task | unrestricted | restricted | delta | right->wrong | wrong->right |")
    print("|---|---:|---:|---:|---:|---:|")
    for t in TASKS:
        ids = [i for i in U if U[i]["task"] == t]
        rw = sum(U[i]["correct"] and not R[i]["correct"] for i in ids)
        wr = sum(R[i]["correct"] and not U[i]["correct"] for i in ids)
        print(f"| {t} | {au[t]:.1f} | {ar[t]:.1f} | {ar[t] - au[t]:+.1f} | {rw} | {wr} |")
    g3 = ratio >= 1.10 and q
    print(f"\n**G3: {'PASS' if g3 else 'FAIL'}** (speed {'ok' if ratio >= 1.10 else 'no'}, quality {'ok' if q else 'no'})")
