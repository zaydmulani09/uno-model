"""Stage 1b tables and gates (PREREG3.md) from raw logs: logs/board3/bench_*.jsonl, eval/results/*_v3.jsonl.
95% Wilson intervals per accuracy; paired bootstrap (10000 resamples, seed 0) on differences."""
import json, math, random
from pathlib import Path

ROOT = Path(__file__).parent
TASKS = ("mmlu", "gsm8k", "triviaqa")
PQ, G, V = "Ternary-Bonsai-1.7B-PQ2_0", "Ternary-Bonsai-1.7B-TQ2_0_G128", "Ternary-Bonsai-1.7B-TQ2_0_G128-V2"
REF_TG = 3.94  # PQ2_0 tg64 4T from stage 1 (PREREG3 reference)


def bench(m):
    rows = [json.loads(l) for l in (ROOT / "logs" / "board3" / f"bench_{m}.jsonl").open() if l.startswith("{")]
    return {("pp" if r["n_prompt"] else "tg", r["n_threads"]): r["avg_ts"] for r in rows}


def res(m):
    return {r["id"]: r for r in map(json.loads, (ROOT / "eval" / "results" / f"{m}_v3.jsonl").open(encoding="utf-8"))}


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * (c - h), 100 * (c + h)


def acc(r, t):
    xs = [v["correct"] for v in r.values() if v["task"] == t]
    return 100 * sum(xs) / len(xs), sum(xs), len(xs)


def paired(a, b, t, n_boot=10000):
    ids = sorted(i for i in a if a[i]["task"] == t)
    d = [int(b[i]["correct"]) - int(a[i]["correct"]) for i in ids]
    rng = random.Random(0)
    boots = sorted(100 * sum(rng.choice(d) for _ in d) / len(d) for _ in range(n_boot))
    return (100 * sum(d) / len(d), boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)],
            sum(x == -1 for x in d), sum(x == 1 for x in d))


B = {m: bench(m) for m in (G, V)}
R = {m: res(m) for m in (PQ, G, V)}

print("### Board throughput (tok/s, llama-bench -p 64 -n 64 -r 3)\n")
print("| model | pp64 1T | pp64 4T | tg64 1T | tg64 2T | tg64 3T | tg64 4T | vs PQ2_0 3.94 |")
print("|---|---:|---:|---:|---:|---:|---:|---:|")
for m in (G, V):
    b = B[m]
    print(f"| {m} | {b[('pp', 1)]:.2f} | {b[('pp', 4)]:.2f} | " + " | ".join(f"{b[('tg', t)]:.2f}" for t in (1, 2, 3, 4))
          + f" | {b[('tg', 4)] / REF_TG:.2f}x |")

print("\n### Quality, eval v3 (accuracy % [95% Wilson])\n")
print("| model | MMLU (1000) | GSM8K (300) | TriviaQA (500) |")
print("|---|---:|---:|---:|")
for m in (PQ, G, V):
    cells = []
    for t in TASKS:
        p, k, n = acc(R[m], t)
        lo, hi = wilson(k, n)
        cells.append(f"{p:.1f} [{lo:.1f}, {hi:.1f}]")
    print(f"| {m} | " + " | ".join(cells) + " |")


def gate_table(title, a, b, tol):
    print(f"\n### {title}: {b} vs {a} (difference, paired bootstrap 95% CI, flips)\n")
    print(f"| task | base | new | delta | 95% CI | right->wrong | wrong->right | ok (delta >= -{tol:.0f}) |")
    print("|---|---:|---:|---:|---|---:|---:|---|")
    ok_all = True
    for t in TASKS:
        d, lo, hi, rw, wr = paired(R[a], R[b], t)
        ok = d >= -tol
        ok_all &= ok
        print(f"| {t} | {acc(R[a], t)[0]:.1f} | {acc(R[b], t)[0]:.1f} | {d:+.1f} | [{lo:+.1f}, {hi:+.1f}] | {rw} | {wr} | {'yes' if ok else 'no'} |")
    return ok_all


g4_q = gate_table("G4 quality", PQ, G, 2.0)
g4_s = B[G][("tg", 4)] >= 6.24
print(f"\nG4 speed: {B[G][('tg', 4)]:.2f} tok/s >= 6.24: {'yes' if g4_s else 'no'}")
print(f"\n**G4: {'PASS' if g4_q and g4_s else 'FAIL'}**")

g5_q = gate_table("G5 quality", G, V, 3.0)
ratio = B[V][("tg", 4)] / B[G][("tg", 4)]
print(f"\nG5 speed: {B[G][('tg', 4)]:.2f} -> {B[V][('tg', 4)]:.2f} tok/s = {ratio:.3f}x (>= 1.10: {'yes' if ratio >= 1.10 else 'no'})")
print(f"\n**G5: {'PASS' if g5_q and ratio >= 1.10 else 'FAIL'}**")

eligible = {PQ: REF_TG}
if g4_q:
    eligible[G] = B[G][("tg", 4)]
    if g5_q:
        eligible[V] = B[V][("tg", 4)]
best = max(eligible, key=eligible.get)
mmlu = acc(R[best], "mmlu")[0]
t_ok = eligible[best] >= 6.24 and mmlu >= 51.0
print(f"\n### Target T\n\nEligible: {', '.join(f'{k} ({v:.2f})' for k, v in eligible.items())}")
print(f"Best config: **{best}**, tg64 4T {eligible[best]:.2f} tok/s, MMLU {mmlu:.1f}")
print(f"\n**T: {'PASS' if t_ok else 'FAIL'}**")
