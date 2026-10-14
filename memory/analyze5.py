"""Stage 2a decisions and tables (PREREG5) from eval/results_mem/*.jsonl and runs/*/train_log.jsonl.
  python analyze5.py sanity D          exit 0 iff memory-off (design D, at init) reproduces base within noise
  python analyze5.py continue D        exit 0 iff 100M -> 300M is "still improving" (continue to 1B)
  python analyze5.py g6 D FINAL        exit 0 iff G6 passes for checkpoint FINAL (e.g. 300M or 1000M)
  python analyze5.py report            markdown tables for every design/checkpoint present"""
import glob, json, math, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "eval", "results_mem")
TASKS = ("mmlu", "gsm8k", "triviaqa")


def load(name):
    p = os.path.join(RES, f"{name}.jsonl")
    return {r["id"]: r for r in map(json.loads, open(p, encoding="utf-8"))} if os.path.exists(p) else None


def acc(r, t):
    xs = [v["correct"] for v in r.values() if v["task"] == t]
    return 100 * sum(xs) / len(xs), sum(xs), len(xs)


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * (c - h), 100 * (c + h)


def paired(a, b, t, n_boot=10000):
    ids = sorted(i for i in a if a[i]["task"] == t)
    d = [int(b[i]["correct"]) - int(a[i]["correct"]) for i in ids]
    rng = random.Random(0)
    boots = sorted(100 * sum(rng.choice(d) for _ in d) / len(d) for _ in range(n_boot))
    same = sum(a[i]["output"] == b[i]["output"] for i in ids) / len(ids)
    return (100 * sum(d) / len(d), boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)],
            sum(x == -1 for x in d), sum(x == 1 for x in d), same)


def delta(base, new):
    return {t: acc(new, t)[0] - acc(base, t)[0] for t in TASKS}


def sanity(d):
    base, off = load("base"), load(f"{d}_off_init")
    out = {t: paired(base, off, t) for t in TASKS}
    ok = all(abs(v[0]) <= 1.0 and v[5] >= 0.99 for v in out.values())
    print(json.dumps({t: {"delta": v[0], "identical_outputs": v[5]} for t, v in out.items()}), "OK" if ok else "FAIL")
    return ok


def keep_going(d):
    r100, r300 = load(f"{d}_on_100M"), load(f"{d}_on_300M")
    dl = delta(r100, r300)
    ok = (dl["triviaqa"] >= 1.0 or dl["mmlu"] >= 1.0) and dl["triviaqa"] >= -1.0 and dl["mmlu"] >= -1.0
    print(json.dumps(dl), "CONTINUE" if ok else "STOP")
    return ok


def g6(d, final):
    dl = delta(load("base"), load(f"{d}_on_{final}"))
    ok = dl["triviaqa"] >= 3.0 and dl["mmlu"] >= 2.0 and dl["gsm8k"] >= -2.0
    print(json.dumps(dl), "G6 PASS" if ok else "G6 FAIL")
    return ok


def report():
    base = load("base")
    names = ["base"] + sorted({os.path.basename(p)[:-6] for p in glob.glob(os.path.join(RES, "*.jsonl"))} - {"base"})
    print("### Accuracy (eval v3, % [95% Wilson])\n")
    print("| run | MMLU (1000) | GSM8K (300) | TriviaQA (500) |")
    print("|---|---:|---:|---:|")
    for n in names:
        r = load(n)
        cells = []
        for t in TASKS:
            p, k, m = acc(r, t)
            lo, hi = wilson(k, m)
            cells.append(f"{p:.1f} [{lo:.1f}, {hi:.1f}]")
        print(f"| {n} | " + " | ".join(cells) + " |")
    print("\n### Paired vs base (delta, 95% bootstrap CI, right->wrong / wrong->right, identical outputs)\n")
    print("| run | task | delta | 95% CI | r->w | w->r | identical |")
    print("|---|---|---:|---|---:|---:|---:|")
    for n in names[1:]:
        r = load(n)
        for t in TASKS:
            dd, lo, hi, rw, wr, same = paired(base, r, t)
            print(f"| {n} | {t} | {dd:+.1f} | [{lo:+.1f}, {hi:+.1f}] | {rw} | {wr} | {100 * same:.1f}% |")
    for log in sorted(glob.glob(os.path.join(HERE, "..", "runs", "*", "train_log.jsonl"))):
        recs = [json.loads(l) for l in open(log)]
        vals = [(r["tokens"], r["val_kl"], r["val_ce"]) for r in recs if "val_kl" in r]
        print(f"\n### Validation curve: {os.path.basename(os.path.dirname(log))}\n")
        print("| tokens (M) | val KL | val CE |")
        print("|---:|---:|---:|")
        for tk, kl, ce in vals:
            print(f"| {tk / 1e6:.0f} | {kl:.4f} | {ce:.4f} |")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "report":
        report()
    else:
        sys.exit(0 if {"sanity": sanity, "continue": keep_going, "g6": g6}[cmd](*sys.argv[2:]) else 1)
