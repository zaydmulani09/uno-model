"""Learnability check: on a periodic token stream (next token fully determined by the current one), the
memory alone (base frozen) must cut the CE. Uses the tiny models from smoke_test.py.
Usage: python learn_test.py <smoke work dir> <A|B>"""
import json, os, subprocess, sys
import numpy as np

work, design = sys.argv[1], sys.argv[2]
HERE = os.path.dirname(os.path.abspath(__file__))
data = f"{work}/data_periodic"
os.makedirs(data, exist_ok=True)
period = np.random.default_rng(1).choice(151669, 37, replace=False).astype(np.uint32)
np.tile(period, 400_000 // 37 + 1)[:400_000].tofile(f"{data}/train.bin")
np.tile(period, 40_000 // 37 + 1)[:40_000].tofile(f"{data}/val.bin")
out = f"{work}/learn{design}"
subprocess.run([sys.executable, f"{HERE}/train.py", "--design", design, "--student", f"{work}/student",
                "--teacher", f"{work}/teacher", "--data", data, "--out", out, "--seq", "64", "--global-batch", "4",
                "--micro-batch", "2", "--stop-tokens", str(4 * 64 * 30), "--schedule-tokens", str(4 * 64 * 30),
                "--warmup", "2", "--loss-chunk", "32", "--device", "cpu", "--lr", "3e-2", "--min-lr", "3e-3"],
               check=True, stdout=subprocess.DEVNULL)
recs = [json.loads(l) for l in open(f"{out}/train_log.jsonl")]
v0 = next(r for r in recs if r["step"] == 0)["val_ce"]
v1 = [r for r in recs if "val_ce" in r][-1]["val_ce"]
print(json.dumps({"design": design, "val_ce_start": v0, "val_ce_end": v1}))
assert v1 < v0 - 1.0, "memory did not learn a deterministic next-token map"
print("LEARN OK")
