"""Score one GGUF on the frozen eval set via a local llama-server (greedy, raw completion,
no chat template, so every model gets byte-identical prompts).
Usage: python run_eval.py <llama-server.exe> <model.gguf> <name>
Writes eval/results/<name>.jsonl (every prompt's output) and prints per-task accuracy."""
import json, re, string, subprocess, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
PORT = 8091
SLOTS = 4


def post(payload):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", json.dumps(payload).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["content"]


def norm(s):  # SQuAD/TriviaQA-style answer normalization
    s = "".join(c for c in s.lower() if c not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def last_number(s):
    s = s.split("####")[-1] if "####" in s else s
    nums = re.findall(r"-?\d[\d,]*\.?\d*", s)
    return nums[-1].replace(",", "").rstrip(".") if nums else ""


def run(row):
    base = {"prompt": row["prompt"], "temperature": 0, "top_k": 1, "seed": 0, "cache_prompt": True}
    if row["task"] == "mmlu":
        out = post({**base, "n_predict": 2, "grammar": 'root ::= " "? [ABCD]'})
        ok = out.strip() == row["gold"]
    elif row["task"] == "gsm8k":
        out = post({**base, "n_predict": 256, "stop": ["\n\nQuestion:", "\nQuestion:"]})
        pred = last_number(out)
        ok = pred != "" and float(pred) == float(row["gold"])
    else:
        out = post({**base, "n_predict": 24, "stop": ["\n"]})
        ok = norm(out) in {norm(g) for g in row["gold"]}
    return {"id": row["id"], "task": row["task"], "output": out, "correct": bool(ok)}


def main(server, model, name):
    rows = [json.loads(l) for l in (HERE / "evalset.jsonl").open(encoding="utf-8")]
    log = open(HERE / "results" / f"{name}.server.log", "w")
    proc = subprocess.Popen([server, "-m", model, "--port", str(PORT), "-np", str(SLOTS), "-c", str(4096 * SLOTS),
                             "-t", "8", "--no-webui"], stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(300):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
                break
            except Exception:
                time.sleep(1)
        t0 = time.time()
        with ThreadPoolExecutor(SLOTS) as ex:
            res = list(ex.map(run, rows))
    finally:
        proc.terminate()
        proc.wait()
    with (HERE / "results" / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {"model": name, "seconds": round(time.time() - t0)}
    for t in ("mmlu", "gsm8k", "triviaqa"):
        sub = [r for r in res if r["task"] == t]
        summary[t] = f"{sum(r['correct'] for r in sub)}/{len(sub)}"
    print(json.dumps(summary))


if __name__ == "__main__":
    # self-check of the scorers
    assert last_number("so 5 + 37 = 42\n#### 42") == "42" and last_number("total is $1,234.") == "1234"
    assert norm("The Ghost!") == "ghost"
    (HERE / "results").mkdir(exist_ok=True)
    main(*sys.argv[1:4])
