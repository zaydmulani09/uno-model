"""Eval v3 in PyTorch/transformers (the memory doesn't exist in llama.cpp). Same prompts and scoring as
eval/run_eval.py; greedy everywhere.
  MMLU: the llama.cpp grammar  " "? [ABCD]  done exactly: argmax over {A..D, " A".." D", " "}; if " " wins,
        argmax over {A..D} at the next position.
  GSM8K: 256 new tokens, stops "\\n\\nQuestion:" / "\\nQuestion:".  TriviaQA: 24 new tokens, stop "\\n".
Variants: base (no memory) | off (memory attached, disabled) | on | zero (table rows 0) | shuffle (ids permuted).
Design A is evaluated in its shipping form: folded table, int8 per row.
Usage: python eval_hf.py --name NAME --variant on --design A --ckpt mem_300M.pt [--limit N]"""
import argparse, json, os, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "eval"))
from run_eval import last_number, norm  # same scorers as the llama.cpp eval

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--variant", choices=["base", "off", "on", "zero", "shuffle"], required=True)
ap.add_argument("--design", choices=["A", "B"])
ap.add_argument("--ckpt")
ap.add_argument("--student", default="prism-ml/Ternary-Bonsai-1.7B-unpacked")
ap.add_argument("--evalset", default=os.path.join(HERE, "..", "eval", "evalset_v3.jsonl"))
ap.add_argument("--out", default=os.path.join(HERE, "..", "eval", "results_mem"))
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--device", default="cuda")
ap.add_argument("--limit", type=int, default=None, help="per task, smoke tests only")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = torch.device(a.device)
dt = torch.bfloat16 if dev.type == "cuda" else torch.float32

tok = AutoTokenizer.from_pretrained(a.student)
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(a.student, dtype=dt).to(dev).eval()
use_cache = True
if a.variant != "base":
    st = torch.load(a.ckpt, map_location=dev, weights_only=False) if a.ckpt else None
    if a.design == "A":
        import meki
        mods = meki.attach(model, d_mem=64)
        if st:
            for m, sd in zip(mods, st["mods"]):
                m.load_state_dict(sd)
        for m in mods:
            m.float().fold(model.model.embed_tokens.weight, int8=True)  # shipping form
        meki.set_mode(mods, a.variant if a.variant in ("zero", "shuffle") else "on")
        meki.set_enabled(model, a.variant != "off")
    else:
        import lngram_adapter
        mods = lngram_adapter.attach(model)
        if st:
            for m, sd in zip(mods, st["mods"]):
                m.load_state_dict(sd)
        for m in mods:
            m.float().eval()
            m.mode = a.variant if a.variant in ("zero", "shuffle") else "on"
            m.enabled = a.variant != "off"
        use_cache = False  # no incremental path in the single-table Lngram; full recompute is exact (causal)

rows = [json.loads(l) for l in open(a.evalset, encoding="utf-8")]
if a.limit:
    rows = [r for t in ("mmlu", "gsm8k", "triviaqa") for r in [x for x in rows if x["task"] == t][:a.limit]]
amp = dict(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")


def ids_of(s):
    t = tok.encode(s, add_special_tokens=False)
    assert len(t) == 1, (s, t)
    return t[0]


LET = [ids_of(c) for c in "ABCD"]
SLET = [ids_of(" " + c) for c in "ABCD"]
SPACE = ids_of(" ")


@torch.no_grad()
def last_logits(prompts):
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
    with torch.autocast(**amp):
        return model(**enc, use_cache=False).logits[:, -1].float()


def mmlu(batch):
    lg = last_logits([r["prompt"] for r in batch])
    allowed = torch.tensor(LET + SLET + [SPACE], device=dev)
    pick = allowed[lg[:, allowed].argmax(1)].tolist()
    out = [None] * len(batch)
    redo = []
    for i, p in enumerate(pick):
        if p == SPACE:
            redo.append(i)
        else:
            out[i] = "ABCD"[(LET + SLET).index(p) % 4]
    if redo:
        lg2 = last_logits([batch[i]["prompt"] + " " for i in redo])
        for j, i in enumerate(redo):
            out[i] = "ABCD"[int(lg2[j, LET].argmax())]
    return out


@torch.no_grad()
def generate(batch, n_new, stops):
    enc = tok([r["prompt"] for r in batch], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
    with torch.autocast(**amp):
        g = model.generate(**enc, max_new_tokens=n_new, do_sample=False, use_cache=use_cache,
                           stop_strings=stops, tokenizer=tok, pad_token_id=tok.pad_token_id)
    outs = []
    for seq in g[:, enc["input_ids"].shape[1]:]:
        s = tok.decode(seq, skip_special_tokens=True)
        cut = min([s.find(x) for x in stops if x in s], default=len(s))
        outs.append(s[:cut])
    return outs


results, t0 = [], time.time()
for task in ("mmlu", "gsm8k", "triviaqa"):
    sub = [r for r in rows if r["task"] == task]
    for i in range(0, len(sub), a.batch):
        b = sub[i:i + a.batch]
        if task == "mmlu":
            outs = mmlu(b)
            ok = [o == r["gold"] for o, r in zip(outs, b)]
        elif task == "gsm8k":
            outs = generate(b, 256, ["\n\nQuestion:", "\nQuestion:"])
            ok = [(lambda p: p != "" and float(p) == float(r["gold"]))(last_number(o)) for o, r in zip(outs, b)]
        else:
            outs = generate(b, 24, ["\n"])
            ok = [norm(o) in {norm(g) for g in r["gold"]} for o, r in zip(outs, b)]
        results += [{"id": r["id"], "task": task, "output": o, "correct": bool(c)} for r, o, c in zip(b, outs, ok)]
    print(task, f"{sum(x['correct'] for x in results if x['task'] == task)}/{len(sub)}", f"{time.time() - t0:.0f}s", flush=True)

with open(os.path.join(a.out, f"{a.name}.jsonl"), "w", encoding="utf-8", newline="\n") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
summary = {"name": a.name, "variant": a.variant, "design": a.design, "ckpt": a.ckpt,
           **{t: f"{sum(x['correct'] for x in results if x['task'] == t)}/{sum(x['task'] == t for x in results)}"
              for t in ("mmlu", "gsm8k", "triviaqa")}}
print(json.dumps(summary))
with open(os.path.join(a.out, "summary.jsonl"), "a") as f:
    f.write(json.dumps(summary) + "\n")
