"""Freeze the eval set: 200 MMLU + 100 GSM8K + 200 TriviaQA, sampled with seed 0.
Writes eval/evalset.jsonl (one row per question: task, id, prompt, gold). Run once; the
jsonl is committed so every model sees byte-identical prompts."""
import io, json, random, urllib.request
from pathlib import Path
import pandas as pd

API = "https://huggingface.co/api/datasets"
OUT = Path(__file__).with_name("evalset.jsonl")


def parquet(ds, cfg, split):
    with urllib.request.urlopen(f"{API}/{ds}/parquet/{cfg}/{split}/0.parquet") as r:
        return pd.read_parquet(io.BytesIO(r.read()))


def sample(df, n, seed=0):
    return sorted(random.Random(seed).sample(range(len(df)), n))


rows = []

# MMLU, zero-shot, answer letter constrained by grammar at eval time.
mmlu = parquet("cais/mmlu", "all", "test")
for i in sample(mmlu, 200):
    r = mmlu.iloc[i]
    subj = r.subject.replace("_", " ")
    opts = "\n".join(f"{l}. {c}" for l, c in zip("ABCD", r.choices))
    prompt = (f"The following is a multiple choice question about {subj}.\n\n"
              f"{r.question.strip()}\n{opts}\nAnswer:")
    rows.append(dict(task="mmlu", id=f"mmlu-{i}", prompt=prompt, gold="ABCD"[int(r.answer)]))

# GSM8K, 4-shot chain of thought using the first 4 train problems, answer after '####'.
gtrain, gtest = parquet("openai/gsm8k", "main", "train"), parquet("openai/gsm8k", "main", "test")
shots = "".join(f"Question: {q}\nAnswer: {a}\n\n" for q, a in zip(gtrain.question[:4], gtrain.answer[:4]))
for i in sample(gtest, 100):
    r = gtest.iloc[i]
    gold = r.answer.split("####")[-1].strip().replace(",", "")
    rows.append(dict(task="gsm8k", id=f"gsm8k-{i}", prompt=f"{shots}Question: {r.question}\nAnswer:", gold=gold))

# TriviaQA (rc.nocontext validation), 5-shot from the first 5 train questions; gold = alias list.
ttrain = parquet("mandarjoshi/trivia_qa", "rc.nocontext", "train")
tval = parquet("mandarjoshi/trivia_qa", "rc.nocontext", "validation")
shots = "".join(f"Q: {q}\nA: {a['value']}\n\n" for q, a in zip(ttrain.question[:5], ttrain.answer[:5]))
for i in sample(tval, 200):
    r = tval.iloc[i]
    gold = sorted(set(list(r.answer["normalized_aliases"]) + [r.answer["normalized_value"]]))
    rows.append(dict(task="triviaqa", id=f"tqa-{i}", prompt=f"{shots}Q: {r.question}\nA:", gold=gold))

with OUT.open("w", encoding="utf-8", newline="\n") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"wrote {len(rows)} rows to {OUT}")
