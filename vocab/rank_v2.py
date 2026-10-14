"""Vocab restriction v2 (PREREG3.md): rank Bonsai/Qwen3 tokens on a fixed mixed corpus, never eval data.
Sources (equal weight; a token's score is the mean over sources of count / source total):
  fineweb-edu sample-10BT shard 0, first 20000 docs | ultrachat_200k train_sft, first 10000 conversations
  rendered with Bonsai's chat template (thinking off) | codeparrot-clean-valid shard 0, first 10000 files |
  GSM8K train, all, as "Question: ...\\nAnswer: ...\\n\\n".
Docs sharing any 8-word shingle with an eval question (eval/evalset_v3.jsonl) are dropped.
Always kept: ids 0-255 (single bytes), all added tokens, and every token of the chat-template scaffold.
Usage: python rank_v2.py <bonsai-hf dir> <corpus dir> <evalset_v3.jsonl> > vocab/top32768_v2.json"""
import json, re, sys
from collections import Counter
import pandas as pd
from transformers import AutoTokenizer

K = 32768
hf_dir, corpus, evalset = sys.argv[1:4]
tok = AutoTokenizer.from_pretrained(hf_dir)


def shingles(text, n=8):
    w = re.findall(r"\w+", text.lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


# eval questions only (not the few-shot exemplars, which come from train splits)
banned = set()
for row in map(json.loads, open(evalset, encoding="utf-8")):
    p = row["prompt"]
    q = {"mmlu": lambda: p.split("\n\n", 1)[1].rsplit("\nAnswer:", 1)[0],
         "gsm8k": lambda: p.rsplit("Question: ", 1)[1].rsplit("\nAnswer:", 1)[0],
         "triviaqa": lambda: p.rsplit("Q: ", 1)[1].rsplit("\nA:", 1)[0]}[row["task"]]()
    banned |= shingles(q)

fw = pd.read_parquet(f"{corpus}/fineweb-edu-0.parquet", columns=["text"]).text[:20000].tolist()
uc = [tok.apply_chat_template(list(m), tokenize=False, enable_thinking=False)
      for m in pd.read_parquet(f"{corpus}/ultrachat-0.parquet", columns=["messages"]).messages[:10000]]
cp = pd.read_parquet(f"{corpus}/codeparrot-0.parquet", columns=["content"]).content[:10000].tolist()
gs = pd.read_parquet(f"{corpus}/gsm8k-train.parquet")
gs = [f"Question: {q}\nAnswer: {a}\n\n" for q, a in zip(gs.question, gs.answer)]

stats, score = {}, Counter()
for name, docs in [("fineweb-edu", fw), ("ultrachat", uc), ("codeparrot", cp), ("gsm8k-train", gs)]:
    kept = [d for d in docs if not (shingles(d) & banned)]
    counts = Counter()
    for i in range(0, len(kept), 1000):
        for ids in tok(kept[i:i + 1000], add_special_tokens=False)["input_ids"]:
            counts.update(ids)
    total = sum(counts.values())
    for t, c in counts.items():
        score[t] += c / total / 4
    stats[name] = {"docs": len(docs), "dropped_contaminated": len(docs) - len(kept), "tokens": total}

added = sorted(tok.added_tokens_decoder)
scaffold_text = tok.apply_chat_template([{"role": "system", "content": "x"}, {"role": "user", "content": "x"},
                                         {"role": "assistant", "content": "x"}], tokenize=False, enable_thinking=False)
scaffold_text += tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
scaffold = tok(scaffold_text, add_special_tokens=False)["input_ids"]
forced = sorted(set(range(256)) | set(added) | set(scaffold))
ranked = sorted((t for t in score if t not in set(forced)), key=lambda t: (-score[t], t))
keep = forced + ranked[:K - len(forced)]
assert len(keep) == len(set(keep)) == K
json.dump({"sources": stats, "n_forced": len(forced), "n_added": len(added), "scaffold_ids": sorted(set(scaffold)),
           "weighted_coverage": sum(score[t] for t in keep), "distinct_tokens_seen": len(score),
           "ids": sorted(keep)}, sys.stdout)
