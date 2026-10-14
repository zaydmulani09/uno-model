"""Stage 2a training data (PREREG5): 50% FineWeb-edu + 50% English Wikipedia, documents interleaved,
tokenized with Bonsai's tokenizer, <|endoftext|> between documents, written as one uint32 token stream.
Decontamination: drop any document sharing a word 13-gram with any question or answer of eval v3
(MMLU question + all options; GSM8K question + full gold solution; TriviaQA question + every alias).
Usage: python data_prep.py <tokenizer dir> <evalset_v3.jsonl> <out dir> <train tokens> <val tokens>
       --fw <fineweb parquet>... --wiki <wikipedia parquet>...
Writes train.bin, val.bin (uint32) and manifest.json."""
import argparse, io, json, os, re, urllib.request
from multiprocessing import Pool
import numpy as np
import pandas as pd
from tokenizers import Tokenizer

N = 13
WORD = re.compile(r"\w+")
EOS = 151643  # <|endoftext|>


def grams(text):
    w = WORD.findall(text.lower())
    return {" ".join(w[i:i + N]) for i in range(len(w) - N + 1)}


def eval_texts(evalset):
    """Every question and answer text of eval v3."""
    gsm = pd.read_parquet(io.BytesIO(urllib.request.urlopen(
        "https://huggingface.co/api/datasets/openai/gsm8k/parquet/main/test/0.parquet").read()))
    out = []
    for r in map(json.loads, open(evalset, encoding="utf-8")):
        p = r["prompt"]
        if r["task"] == "mmlu":
            body = p.split("\n\n", 1)[1].rsplit("\nAnswer:", 1)[0]  # question + "A. ..." option lines
            lines = body.split("\n")
            out.append("\n".join(lines[:-4]))
            out += [l[3:] for l in lines[-4:]]
        elif r["task"] == "gsm8k":
            out.append(p.rsplit("Question: ", 1)[1].rsplit("\nAnswer:", 1)[0])
            out.append(gsm.answer[int(r["id"].split("-")[1])])
        else:
            out.append(p.rsplit("Q: ", 1)[1].rsplit("\nA:", 1)[0])
            out += list(r["gold"])
    return out


BANNED = None


def _init(banned):
    global BANNED
    BANNED = banned


def _clean(text):
    w = WORD.findall(text.lower())
    for i in range(len(w) - N + 1):
        if " ".join(w[i:i + N]) in BANNED:
            return False
    return True


def docs(paths, col):
    for p in paths:
        for t in pd.read_parquet(p, columns=[col])[col]:
            if t and t.strip():
                yield t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tok"); ap.add_argument("evalset"); ap.add_argument("out")
    ap.add_argument("train_tokens", type=int); ap.add_argument("val_tokens", type=int)
    ap.add_argument("--fw", nargs="+", required=True); ap.add_argument("--wiki", nargs="+", required=True)
    a = ap.parse_args()

    banned = set().union(*map(grams, eval_texts(a.evalset)))
    tok = Tokenizer.from_file(f"{a.tok}/tokenizer.json")
    half = (a.train_tokens + a.val_tokens) // 2
    streams = {"fineweb-edu": docs(a.fw, "text"), "wikipedia": docs(a.wiki, "text")}
    stats = {k: {"docs_seen": 0, "docs_dropped": 0, "tokens": 0} for k in streams}
    lens = {k: [] for k in streams}

    # pass 1: per source, stream clean tokenized documents to a temp file (keeps RAM flat)
    with Pool(8, initializer=_init, initargs=(banned,)) as pool:
        for name, stream in streams.items():
            with open(f"{a.out}/{name}.tmp", "wb") as f:
                while stats[name]["tokens"] < half:
                    batch = [d for _, d in zip(range(4000), stream)]
                    if not batch:
                        raise SystemExit(f"{name}: ran out of documents at {stats[name]['tokens']} tokens")
                    keep = [d for d, ok in zip(batch, pool.map(_clean, batch, chunksize=50)) if ok]
                    stats[name]["docs_seen"] += len(batch)
                    stats[name]["docs_dropped"] += len(batch) - len(keep)
                    for enc in tok.encode_batch(keep, add_special_tokens=False):
                        ids = np.asarray(enc.ids + [EOS], dtype=np.uint32)
                        f.write(ids.tobytes())
                        lens[name].append(ids.size)
                        stats[name]["tokens"] += ids.size
                    print(name, stats[name], flush=True)

    # pass 2: per source the first val_tokens/2 tokens of whole documents -> val, the rest -> train;
    # interleave documents so the two sources' cumulative token counts stay balanced
    src = {k: np.memmap(f"{a.out}/{k}.tmp", dtype=np.uint32, mode="r") for k in streams}
    spans = {}
    for k in streams:
        starts = np.concatenate([[0], np.cumsum(lens[k])])
        cut = int(np.searchsorted(starts, a.val_tokens // 2))
        spans[k] = ([(starts[i], starts[i + 1]) for i in range(cut)],
                    [(starts[i], starts[i + 1]) for i in range(cut, len(lens[k]))])

    def interleave(path, sa, sb, limit):
        ia = ib = na = nb = 0
        with open(path, "wb") as f:
            while (ia < len(sa) or ib < len(sb)) and na + nb < limit:
                if ib >= len(sb) or (ia < len(sa) and na <= nb):
                    s, e = sa[ia]; chunk = src["fineweb-edu"][s:e]; na += e - s; ia += 1
                else:
                    s, e = sb[ib]; chunk = src["wikipedia"][s:e]; nb += e - s; ib += 1
                f.write(np.asarray(chunk[:max(0, limit - (na + nb - (e - s)))]).tobytes())
        return min(limit, na + nb)

    n_train = interleave(f"{a.out}/train.bin", spans["fineweb-edu"][1], spans["wikipedia"][1], a.train_tokens)
    n_val = interleave(f"{a.out}/val.bin", spans["fineweb-edu"][0], spans["wikipedia"][0], a.val_tokens)
    del src
    for k in streams:
        os.remove(f"{a.out}/{k}.tmp")
    json.dump({"tokenizer": "prism-ml/Ternary-Bonsai-1.7B-unpacked", "eos": EOS, "dtype": "uint32",
               "train_tokens": int(n_train), "val_tokens": int(n_val), "ngram": N,
               "banned_ngrams": len(banned), "sources": stats,
               "fineweb_files": a.fw, "wiki_files": a.wiki}, open(f"{a.out}/manifest.json", "w"), indent=1)
    print("done", n_train, n_val)


if __name__ == "__main__":
    main()
