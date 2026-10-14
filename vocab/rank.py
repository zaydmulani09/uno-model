"""Rank Qwen3/Bonsai tokens by frequency on wikitext-103-raw-v1 train (not the eval set) and
keep the top 32768: all special/added tokens first, the rest by count, ties to the lower id.
Usage: python rank.py [--force-bytes] <tokenizer.json> <train-0.parquet> <train-1.parquet> > vocab/top32768.json
--force-bytes (POST-HOC, stage 1 exploratory): also always keep the 256 single-byte tokens (ids 0-255),
which include the newline and unspaced punctuation that wikitext's space-separated formatting never produces."""
import json, sys
from collections import Counter
import pandas as pd
from tokenizers import Tokenizer

K = 32768
force_bytes = sys.argv[1] == "--force-bytes"
args = sys.argv[2:] if force_bytes else sys.argv[1:]
tok = Tokenizer.from_file(args[0])
added = sorted(t.id if hasattr(t, "id") else i for i, t in tok.get_added_tokens_decoder().items())

counts = Counter()
n_lines = n_tokens = 0
for path in args[1:]:
    lines = [l for l in pd.read_parquet(path).text if l.strip()]
    for i in range(0, len(lines), 10000):
        for enc in tok.encode_batch(lines[i:i + 10000], add_special_tokens=False):
            counts.update(enc.ids)
            n_tokens += len(enc.ids)
    n_lines += len(lines)

keep = list(added) + (list(range(256)) if force_bytes else [])
ranked = sorted((t for t in counts if t not in set(keep)), key=lambda t: (-counts[t], t))
keep += ranked[:K - len(keep)]
covered = sum(counts[t] for t in keep)
assert len(keep) == len(set(keep)) == K
json.dump({"corpus": "Salesforce/wikitext wikitext-103-raw-v1 train (all non-empty lines)",
           "lines": n_lines, "tokens": n_tokens, "distinct_tokens_seen": len(counts),
           "corpus_token_coverage": covered / n_tokens, "n_added_forced": len(added), "force_bytes": force_bytes,
           "ids": sorted(keep)}, sys.stdout)
