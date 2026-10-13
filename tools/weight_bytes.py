"""Bytes of weights streamed per generated token, per GGUF, plus weights used in matmuls.
Every tensor is read once per token, except token_embd when a separate output.weight exists
(then the embedding is a single-row lookup). Usage: python weight_bytes.py <gguf>... > logs/weight_bytes.json"""
import json, sys
from pathlib import Path
from gguf import GGUFReader

out = {}
for p in sys.argv[1:]:
    t = {x.name: (int(x.n_bytes), x.tensor_type.name, int(x.n_elements)) for x in GGUFReader(p).tensors}
    total = sum(v[0] for v in t.values())
    tied = "output.weight" not in t
    per_tok = total if tied else total - t["token_embd.weight"][0]
    # weights doing a multiply-add per token (the lookup-only embedding excluded)
    mac = sum(v[2] for k, v in t.items() if k != "token_embd.weight" or tied)
    out[Path(p).stem] = {"tensor_bytes": total, "bytes_per_token": per_tok, "matmul_params": mac, "tied_embeddings": tied,
                         "token_embd": t["token_embd.weight"][1], "output": t.get("output.weight", (0, "tied", 0))[1]}
json.dump(out, sys.stdout, indent=1)
