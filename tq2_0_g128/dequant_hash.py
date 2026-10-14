"""Per-tensor sha256 of the fp32 dequantized weights of a GGUF, for bit-identity checks across formats.
Two hashes per tensor: raw fp32 bytes, and with -0.0 folded to +0.0 (value identity).
Usage: python dequant_hash.py <model.gguf> > hashes.json   (run with the gguf-py that knows the type)"""
import hashlib, json, sys
import numpy as np
from gguf import GGUFReader, GGMLQuantizationType
from gguf.quants import dequantize

PQ2_0 = 142  # PrismML fork type, not dequantizable by any gguf-py


def dequant_pq2_0(data):
    # block: fp16 d, then 32 bytes; weight j of the 128 sits in byte j/4 at bits 2*(j%4); w = (q-1)*d
    b = np.asarray(data, dtype=np.uint8).reshape(-1, 34)
    d = b[:, :2].copy().view(np.float16).astype(np.float32)
    q = (b[:, 2:, None] >> np.array([0, 2, 4, 6], dtype=np.uint8)) & 3
    return (q.reshape(-1, 128).astype(np.int8) - np.int8(1)).astype(np.float32) * d


out = {}
for t in GGUFReader(sys.argv[1]).tensors:
    if len(t.shape) < 2 or t.name.startswith("output_sub"):
        continue  # 1-D norms are F32 in every file; the restricted head is not a model weight
    if int(t.tensor_type) == PQ2_0:
        w = dequant_pq2_0(t.data).reshape(-1)
    else:
        w = dequantize(np.asarray(t.data), t.tensor_type).astype(np.float32).reshape(-1)
    raw = hashlib.sha256(w.tobytes()).hexdigest()
    w = w + np.float32(0.0)  # -0.0 + 0.0 == +0.0
    out[t.name] = {"type": GGMLQuantizationType(t.tensor_type).name, "raw": raw,
                   "value": hashlib.sha256(w.tobytes()).hexdigest(), "n": int(w.size)}
json.dump(out, sys.stdout, indent=0)
