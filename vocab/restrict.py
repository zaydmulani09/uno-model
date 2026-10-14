"""Add a restricted LM head to a (tied-embedding) Qwen3 GGUF for the patched llama.cpp.
  output_sub.weight   : the selected rows of the LM head (raw quantized bytes, same type) + 1 zero row
  output_sub.bias     : F32, 0 for the selected rows, -inf for the zero row
  output_sub_map.weight: I32 [n_vocab], row index of each token in output_sub (K for the rest)
Everything else is copied byte for byte.
Usage: python restrict.py <in.gguf> <top32768.json> <out.gguf>"""
import json, sys
import numpy as np
import gguf
from gguf import GGUFReader, GGUFWriter, GGUFValueType

src, ids_path, dst = sys.argv[1:4]
r = GGUFReader(src)
arch = r.fields["general.architecture"].contents()
assert arch == "qwen3", arch
tensors = {t.name: t for t in r.tensors}
head = tensors.get("output.weight") or tensors["token_embd.weight"]
n_vocab = int(head.shape[1])
ids = np.array(json.load(open(ids_path))["ids"], dtype=np.int64)
assert ids.max() < n_vocab and len(set(ids.tolist())) == len(ids)
K = len(ids)

rows = np.asarray(head.data).reshape(n_vocab, -1)        # raw bytes (quantized) or values per row
sub = np.concatenate([rows[ids], np.zeros_like(rows[:1])])  # zero bytes == zero weights for these formats
bias = np.zeros(K + 1, dtype=np.float32)
bias[K] = -np.inf
idmap = np.full(n_vocab, K, dtype=np.int32)
idmap[ids] = np.arange(K, dtype=np.int32)

w = GGUFWriter(dst, arch=arch, endianess=r.endianess)
for name, f in r.fields.items():
    if name.startswith("GGUF.") or name == "general.architecture":
        continue
    vtype = f.types[0]
    sub_type = f.types[-1] if vtype == GGUFValueType.ARRAY else None
    w.add_key_value(name, f.contents(), vtype, sub_type=sub_type)
for t in r.tensors:
    w.add_tensor(t.name, np.asarray(t.data), raw_dtype=t.tensor_type)  # data is (rows, bytes) for quantized
w.add_tensor("output_sub.weight", sub, raw_dtype=head.tensor_type)
w.add_tensor("output_sub.bias", bias)
w.add_tensor("output_sub_map.weight", idmap)
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print(f"{dst}: K={K} of n_vocab={n_vocab}, head type {head.tensor_type.name}, sub {sub.nbytes} bytes")
