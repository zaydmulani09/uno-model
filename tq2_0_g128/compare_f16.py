"""Element-level comparison of the TQ2_0_G128 GGUF against its F16 source: how many weights differ,
and whether every difference sits in a group whose F16 values hold two distinct magnitudes.
Usage: python compare_f16.py <f16.gguf> <tq2_0_g128.gguf>"""
import json, sys
import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize

f = {t.name: t for t in GGUFReader(sys.argv[1]).tensors}
g = {t.name: t for t in GGUFReader(sys.argv[2]).tensors}
tot = diff = groups = groups_two_mag = 0
for name, tg in g.items():
    if len(tg.shape) < 2:
        continue
    a = dequantize(np.asarray(f[name].data), f[name].tensor_type).astype(np.float32).reshape(-1, 128)
    b = dequantize(np.asarray(tg.data), tg.tensor_type).astype(np.float32).reshape(-1, 128)
    bad = (a + np.float32(0)).view(np.uint32) != (b + np.float32(0)).view(np.uint32)
    tot += a.size
    diff += int(bad.sum())
    for r in np.where(bad.any(axis=1))[0]:
        groups += 1
        mags = np.unique(np.abs(a[r])[a[r] != 0])
        groups_two_mag += len(mags) == 2
json.dump({"weights": tot, "weights_differing": diff, "frac": diff / tot, "groups_differing": groups,
           "of_which_two_magnitudes": groups_two_mag}, sys.stdout, indent=1)
