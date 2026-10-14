"""Export a trained memory checkpoint in shipping form (safetensors) and report its size.
Design A: per layer, folded table int8 + fp16 row scales, bf16 gate/out projections and out norm.
Design B: per Lngram layer, tables int8 + fp16 row scales, everything else bf16.
Usage: python export.py <design A|B> <ckpt.pt> <out.safetensors> [--student ID]"""
import argparse, json, os, sys
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import meki

ap = argparse.ArgumentParser()
ap.add_argument("design"); ap.add_argument("ckpt"); ap.add_argument("out")
ap.add_argument("--student", default="prism-ml/Ternary-Bonsai-1.7B-unpacked")
a = ap.parse_args()
st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
model = AutoModelForCausalLM.from_pretrained(a.student, dtype=torch.float32)
if a.design == "A":
    mods = meki.attach(model, d_mem=64)
    for m, sd in zip(mods, st["mods"]):
        m.float().load_state_dict(sd)
    tensors = meki.export_int8(mods, model.model.embed_tokens.weight)
else:
    import lngram_adapter
    mods = lngram_adapter.attach(model)
    tensors = {}
    for i, (m, sd) in enumerate(zip(mods, st["mods"])):
        m.float().load_state_dict(sd)
        for n, p in m.state_dict().items():
            if ".tables." in f".{n}":
                q, s = meki.quantize_rows(p.float())
                tensors[f"lngram{i}.{n}.int8"], tensors[f"lngram{i}.{n}.scale"] = q, s
            else:
                tensors[f"lngram{i}.{n}"] = p.to(torch.bfloat16)
tensors = {k: v.contiguous() for k, v in tensors.items()}
save_file(tensors, a.out, metadata={"design": a.design, "tokens": str(st.get("tokens"))})
tb = sum(v.numel() * v.element_size() for k, v in tensors.items() if "table" in k)
total = sum(v.numel() * v.element_size() for v in tensors.values())
print(json.dumps({"out": a.out, "table_bytes": tb, "table_MB": tb / 1e6, "total_MB": total / 1e6,
                  "within_600MB": tb <= 600e6}))
