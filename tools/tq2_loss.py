"""How lossy is re-blocking Bonsai's ternary g128 weights into TQ2_0 (one scale per 256)?
For every 2-D weight: native scale per 128-group = max|w|; TQ2_0 per 256-block d = max|w|,
q = round(w / d) in {-1,0,1}. Counts weights whose reconstructed value differs from the original.
Usage: python tq2_loss.py <model.safetensors>"""
import json, sys
import torch
from safetensors import safe_open

tot = changed = zeroed = blocks = mixed = 0
with safe_open(sys.argv[1], "pt") as f:
    for name in f.keys():
        w = f.get_tensor(name).float()
        if w.dim() != 2:
            continue
        g = w.reshape(-1, 2, 128)                       # 256-blocks of two native 128-groups
        s = g.abs().amax(dim=2)                          # native per-group scales
        d = s.amax(dim=1, keepdim=True)                  # TQ2_0 per-block scale
        q = torch.round(g / d.unsqueeze(2).clamp_min(1e-30)).clamp(-1, 1)
        rec = q * d.unsqueeze(2)
        diff = (rec - g).abs() > 1e-3 * d.unsqueeze(2)
        tot += g.numel(); changed += int(diff.sum()); zeroed += int(((q == 0) & (g != 0)).sum())
        blocks += s.shape[0]; mixed += int((s[:, 0] != s[:, 1]).sum())
print(json.dumps({"weights": tot, "changed_frac": changed / tot, "nonzero_rounded_to_zero_frac": zeroed / tot,
                  "blocks_256": blocks, "blocks_with_two_different_scales_frac": mixed / blocks}, indent=1))
