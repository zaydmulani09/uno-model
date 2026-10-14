"""Design A: MeKi-style token-indexed memory (arXiv 2602.03359) bolted onto a frozen HF Qwen3.

Per layer, parallel to the FFN:  H' = FFN(H) + MeKi(H, x) + A,  H = RMSNorm_pre_ffn(A)
  e = alpha * RMSNorm(M[x] + beta * G(E[x]))      static table + SwiGLU of the frozen word embedding
  y = RMSNorm_out(W_out (e + sigmoid(W_gate H)))  additive low-rank sigmoid gate
Retrofit deviation (PREREG5): RMSNorm_out.weight starts at 0, so step 0 is exactly the base model.
Export folds G into the table (paper Eq. 9) and stores it int8 with one fp16 scale per row."""
import torch
from torch import nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, init=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.full((dim,), float(init)))
        self.eps = eps

    def forward(self, x):
        x32 = x.float()
        return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight.float()).to(x.dtype)


class MeKi(nn.Module):
    def __init__(self, vocab, hidden, d_mem=64, eps=1e-6, std=0.02):
        super().__init__()
        self.table = nn.Parameter(torch.randn(vocab, d_mem) * std)               # M
        self.g_gate = nn.Linear(hidden, hidden // 2, bias=False)                # G: SwiGLU hidden -> hidden/2 -> d_mem
        self.g_up = nn.Linear(hidden, hidden // 2, bias=False)
        self.g_down = nn.Linear(hidden // 2, d_mem, bias=False)
        self.gate_proj = nn.Linear(hidden, d_mem, bias=False)                   # W_gate
        self.out_proj = nn.Linear(d_mem, hidden, bias=False)                    # W_out
        for lin in (self.g_gate, self.g_up, self.g_down, self.gate_proj, self.out_proj):
            nn.init.normal_(lin.weight, std=std)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.mix_norm = RMSNorm(d_mem, eps)
        self.out_norm = RMSNorm(hidden, eps, init=0.0)                          # retrofit: zero output at init
        self.register_buffer("folded", None, persistent=False)                 # set by fold(): int8-dequantized M~
        self.mode = "on"                                                        # on | zero | shuffle
        self.register_buffer("perm", None, persistent=False)

    def expert(self, ids, word_emb):
        if self.mode == "shuffle":
            ids = self.perm[ids]
        if self.folded is not None:
            e = self.folded[ids].to(self.out_proj.weight.dtype)
        else:
            dyn = self.g_down(F.silu(self.g_gate(word_emb[ids])) * self.g_up(word_emb[ids]))
            e = self.alpha * self.mix_norm(self.table[ids].to(dyn.dtype) + self.beta * dyn)
        return torch.zeros_like(e) if self.mode == "zero" else e

    def forward(self, h, ids, word_emb):
        v = self.expert(ids, word_emb) + torch.sigmoid(self.gate_proj(h))
        return self.out_norm(self.out_proj(v))

    @torch.no_grad()
    def fold(self, word_emb, chunk=16384, int8=True):
        """M~ = alpha * RMSNorm(M + beta * G(E)), optionally int8 per row (as shipped)."""
        rows = []
        for i in range(0, self.table.shape[0], chunk):
            e = word_emb[i:i + chunk].to(self.g_gate.weight.dtype)
            dyn = self.g_down(F.silu(self.g_gate(e)) * self.g_up(e))
            rows.append((self.alpha * self.mix_norm(self.table[i:i + chunk].to(dyn.dtype) + self.beta * dyn)).float())
        m = torch.cat(rows)
        if int8:
            q, scale = quantize_rows(m)
            m = q.float() * scale.float()[:, None]
        self.folded = m
        return m


def quantize_rows(m):
    scale = (m.abs().amax(dim=1) / 127.0).clamp_min(1e-12)
    q = torch.round(m / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return q, scale.to(torch.float16)


class MLPWithMemory(nn.Module):
    """Wraps a frozen decoder MLP: returns mlp(h) + memory(h, current input ids)."""
    def __init__(self, mlp, memory, ctx):
        super().__init__()
        self.mlp, self.memory, self.ctx = mlp, memory, ctx
        self.enabled = True

    def forward(self, h):
        out = self.mlp(h)
        if self.enabled:
            out = out + self.memory(h, self.ctx["ids"], self.ctx["emb"]).to(out.dtype)
        return out


def attach(model, d_mem=64):
    """Add one MeKi per decoder layer. Freezes the base. Returns the list of memory modules."""
    for p in model.parameters():
        p.requires_grad_(False)
    cfg, base = model.config, model.model
    ctx = {"ids": None, "emb": base.embed_tokens.weight}

    def stash_ids(_mod, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        assert ids is not None, "memory needs input_ids (inputs_embeds-only calls are not supported)"
        ctx["ids"] = ids
    base.register_forward_pre_hook(stash_ids, with_kwargs=True)

    dev, dt = base.embed_tokens.weight.device, base.embed_tokens.weight.dtype
    mems = []
    for layer in base.layers:
        mem = MeKi(base.embed_tokens.weight.shape[0], cfg.hidden_size, d_mem, cfg.rms_norm_eps).to(dev, dt)
        mem.table.data = mem.table.data.float()  # table kept fp32 for the optimizer; cast on lookup
        layer.mlp = MLPWithMemory(layer.mlp, mem, ctx)
        mems.append(mem)
    return mems


def set_enabled(model, on):
    for layer in model.model.layers:
        layer.mlp.enabled = on


def set_mode(mems, mode, seed=0):
    g = torch.Generator().manual_seed(seed)
    for m in mems:
        m.mode = mode
        if mode == "shuffle":
            m.perm = torch.randperm(m.table.shape[0], generator=g).to(m.table.device)


def memory_state(mems):
    return {f"layer{i}.{k}": v for i, m in enumerate(mems) for k, v in m.state_dict().items()}


def load_memory_state(mems, sd):
    for i, m in enumerate(mems):
        m.load_state_dict({k.split(".", 1)[1]: v for k, v in sd.items() if k.startswith(f"layer{i}.")})


@torch.no_grad()
def export_int8(mems, word_emb):
    """Shipping format: per layer int8 folded table + fp16 row scales + bf16 gate/out projections and norm."""
    out = {}
    for i, m in enumerate(mems):
        q, scale = quantize_rows(m.fold(word_emb, int8=False))
        m.folded = None
        out[f"layer{i}.table_int8"] = q.cpu()
        out[f"layer{i}.table_scale"] = scale.cpu()
        out[f"layer{i}.gate_proj"] = m.gate_proj.weight.detach().to(torch.bfloat16).cpu()
        out[f"layer{i}.out_proj"] = m.out_proj.weight.detach().to(torch.bfloat16).cpu()
        out[f"layer{i}.out_norm"] = m.out_norm.weight.detach().to(torch.bfloat16).cpu()
    return out


def table_bytes(state):
    return sum(v.numel() * v.element_size() for k, v in state.items() if "table" in k)
