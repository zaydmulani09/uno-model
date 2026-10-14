"""Stage 2a training (PREREG5): frozen Ternary-Bonsai-1.7B (FP16 Qwen3) + memory (design A MeKi or B Lngram),
distilled from Qwen3-8B. Loss = 0.5 KL(teacher || student) + 0.5 CE. Only memory parameters train.
Resumable: run with --stop-tokens 100e6, then 300e6, then 1e9; each stop writes a checkpoint.
The LR schedule is defined over --schedule-tokens (1e9) from the start, so continuation is seamless."""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--design", choices=["A", "B"], required=True)
ap.add_argument("--student", default="prism-ml/Ternary-Bonsai-1.7B-unpacked")
ap.add_argument("--teacher", default="Qwen/Qwen3-8B")
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--stop-tokens", type=float, required=True)
ap.add_argument("--schedule-tokens", type=float, default=1e9)
ap.add_argument("--seq", type=int, default=2048)
ap.add_argument("--global-batch", type=int, default=64)
ap.add_argument("--micro-batch", type=int, default=8)
ap.add_argument("--lr", type=float, default=4e-4)
ap.add_argument("--min-lr", type=float, default=2e-4)
ap.add_argument("--warmup", type=int, default=100)
ap.add_argument("--loss-chunk", type=int, default=512)
ap.add_argument("--device", default="cuda")
ap.add_argument("--max-steps", type=int, default=None, help="smoke tests only")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
torch.manual_seed(0)
dev = torch.device(a.device)
amp = dict(device_type=dev.type, dtype=torch.bfloat16)

student = AutoModelForCausalLM.from_pretrained(a.student, dtype=torch.bfloat16).to(dev)
teacher = AutoModelForCausalLM.from_pretrained(a.teacher, dtype=torch.bfloat16).to(dev).eval()
for p in teacher.parameters():
    p.requires_grad_(False)
V = student.get_output_embeddings().weight.shape[0]  # 151669: teacher logits are cut to this and renormalized

if a.design == "A":
    import meki
    mods = meki.attach(student, d_mem=64)
    for m in mods:
        m.float()  # fp32 master weights; compute runs under bf16 autocast
    no_decay = lambda n: "norm" in n or n.endswith(("alpha", "beta"))
    named = [(f"mem{i}.{n}", p) for i, m in enumerate(mods) for n, p in m.named_parameters()]
    groups = [{"params": [p for n, p in named if not no_decay(n)], "weight_decay": 0.1, "lr_mult": 1.0},
              {"params": [p for n, p in named if no_decay(n)], "weight_decay": 0.0, "lr_mult": 1.0}]
else:
    import lngram_adapter
    mods = lngram_adapter.attach(student)
    for m in mods:
        m.float()
    named = [(f"lng{i}.{n}", p) for i, m in enumerate(mods) for n, p in m.named_parameters()]
    is_table = lambda n: ".memory.tables." in f".{n}"
    no_decay = lambda n: "norm" in n or n.endswith("bias")
    cfg = lngram_adapter.CFG
    groups = [{"params": [p for n, p in named if is_table(n)], "weight_decay": cfg.table_weight_decay,
               "lr_mult": cfg.table_lr_multiplier},
              {"params": [p for n, p in named if not is_table(n) and not no_decay(n)], "weight_decay": 0.1, "lr_mult": 1.0},
              {"params": [p for n, p in named if not is_table(n) and no_decay(n)], "weight_decay": 0.0, "lr_mult": 1.0}]
groups = [g for g in groups if g["params"]]
opt = torch.optim.AdamW(groups, lr=a.lr, betas=(0.9, 0.95), eps=1e-8)
n_train = sum(p.numel() for g in groups for p in g["params"])
student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
student.config.use_cache = False

tokens_per_step = a.global_batch * a.seq
accum = a.global_batch // a.micro_batch
total_steps = math.ceil(a.schedule_tokens / tokens_per_step)
stop_step = math.ceil(a.stop_tokens / tokens_per_step)


def lr_at(step):
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    t = (step - a.warmup) / max(1, total_steps - a.warmup)
    return a.min_lr + 0.5 * (a.lr - a.min_lr) * (1 + math.cos(math.pi * min(1.0, t)))


train = np.memmap(os.path.join(a.data, "train.bin"), dtype=np.uint32, mode="r")
val = np.memmap(os.path.join(a.data, "val.bin"), dtype=np.uint32, mode="r")
row = a.seq + 1


def rows(arr, start_row, n):
    x = np.asarray(arr[start_row * row:(start_row + n) * row], dtype=np.int64).reshape(n, row)
    return torch.from_numpy(x).to(dev)


def chunk_loss(hs, ht, labels):
    """0.5 KL(t||s) + 0.5 CE over one chunk of positions; checkpointed so the 151k-wide logits are not kept."""
    ls = F.log_softmax(student.lm_head(hs).float(), -1)
    lt = F.log_softmax(teacher.lm_head(ht)[..., :V].float(), -1)
    kl = (lt.exp() * (lt - ls)).sum(-1).sum()
    ce = F.nll_loss(ls.reshape(-1, V), labels.reshape(-1), reduction="sum")
    return kl, ce


def batch_loss(x):
    inp, labels = x[:, :-1], x[:, 1:]
    with torch.no_grad(), torch.autocast(**amp):
        ht = teacher.model(input_ids=inp).last_hidden_state
    with torch.autocast(**amp):
        hs = student.model(input_ids=inp).last_hidden_state
    kl = ce = 0.0
    for s in range(0, inp.shape[1], a.loss_chunk):
        sl = slice(s, s + a.loss_chunk)
        with torch.autocast(**amp):
            k, c = checkpoint(chunk_loss, hs[:, sl], ht[:, sl], labels[:, sl], use_reentrant=False)
        kl, ce = kl + k, ce + c
    n = labels.numel()
    return kl / n, ce / n


@torch.no_grad()
def val_loss(n_rows=256):
    student.eval()
    tot_kl = tot_ce = 0.0
    for r in range(0, n_rows, a.micro_batch):
        k, c = batch_loss(rows(val, r, a.micro_batch))
        tot_kl += k.item(); tot_ce += c.item()
    student.train()
    nb = n_rows // a.micro_batch
    return tot_kl / nb, tot_ce / nb


ck = os.path.join(a.out, "latest.pt")
step = 0
if os.path.exists(ck):
    st = torch.load(ck, map_location=dev, weights_only=False)
    for m, sd in zip(mods, st["mods"]):
        m.load_state_dict(sd)
    opt.load_state_dict(st["opt"])
    step = st["step"]
    print(f"resumed at step {step} ({step * tokens_per_step / 1e6:.0f}M tokens)", flush=True)

log = open(os.path.join(a.out, "train_log.jsonl"), "a")
if step == 0:
    vk, vc = val_loss()
    log.write(json.dumps({"step": 0, "tokens": 0, "val_kl": vk, "val_ce": vc, "trainable_params": n_train}) + "\n")
student.train()
t0 = time.time()
while step < stop_step and (a.max_steps is None or step < a.max_steps):
    lr = lr_at(step)
    for g in opt.param_groups:
        g["lr"] = lr * g["lr_mult"]
    opt.zero_grad(set_to_none=True)
    skl = sce = 0.0
    for micro in range(accum):
        x = rows(train, step * a.global_batch + micro * a.micro_batch, a.micro_batch)
        kl, ce = batch_loss(x)
        ((0.5 * kl + 0.5 * ce) / accum).backward()
        skl += kl.item() / accum; sce += ce.item() / accum
    gn = torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0).item()
    opt.step()
    step += 1
    rec = {"step": step, "tokens": step * tokens_per_step, "lr": lr, "kl": skl, "ce": sce, "loss": 0.5 * skl + 0.5 * sce,
           "grad_norm": gn, "tok_per_s": tokens_per_step / (time.time() - t0)}
    t0 = time.time()
    if step % 200 == 0 or step == stop_step:
        rec["val_kl"], rec["val_ce"] = val_loss()
    log.write(json.dumps(rec) + "\n"); log.flush()
    if step % 10 == 0:
        print(json.dumps(rec), flush=True)

state = {"step": step, "mods": [m.state_dict() for m in mods], "opt": opt.state_dict()}
torch.save(state, ck + ".tmp"); os.replace(ck + ".tmp", ck)
torch.save({"design": a.design, "step": step, "tokens": step * tokens_per_step,
            "mods": [m.state_dict() for m in mods]}, os.path.join(a.out, f"mem_{step * tokens_per_step // 1_000_000}M.pt"))
print(f"stopped at step {step} = {step * tokens_per_step / 1e6:.0f}M tokens", flush=True)
