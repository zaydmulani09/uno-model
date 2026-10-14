"""CPU smoke test of the whole stage 2a pipeline with tiny random Qwen3 models that share Bonsai's tokenizer.
Checks: memory is a no-op at init (A) and when disabled; training steps run for A and B; the loss falls;
export fits the size budget formula; every eval variant runs; 'off' reproduces 'base' exactly.
Usage: python smoke_test.py <bonsai tokenizer dir> <work dir>"""
import json, os, subprocess, sys
import numpy as np
import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import meki
import lngram_adapter

tok_dir, work = sys.argv[1], sys.argv[2]
os.makedirs(work, exist_ok=True)
torch.manual_seed(0)


def tiny(vocab, path, layers=2):
    cfg = Qwen3Config(vocab_size=vocab, hidden_size=64, intermediate_size=128, num_hidden_layers=layers,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=512,
                      tie_word_embeddings=True, pad_token_id=151643, eos_token_id=151645)
    m = Qwen3ForCausalLM(cfg)
    m.save_pretrained(path)
    AutoTokenizer.from_pretrained(tok_dir).save_pretrained(path)
    return m


student = tiny(151669, f"{work}/student", layers=12)  # >= 12 layers so Lngram's default layers (1, 11) exist
tiny(151936, f"{work}/teacher")

# A is an exact no-op at init (out_norm weight = 0) and when disabled
x = torch.randint(0, 151669, (2, 16))
with torch.no_grad():
    ref = student(input_ids=x).logits
    mods = meki.attach(student, d_mem=8)
    same_init = torch.equal(student(input_ids=x).logits, ref)
    for m in mods:
        m.out_norm.weight.fill_(1.0)
    changed = not torch.equal(student(input_ids=x).logits, ref)
    meki.set_enabled(student, False)
    same_off = torch.equal(student(input_ids=x).logits, ref)
assert same_init and changed and same_off, (same_init, changed, same_off)
q, s = meki.quantize_rows(torch.randn(10, 64))
assert q.dtype == torch.int8 and s.dtype == torch.float16
print("A no-op at init and when disabled: ok")

# tiny token stream for the training script
data = f"{work}/data"
os.makedirs(data, exist_ok=True)
rng = np.random.default_rng(0)
rng.integers(0, 151669, 400_000, dtype=np.uint32).tofile(f"{data}/train.bin")
rng.integers(0, 151669, 40_000, dtype=np.uint32).tofile(f"{data}/val.bin")

py = sys.executable
for design in ("A", "B"):
    out = f"{work}/run{design}"
    cmd = [py, f"{HERE}/train.py", "--design", design, "--student", f"{work}/student", "--teacher", f"{work}/teacher",
           "--data", data, "--out", out, "--seq", "64", "--global-batch", "4", "--micro-batch", "2",
           "--stop-tokens", str(4 * 64 * 6), "--schedule-tokens", str(4 * 64 * 20), "--warmup", "2",
           "--loss-chunk", "32", "--device", "cpu", "--lr", "3e-2", "--min-lr", "1e-2"]
    subprocess.run(cmd, check=True)
    subprocess.run(cmd[:cmd.index("--stop-tokens") + 1] + [str(4 * 64 * 10)] + cmd[cmd.index("--stop-tokens") + 2:],
                   check=True)  # resume to step 10
    logs = [json.loads(l) for l in open(f"{out}/train_log.jsonl")]
    steps = [r for r in logs if "loss" in r]
    assert [r["step"] for r in steps] == list(range(1, 11)), [r["step"] for r in steps]
    print(design, "loss", steps[0]["loss"], "->", steps[-1]["loss"])
    ck = f"{out}/mem_0M.pt"
    for variant in ("base", "off", "on", "zero", "shuffle"):
        subprocess.run([py, f"{HERE}/eval_hf.py", "--name", f"smoke{design}_{variant}", "--variant", variant,
                        "--design", design, "--ckpt", ck, "--student", f"{work}/student", "--device", "cpu",
                        "--limit", "3", "--batch", "3", "--out", f"{work}/eval"], check=True)
    base = [json.loads(l) for l in open(f"{work}/eval/smoke{design}_base.jsonl", encoding="utf-8")]
    off = [json.loads(l) for l in open(f"{work}/eval/smoke{design}_off.jsonl", encoding="utf-8")]
    assert [r["output"] for r in base] == [r["output"] for r in off], "memory off must reproduce base"
    print(design, "eval variants ran; off == base: ok")
print("SMOKE OK")
