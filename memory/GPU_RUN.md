# Stage 2a on a rented GPU

Everything except the GPU is ready.
- **Code:** in this repo (`memory/`).
- **Tokens:** decontaminated, public at [zaydmulani/uno-model-stage2a-tokens](https://huggingface.co/datasets/zaydmulani/uno-model-stage2a-tokens).
- **Models:** public on Hugging Face.

The launcher needs no credentials.

## Machine
- **GPU:** 1× **H100 80 GB** (SXM preferred), or 1× A100 80 GB (cheaper per hour, about 2.5× slower).
- **Disk:** ≥ 150 GB (Qwen3-8B 16 GB, Bonsai FP16 3.4 GB, tokens 4 GB, checkpoints with optimizer state about 7 GB each).
- **Image:** any recent PyTorch CUDA image, for example RunPod's "PyTorch 2.x".

## Run

```bash
git clone https://github.com/zaydmulani09/uno-model && cd uno-model
nohup bash memory/run_all.sh > run.out 2>&1 &
```

The launcher is resumable: if the machine restarts, run the same command again. Everything it produces ends up in `stage2a_results.tgz`.

## Cost estimate

**Assumptions (per training token):**

| item | GFLOP per token |
|---|---:|
| Teacher Qwen3-8B forward | ~17 |
| Student forward + recompute + input-gradient backward, through a frozen 1.7B with gradient checkpointing | ~11 |
| LM heads | ~2 |
| **Total** | **≈ 30** |

- For the 300M-token pilot that is ≈ 9e18 FLOP.
- Effective throughput, assumed for eager PyTorch at roughly 30–35% MFU: H100 ≈ 300–350 TFLOP/s, A100 ≈ 110–125 TFLOP/s.
- Hourly prices (2026 on-demand, RunPod/Lambda range): H100 $2.0–3.0/h, A100 80 GB $1.2–1.8/h.

| scenario | H100 hours | H100 cost | A100 hours | A100 cost |
|---|---:|---:|---:|---:|
| **Design A, 300M pilot** (train 7–8 h + 6 evals about 1.5 h + setup 0.5 h) | **9–10** | **$18–30** | 22–24 | $27–43 |
| A continued to 1B (if still improving at 300M) | 25–28 | $50–85 | 62–70 | $75–125 |
| Worst case: A to 1B, fails G6, B to 1B | 50–56 | $100–170 | 125–140 | $150–250 |

The first 10 training steps log `tok_per_s`; multiply 1e9 by that rate's inverse to replace these estimates with measured ones.
