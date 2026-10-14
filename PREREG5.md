# Stage 2a pre-registration: a trained memory for frozen Bonsai

Committed and pushed before any stage 2a result existed. The thresholds below do not change after results. The board is not used in this stage.

## Question
Does a trained token-indexed memory raise knowledge scores of the **frozen** `prism-ml/Ternary-Bonsai-1.7B-unpacked` (FP16 Qwen3 weights)? All original weights stay frozen; only the memory parameters and their gates train.

## Design A: MeKi-style memory (arXiv 2602.03359)
Applied in every one of the 28 decoder layers, **parallel to the FFN**: `H' = FFN(H) + MeKi(H, x) + A`, where `H = RMSNorm_pre_ffn(A)` is the FFN input and `x` is the input token id. Per the paper (Eqs. 1–6):
- **static table:** `m_static = M[x]`, with `M` of size |V| × **d_mem = 64**;
- **dynamic part:** `m_dyn = G(E[x])`, where G is a SwiGLU 2048 → 1024 → 64 on the frozen word embedding E;
- **expert vector:** `e = α · RMSNorm(m_static + β · m_dyn)`, with learnable scalars α and β (init 1);
- **additive low-rank sigmoid gate:** `g = σ(W_gate H)`, with W_gate of size 64 × 2048;
- **output:** `v = e + g`, then `y = RMSNorm(W_out v)`, with W_out of size 2048 × 64.

**Retrofit deviation (declared).** MeKi trains from scratch. Here the base is frozen, so the output RMSNorm's weight is **initialized to 0**. That makes the model exactly the unmodified Bonsai at step 0. Other inits: `M` ~ N(0, 0.02); projections ~ N(0, 0.02).

**Export (as shipped and as evaluated for G6):**
- The dynamic part is folded into the table (paper Eq. 9): `M~ = α · RMSNorm(M + β · G(E))`.
- The table is then quantized to **int8 with one fp16 scale per row**.
- Size: 28 layers × 151,669 rows × 64 bytes = **271.8 MB**, plus 8.5 MB of scales = **280 MB ≤ 600 MB**.
- Gates and output projections stay bf16 (7.4 M params).

## Training
- **Distillation:** teacher `Qwen/Qwen3-8B` (same tokenizer; the teacher's logits are cut to the student's 151,669 ids and renormalized). **Loss = 0.5 · KL(teacher ‖ student) + 0.5 · CE(next token)**, temperature 1, over all positions.
- **Trainable:** the MeKi parameters (M, G, W_gate, W_out, the norms, α, β). Everything else is frozen in bf16.
- **Optimizer:**
  - AdamW, β = (0.9, 0.95), weight decay 0.1 (0 for norms and scalars), gradient clip 1.0, bf16 autocast;
  - peak LR 4e-4, 100 warm-up steps, then a cosine to 2e-4 **defined over the full 1B-token schedule**, so a continuation picks up seamlessly;
  - **batch:** 64 × 2048 tokens = 131,072 tokens per step.
- **Data:**
  - 50% FineWeb-edu (`HuggingFaceFW/fineweb-edu` sample-10BT, shards in order) and 50% English Wikipedia (`wikimedia/wikipedia` 20231101.en, files in order);
  - documents interleaved, packed to 2048 tokens with `<|endoftext|>` separators, tokenized with Bonsai's tokenizer.
  - **Decontamination:** drop any document sharing a word 13-gram (lowercased `\w+` tokens) with any question or answer in `eval/evalset_v3.jsonl`. That covers MMLU question, all four options and the gold answer; GSM8K question and full gold solution; TriviaQA question and every alias. Texts shorter than 13 words can't produce a 13-gram; this is stated, not patched. The dropped count is reported.
- **Budget:**
  - Pilot of **300M tokens**, with the full eval at **100M** and **300M**.
  - **Continue to 1B** only if "still improving": at 300M vs 100M, TriviaQA **or** MMLU is up ≥ 1.0 point, and neither is down by more than 1.0.
  - G6 is scored on the **final checkpoint**: 300M if stopped there, otherwise 1B.

## Evaluation
- **Questions:** eval v3 (MMLU 1000, GSM8K 300, TriviaQA 500) with the same prompts, few-shot exemplars and scoring as stages 0–1c.
- **Harness:** run in PyTorch/transformers, because the memory doesn't exist in llama.cpp. Greedy decoding throughout:
  - MMLU uses the same constraint as the llama.cpp grammar `" "? [ABCD]`: argmax over the letter tokens and the space token, then over the letters;
  - GSM8K: 256 new tokens, stops `\n\nQuestion:` / `\nQuestion:`;
  - TriviaQA: 24 tokens, stop `\n`.
- **Baseline:** the **unmodified model in the same harness**. The stage 1b llama.cpp numbers are reported alongside for reference only.
- **Statistics:** 95% Wilson intervals on every accuracy; 95% paired bootstrap (10,000 resamples, seed 0) on every difference; paired right→wrong / wrong→right flips.
- **Eval-time ablations (reported, not gating):**
  - **memory zeroed:** table rows set to 0, gate and projections kept;
  - **token ids shuffled:** the table is looked up at a fixed random permutation of token ids (seed 0).

## Sanity check (run first)
With the memory **disabled** (module detached), the harness must reproduce the unmodified baseline within noise: every task within ±1.0 point, and ≥ 99% of per-question outputs identical. If this fails, nothing else is scored.

## Gate G6
**PASS** iff the final memory model vs the unmodified model (same harness) has **TriviaQA Δ ≥ +3.0** and **MMLU Δ ≥ +2.0** and **GSM8K Δ ≥ −2.0** points.

## Design B, only if A fails G6
- **Module:** Lngram single-table with approximate gradient backpropagation, from `zyaaa-ux/Lngram` (`Single_table_Lngram.py`, MIT). The file is vendored unchanged, with defaults: target layers (1, 11), 4 bits per route, n-grams (2, 3), memory_dim 16, zero-init tables, surrogate gradient on, table LR ×5, table weight decay 0.
- **Integration:** injected on the pre-attention residual, as in the repo. It is attached with a forward pre-hook, because transformers 5's decoder layer signature differs from the repo's.
- **Size:** 2 × 35.7 M table entries → about 71 MB int8, plus projections.
- **Everything else is the same as A:** frozen base, data, loss, schedule, budget rule, eval, ablations, sanity check and G6.
- **Generation:** Lngram's single-table code has no incremental (KV-cache) path, so its eval decodes with the cache off, recomputing the full sequence each step; that is exact, because the module is causal.

## Compute
- No local GPU: the PC has an Intel Iris Xe and 11.7 GB of RAM. Training needs one 80 GB GPU (H100 or A100): teacher 16.4 GB + student 3.4 GB + memory with optimizer state, plus activations.
- Code, data prep and a CPU smoke test (a tiny random Qwen3 and a Qwen3-0.6B stand-in teacher) are done locally first. The one launcher `memory/run_all.sh` runs everything on the rented GPU.

## Release
If G6 passes, the memory weights (int8 tables + gates) go to a separate public HF repo under `zaydmulani`, with a model card crediting PrismML Bonsai, Qwen3 and MeKi, and no em dashes.
