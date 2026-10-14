# uno-model

Can a ternary LLM quant beat 4-bit on an Arduino UNO Q (4x Cortex-A53, no dot-product instructions)?

Stage 0 results: [REPORT.md](REPORT.md) (G1: **PASS**, TQ2_0 +43.7% faster than Q4_0, but post-training ternary Qwen3-0.6B collapses on quality). Pre-registration: [PREREG.md](PREREG.md).

Stage 1 results: [REPORT2.md](REPORT2.md) (G2 **FAIL**: Bonsai TQ2_0 is 2.05x faster but MMLU 50.5 misses by one question; G3 **FAIL**: vocab restriction is 1.16x faster but the wikitext token set has no newline; T **FAIL**). Pre-registration: [PREREG2.md](PREREG2.md).

Stage 1b results: [REPORT3.md](REPORT3.md) (new lossless type TQ2_0_G128: **G4 PASS**, 6.68 tok/s = 1.70x PQ2_0, bit-identical weights; vocab v2 **G5 FAIL** on TriviaQA -4.2; T **FAIL**, MMLU 46.8 on 1000 questions). Pre-registration: [PREREG3.md](PREREG3.md).
