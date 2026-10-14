# uno-model

Can a ternary LLM quant beat 4-bit on an Arduino UNO Q (4x Cortex-A53, no dot-product instructions)?

Stage 0 results: [REPORT.md](REPORT.md) (G1: **PASS**, TQ2_0 +43.7% faster than Q4_0, but post-training ternary Qwen3-0.6B collapses on quality). Pre-registration: [PREREG.md](PREREG.md).

Stage 1 results: [REPORT2.md](REPORT2.md) (G2 **FAIL**: Bonsai TQ2_0 is 2.05x faster but MMLU 50.5 misses by one question; G3 **FAIL**: vocab restriction is 1.16x faster but the wikitext token set has no newline; T **FAIL**). Pre-registration: [PREREG2.md](PREREG2.md).
