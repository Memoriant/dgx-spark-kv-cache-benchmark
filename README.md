# KV Cache Quantization on NVIDIA DGX Spark GB10

> **Corrected benchmarks** (v3, April 2026) — KV cache quantization behavior on the NVIDIA DGX Spark's GB10 Grace Blackwell unified memory architecture.

**Author:** Nathan Maine, [Memoriant Inc.](https://www.memoriant.ai)
**Date:** March 2026, corrected April 2026
**Hardware:** NVIDIA DGX Spark (GB10, compute 12.1, 128GB unified memory)

> **Correction Notice:** The original v1 benchmarks (March 31) contained methodology errors. Memory was measured via RSS (wrong on unified memory) and some throughput data came from failed requests. v3 uses nvidia-smi + llama.cpp internal reporting. See [CORRECTION-NOTICE.md](CORRECTION-NOTICE.md) for full details. Credit to u/audioen on r/LocalLLaMA for identifying the RSS measurement flaw.

---

## TL;DR

KV cache quantization on DGX Spark GB10 works as expected:

- **q4_0 saves 72% KV buffer memory** (216 MiB vs 768 MiB for f16)
- **q8_0 saves 47% KV buffer memory** (408 MiB vs 768 MiB for f16)
- **Prompt throughput is unaffected** by cache quantization at all context lengths
- **Generation throughput degrades ~37%** at 110K context with q4_0 (24 tps vs 38 tps for f16)

---

## Memory (Corrected — nvidia-smi + llama.cpp internals)

| Cache Type | KV Buffer (llama.cpp) | Total GPU (nvidia-smi) | Savings vs f16 |
|-----------|----------------------|----------------------|---------------|
| **f16** | 768 MiB | 23,092 MiB | baseline |
| **q8_0** | 408 MiB | 22,732 MiB | **-360 MiB (-47% KV)** |
| **q4_0** | 216 MiB | 22,540 MiB | **-552 MiB (-72% KV)** |

At 110K context:

| Cache Type | GPU Memory | vs f16 |
|-----------|-----------|--------|
| f16 | 23,116 MiB | baseline |
| q8_0 | 22,856 MiB | -260 MiB |
| q4_0 | 22,664 MiB | -452 MiB |

---

## Throughput

### Prompt Processing (tokens/sec) — No degradation

| Context | f16 | q8_0 | q4_0 |
|---------|-----|------|------|
| ~1.5K | 923 | 925 | 926 |
| ~6K | 1,211 | 1,207 | 1,206 |
| ~12K | 1,188 | 1,184 | 1,191 |
| ~24K | 1,153 | 1,149 | 1,152 |
| **~110K** | **815** | **810** | **813** |

Prompt throughput is essentially identical across all cache types at all context lengths.

### Generation (tokens/sec) — Degrades at long context

| Context | f16 | q8_0 | q4_0 | q4_0 vs f16 |
|---------|-----|------|------|------------|
| ~1.5K | 45.2 | 45.3 | 45.6 | +0.9% |
| ~6K | 44.7 | 44.9 | 45.0 | +0.7% |
| ~12K | 44.9 | 42.9 | 42.7 | -4.9% |
| ~24K | 44.6 | 39.7 | 39.3 | -11.9% |
| **~110K** | **38.0** | **25.0** | **24.0** | **-36.8%** |

Generation (decode) throughput degrades with quantized KV cache at long context. At 110K tokens, q4_0 is 37% slower than f16 for generation. q8_0 is similar at 34% slower.

---

## Test Setup

```
Model:     Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL.gguf
Hardware:  NVIDIA DGX Spark GB10 (compute 12.1, 124,610 MiB VRAM)
OS:        DGX OS / Ubuntu aarch64
llama.cpp: build 8399 (commit 892e3c333), aarch64 + CUDA
CUDA:      13.0 | Driver: 580.126.09
Flags:     --ctx-size 131072
Protocol:  Server restarted between each configuration
Memory:    nvidia-smi --query-compute-apps for GPU memory
KV size:   llama.cpp verbose output (llama_kv_cache line)
Throughput: llama.cpp response timings via /v1/chat/completions
```

---

## What Was Wrong in v1

The original paper (March 31) made two incorrect claims:

1. **"92.5% prompt throughput collapse at 64K"** — Wrong. Prompt throughput is unaffected by cache quantization. The original data likely came from failed completion requests. The actual effect is a 37% generation speed reduction at 110K context.

2. **"q4_0 uses MORE memory than f16"** — Wrong. This was measured via process RSS, which does not capture GPU/unified memory allocations on GB10. Actual measurement via nvidia-smi + llama.cpp shows q4_0 saves 552 MiB as expected.

See [CORRECTION-NOTICE.md](CORRECTION-NOTICE.md) for full methodology comparison.

---

## Actual Finding: Generation Decode Overhead

The real finding is more nuanced: **KV cache quantization saves memory as expected, but imposes a generation speed tax at long context.** At 110K tokens, q4_0 generation is 37% slower than f16 (24 vs 38 tps). This is likely due to dequantization overhead during the decode attention step, which processes the full KV cache for each generated token.

Prompt processing is unaffected because it processes all tokens in parallel — the dequantization cost is amortized across the batch.

This tradeoff may be acceptable depending on the use case:
- **Long-context RAG** (mostly prompt, few generated tokens): use q4_0, save memory
- **Long-form generation at long context**: use f16, preserve decode speed

---

## Raw Data

- [`data/benchmark_results_v3_complete.csv`](data/benchmark_results_v3_complete.csv) — corrected v3 data
- [`data/benchmark_results.csv`](data/benchmark_results.csv) — original v1 data (flawed, kept for reference)

---

## Citation

```bibtex
@techreport{maine2026kvcache,
  title  = {KV Cache Quantization on NVIDIA DGX Spark GB10},
  author = {Maine, Nathan},
  institution = {Memoriant Inc.},
  year   = {2026},
  note   = {Corrected April 2026},
  url    = {https://github.com/Memoriant/dgx-spark-kv-cache-benchmark}
}
```
