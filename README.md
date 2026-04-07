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

## TurboQuant KV Cache (turbo3/turbo4) - First SM 121 Results

**Date:** April 2026
**Build:** [Madreag/turbo3-cuda](https://github.com/Madreag/turbo3-cuda) `release/cuda-optimized` branch, commit `1766c9133` (build 8793)
**Original TurboQuant:** [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant)

We built Madreag's CUDA-optimized TurboQuant fork on the DGX Spark and ran turbo3/turbo4 vs f16 across multiple context depths. These are the **first SM 121 turbo3/turbo4 results**.

### Token Generation (tg32) - t/s

| Depth | f16 | turbo4 | turbo3 | turbo4 vs f16 | turbo3 vs f16 |
|------:|----:|-------:|-------:|--------------:|--------------:|
| 0 | 45.21 | 44.06 | 43.66 | -2.5% | -3.4% |
| 4,096 | 43.29 | 41.58 | 41.68 | -3.9% | -3.7% |
| 8,192 | 43.37 | 39.49 | 40.60 | -8.9% | -6.4% |
| 16,384 | 43.29 | 36.21 | 36.54 | -16.4% | -15.6% |
| 32,768 | 41.61 | 31.81 | 32.09 | **-23.6%** | **-22.9%** |

### Prompt Processing (pp2048) - t/s

| Depth | f16 | turbo4 | turbo3 | turbo4 vs f16 | turbo3 vs f16 |
|------:|----:|-------:|-------:|--------------:|--------------:|
| 0 | 809.55 | 805.17 | 805.06 | -0.5% | -0.6% |
| 4,096 | 794.71 | 788.86 | 788.90 | -0.7% | -0.7% |
| 8,192 | 780.74 | 776.05 | 776.85 | -0.6% | -0.5% |
| 16,384 | 763.60 | 758.19 | 757.55 | -0.7% | -0.8% |
| 32,768 | 718.57 | 711.26 | 712.34 | -1.0% | -0.9% |

### Analysis

TurboQuant is **consistently slower than f16** on GB10 unified memory, with degradation increasing at deeper context - up to 23.6% slower at 32K. Prompt processing is barely affected (<1%).

The root cause is the same as the standard KV cache quantization findings above: the GB10's 128GB unified LPDDR5X memory (~273 GB/s) eliminates the VRAM pressure that makes KV cache compression beneficial on discrete GPUs like the RTX 5090 (~1,700 GB/s GDDR7). The dequantization compute overhead is not offset by bandwidth savings.

**Recommendation:** Use f16 KV cache on DGX Spark. TurboQuant is designed for — and works great on — VRAM-constrained discrete GPUs. See [research/TURBOQUANT-POLARQUANT-QJL-REVIEW.md](research/TURBOQUANT-POLARQUANT-QJL-REVIEW.md) for a detailed analysis of why.

### Cross-Platform Context

The generation throughput penalty on GB10 is explained by the bandwidth equation: KV cache quantization helps when **memory bandwidth savings > dequantization compute cost**.

| Platform | Memory | Bandwidth | KV Quant Benefit |
|----------|--------|-----------|-----------------|
| H100 SXM | 80GB HBM3 | ~3,350 GB/s | **High** — VRAM-constrained, bandwidth-rich |
| RTX 5090 | 32GB GDDR7 | ~1,700 GB/s | **High** — severely VRAM-constrained |
| RTX 4090 | 24GB GDDR6X | ~1,008 GB/s | **High** — VRAM-constrained |
| **DGX Spark GB10** | **128GB LPDDR5X** | **~273 GB/s** | **Low** — memory-abundant, bandwidth-limited |

On discrete GPUs, TurboQuant's near-optimal 3-bit compression reduces memory traffic over the high-bandwidth HBM/GDDR bus, directly translating to faster attention. On DGX Spark, the 128GB unified pool means memory capacity is rarely the bottleneck, and the lower LPDDR5X bandwidth means dequantization compute dominates.

For DGX Spark users needing to maximize context length, the recommended path is NVIDIA's hardware-accelerated **NVFP4** format (via TensorRT-LLM), which uses dedicated Blackwell tensor core silicon for dequantization rather than software kernels.

### Reproduction

```bash
git clone -b release/cuda-optimized https://github.com/Madreag/turbo3-cuda.git
cd turbo3-cuda
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121
cmake --build build -j$(nproc)

MODEL="/path/to/Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL.gguf"

# Baseline
./build/bin/llama-bench -m "$MODEL" -ctk f16 -ctv f16 \
  -t 20 -ngl 99 -fa 1 -p 512,2048,8192 -n 32 -r 3 \
  -d 0,4096,8192,16384,32768

# TurboQuant
./build/bin/llama-bench -m "$MODEL" -ctk turbo3 -ctv turbo3 \
  -t 20 -ngl 99 -fa 1 -p 512,2048,8192 -n 32 -r 3 \
  -d 0,4096,8192,16384,32768
```

---

## Research

- [**TurboQuant, PolarQuant, and QJL Literature Review**](research/TURBOQUANT-POLARQUANT-QJL-REVIEW.md) — Deep analysis of the three Google Research papers behind TurboQuant, with cross-platform bandwidth analysis explaining why KV cache quantization behaves differently on unified memory vs discrete GPUs.

---

## Raw Data

- [`data/turboquant_benchmark_results.csv`](data/turboquant_benchmark_results.csv) - TurboQuant turbo3/turbo4 vs f16 depth scaling data
- [`data/benchmark_results_v3_complete.csv`](data/benchmark_results_v3_complete.csv) - corrected v3 data (q4_0/q8_0/f16)
- [`data/benchmark_results.csv`](data/benchmark_results.csv) - original v1 data (flawed, kept for reference)

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
