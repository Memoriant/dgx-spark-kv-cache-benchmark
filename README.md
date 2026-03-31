# KV Cache Quantization on NVIDIA DGX Spark GB10

> **Three novel empirical findings** about KV cache quantization behavior on the NVIDIA DGX Spark's GB10 Grace Blackwell unified memory architecture.

**Author:** Nathan Maine, [Memoriant Inc.](https://www.memoriant.ai)
**Date:** March 2026
**Hardware:** NVIDIA DGX Spark (GB10, compute 12.1, 128GB unified memory)

---

## TL;DR

If you have a DGX Spark and are using llama.cpp with KV cache quantization:

- ✅ **Use `--cache-type-k q8_0 --cache-type-v q8_0`** for long contexts (32K–131K)
- ❌ **Never use `--cache-type-k q4_0`** at long context — 92.5% speed collapse at 64K
- ⚠️ **q4_0 uses MORE memory than f16** on unified memory — the opposite of what you'd expect

---

## The Three Findings

### 1. The Dequantization Cliff

q4_0 KV cache degrades non-linearly and collapses at 64K context:

| Context | f16 prompt tps | q4_0 prompt tps | Delta |
|---------|---------------|-----------------|-------|
| ~8K | 371.3 | 363.4 | -2.1% |
| ~16K | 360.7 | 346.2 | -4.0% |
| ~32K | 328.3 | 316.9 | -3.5% |
| **~64K** | **282.7** | **21.3** | **-92.5%** |

**At 64K tokens, q4_0 drops from 283 tps to 21 tps.** This is not a tradeoff — it is a failure mode.

### 2. The Unified Memory Paradox

On conventional GPUs, KV quantization saves VRAM. On GB10 unified memory, q4_0 uses **more** RAM than f16:

| Context | f16 RSS | q4_0 RSS | q4_0 vs f16 |
|---------|---------|---------|-------------|
| ~8K | 1.25 GB | 1.34 GB | **+7%** |
| ~32K | 1.59 GB | 1.69 GB | **+6%** |
| ~64K | 1.94 GB | 2.06 GB | **+6%** |

No crossover point found. The dequantization workspace + metadata overhead exceeds the storage savings at every context length.

### 3. The q8_0 Sweet Spot

q8_0 is the only quantized format that provides a genuine benefit:

| Format | Speed impact (64K) | Memory vs f16 | Verdict |
|--------|-------------------|---------------|---------|
| f16 | Baseline | Baseline | ✅ Default |
| q8_0 | <5% | +~3% | ✅ Best for long context |
| q4_0 | -92.5% at 64K | +6% | ❌ Avoid |

---

## Test Setup

```
Model:    Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL.gguf
Hardware: NVIDIA DGX Spark GB10 (compute 12.1, 124,610 MiB VRAM)
OS:       DGX OS / Ubuntu aarch64
llama.cpp: build 8399 (commit 892e3c333), aarch64 + CUDA
CUDA:     13.0 | Driver: 580.126.09
Flags:    --ctx-size 131072, temperature 0.7, max_tokens 200
Protocol: Server restarted between each configuration
```

---

## Commands

```bash
# f16 — default, fastest
llama-server --model MODEL.gguf --ctx-size 131072 \
  --host 0.0.0.0 --port 8080

# q8_0 — recommended for long context
llama-server --model MODEL.gguf --ctx-size 131072 \
  --host 0.0.0.0 --port 8080 \
  --cache-type-k q8_0 --cache-type-v q8_0

# q4_0 — DO NOT USE for long context
# llama-server --model MODEL.gguf --ctx-size 131072 \
#   --host 0.0.0.0 --port 8080 \
#   --cache-type-k q4_0 --cache-type-v q4_0
```

---

## Why This Happens

**The cliff:** On discrete GPUs, the KV cache lives in VRAM and dequantization uses dedicated tensor cores — overhead scales slowly. On GB10 unified memory, all operations share the same memory bus. At 64K tokens, q4_0 dequantization reads dominate the bus, starving attention computation.

**The paradox:** The dequantization workspace + per-block metadata (scales, zero-points) consumes more bytes than the storage saved by int4 vs float16 at these context lengths. On unified memory there is no separate VRAM budget to optimize against.

---

## Raw Data

See [`data/`](data/) for full CSV results.

---

## Open Questions

- Do q4_1, iq4_nl, q5_0, q5_1 share q4_0's cliff behavior?
- Is the 64K cliff model-architecture-specific or general?
- Does NVFP4 KV cache (TensorRT-LLM path) avoid the cliff?
- What is the q8_0 cliff point (if any)?

Contributions and replications welcome.

---

## Citation

```bibtex
@techreport{maine2026kvcache,
  title  = {KV Cache Quantization on NVIDIA DGX Spark GB10: Three Novel Findings},
  author = {Maine, Nathan},
  institution = {Memoriant Inc.},
  year   = {2026},
  url    = {https://github.com/Memoriant/dgx-spark-kv-cache-benchmark}
}
```
