# Correction Notice — KV Cache Benchmark v3

**Date:** 2026-04-01
**Original publication:** 2026-03-31
**Status:** Correcting methodology and findings

---

## What Changed

### Finding 2 (Memory "Paradox") — RETRACTED

**Original claim:** "q4_0 uses MORE memory than f16 on unified memory (+6%)"
**Methodology used:** Process RSS via `ps`
**Problem:** RSS measures CPU-side process memory, not GPU/unified memory allocations. On GB10 unified memory, KV cache allocations are not visible in RSS.

**Corrected measurement (nvidia-smi + llama.cpp internals):**

| Cache Type | KV Buffer (llama.cpp) | Total GPU (nvidia-smi) | vs f16 |
|-----------|----------------------|----------------------|--------|
| f16 | 768 MiB | 23,092 MiB | baseline |
| q8_0 | 408 MiB | 22,732 MiB | **-360 MiB (saves 47%)** |
| q4_0 | 216 MiB | 22,540 MiB | **-552 MiB (saves 72%)** |

**Conclusion:** KV cache quantization DOES save memory on GB10, as expected. The "paradox" was a measurement error.

### Finding 1 (Speed Cliff) — RETRACTED

**Original claim:** "92.5% prompt throughput collapse at 64K with q4_0"
**Corrected finding:** No prompt throughput cliff exists. q4_0 prompt processing is essentially identical to f16 at all context lengths including 110K tokens.

**What actually happens:** Generation (decode) throughput degrades ~37% at 110K context with q4_0 vs f16. This is a real effect but it is generation speed, not prompt processing, and it is 37% not 92.5%.

**Corrected throughput data (full range):**

| Context | f16 prompt | q8_0 prompt | q4_0 prompt | f16 gen | q8_0 gen | q4_0 gen |
|---------|-----------|-------------|-------------|---------|---------|---------|
| ~1.5K | 923 | 925 | 926 | 45.2 | 45.3 | 45.6 |
| ~6K | 1,211 | 1,207 | 1,206 | 44.7 | 44.9 | 45.0 |
| ~12K | 1,188 | 1,184 | 1,191 | 44.9 | 42.9 | 42.7 |
| ~24K | 1,153 | 1,149 | 1,152 | 44.6 | 39.7 | 39.3 |
| **~110K** | **815** | **810** | **813** | **38.0** | **25.0** | **24.0** |

The original 92.5% collapse was likely caused by failed completion requests returning error data, not actual throughput measurements.

### Finding 3 (q8_0 Sweet Spot) — PARTIALLY VALID

q8_0 does offer genuine KV buffer savings (47%) with minimal throughput impact at short-to-medium context. The generation speed degradation at 24K (~10%) needs further investigation at longer contexts.

---

## Root Cause of Original Errors

1. **RSS is the wrong metric** for KV cache on unified memory. RSS reflects CPU-side process allocations, not GPU-side unified memory usage.
2. **nvidia-smi + llama.cpp verbose output** provide the correct measurements.
3. **The original benchmark did not verify that completions actually succeeded** — some data points may have been from failed requests.

## Credit

Community feedback from u/audioen on r/LocalLLaMA identified the RSS measurement flaw. Their critique was correct.
