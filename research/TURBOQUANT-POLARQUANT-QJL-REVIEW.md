# Research Review: TurboQuant, PolarQuant, and QJL

**Author:** Nathan Maine, [Memoriant Inc.](https://www.memoriant.ai)
**Date:** March 2026
**Purpose:** Literature review of the three Google Research papers that form the theoretical foundation for TurboQuant KV cache compression, with analysis of implications for unified memory architectures like DGX Spark GB10.

---

## Table of Contents

1. [TurboQuant](#turboquant)
2. [PolarQuant](#polarquant)
3. [QJL: 1-Bit Quantized JL Transform](#qjl)
4. [Cross-Platform Analysis: Unified Memory vs Discrete GPU](#cross-platform-analysis)
5. [Key Takeaways and Recommendations](#key-takeaways--recommendations)
6. [References](#references)

---

## 1. TurboQuant

**Full title:** "Online Vector Quantization with Near-optimal Distortion Rate"
**Authors:** Amir Zandieh (Google Research), Majid Daliri (NYU), Majid Hadian (Google DeepMind), Vahab Mirrokni (Google Research)
**Venue:** ICLR 2026
**arxiv:** [2504.19874](https://arxiv.org/abs/2504.19874)
**Google Blog:** [TurboQuant: Redefining AI Efficiency](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)

### 1.1 Problem Definition

* **Goal**: Quantize a *d-dimensional* vector **x** to a binary string of **B = b*d** bits (b bits per coordinate) with an invertible de-quantization map.
* **Distortion measures**:
  - **MSE**: Expected squared L2 reconstruction error
  - **Inner-product error**: Expected squared error of inner products (critical for attention scores)
* **Requirement**: *Unbiased* inner-product estimator for downstream KV cache tasks.

### 1.2 Core Algorithm

| Stage | What is done | Why it works |
|------|--------------|--------------|
| **Random rotation** | Multiply input by an orthogonal matrix. Coordinates become i.i.d. Beta, approximated by Gaussian N(0,1/d). | In high-dim, coordinates concentrate and become almost independent, enabling per-coordinate scalar quantizers. |
| **Scalar Lloyd-Max quantizer** | Solve continuous 1-D k-means for each coordinate, yielding optimal centroids. | Gives *optimal* MSE for the chosen bit-width. |
| **De-quantization** | Retrieve centroids, rotate back with the transpose. | Exact inverse of the quantizer (up to reconstruction error). |
| **Two-stage inner-product path** | (1) MSE-optimal quantizer with **b-1** bits. (2) Apply **QJL** (1-bit sign quantizer) on the residual. | MSE quantizer alone is biased for inner products; adding QJL removes the bias while preserving near-optimal distortion. |

### 1.3 Theoretical Guarantees

* **MSE bound** (Theorem 1): Distortion decays as 1/4^b. For b = 1..4: 0.36, 0.117, 0.03, 0.009.
* **Inner-product bound** (Theorem 2): Scales with ||y||^2 * d * 1/4^b.
* **Lower bound** (Theorem 3): Shannon + Yao shows any quantizer must incur at least 1/4^b distortion. TurboQuant is within a factor of ~2.7 of optimal (~1.45 for b=1).

### 1.4 Experimental Results

* **KV cache quantization**: Quality-neutral retrieval at **3.5 bits/channel**; modest degradation at **2.5 bits**.
* **Nearest-neighbor search**: Outperforms classic Product Quantization (PQ) in recall while indexing time approaches zero.
* **H100 benchmark**: Up to **8x speedup** in attention logit computation vs 32-bit.
* Demonstrated on DBpedia, GloVe, and LongBench (needle-in-haystack) tasks.
* Data-oblivious: no calibration, no codebook, no training required.

---

## 2. PolarQuant

**Full title:** "Quantizing KV Caches with Polar Transformation"
**Authors:** Insu Han, Piotr Kacham, Amin Karbasi, Vahab Mirrokni, Amir Zandieh
**Venue:** AISTATS 2026
**arxiv:** [2502.02617](https://arxiv.org/abs/2502.02617)

> **Note:** There is a separate, unrelated paper also named "PolarQuant" by Wu et al. (NeurIPS 2025, [2502.00527](https://arxiv.org/abs/2502.00527)). The review below covers the Zandieh et al. paper which feeds into TurboQuant.

### 2.1 Motivation

KV cache compression suffers from *memory overhead* caused by per-block zero-point and scale storage. Goal: **zero-overhead** quantization with high fidelity for long-context LLMs.

### 2.2 Polar Transformation

* Recursive polar mapping converts a Cartesian vector into a radius and a stack of angle vectors (one per level in a log2(d) hierarchy).
* Groups coordinates in powers of two, turning each pair into 2-D radius/angle, then feeding radii forward.

### 2.3 Angle Distribution and Preconditioning

* After random preconditioning (Gaussian sketch), angles concentrate around pi/4 and become highly predictable.
* The angle PDF can be derived analytically; concentration and variance = O(1/sqrt(d)).

### 2.4 Quantization and Codebooks

* Angles quantized independently using optimal Lloyd-Max partitions for each level.
* Because angles are concentrated, tiny codebooks suffice: <= 4 bits per coordinate overall.
* No per-block scale needed: ~3.875 bits per original coordinate when using 16-bit FP as base.

### 2.5 Experimental Results

* **Compression**: >4.2x reduction in KV cache size.
* **Long-context tasks** (needle-in-haystack, LongBench): Identical recall at 4x compression; marginal loss at 2.5x.
* **Runtime**: 14% faster token generation vs KIVI; prefill time reduced with offline codebooks.
* Uses CUDA kernels for polar-to-Cartesian conversion and angle-lookup.

---

## 3. QJL: 1-Bit Quantized JL Transform

**Full title:** "1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead"
**Authors:** Amir Zandieh (Google Research)
**Venue:** AAAI 2025
**arxiv:** [2406.03482](https://arxiv.org/abs/2406.03482)
**Code:** [github.com/amirzandieh/QJL](https://github.com/amirzandieh/QJL)

### 3.1 Background and Unbiased Estimator

* Classic Johnson-Lindenstrauss (JL) sketch: multiply by a Gaussian matrix **S**, inner products are preserved unbiased with low distortion.
* **QJL** = sign(S*k) (1-bit output) + an asymmetric estimator that keeps the query vector unquantized.
* The estimator is **exactly unbiased**: E[Prod_QJL(q,k)] = <q,k>.

### 3.2 Practical Implementation

* **Key cache quantization**: Generate sketch S once. Store sign(S*k) and ||k||_2 (norm).
* **Score estimation**: s_i = (pi/2m) * ||k_i||_2 * <Sq, sign(Sk_i)>.
* CUDA kernels accelerate both sketching and inner-product aggregation.

### 3.3 Experimental Results

* KV cache stored in **3 bits per coordinate**: >5x compression, no accuracy loss on LongBench.
* Up to 14% faster token generation vs KIVI.
* Works on Llama-3-8B (BF16) and Llama-2-7B, enabling 3-bit quantization across all layers.

---

## 4. Cross-Platform Analysis: Unified Memory vs Discrete GPU

This section connects the theoretical papers above with our empirical DGX Spark GB10 benchmark results and contextualizes them against discrete GPU architectures.

### 4.1 The Bandwidth Equation

KV cache quantization delivers value when **memory bandwidth savings > dequantization compute cost**. The balance depends on hardware:

| Platform | Memory Type | Bandwidth | KV Quant Benefit |
|----------|-----------|-----------|-----------------|
| **H100 SXM** | HBM3 (80GB) | ~3,350 GB/s | **High** — VRAM-constrained, bandwidth-rich |
| **RTX 5090** | GDDR7 (32GB) | ~1,700 GB/s | **High** — severely VRAM-constrained |
| **RTX 4090** | GDDR6X (24GB) | ~1,008 GB/s | **High** — VRAM-constrained |
| **DGX Spark GB10** | Unified LPDDR5X (128GB) | ~273 GB/s | **Low** — memory-abundant, bandwidth-limited |

On discrete GPUs, KV cache quantization reduces memory traffic over the high-bandwidth HBM/GDDR bus, which directly translates to faster attention. On DGX Spark, the 128GB unified pool means memory capacity is rarely the bottleneck, and the lower LPDDR5X bandwidth means dequantization compute dominates.

### 4.2 Our Empirical Evidence

**Standard quantization (q4_0/q8_0) on DGX Spark:**
- Memory savings work as expected (q4_0 saves 72% of KV buffer)
- Generation throughput degrades 37% at 110K context due to dequantization overhead
- Prompt processing is unaffected (dequant cost amortized across batch)

**TurboQuant (turbo3/turbo4) on DGX Spark:**
- Consistently slower than f16 at all context depths
- Up to 23.6% slower at 32K context
- Prompt processing barely affected (<1%)

**TurboQuant on discrete GPUs (community results):**
- RTX 5090: Significant speedups reported at long context (where VRAM savings matter)
- H100: Up to 8x speedup in attention logit computation (Google's paper results)

### 4.3 Why TurboQuant's Theoretical Advantages Don't Transfer to Spark

TurboQuant achieves near-optimal distortion at extreme bit-widths (3-4 bits). On H100/RTX 5090, this translates to:

1. **Less memory traffic** through the high-bandwidth bus (direct speedup)
2. **Longer effective context** within fixed VRAM (enabling workloads that wouldn't fit otherwise)
3. **Unbiased attention scores** (quality preservation at extreme compression)

On DGX Spark, advantage (1) reverses — the dequantization ALU work exceeds the bandwidth savings on LPDDR5X. Advantage (2) is moot with 128GB unified memory. Only advantage (3) remains relevant, and only if you actually need the memory savings for fitting a larger model.

### 4.4 The NVFP4 Alternative Path

For DGX Spark specifically, the recommended quantization path for KV cache is NVIDIA's hardware-accelerated NVFP4 format:

- Dedicated silicon pathways on Blackwell tensor cores
- Two-level scaling (E4M3 fine-grained + FP32 block scalar) retains dynamic range
- Hardware dequantization eliminates the software overhead that penalizes TurboQuant
- Available via TensorRT-LLM and NVIDIA Model Optimizer

The optimal DGX Spark configuration for maximum context is:
```
NVFP4 model weights (shrinks 70B model to ~37GB)
  + NVFP4 KV cache (hardware-accelerated, no software dequant penalty)
  = 70B model + very long context on single 128GB Spark
```

### 4.5 When to Use What

| Scenario | Recommendation |
|----------|---------------|
| Discrete GPU (RTX/H100), VRAM-constrained | TurboQuant — significant speedups and memory savings |
| DGX Spark, general inference | f16 KV cache — no overhead, 128GB is sufficient |
| DGX Spark, need to fit larger model | NVFP4 via TensorRT-LLM — hardware-accelerated |
| DGX Spark, many concurrent users | q8_0 — modest savings compound across sessions |
| Any platform, extreme compression research | TurboQuant — near-optimal distortion at 3-bit |

---

## 5. Key Takeaways and Recommendations

| Aspect | Recommendation |
|--------|----------------|
| **Bit-width selection** | Use 4 bits per coordinate for a good tradeoff (~3.875 bits after PolarQuant). For ultra-low-bit budgets (<2 bits) prefer TurboQuant-Prod + QJL for unbiased inner-product with minimal distortion. |
| **Pre-computation** | Compute centroids and rotation matrices offline; broadcast once. Eliminates per-query overhead. |
| **Outlier handling** | Split vectors into inlier and outlier sub-spaces (PolarQuant approach); allocate extra bits to outliers. |
| **Hardware matching** | TurboQuant excels on bandwidth-rich, VRAM-constrained GPUs. Use f16 or hardware-accelerated formats on memory-abundant unified architectures. |
| **Error control** | Theoretical bounds guarantee distortion <= (sqrt(3)*pi/2) * 2^(-b). Choose b to match application-specific tolerance. |
| **K/V asymmetry** | Keys carry positional (RoPE) structure requiring higher precision. Values are more uniform and tolerate more aggressive quantization. Consider asymmetric configs (e.g., K at 4-bit, V at 3-bit). |

---

## 6. References

1. **TurboQuant** — A. Zandieh, M. Daliri, M. Hadian, V. Mirrokni, "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate," Google Research & NYU, ICLR 2026. [arxiv:2504.19874](https://arxiv.org/abs/2504.19874)
2. **PolarQuant** — I. Han, P. Kacham, A. Karbasi, V. Mirrokni, A. Zandieh, "Quantizing KV Caches with Polar Transformation," AISTATS 2026. [arxiv:2502.02617](https://arxiv.org/abs/2502.02617)
3. **QJL** — A. Zandieh, M. Daliri, "1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead," AAAI 2025. [arxiv:2406.03482](https://arxiv.org/abs/2406.03482)
4. **KIVI** — Z. Liu et al., "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache," 2024. [arxiv:2402.02750](https://arxiv.org/abs/2402.02750)
5. **Google TurboQuant Blog** — [research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
6. **TurboQuant llama.cpp Discussion** — [github.com/ggml-org/llama.cpp/discussions/20969](https://github.com/ggml-org/llama.cpp/discussions/20969)

---

## Implementation Status Tracker

Active community implementations of TurboQuant (as of April 2026):

| Implementation | Platform | Status | Notes |
|---------------|----------|--------|-------|
| [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) | Metal (Apple) | Working | Original llama.cpp implementation |
| [Madreag/turbo3-cuda](https://github.com/Madreag/turbo3-cuda) | CUDA | Working | Used for our SM 121 benchmarks |
| [spiritbuun/llama-cpp-turboquant-cuda](https://github.com/spiritbuun/llama-cpp-turboquant-cuda) | CUDA | Working | Flash Attention + FWHT |
| [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp/issues/1509) | CPU | Working | 18/18 tests, MSE matches paper |
| [vLLM PR #38280](https://github.com/vllm-project/vllm/pull/38280) | Triton | Phase 1 merged | Paged KV cache support |
| llama.cpp upstream [PR #21089](https://github.com/ggml-org/llama.cpp/pull/21089) | CPU | Open PR | TBQ3_0/TBQ4_0 ggml types |
| Google official CUDA | CUDA | Q2 2026 | Announced at ICLR 2026 |
