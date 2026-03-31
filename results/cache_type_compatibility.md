# Cache Type Compatibility on DGX Spark GB10

## Supported Types (from --help, llama.cpp build 8399, aarch64+CUDA)

```
--cache-type-k: f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1
--cache-type-v: f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1
Default: f16 for both K and V
```

## Test Results

| Format | Loads | Long-ctx safe | Memory benefit | Recommended |
|--------|-------|--------------|----------------|-------------|
| f16 | ✅ | ✅ | Baseline | ✅ Default |
| q8_0 | ✅ | ✅ | None (paradox) | ✅ Long context |
| q4_0 | ✅ | ❌ cliff at 64K | Negative (+6%) | ❌ Avoid |
| q4_1 | untested | unknown | unknown | — |
| iq4_nl | untested | unknown | unknown | — |
| q5_0 | untested | unknown | unknown | — |
| q5_1 | untested | unknown | unknown | — |
| bf16 | untested | unknown | unknown | — |
| f32 | untested | unknown | unknown | — |
