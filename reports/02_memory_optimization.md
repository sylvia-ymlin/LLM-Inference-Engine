# Stage 2: Memory Optimization (KV Cache)

## 1. Problem Statement
In autoregressive decoding (generating one token at a time), the Key and Value matrices for all previous tokens must be stored to attend to them.
-   **Fragmentation**: Naive implementations allocate small tensors for each new token or layer. This leads to massive fragmentation and overhead from `cudaMalloc`.
-   **Latency**: The overhead of finding free memory blocks during the critical generation loop adds significant jitter.

## 2. Solution: Pre-allocated KV Cache Manager
We implemented a **Block Manager** approach (`KVCacheManager`) to completely eliminate dynamic allocation during inference.

### 2.1 Architecture Design
-   **Arena Allocation**: We allocate **one giant contiguous block** of GPU memory at model initialization.
    -   `Size = Layers * MaxBatch * Heads * MaxSeqLen * HeadDim * SizeOf(Type)`
-   **Virtual Addressing**: We provide pointers (views) into this block to the attention layers.

### 2.2 Implementation Details
We modified `init_mem` in `llama3.cpp`:
1.  Calculate total required cache size based on `max_seq_len` (e.g., 4096).
2.  Allocate `raw_buffer_` via `KVCacheManager`.
3.  Map `Tensor` objects to this buffer using offset arithmetic.

```cpp
// Logic in KVCacheManager::get_cache_ptr
size_t layer_offset = layer_idx * size_per_layer;
return ptr + layer_offset;
```

## 3. Impact
-   **Performance**: Zero `cudaMalloc` calls during `forward()`.
-   **Stability**: No Out-Of-Memory (OOM) crashes due to fragmentation.
-   **Predictability**: Memory usage is constant and known at startup.
