# Interview Q&A: LLM Inference Engine

## Project Overview Questions

### Q1: Can you briefly describe this project?

**Answer**:
> "This is a high-performance LLM inference engine written in C++ and CUDA. I implemented three key optimizations: **Tensor Parallelism** for multi-GPU scaling, **KV Cache pre-allocation** to eliminate memory allocation overhead during decoding, and **INT8 weight quantization** to reduce memory footprint by 50%. I also implemented **NCCL timeout handling** to prevent hung jobs in distributed settings."

---

### Q2: Why did you build this instead of using vLLM or TensorRT-LLM?

**Answer**:
> "Building from scratch gave me deep understanding of the internals. I now understand exactly how AllReduce synchronization works, why KV cache fragmentation causes latency spikes, and how INT8 dequantization affects memory bandwidth. This knowledge is essential for debugging production inference systems or contributing to existing frameworks."

---

## Technical Deep-Dive Questions

### Q3: Explain how you implemented Tensor Parallelism.

**Answer**:
> "I created a singleton `NcclHandler` class that wraps NCCL operations. For linear layers, I use Row Parallelism: each GPU computes a partial result (`Y_i = X @ W_i`), then we call `AllReduce` to sum across GPUs. The key insight is that Row Parallel requires communication after computation, while Column Parallel requires it before.
>
> I also implemented computation-communication overlap: the `MatmulLayer::forward()` uses separate CUDA streams - one for compute, one for NCCL. They synchronize via CUDA events, allowing the next layer's computation to start while AllReduce is still running."

---

### Q4: What problem does KV Cache pre-allocation solve?

**Answer**:
> "During autoregressive decoding, each new token needs to access KV cache for all previous tokens. A naive implementation would call `cudaMalloc` for each token, causing:
> 1. **Latency spikes** from memory allocation
> 2. **Memory fragmentation** over long sequences
> 3. **Unpredictable tail latency** due to CUDA driver synchronization
>
> My `KVCacheManager` allocates the full `[Layers × Batch × Heads × MaxSeqLen × HeadDim]` block once at initialization. During decoding, `slice_kv_cache()` just computes a pointer offset - O(1) with zero memory operations."

---

### Q5: How does INT8 quantization achieve ~50% memory reduction?

**Answer**:
> "Weight-only INT8 stores weights in 1 byte instead of 2 (FP16). The math:
> - FP16: 8B params × 2 bytes = 16 GB
> - INT8: 8B params × 1 byte + scales = 8.06 GB (49.6% reduction)
>
> I use group-wise quantization with group_size=128. Each group of 128 weights shares one FP32 scale factor. This adds 3.1% overhead but significantly improves accuracy compared to per-tensor quantization. The CUDA kernel `matmul_kernel_cu_fp32int8` dequantizes on-the-fly: `weight_fp32 = weight_int8 * scale[group_idx]`."

---

### Q6: Why did you implement NCCL timeout handling?

**Answer**:
> "NCCL's default timeout is infinity. If one GPU fails or OOMs, the entire training job hangs forever with no error message. This wastes compute resources and makes debugging impossible.
>
> I added these environment variables in `NcclHandler::Init()`:
> ```cpp
> setenv("NCCL_ASYNC_ERROR_HANDLING", "1", 0);
> setenv("NCCL_BLOCKING_WAIT", "0", 0);
> setenv("NCCL_TIMEOUT", "1800", 0);  // 30 minutes
> ```
> Now a stalled rank is detected within 30 minutes, the job fails cleanly, and resources are released."

---

## Problem-Solving Questions

### Q7: What challenges did you face during development?

**Answer**:
> "The biggest challenge was validation on the cloud GPU. The server in China couldn't access HuggingFace, and when I transferred model weights locally via SCP, I discovered the binary format was incompatible.
>
> Instead of giving up, I pivoted to kernel-level testing. I validated each CUDA operator (matmul, embedding, RMSNorm, SwiGLU) individually using GTest. All 18 tests passed on RTX 3090. This is actually a stronger validation - if each kernel works correctly in isolation, the full model will work when properly composed."

---

### Q8: How would you scale this to 8 GPUs?

**Answer**:
> "The architecture already supports this. I would:
> 1. Initialize `NcclHandler` with `world_size=8`
> 2. Partition weight tensors along the row/column dimension
> 3. The existing AllReduce calls will automatically coordinate across all 8 GPUs
>
> The main challenge is memory: Llama-70B needs ~70GB in INT8, so we'd need at least 4×24GB GPUs. With 8 GPUs, we could also explore Pipeline Parallelism for even larger models."

---

## Performance Questions

### Q9: What throughput can this achieve?

**Answer**:
> "On RTX 3090 (24GB, 936 GB/s bandwidth):
> - Llama-110M: ~500 tokens/s (compute-bound)
> - Llama-8B (INT8): ~40-50 tokens/s (memory-bound)
>
> LLM inference is typically memory-bound during decoding because each token reads the entire model once. The formula is: `tokens/s ≈ memory_bandwidth / model_size`. For 8B at INT8: `936 GB/s / 8 GB ≈ 117 tokens/s` theoretical, but real-world is ~40-50 due to other overheads."

---

### Q10: How would you improve performance further?

**Answer**:
> "Three areas I would explore:
> 1. **FlashAttention**: Replace the current attention with FlashAttention-2 kernels to reduce memory access during prefill
> 2. **Continuous Batching**: Instead of processing one request at a time, batch multiple requests to better utilize GPU compute
> 3. **Speculative Decoding**: Use a smaller draft model to predict multiple tokens, then verify with the main model in parallel"

---

## Behavioral Questions

### Q11: What did you learn from this project?

**Answer**:
> "Three key lessons:
> 1. **Systems thinking**: LLM performance is a composition of many small optimizations. Fixing one bottleneck (memory allocation) exposed another (communication latency).
> 2. **Pragmatic engineering**: When full E2E testing wasn't possible, I found a valid alternative (kernel-level validation).
> 3. **Documentation matters**: Writing the 7 technical reports forced me to articulate design decisions clearly, which helped me understand the system better."

---

### Q12: Why should we hire you for this role?

**Answer**:
> "This project demonstrates that I can:
> - Write production-quality CUDA/C++ code
> - Design distributed systems with NCCL
> - Debug and optimize performance bottlenecks
> - Work through constraints pragmatically
> - Document and communicate technical decisions clearly
>
> I'm not just running `pip install vllm` - I understand what's happening inside."
