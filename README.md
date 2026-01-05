# High-Performance LLM Inference Engine

A highly optimized, distributed Llama3/Qwen2 inference engine implemented in C++ and CUDA. 
This project demonstrates advanced techniques in **Distributed Computing (Tensor Parallelism)**, **Memory Management (Paged KV Cache)**, and **Low-Precision Arithmetic (INT8 Quantization)** to achieve state-of-the-art performance.

## Key Features

### 1. Distributed Inference (Tensor Parallelism)
- **Scalable Architecture**: Implemented custom `NcclHandler` to manage NCCL communicators for multi-GPU inference.
- **Synchronized Design**: Integrated `AllReduce` and `AllGather` collectives into the `MatmulLayer` to shard linear operations across multiple GPUs.
- **Computation-Communication Overlap**: Engineered a multi-stream scheduling mechanism that hides communication latency by overlapping `AllReduce` operations with independent computations, ensuring maximum GPU utilization.

### 2. Efficient Memory Management
- **Pre-allocated KV Cache**: Developed `KVCacheManager` to pre-allocate contiguous memory blocks for Key and Value caches at model initialization.
- **Zero Dynamic Allocation**: Eliminated runtime memory allocation overhead during autoregressive decoding, significantly reducing latency jitter/tail latency.
- **Optimized Layout**: Aligned cache memory layout for coalesced global memory access in CUDA kernels.

### 3. Weight-Only INT8 Quantization
- **Memory Efficiency**: Implemented Group-wise Weight-Only INT8 Quantization (W8A16), reducing model memory footprint by ~50%.
- **Custom Kernels**: Utilized optimized CUDA kernels (`matmul_kernel_cu_fp32int8`) that perform on-the-fly dequantization, maintaining high compute throughput while minimizing memory bandwidth usage.

## Architecture Highlights

- **Languages**: C++17, CUDA C++
- **Dependencies**: NCCL, GLOG, GTest, CPM
- **Model Support**: Llama 3, Qwen 2, Llama 2 (and other RoPE-based architectures)

## Implementation Details

### Tensor Parallelism
The linear layers (Wq, Wk, Wv, Wo, Gate, Up, Down) are sharded across GPUs.
- **Row Parallelism**: Used for the second half of MLPs and Output projections. Requires an `AllReduce` sum reduction after computation.
- **Column Parallelism**: Used for Attention projections and first half of MLPs. Typically requires `AllGather` (or `AllReduce` if followed by Row Parallel) but allows independent computation.

### Paged Attention & KV Cache
The `KVCacheManager` treats the GPU memory as a pool of blocks. 
- **Allocation**: `allocate()` reserves the full `[Layers, Batch, Heads, SeqLen, Dim]` block upfront.
- **Access**: `slice_kv_cache()` computes offset pointers instantly using pointer arithmetic, avoiding costly `malloc`/`cudaMalloc` calls during the time-critical decoding loop.

## Usage

### Prerequisites
- NVIDIA GPU(s) with drivers installed
- CUDA Toolkit (11.x or 12.x)
- NCCL (for distributed inference)

### Building
```bash
mkdir -p build && cd build
cmake .. -DUSE_NCCL=ON
make -j$(nproc)
```

### Running Documentation Coverage
```bash
# Verify distributed components (Requires MPI)
mpirun -n 2 ./test_llm --gtest_filter=TestNCCL.*

# Run Inference Demo
./demo/llama_inference --model_path /path/to/weights --tokenizer_path /path/to/tokenizer
```

## Future Work
- **FlashAttention Integration**: Integrate FlashAttention-2 kernels to further optimize the Attention phase.
- **Continuous Batching**: Implement continuous batching scheduler for higher throughput in serving scenarios.
