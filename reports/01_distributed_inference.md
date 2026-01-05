# Stage 1: Distributed Inference Architecture

## 1. Problem Statement
Large Language Models (LLMs) like Llama-3-70B require massive memory and compute bandwidth. A single GPU often cannot hold the entire model parameters or process tokens fast enough.
-   **Memory Constraint**: 70B parameters @ FP16 ~= 140GB. (A typical A100 has 80GB).
-   **Latency Constraint**: Generating tokens sequentially is memory-bound.

## 2. Solution: Tensor Parallelism (TP)
We implemented **Tensor Parallelism**, where individual weight matrices are split across multiple GPUs. This reduces memory per GPU and increases aggregate memory bandwidth.

### 2.1 Architecture Design
We introduced a **Communication Layer** to handle synchronization.
-   **Component**: `NcclHandler` (Singleton)
-   **Library**: NVIDIA NCCL (Optimized for NVLink/PCIe)

### 2.2 Implementation Details
We modified the `MatmulLayer` to support distributed execution:
1.  **Row Parallelism**: Used for the output projection. We split the weight matrix row-wise. Each GPU computes a partial sum.
2.  **All-Reduce**: We inject an `AllReduce` sum operation after the computation to aggregate the partial sums into the final result on every GPU.

```cpp
// Pseudocode from source/op/matmul.cpp
void MatmulLayer::forward() {
    // 1. Compute local part
    kernel::matmul(input, local_weight, local_output);

    // 2. Synchronize across GPUs
    if (tp_size > 1) {
        nccl_handler->AllReduce(local_output, ...);
    }
}
```

## 3. Computational-Communication Overlap
To hide the latency of `AllReduce` (which involves moving data over physical wires), we implemented **Multi-stream Scheduling**:
1.  **Compute Stream**: Launches the Matrix Multiplication.
2.  **Comm Stream**: Waits for Compute to finish, then launches `AllReduce`.
3.  **Synchronization**: The CPU proceeds to the next layer immediately, queueing work. The GPU executes Comm and Next-Compute in parallel (if resources allow) or pipelined without CPU gaps.

## 4. Results (Expected)
-   **Memory**: Linear reduction with number of GPUs (N).
-   **Throughput**: Higher aggregate bandwidth.
-   **Latency**: Reduced due to parallel compute, though exact speedup depends on communication overhead (NVLink vs PCIe).
