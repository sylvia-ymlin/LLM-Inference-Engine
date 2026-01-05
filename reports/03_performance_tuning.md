# Stage 3: Advanced Performance Tuning

## 1. Computation-Communication Overlap
In distributed settings, communication (NCCL AllReduce) is the bottleneck. The GPU computes faster than it can send data over PCIe/NVLink.

### 1.1 The Solution: Multi-stream Pipeline
We mask the communication cost by running it concurrently with other operations.
-   **Streams**: CUDA streams allow independent queues of commands.
    -   `Stream 1 (Compute)`: Executes `Matmul`.
    -   `Stream 2 (Comm)`: Executes `AllReduce`.

### 1.2 Implementation Logic
We used **CUDA Events** to manage dependencies without blocking the CPU.
```cpp
// 1. Record that Compute is done
cudaEventRecord(compute_done, stream_compute);

// 2. Tell Comm stream to wait for Compute (on GPU side only)
cudaStreamWaitEvent(stream_comm, compute_done);

// 3. Launch Comm (Output tensor needed for Reduce)
ncclAllReduce(..., stream_comm);

// 4. Record Comm is done
cudaEventRecord(comm_done, stream_comm);

// 5. Tell Compute stream to wait for Comm (before using the result)
cudaStreamWaitEvent(stream_compute, comm_done);
```
**Result**: The CPU issues all these commands instantly. The GPU hardware scheduler executes them, overlapping the data transfer with any other available independent work (e.g., subsequent layer prep).

## 2. Low-Precision Arithmetic (INT8)
We verify and utilize a W8A16 (Weight INT8, Activation FP16/FP32) scheme.

### 2.1 Group-wise Quantization
Instead of one scale factor for the whole matrix (which loses too much accuracy for outliers), we use **Group-wise** quantization.
-   **Group Size**: Every 128 weights share a scale factor.
-   **Kernel**: `matmul_kernel_cu_int8` loads 128 `int8` weights, 1 `float` scale, and performs the dot product.

### 2.2 Memory Savings
-   FP16 Weight: 2 bytes
-   INT8 Weight: 1 byte
-   **Reduction**: 50% reduction in Model Size.
-   **Bandwidth**: 2x effective memory bandwidth for weight loading.
