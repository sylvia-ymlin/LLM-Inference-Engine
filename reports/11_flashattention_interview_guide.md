# FlashAttention Interview Guide

> **Core Concept**: For interviews and project showcases, understanding the *principles* (Why is it fast? What bottlenecks does it solve?) is more valuable than achieving vendor-level optimization performance.

## 1. Project Context: "Why is your implementation slower than cuDNN?"

**Your Answer**:
"My current implementation is a **Correctness Baseline** (FlashAttention-1 style naive implementation). It proves the concept of Tiling and Recomputation to reduce memory from $O(N^2)$ to $O(N)$. However, it lacks the hardware-specific optimizations found in official libraries, such as FP16 Tensor Core instructions, software pipelining, and aggressive loop unrolling. My goal was to structurally integrate the algorithm, not to compete with NVIDIA's 10-year optimized kernels."

---

## 2. FlashAttention Versions: Core Differences

Interviewers often ask: *"What are the key improvements in FlashAttention 2 and 3?"*

### Flash Attention 1 (Standard)
*   **Core Idea**: **Tiling (分块) + Recomputation (重计算)**.
*   **The Problem Solved**:
    *   Standard Attention computes a huge $N \times N$ matrix (Scores) to do Softmax, which creates a memory bottleneck ($O(N^2)$ writes/reads).
    *   FA1 computes Attention in small blocks (Tiles), computes Softmax on-the-fly, and writes only the final output.
    *   **Result**: Memory complexity drops to $O(N)$ (linear), enabling much longer sequences.

### Flash Attention 2 (Parallelism)
*   **Core Improvement**: **Better Parallelism & Work Partitioning**.
*   **Key Changes**:
    1.  **Sequence Parallelism**: FA1 only parallelized across Batch and Heads. FA2 adds parallelization across the **Sequence Length (Q dimension)**. This is crucial for long-context training where $Q$ is very long and becomes a single-thread bottleneck.
    2.  **Work Distribution**: Optimized the inner loops to minimize non-matmul operations (like Softmax statistics) ensuring the GPU spends most of its time on matrix multiplication.
    3.  **Tensor Cores**: Explicitly designed to keep Tensor Cores saturated.

### Flash Attention 3 (Hopper/FP8)
*   **Core Improvement**: **Hardware-Specific Optimization (H100 Hopper)**.
*   **Key Changes**:
    1.  **Async Data Movement (TMA)**: Uses the H100's "Tensor Memory Accelerator" to copy data from Global Memory to Shared Memory *asychronously*.
    2.  **Software Pipelining**: Hardware allows "Warp Overlapping", meaning the GPU can calculate the current block *while* the next block is loading, effectively hiding memory latency completely.
    3.  **FP8 Support**: Native support for 8-bit floating point math for extreme speedups.
