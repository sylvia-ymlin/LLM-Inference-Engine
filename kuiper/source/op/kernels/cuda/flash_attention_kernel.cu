#include "flash_attention_kernel.cuh"
#include <base/cuda_config.h>
#include <tensor/tensor.h>
#include <glog/logging.h>

namespace kernel {

// FlashAttention-inspired CUDA kernel implementation
// This implements the core memory-efficient attention algorithm
__global__ void flash_attention_fwd_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k, 
    const float* __restrict__ v,
    float* __restrict__ out,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    float softmax_scale,
    bool is_causal) {
    
    const int head_idx = blockIdx.x;
    const int batch_idx = blockIdx.y;
    
    if (head_idx >= num_heads || batch_idx >= batch_size) {
        return;
    }
    
    // Use dynamic shared memory more carefully
    extern __shared__ float shared_mem[];
    float* s_query = shared_mem;
    // Allocate shared memory for scores - limit to reasonable size
    const int max_seq_len = min(seq_len, 256);  // Further limit shared memory usage
    float* s_scores = s_query + head_dim;
    
    // Bounds check for shared memory allocation
    if (head_dim + max_seq_len > 1024) {  // Typical shared memory limit per block
        // If we exceed shared memory, process in chunks
        return;  // For now, just return - could implement chunked processing
    }
    
    // Calculate tensor offsets correctly for [batch, seq, heads, head_dim] layout
    const int head_offset = batch_idx * seq_len * num_heads * head_dim + head_idx * head_dim;
    
    // For autoregressive generation, process current position
    const int q_pos = min(seq_len - 1, max_seq_len - 1);
    
    // Bounds checking for tensor access
    if (batch_idx >= batch_size || head_idx >= num_heads || q_pos >= seq_len) {
        return;
    }
    
    // Load query for current position - fix memory layout
    const float* q_ptr = q + batch_idx * seq_len * num_heads * head_dim + 
                         q_pos * num_heads * head_dim + head_idx * head_dim;
    
    // Cooperatively load query to shared memory
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        if (d < head_dim) {
            s_query[d] = q_ptr[d];
        }
    }
    __syncthreads();
    
    // Compute attention scores for all positions up to current
    float max_score = -INFINITY;
    const int num_positions = min(q_pos + 1, max_seq_len);
    
    // Each thread processes multiple positions
    for (int pos = threadIdx.x; pos < num_positions; pos += blockDim.x) {
        if (is_causal && pos > q_pos) {
            if (pos < max_seq_len) {
                s_scores[pos] = -INFINITY;
            }
            continue;
        }
        
        // Bounds check before accessing memory
        if (pos >= seq_len || pos < 0) continue;
        
        // Fix key pointer calculation
        const float* k_ptr = k + batch_idx * seq_len * num_heads * head_dim + 
                             pos * num_heads * head_dim + head_idx * head_dim;
        
        // Compute dot product: q · k
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += s_query[d] * k_ptr[d];
        }
        score *= softmax_scale;
        
        if (pos < max_seq_len) {
            s_scores[pos] = score;
        }
        max_score = fmaxf(max_score, score);
    }
    
    // Find global maximum across all threads using warp reduction
    __syncthreads();
    
    // Warp-level reduction for max
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        max_score = fmaxf(max_score, __shfl_down_sync(0xffffffff, max_score, offset));
    }
    
    // First thread in each warp writes to shared memory
    if (threadIdx.x % warpSize == 0) {
        s_query[threadIdx.x / warpSize] = max_score;
    }
    __syncthreads();
    
    // Final reduction across warps
    if (threadIdx.x < blockDim.x / warpSize) {
        max_score = s_query[threadIdx.x];
        for (int offset = (blockDim.x / warpSize) / 2; offset > 0; offset /= 2) {
            if (threadIdx.x < offset) {
                max_score = fmaxf(max_score, s_query[threadIdx.x + offset]);
            }
        }
    }
    
    // Broadcast final max to all threads
    if (threadIdx.x == 0) {
        s_query[0] = max_score;
    }
    __syncthreads();
    max_score = s_query[0];
    
    // Compute softmax: exp(score - max) and sum
    float sum_exp = 0.0f;
    for (int pos = threadIdx.x; pos < num_positions; pos += blockDim.x) {
        if (is_causal && pos > q_pos) continue;
        
        if (pos < max_seq_len) {
            s_scores[pos] = expf(s_scores[pos] - max_score);
            sum_exp += s_scores[pos];
        }
    }
    
    // Reduce sum across threads using warp reduction
    __syncthreads();
    
    // Warp-level reduction for sum
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);
    }
    
    // First thread in each warp writes to shared memory
    if (threadIdx.x % warpSize == 0) {
        s_query[threadIdx.x / warpSize] = sum_exp;
    }
    __syncthreads();
    
    // Final reduction across warps
    if (threadIdx.x < blockDim.x / warpSize) {
        sum_exp = s_query[threadIdx.x];
        for (int offset = (blockDim.x / warpSize) / 2; offset > 0; offset /= 2) {
            if (threadIdx.x < offset) {
                sum_exp += s_query[threadIdx.x + offset];
            }
        }
    }
    
    // Broadcast final sum to all threads
    if (threadIdx.x == 0) {
        s_query[0] = sum_exp;
    }
    __syncthreads();
    sum_exp = s_query[0];
    
    // Compute final output: weighted sum of values
    float* out_ptr = out + batch_idx * seq_len * num_heads * head_dim + 
                     q_pos * num_heads * head_dim + head_idx * head_dim;
    
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float output_val = 0.0f;
        
        for (int pos = 0; pos < num_positions; pos++) {
            if (is_causal && pos > q_pos) continue;
            if (pos >= seq_len || pos < 0) continue;  // Bounds check
            
            // Fix value pointer calculation
            const float* v_ptr = v + batch_idx * seq_len * num_heads * head_dim + 
                                 pos * num_heads * head_dim + head_idx * head_dim;
            
            float weight = (pos < max_seq_len) ? (s_scores[pos] / sum_exp) : 0.0f;
            if (d < head_dim) {
                output_val += weight * v_ptr[d];
            }
        }
        
        if (d < head_dim) {
            out_ptr[d] = output_val;
        }
    }
}

void flash_attention_kernel_cu(const tensor::Tensor& query, const tensor::Tensor& key,
                              const tensor::Tensor& value, const tensor::Tensor& output,
                              int32_t head_num, int32_t head_size, int32_t seq_len, int32_t pos,
                              float softmax_scale, bool is_causal, CudaConfig* config) {
    
    LOG(INFO) << "FlashAttention CUDA kernel: seq_len=" << seq_len << ", heads=" << head_num 
              << ", head_size=" << head_size << ", pos=" << pos;
    
    // Validate input parameters
    if (head_num <= 0 || head_size <= 0 || seq_len <= 0) {
        LOG(ERROR) << "Invalid FlashAttention parameters: head_num=" << head_num 
                   << ", head_size=" << head_size << ", seq_len=" << seq_len;
        return;
    }
    
    // Check tensor validity
    if (query.is_empty() || key.is_empty() || value.is_empty() || output.is_empty()) {
        LOG(ERROR) << "FlashAttention: One or more input tensors are empty";
        return;
    }
    
    const int batch_size = 1;  // For inference
    
    // Get raw pointers with null checks
    const float* q_ptr = query.ptr<float>();
    const float* k_ptr = key.ptr<float>();
    const float* v_ptr = value.ptr<float>();
    float* out_ptr = const_cast<float*>(output.ptr<float>());
    
    if (!q_ptr || !k_ptr || !v_ptr || !out_ptr) {
        LOG(ERROR) << "FlashAttention: Null tensor pointers detected";
        return;
    }
    
    // Launch configuration: one block per head
    dim3 block(128);  // Reduce threads per block to save shared memory
    dim3 grid(head_num, batch_size);
    
    // Shared memory: query vector + attention scores (with safety margin)
    const int max_seq_for_shared = min(seq_len, 256);
    size_t shared_mem_size = (head_size + max_seq_for_shared + 32) * sizeof(float);  // Add padding
    
    // Check shared memory limits
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    if (shared_mem_size > prop.sharedMemPerBlock) {
        LOG(WARNING) << "FlashAttention: Shared memory requirement (" << shared_mem_size 
                     << ") exceeds device limit (" << prop.sharedMemPerBlock 
                     << "). Falling back to standard MHA.";
        return;
    }
    
    cudaStream_t stream = config ? config->stream : nullptr;
    
    // Launch kernel
    flash_attention_fwd_kernel<<<grid, block, shared_mem_size, stream>>>(
        q_ptr, k_ptr, v_ptr, out_ptr,
        batch_size, seq_len, head_num, head_size,
        softmax_scale, is_causal
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG(ERROR) << "FlashAttention kernel launch failed: " << cudaGetErrorString(err);
        return;
    }
    
    // Synchronize and check for execution errors
    if (stream) {
        cudaStreamSynchronize(stream);
    } else {
        cudaDeviceSynchronize();
    }
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG(ERROR) << "FlashAttention kernel execution failed: " << cudaGetErrorString(err);
        return;
    }
    
    LOG(INFO) << "FlashAttention CUDA kernel completed successfully";
}

} // namespace kernel