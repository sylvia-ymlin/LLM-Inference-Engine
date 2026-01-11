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
    
    // Shared memory for query and attention scores
    extern __shared__ float shared_mem[];
    float* s_query = shared_mem;
    float* s_scores = s_query + head_dim;
    
    // Calculate base offsets for this head
    const int head_offset = batch_idx * seq_len * num_heads * head_dim + head_idx * head_dim;
    
    // Load query for current position (last token in sequence)
    const int q_pos = seq_len - 1;  // For autoregressive generation
    const float* q_ptr = q + head_offset + q_pos * num_heads * head_dim;
    
    // Cooperatively load query to shared memory
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        s_query[d] = q_ptr[d];
    }
    __syncthreads();
    
    // Compute attention scores for all positions up to current
    float max_score = -INFINITY;
    
    for (int pos = threadIdx.x; pos <= q_pos; pos += blockDim.x) {
        if (is_causal && pos > q_pos) {
            s_scores[pos] = -INFINITY;
            continue;
        }
        
        const float* k_ptr = k + head_offset + pos * num_heads * head_dim;
        
        // Compute dot product: q · k
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += s_query[d] * k_ptr[d];
        }
        score *= softmax_scale;
        
        s_scores[pos] = score;
        max_score = fmaxf(max_score, score);
    }
    
    // Find global maximum across all threads
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            max_score = fmaxf(max_score, __shfl_down_sync(0xffffffff, max_score, stride));
        }
    }
    
    // Broadcast max to all threads
    if (threadIdx.x == 0) {
        s_query[0] = max_score;  // Reuse shared memory
    }
    __syncthreads();
    max_score = s_query[0];
    
    // Compute softmax: exp(score - max) and sum
    float sum_exp = 0.0f;
    for (int pos = threadIdx.x; pos <= q_pos; pos += blockDim.x) {
        if (is_causal && pos > q_pos) continue;
        
        s_scores[pos] = expf(s_scores[pos] - max_score);
        sum_exp += s_scores[pos];
    }
    
    // Reduce sum across threads
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            sum_exp += __shfl_down_sync(0xffffffff, sum_exp, stride);
        }
    }
    
    // Broadcast sum to all threads
    if (threadIdx.x == 0) {
        s_query[0] = sum_exp;
    }
    __syncthreads();
    sum_exp = s_query[0];
    
    // Compute final output: weighted sum of values
    float* out_ptr = out + head_offset + q_pos * num_heads * head_dim;
    
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float output_val = 0.0f;
        
        for (int pos = 0; pos <= q_pos; pos++) {
            if (is_causal && pos > q_pos) continue;
            
            const float* v_ptr = v + head_offset + pos * num_heads * head_dim;
            float weight = s_scores[pos] / sum_exp;
            output_val += weight * v_ptr[d];
        }
        
        out_ptr[d] = output_val;
    }
}

void flash_attention_kernel_cu(const tensor::Tensor& query, const tensor::Tensor& key,
                              const tensor::Tensor& value, const tensor::Tensor& output,
                              int32_t head_num, int32_t head_size, int32_t seq_len, int32_t pos,
                              float softmax_scale, bool is_causal, CudaConfig* config) {
    
    LOG(INFO) << "FlashAttention CUDA kernel: seq_len=" << seq_len << ", heads=" << head_num 
              << ", head_size=" << head_size << ", pos=" << pos;
    
    const int batch_size = 1;  // For inference
    
    // Get raw pointers
    const float* q_ptr = query.ptr<float>();
    const float* k_ptr = key.ptr<float>();
    const float* v_ptr = value.ptr<float>();
    float* out_ptr = const_cast<float*>(output.ptr<float>());
    
    // Launch configuration: one block per head
    dim3 block(256);  // 256 threads per block
    dim3 grid(head_num, batch_size);
    
    // Shared memory: query vector + attention scores
    size_t shared_mem_size = (head_size + seq_len) * sizeof(float);
    
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
    
    // Synchronize if needed
    if (stream) {
        cudaStreamSynchronize(stream);
    } else {
        cudaDeviceSynchronize();
    }
    
    LOG(INFO) << "FlashAttention CUDA kernel completed successfully";
}

} // namespace kernel