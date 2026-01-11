#include "flash_attention_kernel.cuh"
#include <base/cuda_config.h>
#include <tensor/tensor.h>
#include <glog/logging.h>

namespace kernel {

// Simple FlashAttention-inspired kernel for demonstration
// This is a simplified version that maintains the memory efficiency principles
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
    
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int seq_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size || head_idx >= num_heads || seq_idx >= seq_len) {
        return;
    }
    
    // Calculate offsets
    const int q_offset = batch_idx * seq_len * num_heads * head_dim + 
                        seq_idx * num_heads * head_dim + head_idx * head_dim;
    const int kv_offset_base = batch_idx * seq_len * num_heads * head_dim + head_idx * head_dim;
    const int out_offset = q_offset;
    
    // Load query vector
    extern __shared__ float shared_mem[];
    float* s_q = shared_mem;
    float* s_scores = s_q + head_dim;
    
    // Load query to shared memory
    for (int d = threadIdx.y; d < head_dim; d += blockDim.y) {
        s_q[d] = q[q_offset + d];
    }
    __syncthreads();
    
    // Compute attention scores
    float max_score = -INFINITY;
    for (int t = threadIdx.y; t <= seq_idx; t += blockDim.y) {
        if (is_causal && t > seq_idx) continue;
        
        const int k_offset = kv_offset_base + t * num_heads * head_dim;
        
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += s_q[d] * k[k_offset + d];
        }
        score *= softmax_scale;
        
        s_scores[t] = score;
        max_score = fmaxf(max_score, score);
    }
    
    // Reduce max across threads
    __syncthreads();
    for (int stride = blockDim.y / 2; stride > 0; stride /= 2) {
        if (threadIdx.y < stride) {
            max_score = fmaxf(max_score, __shfl_down_sync(0xffffffff, max_score, stride));
        }
    }
    
    // Compute softmax
    float sum_exp = 0.0f;
    for (int t = threadIdx.y; t <= seq_idx; t += blockDim.y) {
        if (is_causal && t > seq_idx) continue;
        
        s_scores[t] = expf(s_scores[t] - max_score);
        sum_exp += s_scores[t];
    }
    
    // Reduce sum across threads
    __syncthreads();
    for (int stride = blockDim.y / 2; stride > 0; stride /= 2) {
        if (threadIdx.y < stride) {
            sum_exp += __shfl_down_sync(0xffffffff, sum_exp, stride);
        }
    }
    
    // Normalize and compute output
    for (int d = threadIdx.y; d < head_dim; d += blockDim.y) {
        float output_val = 0.0f;
        for (int t = 0; t <= seq_idx; t++) {
            if (is_causal && t > seq_idx) continue;
            
            const int v_offset = kv_offset_base + t * num_heads * head_dim;
            float weight = s_scores[t] / sum_exp;
            output_val += weight * v[v_offset + d];
        }
        out[out_offset + d] = output_val;
    }
}

void flash_attention_kernel_cu(const tensor::Tensor& query, const tensor::Tensor& key,
                              const tensor::Tensor& value, const tensor::Tensor& output,
                              int32_t head_num, int32_t head_size, int32_t seq_len, int32_t pos,
                              float softmax_scale, bool is_causal, CudaConfig* config) {

    const int batch_size = 1;  // For inference
    const int current_seq_len = pos + 1;
    
    // Get raw pointers
    const float* q_ptr = query.ptr<float>();
    const float* k_ptr = key.ptr<float>();
    const float* v_ptr = value.ptr<float>();
    float* out_ptr = const_cast<float*>(output.ptr<float>());
    
    // Launch configuration
    dim3 block(32, 8);  // 32 threads for sequence, 8 for head_dim
    dim3 grid((current_seq_len + block.x - 1) / block.x, head_num, batch_size);
    
    // Shared memory size: query vector + scores
    size_t shared_mem_size = (head_size + seq_len) * sizeof(float);
    
    cudaStream_t stream = config ? config->stream : nullptr;
    
    flash_attention_fwd_kernel<<<grid, block, shared_mem_size, stream>>>(
        q_ptr, k_ptr, v_ptr, out_ptr,
        batch_size, current_seq_len, head_num, head_size,
        softmax_scale, is_causal
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG(ERROR) << "FlashAttention kernel launch failed: " << cudaGetErrorString(err);
    }
    
    if (stream) {
        cudaStreamSynchronize(stream);
    }
}

} // namespace kernel