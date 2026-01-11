#include "flash_attention_kernel.cuh"
#include <base/cuda_config.h>
#include <tensor/tensor.h>
#include <glog/logging.h>

namespace kernel {

// Simplified FlashAttention CUDA kernel based on minimal reference implementation
// Block size is fixed at compile time for simplicity
#define BLOCK_SIZE 32

__global__ void flash_attention_fwd_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K, 
    const float* __restrict__ V,
    float* __restrict__ O,
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
    
    // Shared memory allocation
    extern __shared__ float smem[];
    float* s_q = smem;                                    // [BLOCK_SIZE, head_dim]
    float* s_k = s_q + BLOCK_SIZE * head_dim;            // [BLOCK_SIZE, head_dim]  
    float* s_v = s_k + BLOCK_SIZE * head_dim;            // [BLOCK_SIZE, head_dim]
    float* s_qk = s_v + BLOCK_SIZE * head_dim;           // [BLOCK_SIZE, BLOCK_SIZE]
    
    const int tid = threadIdx.x;
    const int num_blocks = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Process each query block
    for (int q_block = 0; q_block < num_blocks; q_block++) {
        const int q_start = q_block * BLOCK_SIZE;
        const int q_end = min(q_start + BLOCK_SIZE, seq_len);
        const int q_size = q_end - q_start;
        
        // Load Q block into shared memory
        for (int i = tid; i < q_size * head_dim; i += blockDim.x) {
            const int q_row = i / head_dim;
            const int q_col = i % head_dim;
            const int q_idx = batch_idx * seq_len * num_heads * head_dim + 
                             (q_start + q_row) * num_heads * head_dim + 
                             head_idx * head_dim + q_col;
            s_q[q_row * head_dim + q_col] = Q[q_idx];
        }
        
        // Initialize output accumulator and softmax statistics
        float row_max[BLOCK_SIZE];
        float row_sum[BLOCK_SIZE];
        float output[BLOCK_SIZE * 64]; // Assume head_dim <= 64
        
        for (int i = 0; i < q_size; i++) {
            row_max[i] = -INFINITY;
            row_sum[i] = 0.0f;
            for (int j = 0; j < head_dim; j++) {
                output[i * head_dim + j] = 0.0f;
            }
        }
        
        __syncthreads();
        
        // Process each key-value block
        for (int kv_block = 0; kv_block < num_blocks; kv_block++) {
            const int kv_start = kv_block * BLOCK_SIZE;
            const int kv_end = min(kv_start + BLOCK_SIZE, seq_len);
            const int kv_size = kv_end - kv_start;
            
            // Load K and V blocks into shared memory
            for (int i = tid; i < kv_size * head_dim; i += blockDim.x) {
                const int kv_row = i / head_dim;
                const int kv_col = i % head_dim;
                const int k_idx = batch_idx * seq_len * num_heads * head_dim + 
                                 (kv_start + kv_row) * num_heads * head_dim + 
                                 head_idx * head_dim + kv_col;
                s_k[kv_row * head_dim + kv_col] = K[k_idx];
                s_v[kv_row * head_dim + kv_col] = V[k_idx];
            }
            __syncthreads();
            
            // Compute QK^T for this block
            for (int i = tid; i < q_size * kv_size; i += blockDim.x) {
                const int q_row = i / kv_size;
                const int kv_row = i % kv_size;
                
                // Apply causal mask
                if (is_causal && (kv_start + kv_row) > (q_start + q_row)) {
                    s_qk[q_row * BLOCK_SIZE + kv_row] = -INFINITY;
                    continue;
                }
                
                float dot_product = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    dot_product += s_q[q_row * head_dim + d] * s_k[kv_row * head_dim + d];
                }
                s_qk[q_row * BLOCK_SIZE + kv_row] = dot_product * softmax_scale;
            }
            __syncthreads();
            
            // Online softmax update for each query row
            if (tid < q_size) {
                const int q_row = tid;
                
                // Find max in current block
                float block_max = -INFINITY;
                for (int kv_row = 0; kv_row < kv_size; kv_row++) {
                    block_max = fmaxf(block_max, s_qk[q_row * BLOCK_SIZE + kv_row]);
                }
                
                // Update global max
                float new_max = fmaxf(row_max[q_row], block_max);
                
                // Compute exponentials and sum for current block
                float block_sum = 0.0f;
                for (int kv_row = 0; kv_row < kv_size; kv_row++) {
                    float exp_val = expf(s_qk[q_row * BLOCK_SIZE + kv_row] - new_max);
                    s_qk[q_row * BLOCK_SIZE + kv_row] = exp_val;
                    block_sum += exp_val;
                }
                
                // Update running sum with correction factor
                float correction = expf(row_max[q_row] - new_max);
                float new_sum = correction * row_sum[q_row] + block_sum;
                
                // Update output with weighted average
                float old_weight = (row_sum[q_row] * correction) / new_sum;
                float new_weight = block_sum / new_sum;
                
                for (int d = 0; d < head_dim; d++) {
                    float new_output = 0.0f;
                    for (int kv_row = 0; kv_row < kv_size; kv_row++) {
                        new_output += s_qk[q_row * BLOCK_SIZE + kv_row] * s_v[kv_row * head_dim + d];
                    }
                    output[q_row * head_dim + d] = old_weight * output[q_row * head_dim + d] + 
                                                   new_weight * new_output;
                }
                
                // Update statistics
                row_max[q_row] = new_max;
                row_sum[q_row] = new_sum;
            }
            __syncthreads();
        }
        
        // Write final output to global memory
        for (int i = tid; i < q_size * head_dim; i += blockDim.x) {
            const int q_row = i / head_dim;
            const int q_col = i % head_dim;
            const int o_idx = batch_idx * seq_len * num_heads * head_dim + 
                             (q_start + q_row) * num_heads * head_dim + 
                             head_idx * head_dim + q_col;
            O[o_idx] = output[q_row * head_dim + q_col];
        }
        __syncthreads();
    }
}

void flash_attention_kernel_cu(const tensor::Tensor& query, const tensor::Tensor& key,
                              const tensor::Tensor& value, const tensor::Tensor& output,
                              int32_t head_num, int32_t head_size, int32_t seq_len, int32_t pos,
                              float softmax_scale, bool is_causal, CudaConfig* config) {
    
    LOG(INFO) << "FlashAttention CUDA kernel (simplified): seq_len=" << seq_len 
              << ", heads=" << head_num << ", head_size=" << head_size;
    
    // Validate input parameters
    if (head_num <= 0 || head_size <= 0 || seq_len <= 0) {
        LOG(ERROR) << "Invalid FlashAttention parameters";
        return;
    }
    
    if (head_size > 64) {
        LOG(WARNING) << "Head size " << head_size << " > 64, may cause issues";
    }
    
    // Check tensor validity
    if (query.is_empty() || key.is_empty() || value.is_empty() || output.is_empty()) {
        LOG(ERROR) << "FlashAttention: Empty input tensors";
        return;
    }
    
    const int batch_size = 1;  // For inference
    
    // Get raw pointers
    const float* q_ptr = query.ptr<float>();
    const float* k_ptr = key.ptr<float>();
    const float* v_ptr = value.ptr<float>();
    float* out_ptr = const_cast<float*>(output.ptr<float>());
    
    if (!q_ptr || !k_ptr || !v_ptr || !out_ptr) {
        LOG(ERROR) << "FlashAttention: Null tensor pointers";
        return;
    }
    
    // Launch configuration
    dim3 block(256);  // Number of threads per block
    dim3 grid(head_num, batch_size);  // One block per head
    
    // Shared memory: Q, K, V blocks + QK matrix
    size_t shared_mem_size = (3 * BLOCK_SIZE * head_size + BLOCK_SIZE * BLOCK_SIZE) * sizeof(float);
    
    // Check shared memory limits
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    if (shared_mem_size > prop.sharedMemPerBlock) {
        LOG(ERROR) << "FlashAttention: Shared memory requirement (" << shared_mem_size 
                   << ") exceeds device limit (" << prop.sharedMemPerBlock << ")";
        return;
    }
    
    cudaStream_t stream = config ? config->stream : nullptr;
    
    // Launch kernel
    flash_attention_fwd_kernel<<<grid, block, shared_mem_size, stream>>>(
        q_ptr, k_ptr, v_ptr, out_ptr,
        batch_size, seq_len, head_num, head_size,
        softmax_scale, is_causal
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG(ERROR) << "FlashAttention kernel launch failed: " << cudaGetErrorString(err);
        return;
    }
    
    // Synchronize
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