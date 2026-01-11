#include "flash_attention_kernel.h"
#include "../kernels_interface.h"
#include "../cuda/flash_attention_kernel.cuh"
#include <cuda_runtime_api.h>

namespace kernel {

void flash_attention_kernel(const tensor::Tensor& query, const tensor::Tensor& key,
                           const tensor::Tensor& value, const tensor::Tensor& output,
                           int32_t head_num, int32_t head_size, int32_t seq_len, int32_t pos,
                           float softmax_scale, bool is_causal,
                           base::DeviceType device_type, CudaConfig* config) {
  
  if (device_type == base::DeviceType::kDeviceCUDA) {
#ifdef KUIPER_USE_FLASH_ATTENTION
    // Temporarily disable FlashAttention CUDA kernel due to memory access issues
    // TODO: Fix tensor layout compatibility
    LOG(WARNING) << "FlashAttention CUDA kernel temporarily disabled, using standard MHA";
    get_mha_kernel(device_type)(pos, head_num, 0, seq_len, head_num * head_size, 1, head_size,
                               output, query, tensor::Tensor(), key, value, device_type, config);
#else
    LOG(WARNING) << "FlashAttention CUDA support not compiled. Falling back to standard MHA";
    // Fallback to standard attention
    get_mha_kernel(device_type)(pos, head_num, 0, seq_len, head_num * head_size, 1, head_size,
                               output, query, tensor::Tensor(), key, value, device_type, config);
#endif
  } else {
    // CPU implementation - fallback to standard attention
    LOG(INFO) << "FlashAttention CPU fallback to standard attention";
    get_mha_kernel(device_type)(pos, head_num, 0, seq_len, head_num * head_size, 1, head_size,
                               output, query, tensor::Tensor(), key, value, device_type, config);
  }
}

} // namespace kernel