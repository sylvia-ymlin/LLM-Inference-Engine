#include "flash_attention_kernel.h"
#include "../kernels_interface.h"
#include "../cuda/flash_attention_kernel.cuh"
#include "base/alloc.h"
#include <cuda_runtime_api.h>

namespace kernel {

void flash_attention_kernel(const tensor::Tensor& query, const tensor::Tensor& key,
                           const tensor::Tensor& value, const tensor::Tensor& output,
                           int32_t head_num, int32_t head_size, int32_t seq_len, int32_t pos,
                           float softmax_scale, bool is_causal,
                           base::DeviceType device_type, CudaConfig* config) {
  
  // Create score tensor for MHA kernel
  std::shared_ptr<base::DeviceAllocator> allocator;
  if (device_type == base::DeviceType::kDeviceCPU) {
    allocator = base::CPUDeviceAllocatorFactory::get_instance();
  } else {
    allocator = base::CUDADeviceAllocatorFactory::get_instance();
  }
  
  // Create score tensor with proper dimensions (head_num * seq_len)
  tensor::Tensor score_tensor(base::DataType::kDataTypeFp32, head_num, seq_len, true, allocator);
  score_tensor.set_device_type(device_type);
  
  if (device_type == base::DeviceType::kDeviceCUDA) {
#ifdef KUIPER_USE_FLASH_ATTENTION
    // Use FlashAttention CUDA kernel
    LOG(INFO) << "Using FlashAttention CUDA kernel";
    flash_attention_kernel_cu(query, key, value, output, head_num, head_size, seq_len, pos,
                             softmax_scale, is_causal, config);
#else
    LOG(WARNING) << "FlashAttention CUDA support not compiled. Falling back to standard MHA";
    // Fallback to standard attention
    get_mha_kernel(device_type)(pos, head_num, 0, seq_len, head_num * head_size, 1, head_size,
                               output, query, score_tensor, key, value, device_type, config);
#endif
  } else {
    // CPU implementation - fallback to standard attention
    LOG(INFO) << "FlashAttention CPU fallback to standard attention";
    get_mha_kernel(device_type)(pos, head_num, 0, seq_len, head_num * head_size, 1, head_size,
                               output, query, score_tensor, key, value, device_type, config);
  }
}

} // namespace kernel