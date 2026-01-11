#include "flash_attention_kernel.h"
#include "../kernels_interface.h"
#include <cuda_runtime_api.h>

namespace kernel {

void flash_attention_kernel(const tensor::Tensor& query, const tensor::Tensor& key,
                           const tensor::Tensor& value, const tensor::Tensor& output,
                           int32_t head_num, int32_t head_size, int32_t seq_len, int32_t pos,
                           float softmax_scale, bool is_causal,
                           base::DeviceType device_type, CudaConfig* config) {
  
  if (device_type == base::DeviceType::kDeviceCUDA) {
#ifdef KUIPER_USE_FLASH_ATTENTION
    // Call CUDA FlashAttention kernel
    flash_attention_kernel_cu(query, key, value, output, head_num, head_size, 
                             seq_len, pos, softmax_scale, is_causal, config);
#else
    LOG(ERROR) << "FlashAttention CUDA support not compiled. Please build with -DUSE_FLASH_ATTENTION=ON";
    // Fallback to standard attention
    LOG(INFO) << "Falling back to standard attention implementation";
    // TODO: Call standard MHA kernel as fallback
#endif
  } else {
    // CPU implementation - fallback to standard attention
    LOG(INFO) << "FlashAttention CPU fallback to standard attention";
    // TODO: Implement CPU fallback or call standard MHA kernel
  }
}

} // namespace kernel