#ifndef KUIPER_FLASH_ATTENTION_KERNEL_H
#define KUIPER_FLASH_ATTENTION_KERNEL_H

#include <base/cuda_config.h>
#include "base/base.h"
#include "tensor/tensor.h"

namespace kernel {

void flash_attention_kernel(const tensor::Tensor& query, const tensor::Tensor& key,
                           const tensor::Tensor& value, const tensor::Tensor& output,
                           int32_t head_num, int32_t head_size, int32_t seq_len, int32_t pos,
                           float softmax_scale, bool is_causal,
                           base::DeviceType device_type, CudaConfig* config);

} // namespace kernel

#endif // KUIPER_FLASH_ATTENTION_KERNEL_H