#include "op/flash_attention.h"
#include "kernels/cpu/flash_attention_kernel.h"
#include "kernels/kernels_interface.h"
#include <cmath>

/*
FlashAttention layer contract:
- Inputs: Q, K, V
- Output: O with shape [batch, seq_len, num_heads, head_dim]
- softmax_scale defaults to 1/sqrt(head_size) when not provided
- forward() routes to kernel::get_flash_attention_kernel with device-specific config
Notes:
- CUDA execution requires a valid cuda_config_ (stream etc.)
*/

namespace op {

FlashAttention::FlashAttention(base::DeviceType device_type, int32_t layer_index,
                              int32_t kv_mul, int32_t kv_dim, int32_t seq_len,
                              int32_t head_num, int32_t head_size, float scale)
    : Layer(device_type, LayerType::kLayerMHA, "FlashAttention"),
      layer_index_(layer_index),
      kv_mul_(kv_mul),
      kv_dim_(kv_dim),
      seq_len_(seq_len),
      head_num_(head_num),
      head_size_(head_size),
      scale_(scale) {
  
  // Set default scale if not provided
  if (scale_ == 0.0f) {
    scale_ = 1.0f / std::sqrt(static_cast<float>(head_size_));
  }
  softmax_scale_ = scale_;
  
  // FlashAttention expects 3 inputs: Q, K, V and 1 output
  reset_input_size(3);
  reset_output_size(1);
}

base::Status FlashAttention::forward() {
  auto status = check();
  if (!status) {
    return status;
  }

  const tensor::Tensor& query_tensor = this->get_input(0);
  const tensor::Tensor& key_tensor = this->get_input(1);
  const tensor::Tensor& value_tensor = this->get_input(2);
  const tensor::Tensor& output_tensor = this->get_output(0);

  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    CHECK(cuda_config_ != nullptr);
  }

  // Call FlashAttention kernel
  kernel::get_flash_attention_kernel(device_type_)(
      query_tensor, key_tensor, value_tensor, output_tensor,
      head_num_, head_size_, seq_len_, pos_, softmax_scale_, is_causal_,
      device_type_, cuda_config_ ? cuda_config_.get() : nullptr);

  return base::error::Success();
}

void FlashAttention::set_pos(int32_t pos) { 
  this->pos_ = pos; 
}

void FlashAttention::set_layer_idx(int32_t layer_idx) { 
  this->layer_index_ = layer_idx; 
}

base::Status FlashAttention::check() const {
  base::Status status;
  
  // Check input tensors (Q, K, V)
  for (int32_t i = 0; i < 3; ++i) {
    status = check_tensor(get_input(i), device_type_, data_type_);
    if (!status) {
      LOG(ERROR) << "The input tensor " << std::to_string(i) << " error in the FlashAttention layer.";
      return status;
    }
  }
  
  // Check output tensor
  return check_tensor(get_output(0), device_type_, data_type_);
}

} // namespace op
