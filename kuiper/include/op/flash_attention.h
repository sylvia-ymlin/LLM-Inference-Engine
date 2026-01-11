#ifndef KUIPER_INCLUDE_FLASH_ATTENTION_H
#define KUIPER_INCLUDE_FLASH_ATTENTION_H

#include <base/cuda_config.h>
#include "layer.h"

namespace op {

class FlashAttention : public op::Layer {
 public:
  explicit FlashAttention(base::DeviceType device_type, int32_t layer_index,
                         int32_t kv_mul, int32_t kv_dim, int32_t seq_len,
                         int32_t head_num, int32_t head_size, float scale = 0.0f);

  base::Status check() const override;

  void set_pos(int32_t pos);
  void set_layer_idx(int32_t layer_idx);

  base::Status forward() override;

 private:
  int32_t layer_index_ = 0;
  int32_t pos_ = 0;
  int32_t kv_mul_ = 0;
  int32_t kv_dim_ = 0;
  int32_t seq_len_ = 0;
  int32_t head_num_ = 0;
  int32_t head_size_ = 0;
  float scale_ = 0.0f;
  
  // FlashAttention specific parameters
  bool is_causal_ = true;
  float softmax_scale_ = 0.0f;
};

} // namespace op

#endif // KUIPER_INCLUDE_FLASH_ATTENTION_H