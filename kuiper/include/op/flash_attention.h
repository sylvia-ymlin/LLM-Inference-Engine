#ifndef KUIPER_FLASH_ATTENTION_H
#define KUIPER_FLASH_ATTENTION_H

#include "op/layer.h"
#include "base/base.h"

namespace op {

class FlashAttention : public Layer {
 public:
  explicit FlashAttention(base::DeviceType device_type, int32_t layer_index,
                         int32_t kv_mul, int32_t kv_dim, int32_t seq_len,
                         int32_t head_num, int32_t head_size, float scale = 0.0f);

  base::Status forward() override;

  void set_pos(int32_t pos);
  void set_layer_idx(int32_t layer_idx);

 private:
  base::Status check() const override;

  int32_t layer_index_ = 0;
  int32_t kv_mul_ = 0;
  int32_t kv_dim_ = 0;
  int32_t seq_len_ = 0;
  int32_t head_num_ = 0;
  int32_t head_size_ = 0;
  int32_t pos_ = 0;
  float scale_ = 0.0f;
  float softmax_scale_ = 0.0f;
  bool is_causal_ = true;
};

} // namespace op

#endif // KUIPER_FLASH_ATTENTION_H