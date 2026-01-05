//
// Created by hello on 2024/5/2.
//

#ifndef KUIPER_INCLUDE_OP_MATMUL_H_
#define KUIPER_INCLUDE_OP_MATMUL_H_
#include <base/cuda_config.h>
#include "layer.h"
namespace op {
class MatmulLayer : public LayerParam {
 public:
  explicit MatmulLayer(base::DeviceType device_type, int32_t dim0, int32_t dim1,
                       bool is_quant_layer = false, bool has_bias = false);

  base::Status check() const override;

  base::Status forward() override;

  base::Status set_bias(int32_t idx, int32_t& dims, const void* bias_ptr,
                        base::DeviceType device_type);

  tensor::Tensor& get_bias(int32_t idx);

  const tensor::Tensor& get_bias(int32_t idx) const;

  void to_cuda() override;

  // ==========================================================================
  // Tensor Parallelism (TP) Configuration
  // ==========================================================================
  // Configures this layer for distributed execution across multiple GPUs.
  //
  // Parameters:
  // - tp_size: Total number of GPUs in the tensor parallel group.
  // - tp_rank: This GPU's rank within the group (0 to tp_size-1).
  // - need_all_reduce: If true, an AllReduce sum operation is performed after
  //                    the forward pass. This is needed for Row-Parallel layers.
  //
  // Typical Usage (in Llama Attention):
  // - Query/Key/Value projections: Column Parallel (no AllReduce needed after).
  // - Output projection (wo): Row Parallel (need_all_reduce = true).
  void set_tp_config(int32_t tp_size, int32_t tp_rank, bool need_all_reduce = false);

 private:
  int32_t dim0_ = 0;
  int32_t dim1_ = 0;
  bool has_bias_ = false;
  std::vector<tensor::Tensor> bias_;

  int32_t tp_size_ = 1;
  int32_t tp_rank_ = 0;
  bool need_all_reduce_ = false;
};
}  // namespace op
#endif  // KUIPER_INCLUDE_OP_MATMUL_H_
