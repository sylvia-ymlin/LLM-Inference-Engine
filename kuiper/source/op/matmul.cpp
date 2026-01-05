#include "op/matmul.h"
#include "base/nccl_handler.h"
#include "kernels/cpu/matmul_kernel.h"
#include "kernels/kernels_interface.h"
namespace op {
MatmulLayer::MatmulLayer(base::DeviceType device_type, int32_t dim0, int32_t dim1,
                         bool is_quant_layer, bool has_bias)
    : LayerParam(device_type, LayerType::kLayerMatmul, is_quant_layer, "Matmul"),
      dim0_(dim0),
      dim1_(dim1),
      has_bias_(has_bias) {
  reset_input_size(1);
  reset_output_size(1);
  reset_weight_size(1);
  if (has_bias_) {
    bias_.resize(1);
  }
}

base::Status MatmulLayer::check() const {
  auto status = check_tensor_with_dim(get_input(0), device_type_, data_type_, dim1_);
  if (!status) {
    LOG(ERROR) << "The input tensor error in the matmul layer.";
    return status;
  }

  if (!is_quant_layer_) {
    status = check_tensor_with_dim(get_weight(0), device_type_, data_type_, dim0_, dim1_);
    if (!status) {
      LOG(ERROR) << "The weight tensor error in the matmul layer.";
      return status;
    }
  } else {
    status = check_tensor_with_dim(get_weight(0), device_type_, base::DataType::kDataTypeInt8,
                                   dim0_, dim1_);
    if (!status) {
      LOG(ERROR) << "The weight tensor error in the matmul layer.";
      return status;
    }
  }

  if (is_quant_layer_) {
    status =
        check_tensor_with_dim(scales_, device_type_, base::DataType::kDataTypeFp32, scales_.size());
    if (!status) {
      LOG(ERROR) << "The scale tensor error in the matmul layer.";
      return status;
    }
  }

  status = check_tensor_with_dim(get_output(0), device_type_, data_type_, dim0_);
  if (!status) {
    LOG(ERROR) << "The output tensor error in the matmul layer.";
    return status;
  }
  return base::error::Success();
}

base::Status MatmulLayer::forward() {
  auto status = check();
  if (!status) {
    return status;
  }
  if (device_type_ == base::DeviceType::kDeviceCUDA) {
    CHECK(cuda_config_ != nullptr);
  }
  if (is_quant_layer_) {
    kernel::get_matmul_kernel_quant8(device_type_)(get_input(0), get_weight(0), get_output(0),
                                                   group_size_, scales_,
                                                   cuda_config_ ? cuda_config_.get() : nullptr);
  } else {
    kernel::get_matmul_kernel(device_type_)(get_input(0), get_weight(0), get_output(0), 1.f,
                                            cuda_config_ ? cuda_config_.get() : nullptr);
  }

  if (has_bias_) {
    kernel::get_add_kernel(device_type_)(get_output(0), get_bias(0), get_output(0),
                                         cuda_config_ ? cuda_config_->stream : nullptr);
  }

  if (need_all_reduce_ && tp_size_ > 1) {
    if (device_type_ == base::DeviceType::kDeviceCUDA) {
      void* ptr = get_output(0).ptr<void>();
      size_t size = get_output(0).size();

      if (cuda_config_->comm_stream) {
        // =================================================================================
        // Computation-Communication Overlap Strategy
        // =================================================================================
        // Goal: Hide the latency of AllReduce by allowing the CPU to proceed and
        //       letting the GPU manage dependencies via Streams/Events.

        // 1. Record event on compute stream: "Computation (MatMul) is finished"
        //    This ensures we don't start reducing before the data is ready.
        cudaEvent_t compute_done;
        cudaEventCreate(&compute_done);
        cudaEventRecord(compute_done, cuda_config_->stream);

        // 2. Make comm stream wait for computation
        //    The Comm stream will block here until MatMul is done.
        cudaStreamWaitEvent(cuda_config_->comm_stream, compute_done, 0);

        // 3. Launch AllReduce on comm stream
        //    This happens asynchronously relative to the CPU.
        base::NcclHandler::get_instance()->AllReduce(ptr, ptr, size, data_type_,
                                                     cuda_config_->comm_stream);

        // 4. Record event on comm stream: "Communication is finished"
        cudaEvent_t comm_done;
        cudaEventCreate(&comm_done);
        cudaEventRecord(comm_done, cuda_config_->comm_stream);

        // 5. Make compute stream wait for communication
        //    Any subsequent operation on 'stream' (e.g., Activation, Next Layer)
        //    will wait until the AllReduce is complete.
        cudaStreamWaitEvent(cuda_config_->stream, comm_done, 0);

        // Cleanup events (Note: In production, reuse events to avoid creation overhead)
        cudaEventDestroy(compute_done);
        cudaEventDestroy(comm_done);
      } else {
        // Fallback to single stream
        base::NcclHandler::get_instance()->AllReduce(ptr, ptr, size, data_type_,
                                                     cuda_config_->stream);
      }
    }
  }

  return base::error::Success();
}

void MatmulLayer::set_tp_config(int32_t tp_size, int32_t tp_rank, bool need_all_reduce) {
  tp_size_ = tp_size;
  tp_rank_ = tp_rank;
  need_all_reduce_ = need_all_reduce;
}

base::Status MatmulLayer::set_bias(int32_t idx, int32_t& dim, const void* bias_ptr,
                                   base::DeviceType device_type) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, bias_.size());
  CHECK_NE(bias_ptr, nullptr);

  size_t size = dim * sizeof(float);
  std::shared_ptr<base::Buffer> buffer =
      std::make_shared<base::Buffer>(size, nullptr, const_cast<void*>(bias_ptr), true);
  if (device_type != base::DeviceType::kDeviceUnknown) {
    buffer->set_device_type(device_type);
  }

  if (!is_quant_layer_) {
    tensor::Tensor bias(base::DataType::kDataTypeFp32, dim);
    bias.set_device_type(device_type);
    CHECK(bias.assign(buffer));
    // LOG(INFO) << "bias:" << bias.index<float>(0);
    bias_.at(idx) = bias;
  } else {
    // is quant layer
    tensor::Tensor bias(base::DataType::kDataTypeInt8, dim);
    bias.set_device_type(device_type);
    CHECK(bias.assign(buffer));
    bias_.at(idx) = bias;

    const int32_t bias_size = static_cast<int32_t>(bias.size());
    CHECK(bias_size % group_size_ == 0);

    int32_t scale_nums = bias_size / group_size_;
    scales_ = tensor::Tensor{base::DataType::kDataTypeFp32, scale_nums, false, nullptr,
                             reinterpret_cast<float*>((int8_t*)bias_ptr + bias_size)};
    scales_.set_device_type(device_type);
  }

  return base::error::Success();
}

tensor::Tensor& MatmulLayer::get_bias(int32_t idx) {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, bias_.size());
  return bias_.at(idx);
}

const tensor::Tensor& MatmulLayer::get_bias(int32_t idx) const {
  CHECK_GE(idx, 0);
  CHECK_LT(idx, bias_.size());
  return bias_.at(idx);
}

void MatmulLayer::to_cuda() {
  LayerParam::to_cuda();
  if (has_bias_) {
    for (auto& bias : bias_) {
      bias.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
    }
  }
}

}  // namespace op