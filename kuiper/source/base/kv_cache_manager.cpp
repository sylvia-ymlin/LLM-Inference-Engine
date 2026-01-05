#include "base/kv_cache_manager.h"
#include <glog/logging.h>

namespace base {

KVCacheManager::KVCacheManager(DeviceType device_type, const KVCacheConfig& config)
    : device_type_(device_type), config_(config) {
  if (device_type_ == DeviceType::kDeviceCPU) {
    allocator_ = CPUDeviceAllocatorFactory::get_instance();
  } else if (device_type_ == DeviceType::kDeviceCUDA) {
    allocator_ = CUDADeviceAllocatorFactory::get_instance();
  }
}

KVCacheManager::~KVCacheManager() {
  if (raw_buffer_ && allocator_) {
    allocator_->release(raw_buffer_);
  }
}

Status KVCacheManager::allocate() {
  if (!allocator_) {
    return error::InternalError("Allocator not initialized in KVCacheManager");
  }

  size_t element_size = DataTypeSize(config_.dtype);
  // Size = Layers * Batch * Heads * MaxSeqLen * HeadDim
  size_t elements_per_layer =
      (size_t)config_.max_batch_size * config_.kv_head_num * config_.max_seq_len * config_.kv_dim;
  // We need both K and V cache, so multiply by 2
  size_t total_elements = config_.layer_num * elements_per_layer * 2;

  total_bytes_ = total_elements * element_size;

  LOG(INFO) << "Allocating KV Cache: " << total_bytes_ / (1024.0 * 1024.0) << " MB";

  raw_buffer_ = allocator_->allocate(total_bytes_);
  if (!raw_buffer_) {
    return error::InternalError("Failed to allocate KV cache memory");
  }

  // Zero out memory
  if (device_type_ == DeviceType::kDeviceCUDA) {
    allocator_->memset_zero(raw_buffer_, total_bytes_, nullptr, false);
  } else {
    allocator_->memset_zero(raw_buffer_, total_bytes_, nullptr, false);
  }

  return error::Success();
}

std::pair<void*, void*> KVCacheManager::get_cache_ptr(int32_t layer_idx, int32_t batch_idx) {
  if (!raw_buffer_) return {nullptr, nullptr};

  size_t element_size = DataTypeSize(config_.dtype);
  // Elements per K or V block (all layers)
  // Actually, to match original [Layer, Seq, Dim], we need [Layer, Batch, Head, Seq, Dim]
  // flattened? Original: [Layer, Seq, Dim] (Batch=1 probably?)

  // Original layout is: KeyCache = [Layers, Seq, KVDim].
  // So for Layer L: ptr + L * Seq * KVDim.

  size_t single_layer_size = (size_t)config_.max_batch_size * config_.kv_head_num *
                             config_.max_seq_len * config_.kv_dim * element_size;

  size_t k_cache_total_size = config_.layer_num * single_layer_size;

  int8_t* k_base = static_cast<int8_t*>(raw_buffer_);
  int8_t* v_base = k_base + k_cache_total_size;

  int8_t* k_layer_ptr = k_base + layer_idx * single_layer_size;
  int8_t* v_layer_ptr = v_base + layer_idx * single_layer_size;

  // Assuming batch=0 for simplicity heavily used in this single-batch-focused codebase for now
  // But adding batch offset support:
  size_t batch_offset =
      batch_idx * config_.kv_head_num * config_.max_seq_len * config_.kv_dim * element_size;

  return {k_layer_ptr + batch_offset, v_layer_ptr + batch_offset};
}

}  // namespace base
