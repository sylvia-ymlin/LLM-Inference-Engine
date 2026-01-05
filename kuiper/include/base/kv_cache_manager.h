#ifndef KUIPER_INCLUDE_BASE_KV_CACHE_MANAGER_H_
#define KUIPER_INCLUDE_BASE_KV_CACHE_MANAGER_H_

#include <map>
#include <memory>
#include <vector>
#include "alloc.h"
#include "base.h"

namespace base {

struct KVCacheConfig {
  int32_t max_batch_size = 1;
  int32_t max_seq_len = 0;
  int32_t layer_num = 0;
  int32_t kv_head_num = 0;
  int32_t kv_dim = 0;  // per head dim
  DataType dtype = DataType::kDataTypeFp32;
};

// ======================================================================================
// KV Cache Block Manager
// ======================================================================================
// This class implements a Pre-allocated Memory Pool for Key-Value Caches.
//
// Motivation:
// 1. Elimination of Dynamic Allocation: Malloc/Free during decoding is expensive and causes
// fragmentation.
// 2. Contiguous Layout: Ensures KV blocks are contiguous in VRAM for optimized kernel access.
// 3. Paged Attention Readiness: Serves as the foundation for paged attention (block-based
// management).
//
// Mechanism:
// - Allocates one massive buffer at Startup.
// - Hands out pointers ("views") to specific layers upon request.
class KVCacheManager {
 public:
  explicit KVCacheManager(DeviceType device_type, const KVCacheConfig& config);
  ~KVCacheManager();

  Status allocate();

  // Get pointers to K and V cache for a specific layer
  // Returns pair of {k_ptr, v_ptr}
  std::pair<void*, void*> get_cache_ptr(int32_t layer_idx, int32_t batch_idx = 0);

  void* get_raw_buffer_k() { return raw_buffer_; }
  void* get_raw_buffer_v() {
    if (!raw_buffer_) return nullptr;
    size_t k_size = total_bytes_ / 2;
    return static_cast<int8_t*>(raw_buffer_) + k_size;
  }

 private:
  DeviceType device_type_;
  KVCacheConfig config_;

  // Single large buffer for all layers and batches
  // Layout: [Layer, Batch, Head, Seq, Dim] or similar
  // Usually: [Layer, Batch, KV_Head, Max_Seq, Head_Dim]
  void* raw_buffer_ = nullptr;
  size_t total_bytes_ = 0;

  std::shared_ptr<DeviceAllocator> allocator_;
};

}  // namespace base

#endif  // KUIPER_INCLUDE_BASE_KV_CACHE_MANAGER_H_
