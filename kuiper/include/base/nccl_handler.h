#ifndef KUIPER_INCLUDE_BASE_NCCL_HANDLER_H_
#define KUIPER_INCLUDE_BASE_NCCL_HANDLER_H_

#include <memory>
#include <vector>
#include "base.h"

#if defined(KUIPER_USE_NCCL)
#include <cuda_runtime.h>
#include <nccl.h>
#endif

namespace base {

// ======================================================================================
// Distributed Communication Handler
// ======================================================================================
// This class encapsulates all NCCL (NVIDIA Collective Communications Library) operations.
// Design Pattern: Singleton
//
// Motivation:
// 1. Decouple model logic from low-level MPI/NCCL communication primitives.
// 2. Manage the lifecycle of NCCL communicators (init/destroy) in one place.
// 3. Provide a simplified API (AllReduce, Broadcast) for the inference layers.
//
// Usage:
// - Call Init() at the start of the program to establish the mesh.
// - Use AllReduce() in MatmulLayer to sum partial results across GPUs.
class NcclHandler {
 public:
  NcclHandler() = default;
  ~NcclHandler();

  static std::shared_ptr<NcclHandler> get_instance();

  Status Init(int32_t rank, int32_t world_size, const std::string& init_method = "");

  Status AllReduce(const void* sendbuff, void* recvbuff, size_t count, DataType datatype,
                   cudaStream_t stream = nullptr);

  Status Broadcast(const void* sendbuff, void* recvbuff, size_t count, DataType datatype,
                   int32_t root, cudaStream_t stream = nullptr);

  Status AllGather(const void* sendbuff, void* recvbuff, size_t send_count, DataType datatype,
                   cudaStream_t stream = nullptr);

  int32_t rank() const { return rank_; }
  int32_t world_size() const { return world_size_; }

 private:
  int32_t rank_ = 0;
  int32_t world_size_ = 1;

#if defined(KUIPER_USE_NCCL)
  ncclComm_t comm_ = nullptr;
  ncclUniqueId id_;
#endif
};

}  // namespace base

#endif  // KUIPER_INCLUDE_BASE_NCCL_HANDLER_H_
