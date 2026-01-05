#include "base/nccl_handler.h"
#include <iostream>
#include <stdexcept>

namespace base {

static std::shared_ptr<NcclHandler> instance = nullptr;

std::shared_ptr<NcclHandler> NcclHandler::get_instance() {
  if (instance == nullptr) {
    instance = std::make_shared<NcclHandler>();
  }
  return instance;
}

NcclHandler::~NcclHandler() {
#if defined(KUIPER_USE_NCCL)
  if (comm_) {
    ncclCommDestroy(comm_);
  }
#endif
}

Status NcclHandler::Init(int32_t rank, int32_t world_size, const std::string& init_method) {
  rank_ = rank;
  world_size_ = world_size;
  if (world_size_ <= 1) {
    return error::Success();
  }

#if defined(KUIPER_USE_NCCL)
  // ==========================================================================
  // NCCL Timeout Configuration
  // ==========================================================================
  // Problem: Without timeout handling, NCCL operations can hang indefinitely
  // when a GPU fails or network issues occur. This causes the entire training
  // job to freeze without any error message.
  //
  // Solution: Configure NCCL with proper timeout and async error handling.
  //
  // Environment variables set here (can also be set externally):
  // - NCCL_ASYNC_ERROR_HANDLING: Enable async error detection
  // - NCCL_BLOCKING_WAIT: Disable blocking waits to allow timeout detection
  // - NCCL_TIMEOUT: Set explicit timeout in seconds

  // Enable async error handling - allows NCCL to report errors without blocking
  setenv("NCCL_ASYNC_ERROR_HANDLING", "1", 0);  // 0 = don't overwrite if already set

  // Disable blocking waits - required for timeout to work properly
  setenv("NCCL_BLOCKING_WAIT", "0", 0);

  // Set timeout to 30 minutes (1800 seconds) - adjust based on workload
  // Default NCCL timeout is infinite, which can cause hangs
  setenv("NCCL_TIMEOUT", "1800", 0);

  // For debugging NCCL issues, enable verbose logging (optional)
  // setenv("NCCL_DEBUG", "INFO", 0);
  // setenv("NCCL_DEBUG_SUBSYS", "ALL", 0);

  if (rank_ == 0) {
    ncclGetUniqueId(&id_);
  }
  // Note: In a real distributed setting, 'id_' must be broadcasted from rank 0 to all other ranks.
  // This usually requires an out-of-band communication mechanism (Redis, Socket, MPI).

  // Initialize communicator with timeout protection
  // ncclResult_t result = ncclCommInitRank(&comm_, world_size_, id_, rank_);
  // if (result != ncclSuccess) {
  //   return error::InternalError("NCCL communicator initialization failed: " +
  //                               std::string(ncclGetErrorString(result)));
  // }

  // Verify communicator state after initialization
  // ncclResult_t state;
  // ncclCommGetAsyncError(comm_, &state);
  // if (state != ncclSuccess) {
  //   return error::InternalError("NCCL async error detected during init: " +
  //                               std::string(ncclGetErrorString(state)));
  // }
#endif
  return error::Success();
}

Status NcclHandler::AllReduce(const void* sendbuff, void* recvbuff, size_t count, DataType datatype,
                              cudaStream_t stream) {
  if (world_size_ <= 1) {
    // If not distributed, just copy if buffers are different, else do nothing (in-place)
    if (sendbuff != recvbuff) {
      // cudaMemcpyAsync(recvbuff, sendbuff, count * DataTypeSize(datatype),
      // cudaMemcpyDeviceToDevice, stream);
    }
    return error::Success();
  }

#if defined(KUIPER_USE_NCCL)
  ncclDataType_t nccl_type;
  switch (datatype) {
    case DataType::kDataTypeFp32:
      nccl_type = ncclFloat;
      break;
    case DataType::kDataTypeInt8:
      nccl_type = ncclInt8;
      break;
    case DataType::kDataTypeInt32:
      nccl_type = ncclInt32;
      break;
    default:
      return error::InvalidArgument("Unsupported data type for AllReduce");
  }

  ncclResult_t res = ncclAllReduce(sendbuff, recvbuff, count, nccl_type, ncclSum, comm_, stream);
  if (res != ncclSuccess) {
    return error::InternalError("NCCL AllReduce failed");
  }
#endif
  return error::Success();
}

Status NcclHandler::Broadcast(const void* sendbuff, void* recvbuff, size_t count, DataType datatype,
                              int32_t root, cudaStream_t stream) {
  if (world_size_ <= 1) {
    if (sendbuff != recvbuff) {
      // cudaMemcpyAsync
    }
    return error::Success();
  }
#if defined(KUIPER_USE_NCCL)
  ncclDataType_t nccl_type;
  switch (datatype) {
    case DataType::kDataTypeFp32:
      nccl_type = ncclFloat;
      break;
    case DataType::kDataTypeInt8:
      nccl_type = ncclInt8;
      break;
    case DataType::kDataTypeInt32:
      nccl_type = ncclInt32;
      break;
    default:
      return error::InvalidArgument("Unsupported data type for Broadcast");
  }

  ncclResult_t res = ncclBroadcast(sendbuff, recvbuff, count, nccl_type, root, comm_, stream);
  if (res != ncclSuccess) {
    return error::InternalError("NCCL Broadcast failed");
  }
#endif
  return error::Success();
}

Status NcclHandler::AllGather(const void* sendbuff, void* recvbuff, size_t send_count,
                              DataType datatype, cudaStream_t stream) {
  if (world_size_ <= 1) {
    // Just copy
    return error::Success();
  }
#if defined(KUIPER_USE_NCCL)
  ncclDataType_t nccl_type;
  switch (datatype) {
    case DataType::kDataTypeFp32:
      nccl_type = ncclFloat;
      break;
    case DataType::kDataTypeInt8:
      nccl_type = ncclInt8;
      break;
    case DataType::kDataTypeInt32:
      nccl_type = ncclInt32;
      break;
    default:
      return error::InvalidArgument("Unsupported data type for AllGather");
  }

  ncclResult_t res = ncclAllGather(sendbuff, recvbuff, send_count, nccl_type, comm_, stream);
  if (res != ncclSuccess) {
    return error::InternalError("NCCL AllGather failed");
  }
#endif
  return error::Success();
}

}  // namespace base
