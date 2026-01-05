#ifndef BLAS_HELPER_H
#define BLAS_HELPER_H
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
namespace kernel {
struct CudaConfig {
  cudaStream_t stream = nullptr;
  cudaStream_t comm_stream = nullptr;
  ~CudaConfig() {
    if (stream) {
      cudaStreamDestroy(stream);
    }
    if (comm_stream) {
      cudaStreamDestroy(comm_stream);
    }
  }
};
}  // namespace kernel
#endif  // BLAS_HELPER_H
