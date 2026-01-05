#include <cuda_runtime.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <chrono>
#include <thread>
#include "base/nccl_handler.h"

#ifdef KUIPER_USE_NCCL
TEST(TestNCCL, InitAndAllReduce) {
  // This test requires running with multiple processes (e.g. mpirun -n 2)
  // In a single process unit test run, this will likely fail or just test Init(rank=0, size=1)
  // correctness.

  // We can simulate a "single node single rank" case which effectively does nothing but verify API
  // calls don't crash. Real distributed testing requires an integration test script.

  // Mocking rank 0 of 1 for unit test safety if not under MPI
  int rank = 0;
  int world_size = 1;

  // In a real environment, we'd get these from env vars or MPI
  const char* env_rank = std::getenv("OMPI_COMM_WORLD_RANK");
  if (env_rank) rank = std::atoi(env_rank);

  const char* env_size = std::getenv("OMPI_COMM_WORLD_SIZE");
  if (env_size) world_size = std::atoi(env_size);

  LOG(INFO) << "Testing NCCL with Rank: " << rank << " World Size: " << world_size;

  auto handler = base::NcclHandler::get_instance();
  // Note: Init logic in our implementation currently requires manual ID exchange or single process
  // Our current implementation has a placeholder.
  // If we want this to pass in CI without GPUs, we need to be careful.

  // Check if CUDA is available
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    LOG(WARNING) << "No CUDA devices found. Skipping NCCL test.";
    return;
  }

  // This init might hang if we don't have a real ID exchange mechanism for >1 ranks
  if (world_size == 1) {
    handler->Init(rank, world_size);

    // Test Broadcast (should be no-op or self-copy)
    std::vector<float> data(10, 1.0f);
    void* d_data;
    cudaMalloc(&d_data, 10 * sizeof(float));
    cudaMemcpy(d_data, data.data(), 10 * sizeof(float), cudaMemcpyHostToDevice);

    handler->Broadcast(d_data, 10, rank);
    handler->AllReduce(d_data, d_data, 10);

    std::vector<float> data_out(10);
    cudaMemcpy(data_out.data(), d_data, 10 * sizeof(float), cudaMemcpyDeviceToHost);

    for (float f : data_out) {
      EXPECT_FLOAT_EQ(f, 1.0f);
    }

    cudaFree(d_data);
  } else {
    LOG(INFO) << "Skipping single-process unit test logic for multi-rank run.";
  }
}
#endif
