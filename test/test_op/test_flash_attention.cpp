#include <gtest/gtest.h>
#include <glog/logging.h>
#include "op/flash_attention.h"
#include "base/alloc.h"

class TestFlashAttention : public ::testing::Test {
 protected:
  void SetUp() override {
    // Google Logging may already be initialized by main test runner
    // Only initialize if not already done
    if (!google::IsGoogleLoggingInitialized()) {
      google::InitGoogleLogging("TestFlashAttention");
    }
  }

  void TearDown() override {
    // Don't shutdown logging as it may be used by other tests
  }
};

TEST_F(TestFlashAttention, CreateFlashAttentionLayer) {
  const int32_t layer_index = 0;
  const int32_t kv_mul = 1;
  const int32_t kv_dim = 512;
  const int32_t seq_len = 128;
  const int32_t head_num = 8;
  const int32_t head_size = 64;

  auto flash_attn = std::make_shared<op::FlashAttention>(
      base::DeviceType::kDeviceCPU, layer_index, kv_mul, kv_dim, 
      seq_len, head_num, head_size);

  ASSERT_NE(flash_attn, nullptr);
  EXPECT_EQ(flash_attn->input_size(), 3);  // Q, K, V
  EXPECT_EQ(flash_attn->output_size(), 1); // Output
}

TEST_F(TestFlashAttention, FlashAttentionForwardCPU) {
  const int32_t batch_size = 1;
  const int32_t seq_len = 4;
  const int32_t head_num = 2;
  const int32_t head_size = 8;
  const int32_t kv_mul = 1;
  const int32_t kv_dim = head_num * head_size;

  auto allocator = base::CPUDeviceAllocatorFactory::get_instance();

  // Create FlashAttention layer
  auto flash_attn = std::make_shared<op::FlashAttention>(
      base::DeviceType::kDeviceCPU, 0, kv_mul, kv_dim, seq_len, head_num, head_size);

  // Create input tensors with proper initialization
  tensor::Tensor query(base::DataType::kDataTypeFp32, batch_size, seq_len, head_num, head_size, true, allocator);
  tensor::Tensor key(base::DataType::kDataTypeFp32, batch_size, seq_len, head_num, head_size, true, allocator);
  tensor::Tensor value(base::DataType::kDataTypeFp32, batch_size, seq_len, head_num, head_size, true, allocator);
  tensor::Tensor output(base::DataType::kDataTypeFp32, batch_size, seq_len, head_num, head_size, true, allocator);

  // Ensure tensors are properly allocated
  ASSERT_FALSE(query.is_empty());
  ASSERT_FALSE(key.is_empty());
  ASSERT_FALSE(value.is_empty());
  ASSERT_FALSE(output.is_empty());

  // Initialize with simple test data
  float* q_ptr = query.ptr<float>();
  float* k_ptr = key.ptr<float>();
  float* v_ptr = value.ptr<float>();
  
  for (int i = 0; i < query.size(); ++i) {
    q_ptr[i] = 0.1f * (i % 10);
    k_ptr[i] = 0.1f * ((i + 1) % 10);
    v_ptr[i] = 0.1f * ((i + 2) % 10);
  }

  // Set inputs and outputs
  flash_attn->set_input(0, query);
  flash_attn->set_input(1, key);
  flash_attn->set_input(2, value);
  flash_attn->set_output(0, output);

  flash_attn->set_pos(seq_len - 1);

  // Forward pass - this will fallback to standard MHA on CPU
  auto status = flash_attn->forward();
  EXPECT_TRUE(status);

  // Check output is not all zeros
  float* out_ptr = output.ptr<float>();
  bool has_non_zero = false;
  for (int i = 0; i < output.size(); ++i) {
    if (std::abs(out_ptr[i]) > 1e-6) {
      has_non_zero = true;
      break;
    }
  }
  EXPECT_TRUE(has_non_zero);
}

#ifdef KUIPER_USE_FLASH_ATTENTION
TEST_F(TestFlashAttention, FlashAttentionForwardCUDA) {
  const int32_t batch_size = 1;
  const int32_t seq_len = 4;
  const int32_t head_num = 2;
  const int32_t head_size = 8;
  const int32_t kv_mul = 1;
  const int32_t kv_dim = head_num * head_size;

  auto allocator = base::CUDADeviceAllocatorFactory::get_instance();
  auto cuda_config = std::make_shared<kernel::CudaConfig>();

  // Create FlashAttention layer
  auto flash_attn = std::make_shared<op::FlashAttention>(
      base::DeviceType::kDeviceCUDA, 0, kv_mul, kv_dim, seq_len, head_num, head_size);
  flash_attn->set_cuda_config(cuda_config);

  // Create input tensors
  tensor::Tensor query(base::DataType::kDataTypeFp32, batch_size, seq_len, head_num, head_size, true, allocator);
  tensor::Tensor key(base::DataType::kDataTypeFp32, batch_size, seq_len, head_num, head_size, true, allocator);
  tensor::Tensor value(base::DataType::kDataTypeFp32, batch_size, seq_len, head_num, head_size, true, allocator);
  tensor::Tensor output(base::DataType::kDataTypeFp32, batch_size, seq_len, head_num, head_size, true, allocator);

  // Initialize with test data on CPU first
  std::vector<float> q_data(query.size(), 0.1f);
  std::vector<float> k_data(key.size(), 0.2f);
  std::vector<float> v_data(value.size(), 0.3f);

  // Copy to GPU
  query.to_cuda();
  key.to_cuda();
  value.to_cuda();
  output.to_cuda();

  cudaMemcpy(query.ptr<float>(), q_data.data(), query.byte_size(), cudaMemcpyHostToDevice);
  cudaMemcpy(key.ptr<float>(), k_data.data(), key.byte_size(), cudaMemcpyHostToDevice);
  cudaMemcpy(value.ptr<float>(), v_data.data(), value.byte_size(), cudaMemcpyHostToDevice);

  // Set inputs and outputs
  flash_attn->set_input(0, query);
  flash_attn->set_input(1, key);
  flash_attn->set_input(2, value);
  flash_attn->set_output(0, output);

  flash_attn->set_pos(seq_len - 1);

  // Forward pass
  auto status = flash_attn->forward();
  EXPECT_TRUE(status);

  // Copy result back to CPU for verification
  std::vector<float> output_data(output.size());
  cudaMemcpy(output_data.data(), output.ptr<float>(), output.byte_size(), cudaMemcpyDeviceToHost);

  // Check output is not all zeros
  bool has_non_zero = false;
  for (float val : output_data) {
    if (std::abs(val) > 1e-6) {
      has_non_zero = true;
      break;
    }
  }
  EXPECT_TRUE(has_non_zero);
}
#endif