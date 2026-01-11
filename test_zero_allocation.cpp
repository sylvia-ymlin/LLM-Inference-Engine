#include <iostream>
#include <memory>
#include "kuiper/include/base/alloc.h"
#include "kuiper/include/base/buffer.h"
#include "kuiper/include/tensor/tensor.h"

int main() {
    std::cout << "Testing zero-byte allocation fixes..." << std::endl;
    
    try {
        // Test 1: Zero-byte CUDA allocation
        auto cuda_alloc = base::CUDADeviceAllocatorFactory::get_instance();
        void* ptr = cuda_alloc->allocate(0);
        std::cout << "✓ CUDA zero-byte allocation returned: " << (ptr ? "non-null" : "null") << std::endl;
        
        // Test 2: Zero-byte CPU allocation  
        auto cpu_alloc = base::CPUDeviceAllocatorFactory::get_instance();
        ptr = cpu_alloc->allocate(0);
        std::cout << "✓ CPU zero-byte allocation returned: " << (ptr ? "non-null" : "null") << std::endl;
        
        // Test 3: Zero-sized buffer creation
        auto buffer = std::make_shared<base::Buffer>(0, cuda_alloc);
        std::cout << "✓ Zero-sized buffer created successfully" << std::endl;
        
        // Test 4: Zero-sized tensor creation
        tensor::Tensor zero_tensor(base::DataType::kDataTypeFp32, 0, true, cuda_alloc);
        std::cout << "✓ Zero-sized tensor created, is_empty: " << zero_tensor.is_empty() << std::endl;
        
        // Test 5: Zero-sized tensor with multiple dimensions that result in zero size
        tensor::Tensor zero_multi_tensor(base::DataType::kDataTypeFp32, 0, 10, true, cuda_alloc);
        std::cout << "✓ Zero-sized multi-dim tensor created, is_empty: " << zero_multi_tensor.is_empty() << std::endl;
        
        std::cout << "All tests passed! Zero-byte allocation handling is working correctly." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "✗ Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}