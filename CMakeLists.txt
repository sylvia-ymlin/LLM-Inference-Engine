cmake_minimum_required(VERSION 3.16)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set the CUDA compiler path or name (only if CUDA is available)
if(EXISTS "/usr/local/cuda/bin/nvcc")
  set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
  project(llama_infer CXX CUDA)
  include(cmake/cuda.cmake)
else()
  project(llama_infer CXX)
  message(WARNING "CUDA not found, building CPU-only version")
endif()

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)

option(USE_NCCL "Enable NCCL for distributed inference" ON)
if(USE_NCCL)
  add_definitions(-DKUIPER_USE_NCCL)
endif()

option(USE_FLASH_ATTENTION "Enable FlashAttention-2 for optimized attention" ON)
if(USE_FLASH_ATTENTION)
  add_definitions(-DKUIPER_USE_FLASH_ATTENTION)
endif()

option(LLAMA3_SUPPORT OFF)
if (LLAMA3_SUPPORT)
  message(STATUS "LLAMA3 SUPPORT")
  add_definitions(-DLLAMA3_SUPPORT)
endif()
option(QWEN2_SUPPORT OFF)
if (QWEN2_SUPPORT)
  message(STATUS "QWEN2 SUPPORT")
  add_definitions(-DQWEN2_SUPPORT)
endif()

option(QWEN3_SUPPORT OFF)
if (QWEN3_SUPPORT)
  message(STATUS "QWEN3 SUPPORT")
  add_definitions(-DQWEN3_SUPPORT)
endif()
# ---- Add dependencies via CPM ----
# see https://github.com/TheLartians/CPM.cmake for more info
option(USE_CPM "Use CPM for dependency management" ON)

if(USE_CPM)
  # Use CPM to manage dependencies
  include(cmake/CPM.cmake)

  CPMAddPackage(
    NAME GTest
    GITHUB_REPOSITORY google/googletest
    VERSION 1.15.0
  )

  CPMAddPackage(
    NAME glog
    GITHUB_REPOSITORY google/glog
    VERSION 0.7.1
    OPTIONS "BUILD_TESTING Off"
  )

  CPMAddPackage(
    NAME Armadillo
    GITLAB_REPOSITORY conradsnicta/armadillo-code
    GIT_TAG 14.0.1
  )

  CPMAddPackage(
    NAME sentencepiece
    GITHUB_REPOSITORY google/sentencepiece
    VERSION 0.2.0
  )
  find_package(sentencepiece REQUIRED)

  if (LLAMA3_SUPPORT OR QWEN2_SUPPORT OR QWEN3_SUPPORT)
    CPMAddPackage(
        NAME absl
        GITHUB_REPOSITORY abseil/abseil-cpp
        GIT_TAG 20240722.0
        OPTIONS "BUILD_TESTING Off" "ABSL_PROPAGATE_CXX_STD ON" "ABSL_ENABLE_INSTALL ON"
    )
    CPMAddPackage(
        NAME re2
        GITHUB_REPOSITORY google/re2
        GIT_TAG 2024-07-02
    )
    CPMAddPackage(
        NAME nlohmann_json
        GITHUB_REPOSITORY nlohmann/json
        VERSION 3.11.3
    )
  endif()

else()
  # Fallback to system packages if CPM is disabled
  find_package(GTest REQUIRED)
  find_package(glog REQUIRED)
  find_package(Armadillo REQUIRED)
endif()

aux_source_directory(kuiper/source/tensor/ DIR_TENSOR)
aux_source_directory(kuiper/source/base/ DIR_BASE)
aux_source_directory(kuiper/source/op/ DIR_OP)
aux_source_directory(kuiper/source/model/ DIR_MODEL)
aux_source_directory(kuiper/source/op/kernels/cpu DIR_KERNEL_CPU)
aux_source_directory(kuiper/source/op/kernels/cuda DIR_KERNEL_CUDA)
aux_source_directory(kuiper/source/op/kernels/ DIR_KERNEL)
aux_source_directory(kuiper/source/sampler DIR_SAMPLE)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(llama SHARED ${DIR_TENSOR} ${DIR_BASE} ${DIR_OP} ${DIR_KERNEL} ${DIR_MODEL} ${DIR_KERNEL_CPU} ${DIR_KERNEL_CUDA} ${DIR_KERNEL} ${DIR_SAMPLE})
target_link_libraries(llama sentencepiece glog::glog gtest gtest_main pthread cudart armadillo)
if(USE_NCCL)
  target_link_libraries(llama nccl)
endif()
target_link_directories(llama PUBLIC ${CMAKE_CUDA_COMPILER_LIBRARY_ROOT}/lib64)

target_include_directories(llama PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(llama PUBLIC ${PROJECT_SOURCE_DIR}/kuiper/include)
target_include_directories(llama PUBLIC ${Armadillo_INCLUDE_DIR})
target_include_directories(llama PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

if(USE_CPM)
  # Add sentencepiece include directory
  target_include_directories(llama PUBLIC ${sentencepiece_SOURCE_DIR}/src)
endif()

# FlashAttention-2 integration (Simplified version)
if(USE_FLASH_ATTENTION AND USE_CPM)
  message(STATUS "FlashAttention support enabled (simplified implementation)")
  # Note: This uses a simplified FlashAttention-inspired kernel
  # For full FlashAttention-2, additional dependencies would be needed
endif()

set_target_properties(llama PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_subdirectory(test)
add_subdirectory(demo)
