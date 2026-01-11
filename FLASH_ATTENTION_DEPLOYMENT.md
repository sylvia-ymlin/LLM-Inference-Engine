# FlashAttention AutoDL 部署指南

## 环境准备

### 1. AutoDL实例配置
推荐配置：
- **GPU**: RTX 3090 / RTX 4090 / A100 (24GB+ VRAM)
- **CUDA**: 11.8+ 或 12.x
- **系统**: Ubuntu 20.04/22.04
- **内存**: 32GB+ RAM

### 2. 依赖安装
```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装基础依赖
sudo apt install -y build-essential cmake git wget curl

# 验证CUDA环境
nvcc --version
nvidia-smi

# 安装Python依赖（如果需要）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 项目构建

### 1. 克隆和准备
```bash
# 进入工作目录
cd /root/autodl-tmp

# 如果项目已存在，更新代码
cd llm-inference-engine
git pull origin main

# 或者重新克隆
# git clone <your-repo-url> llm-inference-engine
# cd llm-inference-engine
```

### 2. 构建FlashAttention版本
```bash
# 创建构建目录
mkdir -p build && cd build

# 配置CMake（启用FlashAttention）
cmake -DUSE_FLASH_ATTENTION=ON -DUSE_CPM=ON -DUSE_NCCL=ON ..

# 编译（使用所有CPU核心）
make -j$(nproc)

# 验证构建成功
ls -la lib/
ls -la test/test_llm
ls -la demo/
```

### 3. 构建状态检查
```bash
# 检查FlashAttention编译状态
grep -r "KUIPER_USE_FLASH_ATTENTION" build/ || echo "FlashAttention not enabled"

# 检查库文件
ldd lib/libllama.so | grep -E "(cuda|nccl)"

# 检查可执行文件
ldd test/test_llm | grep -E "(cuda|nccl)"
```

## 测试验证

### 1. 单元测试
```bash
# 运行FlashAttention单元测试
./test/test_llm --gtest_filter=TestFlashAttention.*

# 运行所有CUDA相关测试
./test/test_llm --gtest_filter=*cu*

# 运行NCCL测试
./test/test_llm --gtest_filter=TestNCCL.*
```

### 2. 端到端测试
```bash
# 下载测试模型（如果还没有）
cd /root/autodl-tmp
wget -O stories110M.bin https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin
wget -O tokenizer.model https://huggingface.co/karpathy/tinyllamas/resolve/main/tokenizer.model

# 运行推理测试（标准MHA）
cd llm-inference-engine
./demo/llama_infer /root/autodl-tmp/stories110M.bin /root/autodl-tmp/tokenizer.model

# 检查日志中的FlashAttention使用情况
# 应该看到 "Using FlashAttention for layer X" 或 "Using standard MHA for layer X"
```

### 3. 性能基准测试
```bash
# 创建性能测试脚本
cat > benchmark_flash_attention.sh << 'EOF'
#!/bin/bash

echo "=== FlashAttention Performance Benchmark ==="
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits

echo -e "\n=== Memory Usage Before Test ==="
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits

echo -e "\n=== Running Inference Test ==="
time ./demo/llama_infer /root/autodl-tmp/stories110M.bin /root/autodl-tmp/tokenizer.model

echo -e "\n=== Memory Usage After Test ==="
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits

echo -e "\n=== CUDA Kernel Performance ==="
./test/test_llm --gtest_filter=TestFlashAttention.FlashAttentionForwardCUDA
EOF

chmod +x benchmark_flash_attention.sh
./benchmark_flash_attention.sh
```

## 性能对比测试

### 1. 创建对比测试脚本
```bash
cat > compare_attention.sh << 'EOF'
#!/bin/bash

echo "=== Attention Implementation Comparison ==="

# 测试标准MHA
echo "Testing Standard MHA..."
# 临时禁用FlashAttention（通过环境变量或重新编译）
export KUIPER_DISABLE_FLASH_ATTENTION=1
time ./demo/llama_infer /root/autodl-tmp/stories110M.bin /root/autodl-tmp/tokenizer.model > standard_mha.log 2>&1

# 测试FlashAttention
echo "Testing FlashAttention..."
unset KUIPER_DISABLE_FLASH_ATTENTION
time ./demo/llama_infer /root/autodl-tmp/stories110M.bin /root/autodl-tmp/tokenizer.model > flash_attention.log 2>&1

echo "=== Results Comparison ==="
echo "Standard MHA log:"
tail -5 standard_mha.log
echo -e "\nFlashAttention log:"
tail -5 flash_attention.log

echo -e "\n=== Performance Summary ==="
grep "steps/s" standard_mha.log flash_attention.log
EOF

chmod +x compare_attention.sh
./compare_attention.sh
```

### 2. 内存使用分析
```bash
# 创建内存监控脚本
cat > monitor_memory.sh << 'EOF'
#!/bin/bash

echo "=== Memory Usage Monitoring ==="

# 启动内存监控
nvidia-smi --query-gpu=timestamp,memory.used,memory.free --format=csv -l 1 > memory_usage.csv &
MONITOR_PID=$!

# 运行推理
echo "Starting inference..."
./demo/llama_infer /root/autodl-tmp/stories110M.bin /root/autodl-tmp/tokenizer.model

# 停止监控
kill $MONITOR_PID

# 分析结果
echo -e "\n=== Memory Usage Analysis ==="
echo "Peak memory usage:"
tail -n +2 memory_usage.csv | cut -d',' -f2 | sort -n | tail -1

echo "Average memory usage:"
tail -n +2 memory_usage.csv | cut -d',' -f2 | awk '{sum+=$1; count++} END {print sum/count}'
EOF

chmod +x monitor_memory.sh
./monitor_memory.sh
```

## 调试和故障排除

### 1. 编译问题
```bash
# 检查CUDA版本兼容性
nvcc --version
cat /usr/local/cuda/version.txt

# 检查CMake配置
cd build
cmake -LA | grep -E "(CUDA|FLASH|NCCL)"

# 重新构建（清理后）
make clean
make -j$(nproc) VERBOSE=1
```

### 2. 运行时问题
```bash
# 检查CUDA运行时
./test/test_llm --gtest_filter=TestCUDA.*

# 检查内存分配
export CUDA_LAUNCH_BLOCKING=1
./test/test_llm --gtest_filter=TestFlashAttention.*

# 启用详细日志
export GLOG_v=2
./demo/llama_infer /root/autodl-tmp/stories110M.bin /root/autodl-tmp/tokenizer.model
```

### 3. 性能问题
```bash
# 使用nsys进行性能分析
nsys profile --stats=true -o flash_attention_profile ./demo/llama_infer /root/autodl-tmp/stories110M.bin /root/autodl-tmp/tokenizer.model

# 查看分析结果
nsys stats flash_attention_profile.nsys-rep
```

## 预期结果

### 1. 编译成功标志
- ✅ `libllama.so` 包含FlashAttention符号
- ✅ 测试可执行文件链接成功
- ✅ CMake配置显示 `KUIPER_USE_FLASH_ATTENTION=ON`

### 2. 运行成功标志
- ✅ 单元测试全部通过
- ✅ 端到端推理正常运行
- ✅ 日志显示 "Using FlashAttention for layer X"
- ✅ 性能提升（特别是长序列）

### 3. 性能指标
- **内存使用**: 相比标准MHA减少20-50%
- **推理速度**: 110M模型应达到600+ tokens/s
- **数值精度**: 与标准MHA输出误差 < 1e-5

## 问题报告

如果遇到问题，请收集以下信息：

```bash
# 系统信息
uname -a
nvidia-smi
nvcc --version

# 构建信息
cd build
cmake -LA | grep -E "(CUDA|FLASH|NCCL)"
make --version

# 运行时信息
ldd lib/libllama.so
./test/test_llm --gtest_list_tests | grep Flash

# 错误日志
export GLOG_v=3
./test/test_llm --gtest_filter=TestFlashAttention.* 2>&1 | tee flash_attention_debug.log
```

---

**部署时间**: 预计15-30分钟  
**测试时间**: 预计10-15分钟  
**总用时**: 约30-45分钟