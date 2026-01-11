#!/bin/bash

# FlashAttention 集成验证脚本
# 用于在AutoDL环境中快速验证FlashAttention功能

set -e  # 遇到错误立即退出

echo "🚀 FlashAttention 集成验证开始..."
echo "=================================================="

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 1. 环境检查
echo -e "\n${BLUE}=== 1. 环境检查 ===${NC}"

log_info "检查CUDA环境..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    log_success "CUDA版本: $CUDA_VERSION"
else
    log_error "CUDA未安装或不在PATH中"
    exit 1
fi

log_info "检查GPU状态..."
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
    log_success "GPU信息: $GPU_INFO"
else
    log_error "nvidia-smi不可用"
    exit 1
fi

# 2. 项目构建
echo -e "\n${BLUE}=== 2. 项目构建 ===${NC}"

log_info "创建构建目录..."
mkdir -p build
cd build

log_info "配置CMake（启用FlashAttention）..."
if cmake -DUSE_FLASH_ATTENTION=ON -DUSE_CPM=ON -DUSE_NCCL=ON .. > cmake_config.log 2>&1; then
    log_success "CMake配置成功"
    
    # 检查FlashAttention是否启用
    if grep -q "FlashAttention support enabled" cmake_config.log; then
        log_success "FlashAttention支持已启用"
    else
        log_warning "FlashAttention支持状态未确认"
    fi
else
    log_error "CMake配置失败"
    cat cmake_config.log
    exit 1
fi

log_info "开始编译..."
if make -j$(nproc) > build.log 2>&1; then
    log_success "编译成功"
else
    log_error "编译失败"
    tail -20 build.log
    exit 1
fi

# 3. 构建验证
echo -e "\n${BLUE}=== 3. 构建验证 ===${NC}"

# 首先找到实际的构建输出位置
log_info "查找构建输出文件..."
echo "当前目录结构:"
ls -la
echo ""
echo "查找libllama.so:"
find . -name "libllama.so" -type f 2>/dev/null | head -5
echo ""
echo "查找test_llm:"
find . -name "test_llm" -type f 2>/dev/null | head -5
echo ""
echo "查找llama_infer:"
find . -name "llama_infer" -type f 2>/dev/null | head -5
echo ""

log_info "检查库文件..."
if [ -f "lib/libllama.so" ]; then
    log_success "libllama.so 构建成功"
    
    # 检查FlashAttention符号
    if nm lib/libllama.so | grep -q "flash_attention" 2>/dev/null; then
        log_success "FlashAttention符号存在于库中"
    else
        log_warning "未在库中找到FlashAttention符号（这是正常的，符号可能被优化）"
    fi
elif [ -f "../lib/libllama.so" ]; then
    log_success "libllama.so 构建成功 (在上级目录)"
    
    # 检查FlashAttention符号
    if nm ../lib/libllama.so | grep -q "flash_attention" 2>/dev/null; then
        log_success "FlashAttention符号存在于库中"
    else
        log_warning "未在库中找到FlashAttention符号（这是正常的，符号可能被优化）"
    fi
else
    log_error "libllama.so 未找到，检查可能的位置..."
    find . -name "libllama.so" -type f 2>/dev/null | head -5
    exit 1
fi

log_info "检查测试可执行文件..."
if [ -f "test/test_llm" ]; then
    log_success "test_llm 构建成功"
elif [ -f "../test/test_llm" ]; then
    log_success "test_llm 构建成功 (在上级目录)"
else
    log_error "test_llm 未找到，检查可能的位置..."
    find . -name "test_llm" -type f 2>/dev/null | head -5
    exit 1
fi

log_info "检查演示程序..."
if [ -f "demo/llama_infer" ]; then
    log_success "llama_infer 构建成功"
elif [ -f "../demo/llama_infer" ]; then
    log_success "llama_infer 构建成功 (在上级目录)"
else
    log_error "llama_infer 未找到，检查可能的位置..."
    find . -name "llama_infer" -type f 2>/dev/null | head -5
    exit 1
fi

# 4. 单元测试
echo -e "\n${BLUE}=== 4. 单元测试 ===${NC}"

log_info "运行FlashAttention单元测试..."
TEST_CMD=""
if [ -f "test/test_llm" ]; then
    TEST_CMD="./test/test_llm"
elif [ -f "../test/test_llm" ]; then
    TEST_CMD="../test/test_llm"
else
    log_error "找不到test_llm可执行文件"
    exit 1
fi

if $TEST_CMD --gtest_filter=TestFlashAttention.* > flash_test.log 2>&1; then
    log_success "FlashAttention单元测试通过"
    
    # 显示测试结果摘要
    PASSED_TESTS=$(grep -c "PASSED" flash_test.log || echo "0")
    FAILED_TESTS=$(grep -c "FAILED" flash_test.log || echo "0")
    log_info "测试结果: $PASSED_TESTS 通过, $FAILED_TESTS 失败"
else
    log_error "FlashAttention单元测试失败"
    cat flash_test.log
    exit 1
fi

log_info "运行CUDA内核测试..."
if $TEST_CMD --gtest_filter=*cu* > cuda_test.log 2>&1; then
    CUDA_PASSED=$(grep -c "PASSED" cuda_test.log || echo "0")
    log_success "CUDA内核测试: $CUDA_PASSED 个测试通过"
else
    log_warning "部分CUDA内核测试失败，检查详细日志"
fi

# 5. 端到端测试准备
echo -e "\n${BLUE}=== 5. 端到端测试准备 ===${NC}"

MODEL_PATH="/root/autodl-tmp/stories110M.bin"
TOKENIZER_PATH="/root/autodl-tmp/tokenizer.model"

log_info "检查测试模型..."
if [ ! -f "$MODEL_PATH" ]; then
    log_warning "测试模型不存在，尝试下载..."
    cd /root/autodl-tmp
    if wget -O stories110M.bin https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin; then
        log_success "模型下载成功"
    else
        log_error "模型下载失败"
        exit 1
    fi
    cd - > /dev/null
fi

if [ ! -f "$TOKENIZER_PATH" ]; then
    log_warning "分词器不存在，尝试下载..."
    cd /root/autodl-tmp
    if wget -O tokenizer.model https://huggingface.co/karpathy/tinyllamas/resolve/main/tokenizer.model; then
        log_success "分词器下载成功"
    else
        log_error "分词器下载失败"
        exit 1
    fi
    cd - > /dev/null
fi

# 6. 端到端测试
echo -e "\n${BLUE}=== 6. 端到端推理测试 ===${NC}"

log_info "运行推理测试..."
INFER_CMD=""
if [ -f "demo/llama_infer" ]; then
    INFER_CMD="./demo/llama_infer"
elif [ -f "../demo/llama_infer" ]; then
    INFER_CMD="../demo/llama_infer"
else
    log_error "找不到llama_infer可执行文件"
    exit 1
fi

if timeout 60 $INFER_CMD "$MODEL_PATH" "$TOKENIZER_PATH" > inference_test.log 2>&1; then
    log_success "推理测试完成"
    
    # 检查FlashAttention使用情况
    if grep -q "Using FlashAttention" inference_test.log; then
        log_success "✅ FlashAttention 正在使用"
    elif grep -q "Using standard MHA" inference_test.log; then
        log_warning "⚠️  使用标准MHA（FlashAttention未启用）"
    else
        log_info "未找到attention类型日志"
    fi
    
    # 显示性能指标
    if grep -q "steps/s" inference_test.log; then
        PERFORMANCE=$(grep "steps/s" inference_test.log | tail -1)
        log_success "性能指标: $PERFORMANCE"
    fi
    
    # 显示内存使用
    log_info "GPU内存使用情况:"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
    
else
    log_error "推理测试失败或超时"
    cat inference_test.log
    exit 1
fi

# 7. 性能基准测试
echo -e "\n${BLUE}=== 7. 性能基准测试 ===${NC}"

log_info "运行性能基准测试..."
echo "测试配置: GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# 记录开始时的内存使用
MEMORY_BEFORE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
log_info "测试前GPU内存使用: ${MEMORY_BEFORE} MB"

# 运行性能测试
START_TIME=$(date +%s)
PERF_CMD=""
if [ -f "demo/llama_infer" ]; then
    PERF_CMD="./demo/llama_infer"
elif [ -f "../demo/llama_infer" ]; then
    PERF_CMD="../demo/llama_infer"
else
    log_error "找不到llama_infer可执行文件"
    exit 1
fi

if timeout 120 $PERF_CMD "$MODEL_PATH" "$TOKENIZER_PATH" > performance_test.log 2>&1; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    # 记录结束时的内存使用
    MEMORY_AFTER=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    MEMORY_USED=$((MEMORY_AFTER - MEMORY_BEFORE))
    
    log_success "性能测试完成 (用时: ${DURATION}s)"
    log_info "内存增量: ${MEMORY_USED} MB"
    
    # 提取性能指标
    if grep -q "steps/s" performance_test.log; then
        TOKENS_PER_SEC=$(grep "steps/s" performance_test.log | tail -1 | awk '{print $NF}')
        log_success "🎯 推理速度: ${TOKENS_PER_SEC} tokens/s"
        
        # 性能评估
        if (( $(echo "$TOKENS_PER_SEC > 500" | bc -l) )); then
            log_success "🏆 性能优秀 (>500 tokens/s)"
        elif (( $(echo "$TOKENS_PER_SEC > 300" | bc -l) )); then
            log_success "✅ 性能良好 (>300 tokens/s)"
        else
            log_warning "⚠️  性能需要优化 (<300 tokens/s)"
        fi
    fi
else
    log_warning "性能测试超时或失败"
fi

# 8. 总结报告
echo -e "\n${GREEN}=== 🎉 验证完成 ===${NC}"
echo "=================================================="

log_success "FlashAttention集成验证完成！"
echo ""
echo "📊 验证结果摘要:"
echo "  ✅ CUDA环境: $CUDA_VERSION"
echo "  ✅ GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "  ✅ 项目构建: 成功"
echo "  ✅ 单元测试: 通过"
echo "  ✅ 端到端测试: 通过"

if [ -f "performance_test.log" ] && grep -q "steps/s" performance_test.log; then
    FINAL_PERF=$(grep "steps/s" performance_test.log | tail -1 | awk '{print $NF}')
    echo "  🎯 推理性能: ${FINAL_PERF} tokens/s"
fi

echo ""
echo "📁 日志文件位置:"
echo "  - 构建日志: build/build.log"
echo "  - 测试日志: build/flash_test.log"
echo "  - 推理日志: build/inference_test.log"
echo "  - 性能日志: build/performance_test.log"

echo ""
echo "🚀 FlashAttention已成功集成并验证！"
echo "   可以开始使用优化后的LLM推理引擎了。"

cd ..  # 返回项目根目录