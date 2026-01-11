#!/bin/bash

# AutoDL服务器快速部署脚本
# 在AutoDL服务器上运行此脚本来拉取和验证FlashAttention集成

set -e

echo "🚀 AutoDL FlashAttention 部署开始..."
echo "=================================================="

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

log_info "当前目录: $(pwd)"
log_info "用户: $(whoami)"

# 检查CUDA
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    log_success "CUDA版本: $CUDA_VERSION"
else
    log_error "CUDA未找到，请检查环境"
    exit 1
fi

# 检查GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
    log_success "GPU信息: $GPU_INFO"
else
    log_error "nvidia-smi不可用"
    exit 1
fi

# 2. 项目拉取
echo -e "\n${BLUE}=== 2. 项目拉取 ===${NC}"

PROJECT_DIR="/root/autodl-tmp/llm-inference-engine"

if [ -d "$PROJECT_DIR" ]; then
    log_info "项目目录已存在，更新代码..."
    cd "$PROJECT_DIR"
    git pull origin main
    log_success "代码更新完成"
else
    log_info "克隆项目..."
    cd /root/autodl-tmp
    git clone https://github.com/sylvia-ymlin/LLM-Inference-Engine.git llm-inference-engine
    cd llm-inference-engine
    log_success "项目克隆完成"
fi

# 3. 检查FlashAttention文件
echo -e "\n${BLUE}=== 3. 检查FlashAttention文件 ===${NC}"

FLASH_FILES=(
    "kuiper/include/op/flash_attention.h"
    "kuiper/source/op/flash_attention.cpp"
    "kuiper/source/op/kernels/cuda/flash_attention_kernel.cu"
    "scripts/verify_flash_attention.sh"
    "reports/08_flash_attention_integration.md"
)

for file in "${FLASH_FILES[@]}"; do
    if [ -f "$file" ]; then
        log_success "✅ $file"
    else
        log_error "❌ $file 缺失"
        exit 1
    fi
done

# 4. 安装依赖
echo -e "\n${BLUE}=== 4. 安装依赖 ===${NC}"

log_info "更新系统包..."
apt update -qq

log_info "安装构建依赖..."
apt install -y build-essential cmake git wget curl bc > /dev/null 2>&1

log_success "依赖安装完成"

# 5. 下载测试模型
echo -e "\n${BLUE}=== 5. 下载测试模型 ===${NC}"

MODEL_PATH="/root/autodl-tmp/stories110M.bin"
TOKENIZER_PATH="/root/autodl-tmp/tokenizer.model"

if [ ! -f "$MODEL_PATH" ]; then
    log_info "下载测试模型..."
    cd /root/autodl-tmp
    wget -q --show-progress -O stories110M.bin https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin
    log_success "模型下载完成"
else
    log_success "测试模型已存在"
fi

if [ ! -f "$TOKENIZER_PATH" ]; then
    log_info "下载分词器..."
    cd /root/autodl-tmp
    wget -q --show-progress -O tokenizer.model https://huggingface.co/karpathy/tinyllamas/resolve/main/tokenizer.model
    log_success "分词器下载完成"
else
    log_success "分词器已存在"
fi

# 6. 运行验证脚本
echo -e "\n${BLUE}=== 6. 运行FlashAttention验证 ===${NC}"

cd "$PROJECT_DIR"

log_info "开始FlashAttention验证..."
if ./scripts/verify_flash_attention.sh; then
    log_success "🎉 FlashAttention验证成功！"
else
    log_error "FlashAttention验证失败"
    exit 1
fi

# 7. 完成总结
echo -e "\n${GREEN}=== 🎉 部署完成 ===${NC}"
echo "=================================================="

log_success "AutoDL FlashAttention部署成功！"
echo ""
echo "📁 项目位置: $PROJECT_DIR"
echo "🔧 可执行文件:"
echo "  - 测试程序: $PROJECT_DIR/build/test/test_llm"
echo "  - 推理程序: $PROJECT_DIR/build/demo/llama_infer"
echo ""
echo "🚀 快速测试命令:"
echo "  cd $PROJECT_DIR/build"
echo "  ./test/test_llm --gtest_filter=TestFlashAttention.*"
echo "  ./demo/llama_infer $MODEL_PATH $TOKENIZER_PATH"
echo ""
echo "📊 查看性能日志:"
echo "  cat $PROJECT_DIR/build/performance_test.log"
echo ""
echo "✅ FlashAttention已成功集成并验证！"