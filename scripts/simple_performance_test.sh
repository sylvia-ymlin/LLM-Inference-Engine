#!/bin/bash

# 简化版性能对比测试 (适用于受限环境)
# 作者: FlashAttention集成项目
# 日期: 2026-01-11

set -e

# 测试配置
ITERATIONS=5
MODEL_PATH="/root/autodl-tmp/stories110M.bin"
TOKENIZER_PATH="/root/autodl-tmp/tokenizer.model"
PROJECT_ROOT="/root/autodl-tmp/llm-inference-engine"

echo "🔬 简化性能对比测试开始..."
echo "=================================================="
echo "测试配置: $ITERATIONS 次迭代"
echo "=================================================="

cd "$PROJECT_ROOT"

# 单次测试函数
run_test() {
    local version=$1
    local iteration=$2
    
    echo "  运行 $version - 第 $iteration 次..."
    
    # 运行推理并提取性能数据
    local output=$(timeout 60s ./build/demo/llama_infer "$MODEL_PATH" "$TOKENIZER_PATH" 2>&1)
    local steps_per_sec=$(echo "$output" | grep "steps/s:" | tail -1 | sed 's/.*steps\/s:\([0-9.]*\).*/\1/')
    
    if [ ! -z "$steps_per_sec" ]; then
        echo "    ✓ $steps_per_sec steps/s"
        echo "$steps_per_sec"
        return 0
    else
        echo "    ✗ 测试失败"
        return 1
    fi
}

# 计算统计数据
calculate_stats() {
    local values=("$@")
    local sum=0
    local count=${#values[@]}
    
    # 计算平均值
    for val in "${values[@]}"; do
        sum=$(echo "$sum + $val" | bc -l)
    done
    local mean=$(echo "scale=2; $sum / $count" | bc -l)
    
    # 计算标准差
    local variance_sum=0
    for val in "${values[@]}"; do
        local diff=$(echo "$val - $mean" | bc -l)
        local squared=$(echo "$diff * $diff" | bc -l)
        variance_sum=$(echo "$variance_sum + $squared" | bc -l)
    done
    local variance=$(echo "scale=4; $variance_sum / $count" | bc -l)
    local std_dev=$(echo "scale=2; sqrt($variance)" | bc -l)
    
    echo "$mean $std_dev"
}

# 测试标准MHA版本
test_standard() {
    echo "🔨 构建标准MHA版本..."
    rm -rf build
    mkdir build
    cd build
    cmake -DUSE_FLASH_ATTENTION=OFF -DUSE_CPM=ON -DUSE_NCCL=ON .. > /dev/null 2>&1
    make -j4 > /dev/null 2>&1
    cd ..
    
    echo "🧪 测试标准MHA版本..."
    local results=()
    local success_count=0
    
    for i in $(seq 1 $ITERATIONS); do
        if result=$(run_test "标准MHA" "$i"); then
            results+=("$result")
            success_count=$((success_count + 1))
        fi
        sleep 2
    done
    
    if [ $success_count -gt 0 ]; then
        local stats=($(calculate_stats "${results[@]}"))
        echo "📊 标准MHA结果: ${stats[0]} ± ${stats[1]} steps/s (n=$success_count)"
        echo "${stats[0]}" > /tmp/standard_mean.txt
        echo "${stats[1]}" > /tmp/standard_std.txt
    else
        echo "❌ 标准MHA测试全部失败"
    fi
}

# 测试FlashAttention版本
test_flashattention() {
    echo "🔨 构建FlashAttention版本..."
    rm -rf build
    mkdir build
    cd build
    cmake -DUSE_FLASH_ATTENTION=ON -DUSE_CPM=ON -DUSE_NCCL=ON .. > /dev/null 2>&1
    make -j4 > /dev/null 2>&1
    cd ..
    
    echo "🧪 测试FlashAttention版本..."
    local results=()
    local success_count=0
    
    for i in $(seq 1 $ITERATIONS); do
        if result=$(run_test "FlashAttention" "$i"); then
            results+=("$result")
            success_count=$((success_count + 1))
        fi
        sleep 2
    done
    
    if [ $success_count -gt 0 ]; then
        local stats=($(calculate_stats "${results[@]}"))
        echo "📊 FlashAttention结果: ${stats[0]} ± ${stats[1]} steps/s (n=$success_count)"
        echo "${stats[0]}" > /tmp/flash_mean.txt
        echo "${stats[1]}" > /tmp/flash_std.txt
    else
        echo "❌ FlashAttention测试全部失败"
    fi
}

# 比较结果
compare_results() {
    if [ -f /tmp/standard_mean.txt ] && [ -f /tmp/flash_mean.txt ]; then
        local standard_mean=$(cat /tmp/standard_mean.txt)
        local flash_mean=$(cat /tmp/flash_mean.txt)
        local standard_std=$(cat /tmp/standard_std.txt)
        local flash_std=$(cat /tmp/flash_std.txt)
        
        echo ""
        echo "=================================================="
        echo "📈 最终对比结果"
        echo "=================================================="
        echo "标准MHA版本:      $standard_mean ± $standard_std steps/s"
        echo "FlashAttention版本: $flash_mean ± $flash_std steps/s"
        
        local diff=$(echo "$flash_mean - $standard_mean" | bc -l)
        local diff_percent=$(echo "scale=2; ($diff / $standard_mean) * 100" | bc -l)
        
        echo "绝对差异:         $diff steps/s"
        echo "相对差异:         $diff_percent%"
        
        # 简单的显著性判断 (基于标准差)
        local combined_std=$(echo "sqrt($standard_std^2 + $flash_std^2)" | bc -l)
        local abs_diff=$(echo "$diff" | sed 's/-//')
        local significance=$(echo "$abs_diff > (2 * $combined_std)" | bc -l)
        
        if [ "$significance" -eq 1 ]; then
            echo "统计显著性:       可能显著 (|差异| > 2σ)"
        else
            echo "统计显著性:       不显著 (|差异| ≤ 2σ)"
        fi
        
        echo ""
        if [ $(echo "$diff_percent > 5" | bc -l) -eq 1 ]; then
            echo "🚀 FlashAttention版本明显更快"
        elif [ $(echo "$diff_percent < -5" | bc -l) -eq 1 ]; then
            echo "⚡ 标准MHA版本明显更快"
        else
            echo "⚖️  两个版本性能基本相同"
        fi
    else
        echo "❌ 无法比较结果，某些测试失败"
    fi
}

# 清理函数
cleanup() {
    rm -f /tmp/standard_mean.txt /tmp/flash_mean.txt /tmp/standard_std.txt /tmp/flash_std.txt
}
trap cleanup EXIT

# 主流程
main() {
    # 随机化测试顺序
    if [ $((RANDOM % 2)) -eq 0 ]; then
        echo "🎲 随机顺序: 先测试标准MHA"
        test_standard
        test_flashattention
    else
        echo "🎲 随机顺序: 先测试FlashAttention"
        test_flashattention
        test_standard
    fi
    
    compare_results
    echo ""
    echo "🎉 测试完成！"
}

# 检查基本依赖
if ! command -v bc >/dev/null 2>&1; then
    echo "❌ 需要安装 bc: apt-get install bc"
    exit 1
fi

main "$@"