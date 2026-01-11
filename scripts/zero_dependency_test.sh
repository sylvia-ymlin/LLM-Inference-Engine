#!/bin/bash

# 零依赖性能对比测试 (纯bash实现)
# 作者: FlashAttention集成项目
# 日期: 2026-01-11

set -e

# 测试配置
ITERATIONS=5
MODEL_PATH="/root/autodl-tmp/stories110M.bin"
TOKENIZER_PATH="/root/autodl-tmp/tokenizer.model"
PROJECT_ROOT="/root/autodl-tmp/llm-inference-engine"

echo "🔬 零依赖性能对比测试开始..."
echo "=================================================="
echo "测试配置: $ITERATIONS 次迭代"
echo "=================================================="

cd "$PROJECT_ROOT"

# 纯bash浮点数计算函数
float_add() {
    local a=$1
    local b=$2
    awk "BEGIN {printf \"%.6f\", $a + $b}"
}

float_subtract() {
    local a=$1
    local b=$2
    awk "BEGIN {printf \"%.6f\", $a - $b}"
}

float_multiply() {
    local a=$1
    local b=$2
    awk "BEGIN {printf \"%.6f\", $a * $b}"
}

float_divide() {
    local a=$1
    local b=$2
    awk "BEGIN {printf \"%.6f\", $a / $b}"
}

float_sqrt() {
    local a=$1
    awk "BEGIN {printf \"%.6f\", sqrt($a)}"
}

float_compare() {
    local a=$1
    local op=$2
    local b=$3
    awk "BEGIN {print ($a $op $b) ? 1 : 0}"
}

# 计算平均值
calculate_mean() {
    local values=("$@")
    local sum=0
    local count=${#values[@]}
    
    for val in "${values[@]}"; do
        sum=$(float_add "$sum" "$val")
    done
    
    float_divide "$sum" "$count"
}

# 计算标准差
calculate_std() {
    local mean=$1
    shift
    local values=("$@")
    local count=${#values[@]}
    local variance_sum=0
    
    for val in "${values[@]}"; do
        local diff=$(float_subtract "$val" "$mean")
        local squared=$(float_multiply "$diff" "$diff")
        variance_sum=$(float_add "$variance_sum" "$squared")
    done
    
    local variance=$(float_divide "$variance_sum" "$count")
    float_sqrt "$variance"
}

# 单次测试函数
run_test() {
    local version=$1
    local iteration=$2
    
    echo "  运行 $version - 第 $iteration 次..."
    
    # 运行推理并提取性能数据
    local output_file="/tmp/test_output_${version}_${iteration}.log"
    timeout 60s ./build/demo/llama_infer "$MODEL_PATH" "$TOKENIZER_PATH" > "$output_file" 2>&1
    
    local steps_per_sec=$(grep "steps/s:" "$output_file" | tail -1 | sed 's/.*steps\/s:\([0-9.]*\).*/\1/')
    
    if [ ! -z "$steps_per_sec" ]; then
        echo "    ✓ $steps_per_sec steps/s"
        echo "$steps_per_sec"
        return 0
    else
        echo "    ✗ 测试失败"
        return 1
    fi
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
        if result=$(run_test "standard" "$i"); then
            results+=("$result")
            success_count=$((success_count + 1))
        fi
        sleep 2
    done
    
    if [ $success_count -gt 0 ]; then
        local mean=$(calculate_mean "${results[@]}")
        local std=$(calculate_std "$mean" "${results[@]}")
        echo "📊 标准MHA结果: $mean ± $std steps/s (n=$success_count)"
        
        # 保存结果
        echo "$mean" > /tmp/standard_mean.txt
        echo "$std" > /tmp/standard_std.txt
        echo "$success_count" > /tmp/standard_count.txt
        
        # 保存所有数据点
        printf "%s\n" "${results[@]}" > /tmp/standard_data.txt
    else
        echo "❌ 标准MHA测试全部失败"
        return 1
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
        if result=$(run_test "flashattention" "$i"); then
            results+=("$result")
            success_count=$((success_count + 1))
        fi
        sleep 2
    done
    
    if [ $success_count -gt 0 ]; then
        local mean=$(calculate_mean "${results[@]}")
        local std=$(calculate_std "$mean" "${results[@]}")
        echo "📊 FlashAttention结果: $mean ± $std steps/s (n=$success_count)"
        
        # 保存结果
        echo "$mean" > /tmp/flash_mean.txt
        echo "$std" > /tmp/flash_std.txt
        echo "$success_count" > /tmp/flash_count.txt
        
        # 保存所有数据点
        printf "%s\n" "${results[@]}" > /tmp/flash_data.txt
    else
        echo "❌ FlashAttention测试全部失败"
        return 1
    fi
}

# 比较结果
compare_results() {
    if [ -f /tmp/standard_mean.txt ] && [ -f /tmp/flash_mean.txt ]; then
        local standard_mean=$(cat /tmp/standard_mean.txt)
        local flash_mean=$(cat /tmp/flash_mean.txt)
        local standard_std=$(cat /tmp/standard_std.txt)
        local flash_std=$(cat /tmp/flash_std.txt)
        local standard_count=$(cat /tmp/standard_count.txt)
        local flash_count=$(cat /tmp/flash_count.txt)
        
        echo ""
        echo "=================================================="
        echo "📈 最终对比结果"
        echo "=================================================="
        echo "标准MHA版本:      $standard_mean ± $standard_std steps/s (n=$standard_count)"
        echo "FlashAttention版本: $flash_mean ± $flash_std steps/s (n=$flash_count)"
        
        local diff=$(float_subtract "$flash_mean" "$standard_mean")
        local diff_percent=$(float_multiply $(float_divide "$diff" "$standard_mean") "100")
        
        echo "绝对差异:         $diff steps/s"
        echo "相对差异:         $diff_percent%"
        
        # 简单的显著性判断 (基于2倍标准误差)
        local standard_se=$(float_divide "$standard_std" $(float_sqrt "$standard_count"))
        local flash_se=$(float_divide "$flash_std" $(float_sqrt "$flash_count"))
        local combined_se=$(float_sqrt $(float_add $(float_multiply "$standard_se" "$standard_se") $(float_multiply "$flash_se" "$flash_se")))
        local threshold=$(float_multiply "$combined_se" "2")
        
        local abs_diff=${diff#-}  # 去掉负号
        local is_significant=$(float_compare "$abs_diff" ">" "$threshold")
        
        if [ "$is_significant" -eq 1 ]; then
            echo "统计显著性:       可能显著 (|差异| > 2SE)"
        else
            echo "统计显著性:       不显著 (|差异| ≤ 2SE)"
        fi
        
        echo ""
        echo "=== 详细数据 ==="
        echo "标准MHA数据点:"
        cat /tmp/standard_data.txt | tr '\n' ' ' && echo
        echo "FlashAttention数据点:"
        cat /tmp/flash_data.txt | tr '\n' ' ' && echo
        
        echo ""
        if [ $(float_compare "$diff_percent" ">" "5") -eq 1 ]; then
            echo "🚀 FlashAttention版本明显更快 (+$diff_percent%)"
        elif [ $(float_compare "$diff_percent" "<" "-5") -eq 1 ]; then
            echo "⚡ 标准MHA版本明显更快 ($diff_percent%)"
        else
            echo "⚖️  两个版本性能基本相同 ($diff_percent%)"
        fi
        
        # 生成简单报告
        generate_simple_report "$standard_mean" "$standard_std" "$standard_count" \
                              "$flash_mean" "$flash_std" "$flash_count" \
                              "$diff" "$diff_percent" "$is_significant"
    else
        echo "❌ 无法比较结果，某些测试失败"
        return 1
    fi
}

# 生成简单报告
generate_simple_report() {
    local std_mean=$1 std_std=$2 std_count=$3
    local flash_mean=$4 flash_std=$5 flash_count=$6
    local diff=$7 diff_percent=$8 is_significant=$9
    
    local report_file="performance_results/zero_dependency_test_report.md"
    mkdir -p performance_results
    
    cat > "$report_file" << EOF
# 零依赖性能对比测试报告

## 测试配置

- **测试时间**: $(date)
- **迭代次数**: $ITERATIONS
- **模型**: $MODEL_PATH
- **测试环境**: AutoDL RTX 3090

## 测试结果

### 标准MHA版本
- **平均性能**: $std_mean steps/s
- **标准差**: $std_std
- **有效测试**: $std_count/$ITERATIONS
- **数据点**: $(cat /tmp/standard_data.txt | tr '\n' ' ')

### FlashAttention版本
- **平均性能**: $flash_mean steps/s
- **标准差**: $flash_std
- **有效测试**: $flash_count/$ITERATIONS
- **数据点**: $(cat /tmp/flash_data.txt | tr '\n' ' ')

## 对比分析

- **绝对差异**: $diff steps/s
- **相对差异**: $diff_percent%
- **统计显著性**: $([ "$is_significant" -eq 1 ] && echo "可能显著" || echo "不显著")

## 结论

$(if [ $(awk "BEGIN {print ($diff_percent > 5) ? 1 : 0}") -eq 1 ]; then
    echo "FlashAttention版本性能更优，提升 $diff_percent%"
elif [ $(awk "BEGIN {print ($diff_percent < -5) ? 1 : 0}") -eq 1 ]; then
    echo "标准MHA版本性能更优，FlashAttention版本下降 ${diff_percent#-}%"
else
    echo "两个版本性能基本相同，差异在误差范围内"
fi)

---
**生成时间**: $(date)
**测试脚本**: scripts/zero_dependency_test.sh
EOF

    echo "📝 报告已生成: $report_file"
}

# 清理函数
cleanup() {
    rm -f /tmp/standard_*.txt /tmp/flash_*.txt /tmp/test_output_*.log
}
trap cleanup EXIT

# 主流程
main() {
    echo "📊 收集系统信息..."
    echo "时间: $(date)"
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null || echo 'N/A')"
    echo ""
    
    # 随机化测试顺序
    if [ $((RANDOM % 2)) -eq 0 ]; then
        echo "🎲 随机顺序: 先测试标准MHA"
        test_standard && test_flashattention
    else
        echo "🎲 随机顺序: 先测试FlashAttention"
        test_flashattention && test_standard
    fi
    
    compare_results
    echo ""
    echo "🎉 测试完成！"
}

main "$@"