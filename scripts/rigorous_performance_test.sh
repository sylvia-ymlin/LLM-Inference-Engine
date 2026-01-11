#!/bin/bash

# 严格的FlashAttention vs 标准MHA性能对比测试
# 作者: FlashAttention集成项目
# 日期: 2026-01-11

set -e

# 测试配置
ITERATIONS=10
MODEL_PATH="/root/autodl-tmp/stories110M.bin"
TOKENIZER_PATH="/root/autodl-tmp/tokenizer.model"
PROJECT_ROOT="/root/autodl-tmp/llm-inference-engine"
RESULTS_DIR="$PROJECT_ROOT/performance_results"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔬 严格性能对比测试开始...${NC}"
echo "=================================================="
echo "测试配置:"
echo "- 迭代次数: $ITERATIONS"
echo "- 模型: $MODEL_PATH"
echo "- 分词器: $TOKENIZER_PATH"
echo "- 结果目录: $RESULTS_DIR"
echo "=================================================="

# 创建结果目录
mkdir -p "$RESULTS_DIR"
cd "$PROJECT_ROOT"

# 清理函数
cleanup() {
    echo -e "${YELLOW}清理临时文件...${NC}"
    rm -f /tmp/test_output_*.log
}
trap cleanup EXIT

# 系统信息收集
collect_system_info() {
    echo -e "${BLUE}📊 收集系统信息...${NC}"
    {
        echo "=== 系统信息 ==="
        echo "时间: $(date)"
        echo "主机名: $(hostname)"
        echo "CPU信息:"
        cat /proc/cpuinfo | grep "model name" | head -1
        echo "内存信息:"
        free -h
        echo "GPU信息:"
        nvidia-smi --query-gpu=name,memory.total,memory.used,temperature.gpu,utilization.gpu --format=csv,noheader,nounits
        echo "CUDA版本:"
        nvcc --version | grep "release"
        echo ""
    } > "$RESULTS_DIR/system_info.txt"
}

# 预热GPU
warmup_gpu() {
    echo -e "${YELLOW}🔥 GPU预热...${NC}"
    nvidia-smi -pm 1  # 设置持久模式
    nvidia-smi -ac 1215,1410  # 设置最大时钟频率 (如果支持)
    
    # 运行一次推理预热
    ./build/demo/llama_infer "$MODEL_PATH" "$TOKENIZER_PATH" > /tmp/warmup.log 2>&1 || true
    sleep 2
}

# 单次测试函数
run_single_test() {
    local test_name=$1
    local output_file=$2
    local iteration=$3
    
    echo -e "${BLUE}  运行 $test_name - 第 $iteration 次...${NC}"
    
    # 清理GPU内存
    nvidia-smi --gpu-reset-ecc=0 > /dev/null 2>&1 || true
    sleep 1
    
    # 记录开始时间和GPU状态
    local start_time=$(date +%s.%N)
    local gpu_before=$(nvidia-smi --query-gpu=memory.used,temperature.gpu,utilization.gpu --format=csv,noheader,nounits)
    
    # 运行推理测试
    timeout 60s ./build/demo/llama_infer "$MODEL_PATH" "$TOKENIZER_PATH" > "$output_file" 2>&1
    local exit_code=$?
    
    # 记录结束时间和GPU状态
    local end_time=$(date +%s.%N)
    local gpu_after=$(nvidia-smi --query-gpu=memory.used,temperature.gpu,utilization.gpu --format=csv,noheader,nounits)
    
    # 计算运行时间
    local duration=$(echo "$end_time - $start_time" | bc -l)
    
    # 提取性能数据
    local steps_per_sec=$(grep "steps/s:" "$output_file" | tail -1 | sed 's/.*steps\/s:\([0-9.]*\).*/\1/')
    
    if [ $exit_code -eq 0 ] && [ ! -z "$steps_per_sec" ]; then
        echo "$iteration,$steps_per_sec,$duration,$gpu_before,$gpu_after" >> "$RESULTS_DIR/${test_name}_raw_results.csv"
        echo -e "${GREEN}    ✓ 成功: ${steps_per_sec} steps/s (${duration}s)${NC}"
        return 0
    else
        echo -e "${RED}    ✗ 失败 (退出码: $exit_code)${NC}"
        echo "$iteration,FAILED,$duration,$gpu_before,$gpu_after" >> "$RESULTS_DIR/${test_name}_raw_results.csv"
        return 1
    fi
}

# 构建标准版本
build_standard() {
    echo -e "${BLUE}🔨 构建标准MHA版本...${NC}"
    rm -rf build
    mkdir build
    cd build
    cmake -DUSE_FLASH_ATTENTION=OFF -DUSE_CPM=ON -DUSE_NCCL=ON .. > /dev/null 2>&1
    make -j$(nproc) > /dev/null 2>&1
    cd ..
    echo -e "${GREEN}✓ 标准版本构建完成${NC}"
}

# 构建FlashAttention版本
build_flashattention() {
    echo -e "${BLUE}🔨 构建FlashAttention版本...${NC}"
    rm -rf build
    mkdir build
    cd build
    cmake -DUSE_FLASH_ATTENTION=ON -DUSE_CPM=ON -DUSE_NCCL=ON .. > /dev/null 2>&1
    make -j$(nproc) > /dev/null 2>&1
    cd ..
    echo -e "${GREEN}✓ FlashAttention版本构建完成${NC}"
}

# 运行测试套件
run_test_suite() {
    local test_name=$1
    local results_file="$RESULTS_DIR/${test_name}_raw_results.csv"
    
    echo -e "${BLUE}🧪 开始 $test_name 测试套件...${NC}"
    
    # 创建CSV头部
    echo "iteration,steps_per_sec,duration,gpu_before,gpu_after" > "$results_file"
    
    local success_count=0
    local total_steps=0
    local total_duration=0
    
    for i in $(seq 1 $ITERATIONS); do
        if run_single_test "$test_name" "/tmp/test_output_${test_name}_${i}.log" "$i"; then
            success_count=$((success_count + 1))
            local steps=$(grep "steps/s:" "/tmp/test_output_${test_name}_${i}.log" | tail -1 | sed 's/.*steps\/s:\([0-9.]*\).*/\1/')
            total_steps=$(echo "$total_steps + $steps" | bc -l)
        fi
        
        # 测试间隔，让系统稳定
        sleep 3
    done
    
    echo -e "${BLUE}📊 $test_name 测试完成:${NC}"
    echo -e "  成功: $success_count/$ITERATIONS"
    
    if [ $success_count -gt 0 ]; then
        local avg_steps=$(echo "scale=2; $total_steps / $success_count" | bc -l)
        echo -e "  平均性能: ${GREEN}$avg_steps steps/s${NC}"
    fi
    echo ""
}

# 统计分析
analyze_results() {
    echo -e "${BLUE}📈 统计分析...${NC}"
    
    python3 << 'EOF'
import pandas as pd
import numpy as np
from scipy import stats
import os

results_dir = os.environ['RESULTS_DIR']

try:
    # 读取数据
    standard_df = pd.read_csv(f'{results_dir}/standard_mha_raw_results.csv')
    flash_df = pd.read_csv(f'{results_dir}/flashattention_raw_results.csv')
    
    # 过滤成功的测试
    standard_valid = standard_df[standard_df['steps_per_sec'] != 'FAILED']['steps_per_sec'].astype(float)
    flash_valid = flash_df[flash_df['steps_per_sec'] != 'FAILED']['steps_per_sec'].astype(float)
    
    print("=== 详细统计分析 ===")
    print(f"标准MHA版本:")
    print(f"  有效测试: {len(standard_valid)}")
    print(f"  平均值: {standard_valid.mean():.2f} steps/s")
    print(f"  标准差: {standard_valid.std():.2f}")
    print(f"  最小值: {standard_valid.min():.2f}")
    print(f"  最大值: {standard_valid.max():.2f}")
    print(f"  中位数: {standard_valid.median():.2f}")
    
    print(f"\nFlashAttention版本:")
    print(f"  有效测试: {len(flash_valid)}")
    print(f"  平均值: {flash_valid.mean():.2f} steps/s")
    print(f"  标准差: {flash_valid.std():.2f}")
    print(f"  最小值: {flash_valid.min():.2f}")
    print(f"  最大值: {flash_valid.max():.2f}")
    print(f"  中位数: {flash_valid.median():.2f}")
    
    # 统计显著性检验
    if len(standard_valid) > 1 and len(flash_valid) > 1:
        t_stat, p_value = stats.ttest_ind(standard_valid, flash_valid)
        print(f"\n=== 统计显著性检验 ===")
        print(f"t统计量: {t_stat:.4f}")
        print(f"p值: {p_value:.4f}")
        
        if p_value < 0.05:
            print("结论: 性能差异统计显著 (p < 0.05)")
        else:
            print("结论: 性能差异不显著 (p >= 0.05)")
        
        # 效应大小 (Cohen's d)
        pooled_std = np.sqrt(((len(standard_valid)-1)*standard_valid.var() + (len(flash_valid)-1)*flash_valid.var()) / (len(standard_valid)+len(flash_valid)-2))
        cohens_d = (standard_valid.mean() - flash_valid.mean()) / pooled_std
        print(f"效应大小 (Cohen's d): {cohens_d:.4f}")
        
        if abs(cohens_d) < 0.2:
            print("效应大小: 微小")
        elif abs(cohens_d) < 0.5:
            print("效应大小: 小")
        elif abs(cohens_d) < 0.8:
            print("效应大小: 中等")
        else:
            print("效应大小: 大")
    
    # 性能差异
    if len(standard_valid) > 0 and len(flash_valid) > 0:
        diff_percent = ((flash_valid.mean() - standard_valid.mean()) / standard_valid.mean()) * 100
        print(f"\n=== 性能对比 ===")
        print(f"性能差异: {diff_percent:+.2f}%")
        
        if abs(diff_percent) < 1:
            print("结论: 性能基本相同")
        elif diff_percent > 0:
            print("结论: FlashAttention版本更快")
        else:
            print("结论: 标准MHA版本更快")

except Exception as e:
    print(f"分析出错: {e}")
    print("请检查数据文件是否存在且格式正确")
EOF
}

# 生成报告
generate_report() {
    echo -e "${BLUE}📝 生成测试报告...${NC}"
    
    local report_file="$RESULTS_DIR/performance_comparison_report.md"
    
    cat > "$report_file" << EOF
# 严格性能对比测试报告

## 测试配置

- **测试时间**: $(date)
- **迭代次数**: $ITERATIONS
- **模型**: $MODEL_PATH
- **测试环境**: AutoDL RTX 3090

## 系统信息

\`\`\`
$(cat "$RESULTS_DIR/system_info.txt")
\`\`\`

## 测试方法

1. **严格控制变量**: 相同的系统环境、模型、输入
2. **多次重复**: 每个版本运行 $ITERATIONS 次
3. **随机化**: 测试顺序随机化
4. **预热**: GPU预热避免冷启动影响
5. **统计分析**: 使用t检验和效应大小分析

## 原始数据

### 标准MHA版本
\`\`\`
$(cat "$RESULTS_DIR/standard_mha_raw_results.csv" 2>/dev/null || echo "数据文件不存在")
\`\`\`

### FlashAttention版本
\`\`\`
$(cat "$RESULTS_DIR/flashattention_raw_results.csv" 2>/dev/null || echo "数据文件不存在")
\`\`\`

## 结论

基于严格的统计分析，本测试提供了FlashAttention集成对性能影响的客观评估。

---
**生成时间**: $(date)
**测试脚本**: scripts/rigorous_performance_test.sh
EOF

    echo -e "${GREEN}✓ 报告已生成: $report_file${NC}"
}

# 主测试流程
main() {
    collect_system_info
    warmup_gpu
    
    # 随机化测试顺序
    if [ $((RANDOM % 2)) -eq 0 ]; then
        echo -e "${YELLOW}🎲 随机顺序: 先测试标准MHA，后测试FlashAttention${NC}"
        
        build_standard
        run_test_suite "standard_mha"
        
        build_flashattention
        run_test_suite "flashattention"
    else
        echo -e "${YELLOW}🎲 随机顺序: 先测试FlashAttention，后测试标准MHA${NC}"
        
        build_flashattention
        run_test_suite "flashattention"
        
        build_standard
        run_test_suite "standard_mha"
    fi
    
    analyze_results
    generate_report
    
    echo -e "${GREEN}🎉 严格性能测试完成！${NC}"
    echo -e "结果保存在: ${BLUE}$RESULTS_DIR${NC}"
}

# 检查依赖
check_dependencies() {
    local missing_deps=()
    
    command -v bc >/dev/null 2>&1 || missing_deps+=("bc")
    command -v python3 >/dev/null 2>&1 || missing_deps+=("python3")
    command -v nvidia-smi >/dev/null 2>&1 || missing_deps+=("nvidia-smi")
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        echo -e "${RED}❌ 缺少依赖: ${missing_deps[*]}${NC}"
        echo "请安装缺少的依赖后重新运行"
        exit 1
    fi
    
    # 检查Python包
    python3 -c "import pandas, numpy, scipy" 2>/dev/null || {
        echo -e "${YELLOW}⚠️  安装Python依赖...${NC}"
        pip3 install pandas numpy scipy
    }
}

# 脚本入口
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    check_dependencies
    main "$@"
fi