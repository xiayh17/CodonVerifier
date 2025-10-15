#!/bin/bash

# 高效批量处理所有数据集的脚本
# 硬件配置: GPU 24GB, CPU 24核, 内存 127GB
# 数据库: UniRef50 (约37GB)

set -e

echo "🚀 高效批量处理所有数据集"
echo "================================"
echo "硬件配置:"
echo "  GPU: 24GB VRAM"
echo "  CPU: 24 cores"
echo "  内存: 127GB"
echo "  数据库: UniRef50 (~37GB)"
echo ""

# 配置参数
HOST_DATA_DIR="$(pwd)/data"  # 宿主机数据目录
DOCKER_DATABASE="/data/mmseqs_db/production/UniRef50"  # Docker容器内数据库路径
DOCKER_OUTPUT_DIR="/data/real_msa"  # Docker容器内输出目录
HOST_OUTPUT_DIR="data/real_msa"  # 宿主机输出目录
THREADS=20  # 留4个核心给系统
GPU_ID=0
DB_INIT_TIMEOUT=600
SEARCH_TIMEOUT=1800  # 30分钟超时

# 根据数据规模优化的batch-size
declare -A BATCH_SIZES=(
    ["Ec_complete_v2.jsonl"]=18780    # 18,780条 - 大批次
    ["Human_complete_v2.jsonl"]=13421 # 13,421条 - 中大批次
    ["mouse_complete_v2.jsonl"]=13253 # 13,253条 - 中大批次
    ["Sac_complete_v2.jsonl"]=6384   # 6,384条 - 中批次
    ["Pic_complete_v2.jsonl"]=320    # 320条 - 小批次
)

# 数据文件列表
DATA_FILES=(
    "Ec_complete_v2.jsonl"
    "Human_complete_v2.jsonl" 
    "mouse_complete_v2.jsonl"
    "Sac_complete_v2.jsonl"
    "Pic_complete_v2.jsonl"
)

# 创建输出目录
mkdir -p "$HOST_OUTPUT_DIR"

# 检查GPU状态
echo "🔍 检查GPU状态..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits
    echo "✅ GPU可用"
else
    echo "⚠️  nvidia-smi不可用，将使用CPU模式"
fi

echo ""
echo "📊 处理计划:"
for file in "${DATA_FILES[@]}"; do
    count=$(wc -l < "data/enhanced/$file")
    batch_size=${BATCH_SIZES[$file]}
    estimated_batches=$(( (count + batch_size - 1) / batch_size ))
    echo "  $file: $count条记录, batch-size=$batch_size, 预计$estimated_batches个批次"
done

echo ""
echo "⏰ 开始处理 (开始时间: $(date))"
echo "================================"

# 总体统计
TOTAL_START_TIME=$(date +%s)
TOTAL_PROCESSED=0
TOTAL_ERRORS=0

# 处理每个文件
for file in "${DATA_FILES[@]}"; do
    echo ""
    echo "📁 处理文件: $file"
    echo "----------------------------------------"
    
    # 文件级统计
    FILE_START_TIME=$(date +%s)
    input_file="data/enhanced/$file"
    output_file="$HOST_OUTPUT_DIR/${file%.jsonl}_msa_features.jsonl"
    batch_size=${BATCH_SIZES[$file]}
    
    # 检查输入文件
    if [ ! -f "$input_file" ]; then
        echo "❌ 输入文件不存在: $input_file"
        continue
    fi
    
    # 获取记录数
    total_records=$(wc -l < "$input_file")
    echo "📊 记录数: $total_records"
    echo "📦 批次大小: $batch_size"
    echo "🎯 输出文件: $output_file"
    
    # 运行处理
    echo "🏃 开始处理..."
    docker run --rm \
        --gpus all \
        --entrypoint="" \
        -v "$HOST_DATA_DIR":/data \
        codon-verifier/msa-features-lite:latest \
        python3 app.py \
        --input "/data/enhanced/$file" \
        --output "/data/real_msa/${file%.jsonl}_msa_features.jsonl" \
        --use-mmseqs2 \
        --database "$DOCKER_DATABASE" \
        --use-gpu \
        --gpu-id $GPU_ID \
        --threads $THREADS \
        --batch-size $batch_size \
        --db-init-timeout $DB_INIT_TIMEOUT \
        --search-timeout $SEARCH_TIMEOUT \
        --log-level INFO
    
    # 计算文件处理时间
    FILE_END_TIME=$(date +%s)
    FILE_DURATION=$((FILE_END_TIME - FILE_START_TIME))
    FILE_HOURS=$((FILE_DURATION / 3600))
    FILE_MINUTES=$(((FILE_DURATION % 3600) / 60))
    FILE_SECONDS=$((FILE_DURATION % 60))
    
    # 检查输出文件
    if [ -f "$output_file" ]; then
        output_records=$(wc -l < "$output_file")
        echo "✅ 文件处理完成: $output_records条记录"
        echo "⏱️  用时: ${FILE_HOURS}h ${FILE_MINUTES}m ${FILE_SECONDS}s"
        TOTAL_PROCESSED=$((TOTAL_PROCESSED + output_records))
    else
        echo "❌ 输出文件未生成: $output_file"
        TOTAL_ERRORS=$((TOTAL_ERRORS + 1))
    fi
    
    echo "----------------------------------------"
done

# 总体统计
TOTAL_END_TIME=$(date +%s)
TOTAL_DURATION=$((TOTAL_END_TIME - TOTAL_START_TIME))
TOTAL_HOURS=$((TOTAL_DURATION / 3600))
TOTAL_MINUTES=$(((TOTAL_DURATION % 3600) / 60))
TOTAL_SECONDS=$((TOTAL_DURATION % 60))

echo ""
echo "🎉 所有文件处理完成!"
echo "================================"
echo "📊 总体统计:"
echo "  总处理时间: ${TOTAL_HOURS}h ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s"
echo "  成功处理记录: $TOTAL_PROCESSED"
echo "  错误文件数: $TOTAL_ERRORS"
echo "  平均处理速度: $((TOTAL_PROCESSED / (TOTAL_DURATION / 60))) 记录/分钟"

echo ""
echo "📁 输出文件:"
for file in "${DATA_FILES[@]}"; do
    output_file="$HOST_OUTPUT_DIR/${file%.jsonl}_msa_features.jsonl"
    if [ -f "$output_file" ]; then
        size=$(du -h "$output_file" | cut -f1)
        records=$(wc -l < "$output_file")
        echo "  ✅ $output_file ($records条记录, $size)"
    else
        echo "  ❌ $output_file (未生成)"
    fi
done

echo ""
echo "💡 性能优化建议:"
echo "  - 使用GPU加速 (24GB VRAM)"
echo "  - 优化批次大小 (根据数据规模)"
echo "  - 并行处理 (20个CPU核心)"
echo "  - 大数据库缓存 (37GB UniRef50)"
echo "  - 合理超时设置 (30分钟搜索超时)"

echo ""
echo "🔍 质量检查:"
if [ -f "$HOST_OUTPUT_DIR/Ec_complete_v2_msa_features.jsonl" ]; then
    echo "  检查Ec数据集结果质量..."
    head -1 "$HOST_OUTPUT_DIR/Ec_complete_v2_msa_features.jsonl" | python3 -c "
import json, sys
try:
    data = json.loads(sys.stdin.read())
    msa = data.get('msa_features', {})
    print(f'    MSA深度: {msa.get(\"msa_depth\", \"N/A\")}')
    print(f'    MSA覆盖度: {msa.get(\"msa_coverage\", \"N/A\")}')
    print(f'    保守性: {msa.get(\"conservation_mean\", \"N/A\")}')
    print(f'    共进化分数: {msa.get(\"coevolution_score\", \"N/A\")}')
except Exception as e:
    print(f'    解析错误: {e}')
"
fi

echo ""
echo "✅ 批量处理完成! (结束时间: $(date))"
