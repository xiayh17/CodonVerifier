#!/bin/bash

# 优化版批量处理脚本 - 动态超时设置

set -e

echo "🚀 优化版批量处理 - 动态超时设置"
echo "================================"
echo "硬件配置:"
echo "  GPU: 24GB VRAM"
echo "  CPU: 24 cores"
echo "  内存: 127GB"
echo "  数据库: UniRef50 (~37GB)"
echo ""

# 配置参数
HOST_DATA_DIR="$(pwd)/data"
DOCKER_DATABASE="/data/mmseqs_db/production/UniRef50"
DOCKER_OUTPUT_DIR="/data/real_msa"
HOST_OUTPUT_DIR="data/real_msa"
THREADS=20
GPU_ID=0

# 动态超时设置函数
calculate_timeouts() {
    local file=$1
    local record_count=$2
    local batch_size=$3
    
    # 数据库初始化超时 (基于数据库大小)
    local db_init_timeout=600  # 基础10分钟
    
    # 搜索超时 (基于批次大小和记录数)
    local search_timeout=1800  # 基础30分钟
    
    # 根据记录数调整数据库初始化超时
    if [ $record_count -lt 1000 ]; then
        db_init_timeout=300  # 5分钟
    elif [ $record_count -gt 10000 ]; then
        db_init_timeout=900  # 15分钟
    fi
    
    # 根据批次大小调整搜索超时
    if [ $batch_size -lt 50 ]; then
        search_timeout=900   # 15分钟
    elif [ $batch_size -gt 150 ]; then
        search_timeout=3600  # 60分钟
    fi
    
    # 根据记录数进一步调整搜索超时
    if [ $record_count -gt 15000 ]; then
        search_timeout=$((search_timeout + 1800))  # 增加30分钟
    fi
    
    echo "$db_init_timeout $search_timeout"
}

# 根据数据规模优化的batch-size
declare -A BATCH_SIZES=(
    ["Ec_complete_v2.jsonl"]=200
    ["Human_complete_v2.jsonl"]=150
    ["mouse_complete_v2.jsonl"]=150
    ["Sac_complete_v2.jsonl"]=100
    ["Pic_complete_v2.jsonl"]=50
)

# 数据文件列表
DATA_FILES=(
    "Pic_complete_v2.jsonl"
    "Sac_complete_v2.jsonl"
    "mouse_complete_v2.jsonl"
    "Human_complete_v2.jsonl"
    "Ec_complete_v2.jsonl"
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
echo "📊 动态超时设置预览:"
for file in "${DATA_FILES[@]}"; do
    if [ -f "data/enhanced/$file" ]; then
        count=$(wc -l < "data/enhanced/$file")
        batch_size=${BATCH_SIZES[$file]}
        timeouts=($(calculate_timeouts "$file" "$count" "$batch_size"))
        db_timeout=${timeouts[0]}
        search_timeout=${timeouts[1]}
        
        echo "  $file ($count条记录, batch-size=$batch_size):"
        echo "    数据库初始化: ${db_timeout}秒 ($((db_timeout/60))分钟)"
        echo "    搜索超时: ${search_timeout}秒 ($((search_timeout/60))分钟)"
    fi
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
    
    # 获取记录数并计算动态超时
    total_records=$(wc -l < "$input_file")
    timeouts=($(calculate_timeouts "$file" "$total_records" "$batch_size"))
    db_timeout=${timeouts[0]}
    search_timeout=${timeouts[1]}
    
    echo "📊 记录数: $total_records"
    echo "📦 批次大小: $batch_size"
    echo "⏰ 数据库初始化超时: ${db_timeout}秒 ($((db_timeout/60))分钟)"
    echo "⏰ 搜索超时: ${search_timeout}秒 ($((search_timeout/60))分钟)"
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
        --db-init-timeout $db_timeout \
        --search-timeout $search_timeout \
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
        echo "📈 实际速度: $((output_records * 60 / FILE_DURATION)) 记录/分钟"
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
echo "💡 动态超时优化效果:"
echo "  - 根据数据规模自动调整超时"
echo "  - 小文件使用较短超时，提高效率"
echo "  - 大文件使用较长超时，避免中断"
echo "  - 基于批次大小和记录数智能计算"

echo ""
echo "✅ 优化版批量处理完成! (结束时间: $(date))"
