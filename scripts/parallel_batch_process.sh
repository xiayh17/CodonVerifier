#!/bin/bash

# 并行批量处理脚本 - 最高效率版本
# 同时处理多个文件，最大化硬件利用率

set -e

echo "🚀 并行批量处理 - 最高效率版本"
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
THREADS_PER_JOB=8  # 每个任务使用8个核心，可以并行3个任务
MAX_PARALLEL_JOBS=3  # 最大并行任务数
GPU_ID=0
DB_INIT_TIMEOUT=600
SEARCH_TIMEOUT=18000

# 根据数据规模优化的batch-size
declare -A BATCH_SIZES=(
    ["Ec_complete_v2.jsonl"]=200    # 18,780条 - 大批次
    ["Human_complete_v2.jsonl"]=150 # 13,421条 - 中大批次
    ["mouse_complete_v2.jsonl"]=150 # 13,253条 - 中大批次
    ["Sac_complete_v2.jsonl"]=100   # 6,384条 - 中批次
    ["Pic_complete_v2.jsonl"]=50    # 320条 - 小批次
)

# 数据文件列表 (按大小排序，先处理小的)
DATA_FILES=(
    "Pic_complete_v2.jsonl"     # 320条 - 最快
    "Sac_complete_v2.jsonl"     # 6,384条
    "mouse_complete_v2.jsonl"   # 13,253条
    "Human_complete_v2.jsonl"   # 13,421条
    "Ec_complete_v2.jsonl"      # 18,780条 - 最慢
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
echo "📊 并行处理计划:"
echo "  最大并行任务: $MAX_PARALLEL_JOBS"
echo "  每任务CPU核心: $THREADS_PER_JOB"
echo "  总CPU核心使用: $((MAX_PARALLEL_JOBS * THREADS_PER_JOB))"
echo ""

for file in "${DATA_FILES[@]}"; do
    count=$(wc -l < "data/enhanced/$file")
    batch_size=${BATCH_SIZES[$file]}
    estimated_batches=$(( (count + batch_size - 1) / batch_size ))
    echo "  $file: $count条记录, batch-size=$batch_size, 预计$estimated_batches个批次"
done

echo ""
echo "⏰ 开始并行处理 (开始时间: $(date))"
echo "================================"

# 总体统计
TOTAL_START_TIME=$(date +%s)
TOTAL_PROCESSED=0
TOTAL_ERRORS=0

# 并行处理函数
process_file() {
    local file=$1
    local job_id=$2
    
    echo "[Job $job_id] 📁 开始处理: $file"
    
    local FILE_START_TIME=$(date +%s)
    local input_file="data/enhanced/$file"
    local output_file="$HOST_OUTPUT_DIR/${file%.jsonl}_msa_features.jsonl"
    local batch_size=${BATCH_SIZES[$file]}
    
    # 检查输入文件
    if [ ! -f "$input_file" ]; then
        echo "[Job $job_id] ❌ 输入文件不存在: $input_file"
        return 1
    fi
    
    local total_records=$(wc -l < "$input_file")
    echo "[Job $job_id] 📊 记录数: $total_records, 批次大小: $batch_size"
    
    # 运行处理
    echo "[Job $job_id] 🏃 开始处理..."
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
        --threads $THREADS_PER_JOB \
        --batch-size $batch_size \
        --db-init-timeout $DB_INIT_TIMEOUT \
        --search-timeout $SEARCH_TIMEOUT \
        --log-level INFO
    
    # 计算处理时间
    local FILE_END_TIME=$(date +%s)
    local FILE_DURATION=$((FILE_END_TIME - FILE_START_TIME))
    local FILE_HOURS=$((FILE_DURATION / 3600))
    local FILE_MINUTES=$(((FILE_DURATION % 3600) / 60))
    local FILE_SECONDS=$((FILE_DURATION % 60))
    
    # 检查输出文件
    if [ -f "$output_file" ]; then
        local output_records=$(wc -l < "$output_file")
        echo "[Job $job_id] ✅ 完成: $file ($output_records条记录, ${FILE_HOURS}h ${FILE_MINUTES}m ${FILE_SECONDS}s)"
        echo "$output_records" > "/tmp/job_${job_id}_result"
    else
        echo "[Job $job_id] ❌ 失败: $file (输出文件未生成)"
        echo "0" > "/tmp/job_${job_id}_result"
    fi
}

# 启动并行任务
echo "🚀 启动并行任务..."
PIDS=()
JOB_IDS=()

for i in "${!DATA_FILES[@]}"; do
    file="${DATA_FILES[$i]}"
    job_id=$((i + 1))
    
    # 如果达到最大并行数，等待一个任务完成
    if [ ${#PIDS[@]} -ge $MAX_PARALLEL_JOBS ]; then
        echo "⏳ 等待任务完成 (当前并行: ${#PIDS[@]})..."
        wait ${PIDS[0]}
        unset PIDS[0]
        PIDS=("${PIDS[@]}")  # 重新索引数组
    fi
    
    # 启动新任务
    process_file "$file" "$job_id" &
    pid=$!
    PIDS+=($pid)
    JOB_IDS+=($job_id)
    
    echo "🎯 启动任务 $job_id: $file (PID: $pid)"
    sleep 5  # 避免同时启动造成资源竞争
done

# 等待所有任务完成
echo ""
echo "⏳ 等待所有任务完成..."
for pid in "${PIDS[@]}"; do
    wait $pid
done

# 收集结果
echo ""
echo "📊 收集处理结果..."
for job_id in "${JOB_IDS[@]}"; do
    if [ -f "/tmp/job_${job_id}_result" ]; then
        result=$(cat "/tmp/job_${job_id}_result")
        TOTAL_PROCESSED=$((TOTAL_PROCESSED + result))
        rm "/tmp/job_${job_id}_result"
    fi
done

# 总体统计
TOTAL_END_TIME=$(date +%s)
TOTAL_DURATION=$((TOTAL_END_TIME - TOTAL_START_TIME))
TOTAL_HOURS=$((TOTAL_DURATION / 3600))
TOTAL_MINUTES=$(((TOTAL_DURATION % 3600) / 60))
TOTAL_SECONDS=$((TOTAL_DURATION % 60))

echo ""
echo "🎉 并行处理完成!"
echo "================================"
echo "📊 总体统计:"
echo "  总处理时间: ${TOTAL_HOURS}h ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s"
echo "  成功处理记录: $TOTAL_PROCESSED"
echo "  平均处理速度: $((TOTAL_PROCESSED / (TOTAL_DURATION / 60))) 记录/分钟"
echo "  并行效率: $((TOTAL_PROCESSED * 100 / (TOTAL_DURATION / 60))) 记录/分钟/核心"

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
echo "💡 并行优化效果:"
echo "  - 同时处理 $MAX_PARALLEL_JOBS 个文件"
echo "  - 充分利用 $((MAX_PARALLEL_JOBS * THREADS_PER_JOB)) 个CPU核心"
echo "  - GPU资源最大化利用"
echo "  - 预计比串行处理快 $MAX_PARALLEL_JOBS 倍"

echo ""
echo "✅ 并行批量处理完成! (结束时间: $(date))"
