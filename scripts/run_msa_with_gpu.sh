#!/bin/bash

# GPU 加速 MSA 特征生成脚本
# 使用方法: ./scripts/run_msa_with_gpu.sh [input_file] [output_file] [database] [gpu_id]

set -e

# 默认参数
INPUT_FILE=${1:-"data/enhanced/Pic_complete_v2.jsonl"}
OUTPUT_FILE=${2:-"data/real_msa/Pic_gpu.jsonl"}
DATABASE=${3:-"data/mmseqs_db/production/UniRef50"}
GPU_ID=${4:-0}
THREADS=${5:-16}
BATCH_SIZE=${6:-100}

echo "🚀 GPU 加速 MSA 特征生成"
echo "📁 输入文件: $INPUT_FILE"
echo "📁 输出文件: $OUTPUT_FILE"
echo "🗄️  数据库: $DATABASE"
echo "🎮 GPU ID: $GPU_ID"
echo "🧵 线程数: $THREADS"
echo "📦 批次大小: $BATCH_SIZE"
echo ""

# 检查输入文件
if [ ! -f "$INPUT_FILE" ]; then
    echo "❌ 输入文件不存在: $INPUT_FILE"
    exit 1
fi

# 检查数据库
if [ ! -d "$DATABASE" ]; then
    echo "❌ 数据库不存在: $DATABASE"
    echo "请先下载数据库: ./scripts/quick_download_uniref50.sh"
    exit 1
fi

# 创建输出目录
mkdir -p "$(dirname "$OUTPUT_FILE")"

# 检查 GPU 可用性
echo "🔍 检查 GPU 可用性..."
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | sed -n "${GPU_ID}p")
    if [ -n "$GPU_INFO" ]; then
        echo "✅ 找到 GPU: $GPU_INFO"
    else
        echo "❌ GPU $GPU_ID 不存在"
        exit 1
    fi
else
    echo "⚠️  nvidia-smi 不可用，将尝试使用 GPU"
fi

echo ""
echo "🏃 开始 GPU 加速 MSA 生成..."
echo "⏱️  这可能需要较长时间，请耐心等待..."

# 使用 Docker 运行 GPU 加速的 MSA 生成
docker run --rm \
    --gpus all \
    -v "$(pwd)/data":/data \
    codon-verifier/msa-features-lite:latest \
    python app.py \
    --input "/data/$(echo $INPUT_FILE | sed 's|^data/||')" \
    --output "/data/$(echo $OUTPUT_FILE | sed 's|^data/||')" \
    --use-mmseqs2 \
    --database "/data/$(echo $DATABASE | sed 's|^data/||')" \
    --use-gpu \
    --gpu-id "$GPU_ID" \
    --threads "$THREADS" \
    --batch-size "$BATCH_SIZE"

echo ""
echo "🎉 GPU 加速 MSA 生成完成!"
echo "📁 输出文件: $OUTPUT_FILE"

# 检查输出文件
if [ -f "$OUTPUT_FILE" ]; then
    RECORD_COUNT=$(wc -l < "$OUTPUT_FILE")
    FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
    echo "📊 结果统计:"
    echo "   记录数: $RECORD_COUNT"
    echo "   文件大小: $FILE_SIZE"
    
    # 显示前几条记录的特征
    echo ""
    echo "🔍 特征预览 (前3条记录):"
    head -3 "$OUTPUT_FILE" | while read -r line; do
        echo "$line" | python -c "
import json, sys
try:
    data = json.loads(sys.stdin.read())
    msa = data.get('msa_features', {})
    print(f'  MSA深度: {msa.get(\"msa_depth\", \"N/A\"):.1f}, 覆盖度: {msa.get(\"msa_coverage\", \"N/A\"):.2f}')
except:
    print('  (无法解析)')
"
    done
else
    echo "❌ 输出文件未生成"
    exit 1
fi

echo ""
echo "✅ 完成!"
