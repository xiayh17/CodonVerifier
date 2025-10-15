#!/bin/bash

# 实时监控处理进度的脚本

echo "📊 实时监控处理进度"
echo "===================="

# 检查输出目录
HOST_OUTPUT_DIR="data/real_msa"

if [ ! -d "$HOST_OUTPUT_DIR" ]; then
    echo "❌ 输出目录不存在: $HOST_OUTPUT_DIR"
    exit 1
fi

# 数据文件列表
DATA_FILES=(
    "Ec_complete_v2.jsonl"
    "Human_complete_v2.jsonl" 
    "mouse_complete_v2.jsonl"
    "Sac_complete_v2.jsonl"
    "Pic_complete_v2.jsonl"
)

# 获取输入文件总记录数
declare -A INPUT_COUNTS
TOTAL_INPUT=0
for file in "${DATA_FILES[@]}"; do
    input_file="data/enhanced/$file"
    if [ -f "$input_file" ]; then
        count=$(wc -l < "$input_file")
        INPUT_COUNTS[$file]=$count
        TOTAL_INPUT=$((TOTAL_INPUT + count))
    else
        INPUT_COUNTS[$file]=0
    fi
done

echo "📋 输入数据统计:"
echo "  总记录数: $TOTAL_INPUT"
for file in "${DATA_FILES[@]}"; do
    echo "  $file: ${INPUT_COUNTS[$file]}条记录"
done

echo ""
echo "🔄 开始监控 (按Ctrl+C停止)..."
echo ""

# 监控循环
while true; do
    clear
    echo "📊 实时处理进度监控 - $(date)"
    echo "=========================================="
    
    TOTAL_OUTPUT=0
    COMPLETED_FILES=0
    
    for file in "${DATA_FILES[@]}"; do
        input_count=${INPUT_COUNTS[$file]}
        output_file="$HOST_OUTPUT_DIR/${file%.jsonl}_msa_features.jsonl"
        
        if [ -f "$output_file" ]; then
            output_count=$(wc -l < "$output_file")
            TOTAL_OUTPUT=$((TOTAL_OUTPUT + output_count))
            
            if [ $output_count -eq $input_count ]; then
                status="✅ 完成"
                COMPLETED_FILES=$((COMPLETED_FILES + 1))
            else
                progress=$((output_count * 100 / input_count))
                status="🔄 $progress%"
            fi
            
            size=$(du -h "$output_file" | cut -f1)
            echo "  $file: $output_count/$input_count ($status) - $size"
        else
            echo "  $file: 0/$input_count (⏳ 等待)"
        fi
    done
    
    echo ""
    echo "📈 总体进度:"
    overall_progress=$((TOTAL_OUTPUT * 100 / TOTAL_INPUT))
    echo "  处理记录: $TOTAL_OUTPUT/$TOTAL_INPUT ($overall_progress%)"
    echo "  完成文件: $COMPLETED_FILES/${#DATA_FILES[@]}"
    
    # 显示GPU状态
    echo ""
    echo "🖥️  GPU状态:"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | while read line; do
            echo "  GPU: $line"
        done
    else
        echo "  nvidia-smi不可用"
    fi
    
    # 显示Docker容器状态
    echo ""
    echo "🐳 Docker容器:"
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "(msa|mmseqs)" || echo "  无相关容器运行"
    
    echo ""
    echo "⏰ 下次更新: 10秒后 (按Ctrl+C停止监控)"
    
    sleep 10
done
