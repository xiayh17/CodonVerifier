#!/bin/bash

# 快速测试配置脚本 - 验证所有设置是否正确

echo "🧪 快速配置测试"
echo "================"

# 测试参数
TEST_RECORDS=5
HOST_DATA_DIR="$(pwd)/data"  # 宿主机数据目录
DOCKER_DATABASE="/data/mmseqs_db/production/UniRef50"  # Docker容器内数据库路径
DOCKER_OUTPUT_DIR="/data/real_msa"  # Docker容器内输出目录
HOST_OUTPUT_DIR="data/real_msa"  # 宿主机输出目录

echo "📋 测试配置:"
echo "  测试记录数: $TEST_RECORDS"
echo "  宿主机数据目录: $HOST_DATA_DIR"
echo "  Docker数据库路径: $DOCKER_DATABASE"
echo "  Docker输出目录: $DOCKER_OUTPUT_DIR"
echo "  宿主机输出目录: $HOST_OUTPUT_DIR"
echo ""

# 检查输入文件
echo "🔍 检查输入文件..."
TEST_FILE="data/enhanced/Pic_complete_v2.jsonl"
if [ -f "$TEST_FILE" ]; then
    count=$(wc -l < "$TEST_FILE")
    echo "✅ 测试文件存在: $TEST_FILE ($count条记录)"
else
    echo "❌ 测试文件不存在: $TEST_FILE"
    exit 1
fi

# 检查数据库
echo ""
echo "🔍 检查数据库..."
HOST_DATABASE="data/mmseqs_db/production/UniRef50"
if [ -f "$HOST_DATABASE" ]; then
    size=$(du -h "$HOST_DATABASE" | cut -f1)
    echo "✅ 宿主机数据库存在: $HOST_DATABASE ($size)"
    echo "✅ Docker容器内路径: $DOCKER_DATABASE"
else
    echo "❌ 宿主机数据库不存在: $HOST_DATABASE"
    exit 1
fi

# 检查Docker镜像
echo ""
echo "🔍 检查Docker镜像..."
if docker image inspect codon-verifier/msa-features-lite:latest > /dev/null 2>&1; then
    echo "✅ Docker镜像存在: codon-verifier/msa-features-lite:latest"
else
    echo "❌ Docker镜像不存在，请先构建镜像"
    exit 1
fi

# 检查GPU
echo ""
echo "🔍 检查GPU..."
if command -v nvidia-smi &> /dev/null; then
    gpu_info=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
    echo "✅ GPU可用: $gpu_info"
else
    echo "⚠️  nvidia-smi不可用，将使用CPU模式"
fi

# 创建输出目录
echo ""
echo "🔍 准备输出目录..."
mkdir -p "$HOST_OUTPUT_DIR"
echo "✅ 宿主机输出目录已创建: $HOST_OUTPUT_DIR"
echo "✅ Docker容器内输出路径: $DOCKER_OUTPUT_DIR"

# 运行快速测试
echo ""
echo "🏃 运行快速测试..."
echo "开始时间: $(date)"

docker run --rm \
    --gpus all \
    --entrypoint="" \
    -v "$HOST_DATA_DIR":/data \
    codon-verifier/msa-features-lite:latest \
    python3 app.py \
    --input "/data/enhanced/Pic_complete_v2.jsonl" \
    --output "/data/real_msa/test_config_output.jsonl" \
    --use-mmseqs2 \
    --database "$DOCKER_DATABASE" \
    --use-gpu \
    --gpu-id 0 \
    --threads 8 \
    --batch-size 5 \
    --limit $TEST_RECORDS \
    --db-init-timeout 300 \
    --search-timeout 600 \
    --log-level INFO

# 检查测试结果
echo ""
echo "🔍 检查测试结果..."
HOST_OUTPUT_FILE="$HOST_OUTPUT_DIR/test_config_output.jsonl"
if [ -f "$HOST_OUTPUT_FILE" ]; then
    output_count=$(wc -l < "$HOST_OUTPUT_FILE")
    size=$(du -h "$HOST_OUTPUT_FILE" | cut -f1)
    echo "✅ 测试成功: $output_count条记录, $size"
    
    # 显示第一条记录的特征
    echo ""
    echo "📊 特征示例:"
    head -1 "$HOST_OUTPUT_FILE" | python3 -c "
import json, sys
try:
    data = json.loads(sys.stdin.read())
    msa = data.get('msa_features', {})
    print(f'  MSA深度: {msa.get(\"msa_depth\", \"N/A\")}')
    print(f'  MSA覆盖度: {msa.get(\"msa_coverage\", \"N/A\")}')
    print(f'  保守性: {msa.get(\"conservation_mean\", \"N/A\")}')
    print(f'  共进化分数: {msa.get(\"coevolution_score\", \"N/A\")}')
except Exception as e:
    print(f'  解析错误: {e}')
"
    
    # 清理测试文件
    rm "$HOST_OUTPUT_FILE"
    echo ""
    echo "🧹 测试文件已清理"
    
else
    echo "❌ 测试失败: 输出文件未生成"
    exit 1
fi

echo ""
echo "结束时间: $(date)"
echo ""
echo "✅ 配置测试通过! 可以开始批量处理"
echo ""
echo "🚀 推荐运行命令:"
echo "  # 串行处理 (稳定)"
echo "  bash scripts/batch_process_all_datasets.sh"
echo ""
echo "  # 并行处理 (最快)"
echo "  bash scripts/parallel_batch_process.sh"
echo ""
echo "  # 监控进度 (另开终端)"
echo "  bash scripts/monitor_processing.sh"
