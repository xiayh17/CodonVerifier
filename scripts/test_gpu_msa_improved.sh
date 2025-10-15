#!/bin/bash

# 改进的 GPU 加速 MSA 测试脚本
# 解决 GPU 超时问题

set -e

echo "🧪 改进的 GPU 加速 MSA 功能测试"
echo ""

# 检查 GPU
echo "🔍 检查 GPU 状态..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
    echo "✅ GPU 可用"
else
    echo "⚠️  nvidia-smi 不可用"
fi

echo ""
echo "📊 测试策略:"
echo "  1. 小批次 (10条) - 自动使用CPU (避免GPU初始化开销)"
echo "  2. 中等批次 (100条) - 使用GPU加速"
echo "  3. 使用Swiss-Prot数据库 (更快)"
echo "  4. 中等数据库测试 - 使用UniRef50数据库"

# 检查Swiss-Prot数据库是否存在
SWISS_PROT_DB="data/mmseqs_db/test_production/SwissProt"
if [ ! -f "$SWISS_PROT_DB" ]; then
    echo "⚠️  Swiss-Prot数据库不存在，下载中..."
    ./scripts/download_test_database.sh
else
    echo "✅ Swiss-Prot数据库已存在，直接使用"
fi

# 检查UniRef50数据库是否存在
PRODUCTION_DB="data/mmseqs_db/production/UniRef50"
if [ ! -f "$PRODUCTION_DB" ]; then
    echo "⚠️  UniRef50数据库不存在，请确保数据库已正确安装"
    echo "   数据库路径: $PRODUCTION_DB"
else
    echo "✅ UniRef50数据库已存在，可用于中等数据库测试"
fi

echo ""
echo "🏃 测试1: 小批次CPU模式 (10条记录)..."
docker run --rm \
    --gpus all \
    --entrypoint="" \
    -v "$(pwd)/data":/data \
    codon-verifier/msa-features-lite:latest \
    python3 app.py \
    --input /data/enhanced/Pic_complete_v2.jsonl \
    --output /data/real_msa/Pic_cpu_test.jsonl \
    --use-mmseqs2 \
    --database "/data/mmseqs_db/test_production/SwissProt" \
    --threads 20 \
    --batch-size 10 \
    --limit 10

echo ""
echo "🏃 测试2: GPU模式 (10条记录，但会自动切换到CPU)..."
docker run --rm \
    --gpus all \
    --entrypoint="" \
    -v "$(pwd)/data":/data \
    codon-verifier/msa-features-lite:latest \
    python3 app.py \
    --input /data/enhanced/Pic_complete_v2.jsonl \
    --output /data/real_msa/Pic_gpu_smart_test.jsonl \
    --use-mmseqs2 \
    --database "/data/mmseqs_db/test_production/SwissProt" \
    --use-gpu \
    --gpu-id 0 \
    --threads 20 \
    --batch-size 10 \
    --limit 10

echo ""
echo "🏃 测试3: 中等批次GPU模式 (50条记录)..."
docker run --rm \
    --gpus all \
    --entrypoint="" \
    -v "$(pwd)/data":/data \
    codon-verifier/msa-features-lite:latest \
    python3 app.py \
    --input /data/enhanced/Pic_complete_v2.jsonl \
    --output /data/real_msa/Pic_gpu_medium_test.jsonl \
    --use-mmseqs2 \
    --database "/data/mmseqs_db/test_production/SwissProt" \
    --use-gpu \
    --gpu-id 0 \
    --threads 20 \
    --batch-size 50 \
    --limit 50

echo ""
echo "🏃 测试4: 中等数据库测试 (100条记录，使用production数据库)..."
if [ -f "$PRODUCTION_DB" ]; then
    docker run --rm \
        --gpus all \
        --entrypoint="" \
        -v "$(pwd)/data":/data \
        codon-verifier/msa-features-lite:latest \
        python3 app.py \
        --input /data/enhanced/Pic_complete_v2.jsonl \
        --output /data/real_msa/Pic_production_db_test.jsonl \
        --use-mmseqs2 \
        --database "/data/mmseqs_db/production/UniRef50" \
        --use-gpu \
        --gpu-id 0 \
        --threads 8 \
        --batch-size 100 \
        --limit 100 \
        --db-init-timeout 600 \
        --search-timeout 6000
else
    echo "⚠️  跳过中等数据库测试 - Production数据库不存在"
fi

echo ""
echo "🎉 所有测试完成!"

# 检查结果
echo ""
echo "📊 测试结果对比:"

for test_file in "Pic_cpu_test.jsonl" "Pic_gpu_smart_test.jsonl" "Pic_gpu_medium_test.jsonl" "Pic_production_db_test.jsonl"; do
    if [ -f "data/real_msa/$test_file" ]; then
        echo "  $test_file:"
        echo "    记录数: $(wc -l < data/real_msa/$test_file)"
        echo "    文件大小: $(du -h data/real_msa/$test_file | cut -f1)"
    else
        echo "  $test_file: ❌ 测试失败"
    fi
done

echo ""
echo "🔍 特征质量检查:"
if [ -f "data/real_msa/Pic_gpu_medium_test.jsonl" ]; then
    echo "  GPU中等批次结果 (Swiss-Prot数据库):"
    head -1 data/real_msa/Pic_gpu_medium_test.jsonl | python -c "
import json, sys
try:
    data = json.loads(sys.stdin.read())
    msa = data.get('msa_features', {})
    print(f'    MSA深度: {msa.get(\"msa_depth\", \"N/A\")}')
    print(f'    MSA覆盖度: {msa.get(\"msa_coverage\", \"N/A\")}')
    print(f'    保守性: {msa.get(\"conservation_mean\", \"N/A\")}')
except Exception as e:
    print(f'    解析错误: {e}')
"
fi

if [ -f "data/real_msa/Pic_production_db_test.jsonl" ]; then
    echo "  Production数据库结果:"
    head -1 data/real_msa/Pic_production_db_test.jsonl | python -c "
import json, sys
try:
    data = json.loads(sys.stdin.read())
    msa = data.get('msa_features', {})
    print(f'    MSA深度: {msa.get(\"msa_depth\", \"N/A\")}')
    print(f'    MSA覆盖度: {msa.get(\"msa_coverage\", \"N/A\")}')
    print(f'    保守性: {msa.get(\"conservation_mean\", \"N/A\")}')
except Exception as e:
    print(f'    解析错误: {e}')
"
fi

echo ""
echo "✅ 改进的GPU加速功能测试完成!"
echo ""
echo "💡 关键改进:"
echo "  - 小批次自动使用CPU (避免GPU初始化开销)"
echo "  - 使用Swiss-Prot数据库 (更快)"
echo "  - 增加超时时间 (GPU需要更长初始化)"
echo "  - 智能批次大小选择"
echo "  - 新增Production数据库测试 (更全面的数据库覆盖)"
