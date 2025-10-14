#!/bin/bash

# 快速 MSA 测试脚本
# 使用小数据库进行快速测试

set -e

echo "🧪 快速 MSA 功能测试"
echo ""

# 创建小测试数据库
echo "📝 创建测试数据库..."
mkdir -p data/test_db
cd data/test_db

# 创建测试序列
cat > test_db.fasta << 'EOF'
>seq1
MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNEL
>seq2
MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNEL
>seq3
MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNEL
>seq4
MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNEL
>seq5
MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNEL
EOF

# 创建数据库
echo "🔧 创建 MMseqs2 数据库..."
docker run --rm -v $(pwd):/data codon-verifier/msa-features-lite:latest \
  mmseqs createdb /data/test_db.fasta /data/test_db

docker run --rm -v $(pwd):/data codon-verifier/msa-features-lite:latest \
  mmseqs createindex /data/test_db /data/tmp

echo "✅ 测试数据库创建完成"
echo ""

# 创建测试输入
echo "📝 创建测试输入..."
cd ../..
cat > data/test_input.jsonl << 'EOF'
{"protein_id": "test1", "protein_aa": "MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNEL", "host": "test"}
{"protein_id": "test2", "protein_aa": "MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNEL", "host": "test"}
EOF

echo "✅ 测试输入创建完成"
echo ""

# 运行测试
echo "🏃 运行 MSA 测试..."
docker run --rm -v $(pwd)/data:/data codon-verifier/msa-features-lite:latest \
  python app.py \
  --input /data/test_input.jsonl \
  --output /data/test_output.jsonl \
  --use-mmseqs2 \
  --database /data/test_db/test_db \
  --threads 4 \
  --batch-size 2

echo ""
echo "🔍 检查结果..."
if [ -f "data/test_output.jsonl" ]; then
    echo "✅ 测试成功!"
    echo "📊 结果:"
    head -1 data/test_output.jsonl | python -c "
import json, sys
try:
    data = json.loads(sys.stdin.read())
    msa = data.get('msa_features', {})
    print(f'  MSA深度: {msa.get(\"msa_depth\", \"N/A\")}')
    print(f'  MSA覆盖度: {msa.get(\"msa_coverage\", \"N/A\")}')
    print(f'  保守性: {msa.get(\"conservation_mean\", \"N/A\")}')
except Exception as e:
    print(f'  解析错误: {e}')
"
else
    echo "❌ 测试失败"
fi

echo ""
echo "🧹 清理测试文件..."
rm -rf data/test_db data/test_input.jsonl data/test_output.jsonl

echo "✅ 测试完成"
