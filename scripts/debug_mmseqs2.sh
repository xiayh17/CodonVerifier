#!/bin/bash

# MMseqs2 调试脚本
# 用于诊断 MSA 生成过程中的问题

set -e

echo "🔍 MMseqs2 调试诊断"
echo ""

# 创建测试目录
mkdir -p data/debug_msa
cd data/debug_msa

echo "📝 创建测试序列..."
cat > test_sequences.fasta << 'EOF'
>test_seq1
MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNEL
>test_seq2
MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNEL
EOF

echo "✅ 测试序列创建完成"
echo ""

echo "🔧 测试 MMseqs2 基本功能..."

# 测试 1: 创建查询数据库
echo "1. 创建查询数据库..."
docker run --rm -v $(pwd):/data codon-verifier/msa-features-lite:latest \
  mmseqs createdb /data/test_sequences.fasta /data/query_db

echo "✅ 查询数据库创建成功"
echo ""

# 测试 2: 检查目标数据库
echo "2. 检查目标数据库..."
docker run --rm -v $(pwd):/data -v $(pwd)/../mmseqs_db/production:/db \
  codon-verifier/msa-features-lite:latest \
  mmseqs view /db/UniRef50 | head -3

echo "✅ 目标数据库可访问"
echo ""

# 测试 3: 小规模搜索测试
echo "3. 小规模搜索测试..."
docker run --rm -v $(pwd):/data -v $(pwd)/../mmseqs_db/production:/db \
  codon-verifier/msa-features-lite:latest \
  mmseqs search /data/query_db /db/UniRef50 /data/result_db /data/tmp \
  --threads 4 -e 0.001 --min-seq-id 0.3 -c 0.5 --alignment-mode 3 \
  --max-seqs 10

echo "✅ 搜索测试完成"
echo ""

# 测试 4: 转换结果
echo "4. 转换搜索结果..."
docker run --rm -v $(pwd):/data -v $(pwd)/../mmseqs_db/production:/db \
  codon-verifier/msa-features-lite:latest \
  mmseqs convertalis /data/query_db /db/UniRef50 /data/result_db /data/result.tsv \
  --format-output 'query,target,pident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits'

echo "✅ 结果转换完成"
echo ""

# 测试 5: 查看结果
echo "5. 搜索结果:"
if [ -f "result.tsv" ]; then
    echo "找到 $(wc -l < result.tsv) 个结果"
    head -5 result.tsv
else
    echo "❌ 没有找到结果文件"
fi

echo ""
echo "🧹 清理测试文件..."
cd ..
rm -rf debug_msa

echo "✅ 调试完成"
