#!/bin/bash

# 下载测试数据库脚本
# 使用 Swiss-Prot 数据库进行测试 (较小，适合测试)

set -e

echo "🧪 下载测试数据库 (Swiss-Prot)"
echo "📊 大小: ~500MB"
echo "⏱️  预计时间: 5-10分钟"
echo ""

# 创建目录
mkdir -p data/mmseqs_db/test_production
mkdir -p data/mmseqs_db/tmp

# 检查磁盘空间
echo "💾 检查磁盘空间..."
AVAILABLE_SPACE=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$AVAILABLE_SPACE" -lt "2" ]; then
    echo "❌ 磁盘空间不足! 需要至少 2GB，当前可用: ${AVAILABLE_SPACE}GB"
    exit 1
fi
echo "✅ 磁盘空间充足"

# 下载 Swiss-Prot
echo "📥 开始下载 Swiss-Prot 数据库..."

docker run --rm \
    -v "$(pwd)/data/mmseqs_db/test_production":/data \
    -v "$(pwd)/data/mmseqs_db/tmp":/tmp \
    codon-verifier/msa-features-lite:latest \
    mmseqs databases "UniProtKB/Swiss-Prot" /data/SwissProt /tmp \
    --threads 8

echo ""
echo "🔍 验证数据库..."

# 验证数据库
docker run --rm \
    -v "$(pwd)/data/mmseqs_db/test_production":/data \
    codon-verifier/msa-features-lite:latest \
    bash -c "echo '数据库验证:' && mmseqs view /data/SwissProt | head -3"

echo ""
echo "📊 数据库信息:"
echo "   路径: data/mmseqs_db/test_production/SwissProt"
echo "   大小: $(du -sh data/mmseqs_db/test_production/SwissProt* 2>/dev/null | awk '{sum+=$1} END {print sum "B"}' || echo "计算中...")"

echo ""
echo "🎉 Swiss-Prot 测试数据库准备完成!"
echo ""
echo "💡 测试真实 MSA 功能:"
echo "   docker run --rm -v \$(pwd)/data:/data codon-verifier/msa-features-lite:latest \\"
echo "     python app.py --input /data/enhanced/Pic_complete_v2.jsonl \\"
echo "     --output /data/real_msa/Pic_test_swiss.jsonl \\"
echo "     --use-mmseqs2 --database /data/mmseqs_db/test_production/SwissProt \\"
echo "     --limit 10"
echo ""

# 清理临时文件
rm -rf data/mmseqs_db/tmp
echo "🧹 临时文件已清理"
echo "✅ 完成!"
