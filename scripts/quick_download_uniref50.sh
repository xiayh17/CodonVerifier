#!/bin/bash

# 快速下载 UniRef50 数据库脚本
# 这是生产推荐的最小数据库，平衡了覆盖度和计算效率

set -e

echo "🚀 快速下载 UniRef50 数据库 (推荐生产使用)"
echo "📊 大小: ~20GB"
echo "⏱️  预计时间: 30-60分钟 (取决于网络速度)"
echo ""

# 创建目录
mkdir -p data/mmseqs_db/production
mkdir -p data/mmseqs_db/tmp

# 检查磁盘空间
echo "💾 检查磁盘空间..."
AVAILABLE_SPACE=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$AVAILABLE_SPACE" -lt "25" ]; then
    echo "❌ 磁盘空间不足! 需要至少 25GB，当前可用: ${AVAILABLE_SPACE}GB"
    exit 1
fi
echo "✅ 磁盘空间充足"

# 下载 UniRef50
echo "📥 开始下载 UniRef50..."
echo "⏱️  请耐心等待，这可能需要较长时间..."

docker run --rm \
    -v "$(pwd)/data/mmseqs_db/production":/data \
    -v "$(pwd)/data/mmseqs_db/tmp":/tmp \
    codon-verifier/msa-features-lite:latest \
    mmseqs databases UniRef50 /data/UniRef50 /tmp \
    --threads 16

echo ""
echo "🔍 验证数据库..."

# 验证数据库
docker run --rm \
    -v "$(pwd)/data/mmseqs_db/production":/data \
    codon-verifier/msa-features-lite:latest \
    bash -c "echo '数据库验证:' && mmseqs view /data/UniRef50 | head -3"

echo ""
echo "📊 数据库信息:"
echo "   路径: data/mmseqs_db/production/UniRef50"
echo "   大小: $(du -sh data/mmseqs_db/production/UniRef50* | awk '{sum+=$1} END {print sum "B"}')"

echo ""
echo "🎉 UniRef50 数据库准备完成!"
echo ""
echo "💡 现在可以使用真实 MSA 功能:"
echo "   docker run --rm -v \$(pwd)/data:/data codon-verifier/msa-features-lite:latest \\"
echo "     python app.py --input /data/enhanced/Pic_complete_v2.jsonl \\"
echo "     --output /data/real_msa/Pic_real.jsonl \\"
echo "     --use-mmseqs2 --database /data/mmseqs_db/production/UniRef50"
echo ""

# 清理临时文件
rm -rf data/mmseqs_db/tmp
echo "🧹 临时文件已清理"
echo "✅ 完成!"
