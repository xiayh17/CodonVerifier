#!/bin/bash

# 生产级 MMseqs2 数据库下载脚本
# 使用方法: ./scripts/download_production_database.sh [database_name] [threads]

set -e

# 默认参数
DATABASE_NAME=${1:-"UniRef50"}
THREADS=${2:-16}
OUTPUT_DIR="data/mmseqs_db/production"
TEMP_DIR="data/mmseqs_db/tmp"

echo "🚀 开始下载生产级 MMseqs2 数据库"
echo "📊 数据库: $DATABASE_NAME"
echo "🧵 线程数: $THREADS"
echo "📁 输出目录: $OUTPUT_DIR"
echo ""

# 创建目录
mkdir -p "$OUTPUT_DIR"
mkdir -p "$TEMP_DIR"

# 检查可用空间
echo "💾 检查磁盘空间..."
REQUIRED_SPACE=0
case $DATABASE_NAME in
    "UniRef50")
        REQUIRED_SPACE=25  # GB
        ;;
    "UniRef90")
        REQUIRED_SPACE=70  # GB
        ;;
    "UniRef100")
        REQUIRED_SPACE=120 # GB
        ;;
    "NR")
        REQUIRED_SPACE=250 # GB
        ;;
    *)
        echo "❌ 不支持的数据库: $DATABASE_NAME"
        echo "支持的数据库: UniRef50, UniRef90, UniRef100, NR"
        exit 1
        ;;
esac

AVAILABLE_SPACE=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$AVAILABLE_SPACE" -lt "$REQUIRED_SPACE" ]; then
    echo "❌ 磁盘空间不足!"
    echo "   需要: ${REQUIRED_SPACE}GB"
    echo "   可用: ${AVAILABLE_SPACE}GB"
    exit 1
fi

echo "✅ 磁盘空间充足 (需要: ${REQUIRED_SPACE}GB, 可用: ${AVAILABLE_SPACE}GB)"
echo ""

# 下载数据库
echo "📥 开始下载 $DATABASE_NAME 数据库..."
echo "⏱️  这可能需要很长时间，请耐心等待..."

docker run --rm \
    -v "$(pwd)/$OUTPUT_DIR":/data \
    -v "$(pwd)/$TEMP_DIR":/tmp \
    codon-verifier/msa-features-lite:latest \
    mmseqs databases "$DATABASE_NAME" /data/"$DATABASE_NAME" /tmp \
    --threads "$THREADS"

echo ""
echo "🔍 验证数据库完整性..."

# 验证数据库
docker run --rm \
    -v "$(pwd)/$OUTPUT_DIR":/data \
    codon-verifier/msa-features-lite:latest \
    mmseqs view /data/"$DATABASE_NAME" | head -5

echo ""
echo "📊 数据库统计信息:"
docker run --rm \
    -v "$(pwd)/$OUTPUT_DIR":/data \
    codon-verifier/msa-features-lite:latest \
    bash -c "echo '序列数量:' && mmseqs view /data/$DATABASE_NAME | wc -l"

echo ""
echo "🎉 数据库下载完成!"
echo "📁 数据库路径: $OUTPUT_DIR/$DATABASE_NAME"
echo ""
echo "💡 使用方法:"
echo "   docker run --rm -v \$(pwd)/data:/data codon-verifier/msa-features-lite:latest \\"
echo "     python app.py --input /data/enhanced/your_file.jsonl \\"
echo "     --output /data/real_msa/your_output.jsonl \\"
echo "     --use-mmseqs2 --database /data/mmseqs_db/production/$DATABASE_NAME"
echo ""
echo "🧹 清理临时文件..."
rm -rf "$TEMP_DIR"
echo "✅ 完成!"
