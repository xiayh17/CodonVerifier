#!/bin/bash

# 超时设置分析脚本

echo "⏰ 超时设置分析"
echo "================"

echo "📊 当前超时设置:"
echo "  数据库初始化超时: 600秒 (10分钟)"
echo "  搜索超时: 1800秒 (30分钟)"
echo ""

echo "🔍 硬件配置分析:"
echo "  GPU: 24GB VRAM"
echo "  CPU: 24核心"
echo "  内存: 127GB"
echo "  数据库: UniRef50 (~37GB)"
echo ""

echo "📈 数据规模分析:"
echo "  总记录数: ~52,000条"
echo "  最大文件: Ec_complete_v2.jsonl (18,780条)"
echo "  批次大小: 50-200 (根据文件大小)"
echo ""

echo "🧮 超时计算分析:"
echo ""

# 计算不同场景的超时需求
echo "1. 数据库初始化超时 (db-init-timeout):"
echo "   - 当前设置: 600秒 (10分钟)"
echo "   - 数据库大小: 37GB"
echo "   - 磁盘I/O: 假设100MB/s → 需要 ~6分钟"
echo "   - GPU初始化: 额外1-2分钟"
echo "   - 建议: 600-900秒 (10-15分钟)"
echo ""

echo "2. 搜索超时 (search-timeout):"
echo "   - 当前设置: 1800秒 (30分钟)"
echo "   - 最大批次: 200条记录"
echo "   - GPU加速: 预计2-5分钟/批次"
echo "   - CPU fallback: 预计5-10分钟/批次"
echo "   - 建议: 1800-3600秒 (30-60分钟)"
echo ""

echo "3. 不同数据规模的时间估算:"
echo ""

# 计算不同文件的时间需求
declare -A FILE_TIMES=(
    ["Pic_complete_v2.jsonl"]="5-10分钟"
    ["Sac_complete_v2.jsonl"]="30-60分钟"
    ["mouse_complete_v2.jsonl"]="60-120分钟"
    ["Human_complete_v2.jsonl"]="60-120分钟"
    ["Ec_complete_v2.jsonl"]="120-180分钟"
)

for file in "${!FILE_TIMES[@]}"; do
    count=$(wc -l < "data/enhanced/$file" 2>/dev/null || echo "0")
    if [ "$count" -gt 0 ]; then
        echo "   $file ($count条记录): ${FILE_TIMES[$file]}"
    fi
done

echo ""
echo "💡 优化建议:"
echo ""

echo "1. 数据库初始化超时优化:"
echo "   - 小文件 (<1000条): 300秒 (5分钟)"
echo "   - 中等文件 (1000-10000条): 600秒 (10分钟)"
echo "   - 大文件 (>10000条): 900秒 (15分钟)"
echo ""

echo "2. 搜索超时优化:"
echo "   - 小批次 (<50条): 600秒 (10分钟)"
echo "   - 中等批次 (50-100条): 1800秒 (30分钟)"
echo "   - 大批次 (>100条): 3600秒 (60分钟)"
echo ""

echo "3. 动态超时策略:"
echo "   - 根据批次大小动态调整"
echo "   - 根据数据库大小调整"
echo "   - 考虑GPU/CPU模式差异"
echo ""

echo "4. 并行处理考虑:"
echo "   - 并行任务共享GPU资源"
echo "   - 可能需要更长的超时时间"
echo "   - 建议增加20-30%的缓冲时间"
echo ""

echo "🎯 推荐配置:"
echo ""

echo "快速测试 (5条记录):"
echo "  --db-init-timeout 300"
echo "  --search-timeout 600"
echo ""

echo "小文件处理 (Pic: 320条):"
echo "  --db-init-timeout 300"
echo "  --search-timeout 900"
echo ""

echo "中等文件处理 (Sac: 6,384条):"
echo "  --db-init-timeout 600"
echo "  --search-timeout 1800"
echo ""

echo "大文件处理 (Ec: 18,780条):"
echo "  --db-init-timeout 900"
echo "  --search-timeout 3600"
echo ""

echo "并行处理 (3个任务同时):"
echo "  --db-init-timeout 1200"
echo "  --search-timeout 4500"
echo ""

echo "⚠️  注意事项:"
echo "1. 超时过短: 可能导致正常处理被中断"
echo "2. 超时过长: 可能导致死锁或资源浪费"
echo "3. GPU初始化: 第一次使用GPU需要额外时间"
echo "4. 数据库缓存: 后续处理会更快"
echo "5. 并行竞争: 多个任务同时访问GPU/数据库"
echo ""

echo "🔧 当前设置评估:"
echo "  数据库初始化: 600秒 ✅ 合理"
echo "  搜索超时: 1800秒 ⚠️  对大文件可能不够"
echo "  建议: 增加大文件的搜索超时到3600秒"
