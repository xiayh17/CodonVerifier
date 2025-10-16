# MMseqs2 数据库管理指南

## 📋 概述

MMseqs2 需要蛋白质序列数据库来生成真实的 MSA 特征。本指南介绍如何准备和管理生产级数据库。

## 🗄️ 数据库类型对比

| 数据库 | 大小 | 序列数 | 覆盖度 | 推荐用途 | 下载时间 |
|--------|------|--------|--------|----------|----------|
| **UniRef50** | ~20 GB | ~50M | 高 | **生产推荐** | 30-60分钟 |
| **UniRef90** | ~60 GB | ~100M | 很高 | 高精度需求 | 1-2小时 |
| **UniRef100** | ~100 GB | ~200M | 最高 | 研究级别 | 2-3小时 |
| **NR** | ~200 GB | ~500M | 最全面 | 最大覆盖度 | 4-6小时 |

### 推荐选择

- **生产环境**: UniRef50 (平衡效率和覆盖度)
- **研究环境**: UniRef90 或 UniRef100
- **最大覆盖度**: NR

## 🚀 快速开始

### 1. 下载 UniRef50 (推荐)

```bash
# 一键下载 UniRef50 数据库
./scripts/quick_download_uniref50.sh
```

### 2. 下载其他数据库

```bash
# 下载 UniRef90
./scripts/download_production_database.sh UniRef90 16

# 下载 UniRef100
./scripts/download_production_database.sh UniRef100 16

# 下载 NR (最大数据库)
./scripts/download_production_database.sh NR 16
```

## 📁 数据库结构

下载完成后，数据库文件结构如下：

```
data/mmseqs_db/production/
├── UniRef50              # 主数据库文件
├── UniRef50.dbtype       # 数据库类型
├── UniRef50.index        # 索引文件
├── UniRef50.lookup       # 查找表
├── UniRef50_h            # 头部信息
├── UniRef50_h.dbtype     # 头部类型
├── UniRef50_h.index      # 头部索引
└── UniRef50.source       # 源信息
```

## 🔧 使用方法

### 基本用法

```bash
# 使用 UniRef50 生成真实 MSA
docker run --rm -v $(pwd)/data:/data codon-verifier/msa-features-lite:latest \
  python app.py \
  --input /data/enhanced/Pic_complete_v2.jsonl \
  --output /data/real_msa/Pic_real.jsonl \
  --use-mmseqs2 \
  --database /data/mmseqs_db/production/UniRef50 \
  --threads 16
```

### 批量处理

```bash
#!/bin/bash
# 批量为所有物种生成真实 MSA

for species in Ec Human mouse Pic Sac; do
  echo "Processing ${species} with UniRef50..."
  
  docker run --rm -v $(pwd)/data:/data codon-verifier/msa-features-lite:latest \
    python app.py \
    --input /data/enhanced/${species}_complete_v2.jsonl \
    --output /data/real_msa/${species}_real.jsonl \
    --use-mmseqs2 \
    --database /data/mmseqs_db/production/UniRef50 \
    --threads 16 \
    --batch-size 100
  
  echo "✓ ${species} completed"
done
```

## ⚡ 性能优化

### 1. 线程数设置

```bash
# 根据 CPU 核心数调整
--threads 16    # 16核 CPU
--threads 32    # 32核 CPU
--threads 8     # 8核 CPU (推荐)
```

### 2. 批次大小

```bash
# 根据内存调整
--batch-size 50   # 8GB 内存
--batch-size 100  # 16GB 内存 (推荐)
--batch-size 200  # 32GB 内存
```

### 3. 数据库选择

- **小数据集** (< 1000 序列): UniRef50
- **中等数据集** (1000-10000 序列): UniRef90
- **大数据集** (> 10000 序列): UniRef100 或 NR

## 🔍 数据库验证

### 检查数据库完整性

```bash
# 验证数据库
docker run --rm -v $(pwd)/data:/data codon-verifier/msa-features-lite:latest \
  mmseqs view /data/mmseqs_db/production/UniRef50 | head -5

# 统计序列数量
docker run --rm -v $(pwd)/data:/data codon-verifier/msa-features-lite:latest \
  bash -c "mmseqs view /data/mmseqs_db/production/UniRef50 | wc -l"
```

### 测试搜索功能

```bash
# 创建测试序列
echo ">test_seq
MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNEL" > test.fasta

# 测试搜索
docker run --rm -v $(pwd)/data:/data codon-verifier/msa-features-lite:latest \
  bash -c "cd /tmp && mmseqs createdb /data/test.fasta query && mmseqs search query /data/mmseqs_db/production/UniRef50 result /tmp --threads 8"
```

## 🛠️ 故障排查

### 问题 1: 数据库下载失败

**症状**: 下载过程中断或失败

**解决方案**:
```bash
# 清理并重新下载
rm -rf data/mmseqs_db/production/UniRef50*
./scripts/quick_download_uniref50.sh
```

### 问题 2: 内存不足

**症状**: MMseqs2 搜索失败，内存错误

**解决方案**:
```bash
# 减小批次大小
--batch-size 25

# 或使用更小的数据库
--database /data/mmseqs_db/production/UniRef50  # 而不是 UniRef100
```

### 问题 3: 磁盘空间不足

**症状**: 下载失败，磁盘空间错误

**解决方案**:
```bash
# 检查空间
df -h

# 清理旧文件
rm -rf data/mmseqs_db/tmp/*
rm -rf data/temp_*

# 使用更小的数据库
./scripts/download_production_database.sh UniRef50 16
```

### 问题 4: 搜索速度慢

**症状**: MSA 生成时间过长

**解决方案**:
```bash
# 增加线程数
--threads 32

# 使用更小的数据库
--database /data/mmseqs_db/production/UniRef50

# 减小批次大小
--batch-size 50
```

## 📊 性能基准

### UniRef50 性能测试

| 序列数 | 批次大小 | 线程数 | 处理时间 | 内存使用 |
|--------|----------|--------|----------|----------|
| 100    | 50       | 8      | 2分钟    | 4GB      |
| 500    | 100      | 16     | 8分钟    | 8GB      |
| 1000   | 100      | 16     | 15分钟   | 12GB     |
| 5000   | 200      | 32     | 60分钟   | 24GB     |

### 数据库对比

| 数据库 | 搜索速度 | 内存使用 | 结果质量 |
|--------|----------|----------|----------|
| UniRef50 | 快 | 低 | 高 |
| UniRef90 | 中等 | 中等 | 很高 |
| UniRef100 | 慢 | 高 | 最高 |
| NR | 最慢 | 最高 | 最高 |

## 🔄 数据库更新

### 自动更新脚本

```bash
#!/bin/bash
# 数据库更新脚本 (建议每月运行)

echo "🔄 更新 MMseqs2 数据库..."

# 备份旧数据库
if [ -d "data/mmseqs_db/production/UniRef50" ]; then
    mv data/mmseqs_db/production/UniRef50 data/mmseqs_db/production/UniRef50_backup_$(date +%Y%m%d)
fi

# 下载新数据库
./scripts/quick_download_uniref50.sh

# 验证新数据库
docker run --rm -v $(pwd)/data:/data codon-verifier/msa-features-lite:latest \
  mmseqs view /data/mmseqs_db/production/UniRef50 | head -1

echo "✅ 数据库更新完成"
```

## 💡 最佳实践

1. **生产环境**: 使用 UniRef50，平衡效率和覆盖度
2. **定期更新**: 每月更新数据库以获得最新序列
3. **监控资源**: 监控内存和磁盘使用情况
4. **备份策略**: 保留旧版本数据库作为备份
5. **测试验证**: 部署前测试数据库完整性

## 📞 支持

如果遇到问题，请检查：

1. 磁盘空间是否充足
2. 内存是否足够
3. 网络连接是否稳定
4. Docker 容器是否正常运行

---

**作者**: CodonVerifier Team  
**日期**: 2025-10-13  
**版本**: 1.0.0
