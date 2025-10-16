# 真实 MSA 特征生成流水线

## 概述

`scripts/generate_real_msa_features.py` 是一个生产级别的 MSA（Multiple Sequence Alignment）特征生成流水线，使用 **MMseqs2** 进行真实的序列比对，而不是启发式近似。

## 与 MSA Lite 的区别

| 特性 | MSA Lite (近似) | Real MSA (真实) |
|------|----------------|----------------|
| 速度 | 快 (~1000-5000 蛋白/分钟) | 慢 (~10-100 蛋白/分钟) |
| 准确性 | 近似估计 | 基于真实比对 |
| 依赖 | 无外部依赖 | 需要 MMseqs2 + 数据库 |
| 用途 | 快速筛选、测试 | 生产环境、精确分析 |

## 前置要求

### 1. 安装 MMseqs2

```bash
# Ubuntu/Debian
sudo apt-get install mmseqs2

# macOS
brew install mmseqs2

# 或从源码编译
git clone https://github.com/soedinglab/MMseqs2.git
cd MMseqs2
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j 4
sudo make install
```

验证安装：
```bash
mmseqs version
```

### 2. 下载序列数据库

推荐使用 **UniRef50** 或 **UniRef90**：

```bash
# 创建数据库目录
mkdir -p data/mmseqs_db
cd data/mmseqs_db

# 下载 UniRef50 (推荐，平衡速度和覆盖度)
mmseqs databases UniRef50 uniref50 tmp --threads 8

# 或下载 UniRef90 (更全面，但更慢)
# mmseqs databases UniRef90 uniref90 tmp --threads 8

# 清理临时文件
rm -rf tmp
```

**数据库大小参考**：
- UniRef50: ~20 GB
- UniRef90: ~60 GB

## 使用方法

### 基本用法

```bash
python scripts/generate_real_msa_features.py \
  --input data/enhanced/Pic_complete_v2.jsonl \
  --output data/real_msa/Pic_msa.jsonl \
  --database data/mmseqs_db/uniref50 \
  --threads 8
```

### 测试模式（限制记录数）

```bash
python scripts/generate_real_msa_features.py \
  --input data/enhanced/Pic_complete_v2.jsonl \
  --output data/real_msa/Pic_test_msa.jsonl \
  --database data/mmseqs_db/uniref50 \
  --threads 8 \
  --limit 10
```

### 高性能配置

```bash
python scripts/generate_real_msa_features.py \
  --input data/enhanced/Human_complete_v2.jsonl \
  --output data/real_msa/Human_msa.jsonl \
  --database data/mmseqs_db/uniref50 \
  --threads 32 \
  --batch-size 500 \
  --evalue 1e-5 \
  --min-seq-id 0.4
```

### 自定义参数

```bash
python scripts/generate_real_msa_features.py \
  --input data/enhanced/Ec_complete_v2.jsonl \
  --output data/real_msa/Ec_msa.jsonl \
  --database /path/to/custom/db \
  --threads 16 \
  --evalue 1e-3 \          # E-value 阈值
  --min-seq-id 0.3 \       # 最小序列相似度
  --coverage 0.5 \         # 最小覆盖度
  --batch-size 1000 \      # 批次大小
  --work-dir /tmp/msa_work # 工作目录
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` | 必需 | 输入 JSONL 文件（包含 `protein_aa` 字段） |
| `--output` | 必需 | 输出 JSONL 文件 |
| `--database` | `uniref50` | MMseqs2 数据库路径 |
| `--threads` | `8` | 线程数 |
| `--evalue` | `1e-3` | E-value 阈值（越小越严格） |
| `--min-seq-id` | `0.3` | 最小序列相似度 (0-1) |
| `--coverage` | `0.5` | 最小覆盖度 (0-1) |
| `--batch-size` | `1000` | 批次处理大小 |
| `--limit` | 无 | 限制处理记录数（测试用） |
| `--work-dir` | 临时目录 | 工作目录（存放中间文件） |

## 输出格式

输出 JSONL 文件，每行一条记录：

```json
{
  "protein_id": "P12345",
  "msa_features": {
    "msa_depth": 234.0,
    "msa_effective_depth": 187.0,
    "msa_coverage": 0.92,
    "conservation_mean": 0.68,
    "conservation_min": 0.42,
    "conservation_max": 0.95,
    "conservation_entropy_mean": 0.15,
    "coevolution_score": 0.52,
    "contact_density": 0.35,
    "pfam_count": 2.0,
    "domain_count": 2.0
  }
}
```

## 与特征刷新脚本集成

生成真实 MSA 后，可以用 `refresh_features_keep_evo2.py` 整合：

```bash
# 1. 生成真实 MSA 特征
python scripts/generate_real_msa_features.py \
  --input data/enhanced/Pic_complete_v2.jsonl \
  --output data/real_msa/Pic_msa.jsonl \
  --database data/mmseqs_db/uniref50 \
  --threads 16

# 2. 刷新特征文件（使用真实 MSA）
python scripts/refresh_features_keep_evo2.py \
  --input data/enhanced/Pic_complete_v2.jsonl \
  --output data/enhanced/Pic_complete_v3.jsonl \
  --real-msa-jsonl data/real_msa/Pic_msa.jsonl \
  --evo2-json data/temp_complete_xxx/evo2_features.json
```

## 批量处理所有物种

```bash
#!/bin/bash
# 批量生成所有物种的真实 MSA 特征

DB="data/mmseqs_db/uniref50"
THREADS=32

for species in Ec Human mouse Pic Sac; do
  echo "=========================================="
  echo "Processing ${species}..."
  echo "=========================================="
  
  python scripts/generate_real_msa_features.py \
    --input data/enhanced/${species}_complete_v2.jsonl \
    --output data/real_msa/${species}_msa.jsonl \
    --database $DB \
    --threads $THREADS \
    --batch-size 500
  
  if [ $? -eq 0 ]; then
    echo "✓ ${species} MSA features generated"
  else
    echo "✗ ${species} failed"
  fi
done

echo ""
echo "All done!"
```

## 性能优化建议

### 1. 线程数配置
- CPU 核心数 ≤ 8: 使用 `--threads 4-8`
- CPU 核心数 > 8: 使用 `--threads 16-32`
- 避免超过物理核心数

### 2. 批次大小
- 小数据集 (<1000): `--batch-size 100-500`
- 中等数据集 (1000-10000): `--batch-size 500-1000`
- 大数据集 (>10000): `--batch-size 1000-2000`

### 3. 数据库选择
- **快速筛选**: UniRef50
- **高精度**: UniRef90 或 UniRef100
- **自定义**: 特定物种数据库

### 4. 磁盘空间
- 预留至少 **50 GB** 用于临时文件
- 使用 SSD 可显著提升速度

## 估算运行时间

| 数据集大小 | UniRef50 | UniRef90 | 备注 |
|-----------|----------|----------|------|
| 100 条 | ~5 分钟 | ~15 分钟 | 测试用 |
| 1000 条 | ~30 分钟 | ~2 小时 | 小规模 |
| 10000 条 | ~5 小时 | ~20 小时 | 中等规模 |
| 50000 条 | ~24 小时 | ~4 天 | 大规模 |

*基于 16 核 CPU，实际时间取决于硬件配置*

## Fallback 机制

如果 MMseqs2 不可用或搜索失败，脚本会自动回退到 MSA Lite 近似：

```
WARNING - MMseqs2 not found. Will use fallback methods.
WARNING - MSA search failed, using fallback features
```

这确保流水线不会因为外部依赖而中断。

## 故障排查

### 问题 1: MMseqs2 未找到
```
ERROR - MMseqs2 not available
```
**解决**: 安装 MMseqs2 并确保在 PATH 中

### 问题 2: 数据库未找到
```
ERROR - Database not found: uniref50
```
**解决**: 下载数据库或指定正确路径

### 问题 3: 内存不足
```
ERROR - Out of memory
```
**解决**: 
- 减小 `--batch-size`
- 增加系统内存
- 使用更小的数据库

### 问题 4: 磁盘空间不足
```
ERROR - No space left on device
```
**解决**:
- 清理临时文件
- 指定 `--work-dir` 到大容量磁盘
- 减小批次大小

## 高级用法

### 使用自定义数据库

```bash
# 从 FASTA 创建 MMseqs2 数据库
mmseqs createdb sequences.fasta custom_db

# 使用自定义数据库
python scripts/generate_real_msa_features.py \
  --input data/enhanced/Pic_complete_v2.jsonl \
  --output data/real_msa/Pic_msa.jsonl \
  --database custom_db \
  --threads 16
```

### 并行处理多个物种

```bash
# 使用 GNU parallel
parallel -j 2 "python scripts/generate_real_msa_features.py \
  --input data/enhanced/{}_complete_v2.jsonl \
  --output data/real_msa/{}_msa.jsonl \
  --database data/mmseqs_db/uniref50 \
  --threads 8" ::: Ec Human mouse Pic Sac
```

## 与完整流水线集成

如果你想从头开始生成完整特征（包括真实 MSA），可以修改 `generate_complete_features.py` 的 Step 3，或者分步执行：

```bash
# Step 1-2: 生成 base + structure（快速）
python scripts/generate_complete_features.py \
  --input data/2025_bio-os_data/dataset/Pic.tsv \
  --output data/enhanced/Pic_partial.jsonl \
  --use-docker \
  --skip-evo2

# Step 3: 生成真实 MSA（慢）
python scripts/generate_real_msa_features.py \
  --input data/enhanced/Pic_partial.jsonl \
  --output data/real_msa/Pic_msa.jsonl \
  --database data/mmseqs_db/uniref50 \
  --threads 16

# Step 4: 整合所有特征
python scripts/refresh_features_keep_evo2.py \
  --input data/enhanced/Pic_partial.jsonl \
  --output data/enhanced/Pic_complete_v3.jsonl \
  --real-msa-jsonl data/real_msa/Pic_msa.jsonl \
  --evo2-json data/temp_complete_xxx/evo2_features.json
```

---

**作者**: CodonVerifier Team  
**日期**: 2025-10-12  
**版本**: 1.0.0
