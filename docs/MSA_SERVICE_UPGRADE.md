# MSA 服务升级说明

## 概述

`services/msa_features_lite` 已升级为支持**双模式**的 MSA 特征生成服务：

1. **Lite 模式**（默认）：快速近似，无需外部依赖
2. **Real 模式**（`--use-mmseqs2`）：真实 MMseqs2 MSA 生成

## 升级内容

### 1. Dockerfile 更新
- ✅ 安装 MMseqs2
- ✅ 添加必要的系统依赖

### 2. app.py 功能增强
- ✅ 新增 `RealMSAGenerator` 类
- ✅ 支持 `--use-mmseqs2` 参数
- ✅ 批次处理大规模数据
- ✅ 自动 fallback 到 Lite 模式

### 3. 新增参数
- `--use-mmseqs2`: 启用真实 MSA 模式
- `--database`: MMseqs2 数据库路径（默认：`/data/mmseqs_db/uniref50`）
- `--threads`: 线程数（默认：8）
- `--batch-size`: 批次大小（默认：100）

## 使用方法

### 重新构建 Docker 镜像

```bash
# 重新构建包含 MMseqs2 的镜像
docker-compose -f docker-compose.microservices.yml build msa_features_lite

# 或使用 Docker 直接构建
docker build -f services/msa_features_lite/Dockerfile -t codon-verifier/msa-features:latest .
```

### Lite 模式（默认，无变化）

```bash
# 使用 Docker Compose
docker-compose -f docker-compose.microservices.yml run --rm \
  -v $(pwd)/data:/data \
  msa_features_lite \
  --input /data/enhanced/Pic_complete_v2.jsonl \
  --output /data/enhanced/Pic_with_msa.jsonl
```

### Real 模式（新功能）

#### 前置条件：准备 MMseqs2 数据库

```bash
# 创建数据库目录
mkdir -p data/mmseqs_db
cd data/mmseqs_db

# 下载 UniRef50（推荐）
docker run --rm -v $(pwd):/data \
  codon-verifier/msa-features:latest \
  mmseqs databases UniRef50 /data/uniref50 /data/tmp --threads 8

# 清理临时文件
rm -rf tmp
cd ../..
```

#### 使用 Real 模式

```bash
# 使用 Docker Compose + MMseqs2
docker-compose -f docker-compose.microservices.yml run --rm \
  -v $(pwd)/data:/data \
  msa_features_lite \
  --input /data/enhanced/Pic_complete_v2.jsonl \
  --output /data/real_msa/Pic_msa.jsonl \
  --use-mmseqs2 \
  --database /data/mmseqs_db/uniref50 \
  --threads 16 \
  --batch-size 100
```

## 与 refresh_features_keep_evo2.py 集成

现在可以直接使用 Docker 服务生成真实 MSA，然后用 refresh 脚本整合：

```bash
# Step 1: 使用升级后的服务生成真实 MSA
docker-compose -f docker-compose.microservices.yml run --rm \
  -v $(pwd)/data:/data \
  msa_features_lite \
  --input /data/enhanced/Pic_complete_v2.jsonl \
  --output /data/real_msa/Pic_msa.jsonl \
  --use-mmseqs2 \
  --database /data/mmseqs_db/uniref50 \
  --threads 16

# Step 2: 刷新特征（整合真实 MSA + Evo2）
python scripts/refresh_features_keep_evo2.py \
  --input data/enhanced/Pic_complete_v2.jsonl \
  --output data/enhanced/Pic_complete_v3.jsonl \
  --real-msa-jsonl data/real_msa/Pic_msa.jsonl \
  --evo2-json data/temp_complete_xxx/evo2_features.json
```

## 性能对比

| 模式 | 速度（1000条） | 准确性 | 依赖 |
|------|---------------|--------|------|
| Lite | ~10 秒 | 近似 | 无 |
| Real | ~30 分钟 | 高 | MMseqs2 + 数据库 |

## 批量处理示例

```bash
#!/bin/bash
# 批量为所有物种生成真实 MSA

for species in Ec Human mouse Pic Sac; do
  echo "Processing ${species}..."
  
  docker-compose -f docker-compose.microservices.yml run --rm \
    -v $(pwd)/data:/data \
    msa_features_lite \
    --input /data/enhanced/${species}_complete_v2.jsonl \
    --output /data/real_msa/${species}_msa.jsonl \
    --use-mmseqs2 \
    --database /data/mmseqs_db/uniref50 \
    --threads 16 \
    --batch-size 100
  
  echo "✓ ${species} done"
done
```

## 故障排查

### 问题 1: MMseqs2 not found

**原因**: Docker 镜像未重新构建

**解决**:
```bash
docker-compose -f docker-compose.microservices.yml build msa_features_lite
```

### 问题 2: Database not found

**原因**: MMseqs2 数据库未下载或路径错误

**解决**:
```bash
# 检查数据库是否存在
ls -lh data/mmseqs_db/

# 确保路径正确（容器内路径为 /data/...）
--database /data/mmseqs_db/uniref50
```

### 问题 3: 内存不足

**原因**: MMseqs2 需要大量内存

**解决**:
- 减小 `--batch-size`（如 50 或 25）
- 增加 Docker 内存限制
- 使用更小的数据库

## 向后兼容性

✅ **完全向后兼容**

- 默认行为未改变（仍然是 Lite 模式）
- 现有脚本和管线无需修改
- 新功能通过 `--use-mmseqs2` 选择性启用

## 数据库大小参考

| 数据库 | 大小 | 推荐用途 |
|--------|------|---------|
| UniRef50 | ~20 GB | 平衡速度和覆盖度（推荐） |
| UniRef90 | ~60 GB | 高覆盖度 |
| UniRef100 | ~100 GB | 最全面 |

## 下一步

1. **重新构建镜像**:
   ```bash
   docker-compose -f docker-compose.microservices.yml build msa_features_lite
   ```

2. **下载数据库**（可选，仅 Real 模式需要）:
   ```bash
   mkdir -p data/mmseqs_db
   # 下载小数据库进行测试
   ./scripts/download_test_database.sh
   # 下载 UniRef50...
   # 下载生产推荐数据库
   ./scripts/quick_download_uniref50.sh
   # 下载更大数据库
   ./scripts/download_production_database.sh UniRef90 16
   ```

3. **测试 Real 模式**:
   ```bash
   docker-compose -f docker-compose.microservices.yml run --rm \
     -v $(pwd)/data:/data \
     msa_features_lite \
     --input /data/enhanced/Pic_complete_v2.jsonl \
     --output /data/real_msa/Pic_test.jsonl \
     --use-mmseqs2 \
     --database /data/mmseqs_db/uniref50 \
     --limit 10
   ```

---

**作者**: CodonVerifier Team  
**日期**: 2025-10-13  
**版本**: 2.0.0
