# 特征更新快速入门指南

## 概述

本指南介绍如何在**不重跑 Evo2 模型**的情况下，快速更新特征文件。

## 三种场景

### 场景 1: 仅更新快速特征（推荐用于测试）

**用途**: 更新 codon、structure、MSA（Lite）特征，保留 Evo2  
**速度**: 快 (~10 秒/1000 条)

```bash
python scripts/refresh_features_keep_evo2.py \
  --input data/enhanced/Pic_complete_v2.jsonl \
  --output data/enhanced/Pic_complete_v3.jsonl
```

### 场景 2: 加载 Evo2 推理结果 + 更新快速特征（推荐用于生产）

**用途**: 从 Evo2 JSON 加载推理结果，计算 bpb 和 Δ，更新其他特征  
**速度**: 快 (~10 秒/1000 条)

```bash
python scripts/refresh_features_keep_evo2.py \
  --input data/enhanced/Pic_complete_v2.jsonl \
  --output data/enhanced/Pic_complete_v3.jsonl \
  --evo2-json data/temp_complete_1760161459/evo2_features.json
```

### 场景 3: 加载 Evo2 + 真实 MSA（最完整）

**用途**: 使用真实 MSA 特征替代 Lite 近似  
**速度**: 取决于 MSA 生成时间

```bash
# Step 1: 生成真实 MSA（慢，一次性）
python scripts/generate_real_msa_features.py \
  --input data/enhanced/Pic_complete_v2.jsonl \
  --output data/real_msa/Pic_msa.jsonl \
  --database data/mmseqs_db/uniref50 \
  --threads 16

# Step 2: 刷新特征（快）
python scripts/refresh_features_keep_evo2.py \
  --input data/enhanced/Pic_complete_v2.jsonl \
  --output data/enhanced/Pic_complete_v3.jsonl \
  --evo2-json data/temp_complete_1760161459/evo2_features.json \
  --real-msa-jsonl data/real_msa/Pic_msa.jsonl
```

## 完整工作流程示例

### 从 TSV 到最终特征文件

```bash
# 假设你已经有了 Evo2 推理结果
EVO2_JSON="data/temp_complete_1760161459/evo2_features.json"

# 1. 如果还没有 v2 文件，先生成（跳过 Evo2）
if [ ! -f "data/enhanced/Pic_complete_v2.jsonl" ]; then
  python scripts/generate_complete_features.py \
    --input data/2025_bio-os_data/dataset/Pic.tsv \
    --output data/enhanced/Pic_complete_v2.jsonl \
    --use-docker \
    --skip-evo2
fi

# 2. 刷新特征（加载 Evo2，更新其他）
python scripts/refresh_features_keep_evo2.py \
  --input data/enhanced/Pic_complete_v2.jsonl \
  --output data/enhanced/Pic_complete_v3.jsonl \
  --evo2-json $EVO2_JSON

# 3. (可选) 如果需要真实 MSA，再运行一次
python scripts/generate_real_msa_features.py \
  --input data/enhanced/Pic_complete_v3.jsonl \
  --output data/real_msa/Pic_msa.jsonl \
  --database data/mmseqs_db/uniref50 \
  --threads 16

python scripts/refresh_features_keep_evo2.py \
  --input data/enhanced/Pic_complete_v3.jsonl \
  --output data/enhanced/Pic_complete_v3.jsonl \
  --real-msa-jsonl data/real_msa/Pic_msa.jsonl
```

## 批量处理所有物种

```bash
#!/bin/bash
# 批量更新所有物种的特征

EVO2_JSON="data/temp_complete_1760161459/evo2_features.json"

for species in Ec Human mouse Pic Sac; do
  echo "Processing ${species}..."
  
  python scripts/refresh_features_keep_evo2.py \
    --input data/enhanced/${species}_complete_v2.jsonl \
    --output data/enhanced/${species}_complete_v3.jsonl \
    --evo2-json $EVO2_JSON
  
  echo "✓ ${species} done"
done
```

## 特征对比

| 特征类型 | v2 (旧) | v3 (新) | 说明 |
|---------|---------|---------|------|
| `evo2_perplexity` | ❌ 缺失 | ✅ 从 JSON 加载 | Evo2 原始分数 |
| `evo2_avg_loglik` | ❌ 缺失 | ✅ 从 JSON 加载 | Evo2 原始分数 |
| `evo2_bpb` | ❌ 缺失 | ✅ 计算得出 | log2(perplexity) |
| `evo2_nll` | ❌ 缺失 | ✅ 计算得出 | = avg_loglik |
| `evo2_delta_bpb` | ❌ 缺失或不准 | ✅ AA+host 分组 | 组内中位数差值 |
| `evo2_delta_nll` | ❌ 缺失或不准 | ✅ AA+host 分组 | 组内中位数差值 |
| `codon_*` | ⚠️ 可能过时 | ✅ 重新计算 | 密码子特征 |
| `struct_*` | ⚠️ 可能过时 | ✅ 重新计算 | 结构特征 |
| `evo_*` (MSA) | ⚠️ Lite 近似 | ✅ Lite 或真实 | MSA 特征 |

## 验证结果

```bash
# 检查 Evo2 特征是否正确加载
head -n 1 data/enhanced/Pic_complete_v3.jsonl | \
  python3 -m json.tool | \
  grep -E "(evo2_|expression)"

# 应该看到：
# "evo2_perplexity": 3.352,
# "evo2_avg_loglik": -1.210,
# "evo2_bpb": 1.745,
# "evo2_nll": -1.210,
# "evo2_delta_bpb": 0.123,  (如果有足够样本)
# "evo2_delta_nll": -0.045,
```

## 常见问题

### Q1: 为什么 `evo2_delta_*` 为空？

**A**: 需要足够的样本来计算组内中位数。确保：
1. 提供了 `--evo2-json` 参数
2. 同一（AA序列 + host）组有多条记录

### Q2: 如何知道使用的是 Lite 还是真实 MSA？

**A**: 查看日志输出：
```
INFO - Real MSA: Yes          # 使用真实 MSA
INFO - Real MSA: No (using lite)  # 使用 Lite 近似
```

### Q3: 可以原地更新吗？

**A**: 可以！使用相同的输入输出路径：
```bash
python scripts/refresh_features_keep_evo2.py \
  --input data/enhanced/Pic_complete_v2.jsonl \
  --output data/enhanced/Pic_complete_v2.jsonl \
  --evo2-json $EVO2_JSON
```

### Q4: 如何只更新部分记录（测试）？

**A**: 使用 `--limit` 参数：
```bash
python scripts/refresh_features_keep_evo2.py \
  --input data/enhanced/Pic_complete_v2.jsonl \
  --output data/enhanced/Pic_test_v3.jsonl \
  --evo2-json $EVO2_JSON \
  --limit 100
```

## 性能对比

| 操作 | 时间（1000 条） | 时间（10000 条） |
|------|----------------|-----------------|
| 完整流水线（含 Evo2） | ~数小时 | ~数天 |
| 刷新特征（不含 Evo2） | ~10 秒 | ~1-2 分钟 |
| 生成真实 MSA | ~30 分钟 | ~5 小时 |

## 相关文档

- **特征刷新详细说明**: `docs/REFRESH_FEATURES_README.md`
- **真实 MSA 流水线**: `docs/REAL_MSA_PIPELINE.md`
- **完整特征生成**: `docs/FEATURE_GENERATION_GUIDE.md`

---

**作者**: CodonVerifier Team  
**日期**: 2025-10-12
