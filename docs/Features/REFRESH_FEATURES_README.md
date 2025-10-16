# 特征刷新脚本使用说明

## 概述

`scripts/refresh_features_keep_evo2.py` 是一个轻量级特征更新脚本，用于在**不重跑 Evo2 模型推理**的情况下，快速更新其他特征。

## 特征处理策略

### 保留/加载的特征
- **Evo2 原始分数**：`evo2_perplexity`, `evo2_avg_loglik`, `evo2_loglik`, `evo2_geom`
  - 从现有 JSONL 文件中保留
  - 或从 `--evo2-json` 参数指定的 JSON 文件加载
- **Expression**：原始表达量估计值

### 计算/补充的特征
- **evo2_bpb**：从 `perplexity` 计算得出 = `log2(perplexity)`
- **evo2_nll**：与 `avg_loglik` 相同（负对数似然）
- **evo2_delta_bpb** 和 **evo2_delta_nll**：基于（氨基酸序列 + host）分组的中位数差值

### 重新计算的特征
- **codon_\***：密码子使用特征（CAI, TAI, FOP, GC, CPB 等）
- **struct_\***：结构特征（使用 Lite 近似，不调用 AlphaFold DB）
- **evo_\***：MSA 特征（可选使用真实 MSA 或 Lite 近似）

## 使用方法

### 基本用法（仅更新 codon + structure + Δ）
```bash
python scripts/refresh_features_keep_evo2.py \
  --input data/enhanced/Pic_complete_v2.jsonl \
  --output data/enhanced/Pic_complete_v3.jsonl
```

### 加载 Evo2 推理结果
如果你有 Evo2 模型的推理输出（JSON 格式），可以用它来补充 Evo2 特征：

```bash
python scripts/refresh_features_keep_evo2.py \
  --input data/enhanced/Pic_complete_v2.jsonl \
  --output data/enhanced/Pic_complete_v3.jsonl \
  --evo2-json data/temp_complete_xxx/evo2_features.json
```

**Evo2 JSON 格式要求**：
```json
[
  {
    "task": "extract_features",
    "status": "success",
    "output": {
      "sequence": "ATG...",
      "sequence_length": 141,
      "loglik": -175.83,
      "avg_loglik": -1.256,
      "perplexity": 3.511,
      "geom": 0.285
    },
    "metadata": {"request_id": "record_0"}
  },
  ...
]
```

### 使用真实 MSA 特征
```bash
python scripts/refresh_features_keep_evo2.py \
  --input data/enhanced/Pic_complete_v2.jsonl \
  --output data/enhanced/Pic_complete_v3.jsonl \
  --real-msa-jsonl data/real_msa/Pic_msa.jsonl
```

### 完整示例（Evo2 + 真实 MSA）
```bash
python scripts/refresh_features_keep_evo2.py \
  --input data/enhanced/Pic_complete_v2.jsonl \
  --output data/enhanced/Pic_complete_v3.jsonl \
  --evo2-json data/temp_complete_xxx/evo2_features.json \
  --real-msa-jsonl data/real_msa/Pic_msa.jsonl
```

### 原地更新（覆盖输入文件）
```bash
python scripts/refresh_features_keep_evo2.py \
  --input data/enhanced/Pic_complete_v2.jsonl \
  --output data/enhanced/Pic_complete_v2.jsonl \
  --evo2-json data/temp_complete_xxx/evo2_features.json
```

### 测试模式（限制记录数）
```bash
python scripts/refresh_features_keep_evo2.py \
  --input data/enhanced/Pic_complete_v2.jsonl \
  --output data/enhanced/Pic_test_v3.jsonl \
  --evo2-json data/temp_complete_xxx/evo2_features.json \
  --limit 100
```

## 批量更新所有物种

```bash
#!/bin/bash
# 更新所有 v2 文件到 v3

EVO2_DIR="data/temp_complete_1760161459"  # 你的 Evo2 输出目录

for species in Ec Human mouse Pic Sac; do
  echo "Processing ${species}..."
  python scripts/refresh_features_keep_evo2.py \
    --input data/enhanced/${species}_complete_v2.jsonl \
    --output data/enhanced/${species}_complete_v3.jsonl \
    --evo2-json ${EVO2_DIR}/evo2_features.json
done
```

## 性能说明

- **快速**：不需要重跑 Evo2 模型（最耗时的步骤）
- **轻量**：codon、structure（Lite）、MSA（Lite）计算都很快
- **灵活**：可以选择性地提供 Evo2 JSON 或真实 MSA

典型速度：
- 1000 条记录：~10 秒
- 10000 条记录：~1-2 分钟

对比完整管线（包含 Evo2）：
- 1000 条记录：~数小时（取决于 Evo2 推理速度）

## 输出示例

```json
{
  "protein_id": "P12345",
  "sequence": "ATGAGTGGC...",
  "host": "S_cerevisiae",
  "protein_aa": "MSGK...",
  "expression": {"value": 50.0, "unit": "estimated"},
  
  "evo2_perplexity": 3.352,
  "evo2_avg_loglik": -1.210,
  "evo2_bpb": 1.745,
  "evo2_nll": -1.210,
  "evo2_delta_bpb": 0.123,
  "evo2_delta_nll": -0.045,
  
  "codon_gc": 0.177,
  "codon_cai": 0.85,
  "codon_tai": 0.78,
  
  "struct_plddt_mean": 90.26,
  "struct_disorder_ratio": 0.084,
  
  "evo_msa_depth": 121.27,
  "evo_conservation_mean": 0.592
}
```

## 注意事项

1. **Evo2 JSON 顺序**：Evo2 JSON 中的记录顺序必须与输入 JSONL 文件一致（按行号对应）
2. **分组统计**：Δ 特征基于（氨基酸序列 + host）分组计算，需要足够的样本才有意义
3. **原地更新**：使用相同的输入输出路径时，脚本会先写入临时文件再替换，确保安全

## 常见问题

**Q: 为什么 group_stats 为 0？**  
A: 如果输入文件中没有 `evo2_bpb` 或 `evo2_nll` 值，无法计算分组统计。请提供 `--evo2-json` 参数。

**Q: 可以只更新部分特征吗？**  
A: 目前脚本会更新所有快速特征。如果只想更新 codon 特征，请使用 `scripts/update_codon_features.py`。

**Q: 如何验证 Evo2 特征是否正确加载？**  
A: 检查输出文件的第一条记录：
```bash
head -n 1 output.jsonl | python3 -m json.tool | grep evo2_
```

---

**作者**: CodonVerifier Team  
**日期**: 2025-10-12
