# 特征更新指南 - 无需重跑 Evo2

**更新日期**: 2025-10-12  
**版本**: v2.0.0

## 问题背景

您已经花费大量时间使用 Evo2 生成了序列特征，但现在添加了新的密码子特征（FOP, CPS, CPB, CpG/UpA等）。重新运行 Evo2 会造成巨大的时间浪费。

本指南提供两种方案来更新特征而不重跑 Evo2。

---

## 🚀 方案 1：使用 `update_codon_features.py`（推荐）

### 适用场景
- 已有完整的特征文件（包含 Evo2）
- 只需要添加新的密码子特征
- 同时计算 Δbpb/ΔNLL 特征

### 使用方法

```bash
# 基本用法
python scripts/update_codon_features.py \
    --input data/enhanced/Pic_test.jsonl \
    --output data/enhanced/Pic_test_v2.jsonl

# 测试模式（限制记录数）
python scripts/update_codon_features.py \
    --input data/enhanced/Pic_test.jsonl \
    --output data/enhanced/Pic_test_v2.jsonl \
    --limit 100
```

### 功能说明

#### 1. 自动提取新的密码子特征

脚本会为每条序列添加以下特征：

**密码子使用特征**:
- `codon_cai` - Codon Adaptation Index
- `codon_tai` - tRNA Adaptation Index
- `codon_fop` - Frequency of Optimal Codons ⭐ 新增
- `codon_gc` - GC含量

**密码子对特征**:
- `codon_cpb` - Codon Pair Bias（使用宿主特异数据）⭐ 新增数据
- `codon_cps` - Codon Pair Score ⭐ 新增

**二核苷酸特征**:
- `codon_cpg_count`, `codon_cpg_freq`, `codon_cpg_obs_exp` ⭐ 新增
- `codon_upa_count`, `codon_upa_freq`, `codon_upa_obs_exp` ⭐ 新增

**其他特征**:
- `codon_rare_runs`, `codon_rare_run_total_len`
- `codon_homopolymers`, `codon_homopoly_total_len`

#### 2. 自动计算 Δbpb/ΔNLL 特征

基于组内中位数（robust method）：

**计算方式**:
1. 按 `(protein_id, host)` 分组
2. 计算每组的中位数 bpb 作为参考值
3. Δbpb = bpb - 组中位数
4. ΔNLL = Δbpb × 序列长度

**新增字段**:
- `evo2_bpb` - bits per base（如果原来没有）
- `evo2_ref_bpb` - 组中位数参考值
- `evo2_delta_bpb` - 相对于组中位数的偏差
- `evo2_delta_nll` - delta negative log-likelihood（bits）

### 输出示例

```json
{
  "sequence": "ATGCCACAA...",
  "protein_id": "P48882",
  "host": "S_cerevisiae",
  "extra_features": {
    // 原有特征保持不变
    "struct_plddt_mean": 90.26,
    "evo2_loglik": -149.20,
    "evo2_perplexity": 2.778,
    
    // 新增密码子特征 ⭐
    "codon_cai": 0.8527,
    "codon_tai": 0.7891,
    "codon_fop": 0.7234,
    "codon_cpb": 0.1523,
    "codon_cps": -0.0234,
    "codon_cpg_count": 15.0,
    "codon_cpg_obs_exp": 0.8234,
    "codon_upa_count": 8.0,
    "codon_upa_obs_exp": 1.1234,
    
    // 新增 delta 特征 ⭐
    "evo2_bpb": 1.4741,
    "evo2_ref_bpb": 1.5200,
    "evo2_delta_bpb": -0.0459,
    "evo2_delta_nll": -6.7017
  }
}
```

---

## 🔧 方案 2：使用 `generate_complete_features.py` 的新参数

### 适用场景
- 从头开始处理新数据
- 需要完整的 pipeline 但想复用部分 Evo2 特征

### 新增参数

#### `--skip-evo2`
完全跳过 Evo2 调用，只计算其他特征。

```bash
python scripts/generate_complete_features.py \
    --input data/raw/Ec.tsv \
    --output data/enhanced/Ec_v2.jsonl \
    --skip-evo2 \
    --no-docker
```

**用途**:
- 快速测试其他特征
- Evo2 服务不可用时
- 只需要轻量级特征

#### `--reuse-evo2-from`
从已有文件复用 Evo2 特征。

```bash
python scripts/generate_complete_features.py \
    --input data/raw/Ec.tsv \
    --output data/enhanced/Ec_v2.jsonl \
    --reuse-evo2-from data/enhanced/Ec_v1.jsonl \
    --no-docker
```

**工作原理**:
1. 从旧文件加载 Evo2 特征到缓存
2. 使用 `(protein_id, sequence_hash)` 作为匹配键
3. 计算其他所有特征
4. 将缓存的 Evo2 特征合并到结果中

**匹配策略**:
- 优先使用 `protein_id` + `sequence SHA256` 前16位
- 如果没有 `protein_id`，使用完整序列哈希
- 确保相同序列的 Evo2 特征被正确复用

---

## 📊 性能对比

| 方案 | 完整特征（1000序列） | 跳过 Evo2 | 复用 Evo2 |
|------|---------------------|-----------|-----------|
| **时间** | ~6-8 小时 | ~10 分钟 | ~15 分钟 |
| **Evo2** | 完整计算 | 跳过 | 从缓存 |
| **其他特征** | 完整计算 | 完整计算 | 完整计算 |
| **适用场景** | 新数据 | 测试/调试 | 更新特征 |

**时间节省**: 复用 Evo2 可节省 95%+ 的时间！

---

## 🎯 推荐工作流

### 场景 A：更新现有结果文件
```bash
# 1. 使用 update_codon_features.py（最简单）
python scripts/update_codon_features.py \
    --input data/enhanced/Pic_test.jsonl \
    --output data/enhanced/Pic_test_v2.jsonl

# 2. 验证结果
python -c "
import json
with open('data/enhanced/Pic_test_v2.jsonl') as f:
    rec = json.loads(f.readline())
    extra = rec['extra_features']
    print('New features:', [k for k in extra.keys() if k.startswith('codon_')])
    print('Delta features:', [k for k in extra.keys() if 'delta' in k])
"
```

### 场景 B：处理新 TSV 但复用 Evo2
```bash
# 1. 第一次完整处理（包含 Evo2）
python scripts/generate_complete_features.py \
    --input data/raw/Ec.tsv \
    --output data/enhanced/Ec_v1.jsonl \
    --use-docker

# 2. 后续更新（复用 Evo2）
python scripts/generate_complete_features.py \
    --input data/raw/Ec_updated.tsv \
    --output data/enhanced/Ec_v2.jsonl \
    --reuse-evo2-from data/enhanced/Ec_v1.jsonl \
    --no-docker
```

### 场景 C：快速测试
```bash
# 跳过 Evo2 进行快速测试
python scripts/generate_complete_features.py \
    --input data/raw/test.tsv \
    --output data/enhanced/test.jsonl \
    --skip-evo2 \
    --limit 100 \
    --no-docker
```

---

## 💡 技术细节

### Δbpb 计算原理

**bpb (bits per base)** 是模型对序列的困惑度度量：
- bpb 越低 = 模型越"喜欢"这个序列
- bpb 越高 = 序列越"奇怪"

**从现有特征计算 bpb**:

```python
import math

# 方法 1: 从 perplexity
if 'evo2_perplexity' in features:
    bpb = math.log2(features['evo2_perplexity'])

# 方法 2: 从 avg_loglik (nats)
if 'evo2_avg_loglik' in features:
    bpb = -features['evo2_avg_loglik'] / math.log(2)

# 方法 3: 直接使用（如果存在）
if 'evo2_bpb' in features:
    bpb = features['evo2_bpb']
```

**Δbpb 的优势**:
1. **组内比较**: 只比较同一蛋白在同一宿主的变体
2. **鲁棒性**: 使用中位数作为参考，不受异常值影响
3. **可解释性**: Δbpb < 0 表示比平均水平更优化

### 特征缓存机制

**稳定键生成**:
```python
import hashlib

protein_id = record['protein_id']
sequence = record['sequence']

# 生成16位序列哈希
seq_hash = hashlib.sha256(sequence.encode()).hexdigest()[:16]

# 组合键
key = f"{protein_id}_{seq_hash}"
```

**为什么使用序列哈希**:
- 确保相同序列匹配（即使 ID 不同）
- 避免序列本身过长
- SHA256 保证唯一性

---

## 🐛 常见问题

### Q1: 更新后的文件是否向后兼容？

**A**: 完全兼容。新特征都添加到 `extra_features` 中，原有字段保持不变。

### Q2: Δbpb 为什么使用中位数而不是平均值？

**A**: 中位数更鲁棒：
- 不受异常值影响
- 对于偏态分布更准确
- 计算简单高效

### Q3: 如果某些序列在缓存中找不到会怎样？

**A**: 
- `update_codon_features.py`: 仅计算新特征，Evo2 字段保留原值
- `generate_complete_features.py --reuse-evo2-from`: 未匹配的序列 Evo2 特征为空

### Q4: 可以同时使用 `--skip-evo2` 和 `--reuse-evo2-from` 吗？

**A**: 不可以。`--reuse-evo2-from` 会自动禁用 Evo2 计算。

### Q5: 计算 Δbpb 需要多少条记录？

**A**: 建议每组（protein_id, host）至少有 3 条记录以获得有意义的中位数。单条记录的 Δbpb 会是 0。

---

## ✅ 验证清单

更新特征后，建议进行以下检查：

```bash
# 1. 检查文件记录数
wc -l data/enhanced/Pic_test_v2.jsonl

# 2. 检查新特征是否存在
python -c "
import json
with open('data/enhanced/Pic_test_v2.jsonl') as f:
    rec = json.loads(f.readline())
    extra = rec['extra_features']
    
    # 检查密码子特征
    codon_features = [k for k in extra.keys() if k.startswith('codon_')]
    print(f'Codon features: {len(codon_features)}')
    print(codon_features[:5])
    
    # 检查 delta 特征
    has_delta = 'evo2_delta_bpb' in extra
    print(f'Has delta features: {has_delta}')
    
    # 检查 Evo2 保留
    has_evo2 = 'evo2_perplexity' in extra
    print(f'Evo2 preserved: {has_evo2}')
"

# 3. 统计特征分布
python -c "
import json
import statistics

fpbs = []
cpgs = []
with open('data/enhanced/Pic_test_v2.jsonl') as f:
    for line in f:
        rec = json.loads(line)
        extra = rec['extra_features']
        if 'codon_fop' in extra:
            fops.append(extra['codon_fop'])
        if 'codon_cpg_obs_exp' in extra:
            cpgs.append(extra['codon_cpg_obs_exp'])

if fops:
    print(f'FOP: mean={statistics.mean(fops):.3f}, median={statistics.median(fops):.3f}')
if cpgs:
    print(f'CpG obs/exp: mean={statistics.mean(cpgs):.3f}, median={statistics.median(cpgs):.3f}')
"
```

---

## 📚 相关文档

- **新特征详解**: `docs/NEW_FEATURES.md`
- **完整 Pipeline**: `docs/FEATURE_GENERATION_GUIDE.md`
- **更新日志**: `CHANGELOG_v2.0.0.md`

---

## 🎉 总结

使用本指南的方法，您可以：

✅ **节省 95%+ 时间** - 无需重跑 Evo2  
✅ **添加所有新特征** - FOP, CPS, CPB, CpG/UpA  
✅ **自动计算 Δbpb** - 基于鲁棒的组内中位数  
✅ **保持向后兼容** - 原有特征完全保留  
✅ **灵活选择方案** - 根据场景选择最佳工具

**推荐使用**: `update_codon_features.py` - 简单、快速、完整

---

**文档维护**: CodonVerifier Team  
**最后更新**: 2025-10-12

