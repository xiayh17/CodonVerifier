# CodonVerifier 新增特征文档

**更新日期**: 2025-10-12  
**版本**: 2.0.0

## 概述

本次更新为 CodonVerifier 添加了全面的密码子使用特征分析功能，包括以下关键改进：

1. **FOP (Frequency of Optimal Codons)** - 最优密码子频率
2. **CPS (Codon Pair Score)** - 密码子对评分
3. **CPB (Codon Pair Bias)** - 密码子对偏好数据表（5个宿主）
4. **CpG/UpA 二核苷酸统计** - 二核苷酸频次和观测/期望比

---

## 1. FOP (Frequency of Optimal Codons)

### 简介
FOP 衡量序列中使用最优密码子的比例。对于每个氨基酸，最优密码子被定义为在宿主生物高表达基因中使用频率最高的密码子。

### 计算公式
```
FOP = (最优密码子数量) / (总密码子数量 - 终止密码子)
```

### 使用方法
```python
from codon_verifier.metrics import fop
from codon_verifier.hosts.tables import get_host_tables

# 获取宿主密码子使用表
usage_table, _, _ = get_host_tables('E_coli')

# 计算 FOP
sequence = "ATGCGTAAAGGC..."
fop_score = fop(sequence, usage_table)
print(f"FOP: {fop_score:.4f}")  # 范围: [0, 1]
```

### 解释
- **值域**: 0.0 - 1.0
- **高值** (>0.8): 大量使用最优密码子，适合高表达
- **中等值** (0.5-0.8): 平衡的密码子使用
- **低值** (<0.5): 较少使用最优密码子

---

## 2. CPS (Codon Pair Score)

### 简介
CPS 评估密码子对的使用偏好，基于观测频率与期望频率（假设密码子独立）的对比。

### 计算公式
```
CPS = ln(F(AB) / (F(A) × F(B)))
```
其中：
- F(AB): 密码子对 AB 的观测频率
- F(A), F(B): 单个密码子的频率

### 使用方法
```python
from codon_verifier.metrics import codon_pair_score

# 基本用法（假设密码子独立，CPS = 0）
cps = codon_pair_score(sequence, usage_table)

# 使用实际观测的密码子对频率表
cps = codon_pair_score(sequence, usage_table, codon_pair_freq_dict)
```

### 解释
- **正值**: 密码子对被过度使用（相对于期望）
- **零值**: 符合独立假设
- **负值**: 密码子对被避免使用

---

## 3. CPB (Codon Pair Bias) 数据表

### 简介
本次更新为5个主要宿主生物添加了基于文献的 CPB 数据表。CPB 分数衡量特定密码子对相对于期望值的偏离程度。

### 支持的宿主

#### 3.1 大肠杆菌 (E. coli)
```python
from codon_verifier.hosts.tables import E_COLI_CPB

# 包含 30+ 个显著偏好的密码子对
# 负值示例：CGA-CGA (-0.45), ATA-ATA (-0.28)
# 正值示例：CTG-CTG (0.35), ACC-ACC (0.25)
```

#### 3.2 人类 (Human)
```python
from codon_verifier.hosts.tables import HUMAN_CPB

# 特别关注 CpG 相关的密码子对
# 负值：CGA-CGA (-0.50), TCG-TCG (-0.35)
# 正值：CTG-CTG (0.40), ACC-ACC (0.28)
```

#### 3.3 小鼠 (Mouse)
```python
from codon_verifier.hosts.tables import MOUSE_CPB

# 与人类模式相似，略有差异
```

#### 3.4 酿酒酵母 (S. cerevisiae)
```python
from codon_verifier.hosts.tables import S_CEREVISIAE_CPB

# 反映酵母特异的密码子偏好
# 负值：CGG-CGG (-0.55), CTC-CTC (-0.35)
# 正值：TTA-TTA (0.35), AGA-AGA (0.40)
```

#### 3.5 毕赤酵母 (P. pastoris)
```python
from codon_verifier.hosts.tables import P_PASTORIS_CPB

# 工业表达系统常用，与 S. cerevisiae 模式相似
```

### 使用方法
```python
from codon_verifier.metrics import codon_pair_bias_score
from codon_verifier.hosts.tables import get_host_tables

# 获取包含 CPB 表的宿主数据
usage, trna, cpb = get_host_tables('E_coli', include_cpb=True)

# 计算平均 CPB 分数
cpb_score = codon_pair_bias_score(sequence, cpb)
print(f"Average CPB: {cpb_score:.4f}")
```

### CPB 数据来源
- 基于 Coleman et al. (2008) Science 及后续研究
- 包含显著偏离（|CPB| > 0.1）的密码子对
- 涵盖高表达基因的统计分析

---

## 4. CpG/UpA 二核苷酸统计

### 简介
CpG 和 UpA (TpA) 二核苷酸在生物学上具有重要意义：
- **CpG**: 甲基化位点，影响基因调控（原核生物中常被避免）
- **UpA (TpA)**: 影响 mRNA 稳定性和翻译效率

### 功能

#### 4.1 通用二核苷酸计数
```python
from codon_verifier.metrics import count_dinucleotides

# 计数特定二核苷酸
counts = count_dinucleotides(sequence, ["CG", "TA", "AT", "GC"])
print(counts)  # {'CG': 15, 'TA': 8, 'AT': 12, 'GC': 10}

# 计数所有16种二核苷酸
all_counts = count_dinucleotides(sequence, None)
```

#### 4.2 CpG/UpA 综合分析
```python
from codon_verifier.metrics import cpg_upa_content

stats = cpg_upa_content(sequence)
print(stats)
# {
#     'cpg_count': 15,          # CG 二核苷酸数量
#     'cpg_freq': 2.45,         # 每100个二核苷酸中的频率
#     'cpg_obs_exp': 0.73,      # 观测/期望比
#     'upa_count': 8,
#     'upa_freq': 1.31,
#     'upa_obs_exp': 0.89
# }
```

### 观测/期望比的解释
```
obs/exp = (实际计数) / (基于单核苷酸频率的期望计数)
```

- **< 1.0**: 二核苷酸被避免（under-represented）
- **= 1.0**: 符合随机期望
- **> 1.0**: 二核苷酸被过度使用（over-represented）

### 优化建议
- **原核表达** (E. coli): 
  - 避免高 CpG 含量 (obs/exp < 0.8 为佳)
  - 避免高 UpA 含量 (obs/exp < 1.0 为佳)
  
- **真核表达** (Human, CHO):
  - CpG 可以适度使用
  - 但连续 CpG 可能触发甲基化

---

## 5. 集成使用

### 5.1 在 rules_score() 中使用

所有新特征已集成到 `rules_score()` 函数中，并具有可配置的权重：

```python
from codon_verifier.metrics import rules_score
from codon_verifier.hosts.tables import get_host_tables

# 获取宿主表
usage, trna, cpb = get_host_tables('E_coli', include_cpb=True)

# 计算综合评分
scores = rules_score(
    dna=sequence,
    usage=usage,
    trna_w=trna,
    cpb=cpb,
    codon_pair_freq=None,  # 可选：提供观测频率表
    weights={
        "cai": 1.0,
        "tai": 0.5,
        "fop": 0.8,           # ⭐ 新增
        "cpb": 0.2,           # ⭐ 增强（现在有数据）
        "cps": 0.2,           # ⭐ 新增
        "cpg_penalty": -0.3,  # ⭐ 新增（惩罚高 CpG）
        "upa_penalty": -0.2,  # ⭐ 新增（惩罚高 UpA）
        # ... 其他权重
    }
)

# 查看所有指标
print(f"CAI: {scores['cai']:.4f}")
print(f"FOP: {scores['fop']:.4f}")
print(f"CPB: {scores['cpb']:.4f}")
print(f"CPS: {scores['cps']:.4f}")
print(f"CpG count: {scores['cpg_count']}")
print(f"CpG obs/exp: {scores['cpg_obs_exp']:.4f}")
print(f"UpA count: {scores['upa_count']}")
print(f"UpA obs/exp: {scores['upa_obs_exp']:.4f}")
print(f"\nTotal score: {scores['total_rules']:.4f}")
```

### 5.2 在微服务中使用

#### Sequence Analyzer 微服务

```bash
# 准备输入 JSON
cat > input.json << EOF
{
  "task": "analyze",
  "input": {
    "sequence": "ATGCGTAAAGGC...",
    "host": "E_coli",
    "forbidden_motifs": ["AAAAA", "TTTTTT"]
  },
  "metadata": {
    "request_id": "test_001"
  }
}
EOF

# 运行微服务
python services/sequence_analyzer/app.py --input input.json --output output.json

# 输出包含所有新特征：
# - cai, tai, fop
# - cpb, cps
# - cpg_count, cpg_freq, cpg_obs_exp
# - upa_count, upa_freq, upa_obs_exp
```

#### Feature Integrator 微服务

特征集成器会自动提取所有密码子特征并添加 `codon_` 前缀：

```python
# 输出特征示例
{
  "sequence": "ATGCGT...",
  "extra_features": {
    "codon_cai": 0.85,
    "codon_tai": 0.78,
    "codon_fop": 0.72,          # ⭐ 新增
    "codon_gc": 0.52,
    "codon_cpb": 0.15,          # ⭐ 增强
    "codon_cps": -0.02,         # ⭐ 新增
    "codon_cpg_count": 15.0,    # ⭐ 新增
    "codon_cpg_freq": 2.45,     # ⭐ 新增
    "codon_cpg_obs_exp": 0.73,  # ⭐ 新增
    "codon_upa_count": 8.0,     # ⭐ 新增
    "codon_upa_freq": 1.31,     # ⭐ 新增
    "codon_upa_obs_exp": 0.89,  # ⭐ 新增
    "codon_rare_runs": 2.0,
    "codon_homopolymers": 1.0,
    ...
  }
}
```

---

## 6. 测试

运行测试脚本以验证所有新功能：

```bash
# 运行测试
python test_new_features.py

# 预期输出：
# - 二核苷酸计数测试
# - 多宿主密码子指标测试
# - 集成评分测试
# ✓ ALL TESTS COMPLETED SUCCESSFULLY!
```

---

## 7. API 参考

### 新增函数

#### `fop(dna: str, usage: Dict[str, float]) -> float`
计算最优密码子频率。

**参数**:
- `dna`: DNA 序列（必须是有效的 CDS）
- `usage`: 密码子使用频率表

**返回**: 0.0 - 1.0 之间的 FOP 分数

---

#### `codon_pair_score(dna: str, usage: Dict[str,float], codon_pair_freq: Optional[Dict[str,float]] = None) -> float`
计算密码子对评分。

**参数**:
- `dna`: DNA 序列（必须是有效的 CDS）
- `usage`: 密码子使用频率表
- `codon_pair_freq`: 可选的密码子对频率表

**返回**: CPS 平均分数（可正可负）

---

#### `count_dinucleotides(seq: str, dinucleotides: Optional[List[str]] = None) -> Dict[str, int]`
计数二核苷酸出现次数。

**参数**:
- `seq`: DNA/RNA 序列
- `dinucleotides`: 要计数的二核苷酸列表（None = 所有16种）

**返回**: 二核苷酸 → 计数的字典

---

#### `cpg_upa_content(seq: str) -> Dict[str, float]`
综合分析 CpG 和 UpA 二核苷酸。

**参数**:
- `seq`: DNA/RNA 序列

**返回**: 包含计数、频率和 obs/exp 比的字典

---

### 更新的函数

#### `get_host_tables(host: str, include_cpb: bool = True) -> tuple`
获取宿主特异的表格。

**参数**:
- `host`: 'E_coli', 'Human', 'Mouse', 'S_cerevisiae', 'P_pastoris'
- `include_cpb`: 是否包含 CPB 表

**返回**: 
- 如果 `include_cpb=True`: (usage, trna, cpb)
- 如果 `include_cpb=False`: (usage, trna)

---

## 8. 最佳实践

### 8.1 密码子优化流程

```python
from codon_verifier.metrics import rules_score
from codon_verifier.hosts.tables import get_host_tables

def evaluate_sequence(sequence, host='E_coli'):
    """评估序列质量的完整流程"""
    
    # 获取宿主表
    usage, trna, cpb = get_host_tables(host, include_cpb=True)
    
    # 计算所有指标
    scores = rules_score(
        dna=sequence,
        usage=usage,
        trna_w=trna,
        cpb=cpb
    )
    
    # 检查关键指标
    quality_checks = {
        'high_cai': scores['cai'] > 0.8,
        'high_fop': scores['fop'] > 0.7,
        'good_cpb': scores['cpb'] > 0.0,
        'low_cpg': scores['cpg_obs_exp'] < 1.0,
        'low_upa': scores['upa_obs_exp'] < 1.0,
        'no_rare_runs': scores['rare_run_len'] == 0,
        'no_homopolymers': scores['homopoly_len'] == 0
    }
    
    passed = sum(quality_checks.values())
    total = len(quality_checks)
    
    print(f"Quality Score: {passed}/{total} checks passed")
    print(f"Total Rules Score: {scores['total_rules']:.4f}")
    
    return scores, quality_checks

# 使用
scores, checks = evaluate_sequence(my_sequence, 'E_coli')
```

### 8.2 宿主特异优化建议

| 宿主 | CAI 目标 | FOP 目标 | CpG 策略 | UpA 策略 |
|------|---------|---------|---------|---------|
| E. coli | >0.85 | >0.75 | 避免 (<0.8) | 避免 (<1.0) |
| Human | >0.80 | >0.70 | 适度使用 | 中等避免 |
| S. cerevisiae | >0.80 | >0.70 | 较少关注 | 较少关注 |
| P. pastoris | >0.82 | >0.72 | 较少关注 | 较少关注 |

---

## 9. 性能考虑

- **计算复杂度**: 所有新函数都是 O(n)，其中 n = 序列长度
- **内存使用**: CPB 表每个宿主约占用 5-10 KB
- **推荐**: 对于批量处理，预先加载宿主表避免重复查询

```python
# 高效的批量处理
usage, trna, cpb = get_host_tables('E_coli', include_cpb=True)

for sequence in sequences:
    # 重复使用预加载的表
    fop_score = fop(sequence, usage)
    cpb_score = codon_pair_bias_score(sequence, cpb)
    # ...
```

---

## 10. 参考文献

1. **FOP**: Ikemura, T. (1981). "Correlation between the abundance of Escherichia coli transfer RNAs and the occurrence of the respective codons in its protein genes." *Journal of Molecular Biology*.

2. **CPB & CPS**: Coleman, J. R., et al. (2008). "Virus attenuation by genome-scale changes in codon pair bias." *Science*, 320(5884), 1784-1787.

3. **CpG content**: Karlin, S., & Mrázek, J. (1997). "Compositional differences within and between eukaryotic genomes." *Proceedings of the National Academy of Sciences*.

4. **UpA content**: Mueller, S., et al. (2006). "Reduction of the rate of poliovirus protein synthesis through large-scale codon deoptimization." *Genome Biology*.

---

## 11. 版本历史

### v2.0.0 (2025-10-12)
- ✅ 实现 FOP 指标
- ✅ 实现 CPS 指标
- ✅ 添加 CPB 数据表（5个宿主）
- ✅ 实现 CpG/UpA 二核苷酸统计
- ✅ 更新 rules_score() 函数
- ✅ 更新 sequence_analyzer 微服务
- ✅ 更新 feature_integrator 微服务
- ✅ 添加测试脚本和文档

---

## 联系与支持

如有问题或建议，请提交 Issue 或 Pull Request。

**文档维护**: CodonVerifier Team  
**最后更新**: 2025-10-12

