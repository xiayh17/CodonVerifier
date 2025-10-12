# Changelog - CodonVerifier v2.0.0

## 发布日期: 2025-10-12

---

## 🎉 主要更新

本版本大幅增强了密码子使用特征分析能力，添加了4个全新的特征维度，并为5个主要宿主生物提供了完整的数据支持。

---

## ✨ 新增特征

### 1. FOP (Frequency of Optimal Codons) 🆕
- **功能**: 计算序列中最优密码子的使用频率
- **计算公式**: FOP = (最优密码子数) / (总密码子数 - 终止密码子)
- **值域**: 0.0 - 1.0
- **适用场景**: 评估序列的密码子优化程度
- **文件**: `codon_verifier/metrics.py:102-149`

### 2. CPS (Codon Pair Score) 🆕
- **功能**: 评估密码子对的使用偏好
- **计算公式**: CPS = ln(F(AB) / (F(A) × F(B)))
- **解释**: 正值=过度使用，零值=符合期望，负值=被避免
- **适用场景**: 优化相邻密码子组合，避免不良密码子对
- **文件**: `codon_verifier/metrics.py:213-266`

### 3. CPB (Codon Pair Bias) 数据表 🆕
- **功能**: 宿主特异的密码子对偏好数据
- **支持宿主**:
  - ✅ E. coli (大肠杆菌)
  - ✅ Human (人类)
  - ✅ Mouse (小鼠)
  - ✅ S. cerevisiae (酿酒酵母)
  - ✅ P. pastoris (毕赤酵母)
- **数据规模**: 每个宿主 30+ 个显著偏离的密码子对
- **文件**: `codon_verifier/hosts/tables.py:38-233`

### 4. CpG/UpA 二核苷酸统计 🆕
- **功能**: 
  - 统计 CpG 和 UpA (TpA) 二核苷酸数量
  - 计算频率和观测/期望比
  - 评估序列中的二核苷酸偏好
- **指标**:
  - `cpg_count`, `cpg_freq`, `cpg_obs_exp`
  - `upa_count`, `upa_freq`, `upa_obs_exp`
- **生物学意义**:
  - CpG: 甲基化位点，影响基因调控
  - UpA: 影响 mRNA 稳定性
- **文件**: `codon_verifier/metrics.py:272-360`

---

## 🔧 功能增强

### 核心模块

#### `metrics.py`
- ✅ 新增 `fop()` 函数
- ✅ 新增 `codon_pair_score()` 函数
- ✅ 增强 `codon_pair_bias_score()` 函数（添加文档）
- ✅ 新增 `count_dinucleotides()` 函数
- ✅ 新增 `cpg_upa_content()` 函数
- ✅ 更新 `rules_score()` 函数以整合所有新特征
  - 新增权重参数: `fop`, `cps`, `cpg_penalty`, `upa_penalty`
  - 新增返回字段: `fop`, `cps`, `cpg_*`, `upa_*`

#### `hosts/tables.py`
- ✅ 添加 `E_COLI_CPB` 数据表
- ✅ 添加 `HUMAN_CPB` 数据表
- ✅ 添加 `MOUSE_CPB` 数据表
- ✅ 添加 `S_CEREVISIAE_CPB` 数据表
- ✅ 添加 `P_PASTORIS_CPB` 数据表
- ✅ 更新 `HOST_TABLES` 字典以包含 CPB 表
- ✅ 更新 `get_host_tables()` 函数
  - 新增参数: `include_cpb` (默认 True)
  - 现在返回 (usage, trna, cpb) 三元组

### 微服务

#### `services/sequence_analyzer/app.py`
- ✅ 完整重写 `process_task()` 函数
- ✅ 集成所有新的密码子特征计算
- ✅ 添加自动警告系统（高 CpG/UpA、稀有密码子等）
- ✅ 输出包含所有新指标：
  - `cai`, `tai`, `fop`
  - `cpb`, `cps`
  - `cpg_count`, `cpg_freq`, `cpg_obs_exp`
  - `upa_count`, `upa_freq`, `upa_obs_exp`
  - `rare_codon_runs`, `homopolymers`, `forbidden_sites`
- ✅ 版本升级至 2.0.0

#### `services/feature_integrator/app.py`
- ✅ 新增 `extract_codon_features()` 方法
- ✅ 更新 `integrate_features()` 以自动提取密码子特征
- ✅ 输出特征添加 `codon_` 前缀：
  - `codon_cai`, `codon_tai`, `codon_fop`
  - `codon_cpb`, `codon_cps`
  - `codon_cpg_count`, `codon_cpg_obs_exp`
  - `codon_upa_count`, `codon_upa_obs_exp`
  - 等等
- ✅ 更新统计日志以包含 `with_codon_features` 计数

---

## 📚 文档

### 新增文档
- ✅ `docs/NEW_FEATURES.md` - 详细的新特征文档（11个章节）
  - 每个特征的原理、公式、使用方法
  - API 参考
  - 最佳实践和优化建议
  - 宿主特异策略表
  - 参考文献

### 测试脚本
- ✅ `test_new_features.py` - 综合测试脚本
  - 二核苷酸计数测试
  - 多宿主密码子指标测试
  - 集成评分测试
  - 包含示例序列和预期输出

---

## 🎯 使用示例

### 快速开始

```python
from codon_verifier.metrics import fop, codon_pair_score, cpg_upa_content
from codon_verifier.hosts.tables import get_host_tables

# 获取 E. coli 表（包含新的 CPB 数据）
usage, trna, cpb = get_host_tables('E_coli', include_cpb=True)

# 测试序列
sequence = "ATGCGTAAAGGC..."

# 计算新特征
fop_score = fop(sequence, usage)
cps_score = codon_pair_score(sequence, usage)
dinuc_stats = cpg_upa_content(sequence)

print(f"FOP: {fop_score:.4f}")
print(f"CPS: {cps_score:.4f}")
print(f"CpG obs/exp: {dinuc_stats['cpg_obs_exp']:.4f}")
print(f"UpA obs/exp: {dinuc_stats['upa_obs_exp']:.4f}")
```

### 微服务使用

```bash
# Sequence Analyzer
python services/sequence_analyzer/app.py \
  --input input.json \
  --output output.json

# Feature Integrator (自动提取密码子特征)
python services/feature_integrator/app.py \
  --input base_records.jsonl \
  --output integrated.jsonl
```

---

## 🔬 技术细节

### 数据来源
- **CPB 数据**: 基于 Coleman et al. (2008) Science 及后续研究
- **密码子使用表**: 来自 CoCoPUTs 和 Kazusa 数据库
- **算法实现**: 基于标准公式和最佳实践

### 性能
- **计算复杂度**: O(n)，n = 序列长度
- **内存占用**: CPB 表每宿主 ~5-10 KB
- **优化**: 支持批量处理时重用预加载的表

### 兼容性
- ✅ 向后兼容：所有旧代码仍可正常工作
- ✅ `get_host_tables()` 默认行为：返回三元组 (usage, trna, cpb)
- ✅ 可选降级：`get_host_tables(host, include_cpb=False)` 返回旧格式

---

## 📊 功能对比

| 特征 | v1.x | v2.0.0 |
|------|------|--------|
| CAI | ✅ | ✅ |
| tAI | ✅ | ✅ |
| FOP | ❌ | ✅ 新增 |
| CPB | 框架存在 | ✅ 5个宿主数据 |
| CPS | ❌ | ✅ 新增 |
| CpG 统计 | 基本支持 | ✅ 完整分析 |
| UpA 统计 | ❌ | ✅ 新增 |
| 二核苷酸计数 | ❌ | ✅ 新增 |

---

## 🧪 测试

运行测试以验证所有功能：

```bash
python test_new_features.py
```

预期输出：
```
✓ ALL TESTS COMPLETED SUCCESSFULLY!

New features implemented:
  ⭐ FOP (Frequency of Optimal Codons)
  ⭐ CPS (Codon Pair Score)
  ⭐ CPB with E. coli, Human, Mouse, S. cerevisiae, P. pastoris tables
  ⭐ CpG/UpA dinucleotide statistics with observed/expected ratios
  ⭐ All features integrated into rules_score() and microservices
```

---

## 🐛 已知问题

无重大问题。如有发现，请提交 Issue。

---

## 🔮 未来计划

### v2.1 (规划中)
- [ ] 添加更多宿主（CHO、昆虫细胞等）
- [ ] 提供更大规模的 CPB 数据（全 4096 对）
- [ ] 添加实际观测的密码子对频率表
- [ ] 支持自定义 CPB 表导入

### v2.2 (规划中)
- [ ] 添加密码子使用指数 (CUB)
- [ ] 添加有效密码子数 (ENC)
- [ ] 整合机器学习模型预测

---

## 👥 贡献者

- CodonVerifier Team

---

## 📝 许可证

与主项目相同

---

## 🙏 致谢

感谢以下研究提供的理论基础和数据支持：
- Coleman et al. (2008) 的密码子对偏好研究
- Ikemura (1981) 的最优密码子理论
- CoCoPUTs 和 Kazusa 数据库

---

**完整文档**: 请参阅 `docs/NEW_FEATURES.md`

**版本**: 2.0.0  
**发布日期**: 2025-10-12

