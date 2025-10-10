# Complete Feature Generation Pipeline

## 概述

新的 `generate_complete_features.py` 脚本简化了从TSV到最终训练数据的流程，将原来的4个步骤合并为1个步骤：

### 原来的流程 (4个步骤)
```
TSV → Ec_full.jsonl → Ec_full_evo2_features.json → Ec_full_enhanced_expr.jsonl
```

### 新的流程 (1个步骤)
```
TSV → Ec_complete.jsonl
```

## 使用方法

### 基本用法
```bash
# 使用Docker (推荐)
python scripts/generate_complete_features.py \
  --input data/2025_bio-os_data/dataset/Ec.tsv \
  --output data/enhanced/Ec_complete.jsonl \
  --use-docker

# 本地运行 (需要安装所有依赖)
python scripts/generate_complete_features.py \
  --input data/2025_bio-os_data/dataset/Ec.tsv \
  --output data/enhanced/Ec_complete.jsonl \
  --no-docker
```

### 测试用法
```bash
# 限制记录数量进行测试
python scripts/generate_complete_features.py \
  --input data/2025_bio-os_data/dataset/Ec.tsv \
  --output data/enhanced/Ec_test.jsonl \
  --limit 100 \
  --use-docker
```

## 包含的特征

最终生成的 `Ec_complete.jsonl` 文件包含所有必要的特征：

### 1. 基础数据
- `sequence`: DNA序列
- `protein_aa`: 蛋白质序列
- `host`: 宿主信息
- `expression`: 表达值 (增强估计)

### 2. 结构特征 (struct_*)
- `struct_secondary_structure_content`: 二级结构含量
- `struct_disorder_score`: 无序性评分
- `struct_solvent_accessibility`: 溶剂可及性

### 3. 进化特征 (evo_*)
- `evo_conservation_score`: 保守性评分
- `evo_entropy`: 熵值
- `evo_gap_frequency`: Gap频率

### 4. Evo2特征 (evo2_*)
- `evo2_avg_confidence`: 平均置信度
- `evo2_perplexity`: 困惑度
- `evo2_gc_content`: GC含量
- `evo2_codon_entropy`: 密码子熵

### 5. 上下文特征 (ctx_*)
- `ctx_promoter_strength`: 启动子强度
- `ctx_rbs_strength`: RBS强度
- `ctx_temperature_norm`: 温度标准化
- `ctx_growth_phase_score`: 生长阶段评分

## 更新的训练脚本

`train_production.sh` 已经更新为使用新的简化流程：

```bash
# 运行完整的训练流程
./scripts/train_production.sh
```

新的训练脚本只有2个步骤：
1. **生成完整特征** (10-12小时)
2. **训练模型** (45-60分钟)

## 优势

### 1. 简化流程
- 从4个步骤减少到1个步骤
- 只需要一个输出文件
- 减少中间文件管理

### 2. 节省存储
- 原来需要3个文件 (~110MB)
- 现在只需要1个文件 (~55MB)
- 节省50%存储空间

### 3. 提高可靠性
- 减少文件依赖
- 降低出错概率
- 更容易调试

### 4. 保持功能
- 包含所有原有特征
- 保持相同的训练效果
- 向后兼容

## 测试

运行测试脚本验证功能：

```bash
python scripts/test_complete_features.py
```

## 故障排除

### 常见问题

1. **Docker服务不可用**
   ```bash
   # 使用本地模式
   python scripts/generate_complete_features.py \
     --input your_file.tsv \
     --output output.jsonl \
     --no-docker
   ```

2. **内存不足**
   ```bash
   # 限制记录数量
   python scripts/generate_complete_features.py \
     --input your_file.tsv \
     --output output.jsonl \
     --limit 1000
   ```

3. **权限问题**
   ```bash
   # 确保输出目录可写
   mkdir -p data/enhanced
   chmod 755 data/enhanced
   ```

## 性能估算

| 数据集大小 | 预计时间 | 内存使用 |
|-----------|---------|---------|
| 100条记录 | 5-10分钟 | 2-4GB |
| 1,000条记录 | 30-60分钟 | 4-8GB |
| 10,000条记录 | 3-6小时 | 8-16GB |
| 完整数据集 | 10-12小时 | 16-32GB |

*注：时间估算基于Docker模式，本地模式可能更快但需要更多依赖管理*
