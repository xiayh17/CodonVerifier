# Feature Generation Guide

本指南介绍如何为 CodonVerifier 生成完整的特征数据，包括从头开始生成和中断后恢复生成。

---

## 📋 目录

1. [概述](#概述)
2. [前置要求](#前置要求)
3. [从头生成特征](#从头生成特征)
4. [中断后恢复生成](#中断后恢复生成)
5. [进度监控](#进度监控)
6. [常见问题](#常见问题)
7. [性能优化](#性能优化)

---

## 概述

### 特征生成流程

完整的特征生成流程包含 6 个步骤：

1. **TSV → JSONL 转换** - 将原始 TSV 数据转换为 JSONL 格式
2. **Structure 特征生成** - 使用轻量级模型生成结构特征
3. **MSA 特征生成** - 生成多序列比对特征
4. **特征整合** - 将所有特征整合到统一的 JSONL 文件
5. **Evo2 特征提取** - 使用 Evo2 模型提取序列特征（最耗时）
6. **表达增强** - 增强表达水平估计

### 支持的数据集

本项目包含以下数据集（位于 `data/2025_bio-os_data/dataset/`）：

| 数据集 | 文件名 | 记录数 | 物种 |
|--------|--------|--------|------|
| **Pic** | `Pic.tsv` | 321 | Pichia/Komagataella (酵母) |
| Ec | `Ec.tsv` | 18,780 | E. coli (大肠杆菌) |
| Sac | `Sac.tsv` | 6,385 | S. cerevisiae (酿酒酵母) |
| Human | `Human.tsv` | 13,422 | Homo sapiens (人类) |
| Mouse | `mouse.tsv` | 13,254 | Mus musculus (小鼠) |

---

## 前置要求

### 环境准备

1. **Docker Desktop** - 必须启动并启用 WSL2 集成
   ```bash
   # 检查 Docker 是否运行
   docker ps
   ```

2. **Python 环境** - Python 3.8+
   ```bash
   # 激活虚拟环境（如果使用）
   source venv/bin/activate  # Linux/Mac
   # 或
   venv\Scripts\activate     # Windows
   ```

3. **Evo2 模型** - 确保模型文件已下载
   ```bash
   # 检查模型路径配置
   echo $EVO2_MODEL_PATH
   ```

4. **磁盘空间** - 确保有足够空间存储中间文件和最终输出
   - Pic 数据集：约 50 MB
   - Ec 数据集：约 5 GB
   - Human/Mouse 数据集：约 3 GB 每个

---


## 从头生成特征

### 示例：生成 Pic 数据集的完整特征

Pic 数据集是最小的数据集（321 条记录），适合用于测试和验证流程。

#### 1. 快速测试（限制 10 条记录）

```bash
cd /mnt/c/Users/xiayh17/Documents/GitHub/CodonVerifier

python scripts/generate_complete_features.py \
  --input data/2025_bio-os_data/dataset/Pic.tsv \
  --output data/enhanced/Pic_test.jsonl \
  --limit 10 \
  --use-docker
```

**预计时间：** 约 5-10 分钟

#### 2. 完整生成（全部 321 条记录）

```bash
cd /mnt/c/Users/xiayh17/Documents/GitHub/CodonVerifier

python scripts/generate_complete_features.py \
  --input data/2025_bio-os_data/dataset/Pic.tsv \
  --output data/enhanced/Pic_complete.jsonl \
  --use-docker
```

**预计时间：** 约 30-60 分钟（取决于硬件配置）

#### 3. 使用 screen 长时间运行

对于大数据集（如 Ec），推荐使用 `screen` 或 `tmux` 保持后台运行：

```bash
# 创建 screen 会话
screen -S pic_features

# 在 screen 中运行
cd /mnt/c/Users/xiayh17/Documents/GitHub/CodonVerifier

python scripts/generate_complete_features.py \
  --input data/2025_bio-os_data/dataset/Pic.tsv \
  --output data/enhanced/Pic_complete.jsonl \
  --use-docker

# 分离 screen（按 Ctrl+A 然后按 D）

# 稍后重新连接
screen -r pic_features
```

---

## 中断后恢复生成

### 场景：流程在 Evo2 特征提取时中断

假设您在运行 Pic 数据集时，流程在 Step 5（Evo2 特征提取）中断了。

#### 1. 找到临时目录

运行日志会显示临时目录路径，例如：

```
2025-10-10 01:00:00 - INFO - Using temp directory: data/temp_complete_1760010000
```

或者查找最新的临时目录：

```bash
ls -lt data/temp_complete_* | head -1
```

#### 2. 检查已完成的步骤

```bash
# 查看临时目录中的文件
ls -lh data/temp_complete_1760010000/

# 应该看到以下文件（已完成的步骤）：
# base.jsonl             - Step 1: TSV 转换 ✓
# structure.json         - Step 2: 结构特征 ✓
# msa.json               - Step 3: MSA 特征 ✓
# integrated.jsonl       - Step 4: 特征整合 ✓
# evo2_features.json     - Step 5: Evo2 特征 ✗ (未完成或部分完成)
```

#### 3. 使用恢复脚本继续运行

```bash
cd /mnt/c/Users/xiayh17/Documents/GitHub/CodonVerifier

python scripts/resume_complete_features.py \
  --temp-dir data/temp_complete_1760010000 \
  --output data/enhanced/Pic_complete.jsonl \
  --use-docker
```

**恢复脚本的智能行为：**
- ✅ 自动检测已完成的步骤并跳过
- ✅ 从中断的步骤（Evo2 特征提取）继续
- ✅ 完成剩余步骤（表达增强）

#### 4. 测试恢复（限制记录数）

如果您想先测试恢复流程是否正常，可以使用 `--limit` 参数：

```bash
python scripts/resume_complete_features.py \
  --temp-dir data/temp_complete_1760010000 \
  --output data/enhanced/Pic_test_resume.jsonl \
  --limit 10 \
  --use-docker
```

这会：
1. 从 `integrated.jsonl` 中提取前 10 条记录
2. 创建测试子集文件 `integrated_test_10.jsonl`
3. 只处理这 10 条记录的 Evo2 特征
4. 生成测试输出文件

---

## 进度监控

### 实时输出示例

运行脚本时，您会看到类似以下的实时进度输出：

```
============================================================
STEP 1: Converting TSV to JSONL
============================================================
Command: python -m codon_verifier.data_converter --input data/2025_bio-os_data/dataset/Pic.tsv --output data/temp_complete_1760010000/base.jsonl
2025-10-10 01:00:00 - INFO - Processing 321 records...
2025-10-10 01:00:05 - INFO - ✓ Converted 321 records
✓ JSONL dataset created: data/temp_complete_1760010000/base.jsonl

============================================================
STEP 2: Generating Structure Features
============================================================
Running: docker-compose -f docker-compose.microservices.yml run --rm -v /mnt/c/.../data:/data structure_features_lite --input /data/temp_complete_1760010000/base.jsonl --output /data/temp_complete_1760010000/structure.json
2025-10-10 01:00:10 - structure-service - INFO - Processing 321 sequences...
2025-10-10 01:00:15 - structure-service - INFO - Progress: 100/321 (31%)
2025-10-10 01:00:20 - structure-service - INFO - Progress: 200/321 (62%)
2025-10-10 01:00:25 - structure-service - INFO - Progress: 321/321 (100%)
✓ Structure features generated

============================================================
STEP 3: Generating MSA Features
============================================================
Running: docker-compose -f docker-compose.microservices.yml run --rm -v /mnt/c/.../data:/data msa_features_lite --input /data/temp_complete_1760010000/base.jsonl --output /data/temp_complete_1760010000/msa.json
2025-10-10 01:00:30 - msa-service - INFO - Processing 321 sequences...
2025-10-10 01:00:35 - msa-service - INFO - Progress: 100/321 (31%)
2025-10-10 01:00:40 - msa-service - INFO - Progress: 200/321 (62%)
2025-10-10 01:00:45 - msa-service - INFO - Progress: 321/321 (100%)
✓ MSA features generated

============================================================
STEP 4: Integrating All Features
============================================================
Running: docker-compose -f docker-compose.microservices.yml run --rm -v /mnt/c/.../data:/data feature_integrator --input /data/temp_complete_1760010000/base.jsonl --structure-features /data/temp_complete_1760010000/structure.json --msa-features /data/temp_complete_1760010000/msa.json --output /data/temp_complete_1760010000/integrated.jsonl
2025-10-10 01:00:50 - integrator - INFO - Integrating features for 321 records...
✓ Features integrated

============================================================
STEP 5: Extracting Evo2 Features
============================================================
Running Evo2 feature extraction...
⏱️  This may take a long time for large datasets...
2025-10-10 01:01:00 - evo2-service-enhanced - INFO - Loading Evo2 model...
2025-10-10 01:01:30 - evo2-service-enhanced - INFO - Model loaded successfully
2025-10-10 01:02:00 - evo2-service-enhanced - INFO - Progress: 100 records processed (0.50 rec/s, 99 success, 1 failed, ETA: 7.4 min)
2025-10-10 01:04:00 - evo2-service-enhanced - INFO - Progress: 200 records processed (0.50 rec/s, 198 success, 2 failed, ETA: 4.0 min)
2025-10-10 01:06:00 - evo2-service-enhanced - INFO - Progress: 300 records processed (0.50 rec/s, 297 success, 3 failed, ETA: 0.7 min)
2025-10-10 01:06:30 - evo2-service-enhanced - INFO - Processing complete: 321 records (318 success, 3 failed)
✓ Evo2 features extracted: data/temp_complete_1760010000/evo2_features.json

============================================================
STEP 6: Enhancing Expression Estimates
============================================================
Command: python scripts/enhance_expression_estimates.py --input data/temp_complete_1760010000/integrated.jsonl --output data/enhanced/Pic_complete.jsonl --evo2-results data/temp_complete_1760010000/evo2_features.json --mode model_enhanced
2025-10-10 01:06:35 - INFO - Enhancing expression estimates...
2025-10-10 01:06:40 - INFO - ✓ Enhanced 321 records
✓ Enhanced expression data created: data/enhanced/Pic_complete.jsonl

================================================================================
PIPELINE COMPLETED SUCCESSFULLY!
================================================================================
Output file: data/enhanced/Pic_complete.jsonl
Steps completed: convert, structure, msa, integrate, evo2, enhance_expression
Total time: 395.5s
================================================================================
```

### 进度报告频率

- **Step 2-4**：每处理一批记录显示一次进度
- **Step 5 (Evo2)**：默认每 **100 条记录**显示一次进度，包括：
  - 处理速度（rec/s）
  - 成功/失败计数
  - 预计剩余时间（ETA）

可以通过 `--progress-interval` 参数自定义频率（需要直接调用 Evo2 服务）。

---

## 常见问题

### Q1: 如何修改进度报告频率？

如果您想更频繁地看到 Evo2 的进度更新，可以直接调用服务：

```bash
docker-compose -f docker-compose.microservices.yml run --rm \
  -v $(pwd)/data:/data \
  evo2 \
  python services/evo2/app_enhanced.py \
  --input /data/temp_complete_xxx/integrated.jsonl \
  --output /data/temp_complete_xxx/evo2_features.json \
  --progress-interval 50  # 每 50 条记录显示一次
```

### Q2: 如何在没有 Docker 的环境下运行？

使用 `--no-docker` 标志（需要本地安装所有依赖）：

```bash
python scripts/generate_complete_features.py \
  --input data/2025_bio-os_data/dataset/Pic.tsv \
  --output data/enhanced/Pic_complete.jsonl \
  --no-docker
```

### Q3: 内存不足怎么办？

Evo2 服务已经实现了流式处理，但如果仍然遇到内存问题：

1. 使用 `--limit` 参数分批处理
2. 调整 Docker 内存限制（Docker Desktop → Settings → Resources）
3. 关闭其他占用内存的应用

### Q4: 如何清理临时文件？

临时文件保存在 `data/temp_complete_*` 目录中，可以安全删除：

```bash
# 删除特定的临时目录
rm -rf data/temp_complete_1760010000

# 清理所有临时目录（谨慎！）
rm -rf data/temp_complete_*
```

### Q5: 流程中断后，临时文件丢失了怎么办？

如果 `data/temp_complete_*` 目录被删除，您需要重新运行完整流程：

```bash
python scripts/generate_complete_features.py \
  --input data/2025_bio-os_data/dataset/Pic.tsv \
  --output data/enhanced/Pic_complete.jsonl \
  --use-docker
```

### Q6: 如何验证输出文件的完整性？

```bash
# 检查记录数
wc -l data/enhanced/Pic_complete.jsonl

# 检查文件格式（每行应该是有效的 JSON）
head -1 data/enhanced/Pic_complete.jsonl | python -m json.tool

# 检查必需字段
head -1 data/enhanced/Pic_complete.jsonl | python -c "
import json, sys
record = json.load(sys.stdin)
required_fields = ['protein_id', 'sequence', 'protein_aa', 'expression', 'structure_features', 'msa_features', 'evo2_features']
missing = [f for f in required_fields if f not in record]
if missing:
    print(f'❌ Missing fields: {missing}')
else:
    print('✅ All required fields present')
"
```

---

## 性能优化

### 预计处理时间（参考）

基于典型硬件配置（NVIDIA RTX 3090 / 24GB RAM）：

| 数据集 | 记录数 | Step 1-4 | Step 5 (Evo2) | Step 6 | 总时间 |
|--------|--------|----------|---------------|--------|--------|
| Pic | 321 | ~2 分钟 | ~5 分钟 | ~1 分钟 | **~8 分钟** |
| Sac | 6,385 | ~10 分钟 | ~2 小时 | ~5 分钟 | **~2.3 小时** |
| Human | 13,422 | ~20 分钟 | ~4.5 小时 | ~10 分钟 | **~5 小时** |
| Mouse | 13,254 | ~20 分钟 | ~4.5 小时 | ~10 分钟 | **~5 小时** |
| Ec | 18,780 | ~30 分钟 | ~6.5 小时 | ~15 分钟 | **~7 小时** |

**注意：** Evo2 特征提取占据 80-90% 的处理时间。

### 加速技巧

1. **使用 GPU** - 确保 Docker 容器能访问 GPU
   ```bash
   # 检查 GPU 可用性
   docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
   ```

2. **调整批处理大小** - 编辑 `services/evo2/app_enhanced.py` 中的批处理参数

3. **并行处理多个数据集** - 每个数据集使用独立的 Docker 容器
   ```bash
   # 终端 1
   screen -S pic
   python scripts/generate_complete_features.py --input data/.../Pic.tsv --output data/enhanced/Pic_complete.jsonl --use-docker
   
   # 终端 2
   screen -S sac
   python scripts/generate_complete_features.py --input data/.../Sac.tsv --output data/enhanced/Sac_complete.jsonl --use-docker
   ```

4. **预加载模型** - 首次运行会下载 Evo2 模型，后续运行会使用缓存

---

## 其他数据集示例

### 生成 E. coli (Ec) 数据集

```bash
# 测试 100 条记录
python scripts/generate_complete_features.py \
  --input data/2025_bio-os_data/dataset/Ec.tsv \
  --output data/enhanced/Ec_test.jsonl \
  --limit 100 \
  --use-docker

# 完整生成（推荐使用 screen）
screen -S ec_features
python scripts/generate_complete_features.py \
  --input data/2025_bio-os_data/dataset/Ec.tsv \
  --output data/enhanced/Ec_complete.jsonl \
  --use-docker
```

### 生成 Human 数据集

```bash
screen -S human_features
python scripts/generate_complete_features.py \
  --input data/2025_bio-os_data/dataset/Human.tsv \
  --output data/enhanced/Human_complete.jsonl \
  --use-docker
```

---

## 总结

| 操作 | 命令 | 使用场景 |
|------|------|----------|
| **快速测试** | `generate_complete_features.py --limit 10` | 验证流程是否正常 |
| **完整生成** | `generate_complete_features.py` | 首次生成完整特征 |
| **恢复生成** | `resume_complete_features.py --temp-dir ...` | 中断后继续 |
| **测试恢复** | `resume_complete_features.py --limit 10` | 验证恢复流程 |

---

## 相关文档

- [README.md](../README.md) - 项目总览
- [docker-compose.microservices.yml](../docker-compose.microservices.yml) - 微服务配置
- [scripts/generate_complete_features.py](../scripts/generate_complete_features.py) - 完整流程脚本
- [scripts/resume_complete_features.py](../scripts/resume_complete_features.py) - 恢复流程脚本

---

**最后更新：** 2025-10-10  
**版本：** 1.0.0

