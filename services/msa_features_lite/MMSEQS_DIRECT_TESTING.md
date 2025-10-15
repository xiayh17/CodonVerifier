# MMseqs2 Direct Testing Guide

## 🎯 目标
直接在Docker环境中测试MMseqs2命令，调整参数以找到最优配置。

## 🚀 快速开始

### 1. 进入Docker环境
```bash
./enter_docker.sh
```

### 2. 运行快速测试脚本
```bash
./quick_mmseqs_test.sh
```

### 3. 或者使用Python版本（更灵活）
```bash
python mmseqs_command_generator.py
```

## 📊 测试脚本功能

### 快速测试脚本 (`quick_mmseqs_test.sh`)
- ✅ 简单的bash脚本，易于使用
- ✅ 交互式菜单
- ✅ 参数修改功能
- ✅ 多种测试场景

### Python版本 (`mmseqs_command_generator.py`)
- ✅ 更灵活的参数控制
- ✅ 详细的测试结果
- ✅ 批量测试功能
- ✅ 更好的错误处理

## 🧪 测试场景

### 1. 基础测试
- **快速测试**: 1个序列
- **小测试**: 5个序列
- **中等测试**: 10个序列
- **大测试**: 50个序列

### 2. 参数优化测试
- **批次大小**: 测试不同的`--max-seqs`值
- **敏感度**: 测试不同的`-s`值
- **GPU vs CPU**: 性能对比测试

### 3. 参数调整
- 线程数 (`--threads`)
- GPU ID (`--gpu`)
- 内存限制 (`--split-memory-limit`)
- E值 (`-e`)
- 最小序列ID (`--min-seq-id`)
- 覆盖率 (`-c`)

## 🔧 关键参数说明

| 参数 | 默认值 | 说明 | 优化建议 |
|------|--------|------|----------|
| `--threads` | 20 | CPU线程数 | 增加到24-32 |
| `--max-seqs` | 1000 | 最大序列数 | 减少到500-750 |
| `-s` | 7.5 | 敏感度 | 减少到6.0-7.0 |
| `--gpu` | 0 | GPU ID | 保持0 |
| `--split-memory-limit` | 12288 | 内存限制 | 增加到16384 |

## 📈 预期优化效果

### 当前性能
- 批次时间: ~14.5分钟
- 总时间: ~7.8小时

### 优化后性能
- 批次时间: ~5-8分钟
- 总时间: ~3-5小时
- 改进幅度: 30-60%

## 🎯 推荐测试流程

### 1. 基础测试
```bash
# 测试当前配置
./quick_mmseqs_test.sh
# 选择选项3: Medium test (10 sequences)
```

### 2. 参数优化测试
```bash
# 测试不同批次大小
# 选择选项5: Test different batch sizes
```

### 3. 敏感度测试
```bash
# 测试不同敏感度
# 选择选项6: Test different sensitivity values
```

### 4. GPU vs CPU测试
```bash
# 性能对比
# 选择选项7: Test GPU vs CPU
```

## 🔍 监控命令

在测试过程中，可以在另一个终端运行：

### GPU监控
```bash
# 实时GPU使用率
nvidia-smi -l 1

# 详细GPU信息
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader,nounits
```

### 系统监控
```bash
# 系统资源
htop

# 磁盘I/O
iostat -x 1
```

## 📝 测试结果分析

### 性能指标
- **执行时间**: 每个测试的耗时
- **GPU利用率**: GPU使用百分比
- **内存使用**: 内存占用情况
- **结果数量**: 找到的匹配序列数

### 优化建议
1. **如果GPU利用率低**: 增加批次大小
2. **如果内存不足**: 减少批次大小或增加内存限制
3. **如果结果质量差**: 增加敏感度或最大序列数
4. **如果速度慢**: 减少敏感度或最大序列数

## 🚀 应用优化结果

找到最优参数后，应用到完整数据集：

```bash
# 示例：如果批次大小50是最优的
python app.py \
  --input /data/enhanced/Pic_complete_v2.jsonl \
  --output /data/real_msa/Pic_production_db_UniRef50_optimized.jsonl \
  --use-mmseqs2 \
  --database /data/mmseqs_db/production/UniRef50 \
  --use-gpu --gpu-id 0 \
  --threads 24 \
  --batch-size 50 \
  --search-timeout 1800
```

## 💡 提示

1. **从小开始**: 先用少量序列测试
2. **逐步优化**: 一次调整一个参数
3. **记录结果**: 记录每个测试的性能数据
4. **平衡质量**: 在速度和质量之间找到平衡
5. **监控资源**: 确保不超出系统限制

现在您可以开始测试了！🚀
