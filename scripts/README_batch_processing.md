# 批量处理脚本使用指南

## 概述

这套脚本专门为你的硬件配置优化，用于高效处理所有数据集：
- **硬件**: GPU 24GB, CPU 24核, 内存 127GB
- **数据库**: UniRef50 (~37GB)
- **数据规模**: 总计约52,000条记录

## 脚本说明

### 1. 配置测试脚本
```bash
bash scripts/quick_test_config.sh
```
- 验证所有配置是否正确
- 运行5条记录的快速测试
- 检查GPU、数据库、Docker等

### 2. 串行批量处理脚本
```bash
bash scripts/batch_process_all_datasets.sh
```
- 按顺序处理所有文件
- 稳定可靠，适合首次运行
- 详细的进度报告和时间统计

### 3. 并行批量处理脚本 (推荐)
```bash
bash scripts/parallel_batch_process.sh
```
- 同时处理3个文件
- 最大化硬件利用率
- 预计比串行处理快3倍

### 4. 实时监控脚本
```bash
bash scripts/monitor_processing.sh
```
- 实时显示处理进度
- 监控GPU使用情况
- 显示Docker容器状态

## 优化配置

### 批次大小优化
根据数据规模自动选择最优批次大小：

| 文件 | 记录数 | 批次大小 | 原因 |
|------|--------|----------|------|
| Pic_complete_v2.jsonl | 320 | 50 | 小文件，快速处理 |
| Sac_complete_v2.jsonl | 6,384 | 100 | 中等文件 |
| mouse_complete_v2.jsonl | 13,253 | 150 | 大文件，平衡内存 |
| Human_complete_v2.jsonl | 13,421 | 150 | 大文件，平衡内存 |
| Ec_complete_v2.jsonl | 18,780 | 200 | 最大文件，大批次 |

### 硬件资源分配
- **GPU**: 24GB VRAM，支持大批次处理
- **CPU**: 24核心，并行处理时每任务8核心
- **内存**: 127GB，支持大数据库缓存
- **超时设置**: 数据库初始化600s，搜索1800s

## 使用步骤

### 第一步：测试配置
```bash
cd /mnt/c/Users/xiayh17/Documents/GitHub/CodonVerifier
bash scripts/quick_test_config.sh
```

### 第二步：选择处理方式

#### 方式1：串行处理 (稳定)
```bash
bash scripts/batch_process_all_datasets.sh
```

#### 方式2：并行处理 (最快) - 推荐
```bash
bash scripts/parallel_batch_process.sh
```

### 第三步：监控进度 (可选)
在另一个终端运行：
```bash
bash scripts/monitor_processing.sh
```

## 预期性能

### 处理时间估算
基于硬件配置和数据库大小：

| 文件 | 记录数 | 预计时间 | 说明 |
|------|--------|----------|------|
| Pic_complete_v2.jsonl | 320 | 5-10分钟 | 小文件，快速 |
| Sac_complete_v2.jsonl | 6,384 | 30-60分钟 | 中等文件 |
| mouse_complete_v2.jsonl | 13,253 | 1-2小时 | 大文件 |
| Human_complete_v2.jsonl | 13,421 | 1-2小时 | 大文件 |
| Ec_complete_v2.jsonl | 18,780 | 2-3小时 | 最大文件 |

**总计**: 串行处理约4-7小时，并行处理约2-3小时

### 输出文件
所有结果保存在 `data/real_msa/` 目录：
- `Ec_complete_v2_msa_features.jsonl`
- `Human_complete_v2_msa_features.jsonl`
- `mouse_complete_v2_msa_features.jsonl`
- `Sac_complete_v2_msa_features.jsonl`
- `Pic_complete_v2_msa_features.jsonl`

## 注意事项

### 1. 数据库缓存
- UniRef50数据库约37GB
- 首次加载需要时间，后续处理会更快
- 建议保持Docker容器运行以保持缓存

### 2. GPU内存管理
- 24GB VRAM足够处理大批次
- 如果出现GPU内存不足，脚本会自动降级到CPU
- 监控GPU使用情况确保最佳性能

### 3. 错误处理
- 脚本包含完整的错误处理和重试机制
- 如果某个文件处理失败，会继续处理其他文件
- 详细的日志记录便于问题排查

### 4. 资源监控
- 使用 `nvidia-smi` 监控GPU状态
- 使用 `htop` 监控CPU和内存使用
- 使用监控脚本实时查看进度

## 故障排除

### 常见问题

1. **GPU内存不足**
   - 减少批次大小
   - 检查其他程序是否占用GPU

2. **数据库加载超时**
   - 增加 `--db-init-timeout` 参数
   - 检查磁盘I/O性能

3. **处理速度慢**
   - 确保GPU被正确使用
   - 检查CPU核心是否被充分利用
   - 考虑使用并行处理脚本

4. **输出文件不完整**
   - 检查磁盘空间
   - 查看错误日志
   - 重新运行失败的单个文件

### 性能优化建议

1. **首次运行**: 使用串行处理脚本确保稳定性
2. **后续运行**: 使用并行处理脚本提高效率
3. **监控资源**: 使用监控脚本实时查看状态
4. **定期清理**: 清理临时文件和日志

## 联系支持

如果遇到问题，请提供：
1. 错误日志
2. 硬件配置信息
3. 使用的脚本和参数
4. 系统资源使用情况
