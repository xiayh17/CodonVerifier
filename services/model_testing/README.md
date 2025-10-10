# Model Testing Microservice

这是CodonVerifier项目的模型测试微服务，用于对训练好的生产模型进行全面评估。

## 服务特性

- ✅ 全面的性能指标评估（R², MAE, RMSE, MAPE等）
- ✅ Conformal Prediction不确定性量化
- ✅ 特征重要性分析
- ✅ 详细的误差分析
- ✅ 自动生成高质量可视化图表
- ✅ 独立的Docker容器运行环境
- ✅ 支持直接参数和配置文件两种运行模式

## 容器架构

```
┌─────────────────────────────────────┐
│  Model Testing Service Container    │
│                                      │
│  Python 3.10 + ML Libraries:        │
│  - scikit-learn (模型评估)          │
│  - lightgbm (模型加载)              │
│  - matplotlib/seaborn (可视化)      │
│  - pandas/numpy (数据处理)          │
│                                      │
│  应用组件:                           │
│  - app.py (服务入口)                │
│  - test_production_model.py (测试)  │
│  - train_with_all_features.py (特征) │
│                                      │
└─────────────────────────────────────┘
         │
         ├─ Volume: ./data → /data
         ├─ Volume: ./models → /data/models
         ├─ Volume: ./test_results → /data/test_results
         └─ Volume: ./logs → /logs
```

## 快速使用

### 1. 构建服务镜像

```bash
docker-compose -f docker-compose.microservices.yml build model_testing
```

### 2. 运行测试（直接模式）

```bash
docker-compose -f docker-compose.microservices.yml run --rm model_testing \
  --direct \
  --model-dir /data/models/production/ecoli_20251007_194746 \
  --data /data/enhanced/Ec_full_enhanced_expr.jsonl \
  --evo2-features /data/enhanced/Ec_full_evo2_features.json \
  --limit 1000 \
  --output-dir /data/test_results/test_run
```

### 3. 运行测试（配置文件模式）

创建 `config/test_task.json`:
```json
{
  "config": {
    "model_dir": "/data/models/production/ecoli_20251007_194746",
    "data_path": "/data/enhanced/Ec_full_enhanced_expr.jsonl",
    "evo2_features_path": "/data/enhanced/Ec_full_evo2_features.json",
    "host": "E_coli",
    "limit": 1000,
    "output_dir": "/data/test_results/test_run"
  }
}
```

运行：
```bash
docker-compose -f docker-compose.microservices.yml run --rm model_testing \
  --task-file /data/config/test_task.json
```

## 推荐使用方式

**使用Shell脚本（最简单）**：
```bash
# 快速测试
./scripts/test_production.sh --quick

# 完整测试
./scripts/test_production.sh --full

# 自定义测试
./scripts/test_production.sh --model-dir models/production/ecoli_latest --limit 5000
```

## 配置参数

### 必需参数
- `model_dir`: 模型目录路径
- `data_path`: 测试数据JSONL文件路径
- `evo2_features_path`: Evo2特征JSON文件路径

### 可选参数
- `host`: 宿主生物名称（默认：E_coli）
- `limit`: 限制测试样本数量（默认：全部）
- `output_dir`: 输出目录（默认：/data/test_results）

## 输出文件

### 数据文件
- `test_results.json`: 完整的测试结果（包含所有指标）
- `service_result.json`: 服务执行信息（状态、耗时、摘要）

### 可视化图表
- `prediction_vs_true.png`: 预测值vs真实值散点图（带R²和MAE）
- `residual_plot.png`: 残差分布图
- `error_distribution.png`: 绝对误差直方图
- `uncertainty_vs_error.png`: 不确定性vs误差相关性

## 测试指标说明

### 基本回归指标
- **R² Score**: 决定系数，衡量模型拟合优度（0-1，越接近1越好）
- **MAE**: 平均绝对误差
- **RMSE**: 均方根误差
- **MAPE**: 平均绝对百分比误差
- **Median AE**: 中位数绝对误差
- **Max Error**: 最大误差

### 不确定性指标
- **Uncertainty Mean/Std**: 预测不确定性的均值和标准差
- **Coverage**: 预测区间覆盖率（应接近目标覆�率90%）
- **Interval Width**: 预测区间宽度

### 分析结果
- **Feature Importance**: 特征重要性排名
- **Feature Type Breakdown**: 不同类型特征的重要性统计
- **Worst Predictions**: 最差的10个预测样本
- **Error by Expression Range**: 不同表达水平的误差分布

## 依赖项

Python包：
- numpy, scipy: 数值计算
- scikit-learn: 模型评估指标
- pandas: 数据处理
- biopython: 生物序列处理
- lightgbm: 模型加载
- joblib: 模型序列化
- matplotlib, seaborn: 可视化

## 调试

进入容器调试：
```bash
docker-compose -f docker-compose.microservices.yml run --rm model_testing /bin/bash

# 在容器内
python3 /app/scripts/test_production_model.py \
  --model-dir /data/models/production/ecoli_latest \
  --data /data/enhanced/Ec_full_enhanced_expr.jsonl \
  --evo2-features /data/enhanced/Ec_full_evo2_features.json \
  --limit 100 \
  --output-dir /data/test_results/debug
```

## 故障排查

### 常见问题

1. **找不到模型文件**
   - 确保模型路径正确
   - 检查Volume挂载配置

2. **内存不足**
   - 使用 `--limit` 参数减少样本数
   - 增加Docker内存限制

3. **图表生成失败**
   - 检查matplotlib和seaborn是否正确安装
   - 查看容器日志

## 与其他服务的关系

```
Training Service (训练服务)
    │
    ├─ 输出: 训练好的模型
    │         └─ models/production/ecoli_*/
    ▼
Model Testing Service (测试服务)
    │
    ├─ 输入: 模型 + 测试数据
    ├─ 处理: 全面评估
    └─ 输出: 指标 + 图表
              └─ test_results/
```

## 性能考虑

- **小样本测试** (< 1000样本): 几秒钟
- **中等测试** (1000-10000样本): 30秒-2分钟
- **大规模测试** (> 10000样本): 2-10分钟

时间取决于：
- 样本数量
- 特征维度
- 是否生成图表
- 硬件性能

## 版本历史

- v1.0.0 (2025-10-07): 初始版本
  - 完整的测试功能
  - Docker微服务支持
  - 详细的可视化输出

## 维护者

CodonVerifier Team

## 许可证

与主项目保持一致
