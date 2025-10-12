# 蛋白质结构特征服务 - AlphaFold DB 集成版

## 🎯 核心改进

将原有的 `StructureFeaturesLite` 服务改造为**智能双模式**：

1. **AlphaFold DB API（优先）** - 对已知 UniProt 条目，直接获取真实的 AlphaFold 预测数据
2. **Lite 近似（回退）** - API 失败或无 UniProt ID 时，使用序列属性快速估算

## ✨ 主要特性

### 从 AlphaFold DB 获取（当可用时）
- ✅ **真实 pLDDT 分数** - 来自 AlphaFold 的实际预测
- ✅ **模型元数据** - 创建日期、版本信息
- ✅ **置信度分类** - very high, confident, low, very low
- ✅ **下载链接** - PDB/mmCIF 结构文件
- ✅ **PAE 可用性** - 预测对齐误差信息

### Lite 近似（回退模式）
- ✅ **pLDDT 估算** - 基于无序度/复杂度
- ✅ **二级结构预测** - α-螺旋/β-折叠/无规卷曲
- ✅ **无序区域检测**
- ✅ **信号肽/跨膜螺旋预测**
- ✅ **表面积估算**

## 🚀 快速开始

```bash
# 1. 安装依赖
pip install requests

# 2. 基础用法（AFDB + Lite 自动回退）
python app.py --input proteins.jsonl --output features.jsonl

# 3. 仅使用 Lite 模式（不调用 API）
python app.py --input proteins.jsonl --output features.jsonl --no-afdb
```

## 📝 输入格式

支持**混合输入**，可以只有序列、只有 UniProt ID，或两者都有：

```json
{
  "protein_id": "protein_001",
  "protein_aa": "MKTAYIAKQR...",
  "uniprot_id": "P12345"
}
```

### 处理优先级
1. 如果有 `uniprot_id` → 优先尝试 AlphaFold DB API
2. API 失败或无 UniProt ID → 使用 `protein_aa` 序列做 Lite 近似
3. 两者都失败 → 返回默认值并警告

## 📤 输出格式

输出保留所有输入字段，并新增 `structure_features`：

```json
{
  "protein_id": "protein_001",
  "uniprot_id": "P12345",
  "structure_features": {
    "source": "afdb",           // 数据来源："afdb" 或 "lite"
    "plddt_mean": 82.5,
    "plddt_std": 8.3,
    "disorder_ratio": 0.12,
    
    // AFDB 专属字段（当 source="afdb" 时）
    "uniprot_accession": "P12345",
    "afdb_confidence": "very high",
    "model_created": "2021-11-01",
    "pae_available": true,
    "pdb_url": "https://alphafold.ebi.ac.uk/files/...",
    "cif_url": "https://alphafold.ebi.ac.uk/files/...",
    
    // Lite 专属字段
    "helix_ratio": 0.35,
    "sheet_ratio": 0.28,
    "coil_ratio": 0.37
  }
}
```

## 📊 统计输出示例

```
============================================================
✓ 结构特征生成完成！
============================================================
  总处理数: 1000
  AFDB 成功: 850
  AFDB 失败: 150
  Lite 使用: 150
  AFDB 成功率: 85.0%
  错误数: 0
  输出文件: features.jsonl (1000 条记录)
============================================================
```

## 🧪 测试和演示

```bash
# 方式 1: 运行完整测试套件
python test_afdb_integration.py

# 方式 2: 运行快速示例
python quickstart.py

# 方式 3: Bash 演示脚本
./demo.sh
```

## 🔧 命令行参数

```bash
python app.py [选项]

必需参数:
  --input PATH          输入 JSONL 文件
  --output PATH         输出 JSONL 文件

可选参数:
  --limit N            只处理前 N 条记录
  --no-afdb            禁用 AlphaFold DB（仅用 Lite）
  --afdb-retry N       API 重试次数（默认: 2）
  --log-level LEVEL    日志级别: DEBUG, INFO, WARNING, ERROR
```

## 📖 使用场景

### 场景 1: 已知蛋白质（推荐用 AFDB）

```json
// 输入：包含 UniProt ID 的已知蛋白
{"protein_id": "BRCA1", "uniprot_id": "P38398", "protein_aa": "MDLSALR..."}

// 输出：source="afdb"，真实的 AlphaFold 数据
```

### 场景 2: 合成/新序列（自动用 Lite）

```json
// 输入：只有序列，无 UniProt ID
{"protein_id": "synthetic_001", "protein_aa": "MKTAYIAKQR..."}

// 输出：source="lite"，快速近似估算
```

### 场景 3: 混合数据集（推荐默认配置）

```json
// 部分有 uniprot_id，部分只有序列
// 自动选择最佳方式：有 ID 用 AFDB，否则用 Lite
```

## 🎓 实际应用建议

1. **参考/天然蛋白** → 提供 `uniprot_id`，使用 AFDB 获取真实数据
2. **优化/合成变体** → 只提供序列，使用 Lite 快速估算
3. **混合数据集** → 两种字段都提供，启用自动回退（默认）
4. **离线环境** → 使用 `--no-afdb` 标志，完全本地计算

## 🔗 AlphaFold DB API 详情

### 端点
```
https://alphafold.ebi.ac.uk/api/prediction/{uniprot_accession}
```

### 行为
- **超时**: 30 秒
- **重试**: 可配置（默认 2 次）
- **404 处理**: 静默回退到 Lite
- **错误恢复**: 自动降级，不中断流程

### API 变更说明

AlphaFold DB 在 2025-10-07 进行了 breaking changes。本实现已适配：
- 尝试多种字段名称 (`pLDDT`, `confidenceScore`, `summary`)
- 灵活解析下载链接
- 健壮的错误处理

参考：https://www.ebi.ac.uk/pdbe/news/breaking-changes-afdb-predictions-api

## 🛠️ 故障排查

### 问题：AFDB 成功率为 0%

**可能原因及解决方案：**
- `requests` 未安装 → `pip install requests`
- 网络连接问题 → 检查防火墙/代理
- UniProt ID 无效 → 验证 ID 是否在 AFDB 中
- API 停机 → 使用 `--no-afdb` 跳过 AFDB

### 问题：AFDB 失败率高

**可能原因及解决方案：**
- UniProt ID 不在 AlphaFold DB 中 → 正常，会自动回退到 Lite
- 超时 → 增加重试次数 `--afdb-retry 5`
- API 限流 → 批处理任务间添加延迟

### 问题：全部使用 Lite

**可能原因及解决方案：**
- 输入无 `uniprot_id` 字段 → 添加 UniProt 映射
- 使用了 `--no-afdb` → 移除该标志
- `requests` 未安装 → 安装 requests 库

## 📚 完整文档

- **README.md** - 英文快速指南
- **README_AFDB_INTEGRATION.md** - 完整集成文档（英文）
- **CHANGELOG.md** - 变更日志
- **example_input.jsonl** - 示例输入数据

## 🔍 API 使用示例

### Python API

```python
from app import StructureFeaturesLite

# 初始化（启用 AFDB）
predictor = StructureFeaturesLite(use_afdb=True, afdb_retry=2)

# 场景 1: 有 UniProt ID（优先用 AFDB）
features = predictor.predict_structure(
    uniprot_id="P12345",
    aa_sequence="MKTAYIAKQR...",  # 回退用
    protein_id="protein_001"
)

# 场景 2: 仅序列（用 Lite）
features = predictor.predict_structure(
    aa_sequence="MKTAYIAKQR...",
    protein_id="protein_002"
)

# 检查数据来源
if features.source == "afdb":
    print(f"来自 AFDB: 置信度 {features.afdb_confidence}")
    print(f"pLDDT 均值: {features.plddt_mean:.1f}")
    print(f"PDB 下载: {features.pdb_url}")
else:
    print(f"Lite 近似: pLDDT 均值 {features.plddt_mean:.1f}")
```

### 批处理 API

```python
from app import process_jsonl

stats = process_jsonl(
    input_path="data/proteins.jsonl",
    output_path="data/features.jsonl",
    use_afdb=True,
    afdb_retry=2
)

print(f"处理完成:")
print(f"  总数: {stats['total_processed']}")
print(f"  AFDB: {stats['afdb_success']} ({stats['afdb_success_rate']:.1f}%)")
print(f"  Lite: {stats['lite_used']}")
```

## 🎯 关键优势

### vs 纯 Lite 模式
- ✅ 对已知蛋白，获得**真实的 AlphaFold 预测**
- ✅ pLDDT 分数**更准确**（来自实际模型）
- ✅ 额外的**模型元数据**和**下载链接**

### vs 纯 AFDB 模式
- ✅ **无缝回退** - 无 UniProt ID 时不会失败
- ✅ **离线可用** - 可禁用 AFDB 完全本地运行
- ✅ **快速处理** - 合成序列直接用 Lite

### 最佳实践
- ✅ **智能路由** - 自动选择最佳数据源
- ✅ **健壮性** - API 失败不影响流程
- ✅ **向后兼容** - 现有工作流无需修改

## 📦 依赖

```bash
# 必需（用于 AFDB API）
pip install requests

# 可选
# - 无 requests 时自动降级到 Lite-only 模式
# - 其他依赖均为 Python 标准库
```

## 🙏 致谢

- **AlphaFold Database** - 提供免费的程序化访问
- **EBI/EMBL** - 托管和维护 AFDB 基础设施
- **DeepMind/Google** - AlphaFold 及公开预测数据

## 📄 许可

与父项目相同 - 参见 [LICENSE](../../LICENSE)

---

**CodonVerifier 项目的一部分**

