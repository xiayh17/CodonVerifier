# 路径映射说明

## 概述

在Docker环境中，宿主机路径和容器内路径是不同的。本文档说明了所有脚本中的路径映射关系。

## 路径映射表

### 宿主机路径 → Docker容器内路径

| 宿主机路径 | Docker容器内路径 | 说明 |
|------------|------------------|------|
| `$(pwd)/data` | `/data` | 数据根目录 |
| `data/enhanced/` | `/data/enhanced/` | 输入数据目录 |
| `data/mmseqs_db/production/UniRef50` | `/data/mmseqs_db/production/UniRef50` | 数据库文件 |
| `data/real_msa/` | `/data/real_msa/` | 输出结果目录 |

## 脚本中的变量定义

### 1. 快速测试脚本 (`quick_test_config.sh`)

```bash
# 宿主机路径
HOST_DATA_DIR="$(pwd)/data"                    # 宿主机数据目录
HOST_OUTPUT_DIR="data/real_msa"                # 宿主机输出目录

# Docker容器内路径
DOCKER_DATABASE="/data/mmseqs_db/production/UniRef50"  # 数据库路径
DOCKER_OUTPUT_DIR="/data/real_msa"             # 输出目录
```

### 2. 批量处理脚本 (`batch_process_all_datasets.sh`)

```bash
# 宿主机路径
HOST_DATA_DIR="$(pwd)/data"                    # 宿主机数据目录
HOST_OUTPUT_DIR="data/real_msa"                # 宿主机输出目录

# Docker容器内路径
DOCKER_DATABASE="/data/mmseqs_db/production/UniRef50"  # 数据库路径
DOCKER_OUTPUT_DIR="/data/real_msa"             # 输出目录
```

### 3. 并行处理脚本 (`parallel_batch_process.sh`)

```bash
# 宿主机路径
HOST_DATA_DIR="$(pwd)/data"                    # 宿主机数据目录
HOST_OUTPUT_DIR="data/real_msa"                # 宿主机输出目录

# Docker容器内路径
DOCKER_DATABASE="/data/mmseqs_db/production/UniRef50"  # 数据库路径
DOCKER_OUTPUT_DIR="/data/real_msa"             # 输出目录
```

### 4. 监控脚本 (`monitor_processing.sh`)

```bash
# 宿主机路径
HOST_OUTPUT_DIR="data/real_msa"                # 宿主机输出目录
```

## Docker挂载命令

所有脚本都使用以下Docker挂载命令：

```bash
docker run --rm \
    --gpus all \
    --entrypoint="" \
    -v "$HOST_DATA_DIR":/data \              # 挂载宿主机数据目录到容器/data
    codon-verifier/msa-features-lite:latest \
    python3 app.py \
    --input "/data/enhanced/$file" \         # 容器内输入路径
    --output "/data/real_msa/output.jsonl" \ # 容器内输出路径
    --database "/data/mmseqs_db/production/UniRef50"  # 容器内数据库路径
```

## 实际路径示例

假设你的项目在 `/mnt/c/Users/xiayh17/Documents/GitHub/CodonVerifier/`：

### 宿主机实际路径
```
/mnt/c/Users/xiayh17/Documents/GitHub/CodonVerifier/data/enhanced/Pic_complete_v2.jsonl
/mnt/c/Users/xiayh17/Documents/GitHub/CodonVerifier/data/mmseqs_db/production/UniRef50
/mnt/c/Users/xiayh17/Documents/GitHub/CodonVerifier/data/real_msa/
```

### Docker容器内路径
```
/data/enhanced/Pic_complete_v2.jsonl
/data/mmseqs_db/production/UniRef50
/data/real_msa/
```

## 关键要点

1. **宿主机检查**: 脚本在宿主机上检查文件是否存在时，使用宿主机路径
2. **Docker命令**: 在Docker命令中，使用容器内路径
3. **挂载映射**: `-v "$HOST_DATA_DIR":/data` 将宿主机数据目录挂载到容器的 `/data`
4. **相对路径**: 使用 `$(pwd)/data` 确保路径正确，无论从哪个目录运行脚本

## 常见错误

### ❌ 错误示例
```bash
# 错误：在Docker命令中使用宿主机路径
--database "data/mmseqs_db/production/UniRef50"  # 容器内找不到

# 错误：在宿主机检查中使用容器路径
if [ -f "/data/enhanced/file.jsonl" ]; then  # 宿主机上找不到
```

### ✅ 正确示例
```bash
# 正确：在Docker命令中使用容器内路径
--database "/data/mmseqs_db/production/UniRef50"

# 正确：在宿主机检查中使用宿主机路径
if [ -f "data/enhanced/file.jsonl" ]; then
```

## 验证路径

可以使用以下命令验证路径映射：

```bash
# 检查宿主机文件
ls -la data/enhanced/
ls -la data/mmseqs_db/production/

# 检查Docker容器内文件
docker run --rm -v "$(pwd)/data":/data codon-verifier/msa-features-lite:latest ls -la /data/enhanced/
docker run --rm -v "$(pwd)/data":/data codon-verifier/msa-features-lite:latest ls -la /data/mmseqs_db/production/
```
