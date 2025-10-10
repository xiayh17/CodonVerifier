## Evo2微服务功能

这个Evo2微服务是一个**DNA序列分析服务**，主要功能是：

### 1. 核心功能
- **DNA序列特征提取**：使用Evo2模型对DNA序列进行深度分析
- **置信度评估**：计算序列的置信度分数、似然度、困惑度等指标
- **多后端支持**：
  - 真实的Evo2模型（本地或NVIDIA NIM API）
  - 启发式算法（当真实模型不可用时）

### 2. 输出特征
- `avg_confidence`：平均置信度
- `max_confidence`/`min_confidence`：最大/最小置信度
- `confidence_scores`：每个位置的置信度分数
- `avg_loglik`：平均对数似然度
- `perplexity`：困惑度
- `gc_content`：GC含量
- `codon_entropy`：密码子熵

## 使用方法

### 1. 通过Docker Compose运行
```bash
# 处理JSONL数据集
docker-compose -f docker-compose.microservices.yml run --rm evo2 \
    --input /data/converted/merged_dataset.jsonl \
    --output /data/output/evo2/features.json \
    --mode features

# 限制处理数量（测试用）
docker-compose -f docker-compose.microservices.yml run --rm evo2 \
    --input /data/converted/merged_dataset.jsonl \
    --output /data/output/evo2/features_test.json \
    --mode features \
    --limit 1000
```

### 2. 直接运行Python脚本
```bash
# 本地运行（需要安装依赖）
python services/evo2/app_enhanced.py \
    --input /path/to/input.jsonl \
    --output /path/to/output.json \
    --mode features
```

### 3. 环境变量配置
```bash
# 使用真实Evo2模型
export USE_EVO2_LM=1

# NVIDIA NIM API配置
export NVCF_RUN_KEY=your_api_key
export EVO2_NIM_URL=https://health.api.nvidia.com/v1/biology/arc/evo2-40b/generate
```

## 绑定挂载配置

是的，这个服务使用了多个绑定挂载：

### 1. 数据目录挂载
```yaml
volumes:
  - ./data:/data                    # 主数据目录
  - ./logs:/logs                    # 日志目录
  - ${HOME}/.cache/huggingface:/cache/huggingface  # HuggingFace缓存
```

### 2. 具体挂载点
- **`./data:/data`**：输入输出数据共享
- **`./logs:/logs`**：日志文件持久化
- **`${HOME}/.cache/huggingface:/cache/huggingface`**：模型缓存目录

### 3. 使用场景
- **输入**：从`/data/converted/`读取JSONL格式的DNA序列数据
- **输出**：将特征提取结果保存到`/data/output/evo2/`
- **缓存**：HuggingFace模型缓存到宿主机，避免重复下载

## 服务架构

### 1. 容器配置
- **基础镜像**：`nvcr.io/nvidia/pytorch:25.04-py3`
- **GPU支持**：配置了NVIDIA GPU资源
- **环境变量**：`CUDA_VISIBLE_DEVICES=0`、`HF_HOME=/cache/huggingface`

### 2. 网络配置
- 连接到`codon-network`网络
- 可以与其他微服务通信

### 3. 依赖管理
- 自动安装Evo2包和相关依赖
- 支持本地模型和云端API两种模式

这个微服务是CodonVerifier系统中负责DNA序列质量评估的关键组件，通过Evo2模型提供专业的序列分析能力。