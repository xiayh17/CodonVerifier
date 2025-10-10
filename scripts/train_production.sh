#!/bin/bash
# 生产级模型训练脚本
# 完整的端到端训练流程

set -e  # 遇到错误立即退出

echo "🚀 Starting Production Model Training Pipeline"
echo "================================================"
echo ""

# 配置
INPUT_TSV="data/2025_bio-os_data/dataset/Ec.tsv"
FINAL_DATA="data/enhanced/Ec_complete.jsonl"
OUTPUT_DIR="models/production/ecoli_$(date +%Y%m%d_%H%M%S)"
HOST="E_coli"

# 检查输入文件
if [ ! -f "$INPUT_TSV" ]; then
    echo "❌ Error: Input file not found: $INPUT_TSV"
    exit 1
fi

echo "📁 Configuration:"
echo "   Input: $INPUT_TSV"
echo "   Final data: $FINAL_DATA"
echo "   Output: $OUTPUT_DIR"
echo "   Host: $HOST"
echo ""

# 创建输出目录
mkdir -p $OUTPUT_DIR
mkdir -p data/enhanced

# 步骤 1: 生成完整特征 (一步到位)
if [ ! -f "$FINAL_DATA" ]; then
    echo "==========================================="
    echo "📊 Step 1/2: Generating Complete Features"
    echo "==========================================="
    echo ""
    echo "⏱️  Estimated time: ~10-12 hours for full dataset"
    echo "   (includes: TSV conversion + structure + MSA + Evo2 + expression enhancement)"
    echo "   (or use --limit for testing)"
    echo ""
    
    python scripts/generate_complete_features.py \
      --input $INPUT_TSV \
      --output $FINAL_DATA \
      --use-docker
    
    if [ $? -ne 0 ]; then
        echo "❌ Complete feature generation failed!"
        exit 1
    fi
    
    echo "✅ Complete feature generation completed!"
    echo ""
else
    echo "✓ Complete features already exist: $FINAL_DATA"
    echo "  Skipping feature generation..."
    echo ""
fi

# 步骤 2: 训练模型
echo "=========================================="
echo "🎯 Step 2/2: Training Enhanced Model"
echo "=========================================="
echo ""
echo "⏱️  Estimated time: ~45-60 minutes"
echo ""

python scripts/train_with_all_features.py \
  --data $FINAL_DATA \
  --output-dir $OUTPUT_DIR \
  --host $HOST \
  --use-ensemble \
  --use-conformal \
  --n-models 5 \
  --n-estimators 500 \
  --learning-rate 0.05 \
  --max-depth 5 \
  --output-metrics $OUTPUT_DIR/metrics.json

if [ $? -ne 0 ]; then
    echo "❌ Training failed!"
    exit 1
fi

# 步骤 3: 总结
echo ""
echo "=========================================="
echo "✅ Training Completed Successfully!"
echo "=========================================="
echo ""
echo "📊 Results:"
echo "   Final data: $FINAL_DATA"
echo "   Model directory: $OUTPUT_DIR"
echo "   Ensemble model: $OUTPUT_DIR/ensemble.pkl"
echo "   Conformal predictor: $OUTPUT_DIR/conformal.pkl"
echo "   Metrics: $OUTPUT_DIR/metrics.json"
echo ""

# 显示指标
if [ -f "$OUTPUT_DIR/metrics.json" ]; then
    echo "📈 Training Metrics:"
    cat $OUTPUT_DIR/metrics.json | python -m json.tool | grep -E "(r2|mae|n_samples|n_features|total_time)" | head -10
    echo ""
fi

echo "🎉 Done! Your production model is ready."
echo ""
echo "To use the model:"
echo "  from codon_verifier.model_ensemble import DeepEnsemble"
echo "  ensemble = DeepEnsemble.load('$OUTPUT_DIR/ensemble.pkl')"
echo ""
