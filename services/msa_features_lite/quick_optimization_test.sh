#!/bin/bash
# Quick optimization test script for Docker environment

echo "=== MSA Processing Optimization Test ==="
echo "This script will test different configurations to find the optimal settings"
echo

# Check if we're inside Docker
if [ -f /.dockerenv ]; then
    echo "✓ Running inside Docker container"
else
    echo "✗ Not inside Docker container"
    echo "Please run this script inside the Docker container"
    exit 1
fi

echo
echo "=== System Check ==="
echo "CPU cores: $(nproc)"
echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
if command -v nvidia-smi &> /dev/null; then
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"
    echo "GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)MB"
else
    echo "GPU: Not available"
fi

echo
echo "=== Test Configurations ==="

# Test 1: Small batch size (current)
echo "🧪 Test 1: Small batch size (current configuration)"
echo "Command:"
echo "python app.py \\"
echo "  --input /data/enhanced/Pic_complete_v2.jsonl \\"
echo "  --output /tmp/test_small_batch.jsonl \\"
echo "  --use-mmseqs2 \\"
echo "  --database /data/mmseqs_db/production/UniRef50 \\"
echo "  --use-gpu --gpu-id 0 \\"
echo "  --threads 20 \\"
echo "  --batch-size 10 \\"
echo "  --limit 20 \\"
echo "  --search-timeout 600"
echo

# Test 2: Medium batch size
echo "🧪 Test 2: Medium batch size (recommended)"
echo "Command:"
echo "python app.py \\"
echo "  --input /data/enhanced/Pic_complete_v2.jsonl \\"
echo "  --output /tmp/test_medium_batch.jsonl \\"
echo "  --use-mmseqs2 \\"
echo "  --database /data/mmseqs_db/production/UniRef50 \\"
echo "  --use-gpu --gpu-id 0 \\"
echo "  --threads 24 \\"
echo "  --batch-size 50 \\"
echo "  --limit 20 \\"
echo "  --search-timeout 600"
echo

# Test 3: Large batch size
echo "🧪 Test 3: Large batch size (speed optimized)"
echo "Command:"
echo "python app.py \\"
echo "  --input /data/enhanced/Pic_complete_v2.jsonl \\"
echo "  --output /tmp/test_large_batch.jsonl \\"
echo "  --use-mmseqs2 \\"
echo "  --database /data/mmseqs_db/production/UniRef50 \\"
echo "  --use-gpu --gpu-id 0 \\"
echo "  --threads 32 \\"
echo "  --batch-size 100 \\"
echo "  --limit 20 \\"
echo "  --search-timeout 600"
echo

echo "=== Monitoring Commands ==="
echo "Run these in separate terminals to monitor performance:"
echo
echo "1. Monitor GPU usage:"
echo "   watch -n 1 'nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader,nounits'"
echo
echo "2. Monitor system resources:"
echo "   htop"
echo
echo "3. Monitor disk I/O:"
echo "   iostat -x 1"
echo

echo "=== Quick Test Commands ==="
echo "Copy and paste these commands to test different configurations:"
echo

echo "# Test 1: Small batch (current)"
echo "time python app.py --input /data/enhanced/Pic_complete_v2.jsonl --output /tmp/test1.jsonl --use-mmseqs2 --database /data/mmseqs_db/production/UniRef50 --use-gpu --gpu-id 0 --threads 20 --batch-size 10 --limit 5 --search-timeout 300"
echo

echo "# Test 2: Medium batch (recommended)"
echo "time python app.py --input /data/enhanced/Pic_complete_v2.jsonl --output /tmp/test2.jsonl --use-mmseqs2 --database /data/mmseqs_db/production/UniRef50 --use-gpu --gpu-id 0 --threads 24 --batch-size 50 --limit 5 --search-timeout 300"
echo

echo "# Test 3: Large batch (speed optimized)"
echo "time python app.py --input /data/enhanced/Pic_complete_v2.jsonl --output /tmp/test3.jsonl --use-mmseqs2 --database /data/mmseqs_db/production/UniRef50 --use-gpu --gpu-id 0 --threads 32 --batch-size 100 --limit 5 --search-timeout 300"
echo

echo "=== Expected Results ==="
echo "• Test 1: ~2-3 minutes per batch (current performance)"
echo "• Test 2: ~1-2 minutes per batch (30-50% faster)"
echo "• Test 3: ~1 minute per batch (50-70% faster)"
echo

echo "=== Next Steps ==="
echo "1. Run the quick tests above"
echo "2. Compare the timing results"
echo "3. Choose the best configuration for your needs"
echo "4. Apply the optimal settings to your full dataset"
echo

echo "=== Full Dataset Optimization ==="
echo "Once you find the optimal configuration, use it for the full dataset:"
echo
echo "# Example: If medium batch size (50) is optimal:"
echo "python app.py \\"
echo "  --input /data/enhanced/Pic_complete_v2.jsonl \\"
echo "  --output /data/real_msa/Pic_production_db_UniRef50_optimized.jsonl \\"
echo "  --use-mmseqs2 \\"
echo "  --database /data/mmseqs_db/production/UniRef50 \\"
echo "  --use-gpu --gpu-id 0 \\"
echo "  --threads 24 \\"
echo "  --batch-size 50 \\"
echo "  --search-timeout 1800"
echo

echo "=== Performance Monitoring ==="
echo "Monitor the full run with:"
echo "• GPU: nvidia-smi -l 1"
echo "• System: htop"
echo "• Progress: tail -f /path/to/logfile"
echo

echo "Ready to optimize! 🚀"
