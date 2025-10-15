#!/bin/bash
# Interactive Docker exploration script for MSA optimization

echo "=== MSA Features Docker Exploration ==="
echo "This script will help you explore and optimize MSA processing inside Docker"
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
echo "=== System Information ==="
echo "CPU cores: $(nproc)"
echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
echo "Disk space: $(df -h /data | tail -1 | awk '{print $4}')"

echo
echo "=== GPU Information ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits
else
    echo "nvidia-smi not available"
fi

echo
echo "=== MMseqs2 Information ==="
if command -v mmseqs &> /dev/null; then
    echo "MMseqs2 version: $(mmseqs version)"
    echo "MMseqs2 help:"
    mmseqs search --help | grep -E "(gpu|thread|memory|batch)" | head -10
else
    echo "MMseqs2 not available"
fi

echo
echo "=== Database Information ==="
if [ -f "/data/mmseqs_db/production/UniRef50" ]; then
    echo "Database path: /data/mmseqs_db/production/UniRef50"
    echo "Database size: $(du -sh /data/mmseqs_db/production/UniRef50* | awk '{sum+=$1} END {print sum "GB"}')"
    echo "Database files:"
    ls -la /data/mmseqs_db/production/UniRef50* | head -5
else
    echo "Database not found at /data/mmseqs_db/production/UniRef50"
fi

echo
echo "=== Performance Testing Commands ==="
echo "1. Test MMseqs2 GPU support:"
echo "   mmseqs search --help | grep gpu"
echo
echo "2. Test database access speed:"
echo "   time mmseqs view /data/mmseqs_db/production/UniRef50 | head -100"
echo
echo "3. Test GPU memory usage:"
echo "   nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"
echo
echo "4. Monitor GPU during search:"
echo "   watch -n 1 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits'"
echo
echo "5. Test small batch processing:"
echo "   python app.py --input /data/enhanced/Pic_complete_v2.jsonl --output /tmp/test_output.jsonl --use-mmseqs2 --database /data/mmseqs_db/production/UniRef50 --use-gpu --gpu-id 0 --threads 20 --batch-size 1 --limit 1 --search-timeout 300"
echo
echo "=== Optimization Suggestions ==="
echo "• Increase batch size for better GPU utilization"
echo "• Use more threads for CPU operations"
echo "• Monitor GPU memory usage during processing"
echo "• Consider using smaller database for testing"
echo "• Check disk I/O performance"
echo
echo "=== Interactive Commands ==="
echo "Run any of these commands to explore further:"
echo "• 'mmseqs search --help' - See all MMseqs2 options"
echo "• 'nvidia-smi' - Check GPU status"
echo "• 'htop' - Monitor system resources"
echo "• 'iostat -x 1' - Monitor disk I/O"
echo "• 'python app.py --help' - See all app options"
