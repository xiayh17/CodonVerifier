# Docker Interactive Optimization Guide

## 🎯 Current Status

✅ **GPU is now working correctly!**
- GPU acceleration is enabled and functioning
- Each batch takes ~14.5 minutes
- Total estimated time: ~7.8 hours for 320 records

## 🚀 Quick Start - Enter Docker

```bash
# Enter interactive Docker session
./enter_docker.sh
```

This will start an interactive session inside the Docker container with:
- GPU access
- Data volume mounted at `/data`
- Workspace mounted at `/workspace`
- All optimization scripts available

## 📊 Performance Analysis

Based on your current performance data:

| Metric | Current | Optimized Potential |
|--------|---------|-------------------|
| Batch time | 14.5 min | 5-8 min |
| Total time | 7.8 hours | 3-5 hours |
| GPU utilization | Good | Excellent |
| Batch size | 10 records | 50-100 records |

## 🧪 Optimization Testing

Once inside Docker, run these tests:

### 1. System Information
```bash
./docker_explore.sh
```

### 2. Quick Performance Tests
```bash
./quick_optimization_test.sh
```

### 3. Test Different Batch Sizes

**Test 1: Current (baseline)**
```bash
time python app.py \
  --input /data/enhanced/Pic_complete_v2.jsonl \
  --output /tmp/test1.jsonl \
  --use-mmseqs2 \
  --database /data/mmseqs_db/production/UniRef50 \
  --use-gpu --gpu-id 0 \
  --threads 20 \
  --batch-size 10 \
  --limit 5 \
  --search-timeout 300
```

**Test 2: Medium batch (recommended)**
```bash
time python app.py \
  --input /data/enhanced/Pic_complete_v2.jsonl \
  --output /tmp/test2.jsonl \
  --use-mmseqs2 \
  --database /data/mmseqs_db/production/UniRef50 \
  --use-gpu --gpu-id 0 \
  --threads 24 \
  --batch-size 50 \
  --limit 5 \
  --search-timeout 300
```

**Test 3: Large batch (speed optimized)**
```bash
time python app.py \
  --input /data/enhanced/Pic_complete_v2.jsonl \
  --output /tmp/test3.jsonl \
  --use-mmseqs2 \
  --database /data/mmseqs_db/production/UniRef50 \
  --use-gpu --gpu-id 0 \
  --threads 32 \
  --batch-size 100 \
  --limit 5 \
  --search-timeout 300
```

## 📈 Expected Improvements

| Configuration | Batch Size | Expected Time | Improvement |
|---------------|------------|---------------|-------------|
| Current | 10 | 14.5 min | Baseline |
| Medium | 50 | 8-10 min | 30-40% faster |
| Large | 100 | 5-7 min | 50-60% faster |

## 🔧 Additional Optimizations

### 1. MMseqs2 Parameters
```bash
# More aggressive parameters for speed
--max-seqs 500    # Instead of 1000
-s 6.0            # Instead of 7.5
```

### 2. Memory Optimization
```bash
# Increase memory limit if available
--split-memory-limit 16384  # Instead of 12288
```

### 3. Thread Optimization
```bash
# Use more threads if CPU has more cores
--threads 32  # Instead of 20
```

## 📊 Monitoring Commands

Run these in separate terminals to monitor performance:

### GPU Monitoring
```bash
# Real-time GPU usage
nvidia-smi -l 1

# Detailed GPU info
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader,nounits
```

### System Monitoring
```bash
# System resources
htop

# Disk I/O
iostat -x 1

# Memory usage
free -h
```

## 🎯 Recommended Workflow

1. **Enter Docker**: `./enter_docker.sh`
2. **Check system**: `./docker_explore.sh`
3. **Run quick tests**: `./quick_optimization_test.sh`
4. **Test batch sizes**: Run the 3 test commands above
5. **Compare results**: Note the timing differences
6. **Choose optimal config**: Based on speed vs quality trade-offs
7. **Run full dataset**: Apply optimal settings to complete dataset

## 🚀 Full Dataset Command

Once you find the optimal configuration, use it for the full dataset:

```bash
# Example: If batch size 50 is optimal
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

## 📝 Notes

- **GPU is working**: The fixes resolved the parameter issues
- **Current performance**: 14.5 min per batch is actually quite good for 37GB database
- **Optimization potential**: 30-60% speed improvement possible
- **Quality trade-offs**: Larger batch sizes may slightly reduce result quality
- **Memory considerations**: Monitor GPU memory usage with larger batches

## 🎉 Success Metrics

- ✅ GPU acceleration working
- ✅ No parameter errors
- ✅ Consistent batch processing
- ✅ Good error handling and fallback
- ✅ Ready for optimization testing

You're now ready to enter Docker and optimize your MSA processing! 🚀
