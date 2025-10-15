# GPU Support Solution Guide

## 🎯 Problem Identified

**Root Cause**: MMseqs2 was compiled without CUDA support
```
MMseqs2 was compiled without CUDA support
Error: Ungapped prefilter died
```

## 🔧 Solutions

### Solution 1: Use Pre-compiled GPU Version (Recommended)

**Pros**: Fast build, reliable
**Cons**: May not have latest features

```bash
# Build GPU-enabled image
./build_gpu_docker.sh
# Select option 1: Pre-compiled MMseqs2
```

### Solution 2: Compile from Source (Most Reliable)

**Pros**: Guaranteed CUDA support, latest features
**Cons**: Longer build time

```bash
# Build GPU-enabled image
./build_gpu_docker.sh
# Select option 2: Compile from source
```

### Solution 3: Use Official MMseqs2 Docker Image

**Pros**: Official support, maintained by MMseqs2 team
**Cons**: May have different base environment

```bash
# Build GPU-enabled image
./build_gpu_docker.sh
# Select option 3: Official MMseqs2 image
```

## 🚀 Quick Start

### 1. Build GPU-Enabled Image

```bash
cd services/msa_features_lite
./build_gpu_docker.sh
```

### 2. Test GPU Support

```bash
# Test if GPU support is working
docker run --rm --gpus all codon-verifier/msa-features-lite:gpu \
  mmseqs search --help | grep gpu
```

### 3. Run MSA Processing with GPU

```bash
docker run --rm --gpus all \
  -v "$(pwd)/data":/data \
  codon-verifier/msa-features-lite:gpu \
  python app.py \
  --input /data/enhanced/Pic_complete_v2.jsonl \
  --output /data/real_msa/Pic_production_db_UniRef50.jsonl \
  --use-mmseqs2 \
  --database /data/mmseqs_db/production/UniRef50 \
  --use-gpu \
  --gpu-id 0 \
  --threads 20 \
  --batch-size 50 \
  --search-timeout 1800
```

## 📊 Expected Performance Improvements

| Configuration | Batch Size | Expected Time | Improvement |
|---------------|------------|---------------|-------------|
| CPU Only | 10 | 14.5 min | Baseline |
| GPU Enabled | 10 | 8-10 min | 30-40% faster |
| GPU + Large Batch | 50 | 5-7 min | 50-60% faster |
| GPU + Optimized | 100 | 3-5 min | 70-80% faster |

## 🔍 Verification Steps

### 1. Check GPU Support

```bash
# Inside Docker container
mmseqs search --help | grep gpu
# Should show: --gpu <int>    GPU device to use
```

### 2. Test GPU Functionality

```bash
# Run a quick test
./cpu_optimized_test.sh
# Select option 7: Test GPU vs CPU
```

### 3. Monitor GPU Usage

```bash
# Monitor GPU utilization
nvidia-smi -l 1
```

## 🛠️ Troubleshooting

### Issue: "MMseqs2 was compiled without CUDA support"

**Solution**: Use one of the GPU-enabled Dockerfiles

### Issue: "No GPU detected"

**Solution**: 
1. Ensure `--gpus all` is used in docker run
2. Check nvidia-docker2 is installed
3. Verify CUDA drivers are working

### Issue: "GPU search failed"

**Solution**: The system will automatically fall back to CPU

## 📝 Dockerfile Comparison

| Dockerfile | Base Image | MMseqs2 Source | CUDA Support | Build Time |
|------------|------------|----------------|--------------|------------|
| Dockerfile | python:3.10-slim | Pre-compiled | ❌ No | Fast |
| Dockerfile.gpu | nvidia/cuda:11.8 | Pre-compiled | ✅ Yes | Medium |
| Dockerfile.gpu-source | nvidia/cuda:11.8 | Source | ✅ Yes | Slow |
| Dockerfile.official | ghcr.io/soedinglab/mmseqs2 | Official | ✅ Yes | Fast |

## 🎯 Recommended Workflow

1. **Build GPU image**: `./build_gpu_docker.sh`
2. **Test GPU support**: Verify `--gpu` parameter exists
3. **Run optimization tests**: Use `./cpu_optimized_test.sh`
4. **Apply to full dataset**: Use optimal parameters

## 🚀 Performance Tips

### For GPU Processing

1. **Increase batch size**: 50-100 records per batch
2. **Use more threads**: 24-32 threads
3. **Optimize memory**: Use `--split-memory-limit`
4. **Monitor GPU usage**: Ensure GPU is being utilized

### For CPU Fallback

1. **Use more threads**: 32-48 threads
2. **Optimize batch size**: 25-50 records
3. **Reduce sensitivity**: Use `-s 6.0` instead of `7.5`

## 📈 Expected Results

With GPU support enabled:
- **30-80% performance improvement**
- **Better GPU utilization**
- **Faster processing of large datasets**
- **Automatic CPU fallback if GPU fails**

## 🎉 Success Criteria

✅ MMseqs2 shows `--gpu` parameter in help
✅ GPU searches complete without "CUDA support" errors
✅ GPU utilization visible in `nvidia-smi`
✅ Automatic CPU fallback works if GPU fails
✅ Performance improvement over CPU-only version

You're now ready to build and use GPU-enabled MSA processing! 🚀
