# GPU Parameter Fix Summary

## Problem Identified

The MMseqs2 GPU search was failing with the error:
```
Unrecognized parameter "--gpu-memory". Did you mean "--split-memory-limit" (Split memory limit)?
```

## Root Cause

MMseqs2 does not support the `--gpu-memory` parameter that was being used. The correct parameter for memory management is `--split-memory-limit`.

## Fixes Applied

### 1. Removed Unsupported GPU Parameters
- ❌ Removed `--gpu-memory` (not supported)
- ❌ Removed `--batch-size` for GPU (not GPU-specific)
- ✅ Kept only `--gpu <id>` (core GPU parameter)

### 2. Added Proper Memory Management
- ✅ Use `--split-memory-limit` for large databases (30GB+)
- ✅ Automatic memory limit based on database size

### 3. Improved GPU Detection
- ✅ Check if MMseqs2 actually supports GPU before using it
- ✅ Better error messages for GPU support issues

### 4. Added CPU Fallback
- ✅ Automatic fallback to CPU if GPU search fails
- ✅ Remove GPU parameters and retry with CPU
- ✅ Better error handling and logging

## Updated GPU Command Structure

### Before (Failed)
```bash
mmseqs search ... --gpu 0 --gpu-memory 12288 --batch-size 64
```

### After (Fixed)
```bash
mmseqs search ... --gpu 0 --split-memory-limit 12288
```

## Expected Behavior Now

1. **GPU Detection**: Properly detects if MMseqs2 supports GPU
2. **GPU Usage**: Uses only supported GPU parameters
3. **Memory Management**: Uses `--split-memory-limit` for large databases
4. **Fallback**: Automatically falls back to CPU if GPU fails
5. **Error Handling**: Better error messages and recovery

## Usage Examples

### Large Database with GPU
```bash
python app.py \
  --input data/proteins.jsonl \
  --output data/output.jsonl \
  --use-mmseqs2 \
  --database /data/mmseqs_db/uniref50 \
  --use-gpu \
  --gpu-id 0 \
  --search-timeout 1800
```

### Expected Output
```
Found GPU: NVIDIA GeForce RTX 4090
MMseqs2 supports GPU acceleration
Large database (37.0GB), using GPU for batch size 10
Using GPU 0 for MMseqs2 search
  Using split-memory-limit: 12288MB for large database
Running MMseqs2 search with 1800s timeout...
```

### If GPU Fails
```
GPU search failed, attempting CPU fallback...
Retrying with CPU-only search...
CPU fallback search completed successfully
```

## Key Improvements

1. **Parameter Compatibility**: Only uses supported MMseqs2 parameters
2. **Automatic Recovery**: Falls back to CPU if GPU fails
3. **Better Diagnostics**: Clear error messages and solutions
4. **Memory Optimization**: Proper memory limits for large databases
5. **Robust Operation**: Continues processing even if GPU fails

This fix ensures that the MSA features service works reliably with both GPU and CPU, providing the best performance possible while maintaining compatibility.
