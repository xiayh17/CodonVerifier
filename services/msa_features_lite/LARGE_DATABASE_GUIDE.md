# Large Database Support Guide

This guide explains how to use the MSA Features Service with large databases (50GB+) and the new progress display features.

## New Features for Large Databases

### 1. Database Initialization Progress
- Real-time progress indicators during database initialization
- Database size and sequence count statistics
- Configurable timeout settings
- Better error handling for large databases

### 2. Command Line Options

```bash
# Basic usage with progress display
python app.py \
  --input data/proteins.jsonl \
  --output data/output.jsonl \
  --use-mmseqs2 \
  --database /data/mmseqs_db/uniref50

# For very large databases (50GB+)
python app.py \
  --input data/proteins.jsonl \
  --output data/output.jsonl \
  --use-mmseqs2 \
  --database /data/mmseqs_db/uniref50 \
  --db-init-timeout 600 \
  --search-timeout 1200 \
  --threads 16

# Disable progress indicators (for scripts)
python app.py \
  --input data/proteins.jsonl \
  --output data/output.jsonl \
  --use-mmseqs2 \
  --database /data/mmseqs_db/uniref50 \
  --no-progress
```

### 3. New Parameters

- `--db-init-timeout`: Database initialization timeout in seconds (default: 300)
- `--search-timeout`: MMseqs2 search timeout in seconds (default: 600)
- `--no-progress`: Disable progress indicators for database initialization

## Database Size Recommendations

| Database Size | Init Timeout | Search Timeout | Notes |
|---------------|-------------|----------------|-------|
| < 10GB | 300s | 600s | Standard timeouts |
| 10-50GB | 600s | 900s | Increase timeouts |
| 50GB+ | 900s+ | 1200s+ | Consider using smaller database for testing |

## Example Output

```
2025-01-12 10:30:15 - INFO - Starting: Initializing database (this may take several minutes for large databases)
Initializing database (this may take several minutes for large databases)...
2025-01-12 10:32:45 - INFO - ✓ Initializing database (this may take several minutes for large databases) (took 150.2s)
2025-01-12 10:32:45 - INFO - ✓ Database valid: /data/mmseqs_db/uniref50
2025-01-12 10:32:45 - INFO -   Size: 52.3 GB
2025-01-12 10:32:45 - INFO -   Sequences: 45,234,567
2025-01-12 10:32:45 - INFO -   Type: UniRef
2025-01-12 10:32:45 - INFO -   Init time: 150.2s
```

## Performance Tips

### For Large Databases (50GB+)

1. **Increase timeouts**: 
   - `--db-init-timeout 600` or higher
   - `--search-timeout 1200` or higher
2. **Use more threads**: `--threads 16` or higher
3. **Enable GPU acceleration**: `--use-gpu` (automatically optimized for large databases)
4. **Ensure sufficient disk I/O**: Use SSD storage if possible
5. **Monitor memory usage**: Large databases may require significant RAM

### For Testing

1. **Use smaller databases**: Swiss-Prot instead of UniRef50
2. **Limit records**: Use `--limit 100` for testing
3. **Disable progress**: Use `--no-progress` in automated scripts

## Troubleshooting

### Database Initialization Timeout

```
Database initialization timed out after 300s
For large databases (50GB+), consider:
1. Increasing --db-init-timeout (default: 300s)
2. Using a smaller database for testing
3. Ensuring sufficient disk I/O performance
```

**Solutions:**
- Increase `--db-init-timeout` to 600s or higher
- Check disk I/O performance
- Consider using a smaller database for testing

### GPU Issues with Large Databases

```
GPU search failed, this may be due to:
1. MMseqs2 not compiled with GPU support
2. CUDA drivers not properly installed
3. GPU memory insufficient
4. Unsupported GPU parameters
```

**Solutions:**
- The system now automatically falls back to CPU if GPU fails
- Use CPU mode for large databases: remove `--use-gpu`
- Ensure sufficient GPU memory (8GB+ recommended)
- Check CUDA installation
- Verify MMseqs2 GPU support: `mmseqs search --help | grep gpu`

## Docker Usage

```bash
# For large databases with increased timeout
docker run --rm \
  -v $(pwd)/data:/data \
  codon-verifier/msa-features:latest \
  --input /data/input.jsonl \
  --output /data/output.jsonl \
  --use-mmseqs2 \
  --database /data/mmseqs_db/uniref50 \
  --db-init-timeout 600 \
  --threads 16
```

## Testing the Progress Display

Run the test script to see the progress display in action:

```bash
cd services/msa_features_lite
python test_db_progress.py
```

This will demonstrate:
- Progress indicator animation
- Database statistics calculation
- Large database simulation
