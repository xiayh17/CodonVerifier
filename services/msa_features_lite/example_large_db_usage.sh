#!/bin/bash
# Example usage for large database (50GB+) with progress display

echo "=== Large Database MSA Features Example ==="
echo

# Example 1: Basic usage with progress display
echo "1. Basic usage with progress display:"
echo "python app.py \\"
echo "  --input data/proteins.jsonl \\"
echo "  --output data/output.jsonl \\"
echo "  --use-mmseqs2 \\"
echo "  --database /data/mmseqs_db/uniref50"
echo

# Example 2: Large database with increased timeouts
echo "2. Large database (50GB+) with increased timeouts:"
echo "python app.py \\"
echo "  --input data/proteins.jsonl \\"
echo "  --output data/output.jsonl \\"
echo "  --use-mmseqs2 \\"
echo "  --database /data/mmseqs_db/uniref50 \\"
echo "  --db-init-timeout 600 \\"
echo "  --search-timeout 1200 \\"
echo "  --threads 16"
echo

# Example 3: Testing with limited records
echo "3. Testing with limited records:"
echo "python app.py \\"
echo "  --input data/proteins.jsonl \\"
echo "  --output data/test_output.jsonl \\"
echo "  --use-mmseqs2 \\"
echo "  --database /data/mmseqs_db/uniref50 \\"
echo "  --limit 100 \\"
echo "  --db-init-timeout 300"
echo

# Example 4: Script mode (no progress indicators)
echo "4. Script mode (no progress indicators):"
echo "python app.py \\"
echo "  --input data/proteins.jsonl \\"
echo "  --output data/output.jsonl \\"
echo "  --use-mmseqs2 \\"
echo "  --database /data/mmseqs_db/uniref50 \\"
echo "  --no-progress"
echo

# Example 5: GPU acceleration for large databases
echo "5. GPU acceleration for large databases:"
echo "python app.py \\"
echo "  --input data/proteins.jsonl \\"
echo "  --output data/output.jsonl \\"
echo "  --use-mmseqs2 \\"
echo "  --database /data/mmseqs_db/uniref50 \\"
echo "  --use-gpu \\"
echo "  --gpu-id 0 \\"
echo "  --db-init-timeout 600 \\"
echo "  --search-timeout 1200"
echo

echo "=== Key Features for Large Databases ==="
echo "✓ Real-time progress indicators during database initialization"
echo "✓ Database size and sequence count statistics"
echo "✓ Configurable timeout settings (--db-init-timeout, --search-timeout)"
echo "✓ Progress display control (--no-progress)"
echo "✓ Better error handling and timeout management"
echo "✓ Smart GPU acceleration with automatic optimization for large databases"
echo "✓ Improved database sequence count parsing"
echo

echo "=== Performance Recommendations ==="
echo "• For 50GB+ databases: Use --db-init-timeout 600 and --search-timeout 1200 or higher"
echo "• Use SSD storage for better I/O performance"
echo "• Increase threads: --threads 16 or higher"
echo "• Enable GPU acceleration: --use-gpu (automatically optimized for large databases)"
echo "• For testing: Use --limit 100 with smaller databases"
echo "• Monitor memory usage with large databases"
