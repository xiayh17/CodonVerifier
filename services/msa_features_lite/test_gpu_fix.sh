#!/bin/bash
# Test script to verify GPU parameter fix

echo "=== Testing GPU Parameter Fix ==="

# Create a simple test input file
mkdir -p /tmp/test_gpu_fix
cat > /tmp/test_gpu_fix/test_input.jsonl << EOF
{"protein_id": "test_protein_1", "protein_aa": "MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNEL"}
EOF

echo "Created test input file with 1 sequence"

# Test with GPU enabled
echo "Testing with GPU enabled..."
docker run --rm \
    --gpus all \
    --entrypoint="" \
    -v "$(pwd)/data":/data \
    -v "/tmp/test_gpu_fix":/tmp/test_gpu_fix \
    codon-verifier/msa-features-lite:latest \
    python3 app.py \
    --input /tmp/test_gpu_fix/test_input.jsonl \
    --output /tmp/test_gpu_fix/test_output.jsonl \
    --use-mmseqs2 \
    --database "/data/mmseqs_db/test_production/SwissProt" \
    --use-gpu \
    --gpu-id 0 \
    --threads 4 \
    --batch-size 1 \
    --limit 1 \
    --log-level INFO

echo "Test completed. Check the output above for GPU parameter usage."
