#!/bin/bash
# Test script to verify MMseqs2 command logging

echo "=== Testing MMseqs2 Command Logging ==="

# Create a simple test input file
mkdir -p /tmp/test_msa
cat > /tmp/test_msa/test_input.jsonl << EOF
{"protein_id": "test_protein_1", "protein_aa": "MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNEL"}
{"protein_id": "test_protein_2", "protein_aa": "MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNEL"}
EOF

echo "Created test input file with 2 sequences"

# Test with a small database (if available)
if [ -f "data/mmseqs_db/test_production/SwissProt" ]; then
    echo "Testing with Swiss-Prot database..."
    docker run --rm \
        --gpus all \
        --entrypoint="" \
        -v "$(pwd)/data":/data \
        -v "/tmp/test_msa":/tmp/test_msa \
        codon-verifier/msa-features-lite:latest \
        python3 app.py \
        --input /tmp/test_msa/test_input.jsonl \
        --output /tmp/test_msa/test_output.jsonl \
        --use-mmseqs2 \
        --database "/data/mmseqs_db/test_production/SwissProt" \
        --threads 4 \
        --batch-size 2 \
        --limit 2 \
        --log-level INFO
else
    echo "Swiss-Prot database not found, testing Lite mode..."
    docker run --rm \
        --entrypoint="" \
        -v "/tmp/test_msa":/tmp/test_msa \
        codon-verifier/msa-features-lite:latest \
        python3 app.py \
        --input /tmp/test_msa/test_input.jsonl \
        --output /tmp/test_msa/test_output.jsonl \
        --limit 2 \
        --log-level INFO
fi

echo "Test completed. Check the output above for command logging."
