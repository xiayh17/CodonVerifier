#!/bin/bash
# Quick script to build and run MFE test in Docker environment

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "5' UTR MFE Test - Docker Environment"
echo "=========================================="
echo ""

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Error: Docker is not installed or not in PATH"
    exit 1
fi

echo "✓ Docker is available"
echo ""

# Build the image
echo "Building Docker image with ViennaRNA..."
echo "(This may take 5-10 minutes on first build)"
echo ""

docker build -t sequence-analyzer -f services/sequence_analyzer/Dockerfile . || {
    echo "❌ Failed to build Docker image"
    exit 1
}

echo ""
echo "✓ Docker image built successfully"
echo ""

# Verify ViennaRNA
echo "Verifying ViennaRNA installation..."
VIENNA_VERSION=$(docker run --rm sequence-analyzer RNAfold --version 2>&1 | head -1)
echo "✓ $VIENNA_VERSION"
echo ""

# Run the test
echo "=========================================="
echo "Running MFE Test Demo..."
echo "=========================================="
echo ""

docker run --rm \
    -v "$(pwd):/workspace" \
    -w /workspace \
    sequence-analyzer \
    python3 test_utr_mfe_demo.py

echo ""
echo "=========================================="
echo "✓ Test completed successfully!"
echo "=========================================="
echo ""
echo "To run interactive tests, use:"
echo "  docker run --rm -it -v \"\$(pwd):/workspace\" -w /workspace sequence-analyzer bash"
echo ""

