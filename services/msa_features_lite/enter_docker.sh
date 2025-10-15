#!/bin/bash
# Script to enter Docker container for interactive MSA optimization

echo "=== Entering MSA Features Docker Container ==="
echo "This will start an interactive session inside the Docker container"
echo

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "✗ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if the image exists
if ! docker image inspect codon-verifier/msa-features-lite:latest > /dev/null 2>&1; then
    echo "✗ Docker image 'codon-verifier/msa-features-lite:latest' not found."
    echo "Please build the image first:"
    echo "  docker build -f services/msa_features_lite/Dockerfile -t codon-verifier/msa-features-lite:latest ."
    exit 1
fi

echo "✓ Docker is running"
echo "✓ Image found: codon-verifier/msa-features-lite:latest"
echo

echo "🚀 Starting interactive Docker session..."
echo "You can now run optimization tests inside the container."
echo
echo "Useful commands once inside:"
echo "• ./docker_explore.sh - System information and exploration"
echo "• ./quick_optimization_test.sh - Quick performance tests"
echo "• python optimization_analysis.py - Performance analysis"
echo "• nvidia-smi -l 1 - Monitor GPU usage"
echo "• htop - Monitor system resources"
echo

# Start interactive Docker session
docker run --rm -it \
    --gpus all \
    --entrypoint="" \
    -v "$(pwd)/data":/data \
    -v "$(pwd)/services/msa_features_lite":/workspace \
    -w /workspace \
    codon-verifier/msa-features-lite:latest \
    /bin/bash
