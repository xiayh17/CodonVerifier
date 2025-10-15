#!/bin/bash
# Build GPU-enabled Docker image for MSA Features

echo "=== Building GPU-Enabled MSA Features Docker Image ==="
echo

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "✗ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if NVIDIA Docker runtime is available
if ! docker info | grep -q nvidia; then
    echo "⚠️  NVIDIA Docker runtime not detected. GPU support may not work."
    echo "   Make sure you have nvidia-docker2 installed."
fi

echo "Choose build method:"
echo "1. Pre-compiled MMseqs2 (faster build)"
echo "2. Compile from source (guaranteed CUDA support)"
echo "3. Use official MMseqs2 Docker image (Python 3.11)"
echo "4. Use official MMseqs2 Docker image (simple, existing Python)"
echo

read -p "Select option (1-4): " choice

case $choice in
    1)
        echo "Building with pre-compiled MMseqs2..."
        dockerfile="Dockerfile.gpu"
        tag="codon-verifier/msa-features-lite:gpu"
        ;;
    2)
        echo "Building with source-compiled MMseqs2..."
        dockerfile="Dockerfile.gpu-source"
        tag="codon-verifier/msa-features-lite:gpu-source"
        ;;
    3)
        echo "Using official MMseqs2 Docker image with Python 3.11..."
        dockerfile="Dockerfile.official"
        tag="codon-verifier/msa-features-lite:gpu-official"
        ;;
    4)
        echo "Using official MMseqs2 Docker image (simple)..."
        dockerfile="Dockerfile.official-simple"
        tag="codon-verifier/msa-features-lite:gpu-official-simple"
        ;;
    *)
        echo "Invalid option. Using pre-compiled version."
        dockerfile="Dockerfile.gpu"
        tag="codon-verifier/msa-features-lite:gpu"
        ;;
esac

echo "Building Docker image..."
echo "Dockerfile: $dockerfile"
echo "Tag: $tag"
echo

# Build the image
docker build -f "$dockerfile" -t "$tag" .

if [ $? -eq 0 ]; then
    echo "✓ Docker image built successfully!"
    echo "Image tag: $tag"
    echo
    
    # Test the image
    echo "Testing GPU support..."
    docker run --rm --gpus all "$tag" mmseqs search --help | grep -i gpu
    
    if [ $? -eq 0 ]; then
        echo "✓ GPU support confirmed!"
    else
        echo "⚠️  GPU support not detected. The image may not have CUDA support."
    fi
    
    echo
    echo "=== Usage Examples ==="
    echo
    echo "1. Test GPU support:"
    echo "   docker run --rm --gpus all $tag mmseqs search --help | grep gpu"
    echo
    echo "2. Run MSA processing with GPU:"
    echo "   docker run --rm --gpus all \\"
    echo "     -v \$(pwd)/data:/data \\"
    echo "     $tag \\"
    echo "     python app.py \\"
    echo "     --input /data/enhanced/Pic_complete_v2.jsonl \\"
    echo "     --output /data/real_msa/Pic_production_db_UniRef50.jsonl \\"
    echo "     --use-mmseqs2 \\"
    echo "     --database /data/mmseqs_db/production/UniRef50 \\"
    echo "     --use-gpu \\"
    echo "     --gpu-id 0 \\"
    echo "     --threads 20 \\"
    echo "     --batch-size 50 \\"
    echo "     --search-timeout 1800"
    echo
    echo "3. Enter interactive mode:"
    echo "   docker run --rm -it --gpus all \\"
    echo "     -v \$(pwd)/data:/data \\"
    echo "     -v \$(pwd)/services/msa_features_lite:/workspace \\"
    echo "     -w /workspace \\"
    echo "     $tag /bin/bash"
    echo
    
else
    echo "✗ Docker build failed!"
    exit 1
fi
