#!/bin/bash
# Test script to verify Dockerfile fixes

echo "=== Testing Dockerfile Fixes ==="
echo

# Test 1: Official MMseqs2 image with Python 3.11
echo "🧪 Test 1: Official MMseqs2 with Python 3.11"
echo "Building Dockerfile.official..."
docker build -f Dockerfile.official -t test-official:latest . --no-cache

if [ $? -eq 0 ]; then
    echo "✓ Dockerfile.official built successfully!"
    
    # Test GPU support
    echo "Testing GPU support..."
    docker run --rm --gpus all test-official:latest mmseqs search --help | grep -i gpu
    
    if [ $? -eq 0 ]; then
        echo "✓ GPU support confirmed!"
    else
        echo "⚠️  GPU support not detected"
    fi
else
    echo "✗ Dockerfile.official build failed"
fi

echo

# Test 2: Official MMseqs2 image (simple)
echo "🧪 Test 2: Official MMseqs2 (simple)"
echo "Building Dockerfile.official-simple..."
docker build -f Dockerfile.official-simple -t test-official-simple:latest . --no-cache

if [ $? -eq 0 ]; then
    echo "✓ Dockerfile.official-simple built successfully!"
    
    # Test GPU support
    echo "Testing GPU support..."
    docker run --rm --gpus all test-official-simple:latest mmseqs search --help | grep -i gpu
    
    if [ $? -eq 0 ]; then
        echo "✓ GPU support confirmed!"
    else
        echo "⚠️  GPU support not detected"
    fi
else
    echo "✗ Dockerfile.official-simple build failed"
fi

echo

# Cleanup
echo "🧹 Cleaning up test images..."
docker rmi test-official:latest test-official-simple:latest 2>/dev/null

echo
echo "=== Test Summary ==="
echo "If both tests passed, the Dockerfile fixes are working!"
echo "You can now use:"
echo "  ./build_gpu_docker.sh"
echo "  Select option 3 or 4 for official MMseqs2 images"
