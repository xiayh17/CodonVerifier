#!/bin/bash
# Test script to verify MMseqs2 installation

echo "=== Testing MMseqs2 Installation ==="

# Test if mmseqs command exists
if command -v mmseqs &> /dev/null; then
    echo "✓ mmseqs command found"
    mmseqs version
else
    echo "✗ mmseqs command not found"
    echo "Available mmseqs commands:"
    ls -la /usr/local/bin/ | grep mmseqs
fi

echo
echo "=== Testing mmseqs_avx2 ==="
if command -v mmseqs_avx2 &> /dev/null; then
    echo "✓ mmseqs_avx2 command found"
    mmseqs_avx2 version
else
    echo "✗ mmseqs_avx2 command not found"
fi

echo
echo "=== Testing GPU Support ==="
mmseqs search --help | grep -i gpu || echo "No GPU support found in help"

echo
echo "=== Testing Database Path ==="
if [ -d "/data/mmseqs_db/production/UniRef50" ]; then
    echo "✓ Database path exists: /data/mmseqs_db/production/UniRef50"
    ls -la /data/mmseqs_db/production/UniRef50/ | head -5
else
    echo "✗ Database path not found: /data/mmseqs_db/production/UniRef50"
    echo "Available data directories:"
    ls -la /data/ 2>/dev/null || echo "No /data directory found"
fi
