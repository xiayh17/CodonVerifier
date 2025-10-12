#!/bin/bash
# Quick Demo Script for Structure Features Service with AFDB Integration

set -e

echo "============================================================"
echo " Structure Features Service - Quick Demo"
echo "============================================================"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if requests is installed
echo -e "${BLUE}Checking dependencies...${NC}"
if python3 -c "import requests" 2>/dev/null; then
    echo -e "${GREEN}✓ requests library installed${NC}"
else
    echo -e "${YELLOW}! requests library not installed${NC}"
    echo "Installing requests..."
    pip install requests
fi

echo ""
echo "============================================================"
echo " Demo 1: Process Example Input (AFDB + Lite)"
echo "============================================================"
echo ""

python3 app.py \
    --input example_input.jsonl \
    --output demo_output_afdb.jsonl \
    --log-level INFO

echo ""
echo "============================================================"
echo " Demo 2: Process with Lite-Only Mode"
echo "============================================================"
echo ""

python3 app.py \
    --input example_input.jsonl \
    --output demo_output_lite.jsonl \
    --no-afdb \
    --log-level INFO

echo ""
echo "============================================================"
echo " Demo 3: Test First 3 Records Only"
echo "============================================================"
echo ""

python3 app.py \
    --input example_input.jsonl \
    --output demo_output_limited.jsonl \
    --limit 3 \
    --log-level INFO

echo ""
echo "============================================================"
echo " Comparing Results"
echo "============================================================"
echo ""

echo -e "${BLUE}AFDB Mode Output (first 3 lines):${NC}"
head -3 demo_output_afdb.jsonl | python3 -m json.tool --indent 2 | head -30

echo ""
echo -e "${BLUE}Lite Mode Output (first 3 lines):${NC}"
head -3 demo_output_lite.jsonl | python3 -m json.tool --indent 2 | head -30

echo ""
echo "============================================================"
echo " Demo Complete!"
echo "============================================================"
echo ""
echo "Output files created:"
echo "  - demo_output_afdb.jsonl (AFDB + Lite fallback)"
echo "  - demo_output_lite.jsonl (Lite only)"
echo "  - demo_output_limited.jsonl (first 3 records)"
echo ""
echo "To view results:"
echo "  python3 -m json.tool demo_output_afdb.jsonl"
echo ""
echo "To run full test suite:"
echo "  python3 test_afdb_integration.py"
echo ""

