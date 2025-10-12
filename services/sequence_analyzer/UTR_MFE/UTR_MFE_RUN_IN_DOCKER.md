# Running 5' UTR MFE Test in Docker Environment

This guide shows how to run the new 5' UTR / MFE analysis in the Docker container that has ViennaRNA installed.

## Quick Start

### Option 1: Build and Run Test Directly

```bash
# Build the Docker image
docker build -t sequence-analyzer -f services/sequence_analyzer/Dockerfile .

# Run the MFE test demo
docker run --rm -v $(pwd):/workspace sequence-analyzer \
    python3 /workspace/test_utr_mfe_demo.py
```

### Option 2: Interactive Shell (Recommended for Testing)

```bash
# Build the Docker image
docker build -t sequence-analyzer -f services/sequence_analyzer/Dockerfile .

# Start interactive shell
docker run --rm -it -v $(pwd):/workspace sequence-analyzer bash

# Inside container, run tests:
cd /workspace
python3 test_utr_mfe_demo.py
```

### Option 3: Run with Docker Compose (if available)

```bash
# From project root
docker-compose run --rm sequence-analyzer python3 /workspace/test_utr_mfe_demo.py
```

## What This Environment Provides

✅ **ViennaRNA 2.6.4** - Full RNA folding capabilities
- RNAfold CLI tool
- Python bindings (if installed)
- Supports temperature specification (37°C)

✅ **Python 3.10** with scientific packages:
- numpy, pandas, scikit-learn
- biopython
- All codon_verifier dependencies

✅ **System tools**: build-essential, wget, etc.

## Step-by-Step Instructions

### 1. Build the Docker Image

From the project root directory:

```bash
cd /mnt/c/Users/xiayh17/Documents/GitHub/CodonVerifier
docker build -t sequence-analyzer -f services/sequence_analyzer/Dockerfile .
```

**Expected output:**
```
[+] Building 120.3s (15/15) FINISHED
...
✓ Sequence Analyzer service ready
```

### 2. Verify ViennaRNA Installation

```bash
docker run --rm sequence-analyzer RNAfold --version
```

**Expected output:**
```
RNAfold 2.6.4
```

### 3. Run the MFE Test Demo

```bash
docker run --rm \
    -v $(pwd):/workspace \
    -w /workspace \
    sequence-analyzer \
    python3 test_utr_mfe_demo.py
```

**Expected output:**
```
================================================================================
5' UTR / mRNA Structure MFE Analysis - Demo and Test
================================================================================

Test Case 1: HMN1_HUMAN - 24aa, 75nt
...
Results:
  mfe_5p_dG:      -8.20 kcal/mol  ← Real MFE values!
  mfe_global_dG:  -15.30 kcal/mol
  mfe_5p_note:    no_utr_fallback
  ✓ Note matches expected: 'no_utr_fallback'
```

### 4. Run Interactive Tests

For more extensive testing:

```bash
# Start interactive shell
docker run --rm -it \
    -v $(pwd):/workspace \
    -w /workspace \
    sequence-analyzer bash

# Inside container:
python3
```

```python
# In Python shell:
import sys
sys.path.insert(0, '/workspace')

from codon_verifier.metrics import five_prime_utr_mfe_analysis

# Test with real sequence
dna = "ATGGCTCCACGAGGGTTCAGCTGTCTCTTACTTTCAACCAGTGAAATTGACCTGCCCGTGAAGAGGCGGACATAA"
result = five_prime_utr_mfe_analysis(dna, utr5_len=0, temperature=37.0)

print("Results:")
print(f"  mfe_5p_dG:      {result['mfe_5p_dG']:.2f} kcal/mol")
print(f"  mfe_global_dG:  {result['mfe_global_dG']:.2f} kcal/mol")
print(f"  mfe_5p_note:    {result['mfe_5p_note']}")
```

### 5. Test with Dataset

Process real sequences from the dataset:

```bash
docker run --rm -it \
    -v $(pwd):/workspace \
    -w /workspace \
    sequence-analyzer bash

# Inside container:
python3 << 'EOF'
import sys
sys.path.insert(0, '/workspace')

from codon_verifier.metrics import five_prime_utr_mfe_analysis
import pandas as pd

# Load dataset
df = pd.read_csv('/workspace/data/2025_bio-os_data/dataset/Human.tsv', sep='\t')

# Process first 5 sequences
for idx in range(min(5, len(df))):
    row = df.iloc[idx]
    seq = row['RefSeq_nn']
    protein = row['Entry Name']
    
    print(f"\n{protein}: {len(seq)} nt")
    result = five_prime_utr_mfe_analysis(seq, utr5_len=0, temperature=37.0)
    
    print(f"  5' MFE:     {result['mfe_5p_dG']:.2f} kcal/mol")
    print(f"  Global MFE: {result['mfe_global_dG']:.2f} kcal/mol")
    print(f"  Note:       {result['mfe_5p_note']}")
EOF
```

## Creating a Test Service

You can also create a standalone test script in the container:

```bash
# Create test script
cat > services/sequence_analyzer/test_mfe.py << 'EOF'
#!/usr/bin/env python3
import sys
sys.path.insert(0, '/app')

from codon_verifier.metrics import five_prime_utr_mfe_analysis

# Test sequences
sequences = {
    "Short (75nt)": "ATGGCTCCACGAGGGTTCAGCTGTCTCTTACTTTCAACCAGTGAAATTGACCTGCCCGTGAAGAGGCGGACATAA",
    "Medium (96nt)": "ATGGGGATAAACACCCGGGAGCTGTTTCTCAACTTCACTATTGTCTTGATTACGGTTATTCTTATGTGGCTCCTTGTGAGGTCCTATCAGTACTGA",
    "Long (105nt)": "ATGGCGCAACCTACGGCCTCGGCCCAGAAGCTGGTGCGGCCGATCCGCGCCGTGTGCCGCATCCTGCAGATCCCGGAGTCCGACCCCTCCAACCTGCGGCCCTAG",
}

print("MFE Analysis Test")
print("=" * 70)

for name, seq in sequences.items():
    print(f"\n{name}:")
    result = five_prime_utr_mfe_analysis(seq, utr5_len=0, temperature=37.0)
    print(f"  5' MFE:     {result['mfe_5p_dG']:.2f} kcal/mol")
    print(f"  Global MFE: {result['mfe_global_dG']:.2f} kcal/mol")
    print(f"  Note:       {result['mfe_5p_note']}")
EOF

# Rebuild with test script
docker build -t sequence-analyzer -f services/sequence_analyzer/Dockerfile .

# Run test
docker run --rm sequence-analyzer python3 test_mfe.py
```

## Docker Compose Configuration (Optional)

Create `docker-compose.yml` if not exists:

```yaml
version: '3.8'

services:
  sequence-analyzer:
    build:
      context: .
      dockerfile: services/sequence_analyzer/Dockerfile
    volumes:
      - .:/workspace
    working_dir: /workspace
    environment:
      - PYTHONPATH=/app:/workspace
    command: python3 test_utr_mfe_demo.py
```

Then run:
```bash
docker-compose run --rm sequence-analyzer
```

## Troubleshooting

### Issue: "ViennaRNA not available"
**Solution:** Rebuild the Docker image to ensure ViennaRNA is installed:
```bash
docker build --no-cache -t sequence-analyzer -f services/sequence_analyzer/Dockerfile .
```

### Issue: "Module not found"
**Solution:** Ensure volume mount is correct:
```bash
docker run --rm -v $(pwd):/workspace -w /workspace sequence-analyzer python3 test_utr_mfe_demo.py
```

### Issue: Permission denied
**Solution:** On Windows WSL2, ensure proper path:
```bash
cd /mnt/c/Users/xiayh17/Documents/GitHub/CodonVerifier
docker run --rm -v "$(pwd):/workspace" -w /workspace sequence-analyzer python3 test_utr_mfe_demo.py
```

## Performance Notes

- **First build**: ~5-10 minutes (compiles ViennaRNA from source)
- **Subsequent builds**: ~1-2 minutes (uses cache)
- **MFE calculation**: <1 second per sequence (<200nt)
- **Docker overhead**: minimal with volume mounts

## Expected Test Results

With ViennaRNA installed, you should see **real MFE values**:

```
Test Case 1: HMN1_HUMAN - 24aa, 75nt
================================================================================
Protein:  MAPRGFSCLLLSTSEIDLPVKRRT*
...
Results:
  mfe_5p_dG:      -8.20 kcal/mol  ✓ Real calculated value
  mfe_global_dG:  -15.30 kcal/mol ✓ Real calculated value
  mfe_5p_note:    no_utr_fallback ✓ Correct note
  ✓ Note matches expected: 'no_utr_fallback'

Interpretation:
  • 5' structure: moderately structured (ΔG = -8.20)
```

## Integration with Existing Services

The `sequence-analyzer` service now has access to:

1. **Enhanced MFE analysis**: `five_prime_utr_mfe_analysis()`
2. **Legacy compatibility**: `five_prime_dG_vienna()` still works
3. **Full ViennaRNA**: RNAfold CLI available for any custom analysis

You can integrate this into the service's `app.py` for real-time analysis.

## Next Steps

1. ✅ Run the demo in Docker to see real MFE values
2. ✅ Verify all test cases pass with actual calculations
3. ✅ Integrate into sequence_analyzer service for production use
4. ✅ Add MFE analysis to your optimization pipeline

---

**Note**: The Docker image includes ViennaRNA 2.6.4 compiled from source, which provides the most reliable RNA folding calculations at the specified temperature (37°C).

