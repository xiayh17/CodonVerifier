# Structure Features Service - AlphaFold DB Integration

## Overview

This enhanced service provides protein structure features with **two modes**:

1. **AlphaFold DB API (Preferred)**: Fetches real predictions from AlphaFold Database for known UniProt entries
2. **Lite Approximation (Fallback)**: Fast sequence-based estimates when AFDB is unavailable

## Features

### From AlphaFold DB (when available)
- Real pLDDT scores (per-residue confidence)
- Model metadata (creation date, version)
- Confidence categories (very high, confident, low, very low)
- PAE availability status
- Download links for PDB/mmCIF files
- UniProt accession linkage

### From Lite Approximation (fallback)
- pLDDT estimates based on disorder/complexity
- Disorder and flexibility predictions
- Secondary structure estimates (helix/sheet/coil)
- Signal peptide detection
- Transmembrane helix prediction
- SASA approximation

## Installation

```bash
# Install required dependencies
pip install requests  # Required for AFDB API
```

## Usage

### Basic Usage (AFDB + Lite Fallback)

```bash
python app.py --input proteins.jsonl --output features.jsonl
```

### Lite-Only Mode (No AFDB Calls)

```bash
python app.py --input proteins.jsonl --output features.jsonl --no-afdb
```

### With Custom Retry Settings

```bash
python app.py --input proteins.jsonl --output features.jsonl --afdb-retry 3
```

### Test Run (Limit Records)

```bash
python app.py --input proteins.jsonl --output test.jsonl --limit 10
```

## Input Format

Input JSONL file should contain records with **either or both**:

```json
{
  "protein_id": "protein_001",
  "protein_aa": "MKTAYIAKQR...",
  "uniprot_id": "P12345"
}
```

### Field Priority
1. If `uniprot_id` is provided → Try AFDB API first
2. If AFDB fails or no `uniprot_id` → Use `protein_aa` for Lite approximation
3. If neither succeeds → Return default values with warning

## Output Format

Output JSONL file contains all input fields plus `structure_features`:

```json
{
  "protein_id": "protein_001",
  "protein_aa": "MKTAYIAKQR...",
  "uniprot_id": "P12345",
  "structure_features": {
    "source": "afdb",
    "plddt_mean": 82.5,
    "plddt_min": 65.2,
    "plddt_max": 95.8,
    "plddt_std": 8.3,
    "plddt_q25": 76.1,
    "plddt_q75": 89.2,
    "disorder_ratio": 0.12,
    "flexible_ratio": 0.06,
    "uniprot_accession": "P12345",
    "model_created": "2021-11-01",
    "afdb_confidence": "very high",
    "pae_available": true,
    "pdb_url": "https://alphafold.ebi.ac.uk/files/AF-P12345-F1-model_v4.pdb",
    "cif_url": "https://alphafold.ebi.ac.uk/files/AF-P12345-F1-model_v4.cif",
    "helix_ratio": 0.35,
    "sheet_ratio": 0.28,
    "coil_ratio": 0.37,
    ...
  }
}
```

### Source Field
- `"source": "afdb"` → Data from AlphaFold DB API
- `"source": "lite"` → Data from Lite approximation

## Statistics Output

The service reports detailed statistics:

```
============================================================
✓ Structure features generation completed successfully!
============================================================
  Total Processed: 1000
  AFDB Success: 850
  AFDB Failed: 150
  Lite Used: 150
  AFDB Success Rate: 85.0%
  Errors: 0
  Output: features.jsonl (1000 records)
============================================================
```

## AlphaFold DB API Details

### Endpoint
```
https://alphafold.ebi.ac.uk/api/prediction/{uniprot_accession}
```

### Behavior
- **Timeout**: 30 seconds per request
- **Retries**: Configurable (default: 2)
- **Rate Limiting**: Automatic retry with 1-second delay
- **404 Handling**: Silently falls back to Lite mode
- **Error Recovery**: Graceful degradation to Lite approximation

### Recent API Changes (2025-10-07)

AlphaFold DB has made breaking changes to their API. This implementation includes:
- Multiple field name attempts (`pLDDT`, `confidenceScore`, `summary`)
- Flexible download URL parsing
- Robust error handling for structure variations

See: https://www.ebi.ac.uk/pdbe/news/breaking-changes-afdb-predictions-api

## Mixed Input Scenarios

### 1. UniProt IDs Only
```json
{"protein_id": "p1", "uniprot_id": "P12345"}
```
→ Uses AFDB exclusively (no fallback without sequence)

### 2. Sequences Only
```json
{"protein_id": "p1", "protein_aa": "MKTAYIAKQR..."}
```
→ Uses Lite approximation only

### 3. Both Available (Recommended)
```json
{"protein_id": "p1", "uniprot_id": "P12345", "protein_aa": "MKTAYIAKQR..."}
```
→ Tries AFDB first, falls back to Lite if AFDB fails

## Performance Considerations

### AFDB Mode
- **Speed**: ~1-2 seconds per UniProt ID (network dependent)
- **Accuracy**: Real AlphaFold predictions (highest quality)
- **Best for**: Known proteins with UniProt entries

### Lite Mode
- **Speed**: ~0.01 seconds per sequence (instant)
- **Accuracy**: Approximation based on sequence properties
- **Best for**: Novel/synthetic sequences, bulk processing, offline use

### Recommended Strategy
1. Use AFDB for reference/natural proteins (include `uniprot_id`)
2. Use Lite for optimized/synthetic variants
3. Enable both (default) for mixed datasets

## Troubleshooting

### No AFDB Results
```
AFDB Success Rate: 0.0%
```
**Causes**:
- `requests` library not installed → Install via `pip install requests`
- Network connectivity issues → Check firewall/proxy
- Invalid UniProt IDs → Verify IDs are valid and in AFDB
- API downtime → Use `--no-afdb` to skip AFDB calls

### High AFDB Failure Rate
```
AFDB Failed: 800 (80%)
```
**Causes**:
- UniProt IDs not in AlphaFold DB → Use sequences for fallback
- Timeout issues → Increase `--afdb-retry` value
- API rate limiting → Add delays between batch jobs

### All Lite Fallback
```
Lite Used: 1000 (100%)
```
**Causes**:
- No `uniprot_id` fields in input → Add UniProt mappings
- `--no-afdb` flag used → Remove flag for AFDB integration
- AFDB integration disabled → Check `requests` library installation

## Examples

### Example 1: Process Known Human Proteins
```bash
# Input: proteins_with_uniprot.jsonl
# Each record has uniprot_id and protein_aa
python app.py \
  --input data/human_proteins.jsonl \
  --output data/human_features.jsonl \
  --log-level INFO
```

### Example 2: Process Synthetic Variants (Lite Only)
```bash
# Input: synthetic_variants.jsonl
# Only has protein_aa, no uniprot_id
python app.py \
  --input data/synthetic.jsonl \
  --output data/synthetic_features.jsonl \
  --no-afdb
```

### Example 3: Mixed Dataset with Aggressive Retry
```bash
# Some have uniprot_id, some don't
python app.py \
  --input data/mixed_proteins.jsonl \
  --output data/mixed_features.jsonl \
  --afdb-retry 5 \
  --log-level DEBUG
```

## Integration with Complete Features Pipeline

This service integrates seamlessly with the complete features generation pipeline:

```bash
# Step 1: Generate structure features (this service)
python services/structure_features_lite/app.py \
  --input data/base.jsonl \
  --output data/structure.jsonl

# Step 2: Integrate with other features
python services/feature_integrator/app.py \
  --base data/base.jsonl \
  --structure data/structure.jsonl \
  --msa data/msa.json \
  --output data/complete.jsonl
```

## API Reference

### StructureFeaturesLite Class

```python
from app import StructureFeaturesLite

# Initialize with AFDB enabled
predictor = StructureFeaturesLite(use_afdb=True, afdb_retry=2)

# Predict with UniProt ID (tries AFDB)
features = predictor.predict_structure(
    uniprot_id="P12345",
    aa_sequence="MKTAYIAKQR...",  # Fallback
    protein_id="protein_001"
)

# Predict with sequence only (uses Lite)
features = predictor.predict_structure(
    aa_sequence="MKTAYIAKQR...",
    protein_id="protein_002"
)

# Check source
print(features.source)  # "afdb" or "lite"
print(features.plddt_mean)
print(features.afdb_confidence)
```

### Process Function

```python
from app import process_jsonl

stats = process_jsonl(
    input_path="data/input.jsonl",
    output_path="data/output.jsonl",
    limit=None,  # Process all
    use_afdb=True,
    afdb_retry=2
)

print(f"AFDB Success Rate: {stats['afdb_success_rate']:.1f}%")
```

## References

1. AlphaFold Database: https://alphafold.ebi.ac.uk/
2. API Documentation: https://alphafold.ebi.ac.uk/api-docs
3. Breaking Changes Notice: https://www.ebi.ac.uk/pdbe/news/breaking-changes-afdb-predictions-api
4. Biopython AlphaFold DB: https://biopython.org/docs/dev/api/Bio.PDB.alphafold_db.html

## License

Same as parent project (see LICENSE).

