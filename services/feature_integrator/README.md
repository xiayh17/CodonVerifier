# Feature Integrator Service

A microservice that integrates all enhanced features (structural, evolutionary, and contextual) into a unified format ready for machine learning training. This service is part of the CodonVerifier system for protein expression prediction.

## Overview

The Feature Integrator Service combines multiple types of features into training-ready records:

- **Structural Features**: Protein structure-based features (from structure analysis)
- **Evolutionary Features**: Multiple sequence alignment (MSA) features (from evolutionary analysis)
- **Contextual Features**: Expression context features extracted from metadata
- **Base Features**: Original sequence and expression data

## Features

### Context Feature Extraction

The service automatically extracts contextual features from metadata including:

- **Promoter Strength**: Database of known promoter strengths (T7, lacUV5, tac, etc.)
- **RBS/Kozak Strength**: Ribosome binding site strength scoring
- **Vector Properties**: Copy number, selection markers
- **Expression Conditions**: Temperature, inducer concentration, growth phase
- **Localization**: Cellular localization scoring

### Feature Integration

- Merges features from multiple sources with proper prefixes:
  - `struct_*`: Structural features
  - `evo_*`: Evolutionary features  
  - `ctx_*`: Context features
- Handles missing features gracefully
- Validates and normalizes feature values
- Preserves original metadata for reference

## Usage

### Command Line Interface

```bash
python app.py --input base_records.jsonl \
              --structure-features structure_features.json \
              --msa-features msa_features.json \
              --output integrated_features.jsonl \
              --limit 1000
```

### Arguments

- `--input` (required): Input JSONL file with base records
- `--structure-features` (optional): Structure features JSON/JSONL file
- `--msa-features` (optional): MSA features JSON/JSONL file  
- `--output` (required): Output JSONL file for integrated features
- `--limit` (optional): Limit number of records for testing
- `--log-level` (optional): Logging level (DEBUG, INFO, WARNING, ERROR)

### Input Format

#### Base Records (JSONL)
```json
{
  "protein_id": "protein_001",
  "sequence": "ATGAAACGC...",
  "protein_aa": "MKRL...",
  "host": "E_coli",
  "expression": 1250.5,
  "expression_unit": "RFU",
  "assay": "bulk_fluor",
  "metadata": {
    "promoter": "T7",
    "rbs": "strong",
    "rbs_spacing": 8,
    "copy_number": 1,
    "conditions": {
      "temperature": 37.0,
      "inducer_conc": 0.1
    },
    "growth_phase": "log",
    "localization": "cytoplasm"
  }
}
```

#### Structure Features (JSON/JSONL)
```json
{
  "protein_id": "protein_001",
  "structure_features": {
    "secondary_structure_content": 0.65,
    "disorder_score": 0.23,
    "solvent_accessibility": 0.45
  }
}
```

#### MSA Features (JSON/JSONL)
```json
{
  "protein_id": "protein_001", 
  "msa_features": {
    "conservation_score": 0.78,
    "entropy": 1.2,
    "gap_frequency": 0.05
  }
}
```

### Output Format

```json
{
  "sequence": "ATGAAACGC...",
  "protein_aa": "MKRL...",
  "host": "E_coli",
  "expression": {
    "value": 1250.5,
    "unit": "RFU", 
    "assay": "bulk_fluor"
  },
  "extra_features": {
    "struct_secondary_structure_content": 0.65,
    "struct_disorder_score": 0.23,
    "evo_conservation_score": 0.78,
    "evo_entropy": 1.2,
    "ctx_promoter_strength": 1.0,
    "ctx_rbs_strength": 1.0,
    "ctx_temperature_norm": 0.0,
    "ctx_growth_phase_score": 1.0
  },
  "metadata": { ... },
  "protein_id": "protein_001"
}
```

## Docker Usage

### Build Image
```bash
docker build -t feature-integrator .
```

### Run Container
```bash
docker run -v /path/to/data:/data feature-integrator \
  --input /data/base_records.jsonl \
  --structure-features /data/structure_features.json \
  --msa-features /data/msa_features.json \
  --output /data/integrated_features.jsonl
```

### Docker Compose
```yaml
services:
  feature-integrator:
    build: ./services/feature_integrator
    volumes:
      - ./data:/data
    command: [
      "--input", "/data/base_records.jsonl",
      "--structure-features", "/data/structure_features.json", 
      "--msa-features", "/data/msa_features.json",
      "--output", "/data/integrated_features.jsonl"
    ]
```

## Context Feature Database

### Promoter Strengths
- T7: 1.0 (strongest)
- lacUV5: 0.8
- tac: 0.85
- trc: 0.75
- araBAD: 0.7
- AOX1: 0.9 (P. pastoris)
- GAL1: 0.85 (S. cerevisiae)
- CMV: 0.9 (Mammalian)
- EF1A: 0.85
- SV40: 0.7
- unknown/default: 0.5

### RBS Strengths
- strong/optimal: 1.0
- medium: 0.6
- weak/suboptimal: 0.3
- unknown/default: 0.5

### Growth Phase Encoding
- log/exponential: 1.0
- stationary: 0.5
- lag: 0.3

### Localization Scoring
- cytoplasm: 1.0
- periplasm: 0.8
- membrane: 0.6
- secreted/extracellular: 0.7

## Dependencies

- Python 3.10+
- numpy
- pandas

## Logging

The service provides detailed logging including:
- Processing statistics
- Feature integration counts
- Error reporting
- Sample output preview

## Error Handling

- Graceful handling of missing feature files
- Robust JSON parsing with error recovery
- Detailed error logging for debugging
- Continues processing even if individual records fail

## Performance

- Memory-efficient processing of large datasets
- Progress reporting every 100 records
- Optional record limiting for testing
- Support for both JSON and JSONL input formats

## Integration

This service is designed to work with other CodonVerifier microservices:
- Receives base records from data preprocessing
- Integrates features from structure analyzer and evolutionary analyzer
- Outputs training-ready data for ML models

## Author

CodonVerifier Team  
Date: 2025-10-05
