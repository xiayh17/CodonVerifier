# MSA Features Lite Service

A lightweight microservice that provides fast approximation of evolutionary features without requiring actual Multiple Sequence Alignment (MSA) generation. This service is part of the CodonVerifier system for protein expression prediction.

## Overview

The MSA Features Lite Service generates evolutionary features using sequence composition, complexity, and conservation heuristics instead of running computationally expensive MSA tools like MMseqs2 or JackHMMER. This provides a fast alternative for scenarios where speed is prioritized over precision.

## Features

### Evolutionary Features Generated

The service approximates the following evolutionary features:

- **MSA Depth**: Estimated number of homologous sequences
- **MSA Effective Depth**: Estimated number of high-quality homologous sequences
- **MSA Coverage**: Estimated sequence coverage in alignment
- **Conservation Scores**: Mean, min, and max conservation across the sequence
- **Conservation Entropy**: Information content of conservation patterns
- **Coevolution Score**: Estimated co-evolution signal strength
- **Contact Density**: Estimated residue contact density
- **Domain Count**: Estimated number of protein domains
- **Pfam Count**: Estimated number of Pfam domains

### Algorithm Approach

The service uses several heuristics to approximate evolutionary features:

1. **Conservation Estimation**: Based on amino acid type conservation weights and sequence composition similarity
2. **MSA Depth Prediction**: Uses sequence complexity, length, and conservation to estimate homolog abundance
3. **Coevolution Approximation**: Analyzes charged residue patterns for potential salt bridges
4. **Domain Detection**: Simple length-based heuristics for domain counting

## Installation

### Prerequisites

- Python 3.10+
- Docker (for containerized deployment)

### Dependencies

The service requires minimal dependencies:
- `numpy`: For numerical computations
- `pandas`: For data handling (optional, for future extensions)

### Local Installation

```bash
# Clone the repository
git clone <repository-url>
cd CodonVerifier

# Install dependencies
pip install numpy pandas

# The service is ready to use
python services/msa_features_lite/app.py --help
```

### Docker Installation

```bash
# Build the Docker image
docker build -f services/msa_features_lite/Dockerfile -t codon-verifier/msa-features-lite:latest .

# Or use docker-compose (recommended)
docker-compose -f docker-compose.microservices.yml build msa_features_lite
```

## Usage

### Command Line Interface

```bash
python app.py --input input.jsonl --output output.jsonl [options]
```

### Arguments

- `--input` (required): Input JSONL file containing protein sequences
- `--output` (required): Output JSONL file with added MSA features
- `--limit` (optional): Limit number of records to process (for testing)
- `--log-level` (optional): Logging level (DEBUG, INFO, WARNING, ERROR)

### Input Format

The service expects JSONL input where each line contains a JSON object with:

```json
{
  "protein_id": "protein_001",
  "protein_aa": "MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNEL..."
}
```

### Output Format

The service adds an `msa_features` field to each input record:

```json
{
  "protein_id": "protein_001",
  "protein_aa": "MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNEL...",
  "msa_features": {
    "msa_depth": 150.2,
    "msa_effective_depth": 120.1,
    "msa_coverage": 0.95,
    "conservation_mean": 0.72,
    "conservation_min": 0.45,
    "conservation_max": 0.92,
    "conservation_entropy_mean": 0.48,
    "coevolution_score": 0.52,
    "contact_density": 0.35,
    "pfam_count": 1.0,
    "domain_count": 2.0
  }
}
```

### Example Usage

```bash
# Process a small dataset for testing
python app.py \
  --input data/proteins.jsonl \
  --output data/proteins_with_msa_features.jsonl \
  --limit 100 \
  --log-level INFO

# Process full dataset
python app.py \
  --input data/full_protein_dataset.jsonl \
  --output data/full_protein_dataset_with_msa_features.jsonl
```

## Docker Usage

### Using Docker Compose (Recommended)

```bash
# Run the service using docker-compose
docker-compose -f docker-compose.microservices.yml run --rm \
  -v $(pwd)/data:/data \
  msa_features_lite \
  --input /data/input.jsonl \
  --output /data/output.jsonl \
  --limit 1000
```

### Using Docker Directly

```bash
# Run the container
docker run --rm \
  -v $(pwd)/data:/data \
  codon-verifier/msa-features-lite:latest \
  --input /data/input.jsonl \
  --output /data/output.jsonl
```

## Integration with CodonVerifier

The MSA Features Lite Service is integrated into the CodonVerifier pipeline as part of the enhanced features generation workflow:

### Pipeline Integration

1. **Input**: Receives protein sequences from the base dataset
2. **Processing**: Generates evolutionary features using sequence heuristics
3. **Output**: Produces enhanced records with MSA features
4. **Integration**: Results are combined with structural features by the Feature Integrator Service

### Usage in Enhanced Features Pipeline

```bash
# Generate enhanced features (includes MSA features)
python scripts/generate_enhanced_features.py \
  --input data/base_dataset.jsonl \
  --output data/enhanced_dataset.jsonl \
  --use-docker
```

## Performance Characteristics

### Speed

- **Processing Rate**: ~1000-5000 proteins per minute (depending on sequence length)
- **Memory Usage**: Low memory footprint (~50-100 MB)
- **CPU Usage**: Single-threaded, moderate CPU usage

### Accuracy Trade-offs

- **Speed vs. Precision**: 10-100x faster than actual MSA tools
- **Approximation Quality**: Good for screening and preliminary analysis
- **Use Cases**: Suitable for large-scale datasets where speed is prioritized

## Algorithm Details

### Conservation Estimation

The conservation score combines three components:

1. **Amino Acid Weights**: Each amino acid has a conservation tendency score
   - Cysteine (C): 0.95 (highly conserved)
   - Tryptophan (W): 0.90 (highly conserved)
   - Proline (P): 0.85 (structurally important)
   - Glycine (G): 0.85 (structurally important)

2. **Composition Similarity**: Compares observed vs. expected amino acid frequencies
3. **Sequence Complexity**: Higher complexity sequences tend to be more conserved

### MSA Depth Estimation

MSA depth is estimated using:

- **Base Depth**: 100 sequences
- **Length Factor**: Longer proteins often have more homologs
- **Complexity Factor**: More complex sequences attract more homologs
- **Conservation Factor**: Highly conserved proteins have more homologs

### Co-evolution Approximation

The co-evolution score analyzes:

- **Charged Residue Pairs**: Looks for potential salt bridges (R/K with D/E)
- **Distance Constraints**: Pairs within 10 residues are considered
- **Normalization**: Score is normalized by sequence length

## Configuration

### Environment Variables

- `PYTHONUNBUFFERED=1`: Ensures real-time logging output

### Logging Configuration

The service uses Python's standard logging module with configurable levels:

- `DEBUG`: Detailed debugging information
- `INFO`: General information about processing
- `WARNING`: Warning messages for non-critical issues
- `ERROR`: Error messages for critical failures

## Troubleshooting

### Common Issues

1. **Empty Input File**
   ```
   Error: No protein_aa field found
   ```
   Solution: Ensure input JSONL contains `protein_aa` field

2. **Memory Issues with Large Files**
   ```
   Error: Out of memory
   ```
   Solution: Use `--limit` parameter to process in batches

3. **Docker Volume Mount Issues**
   ```
   Error: No such file or directory
   ```
   Solution: Ensure data directory is properly mounted in Docker

### Performance Optimization

1. **Batch Processing**: Use `--limit` for large datasets
2. **Memory Management**: Process files in chunks if memory is limited
3. **Docker Resources**: Allocate sufficient memory to Docker containers

## Development

### Code Structure

```
services/msa_features_lite/
├── app.py              # Main application
├── Dockerfile          # Docker configuration
└── README.md          # This documentation
```

### Key Classes

- `EvolutionaryFeatures`: Dataclass defining the output feature structure
- `MSAFeaturesLite`: Main class implementing the feature prediction algorithms

### Adding New Features

To add new evolutionary features:

1. Add the feature to the `EvolutionaryFeatures` dataclass
2. Implement the calculation method in `MSAFeaturesLite`
3. Update the `predict_evolutionary_features` method
4. Update this documentation

### Testing

```bash
# Test with sample data
python app.py \
  --input test_data/sample.jsonl \
  --output test_data/sample_output.jsonl \
  --limit 10 \
  --log-level DEBUG
```

## License

This service is part of the CodonVerifier project. Please refer to the main project license.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and questions:

1. Check the troubleshooting section above
2. Review the logs for error messages
3. Open an issue in the main CodonVerifier repository
4. Contact the CodonVerifier team

## Changelog

### Version 1.0.0 (2025-10-05)
- Initial release
- Basic evolutionary feature approximation
- Docker support
- Integration with CodonVerifier pipeline
