# Structure Features Service (Lite with AlphaFold DB Integration)

Fast and accurate protein structure feature generation with dual-mode operation:
1. **AlphaFold DB API** - Real predictions for known proteins
2. **Lite Approximation** - Fast sequence-based estimates

## 🚀 Quick Start

```bash
# Install dependencies
pip install requests

# Basic usage (AFDB + Lite fallback)
python app.py --input proteins.jsonl --output features.jsonl

# Lite-only mode (no API calls)
python app.py --input proteins.jsonl --output features.jsonl --no-afdb
```

## 📋 Features

### AlphaFold DB Mode (Preferred for Known Proteins)
- ✓ Real pLDDT scores from AlphaFold predictions
- ✓ Model metadata (creation date, version)
- ✓ Confidence classifications
- ✓ PDB/mmCIF download links
- ✓ PAE availability status

### Lite Mode (Fallback for Novel/Synthetic Sequences)
- ✓ pLDDT approximation from disorder/complexity
- ✓ Secondary structure prediction (helix/sheet/coil)
- ✓ Disorder and flexibility ratios
- ✓ Signal peptide detection
- ✓ Transmembrane helix prediction
- ✓ SASA estimation

## 🔄 How It Works

```
Input Record
    ↓
Has uniprot_id? ──YES──→ Try AlphaFold DB API
    |                            |
    NO                      Success? ──YES──→ Use AFDB Data
    |                           |
    ↓                           NO
Has protein_aa? ──YES──→ Use Lite Approximation
    |
    NO
    ↓
Return Defaults
```

## 📝 Input Format

Input JSONL with **either or both**:

```json
{
  "protein_id": "protein_001",
  "protein_aa": "MKTAYIAKQRIQVLTQERYLRTLNQLASQPVAQARLEALQAKK",
  "uniprot_id": "P12345"
}
```

## 📤 Output Format

```json
{
  "protein_id": "protein_001",
  "structure_features": {
    "source": "afdb",  // or "lite"
    "plddt_mean": 82.5,
    "plddt_min": 65.2,
    "plddt_max": 95.8,
    "plddt_std": 8.3,
    "plddt_q25": 76.1,
    "plddt_q75": 89.2,
    "disorder_ratio": 0.12,
    "flexible_ratio": 0.06,
    "uniprot_accession": "P12345",
    "afdb_confidence": "very high",
    "pae_available": true,
    "pdb_url": "https://alphafold.ebi.ac.uk/files/...",
    "helix_ratio": 0.35,
    "sheet_ratio": 0.28,
    "coil_ratio": 0.37,
    ...
  }
}
```

## 📊 Statistics Output

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

## 🧪 Testing

```bash
# Run test suite
python test_afdb_integration.py

# Test with small dataset
python app.py --input test.jsonl --output test_out.jsonl --limit 10
```

## 🔧 Command Line Options

```bash
python app.py [OPTIONS]

Required:
  --input PATH          Input JSONL file
  --output PATH         Output JSONL file

Optional:
  --limit N            Process only first N records
  --no-afdb            Disable AlphaFold DB (use Lite only)
  --afdb-retry N       Number of API retries (default: 2)
  --log-level LEVEL    Logging level: DEBUG, INFO, WARNING, ERROR
```

## 📚 Documentation

- **[Full AFDB Integration Guide](README_AFDB_INTEGRATION.md)** - Detailed documentation
- **[Test Script](test_afdb_integration.py)** - Example usage and testing

## 🔗 API Integration Details

- **Endpoint**: `https://alphafold.ebi.ac.uk/api/prediction/{accession}`
- **Timeout**: 30 seconds
- **Retries**: Configurable (default: 2)
- **Fallback**: Automatic to Lite mode on failure

## 💡 Best Practices

1. **Include both `uniprot_id` and `protein_aa`** for maximum flexibility
2. **Use AFDB for reference proteins** (natural, well-studied)
3. **Use Lite for synthetic variants** (optimized, novel sequences)
4. **Enable both modes** (default) for mixed datasets
5. **Check `source` field** in output to know which mode was used

## ⚠️ Troubleshooting

### Issue: AFDB Success Rate 0%
**Solution**: 
- Install requests: `pip install requests`
- Check network connectivity
- Verify UniProt IDs are valid

### Issue: High AFDB Failure Rate
**Solution**:
- UniProt IDs may not be in AlphaFold DB
- Increase retry count: `--afdb-retry 5`
- Check API status: https://alphafold.ebi.ac.uk/

### Issue: All using Lite mode
**Solution**:
- Ensure input has `uniprot_id` field
- Don't use `--no-afdb` flag
- Check requests library is installed

## 📖 References

- [AlphaFold Database](https://alphafold.ebi.ac.uk/)
- [AFDB API Documentation](https://alphafold.ebi.ac.uk/api-docs)
- [API Breaking Changes Notice](https://www.ebi.ac.uk/pdbe/news/breaking-changes-afdb-predictions-api)

## 📄 License

Same as parent project - see [LICENSE](../../LICENSE)

---

**Part of the CodonVerifier Project**

