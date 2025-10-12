# Changelog - Structure Features Service

## [2.0.0] - 2025-10-12 - AlphaFold DB Integration

### 🎉 Major Enhancement

Complete rewrite to integrate **AlphaFold Database API** with intelligent fallback to Lite approximation.

### ✨ New Features

#### AlphaFold DB Integration
- **Real pLDDT scores** from AlphaFold predictions via public API
- **Model metadata** extraction (creation date, version)
- **Confidence classification** (very high, confident, low, very low)
- **Download links** for PDB/mmCIF structure files
- **PAE availability** status tracking
- **Automatic retry** with configurable attempts
- **Graceful fallback** to Lite mode on API failure

#### Enhanced Input Support
- **UniProt ID support** (`uniprot_id` field in input JSONL)
- **Mixed input handling** (UniProt ID + sequence)
- **Flexible field names** (`uniprot_id` or `uniprot_accession`)
- **Sequence-only mode** (backwards compatible)

#### Enhanced Output
- **Source tracking** (`source` field: "afdb" or "lite")
- **AFDB-specific fields** when available:
  - `uniprot_accession`
  - `model_created`
  - `afdb_confidence`
  - `pae_available`
  - `pdb_url`
  - `cif_url`
- **Backwards compatible** structure (all Lite fields preserved)

#### Statistics & Monitoring
- **AFDB success/failure counts**
- **Lite usage tracking**
- **Success rate percentage**
- **Real-time progress** with mode breakdown
- **Detailed logging** (configurable levels)

#### Command Line Interface
- `--no-afdb` - Disable AFDB, use Lite only
- `--afdb-retry N` - Configure API retry count
- `--log-level` - Control verbosity (DEBUG, INFO, WARNING, ERROR)
- Enhanced help text with examples

### 📚 Documentation

- **README.md** - Quick start guide
- **README_AFDB_INTEGRATION.md** - Comprehensive documentation
- **CHANGELOG.md** - This file
- **example_input.jsonl** - Sample input data
- **requirements.txt** - Dependency list

### 🧪 Testing & Examples

- **test_afdb_integration.py** - Comprehensive test suite
- **quickstart.py** - Interactive examples (4 scenarios)
- **demo.sh** - Bash demo script

### 🔧 Technical Improvements

- **Robust API parsing** - Handles multiple AFDB field name variations
- **Error recovery** - Automatic fallback on any AFDB failure
- **Timeout handling** - 30-second timeout with retry
- **Network resilience** - Graceful degradation without requests library
- **Type hints** - Improved code quality
- **Comprehensive logging** - Better debugging

### 🔄 Migration Guide

#### For Users

**No breaking changes!** Existing workflows continue to work:

```bash
# Old usage (still works, now with AFDB enhancement)
python app.py --input data.jsonl --output features.jsonl

# New usage (explicitly disable AFDB if needed)
python app.py --input data.jsonl --output features.jsonl --no-afdb
```

**Output changes:**
- New `source` field in `structure_features` indicates data origin
- Additional AFDB-specific fields when available (can be ignored)
- All original Lite fields preserved

#### For Developers

**API Changes:**

```python
# Old API (still works)
predictor = StructureFeaturesLite()
features = predictor.predict_structure(aa_sequence, protein_id)

# New API (with AFDB support)
predictor = StructureFeaturesLite(use_afdb=True, afdb_retry=2)
features = predictor.predict_structure(
    aa_sequence=aa_sequence,
    protein_id=protein_id,
    uniprot_id=uniprot_id  # New optional parameter
)

# Check source
if features.source == "afdb":
    print(f"Real AlphaFold data: {features.afdb_confidence}")
else:
    print("Lite approximation")
```

**Statistics Changes:**

```python
# New statistics fields
stats = {
    'total_processed': 1000,
    'total_errors': 0,
    'afdb_success': 850,      # New
    'afdb_failed': 150,       # New
    'lite_used': 150,         # New
    'afdb_success_rate': 85.0 # New
}
```

### 📦 Dependencies

**New Required:**
- `requests >= 2.25.0` (for AFDB API)

**Optional:**
- Service works without `requests` (Lite-only mode)
- Automatic detection and graceful fallback

### ⚠️ Known Limitations

1. **AFDB API rate limits** - No official rate limit, but recommend reasonable delays for large batches
2. **UniProt coverage** - Not all proteins in AFDB; coverage varies by organism
3. **Network dependency** - AFDB mode requires internet connectivity
4. **API breaking changes** - AFDB recently changed API (2025-10-07); our implementation handles variations

### 🔮 Future Enhancements

- [ ] Bulk download via FTP for large datasets
- [ ] Local AFDB file caching
- [ ] PAE matrix extraction and analysis
- [ ] Structure quality metrics
- [ ] Batch API optimization
- [ ] Progress bar for large files

### 🙏 Acknowledgments

- **AlphaFold Database Team** - For providing free, programmatic access
- **EBI/EMBL** - For hosting and maintaining AFDB infrastructure
- **DeepMind/Google** - For AlphaFold and making predictions public

---

## [1.0.0] - 2025-10-05 - Initial Release

### Features

- Lite approximation based on sequence properties
- pLDDT estimation from disorder/complexity
- Secondary structure prediction
- Disorder and flexibility ratios
- Signal peptide detection
- Transmembrane helix prediction
- SASA approximation
- JSONL batch processing
- Command-line interface

---

**For full documentation, see README.md and README_AFDB_INTEGRATION.md**

