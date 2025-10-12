#!/usr/bin/env python3
"""
Quick Start Example for Structure Features Service

This script demonstrates the basic usage of the enhanced structure features service
with AlphaFold DB integration.
"""

import json
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app import StructureFeaturesLite, process_jsonl


def example_1_single_prediction():
    """Example 1: Predict features for a single protein"""
    
    print("\n" + "=" * 80)
    print("Example 1: Single Protein Prediction")
    print("=" * 80)
    
    predictor = StructureFeaturesLite(use_afdb=True)
    
    # Example with UniProt ID (will try AFDB first)
    print("\n1a. Known protein with UniProt ID (BRCA1):")
    features = predictor.predict_structure(
        uniprot_id="P38398",
        aa_sequence="MDLSALRVEEVQNVINAMQKILECPICLELIKEPVSTKCDHIFCKFCMLK",
        protein_id="BRCA1_HUMAN"
    )
    
    print(f"   Source: {features.source}")
    print(f"   pLDDT Mean: {features.plddt_mean:.2f}")
    print(f"   Disorder Ratio: {features.disorder_ratio:.3f}")
    if features.source == "afdb":
        print(f"   AFDB Confidence: {features.afdb_confidence}")
        print(f"   PDB URL: {features.pdb_url or 'N/A'}")
    
    # Example with sequence only (will use Lite)
    print("\n1b. Synthetic protein (sequence only):")
    features = predictor.predict_structure(
        aa_sequence="MKTAYIAKQRIQVLTQERYLRTLNQLASQPVAQARLEALQAKKAANEAQ",
        protein_id="SYNTHETIC_001"
    )
    
    print(f"   Source: {features.source}")
    print(f"   pLDDT Mean: {features.plddt_mean:.2f}")
    print(f"   Helix Ratio: {features.helix_ratio:.3f}")
    print(f"   Sheet Ratio: {features.sheet_ratio:.3f}")


def example_2_batch_processing():
    """Example 2: Batch process multiple proteins from JSONL"""
    
    print("\n" + "=" * 80)
    print("Example 2: Batch Processing")
    print("=" * 80)
    
    # Create sample input data
    sample_data = [
        {
            "protein_id": "human_brca1",
            "uniprot_id": "P38398",
            "protein_aa": "MDLSALRVEEVQNVINAMQKILECPICLELIKEPVSTKCDHIFCKFCMLK"
        },
        {
            "protein_id": "human_p53",
            "uniprot_id": "P04637",
            "protein_aa": "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDD"
        },
        {
            "protein_id": "synthetic_variant",
            "protein_aa": "MKTAYIAKQRIQVLTQERYLRTLNQLASQPVAQARLEALQAKK"
        }
    ]
    
    # Write to temporary file
    input_file = "/tmp/quickstart_input.jsonl"
    output_file = "/tmp/quickstart_output.jsonl"
    
    print(f"\nCreating sample input file: {input_file}")
    with open(input_file, 'w') as f:
        for record in sample_data:
            f.write(json.dumps(record) + '\n')
    
    # Process the file
    print(f"Processing {len(sample_data)} proteins...")
    stats = process_jsonl(
        input_path=input_file,
        output_path=output_file,
        use_afdb=True,
        afdb_retry=2
    )
    
    print(f"\n✓ Processing complete!")
    print(f"  Total: {stats['total_processed']}")
    print(f"  AFDB Success: {stats['afdb_success']}")
    print(f"  Lite Used: {stats['lite_used']}")
    print(f"  AFDB Success Rate: {stats['afdb_success_rate']:.1f}%")
    
    # Display results
    print(f"\nResults written to: {output_file}")
    print("\nSample results:")
    with open(output_file, 'r') as f:
        for i, line in enumerate(f, 1):
            record = json.loads(line)
            sf = record['structure_features']
            print(f"  {i}. {record['protein_id']:20s} | "
                  f"source={sf['source']:4s} | "
                  f"pLDDT={sf['plddt_mean']:5.1f} | "
                  f"disorder={sf['disorder_ratio']:.3f}")
    
    return output_file


def example_3_lite_only_mode():
    """Example 3: Use Lite-only mode (no AFDB calls)"""
    
    print("\n" + "=" * 80)
    print("Example 3: Lite-Only Mode (Fast, No API Calls)")
    print("=" * 80)
    
    predictor = StructureFeaturesLite(use_afdb=False)
    
    sequences = [
        ("SHORT_PEPTIDE", "MKTAYIAKQR"),
        ("ALPHA_HELIX", "AEEEKEKAEEEKEKAEEEKEKAEEEKEK"),
        ("BETA_SHEET", "VIVIVIVIVIVIVIVIVIVIVIVIVI")
    ]
    
    print("\nProcessing sequences with Lite approximation:")
    for protein_id, seq in sequences:
        features = predictor.predict_structure(
            aa_sequence=seq,
            protein_id=protein_id
        )
        
        print(f"\n  {protein_id}:")
        print(f"    Length: {len(seq)} aa")
        print(f"    pLDDT: {features.plddt_mean:.1f}")
        print(f"    Helix: {features.helix_ratio:.2f}, "
              f"Sheet: {features.sheet_ratio:.2f}, "
              f"Coil: {features.coil_ratio:.2f}")
        print(f"    Disorder: {features.disorder_ratio:.3f}")


def example_4_compare_modes():
    """Example 4: Compare AFDB vs Lite for the same protein"""
    
    print("\n" + "=" * 80)
    print("Example 4: Comparing AFDB vs Lite Mode")
    print("=" * 80)
    
    test_sequence = "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD"
    
    # AFDB mode
    print("\nFetching with AFDB...")
    predictor_afdb = StructureFeaturesLite(use_afdb=True)
    features_afdb = predictor_afdb.predict_structure(
        uniprot_id="P04637",  # Human TP53
        aa_sequence=test_sequence,
        protein_id="TP53_HUMAN"
    )
    
    # Lite mode
    print("Computing with Lite...")
    predictor_lite = StructureFeaturesLite(use_afdb=False)
    features_lite = predictor_lite.predict_structure(
        aa_sequence=test_sequence,
        protein_id="TP53_HUMAN_LITE"
    )
    
    # Compare
    print("\nComparison (Human TP53):")
    print(f"{'Metric':<20s} {'AFDB':<15s} {'Lite':<15s} {'Difference':<15s}")
    print("-" * 65)
    
    metrics = [
        ('Source', features_afdb.source, features_lite.source, 'N/A'),
        ('pLDDT Mean', f"{features_afdb.plddt_mean:.2f}", f"{features_lite.plddt_mean:.2f}", 
         f"{abs(features_afdb.plddt_mean - features_lite.plddt_mean):.2f}"),
        ('pLDDT Std', f"{features_afdb.plddt_std:.2f}", f"{features_lite.plddt_std:.2f}",
         f"{abs(features_afdb.plddt_std - features_lite.plddt_std):.2f}"),
        ('Disorder Ratio', f"{features_afdb.disorder_ratio:.3f}", f"{features_lite.disorder_ratio:.3f}",
         f"{abs(features_afdb.disorder_ratio - features_lite.disorder_ratio):.3f}"),
    ]
    
    for metric, afdb_val, lite_val, diff in metrics:
        print(f"{metric:<20s} {afdb_val:<15s} {lite_val:<15s} {diff:<15s}")
    
    if features_afdb.source == "afdb":
        print(f"\nAFDB-specific info:")
        print(f"  Confidence: {features_afdb.afdb_confidence}")
        print(f"  Model Date: {features_afdb.model_created}")
        print(f"  PAE Available: {features_afdb.pae_available}")


def main():
    """Run all examples"""
    
    print("\n" + "█" * 80)
    print("█  Structure Features Service - Quick Start Examples")
    print("█" * 80)
    
    # Check dependencies
    try:
        import requests
        print("\n✓ requests library available - AFDB integration enabled")
    except ImportError:
        print("\n✗ requests library not available")
        print("  AFDB integration will be disabled")
        print("  Install with: pip install requests")
        print("\n  Continuing with Lite-only examples...")
    
    try:
        # Run examples
        example_1_single_prediction()
        example_2_batch_processing()
        example_3_lite_only_mode()
        example_4_compare_modes()
        
        print("\n" + "=" * 80)
        print("✓ All examples completed successfully!")
        print("=" * 80)
        print("\nNext steps:")
        print("  1. Check output files in /tmp/")
        print("  2. Run full test suite: python test_afdb_integration.py")
        print("  3. Process your own data: python app.py --input your_data.jsonl --output output.jsonl")
        print("  4. Read documentation: cat README.md")
        print("\n")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

