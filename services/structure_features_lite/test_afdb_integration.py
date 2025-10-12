#!/usr/bin/env python3
"""
Test Script for AlphaFold DB Integration

This script demonstrates the enhanced structure features service with AFDB integration.
"""

import json
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app import StructureFeaturesLite

def test_afdb_integration():
    """Test AFDB integration with known UniProt IDs"""
    
    print("=" * 80)
    print("Testing AlphaFold DB Integration")
    print("=" * 80)
    
    # Test cases with known UniProt IDs
    test_cases = [
        {
            "name": "Human BRCA1 (Known in AFDB)",
            "uniprot_id": "P38398",
            "protein_id": "BRCA1_HUMAN",
            "protein_aa": "MDLSALRVEE..."  # Partial sequence as fallback
        },
        {
            "name": "Human TP53 (Known in AFDB)",
            "uniprot_id": "P04637",
            "protein_id": "P53_HUMAN",
            "protein_aa": "MEEPQSDPSV..."
        },
        {
            "name": "Fake UniProt ID (Should fallback to Lite)",
            "uniprot_id": "FAKE123",
            "protein_id": "FAKE_PROTEIN",
            "protein_aa": "MKTAYIAKQRIQVLTQERYLRTLNQLASQPVAQARLEALQAKK"
        },
        {
            "name": "Sequence Only (Lite mode)",
            "protein_id": "SYNTHETIC_1",
            "protein_aa": "MKTAYIAKQRIQVLTQERYLRTLNQLASQPVAQARLEALQAKK"
        }
    ]
    
    # Initialize predictor with AFDB enabled
    predictor = StructureFeaturesLite(use_afdb=True, afdb_retry=1)
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'-' * 80}")
        print(f"Test Case {i}: {test_case['name']}")
        print(f"{'-' * 80}")
        
        features = predictor.predict_structure(
            aa_sequence=test_case.get('protein_aa'),
            protein_id=test_case.get('protein_id'),
            uniprot_id=test_case.get('uniprot_id')
        )
        
        print(f"✓ Source: {features.source}")
        print(f"  pLDDT Mean: {features.plddt_mean:.2f}")
        print(f"  pLDDT Range: [{features.plddt_min:.2f}, {features.plddt_max:.2f}]")
        print(f"  pLDDT Std: {features.plddt_std:.2f}")
        print(f"  Disorder Ratio: {features.disorder_ratio:.3f}")
        
        if features.source == "afdb":
            print(f"  AFDB Confidence: {features.afdb_confidence}")
            print(f"  UniProt Accession: {features.uniprot_accession}")
            print(f"  Model Created: {features.model_created}")
            print(f"  PAE Available: {features.pae_available}")
            if features.pdb_url:
                print(f"  PDB URL: {features.pdb_url}")
        else:
            print(f"  Helix Ratio: {features.helix_ratio:.3f}")
            print(f"  Sheet Ratio: {features.sheet_ratio:.3f}")
            print(f"  Coil Ratio: {features.coil_ratio:.3f}")
        
        results.append({
            "test_case": test_case['name'],
            "source": features.source,
            "plddt_mean": features.plddt_mean,
            "uniprot_id": test_case.get('uniprot_id'),
            "protein_id": test_case.get('protein_id')
        })
    
    # Print summary statistics
    print(f"\n{'=' * 80}")
    print("Summary Statistics")
    print(f"{'=' * 80}")
    print(f"Total Tests: {len(test_cases)}")
    print(f"AFDB Success: {predictor.stats['afdb_success']}")
    print(f"AFDB Failed: {predictor.stats['afdb_failed']}")
    print(f"Lite Used: {predictor.stats['lite_used']}")
    
    afdb_total = predictor.stats['afdb_success'] + predictor.stats['afdb_failed']
    if afdb_total > 0:
        success_rate = predictor.stats['afdb_success'] / afdb_total * 100
        print(f"AFDB Success Rate: {success_rate:.1f}%")
    
    print(f"\n{'=' * 80}")
    print("Test Results Summary")
    print(f"{'=' * 80}")
    for result in results:
        status = "✓ AFDB" if result['source'] == "afdb" else "○ Lite"
        print(f"{status} | {result['test_case']:40s} | pLDDT: {result['plddt_mean']:.1f}")
    
    return results


def test_batch_processing():
    """Test batch processing with JSONL file"""
    
    print(f"\n\n{'=' * 80}")
    print("Testing Batch Processing")
    print(f"{'=' * 80}")
    
    # Create test JSONL file
    test_data = [
        {
            "protein_id": "test_001",
            "uniprot_id": "P38398",
            "protein_aa": "MDLSALRVEEVQNVINAMQKILECPICLELIKEPVSTKCDHIFCKFCMLK"
        },
        {
            "protein_id": "test_002",
            "protein_aa": "MKTAYIAKQRIQVLTQERYLRTLNQLASQPVAQARLEALQAKK"
        },
        {
            "protein_id": "test_003",
            "uniprot_id": "P04637",
            "protein_aa": "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDD"
        }
    ]
    
    test_input = "/tmp/test_structure_input.jsonl"
    test_output = "/tmp/test_structure_output.jsonl"
    
    # Write test input
    print(f"Creating test input: {test_input}")
    with open(test_input, 'w') as f:
        for record in test_data:
            f.write(json.dumps(record) + '\n')
    
    # Process using the main function
    from app import process_jsonl
    
    print(f"Processing {len(test_data)} records...")
    stats = process_jsonl(
        input_path=test_input,
        output_path=test_output,
        use_afdb=True,
        afdb_retry=1
    )
    
    print(f"\n✓ Processing complete!")
    print(f"  Processed: {stats['total_processed']}")
    print(f"  AFDB Success: {stats['afdb_success']}")
    print(f"  Lite Used: {stats['lite_used']}")
    print(f"  AFDB Success Rate: {stats['afdb_success_rate']:.1f}%")
    
    # Read and display results
    print(f"\nOutput file: {test_output}")
    print("Sample results:")
    with open(test_output, 'r') as f:
        for i, line in enumerate(f, 1):
            record = json.loads(line)
            sf = record['structure_features']
            print(f"  {i}. {record['protein_id']}: source={sf['source']}, pLDDT={sf['plddt_mean']:.1f}")
    
    print(f"\nTest files created:")
    print(f"  Input:  {test_input}")
    print(f"  Output: {test_output}")
    
    return test_output


def main():
    """Run all tests"""
    
    print("\n" + "=" * 80)
    print(" AlphaFold DB Integration Test Suite")
    print("=" * 80 + "\n")
    
    # Check if requests is available
    try:
        import requests
        print("✓ requests library available - AFDB integration enabled")
    except ImportError:
        print("✗ requests library not available - AFDB integration disabled")
        print("  Install with: pip install requests")
        return
    
    try:
        # Test 1: Individual predictions
        print("\n" + "█" * 80)
        print("█ TEST 1: Individual Predictions")
        print("█" * 80)
        results = test_afdb_integration()
        
        # Test 2: Batch processing
        print("\n" + "█" * 80)
        print("█ TEST 2: Batch Processing")
        print("█" * 80)
        output_file = test_batch_processing()
        
        print(f"\n\n{'=' * 80}")
        print("✓ All tests completed successfully!")
        print(f"{'=' * 80}")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

