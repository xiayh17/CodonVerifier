#!/usr/bin/env python3
"""
Demo and test script for the new 5' UTR / mRNA structure MFE analysis.

Tests the two-tier window logic:
- If utr5_len >= 20: use [-20..+50] window (or [+1..+50] if only CDS available)
- Otherwise: fallback to [+1..+50] with note "no_utr_fallback"
"""

import sys
from pathlib import Path

# Add the package to path
sys.path.insert(0, str(Path(__file__).parent))

from codon_verifier.metrics import five_prime_utr_mfe_analysis

# Test cases from Human.tsv dataset
test_cases = [
    {
        "name": "HMN1_HUMAN - 24aa, 75nt",
        "sequence": "ATGGCTCCACGAGGGTTCAGCTGTCTCTTACTTTCAACCAGTGAAATTGACCTGCCCGTGAAGAGGCGGACATAA",
        "protein": "MAPRGFSCLLLSTSEIDLPVKRRT*",
        "utr5_len": 0,  # No UTR
        "expected_note": "no_utr_fallback"
    },
    {
        "name": "HMN3_HUMAN - 24aa, 75nt",
        "sequence": "ATGGCCACACGAAGGTTCAGCTGTCTCTTACTTTCAACCAGTGAAATTGATCTATCCGTGAAGAGGCGGATATAA",
        "protein": "MATRRFSCLLLSTSEIDLSVKRRI*",
        "utr5_len": 0,
        "expected_note": "no_utr_fallback"
    },
    {
        "name": "LSP1N_HUMAN - 25aa, 78nt (long sequence)",
        "sequence": "ATGCTGGATATTTTCATCCTGATGTTCTTTGCCATCATAGGCCTGGTCATTCTGTCCTACATTATCTATCTGCTCTAG",
        "protein": "MLDIFILMFFAIIGLVILSYIIYLL*",
        "utr5_len": 0,
        "expected_note": "no_utr_fallback"
    },
    {
        "name": "SARCO_HUMAN - 31aa, 96nt (longer sequence)",
        "sequence": "ATGGGGATAAACACCCGGGAGCTGTTTCTCAACTTCACTATTGTCTTGATTACGGTTATTCTTATGTGGCTCCTTGTGAGGTCCTATCAGTACTGA",
        "protein": "MGINTRELFLNFTIVLITVILMWLLVRSYQY*",
        "utr5_len": 0,
        "expected_note": "no_utr_fallback"
    },
    {
        "name": "HRURF_HUMAN - 34aa, 105nt (even longer)",
        "sequence": "ATGGCGCAACCTACGGCCTCGGCCCAGAAGCTGGTGCGGCCGATCCGCGCCGTGTGCCGCATCCTGCAGATCCCGGAGTCCGACCCCTCCAACCTGCGGCCCTAG",
        "protein": "MAQPTASAQKLVRPIRAVCRILQIPESDPSNLRP*",
        "utr5_len": 0,
        "expected_note": "no_utr_fallback"
    },
    {
        "name": "Simulated with short UTR (15nt)",
        "sequence": "ATGGCTCCACGAGGGTTCAGCTGTCTCTTACTTTCAACCAGTGAAATTGACCTGCCCGTGAAGAGGCGGACATAA",
        "protein": "MAPRGFSCLLLSTSEIDLPVKRRT*",
        "utr5_len": 15,  # Short UTR < 20
        "expected_note": "no_utr_fallback"
    },
    {
        "name": "Simulated with adequate UTR (25nt)",
        "sequence": "ATGGGGATAAACACCCGGGAGCTGTTTCTCAACTTCACTATTGTCTTGATTACGGTTATTCTTATGTGGCTCCTTGTGAGGTCCTATCAGTACTGA",
        "protein": "MGINTRELFLNFTIVLITVILMWLLVRSYQY*",
        "utr5_len": 25,  # UTR >= 20
        "expected_note": "utr_available_but_cds_only"  # We only have CDS
    },
]


def main():
    print("=" * 80)
    print("5' UTR / mRNA Structure MFE Analysis - Demo and Test")
    print("=" * 80)
    print()
    print("This demo tests the two-tier window logic for MFE calculation:")
    print("  • If utr5_len >= 20: window is [-20..+50] (or [+1..+50] if CDS-only)")
    print("  • Otherwise: fallback window [+1..+50] with note 'no_utr_fallback'")
    print("  • Temperature fixed at 37°C for folding")
    print()
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'=' * 80}")
        print(f"Test Case {i}: {test['name']}")
        print(f"{'=' * 80}")
        print(f"Protein:  {test['protein']}")
        print(f"CDS:      {test['sequence'][:60]}{'...' if len(test['sequence']) > 60 else ''}")
        print(f"Length:   {len(test['sequence'])} nt")
        print(f"UTR5 len: {test['utr5_len']} nt")
        print()
        
        # Run MFE analysis
        result = five_prime_utr_mfe_analysis(
            dna=test['sequence'],
            utr5_len=test['utr5_len'],
            temperature=37.0
        )
        
        # Display results
        print("Results:")
        print(f"  mfe_5p_dG:      {result['mfe_5p_dG']:.2f} kcal/mol" if result['mfe_5p_dG'] is not None else "  mfe_5p_dG:      None (ViennaRNA not available)")
        print(f"  mfe_global_dG:  {result['mfe_global_dG']:.2f} kcal/mol" if result['mfe_global_dG'] is not None else "  mfe_global_dG:  None (ViennaRNA not available)")
        print(f"  mfe_5p_note:    {result['mfe_5p_note']}")
        
        # Validate expected note
        if result['mfe_5p_note'] == test['expected_note']:
            print(f"  ✓ Note matches expected: '{test['expected_note']}'")
        else:
            print(f"  ✗ Note mismatch! Expected: '{test['expected_note']}', Got: '{result['mfe_5p_note']}'")
        
        # Interpretation
        print("\nInterpretation:")
        if test['utr5_len'] >= 20:
            print(f"  • UTR length ({test['utr5_len']}nt) >= 20: should use [-20..+50] window")
            print(f"  • Since only CDS provided, using [+1..+50] from CDS")
        else:
            print(f"  • UTR length ({test['utr5_len']}nt) < 20: fallback to [+1..+50] window")
        
        if result['mfe_5p_dG'] is not None:
            if len(test['sequence']) >= 51:
                print(f"  • 5' region: positions [+1..+50] = 50nt analyzed")
            else:
                print(f"  • 5' region: positions [+1..end] = {len(test['sequence'])-1}nt analyzed (sequence too short)")
            print(f"  • Global MFE: entire {len(test['sequence'])}nt sequence")
            
            if result['mfe_5p_dG'] > -3:
                print(f"  • 5' structure: weakly structured (ΔG = {result['mfe_5p_dG']:.2f})")
            elif result['mfe_5p_dG'] > -10:
                print(f"  • 5' structure: moderately structured (ΔG = {result['mfe_5p_dG']:.2f})")
            else:
                print(f"  • 5' structure: strongly structured (ΔG = {result['mfe_5p_dG']:.2f})")
        else:
            print("  • ViennaRNA not available - install Python RNA package or RNAfold CLI")


if __name__ == "__main__":
    main()

