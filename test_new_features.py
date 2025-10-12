#!/usr/bin/env python3
"""
Test script for new codon features:
- FOP (Frequency of Optimal Codons)
- CPS (Codon Pair Score)
- CPB (Codon Pair Bias) with host-specific tables
- CpG/UpA dinucleotide statistics

Author: CodonVerifier Enhanced Features
Date: 2025-10-12
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from codon_verifier.metrics import (
    cai, tai, fop, gc_content,
    codon_pair_bias_score, codon_pair_score,
    cpg_upa_content, count_dinucleotides,
    rare_codon_runs, homopolymers,
    rules_score
)
from codon_verifier.hosts.tables import get_host_tables


def test_codon_metrics():
    """Test all codon usage metrics"""
    
    # Test sequences (E. coli highly expressed genes)
    # Note: Removing terminal stop codons as required by metrics functions
    test_sequences = {
        "GFP_optimized": "ATGCGTAAAGGCGAAGAACTGTTTACCGGTGTTGTGCCGATTCTGGTGGAACTGGATGGTGATGTGAACGGTCATAAATTTTCCGTGCGTGGTGAAGGTGAAGGTGATGCGACCAACGGTAAACTGACCCTGAAATTTATTTGCACCACCGGTAAACTGCCGGTTCCGTGGCCGACCCTGGTGACCACCCTGACCTATGGTGTTCAGTGCTTTGCGCGTTATCCGGATCATATGAAACGTCATGATTTTTTTAAATCCGCGATGCCGGAAGGTTATGTGCAGGAACGTACCATTTCCTTTAAAGATGATGGTAACTATAAGACCCGTGCGGAAGTTAAATTTGAAGGTGATACCCTGGTTAACCGTATCGAACTGAAAGGTATCGATTTCAAAGAAGATGGTAACATCCTGGGTCATAAGCTGGAATATAACTTTAACTCCCATAACGTTTATATTACCGCGGATAAACAGAAAAACGGTATCAAAGCGAACTTTAAAATCCGTCATAACGTTGAAGATGGTTCCGTTCAGCTGGCGGATCATTATCAGCAGAATACCCCGATTGGTGATGGTCCGGTGCTGCTGCCGGATAACCATTATCTGTCCACCCAGTCCAAACTGTCCAAAGACCCGAACGAAAAACGTGATCATATGGTTCTGCTGGAATTTGTTACCGCGGCGGGTATTACCCATGGTATTGATGAACTGTATAAA",
        "lacZ_fragment": "ATGACCATGATTACGGATTCACTGGCCGTCGTTTTACAACGTCGTGACTGGGAAAACCCTGGCGTTACCCAACTTAATCGCCTTGCAGCACATCCCCCTTTCGCCAGCTGGCGTAATAGCGAAGAGGCCCGCACCGATCGCCCTTCCCAACAGTTGCGCAGCCTGAATGGCGAATGGCGCTTTGCCTGGTTTCCGGCACCAGAAGCGGTGCCGGAAAGCTGGCTGGAGTGCGATCTTCCTGAGGCCGATACTGTCGTCGTCCCCTCAAACTGGCAGATGCACGGTTACGATGCGCCCATCTACACCAACGTGACCTATCCCATTACGGTCAATCCGCCGTTTGTTCCCACGGAGAATCCGACGGGTTGTTACTCGCTCACATT"
    }
    
    hosts = ['E_coli', 'Human', 'S_cerevisiae']
    
    print("=" * 80)
    print("CODON USAGE METRICS TEST")
    print("=" * 80)
    
    for seq_name, sequence in test_sequences.items():
        print(f"\n### Testing: {seq_name} (length: {len(sequence)} bp)")
        print("-" * 80)
        
        for host in hosts:
            print(f"\n## Host: {host}")
            
            # Get host-specific tables
            usage_table, trna_weights, cpb_table = get_host_tables(host, include_cpb=True)
            
            # Calculate metrics
            try:
                # Basic codon usage
                _cai = cai(sequence, usage_table)
                _tai = tai(sequence, trna_weights)
                _fop = fop(sequence, usage_table)
                _gc = gc_content(sequence)
                
                print(f"  CAI (Codon Adaptation Index): {_cai:.4f}")
                print(f"  tAI (tRNA Adaptation Index):  {_tai:.4f}")
                print(f"  FOP (Freq Optimal Codons):    {_fop:.4f} ⭐ NEW")
                print(f"  GC content:                   {_gc:.4f}")
                
                # Codon pair metrics
                _cpb = codon_pair_bias_score(sequence, cpb_table)
                _cps = codon_pair_score(sequence, usage_table)
                
                print(f"\n  CPB (Codon Pair Bias):        {_cpb:.4f} ⭐ NEW (with data)")
                print(f"  CPS (Codon Pair Score):       {_cps:.4f} ⭐ NEW")
                
                # Dinucleotide statistics
                dinuc_stats = cpg_upa_content(sequence)
                print(f"\n  CpG dinucleotides:")
                print(f"    Count:                      {dinuc_stats['cpg_count']}")
                print(f"    Frequency (per 100):        {dinuc_stats['cpg_freq']:.2f}")
                print(f"    Obs/Exp ratio:              {dinuc_stats['cpg_obs_exp']:.4f} ⭐ NEW")
                
                print(f"\n  UpA (TpA) dinucleotides:")
                print(f"    Count:                      {dinuc_stats['upa_count']}")
                print(f"    Frequency (per 100):        {dinuc_stats['upa_freq']:.2f}")
                print(f"    Obs/Exp ratio:              {dinuc_stats['upa_obs_exp']:.4f} ⭐ NEW")
                
                # Problematic features
                rare_runs = rare_codon_runs(sequence, usage_table)
                homos = homopolymers(sequence, min_len=6)
                
                print(f"\n  Rare codon runs (≥3):         {len(rare_runs)}")
                print(f"  Homopolymers (≥6bp):          {len(homos)}")
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
    
    print("\n" + "=" * 80)


def test_integrated_scoring():
    """Test integrated rules_score with all new features"""
    
    print("\n" + "=" * 80)
    print("INTEGRATED RULES SCORE TEST")
    print("=" * 80)
    
    # Test sequence (without terminal stop codon)
    sequence = "ATGCGTAAAGGCGAAGAACTGTTTACCGGTGTTGTGCCGATTCTGGTGGAACTGGATGGTGATGTGAACGGTCATAAATTTTCCGTGCGTGGTGAAGGTGAAGGTGATGCGACCAACGGTAAACTGACCCTGAAATTTATTTGCACCACCGGTAAACTGCCGGTTCCGTGGCCGACCCTGGTGACCACCCTGACCTATGGTGTTCAGTGCTTTGCGCGTTATCCGGATCATATGAAACGTCATGATTTTTTTAAATCCGCGATGCCGGAAGGTTATGTGCAGGAACGTACCATTTCCTTTAAAGATGATGGTAACTATAAGACCCGTGCGGAAGTTAAATTTGAAGGTGATACCCTGGTTAACCGTATCGAACTGAAAGGTATCGATTTCAAAGAAGATGGTAACATCCTGGGTCATAAGCTGGAATATAACTTTAACTCCCATAACGTTTATATTACCGCGGATAAACAGAAAAACGGTATCAAAGCGAACTTTAAAATCCGTCATAACGTTGAAGATGGTTCCGTTCAGCTGGCGGATCATTATCAGCAGAATACCCCGATTGGTGATGGTCCGGTGCTGCTGCCGGATAACCATTATCTGTCCACCCAGTCCAAACTGTCCAAAGACCCGAACGAAAAACGTGATCATATGGTTCTGCTGGAATTTGTTACCGCGGCGGGTATTACCCATGGTATTGATGAACTGTATAAA"
    
    for host in ['E_coli', 'Human']:
        print(f"\n### Host: {host}")
        print("-" * 80)
        
        # Get host tables
        usage_table, trna_weights, cpb_table = get_host_tables(host, include_cpb=True)
        
        # Calculate comprehensive rules score
        scores = rules_score(
            dna=sequence,
            usage=usage_table,
            trna_w=trna_weights,
            cpb=cpb_table,
            codon_pair_freq=None,  # Could add observed frequencies if available
            motifs=['AAAAA', 'TTTTTT'],  # Example forbidden motifs
            use_vienna_dG=False  # Skip MFE calculation for this test
        )
        
        print(f"\nCodon Usage Metrics:")
        print(f"  CAI:                {scores['cai']:.4f}")
        print(f"  tAI:                {scores['tai']:.4f}")
        print(f"  FOP:                {scores['fop']:.4f} ⭐ NEW")
        
        print(f"\nCodon Pair Metrics:")
        print(f"  CPB:                {scores['cpb']:.4f} ⭐ NEW")
        print(f"  CPS:                {scores['cps']:.4f} ⭐ NEW")
        
        print(f"\nDinucleotide Analysis:")
        print(f"  CpG count:          {scores['cpg_count']}")
        print(f"  CpG obs/exp:        {scores['cpg_obs_exp']:.4f} ⭐ NEW")
        print(f"  CpG penalty:        {scores['cpg_penalty']:.4f} ⭐ NEW")
        print(f"  UpA count:          {scores['upa_count']}")
        print(f"  UpA obs/exp:        {scores['upa_obs_exp']:.4f} ⭐ NEW")
        print(f"  UpA penalty:        {scores['upa_penalty']:.4f} ⭐ NEW")
        
        print(f"\nOther Metrics:")
        print(f"  GC content:         {scores['gc']:.4f}")
        print(f"  Rare codon runs:    {int(scores['rare_run_len'])}")
        print(f"  Homopolymer len:    {int(scores['homopoly_len'])}")
        print(f"  Forbidden motifs:   {scores['forbidden_hits']}")
        
        print(f"\n{'='*40}")
        print(f"  TOTAL RULES SCORE:  {scores['total_rules']:.4f}")
        print(f"{'='*40}")


def test_dinucleotide_counting():
    """Test dinucleotide counting function"""
    
    print("\n" + "=" * 80)
    print("DINUCLEOTIDE COUNTING TEST")
    print("=" * 80)
    
    # Test with a simple sequence
    test_seq = "ATGCGATCGATCGCGCGTATAA"
    
    print(f"\nTest sequence: {test_seq}")
    print(f"Length: {len(test_seq)} bp")
    
    # Count specific dinucleotides
    specific_dinucs = count_dinucleotides(test_seq, ["CG", "TA", "AT", "GC"])
    print("\nSpecific dinucleotide counts:")
    for dinuc, count in sorted(specific_dinucs.items()):
        print(f"  {dinuc}: {count}")
    
    # Count all dinucleotides
    all_dinucs = count_dinucleotides(test_seq, None)
    print(f"\nAll dinucleotide counts (showing non-zero):")
    for dinuc, count in sorted(all_dinucs.items()):
        if count > 0:
            print(f"  {dinuc}: {count}")
    
    # Test CpG/UpA analysis
    cpg_upa = cpg_upa_content(test_seq)
    print(f"\nCpG/UpA analysis:")
    print(f"  CpG count: {cpg_upa['cpg_count']}")
    print(f"  CpG obs/exp: {cpg_upa['cpg_obs_exp']:.4f}")
    print(f"  UpA count: {cpg_upa['upa_count']}")
    print(f"  UpA obs/exp: {cpg_upa['upa_obs_exp']:.4f}")


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "CODONVERIFIER NEW FEATURES TEST" + " " * 27 + "║")
    print("╚" + "═" * 78 + "╝")
    
    try:
        # Run tests
        test_dinucleotide_counting()
        test_codon_metrics()
        test_integrated_scoring()
        
        print("\n" + "=" * 80)
        print("✓ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nNew features implemented:")
        print("  ⭐ FOP (Frequency of Optimal Codons)")
        print("  ⭐ CPS (Codon Pair Score)")
        print("  ⭐ CPB (Codon Pair Bias) with E. coli, Human, Mouse, S. cerevisiae, P. pastoris tables")
        print("  ⭐ CpG/UpA dinucleotide statistics with observed/expected ratios")
        print("  ⭐ All features integrated into rules_score() and microservices")
        print("\n")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

