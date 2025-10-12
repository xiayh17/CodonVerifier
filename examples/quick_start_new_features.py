#!/usr/bin/env python3
"""
Quick Start Guide for CodonVerifier v2.0.0 New Features

This script demonstrates how to use the new features:
- FOP (Frequency of Optimal Codons)
- CPS (Codon Pair Score)
- CPB (Codon Pair Bias) with host-specific tables
- CpG/UpA dinucleotide statistics

Author: CodonVerifier Team
Date: 2025-10-12
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def example_1_basic_metrics():
    """Example 1: Calculate individual metrics"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Codon Metrics")
    print("="*80)
    
    from codon_verifier.metrics import cai, tai, fop, gc_content
    from codon_verifier.hosts.tables import get_host_tables
    
    # Example: GFP sequence (optimized for E. coli)
    sequence = """
    ATGCGTAAAGGCGAAGAACTGTTTACCGGTGTTGTGCCGATTCTGGTGGAACTGGATGGT
    GATGTGAACGGTCATAAATTTTCCGTGCGTGGTGAAGGTGAAGGTGATGCGACCAACGGT
    AAACTGACCCTGAAATTTATTTGCACCACCGGTAAACTGCCGGTTCCGTGGCCGACCCTG
    """.replace("\n", "").replace(" ", "")
    
    print(f"\nSequence length: {len(sequence)} bp")
    
    # Get E. coli tables (now includes CPB!)
    usage, trna, cpb = get_host_tables('E_coli', include_cpb=True)
    
    # Calculate metrics
    print("\n📊 Codon Usage Metrics:")
    print(f"  CAI (Codon Adaptation Index):     {cai(sequence, usage):.4f}")
    print(f"  tAI (tRNA Adaptation Index):      {tai(sequence, trna):.4f}")
    print(f"  FOP (Freq Optimal Codons):        {fop(sequence, usage):.4f}  ⭐ NEW!")
    print(f"  GC Content:                       {gc_content(sequence):.4f}")
    
    print("\n💡 Interpretation:")
    fop_score = fop(sequence, usage)
    if fop_score > 0.8:
        print("  ✅ High FOP: Excellent use of optimal codons")
    elif fop_score > 0.6:
        print("  ⚠️  Medium FOP: Moderate optimization")
    else:
        print("  ❌ Low FOP: Poor codon optimization")


def example_2_codon_pairs():
    """Example 2: Codon pair analysis"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Codon Pair Metrics")
    print("="*80)
    
    from codon_verifier.metrics import codon_pair_bias_score, codon_pair_score
    from codon_verifier.hosts.tables import get_host_tables
    
    # Two test sequences
    sequences = {
        "Optimized": "ATGCGTAAAGGCGAAGAAGTTCTGAAAGGCGAAATCATTATGGGTAAATAA",
        "Non-optimized": "ATGCGAAGACGGCGACGAAGAAGACTACGACGACGGAGAAGATCGATAA"
    }
    
    usage, trna, cpb = get_host_tables('E_coli', include_cpb=True)
    
    print("\n🔗 Comparing Codon Pair Scores:\n")
    
    for name, seq in sequences.items():
        cpb_score = codon_pair_bias_score(seq, cpb)
        cps_score = codon_pair_score(seq, usage)
        
        print(f"{name}:")
        print(f"  CPB (with E. coli data):  {cpb_score:+.4f}  ⭐ NEW DATA!")
        print(f"  CPS (calculated):         {cps_score:+.4f}  ⭐ NEW!")
        
        if cpb_score > 0:
            print(f"  ✅ Positive CPB: Favored codon pairs")
        elif cpb_score < -0.1:
            print(f"  ❌ Negative CPB: Disfavored codon pairs")
        else:
            print(f"  ➖ Neutral CPB: No strong bias")
        print()


def example_3_dinucleotides():
    """Example 3: CpG and UpA analysis"""
    print("\n" + "="*80)
    print("EXAMPLE 3: CpG/UpA Dinucleotide Analysis")
    print("="*80)
    
    from codon_verifier.metrics import cpg_upa_content, count_dinucleotides
    
    # Test sequences with different CpG/UpA content
    sequences = {
        "High CpG": "ATGCGCGCGCGTCGACGCGCGCGCGTAA",
        "Low CpG":  "ATGAAAGAAAAAGATGATGATGATTAA",
        "High UpA": "ATGTATAATATACTACTATATTAA"
    }
    
    print("\n🧬 Dinucleotide Content Analysis:\n")
    
    for name, seq in sequences.items():
        print(f"### {name}")
        print(f"Sequence: {seq}")
        
        # Count specific dinucleotides
        counts = count_dinucleotides(seq, ["CG", "TA", "AT", "GC"])
        print(f"\nDinucleotide counts:")
        for dinuc, count in counts.items():
            print(f"  {dinuc}: {count}")
        
        # Detailed CpG/UpA analysis
        stats = cpg_upa_content(seq)
        print(f"\n📈 CpG Analysis:  ⭐ NEW!")
        print(f"  Count:              {stats['cpg_count']}")
        print(f"  Frequency:          {stats['cpg_freq']:.2f} per 100")
        print(f"  Observed/Expected:  {stats['cpg_obs_exp']:.4f}")
        
        print(f"\n📉 UpA (TpA) Analysis:  ⭐ NEW!")
        print(f"  Count:              {stats['upa_count']}")
        print(f"  Frequency:          {stats['upa_freq']:.2f} per 100")
        print(f"  Observed/Expected:  {stats['upa_obs_exp']:.4f}")
        
        # Interpretation
        print(f"\n💡 Interpretation:")
        if stats['cpg_obs_exp'] > 1.5:
            print(f"  ⚠️  High CpG content (obs/exp = {stats['cpg_obs_exp']:.2f})")
            print(f"     May cause issues in prokaryotic expression")
        elif stats['cpg_obs_exp'] < 0.5:
            print(f"  ✅ Low CpG content (obs/exp = {stats['cpg_obs_exp']:.2f})")
            print(f"     Good for prokaryotic expression")
        
        if stats['upa_obs_exp'] > 1.5:
            print(f"  ⚠️  High UpA content (obs/exp = {stats['upa_obs_exp']:.2f})")
            print(f"     May affect mRNA stability")
        
        print("\n" + "-"*80 + "\n")


def example_4_multi_host():
    """Example 4: Compare metrics across hosts"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Multi-Host Comparison")
    print("="*80)
    
    from codon_verifier.metrics import fop, codon_pair_bias_score
    from codon_verifier.hosts.tables import get_host_tables
    
    sequence = "ATGCGTAAAGGCGAAGAACTGTTTACCGGTGTTGTGCCGATTCTGGTGGAACTGGATGGTGATGTGAACGGTCATAAATTTTCCGTGCGTGGTGAAGGTGAAGGTGATGCGACCAACGGTAAACTGACCCTGAAATTTATTTGCACCACCGGTAAACTGCCGGTTCCGTGGCCGACCCTGGTGACCACCCTGACCTATGGTGTTCAGTGCTTTGCGCGTTATCCGGATCATATGAAACGTCATGATTTTTTTAAATCCGCGATGCCGGAAGGTTATGTGCAGGAACGTACCATTTCCTTTAAAGATGATGGTAACTATAAGACCCGTGCGGAAGTTAAATTTGAAGGTGATACCCTGGTTAACCGTATCGAACTGAAAGGTATCGATTTCAAAGAAGATGGTAACATCCTGGGTCATAAGCTGGAATATAACTTTAACTCCCATAACGTTTATATTACCGCGGATAAACAGAAAAACGGTATCAAAGCGAACTTTAAAATCCGTCATAACGTTGAAGATGGTTCCGTTCAGCTGGCGGATCATTATCAGCAGAATACCCCGATTGGTGATGGTCCGGTGCTGCTGCCGGATAACCATTATCTGTCCACCCAGTCCAAACTGTCCAAAGACCCGAACGAAAAACGTGATCATATGGTTCTGCTGGAATTTGTTACCGCGGCGGGTATTACCCATGGTATTGATGAACTGTATAAATAA"
    
    hosts = ['E_coli', 'Human', 'S_cerevisiae']
    
    print(f"\n🌍 Same sequence evaluated for different hosts:\n")
    print(f"Sequence: {sequence[:50]}... (length: {len(sequence)} bp)\n")
    print(f"{'Host':<20} {'FOP':>8} {'CPB':>8}  Interpretation")
    print("-" * 70)
    
    for host in hosts:
        usage, trna, cpb = get_host_tables(host, include_cpb=True)
        
        fop_score = fop(sequence, usage)
        cpb_score = codon_pair_bias_score(sequence, cpb)
        
        # Interpretation
        if fop_score > 0.7 and cpb_score > 0:
            interp = "✅ Well optimized"
        elif fop_score > 0.6:
            interp = "⚠️  Moderately optimized"
        else:
            interp = "❌ Poorly optimized"
        
        print(f"{host:<20} {fop_score:>8.4f} {cpb_score:>+8.4f}  {interp}")
    
    print("\n💡 Key Insight:")
    print("  Same sequence can have different scores for different hosts!")
    print("  Always optimize for your target expression system.")


def example_5_complete_workflow():
    """Example 5: Complete evaluation workflow"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Complete Sequence Evaluation Workflow")
    print("="*80)
    
    from codon_verifier.metrics import rules_score
    from codon_verifier.hosts.tables import get_host_tables
    
    sequence = "ATGCGTAAAGGCGAAGAACTGTTTACCGGTGTTGTGCCGATTCTGGTGGAACTGGATGGTGATGTGAACGGTCATAAATTTTCCGTGCGTGGTGAAGGTGAAGGTGATGCGACCAACGGTAAACTGACCCTGAAATTTATTTGCACCACCGGTAAACTGCCGGTTCCGTGGCCGACCCTGGTGACCACCCTGACCTATGGTGTTCAGTGCTTTGCGCGTTATCCGGATCATATGAAACGTCATGATTTTTTTAAATCCGCGATGCCGGAAGGTTATGTGCAGGAACGTACCATTTCCTTTAAAGATGATGGTAACTATAAGACCCGTGCGGAAGTTAAATTTGAAGGTGATACCCTGGTTAACCGTATCGAACTGAAAGGTATCGATTTCAAAGAAGATGGTAACATCCTGGGTCATAAGCTGGAATATAACTTTAACTCCCATAACGTTTATATTACCGCGGATAAACAGAAAAACGGTATCAAAGCGAACTTTAAAATCCGTCATAACGTTGAAGATGGTTCCGTTCAGCTGGCGGATCATTATCAGCAGAATACCCCGATTGGTGATGGTCCGGTGCTGCTGCCGGATAACCATTATCTGTCCACCCAGTCCAAACTGTCCAAAGACCCGAACGAAAAACGTGATCATATGGTTCTGCTGGAATTTGTTACCGCGGCGGGTATTACCCATGGTATTGATGAACTGTATAAATAA"
    
    host = 'E_coli'
    usage, trna, cpb = get_host_tables(host, include_cpb=True)
    
    print(f"\n🔬 Comprehensive Sequence Analysis")
    print(f"Host: {host}")
    print(f"Sequence length: {len(sequence)} bp\n")
    
    # Calculate all metrics at once
    scores = rules_score(
        dna=sequence,
        usage=usage,
        trna_w=trna,
        cpb=cpb,
        motifs=['AAAAA', 'TTTTTT'],
        use_vienna_dG=False  # Skip MFE for quick demo
    )
    
    # Display results by category
    print("📊 Codon Usage Metrics:")
    print(f"  CAI:                    {scores['cai']:.4f}")
    print(f"  tAI:                    {scores['tai']:.4f}")
    print(f"  FOP:                    {scores['fop']:.4f}  ⭐")
    print(f"  GC Content:             {scores['gc']:.4f}")
    
    print("\n🔗 Codon Pair Metrics:")
    print(f"  CPB:                    {scores['cpb']:+.4f}  ⭐")
    print(f"  CPS:                    {scores['cps']:+.4f}  ⭐")
    
    print("\n🧬 Dinucleotide Analysis:")
    print(f"  CpG count:              {scores['cpg_count']}")
    print(f"  CpG obs/exp:            {scores['cpg_obs_exp']:.4f}  ⭐")
    print(f"  UpA count:              {scores['upa_count']}")
    print(f"  UpA obs/exp:            {scores['upa_obs_exp']:.4f}  ⭐")
    
    print("\n⚠️  Problematic Features:")
    print(f"  Rare codon runs:        {int(scores['rare_run_len'])}")
    print(f"  Homopolymer length:     {int(scores['homopoly_len'])}")
    print(f"  Forbidden motifs:       {scores['forbidden_hits']}")
    
    print("\n" + "="*70)
    print(f"  🎯 TOTAL RULES SCORE:   {scores['total_rules']:.4f}")
    print("="*70)
    
    # Overall assessment
    print("\n💯 Overall Assessment:")
    
    quality_checks = []
    if scores['cai'] > 0.8:
        quality_checks.append("✅ Excellent CAI")
    if scores['fop'] > 0.7:
        quality_checks.append("✅ High optimal codon usage")
    if scores['cpb'] > 0:
        quality_checks.append("✅ Favorable codon pairs")
    if scores['cpg_obs_exp'] < 1.0:
        quality_checks.append("✅ Low CpG content")
    if scores['upa_obs_exp'] < 1.0:
        quality_checks.append("✅ Low UpA content")
    if scores['rare_run_len'] == 0:
        quality_checks.append("✅ No rare codon runs")
    if scores['homopoly_len'] == 0:
        quality_checks.append("✅ No long homopolymers")
    
    for check in quality_checks:
        print(f"  {check}")
    
    score_percentage = (len(quality_checks) / 7) * 100
    print(f"\n  Quality Score: {len(quality_checks)}/7 checks passed ({score_percentage:.0f}%)")
    
    if score_percentage >= 80:
        print("\n  🌟 Excellent! This sequence is well optimized.")
    elif score_percentage >= 60:
        print("\n  👍 Good! Minor improvements possible.")
    else:
        print("\n  ⚠️  Consider optimization for better expression.")


def main():
    """Run all examples"""
    print("\n")
    print("╔" + "═"*78 + "╗")
    print("║" + " "*15 + "CodonVerifier v2.0.0 - Quick Start Guide" + " "*23 + "║")
    print("╚" + "═"*78 + "╝")
    
    examples = [
        ("Basic Metrics", example_1_basic_metrics),
        ("Codon Pairs", example_2_codon_pairs),
        ("Dinucleotides", example_3_dinucleotides),
        ("Multi-Host", example_4_multi_host),
        ("Complete Workflow", example_5_complete_workflow),
    ]
    
    for i, (name, func) in enumerate(examples, 1):
        try:
            func()
        except Exception as e:
            print(f"\n❌ Error in Example {i}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("✨ Quick Start Complete!")
    print("="*80)
    print("\n📚 For more details, see:")
    print("  - docs/NEW_FEATURES.md")
    print("  - CHANGELOG_v2.0.0.md")
    print("  - test_new_features.py")
    print("\n")


if __name__ == '__main__':
    main()

