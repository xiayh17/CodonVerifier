#!/usr/bin/env python3
"""
Update Codon Features Without Re-running Evo2

This script updates existing JSONL files with new codon features while
preserving expensive Evo2 features. It also computes Δbpb/ΔNLL based on
group statistics.

Usage:
    # Update features from existing file
    python scripts/update_codon_features.py \\
        --input data/enhanced/Pic_test.jsonl \\
        --output data/enhanced/Pic_test_v2.jsonl

    # With custom limit for testing
    python scripts/update_codon_features.py \\
        --input data/enhanced/Pic_test.jsonl \\
        --output data/enhanced/Pic_test_v2.jsonl \\
        --limit 100

Author: CodonVerifier Team
Date: 2025-10-12
"""

import argparse
import json
import logging
import sys
import math
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
import hashlib

# Add project root to path
project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
try:
    # For AA translation to support grouping by (AA sequence, host)
    from codon_verifier.codon_utils import aa_from_dna
except Exception:
    aa_from_dna = None  # Will detect later and skip grouping if missing


def extract_codon_features(sequence: str, host: str = 'E_coli') -> Dict[str, float]:
    """
    Extract all codon usage features from DNA sequence
    
    Args:
        sequence: DNA sequence
        host: Host organism for codon usage tables
        
    Returns:
        Dictionary of codon features with codon_ prefix
    """
    features = {}
    
    try:
        from codon_verifier.metrics import (
            cai, tai, fop, gc_content, codon_pair_bias_score,
            codon_pair_score, cpg_upa_content, rare_codon_runs,
            homopolymers
        )
        from codon_verifier.hosts.tables import get_host_tables
        from codon_verifier.codon_utils import validate_cds
        
        # Get host tables
        usage_table, trna_weights, cpb_table = get_host_tables(host, include_cpb=True)
        
        # Always try to calculate GC content (no validation needed)
        features['codon_gc'] = gc_content(sequence)
        
        # Validate sequence for codon-specific features
        valid, msg = validate_cds(sequence)
        if not valid:
            logger.debug(f"Invalid CDS ({msg}), computing partial features only")
            # For invalid CDS, still compute what we can
            # Skip codon usage metrics that require valid CDS
        else:
            # Calculate codon usage metrics (requires valid CDS)
            features['codon_cai'] = cai(sequence, usage_table)
            features['codon_tai'] = tai(sequence, trna_weights)
            features['codon_fop'] = fop(sequence, usage_table)
        
        # Codon pair metrics (wrap in try-except as they may validate CDS)
        try:
            features['codon_cpb'] = codon_pair_bias_score(sequence, cpb_table)
        except Exception as e:
            logger.debug(f"Could not compute CPB: {e}")
        
        try:
            features['codon_cps'] = codon_pair_score(sequence, usage_table)
        except Exception as e:
            logger.debug(f"Could not compute CPS: {e}")
        
        # Dinucleotide analysis (doesn't require valid CDS)
        dinuc_stats = cpg_upa_content(sequence)
        features['codon_cpg_count'] = float(dinuc_stats['cpg_count'])
        features['codon_cpg_freq'] = dinuc_stats['cpg_freq']
        features['codon_cpg_obs_exp'] = dinuc_stats['cpg_obs_exp']
        features['codon_upa_count'] = float(dinuc_stats['upa_count'])
        features['codon_upa_freq'] = dinuc_stats['upa_freq']
        features['codon_upa_obs_exp'] = dinuc_stats['upa_obs_exp']
        
        # Rare codon runs (wrap in try-except)
        try:
            rare_runs = rare_codon_runs(sequence, usage_table)
            features['codon_rare_runs'] = float(len(rare_runs))
            features['codon_rare_run_total_len'] = float(sum(length for _, length in rare_runs))
        except Exception as e:
            logger.debug(f"Could not compute rare codon runs: {e}")
        
        # Homopolymers (doesn't require valid CDS)
        homos = homopolymers(sequence, min_len=6)
        features['codon_homopolymers'] = float(len(homos))
        features['codon_homopoly_total_len'] = float(sum(length for _, _, length in homos))
        
    except Exception as e:
        logger.warning(f"Error extracting codon features: {e}")
        
    return features


def compute_bpb_from_existing(evo2_features: Dict) -> Optional[float]:
    """
    Compute bpb (bits per base) from existing Evo2 features
    
    Priority:
    1. evo2_bpb (if exists)
    2. evo2_perplexity -> bpb = log2(perplexity)
    3. evo2_avg_loglik (nats) -> bpb = -avg_loglik / ln(2)
    
    Args:
        evo2_features: Dictionary containing Evo2 features
        
    Returns:
        bpb value or None
    """
    if 'evo2_bpb' in evo2_features:
        return float(evo2_features['evo2_bpb'])
    
    if 'evo2_perplexity' in evo2_features:
        ppl = evo2_features['evo2_perplexity']
        if ppl > 0:
            return math.log2(ppl)
    
    if 'evo2_avg_loglik' in evo2_features:
        # Assuming avg_loglik is in nats (natural log)
        # Convert to bits: bpb = -avg_loglik / ln(2)
        avg_ll = evo2_features['evo2_avg_loglik']
        return -avg_ll / math.log(2)
    
    return None


def compute_group_statistics(records: List[Dict]) -> Dict:
    """
    Compute group statistics for Δbpb calculation
    
    Groups by (amino_acid_sequence, host) and computes median bpb
    
    Args:
        records: List of record dictionaries
        
    Returns:
        Dictionary mapping (protein_id, host) -> statistics
    """
    logger.info("Computing group statistics for Δbpb calculation...")
    
    # Collect bpb values by group
    group_bpbs = defaultdict(list)
    
    for rec in records:
        # Strict grouping: require AA sequence and host; no fallback
        sequence = rec.get('sequence', '') or ''
        host = rec.get('host', '') or ''
        if not sequence or not host:
            continue
        if aa_from_dna is None:
            continue
        aa_seq = aa_from_dna(sequence)
        if not aa_seq:
            continue

        extra = rec.get('extra_features', {})
        bpb = compute_bpb_from_existing(extra)

        if bpb is not None:
            group_bpbs[(aa_seq, host)].append(bpb)
    
    # Compute median for each group
    group_stats = {}
    for key, bpbs in group_bpbs.items():
        if bpbs:
            sorted_bpbs = sorted(bpbs)
            n = len(sorted_bpbs)
            if n % 2 == 0:
                median = (sorted_bpbs[n//2 - 1] + sorted_bpbs[n//2]) / 2
            else:
                median = sorted_bpbs[n//2]
            
            group_stats[key] = {
                'median_bpb': median,
                'count': n,
                'min_bpb': min(bpbs),
                'max_bpb': max(bpbs)
            }
    
    logger.info(f"Computed statistics for {len(group_stats)} groups")
    return group_stats


def compute_delta_features(rec: Dict, group_stats: Dict) -> Dict[str, float]:
    """
    Compute Δbpb and ΔNLL from existing Evo2 features
    
    Args:
        rec: Record dictionary
        group_stats: Group statistics from compute_group_statistics
        
    Returns:
        Dictionary with delta features
    """
    features = {}
    
    extra = rec.get('extra_features', {})
    bpb = compute_bpb_from_existing(extra)
    
    if bpb is None:
        return features
    
    # Add bpb if not already present
    if 'evo2_bpb' not in extra:
        features['evo2_bpb'] = bpb
    
    # Compute delta based strictly on (AA sequence, host)
    sequence = rec.get('sequence', '') or ''
    host = rec.get('host', '') or ''
    if not sequence or not host or aa_from_dna is None:
        return features
    aa_seq = aa_from_dna(sequence)
    if not aa_seq:
        return features
    key = (aa_seq, host)

    if key in group_stats:
        ref_bpb = group_stats[key]['median_bpb']
        delta_bpb = bpb - ref_bpb

        features['evo2_delta_bpb'] = delta_bpb
        features['evo2_ref_bpb'] = ref_bpb

        # Compute ΔNLL (in bits)
        seq_len = len(sequence)
        if seq_len > 0:
            delta_nll = delta_bpb * seq_len
            features['evo2_delta_nll'] = delta_nll
    
    return features


def create_stable_key(rec: Dict) -> str:
    """
    Create a stable key for matching records
    
    Uses (protein_id, sequence) or sequence hash
    
    Args:
        rec: Record dictionary
        
    Returns:
        Stable key string
    """
    protein_id = rec.get('protein_id', '')
    sequence = rec.get('sequence', '')
    
    if protein_id and sequence:
        # Use protein_id + sequence hash for uniqueness
        seq_hash = hashlib.sha256(sequence.encode()).hexdigest()[:16]
        return f"{protein_id}_{seq_hash}"
    elif sequence:
        # Fallback to sequence hash only
        return hashlib.sha256(sequence.encode()).hexdigest()
    else:
        # Last resort: use position
        return f"record_{id(rec)}"


def update_features_pipeline(
    input_jsonl: str,
    output_jsonl: str,
    limit: Optional[int] = None
):
    """
    Main pipeline to update features
    
    Args:
        input_jsonl: Input JSONL file with Evo2 features
        output_jsonl: Output JSONL file with updated features
        limit: Optional limit on records
    """
    logger.info("=" * 80)
    logger.info("CODON FEATURE UPDATE PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Input: {input_jsonl}")
    logger.info(f"Output: {output_jsonl}")
    if limit:
        logger.info(f"Limit: {limit} records (testing mode)")
    logger.info("=" * 80)
    
    # Step 1: Load all records
    logger.info("\n[1/4] Loading records...")
    records = []
    with open(input_jsonl, 'r') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            try:
                rec = json.loads(line.strip())
                records.append(rec)
            except Exception as e:
                logger.error(f"Error loading record {i}: {e}")
    
    logger.info(f"✓ Loaded {len(records)} records")
    
    # Step 2: Compute group statistics for Δbpb
    logger.info("\n[2/4] Computing group statistics...")
    group_stats = compute_group_statistics(records)
    logger.info(f"✓ Computed statistics for {len(group_stats)} groups")
    
    # Step 3: Update each record with new features
    logger.info("\n[3/4] Extracting codon features...")
    updated_records = []
    stats = {
        'total': len(records),
        'with_codon_features': 0,
        'with_delta_features': 0,
        'errors': 0
    }
    
    for i, rec in enumerate(records):
        try:
            # Get or create extra_features
            if 'extra_features' not in rec:
                rec['extra_features'] = {}
            
            # Extract new codon features
            sequence = rec.get('sequence', '')
            host = rec.get('host', 'E_coli')
            
            if sequence:
                codon_feats = extract_codon_features(sequence, host)
                if codon_feats:
                    rec['extra_features'].update(codon_feats)
                    stats['with_codon_features'] += 1
            
            # Compute delta features
            delta_feats = compute_delta_features(rec, group_stats)
            if delta_feats:
                rec['extra_features'].update(delta_feats)
                stats['with_delta_features'] += 1
            
            updated_records.append(rec)
            
            if (i + 1) % 100 == 0:
                logger.info(f"  Processed {i + 1}/{len(records)} records...")
        
        except Exception as e:
            logger.error(f"Error processing record {i}: {e}")
            stats['errors'] += 1
            updated_records.append(rec)  # Keep original record
    
    logger.info(f"✓ Processed {len(updated_records)} records")
    logger.info(f"  - With codon features: {stats['with_codon_features']}")
    logger.info(f"  - With delta features: {stats['with_delta_features']}")
    logger.info(f"  - Errors: {stats['errors']}")
    
    # Step 4: Write output
    logger.info(f"\n[4/4] Writing output to {output_jsonl}...")
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_jsonl, 'w') as f:
        for rec in updated_records:
            f.write(json.dumps(rec) + '\n')
    
    logger.info(f"✓ Written {len(updated_records)} records")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("UPDATE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)
    logger.info(f"Output file: {output_jsonl}")
    logger.info(f"Total records: {len(updated_records)}")
    logger.info(f"New codon features added:")
    logger.info(f"  - codon_cai, codon_tai, codon_fop")
    logger.info(f"  - codon_cpb, codon_cps")
    logger.info(f"  - codon_cpg_*, codon_upa_*")
    logger.info(f"  - codon_rare_runs, codon_homopolymers")
    logger.info(f"New delta features added:")
    logger.info(f"  - evo2_bpb, evo2_delta_bpb, evo2_delta_nll")
    logger.info("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Update codon features without re-running Evo2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update features from existing file
  python scripts/update_codon_features.py \\
    --input data/enhanced/Pic_test.jsonl \\
    --output data/enhanced/Pic_test_v2.jsonl

  # With custom limit for testing
  python scripts/update_codon_features.py \\
    --input data/enhanced/Pic_test.jsonl \\
    --output data/enhanced/Pic_test_v2.jsonl \\
    --limit 100

Features added:
  Codon Usage:
    - codon_cai, codon_tai, codon_fop
    - codon_gc
  
  Codon Pairs:
    - codon_cpb (with host-specific data)
    - codon_cps
  
  Dinucleotides:
    - codon_cpg_count, codon_cpg_freq, codon_cpg_obs_exp
    - codon_upa_count, codon_upa_freq, codon_upa_obs_exp
  
  Others:
    - codon_rare_runs, codon_rare_run_total_len
    - codon_homopolymers, codon_homopoly_total_len
  
  Delta Features (from existing Evo2):
    - evo2_bpb (bits per base)
    - evo2_delta_bpb (relative to group median)
    - evo2_ref_bpb (group median reference)
    - evo2_delta_nll (delta negative log-likelihood in bits)
"""
    )
    
    parser.add_argument('--input', required=True,
                       help="Input JSONL file with Evo2 features")
    parser.add_argument('--output', required=True,
                       help="Output JSONL file with updated features")
    parser.add_argument('--limit', type=int,
                       help="Limit number of records (for testing)")
    
    args = parser.parse_args()
    
    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    # Run pipeline
    try:
        update_features_pipeline(
            input_jsonl=str(input_path),
            output_jsonl=args.output,
            limit=args.limit
        )
        
        logger.info("🎉 Feature update completed successfully!")
    
    except Exception as e:
        logger.error(f"❌ Feature update failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

