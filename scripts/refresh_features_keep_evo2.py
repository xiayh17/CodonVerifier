#!/usr/bin/env python3
"""
Refresh Features While Preserving Evo2

This script updates existing enhanced JSONL files by recomputing fast features
(codon, MSA, structure, Δ) while preserving expensive Evo2 model features.

Features updated:
- codon_* (CAI, TAI, FOP, GC, CPB, etc.)
- evo_* (MSA features, if --real-msa-jsonl provided)
- struct_* (structure features)
- evo2_delta_* (Δbpb, ΔNLL based on AA+host grouping)

Features preserved:
- evo2_bpb (from Evo2 model)
- evo2_nll (from Evo2 model)
- expression (original expression value)

Usage:
    # Basic: update codon + structure + Δ, preserve Evo2
    python scripts/refresh_features_keep_evo2.py \\
        --input data/enhanced/Pic_complete_v2.jsonl \\
        --output data/enhanced/Pic_complete_v3.jsonl

    # With real MSA features
    python scripts/refresh_features_keep_evo2.py \\
        --input data/enhanced/Pic_complete_v2.jsonl \\
        --output data/enhanced/Pic_complete_v3.jsonl \\
        --real-msa-jsonl data/real_msa/Pic_msa.jsonl

    # In-place update (overwrites input)
    python scripts/refresh_features_keep_evo2.py \\
        --input data/enhanced/Pic_complete_v2.jsonl \\
        --output data/enhanced/Pic_complete_v2.jsonl \\
        --real-msa-jsonl data/real_msa/Pic_msa.jsonl

Author: CodonVerifier Team
Date: 2025-10-12
"""

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

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
    from codon_verifier.codon_utils import aa_from_dna
except Exception:
    aa_from_dna = None


def load_real_msa_map(real_msa_jsonl: str) -> Dict[str, Dict[str, float]]:
    """Load real MSA features from JSONL, keyed by protein_id"""
    msa_map: Dict[str, Dict[str, float]] = {}
    with open(real_msa_jsonl, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                protein_id = rec.get('protein_id')
                msa_features = rec.get('msa_features')
                if not protein_id:
                    logger.warning(f"Line {line_num}: missing protein_id, skipping")
                    continue
                if not isinstance(msa_features, dict):
                    logger.warning(f"Line {line_num}: msa_features not a dict, skipping")
                    continue
                msa_map[protein_id] = msa_features
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: JSON decode error: {e}")
    logger.info(f"Loaded MSA features for {len(msa_map)} proteins from {real_msa_jsonl}")
    return msa_map


def load_evo2_features(evo2_json: str) -> List[Dict[str, float]]:
    """
    Load Evo2 features from JSON array and convert to list indexed by record position.
    
    Evo2 JSON format:
    [
      {
        "task": "extract_features",
        "status": "success",
        "output": {
          "sequence": "ATG...",
          "sequence_length": 141,
          "loglik": -175.83,
          "avg_loglik": -1.256,
          "perplexity": 3.511,
          "geom": 0.285,
          ...
        },
        "metadata": {"request_id": "record_6", ...}
      },
      ...
    ]
    
    Returns:
        List of dicts with keys: evo2_perplexity, evo2_avg_loglik, evo2_bpb, evo2_nll
        indexed by record position
    """
    import math
    
    evo2_list = []
    with open(evo2_json, 'r') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        logger.error(f"Expected JSON array in {evo2_json}, got {type(data)}")
        return []
    
    for idx, entry in enumerate(data):
        if entry.get('status') != 'success':
            logger.warning(f"Record {idx}: Evo2 status = {entry.get('status')}, skipping")
            evo2_list.append({})
            continue
        
        output = entry.get('output', {})
        perplexity = output.get('perplexity')
        avg_loglik = output.get('avg_loglik')
        loglik = output.get('loglik')
        geom = output.get('geom')
        
        features = {}
        # Preserve raw Evo2 scores with evo2_ prefix
        if perplexity is not None:
            features['evo2_perplexity'] = float(perplexity)
            # Also compute bpb = log2(perplexity)
            features['evo2_bpb'] = math.log2(float(perplexity))
        if avg_loglik is not None:
            features['evo2_avg_loglik'] = float(avg_loglik)
            # evo2_nll is the same as avg_loglik
            features['evo2_nll'] = float(avg_loglik)
        if loglik is not None:
            features['evo2_loglik'] = float(loglik)
        if geom is not None:
            features['evo2_geom'] = float(geom)
        
        evo2_list.append(features)
    
    logger.info(f"Loaded Evo2 features for {len(evo2_list)} records from {evo2_json}")
    return evo2_list


def extract_codon_features(sequence: str, host: str = 'E_coli') -> Dict[str, float]:
    """Extract codon usage features"""
    features = {}
    try:
        from codon_verifier.metrics import (
            cai, tai, fop, gc_content, codon_pair_bias_score,
            codon_pair_score, cpg_upa_content, rare_codon_runs,
            homopolymers
        )
        from codon_verifier.hosts.tables import get_host_tables
        from codon_verifier.codon_utils import validate_cds
        
        usage_table, trna_weights, cpb_table = get_host_tables(host, include_cpb=True)
        
        features['codon_gc'] = gc_content(sequence)
        
        valid, msg = validate_cds(sequence)
        if valid:
            features['codon_cai'] = cai(sequence, usage_table)
            features['codon_tai'] = tai(sequence, trna_weights)
            features['codon_fop'] = fop(sequence, usage_table)
        
        try:
            features['codon_cpb'] = codon_pair_bias_score(sequence, cpb_table)
        except Exception:
            pass
        
        try:
            features['codon_cps'] = codon_pair_score(sequence, usage_table)
        except Exception:
            pass
        
        features['codon_cpg_upa'] = cpg_upa_content(sequence)
        features['codon_rare_runs'] = rare_codon_runs(sequence, usage_table)
        features['codon_homopolymer'] = homopolymers(sequence)
        
    except Exception as e:
        logger.warning(f"Error extracting codon features: {e}")
    
    return features


def extract_structure_features(protein_aa: str) -> Dict[str, float]:
    """Extract structure features using lite approximation"""
    features = {}
    try:
        # Import structure lite logic
        from services.structure_features_lite.app import StructureFeaturesLite
        from dataclasses import asdict
        
        lite = StructureFeaturesLite(use_afdb=True)  # Disable AFDB for speed
        struct_feat = lite.predict_structure(aa_sequence=protein_aa)
        
        # Flatten structure features with struct_ prefix
        for key, val in asdict(struct_feat).items():
            if isinstance(val, (int, float)):
                features[f'struct_{key}'] = float(val)
        
    except Exception as e:
        logger.warning(f"Error extracting structure features: {e}")
    
    return features


def extract_msa_features_lite(protein_id: str, protein_aa: str) -> Dict[str, float]:
    """Extract MSA features using lite approximation (fallback)"""
    features = {}
    try:
        from services.msa_features_lite.app import MSAFeaturesLite
        from dataclasses import asdict
        
        lite = MSAFeaturesLite()
        msa_feat = lite.predict_evolutionary_features(protein_aa, protein_id)
        
        # Flatten MSA features with evo_ prefix
        for key, val in asdict(msa_feat).items():
            if isinstance(val, (int, float)):
                features[f'evo_{key}'] = float(val)
        
    except Exception as e:
        logger.warning(f"Error extracting MSA features: {e}")
    
    return features


def compute_group_statistics(records: List[Dict]) -> Dict:
    """
    Groups by (amino_acid_sequence, host) and computes median bpb/nll
    
    Returns:
        Dict mapping (aa_seq, host) -> {'median_bpb': float, 'median_nll': float}
    """
    group_bpbs = defaultdict(list)
    group_nlls = defaultdict(list)
    
    for rec in records:
        sequence = rec.get('sequence', '')
        host = rec.get('host', 'E_coli')
        evo2_bpb = rec.get('evo2_bpb')
        evo2_nll = rec.get('evo2_nll')
        
        if not sequence or aa_from_dna is None:
            continue
        
        try:
            aa_seq = aa_from_dna(sequence)
        except Exception:
            continue
        
        group_key = (aa_seq, host)
        
        if evo2_bpb is not None and isinstance(evo2_bpb, (int, float)):
            group_bpbs[group_key].append(float(evo2_bpb))
        if evo2_nll is not None and isinstance(evo2_nll, (int, float)):
            group_nlls[group_key].append(float(evo2_nll))
    
    # Compute medians
    group_stats = {}
    for group_key in set(group_bpbs.keys()) | set(group_nlls.keys()):
        bpbs = sorted(group_bpbs.get(group_key, []))
        nlls = sorted(group_nlls.get(group_key, []))
        
        median_bpb = None
        if bpbs:
            n = len(bpbs)
            median_bpb = bpbs[n // 2] if n % 2 == 1 else (bpbs[n // 2 - 1] + bpbs[n // 2]) / 2
        
        median_nll = None
        if nlls:
            n = len(nlls)
            median_nll = nlls[n // 2] if n % 2 == 1 else (nlls[n // 2 - 1] + nlls[n // 2]) / 2
        
        group_stats[group_key] = {
            'median_bpb': median_bpb,
            'median_nll': median_nll
        }
    
    logger.info(f"Computed group statistics for {len(group_stats)} (AA, host) groups")
    return group_stats


def refresh_record(
    record: Dict,
    group_stats: Dict,
    real_msa_map: Optional[Dict[str, Dict[str, float]]] = None,
    evo2_features: Optional[Dict[str, float]] = None
) -> Dict:
    """
    Refresh a single record:
    - Preserve Evo2 raw scores (perplexity, avg_loglik, etc.) OR load from evo2_features
    - Compute evo2_bpb from perplexity if missing
    - Recompute codon_*, struct_*, evo_*, evo2_delta_*
    """
    import math
    
    sequence = record.get('sequence', '')
    host = record.get('host', 'E_coli')
    protein_id = record.get('protein_id', '')
    protein_aa = record.get('protein_aa', '')
    expression = record.get('expression')
    
    # Start with base fields
    refreshed = {
        'protein_id': protein_id,
        'sequence': sequence,
        'host': host,
        'protein_aa': protein_aa,
        'expression': expression,
    }
    
    # Handle Evo2 features: preserve existing or load from evo2_features
    # Priority: existing record > evo2_features parameter
    evo2_perplexity = record.get('evo2_perplexity')
    evo2_avg_loglik = record.get('evo2_avg_loglik')
    evo2_bpb = record.get('evo2_bpb')
    evo2_nll = record.get('evo2_nll')
    
    # If evo2_features provided (from JSON), use it as fallback
    if evo2_features:
        if evo2_perplexity is None:
            evo2_perplexity = evo2_features.get('evo2_perplexity')
        if evo2_avg_loglik is None:
            evo2_avg_loglik = evo2_features.get('evo2_avg_loglik')
        if evo2_bpb is None:
            evo2_bpb = evo2_features.get('evo2_bpb')
        if evo2_nll is None:
            evo2_nll = evo2_features.get('evo2_nll')
    
    # Compute evo2_bpb from perplexity if missing
    if evo2_bpb is None and evo2_perplexity is not None:
        evo2_bpb = math.log2(float(evo2_perplexity))
    
    # Compute evo2_nll from avg_loglik if missing (they should be the same)
    if evo2_nll is None and evo2_avg_loglik is not None:
        evo2_nll = float(evo2_avg_loglik)
    
    # Store Evo2 raw scores and computed values
    if evo2_perplexity is not None:
        refreshed['evo2_perplexity'] = float(evo2_perplexity)
    if evo2_avg_loglik is not None:
        refreshed['evo2_avg_loglik'] = float(evo2_avg_loglik)
    if evo2_bpb is not None:
        refreshed['evo2_bpb'] = float(evo2_bpb)
    if evo2_nll is not None:
        refreshed['evo2_nll'] = float(evo2_nll)
    
    # 1. Recompute codon features
    codon_features = extract_codon_features(sequence, host)
    refreshed.update(codon_features)
    
    # 2. Recompute structure features
    if protein_aa:
        struct_features = extract_structure_features(protein_aa)
        refreshed.update(struct_features)
    
    # 3. Recompute MSA features
    if real_msa_map and protein_id in real_msa_map:
        # Use real MSA features
        msa_features = real_msa_map[protein_id]
        for key, val in msa_features.items():
            refreshed[f'evo_{key}'] = val
    else:
        # Fallback to lite approximation
        if protein_aa:
            msa_features = extract_msa_features_lite(protein_id, protein_aa)
            refreshed.update(msa_features)
    
    # 4. Recompute Δ features (evo2_delta_bpb, evo2_delta_nll)
    if aa_from_dna and sequence and evo2_bpb is not None:
        try:
            aa_seq = aa_from_dna(sequence)
            group_key = (aa_seq, host)
            stats = group_stats.get(group_key, {})
            median_bpb = stats.get('median_bpb')
            median_nll = stats.get('median_nll')
            
            if median_bpb is not None:
                refreshed['evo2_delta_bpb'] = float(evo2_bpb) - median_bpb
            if median_nll is not None and evo2_nll is not None:
                refreshed['evo2_delta_nll'] = float(evo2_nll) - median_nll
        except Exception as e:
            logger.debug(f"Could not compute Δ for {protein_id}: {e}")
    
    return refreshed


def main():
    parser = argparse.ArgumentParser(
        description='Refresh features while preserving Evo2 model outputs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--input', required=True,
                       help='Input enhanced JSONL file')
    parser.add_argument('--output', required=True,
                       help='Output refreshed JSONL file')
    parser.add_argument('--real-msa-jsonl', type=str,
                       help='Optional: real MSA JSONL to use instead of lite approximation')
    parser.add_argument('--evo2-json', type=str,
                       help='Optional: Evo2 features JSON to load/supplement Evo2 scores')
    parser.add_argument('--limit', type=int,
                       help='Limit number of records (for testing)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    # Load real MSA map if provided
    real_msa_map = None
    if args.real_msa_jsonl:
        real_msa_path = Path(args.real_msa_jsonl)
        if not real_msa_path.exists():
            logger.error(f"Real MSA file not found: {real_msa_path}")
            sys.exit(1)
        real_msa_map = load_real_msa_map(args.real_msa_jsonl)
    
    # Load Evo2 features if provided
    evo2_list = []
    if args.evo2_json:
        evo2_path = Path(args.evo2_json)
        if not evo2_path.exists():
            logger.error(f"Evo2 JSON file not found: {evo2_path}")
            sys.exit(1)
        evo2_list = load_evo2_features(args.evo2_json)
    
    # Load all records
    logger.info(f"Loading records from {input_path}...")
    records = []
    with open(input_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append(rec)
                if args.limit and len(records) >= args.limit:
                    break
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: JSON decode error: {e}")
    
    logger.info(f"Loaded {len(records)} records")
    
    if not records:
        logger.error("No records loaded")
        sys.exit(1)
    
    # Compute group statistics for Δ features
    logger.info("Computing group statistics for Δ features...")
    group_stats = compute_group_statistics(records)
    
    # Refresh each record
    logger.info("Refreshing features...")
    refreshed_records = []
    for i, rec in enumerate(records, 1):
        if i % 1000 == 0:
            logger.info(f"  Processed {i}/{len(records)} records...")
        
        # Get Evo2 features for this record if available
        evo2_features = None
        if evo2_list and i - 1 < len(evo2_list):
            evo2_features = evo2_list[i - 1]
        
        refreshed = refresh_record(rec, group_stats, real_msa_map, evo2_features)
        refreshed_records.append(refreshed)
    
    # Write output
    logger.info(f"Writing {len(refreshed_records)} records to {output_path}...")
    
    # If output == input, write to temp first then move
    if output_path.resolve() == input_path.resolve():
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as tmp:
            tmp_path = Path(tmp.name)
            for rec in refreshed_records:
                tmp.write(json.dumps(rec, ensure_ascii=False) + '\n')
        
        # Move temp to output
        tmp_path.replace(output_path)
        logger.info(f"✓ In-place update completed: {output_path}")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            for rec in refreshed_records:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        logger.info(f"✓ Refresh completed: {output_path}")
    
    logger.info(f"Summary:")
    logger.info(f"  Input: {input_path}")
    logger.info(f"  Output: {output_path}")
    logger.info(f"  Records: {len(refreshed_records)}")
    logger.info(f"  Evo2 JSON: {'Yes' if evo2_list else 'No (preserve existing)'}")
    logger.info(f"  Real MSA: {'Yes' if real_msa_map else 'No (using lite)'}")
    logger.info(f"  Preserved/Loaded: evo2_perplexity, evo2_avg_loglik, expression")
    logger.info(f"  Computed: evo2_bpb (from perplexity), evo2_delta_*")
    logger.info(f"  Refreshed: codon_*, struct_*, evo_*")


if __name__ == '__main__':
    main()

