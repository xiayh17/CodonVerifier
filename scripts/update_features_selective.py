#!/usr/bin/env python3
"""
Selective Feature Update Script

This script updates specific features while preserving:
- MSA features (msa_* from mmseqs)
- Evo2 model direct outputs (evo2_loglik, evo2_avg_loglik, evo2_perplexity, evo2_geom)

Features updated:
- expression (recalculated using enhanced estimator)
- codon_* features (CAI, TAI, FOP, GC, CPB, etc.)
- struct_* features (structure features)
- ctx_* features (context features)
- evo_* features (evolutionary features, but not from MSA)

Features preserved:
- msa_features (from mmseqs)
- evo2_loglik, evo2_avg_loglik, evo2_perplexity, evo2_geom (direct Evo2 outputs)
- evo2_bpb, evo2_delta_*, evo2_ref_* (computed values - not updated)

Usage:
    python scripts/update_features_selective.py \\
        --input data/real_msa/Pic_complete_v2_msa_features.jsonl \\
        --output data/real_msa/Pic_complete_v3_updated.jsonl

Author: CodonVerifier Team
Date: 2025-01-27
"""

import argparse
import json
import logging
import sys
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
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


def extract_codon_features(sequence: str, host: str = 'E_coli') -> Dict[str, float]:
    """
    Extract codon usage features from DNA sequence
    
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


def extract_structure_features(protein_aa: str) -> Dict[str, float]:
    """
    Extract structure features using lite approximation
    
    Args:
        protein_aa: Amino acid sequence
        
    Returns:
        Dictionary of structure features with struct_ prefix
    """
    features = {}
    try:
        from services.structure_features_lite.app import StructureFeaturesLite
        from dataclasses import asdict
        
        lite = StructureFeaturesLite(use_afdb=True)
        struct_feat = lite.predict_structure(aa_sequence=protein_aa)
        
        # Flatten structure features with struct_ prefix
        for key, val in asdict(struct_feat).items():
            if isinstance(val, (int, float)):
                features[f'struct_{key}'] = float(val)
        
    except Exception as e:
        logger.warning(f"Error extracting structure features: {e}")
    
    return features


def extract_context_features(sequence: str, host: str = 'E_coli') -> Dict[str, float]:
    """
    Extract context features (promoter, RBS, etc.)
    
    Args:
        sequence: DNA sequence
        host: Host organism
        
    Returns:
        Dictionary of context features with ctx_ prefix
    """
    features = {}
    try:
        # Default context features - these would normally be computed
        # based on vector design, promoter strength, etc.
        # For now, using placeholder values
        features['ctx_promoter_strength'] = 0.5
        features['ctx_rbs_strength'] = 0.5
        features['ctx_rbs_spacing'] = 8.0
        features['ctx_kozak_score'] = 0.0
        features['ctx_vector_copy_number'] = 1.0
        features['ctx_has_selection_marker'] = 0.0
        features['ctx_temperature_norm'] = 0.0
        features['ctx_inducer_concentration'] = 0.0
        features['ctx_growth_phase_score'] = 1.0
        features['ctx_localization_score'] = 1.0
        
    except Exception as e:
        logger.warning(f"Error extracting context features: {e}")
    
    return features


def extract_evolutionary_features_lite(protein_id: str, protein_aa: str) -> Dict[str, float]:
    """
    Extract evolutionary features using lite approximation (not from MSA)
    
    Args:
        protein_id: Protein identifier
        protein_aa: Amino acid sequence
        
    Returns:
        Dictionary of evolutionary features with evo_ prefix
    """
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
        logger.warning(f"Error extracting evolutionary features: {e}")
    
    return features


def recalculate_expression(
    record: Dict,
    evo2_features: Optional[Dict[str, float]] = None
) -> Dict:
    """
    Recalculate expression value using enhanced estimator
    
    Args:
        record: Record dictionary
        evo2_features: Optional Evo2 features for enhancement
        
    Returns:
        Updated expression dictionary
    """
    try:
        from codon_verifier.expression_estimator import ExpressionEstimator
        
        # Extract metadata
        metadata = record.get("metadata", {})
        extra_features = record.get("extra_features", {})
        
        reviewed = "reviewed" if extra_features.get("reviewed", False) else "unreviewed"
        
        # Handle subcellular_location which might be NaN (float)
        subcellular_raw = metadata.get("subcellular_location", "")
        if isinstance(subcellular_raw, float):
            subcellular = "" if math.isnan(subcellular_raw) else str(subcellular_raw)
        else:
            subcellular = str(subcellular_raw) if subcellular_raw else ""
        
        length = extra_features.get("length", len(record.get("protein_aa", "")))
        organism = metadata.get("organism", "")
        sequence = record.get("sequence", "")
        
        # Prepare model features if available
        model_features = None
        if evo2_features:
            model_features = {
                "avg_confidence": evo2_features.get("evo2_geom"),  # Use geom as confidence proxy
                "avg_loglik": evo2_features.get("evo2_avg_loglik"),
                "perplexity": evo2_features.get("evo2_perplexity")
            }
        
        # Estimate enhanced expression
        estimator = ExpressionEstimator(mode="model_enhanced")
        new_value, new_confidence = estimator.estimate(
            reviewed=reviewed,
            subcellular_location=subcellular,
            protein_length=length,
            organism=organism,
            model_features=model_features,
            sequence=sequence
        )
        
        # Get original value for comparison
        original_expr = record.get("expression", {})
        original_value = original_expr.get("value", 50.0)
        
        return {
            "value": new_value,
            "unit": "estimated_enhanced",
            "assay": "model_enhanced_heuristic",
            "confidence": new_confidence,
            "original_value": original_value
        }
        
    except Exception as e:
        logger.warning(f"Error recalculating expression: {e}")
        # Return original expression if recalculation fails
        return record.get("expression", {"value": 50.0, "unit": "estimated", "assay": "metadata_heuristic", "confidence": "low"})


def update_record_selective(
    record: Dict,
    preserve_evo2_direct: bool = True
) -> Dict:
    """
    Update record with selective feature refresh
    
    Args:
        record: Original record dictionary
        preserve_evo2_direct: Whether to preserve direct Evo2 outputs
        
    Returns:
        Updated record dictionary
    """
    # Start with base fields
    updated = {
        'protein_id': record.get('protein_id', ''),
        'sequence': record.get('sequence', ''),
        'host': record.get('host', 'E_coli'),
        'protein_aa': record.get('protein_aa', ''),
        'metadata': record.get('metadata', {}),
        'msa_features': record.get('msa_features', {})  # Preserve MSA features
    }
    
    # Preserve direct Evo2 outputs if requested
    if preserve_evo2_direct:
        # Check both top level and extra_features for Evo2 features
        evo2_direct = {}
        for key in ['evo2_loglik', 'evo2_avg_loglik', 'evo2_perplexity', 'evo2_geom']:
            if key in record:
                evo2_direct[key] = record[key]
            elif 'extra_features' in record and key in record['extra_features']:
                evo2_direct[key] = record['extra_features'][key]
        updated.update(evo2_direct)
    
    # Also preserve any other evo2_* features that might be in the record
    if preserve_evo2_direct:
        # Check top level
        for key, value in record.items():
            if key.startswith('evo2_') and key not in updated:
                updated[key] = value
        # Check extra_features
        if 'extra_features' in record:
            for key, value in record['extra_features'].items():
                if key.startswith('evo2_') and key not in updated:
                    updated[key] = value
    
    # Extract features
    sequence = record.get('sequence', '')
    host = record.get('host', 'E_coli')
    protein_aa = record.get('protein_aa', '')
    protein_id = record.get('protein_id', '')
    
    # 1. Update codon features
    codon_features = extract_codon_features(sequence, host)
    updated.update(codon_features)
    
    # 2. Update structure features
    if protein_aa:
        struct_features = extract_structure_features(protein_aa)
        updated.update(struct_features)
    
    # 3. Update context features
    ctx_features = extract_context_features(sequence, host)
    updated.update(ctx_features)
    
    # 4. Update evolutionary features (lite, not from MSA)
    if protein_aa:
        evo_features = extract_evolutionary_features_lite(protein_id, protein_aa)
        updated.update(evo_features)
    
    # 5. Recalculate expression
    evo2_features = {
        'evo2_geom': record.get('evo2_geom'),
        'evo2_avg_loglik': record.get('evo2_avg_loglik'),
        'evo2_perplexity': record.get('evo2_perplexity')
    }
    updated['expression'] = recalculate_expression(record, evo2_features)
    
    return updated


def main():
    parser = argparse.ArgumentParser(
        description="Selective feature update while preserving MSA and Evo2 direct outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--input', required=True,
                       help="Input JSONL file with MSA features")
    parser.add_argument('--output', required=True,
                       help="Output JSONL file with updated features")
    parser.add_argument('--limit', type=int,
                       help="Limit number of records (for testing)")
    parser.add_argument('--preserve-evo2-direct', action='store_true', default=True,
                       help="Preserve direct Evo2 outputs (evo2_loglik, evo2_avg_loglik, evo2_perplexity, evo2_geom)")
    
    args = parser.parse_args()
    
    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    logger.info("=" * 80)
    logger.info("SELECTIVE FEATURE UPDATE PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Preserve Evo2 direct outputs: {args.preserve_evo2_direct}")
    if args.limit:
        logger.info(f"Limit: {args.limit} records (testing mode)")
    logger.info("=" * 80)
    
    # Load records
    logger.info("\n[1/3] Loading records...")
    records = []
    with open(input_path, 'r') as f:
        for i, line in enumerate(f):
            if args.limit and i >= args.limit:
                break
            try:
                rec = json.loads(line.strip())
                records.append(rec)
            except Exception as e:
                logger.error(f"Error loading record {i}: {e}")
    
    logger.info(f"✓ Loaded {len(records)} records")
    
    # Update records
    logger.info("\n[2/3] Updating features...")
    updated_records = []
    stats = {
        'total': len(records),
        'updated_expression': 0,
        'updated_codon': 0,
        'updated_structure': 0,
        'updated_context': 0,
        'updated_evolutionary': 0,
        'errors': 0
    }
    
    for i, rec in enumerate(records):
        try:
            updated = update_record_selective(rec, args.preserve_evo2_direct)
            updated_records.append(updated)
            
            # Count updated features
            if 'expression' in updated:
                stats['updated_expression'] += 1
            if any(k.startswith('codon_') for k in updated.keys()):
                stats['updated_codon'] += 1
            if any(k.startswith('struct_') for k in updated.keys()):
                stats['updated_structure'] += 1
            if any(k.startswith('ctx_') for k in updated.keys()):
                stats['updated_context'] += 1
            if any(k.startswith('evo_') for k in updated.keys()):
                stats['updated_evolutionary'] += 1
            
            if (i + 1) % 100 == 0:
                logger.info(f"  Processed {i + 1}/{len(records)} records...")
        
        except Exception as e:
            logger.error(f"Error processing record {i}: {e}")
            stats['errors'] += 1
            updated_records.append(rec)  # Keep original record
    
    logger.info(f"✓ Processed {len(updated_records)} records")
    logger.info(f"  - Updated expression: {stats['updated_expression']}")
    logger.info(f"  - Updated codon features: {stats['updated_codon']}")
    logger.info(f"  - Updated structure features: {stats['updated_structure']}")
    logger.info(f"  - Updated context features: {stats['updated_context']}")
    logger.info(f"  - Updated evolutionary features: {stats['updated_evolutionary']}")
    logger.info(f"  - Errors: {stats['errors']}")
    
    # Write output
    logger.info(f"\n[3/3] Writing output to {args.output}...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for rec in updated_records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    
    logger.info(f"✓ Written {len(updated_records)} records")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SELECTIVE UPDATE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)
    logger.info(f"Output file: {args.output}")
    logger.info(f"Total records: {len(updated_records)}")
    logger.info("Features updated:")
    logger.info("  - expression (recalculated with enhanced estimator)")
    logger.info("  - codon_* (CAI, TAI, FOP, GC, CPB, etc.)")
    logger.info("  - struct_* (structure features)")
    logger.info("  - ctx_* (context features)")
    logger.info("  - evo_* (evolutionary features, lite approximation)")
    logger.info("Features preserved:")
    logger.info("  - msa_features (from mmseqs)")
    if args.preserve_evo2_direct:
        logger.info("  - evo2_loglik, evo2_avg_loglik, evo2_perplexity, evo2_geom")
    logger.info("=" * 80 + "\n")


if __name__ == '__main__':
    main()
