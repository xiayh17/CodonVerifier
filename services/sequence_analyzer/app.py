#!/usr/bin/env python3
"""
Sequence Analyzer Service - DNA Sequence Verification and Analysis
Verifies and analyzes DNA sequences for quality metrics
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('sequence-analyzer-service')


def load_verifier():
    """Initialize Codon Verifier components"""
    try:
        import codon_verifier
        logger.info("✓ Codon Verifier loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load Codon Verifier: {e}")
        raise


def process_task(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single verification task"""
    start_time = time.time()
    
    try:
        from codon_verifier.metrics import (
            cai, tai, fop, gc_content, codon_pair_bias_score, 
            codon_pair_score, cpg_upa_content, rare_codon_runs,
            homopolymers, find_forbidden_sites
        )
        from codon_verifier.hosts.tables import get_host_tables
        from codon_verifier.codon_utils import validate_cds
        
        task_type = task_data.get('task', 'analyze')
        input_data = task_data.get('input', {})
        
        sequence = input_data.get('sequence', '').upper().replace('U', 'T')
        host = input_data.get('host', 'E_coli')
        forbidden_motifs = input_data.get('forbidden_motifs', [])
        
        logger.info(f"Analyzing sequence (length: {len(sequence)}, host: {host})")
        
        # Validate CDS
        valid, msg = validate_cds(sequence)
        if not valid:
            logger.warning(f"Sequence validation failed: {msg}")
        
        # Get host tables
        usage_table, trna_weights, cpb_table = get_host_tables(host, include_cpb=True)
        
        # Calculate codon metrics
        metrics = {}
        warnings = []
        
        try:
            # Basic sequence metrics
            metrics['gc_content'] = gc_content(sequence)
            metrics['length'] = len(sequence)
            metrics['has_start_codon'] = sequence.startswith('ATG')
            metrics['has_stop_codon'] = sequence.endswith(('TAA', 'TAG', 'TGA'))
            
            if valid:
                # Codon usage metrics
                metrics['cai'] = cai(sequence, usage_table)
                metrics['tai'] = tai(sequence, trna_weights)
                metrics['fop'] = fop(sequence, usage_table)
                
                # Codon pair metrics
                metrics['cpb'] = codon_pair_bias_score(sequence, cpb_table)
                metrics['cps'] = codon_pair_score(sequence, usage_table)
                
                # Dinucleotide analysis
                dinuc_stats = cpg_upa_content(sequence)
                metrics.update({
                    'cpg_count': dinuc_stats['cpg_count'],
                    'cpg_freq': dinuc_stats['cpg_freq'],
                    'cpg_obs_exp': dinuc_stats['cpg_obs_exp'],
                    'upa_count': dinuc_stats['upa_count'],
                    'upa_freq': dinuc_stats['upa_freq'],
                    'upa_obs_exp': dinuc_stats['upa_obs_exp'],
                })
                
                # Check for problematic features
                if dinuc_stats['cpg_obs_exp'] > 1.5:
                    warnings.append(f"High CpG content detected (obs/exp ratio: {dinuc_stats['cpg_obs_exp']:.2f})")
                if dinuc_stats['upa_obs_exp'] > 1.5:
                    warnings.append(f"High UpA content detected (obs/exp ratio: {dinuc_stats['upa_obs_exp']:.2f})")
                
                # Rare codon runs
                rare_runs = rare_codon_runs(sequence, usage_table)
                metrics['rare_codon_runs'] = len(rare_runs)
                if rare_runs:
                    warnings.append(f"Found {len(rare_runs)} rare codon runs")
                
                # Homopolymers
                homos = homopolymers(sequence, min_len=6)
                metrics['homopolymers'] = len(homos)
                if homos:
                    warnings.append(f"Found {len(homos)} homopolymer stretches (≥6bp)")
                
                # Forbidden motifs
                if forbidden_motifs:
                    forbidden_hits = find_forbidden_sites(sequence, forbidden_motifs)
                    metrics['forbidden_sites'] = len(forbidden_hits)
                    if forbidden_hits:
                        warnings.append(f"Found {len(forbidden_hits)} forbidden motif sites")
                
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            warnings.append(f"Metric calculation error: {str(e)}")
        
        result = {
            "task": task_type,
            "status": "success" if valid else "warning",
            "output": {
                "valid": valid,
                "validation_message": msg if not valid else "Sequence is valid CDS",
                "metrics": metrics,
                "warnings": warnings
            },
            "metadata": {
                "request_id": task_data.get('metadata', {}).get('request_id', 'unknown'),
                "processing_time_ms": int((time.time() - start_time) * 1000),
                "service": "sequence_analyzer",
                "version": "2.0.0",
                "host": host
            }
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing task: {e}")
        import traceback
        traceback.print_exc()
        return {
            "task": task_data.get('task', 'unknown'),
            "status": "error",
            "error": str(e),
            "metadata": {
                "request_id": task_data.get('metadata', {}).get('request_id', 'unknown'),
                "processing_time_ms": int((time.time() - start_time) * 1000),
                "service": "sequence_analyzer",
                "version": "2.0.0"
            }
        }


def main():
    parser = argparse.ArgumentParser(description='Sequence Analyzer Service')
    parser.add_argument('--input', type=str, required=True,
                       help='Input JSON file path')
    parser.add_argument('--output', type=str,
                       help='Output JSON file path')
    parser.add_argument('--batch', action='store_true',
                       help='Process multiple tasks')
    
    args = parser.parse_args()
    
    load_verifier()
    
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    with open(input_path, 'r') as f:
        input_data = json.load(f)
    
    # Auto-detect batch mode based on input data structure
    if isinstance(input_data, list):
        logger.info(f"Processing {len(input_data)} tasks in batch mode")
        results = [process_task(task) for task in input_data]
    else:
        logger.info("Processing single task")
        results = process_task(input_data)
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = Path('/data/output/sequence_analyzer')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{input_path.stem}_result.json"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"✓ Results written to: {output_path}")


if __name__ == '__main__':
    main()

