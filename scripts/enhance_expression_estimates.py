#!/usr/bin/env python3
"""
Enhance Expression Estimates with Evo2 Features

This script takes the original JSONL dataset and Evo2 model outputs,
then re-estimates expression levels using the enhanced estimator.

Usage:
    python scripts/enhance_expression_estimates.py \\
        --input data/converted/merged_dataset.jsonl \\
        --evo2-results data/output/evo2/merged_dataset_result.json \\
        --output data/converted/merged_dataset_enhanced.jsonl \\
        --mode model_enhanced

Author: Codon Verifier Team
Date: 2025-10-04
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from codon_verifier.expression_estimator import (
    ExpressionEstimator,
    load_evo2_features
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('enhance_expression')


def enhance_dataset(
    input_jsonl: str,
    evo2_results: str,
    output_jsonl: str,
    mode: str = "model_enhanced"
) -> Dict[str, any]:
    """
    Re-estimate expression levels with model enhancement.
    
    Args:
        input_jsonl: Original dataset path
        evo2_results: Evo2 model output JSON
        output_jsonl: Output path for enhanced dataset
        mode: Estimation mode
        
    Returns:
        Statistics dictionary
    """
    logger.info("=" * 60)
    logger.info("Expression Enhancement Pipeline")
    logger.info("=" * 60)
    
    # Load Evo2 features
    logger.info(f"Loading Evo2 features from {evo2_results}")
    evo2_features = load_evo2_features(evo2_results)
    
    if not evo2_features:
        logger.warning("No Evo2 features loaded - will use metadata-only mode")
        mode = "metadata_only"
    else:
        logger.info(f"Loaded features for {len(evo2_features)} records")
    
    # Initialize estimator
    estimator = ExpressionEstimator(mode=mode)
    
    # Process records
    logger.info(f"Processing records from {input_jsonl}")
    
    stats = {
        "total_records": 0,
        "enhanced_records": 0,
        "metadata_only_records": 0,
        "original_avg": 0.0,
        "enhanced_avg": 0.0,
        "max_change": 0.0,
        "changes": []  # Track all changes for statistics
    }
    
    input_path = Path(input_jsonl)
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        for idx, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
                stats["total_records"] += 1
                
                # Extract metadata
                metadata = record.get("metadata", {})
                extra_features = record.get("extra_features", {})
                
                reviewed = "reviewed" if extra_features.get("reviewed", False) else "unreviewed"
                
                # Handle subcellular_location which might be NaN (float)
                subcellular_raw = metadata.get("subcellular_location", "")
                if isinstance(subcellular_raw, float):
                    import math
                    subcellular = "" if math.isnan(subcellular_raw) else str(subcellular_raw)
                else:
                    subcellular = str(subcellular_raw) if subcellular_raw else ""
                
                length = extra_features.get("length", len(record.get("protein_aa", "")))
                organism = metadata.get("organism", "")
                sequence = record.get("sequence", "")
                
                # Get original expression
                original_expr = record.get("expression", {})
                original_value = original_expr.get("value", 50.0)
                
                # Get model features if available
                model_features = evo2_features.get(idx, None)
                
                # Estimate enhanced expression
                new_value, new_confidence = estimator.estimate(
                    reviewed=reviewed,
                    subcellular_location=subcellular,
                    protein_length=length,
                    organism=organism,
                    model_features=model_features,
                    sequence=sequence
                )
                
                # Update record
                record["expression"] = {
                    "value": new_value,
                    "unit": "estimated_enhanced" if model_features else "estimated",
                    "assay": "model_enhanced_heuristic" if model_features else "metadata_heuristic",
                    "confidence": new_confidence,
                    "original_value": original_value  # Keep for comparison
                }
                
                # Merge Evo2 features into extra_features if available
                if model_features and isinstance(model_features, dict):
                    if "extra_features" not in record:
                        record["extra_features"] = {}
                    # Add Evo2 features with evo2_ prefix
                    for key, value in model_features.items():
                        if key not in ("protein_id", "sequence"):  # Skip non-feature fields
                            record["extra_features"][f"evo2_{key}"] = float(value) if isinstance(value, (int, float)) else value
                
                # Track statistics
                change = abs(new_value - original_value)
                stats["changes"].append(change)
                stats["max_change"] = max(stats["max_change"], change)
                
                if model_features:
                    stats["enhanced_records"] += 1
                else:
                    stats["metadata_only_records"] += 1
                
                # Write enhanced record
                fout.write(json.dumps(record, ensure_ascii=False) + '\n')
                
                # Progress logging
                if (idx + 1) % 5000 == 0:
                    logger.info(f"Processed {idx + 1} records...")
            
            except Exception as e:
                logger.error(f"Error processing record {idx}: {e}")
                continue
    
    # Compute final statistics
    if stats["total_records"] > 0:
        enhancement_rate = stats["enhanced_records"] / stats["total_records"] * 100
        logger.info("")
        logger.info("=" * 60)
        logger.info("Enhancement Statistics")
        logger.info("=" * 60)
        logger.info(f"Total records: {stats['total_records']}")
        logger.info(f"Enhanced with Evo2: {stats['enhanced_records']} ({enhancement_rate:.1f}%)")
        logger.info(f"Metadata only: {stats['metadata_only_records']}")
        
        if stats["changes"]:
            import numpy as np
            changes_array = np.array(stats["changes"])
            logger.info(f"Expression value changes:")
            logger.info(f"  Mean absolute change: {np.mean(changes_array):.2f}")
            logger.info(f"  Median absolute change: {np.median(changes_array):.2f}")
            logger.info(f"  Max absolute change: {np.max(changes_array):.2f}")
            logger.info(f"  Std deviation: {np.std(changes_array):.2f}")
            
            # Histogram of changes
            logger.info(f"Change distribution:")
            bins = [0, 5, 10, 15, 20, np.inf]
            hist, _ = np.histogram(changes_array, bins=bins)
            for i in range(len(bins)-1):
                if bins[i+1] == np.inf:
                    logger.info(f"  >{bins[i]}: {hist[i]} records ({hist[i]/len(changes_array)*100:.1f}%)")
                else:
                    logger.info(f"  {bins[i]}-{bins[i+1]}: {hist[i]} records ({hist[i]/len(changes_array)*100:.1f}%)")
        
        logger.info("")
        logger.info(f"✓ Enhanced dataset written to: {output_path}")
        logger.info("=" * 60)
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Enhance expression estimates with Evo2 model features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic enhancement with Evo2 features
  python scripts/enhance_expression_estimates.py \\
    --input data/converted/merged_dataset.jsonl \\
    --evo2-results data/output/evo2/merged_dataset_result.json \\
    --output data/converted/merged_dataset_enhanced.jsonl
  
  # Full mode (includes sequence features)
  python scripts/enhance_expression_estimates.py \\
    --input data/converted/merged_dataset.jsonl \\
    --evo2-results data/output/evo2/merged_dataset_result.json \\
    --output data/converted/merged_dataset_full.jsonl \\
    --mode full
  
  # Metadata-only baseline (no Evo2)
  python scripts/enhance_expression_estimates.py \\
    --input data/converted/merged_dataset.jsonl \\
    --output data/converted/merged_dataset_baseline.jsonl \\
    --mode metadata_only
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input JSONL dataset path'
    )
    parser.add_argument(
        '--evo2-results',
        type=str,
        help='Evo2 model output JSON file (optional for metadata-only mode)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output path for enhanced dataset'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['metadata_only', 'model_enhanced', 'full'],
        default='model_enhanced',
        help='Enhancement mode (default: model_enhanced)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode != 'metadata_only' and not args.evo2_results:
        parser.error("--evo2-results is required for model_enhanced and full modes")
    
    if not Path(args.input).exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    if args.evo2_results and not Path(args.evo2_results).exists():
        logger.error(f"Evo2 results file not found: {args.evo2_results}")
        sys.exit(1)
    
    # Run enhancement
    try:
        stats = enhance_dataset(
            input_jsonl=args.input,
            evo2_results=args.evo2_results or "",
            output_jsonl=args.output,
            mode=args.mode
        )
        
        logger.info("Enhancement completed successfully!")
        
    except Exception as e:
        logger.error(f"Enhancement failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

