#!/usr/bin/env python3
"""
Enhanced Evo2 Service - DNA Sequence Analysis with Real Model Features

This service provides real Evo2 model features (confidence, likelihood, perplexity)
for expression estimation enhancement. Requires a working Evo2 installation.

Usage:
    docker-compose -f docker-compose.microservices.yml run --rm evo2 \\
        --input /data/converted/merged_dataset.jsonl \\
        --mode features
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import time
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('evo2-service-enhanced')


def load_evo2_model():
    """
    Load Evo2 model with actual implementation.
    
    Tries multiple backends:
    1. Local evo2 package
    2. NVIDIA NIM API
    """
    try:
        # Try importing evo2_adapter for real model
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from codon_verifier.evo2_adapter import check_evo2_available, score_sequence
        
        if check_evo2_available():
            logger.info("✓ Real Evo2 model available (local or NIM)")
            return {"backend": "evo2", "score_fn": score_sequence}
        else:
            logger.error("Evo2 model not available. Please ensure Evo2 is properly installed.")
            raise RuntimeError("Evo2 model is required but not available")
    
    except ImportError:
        logger.error("evo2_adapter not found. Please ensure the evo2_adapter module is available.")
        raise ImportError("evo2_adapter module is required but not found")




def process_sequence_features(
    sequence: str,
    model: Dict,
    request_id: str = "unknown"
) -> Dict[str, Any]:
    """
    Process a single DNA sequence and extract features.
    
    Args:
        sequence: DNA sequence string
        model: Model backend dict
        request_id: Request identifier
        
    Returns:
        Result dict with features
    """
    start_time = time.time()
    
    try:
        # Call scoring function
        score_fn = model["score_fn"]
        features = score_fn(sequence)
        
        # Build result
        result = {
            "task": "extract_features",
            "status": "success",
            "output": {
                "sequence": sequence[:50] + "..." if len(sequence) > 50 else sequence,
                "sequence_length": len(sequence),
                **features,  # Include all extracted features
                "model_version": "evo2-enhanced",
                "backend": model["backend"]
            },
            "metadata": {
                "request_id": request_id,
                "processing_time_ms": int((time.time() - start_time) * 1000),
                "service": "evo2-enhanced",
                "version": "1.0.0"
            }
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing sequence {request_id}: {e}")
        return {
            "task": "extract_features",
            "status": "error",
            "error": str(e),
            "metadata": {
                "request_id": request_id,
                "processing_time_ms": int((time.time() - start_time) * 1000),
                "service": "evo2-enhanced",
                "version": "1.0.0"
            }
        }


def process_jsonl_records(
    input_path: Path,
    output_path: Path,
    model: Dict,
    limit: Optional[int] = None,
    progress_interval: int = 100
) -> Dict[str, Any]:
    """
    Process JSONL dataset and extract Evo2 features for each sequence.
    
    Args:
        input_path: Input JSONL file path
        output_path: Output JSON file path
        model: Model backend
        limit: Optional limit on number of records to process
        
    Returns:
        Statistics dict
    """
    logger.info("=" * 60)
    logger.info("Evo2 Feature Extraction Pipeline")
    logger.info("=" * 60)
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Backend: {model['backend']}")
    
    stats = {
        "total_records": 0,
        "successful": 0,
        "failed": 0,
        "total_time_s": 0.0,
        "avg_time_ms": 0.0
    }
    
    start_time = time.time()
    
    # Prepare output file - use streaming JSON array format
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        # Start JSON array
        fout.write('[\n')
        first_record = True
        
        for idx, line in enumerate(fin):
            if limit and idx >= limit:
                logger.info(f"Reached limit of {limit} records")
                break
            
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
                sequence = record.get("sequence", "")
                
                if not sequence:
                    logger.warning(f"Record {idx}: No sequence found")
                    continue
                
                # Process sequence
                result = process_sequence_features(
                    sequence=sequence,
                    model=model,
                    request_id=f"record_{idx}"
                )
                
                # Write result immediately (streaming)
                if not first_record:
                    fout.write(',\n')
                json.dump(result, fout, indent=2)
                first_record = False
                
                stats["total_records"] += 1
                
                if result["status"] == "success":
                    stats["successful"] += 1
                else:
                    stats["failed"] += 1
                
                # Progress logging
                if (idx + 1) % progress_interval == 0:
                    elapsed = time.time() - start_time
                    rate = (idx + 1) / elapsed
                    remaining = 18780 - (idx + 1)  # Approximate total
                    eta_seconds = remaining / rate if rate > 0 else 0
                    eta_minutes = eta_seconds / 60
                    logger.info(
                        f"Progress: {idx + 1} records processed "
                        f"({rate:.2f} rec/s, {stats['successful']} success, {stats['failed']} failed, "
                        f"ETA: {eta_minutes:.1f} min)"
                    )
                    fout.flush()  # Flush to disk periodically
            
            except Exception as e:
                logger.error(f"Error processing record {idx}: {e}")
                stats["failed"] += 1
                continue
        
        # Close JSON array
        fout.write('\n]\n')
    
    # Final statistics
    stats["total_time_s"] = time.time() - start_time
    stats["avg_time_ms"] = (stats["total_time_s"] * 1000) / stats["total_records"] if stats["total_records"] > 0 else 0
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("Processing Complete")
    logger.info("=" * 60)
    logger.info(f"Total records: {stats['total_records']}")
    logger.info(f"Successful: {stats['successful']}")
    logger.info(f"Failed: {stats['failed']}")
    logger.info(f"Total time: {stats['total_time_s']:.2f}s")
    logger.info(f"Average time: {stats['avg_time_ms']:.1f}ms/record")
    logger.info(f"Processing rate: {stats['total_records'] / stats['total_time_s']:.1f} records/s")
    logger.info(f"✓ Results written to: {output_path}")
    logger.info("=" * 60)
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced Evo2 Service - Real Feature Extraction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process JSONL dataset with Evo2 features
  python app_enhanced.py \\
    --input /data/converted/merged_dataset.jsonl \\
    --output /data/output/evo2/features.json \\
    --mode features
  
  # Process with limit (for testing)
  python app_enhanced.py \\
    --input /data/converted/merged_dataset.jsonl \\
    --output /data/output/evo2/features_test.json \\
    --mode features \\
    --limit 1000
  
  # Process with Evo2 model (requires proper installation)
  python app_enhanced.py \\
    --input /data/converted/merged_dataset.jsonl \\
    --output /data/output/evo2/features.json \\
    --mode features
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input JSONL file path'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output JSON file path (default: auto-generated in /data/output/evo2/)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['features', 'generate'],
        default='features',
        help='Processing mode (default: features)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of records to process (for testing)'
    )
    parser.add_argument(
        '--progress-interval',
        type=int,
        default=100,
        help='Progress report interval (default: 100 records)'
    )
    
    args = parser.parse_args()
    
    # Load model
    try:
        model = load_evo2_model()
    except (RuntimeError, ImportError) as e:
        logger.error(f"Failed to load Evo2 model: {e}")
        logger.error("This service requires a working Evo2 installation.")
        sys.exit(1)
    
    # Check input
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = Path('/data/output/evo2')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{input_path.stem}_features.json"
    
    # Process based on mode
    if args.mode == 'features':
        stats = process_jsonl_records(
            input_path=input_path,
            output_path=output_path,
            model=model,
            limit=args.limit,
            progress_interval=args.progress_interval
        )
    else:
        logger.error(f"Mode '{args.mode}' not implemented yet")
        sys.exit(1)


if __name__ == '__main__':
    main()

