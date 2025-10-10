#!/usr/bin/env python3
"""
Resume Complete Feature Generation from a previous run

This script allows you to resume from the last successful step
if the pipeline was interrupted.

Usage:
    python scripts/resume_complete_features.py \\
        --temp-dir data/temp_complete_1760002553 \\
        --output data/enhanced/Ec_complete.jsonl \\
        --use-docker
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.generate_complete_features import CompleteFeaturePipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Resume Complete Feature Generation Pipeline"
    )
    parser.add_argument('--temp-dir', required=True,
                       help="Temporary directory from previous run")
    parser.add_argument('--output', required=True,
                       help="Output JSONL file")
    parser.add_argument('--use-docker', action='store_true', default=True,
                       help="Use Docker services")
    parser.add_argument('--no-docker', action='store_true',
                       help="Run locally without Docker")
    parser.add_argument('--limit', type=int,
                       help="Limit number of records for testing")
    
    args = parser.parse_args()
    
    use_docker = args.use_docker and not args.no_docker
    temp_dir = Path(args.temp_dir)
    
    if not temp_dir.exists():
        logger.error(f"Temp directory not found: {temp_dir}")
        sys.exit(1)
    
    # Check which files exist
    base_jsonl = temp_dir / "base.jsonl"
    structure_json = temp_dir / "structure.json"
    msa_json = temp_dir / "msa.json"
    integrated_jsonl = temp_dir / "integrated.jsonl"
    evo2_json = temp_dir / "evo2_features.json"
    
    logger.info(f"Resuming from: {temp_dir}")
    logger.info(f"Files found:")
    logger.info(f"  base.jsonl: {'✓' if base_jsonl.exists() else '✗'}")
    logger.info(f"  structure.json: {'✓' if structure_json.exists() else '✗'}")
    logger.info(f"  msa.json: {'✓' if msa_json.exists() else '✗'}")
    logger.info(f"  integrated.jsonl: {'✓' if integrated_jsonl.exists() else '✗'}")
    logger.info(f"  evo2_features.json: {'✓' if evo2_json.exists() else '✗'}")
    
    pipeline = CompleteFeaturePipeline(use_docker=use_docker)
    
    # Resume from where we left off
    if not integrated_jsonl.exists():
        logger.error("❌ integrated.jsonl not found. Cannot resume from this point.")
        logger.info("💡 Please re-run the full pipeline from the beginning.")
        sys.exit(1)
    
    # If testing with limit, create a subset of integrated.jsonl
    test_integrated = integrated_jsonl
    test_evo2_json = evo2_json
    
    if args.limit:
        logger.info(f"\n🧪 Testing mode: limiting to {args.limit} records")
        test_integrated = temp_dir / f"integrated_test_{args.limit}.jsonl"
        test_evo2_json = temp_dir / f"evo2_features_test_{args.limit}.json"
        
        # Create subset
        import json
        with open(integrated_jsonl, 'r') as fin, open(test_integrated, 'w') as fout:
            for i, line in enumerate(fin):
                if i >= args.limit:
                    break
                fout.write(line)
        logger.info(f"✓ Created test subset: {test_integrated}")
    
    # Step 5: Extract Evo2 features (if not done)
    if not test_evo2_json.exists():
        logger.info("\n🔄 Resuming from Step 5: Evo2 Feature Extraction")
        if not pipeline.step5_extract_evo2_features(str(test_integrated), str(test_evo2_json)):
            logger.error("❌ Evo2 feature extraction failed")
            sys.exit(1)
    else:
        logger.info("✓ Evo2 features already exist, skipping Step 5")
    
    # Step 6: Enhance expression estimates
    logger.info("\n🔄 Running Step 6: Expression Enhancement")
    if not pipeline.step6_enhance_expression(str(test_integrated), str(test_evo2_json), args.output):
        logger.error("❌ Expression enhancement failed")
        sys.exit(1)
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ Pipeline resumed and completed successfully!")
    logger.info(f"📁 Output: {args.output}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()

