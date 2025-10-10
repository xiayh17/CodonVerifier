#!/usr/bin/env python3
"""
Microservice-Based Expression Enhancement Pipeline

This script orchestrates the complete workflow:
1. Data Conversion (TSV → JSONL)
2. Evo2 Feature Extraction (via microservice)
3. Expression Enhancement (with Evo2 features)
4. Surrogate Model Training (optional)

Usage:
    # Full pipeline
    python scripts/microservice_enhance_expression.py \\
        --input data/input/Ec.tsv \\
        --output data/enhanced/ecoli_enhanced.jsonl \\
        --evo2-service

    # Skip Evo2 (use metadata only)
    python scripts/microservice_enhance_expression.py \\
        --input data/input/Ec.tsv \\
        --output data/enhanced/ecoli_baseline.jsonl

Author: Codon Verifier Team
Date: 2025-10-04
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional
import time

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
logger = logging.getLogger('microservice-pipeline')


class MicroservicePipeline:
    """Orchestrates the complete enhancement pipeline using microservices."""
    
    def __init__(self, use_docker: bool = True, project_root: Optional[Path] = None):
        self.use_docker = use_docker
        self.docker_compose_file = "docker-compose.microservices.yml"
        
        # Save project root directory (absolute path)
        if project_root is None:
            # Assume script is in scripts/ directory
            self.project_root = Path(__file__).parent.parent.resolve()
        else:
            self.project_root = Path(project_root).resolve()
    
    def step1_convert_tsv_to_jsonl(
        self,
        input_tsv: Path,
        output_jsonl: Path
    ) -> bool:
        """
        Step 1: Convert TSV to JSONL format.
        
        Uses codon_verifier.data_converter module.
        """
        logger.info("=" * 60)
        logger.info("STEP 1: Converting TSV to JSONL")
        logger.info("=" * 60)
        
        try:
            cmd = [
                "python", "-m", "codon_verifier.data_converter",
                "--input", str(input_tsv),
                "--output", str(output_jsonl)
            ]
            
            logger.info(f"Command: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=str(self.project_root))
            logger.info(result.stdout)
            logger.info(f"✓ JSONL dataset created: {output_jsonl}")
            return True
        
        except subprocess.CalledProcessError as e:
            logger.error(f"Conversion failed: {e}")
            logger.error(e.stderr)
            return False
    
    def step2_extract_evo2_features(
        self,
        input_jsonl: Path,
        output_features: Path,
        limit: Optional[int] = None
    ) -> bool:
        """
        Step 2: Extract Evo2 features using microservice.
        
        Runs Evo2 service via Docker Compose.
        """
        logger.info("=" * 60)
        logger.info("STEP 2: Extracting Evo2 Features (Microservice)")
        logger.info("=" * 60)
        
        if not self.use_docker:
            # Run locally without Docker
            logger.info("Running Evo2 feature extraction locally")
            try:
                cmd = [
                    "python", "services/evo2/app_enhanced.py",
                    "--input", str(input_jsonl),
                    "--output", str(output_features),
                    "--mode", "features"
                ]
                if limit:
                    cmd.extend(["--limit", str(limit)])
                
                logger.info(f"Command: {' '.join(cmd)}")
                result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=str(self.project_root))
                logger.info(result.stdout)
                logger.info(f"✓ Evo2 features extracted: {output_features}")
                return True
            
            except subprocess.CalledProcessError as e:
                logger.error(f"Feature extraction failed: {e}")
                logger.error(e.stderr)
                return False
        
        else:
            # Run via Docker Compose
            logger.info("Running Evo2 service via Docker Compose")
            logger.warning("NOTE: Docker Evo2 service uses original app.py, not app_enhanced.py")
            logger.warning("For enhanced features, use --no-docker option")
            
            # Convert paths to container paths
            container_input = f"/data/converted/{input_jsonl.name}"
            container_output = f"/data/output/evo2/{output_features.name}"
            
            # Build docker-compose command
            # Note: Evo2 Docker service only supports app.py (placeholder mode)
            cmd = [
                "docker-compose",
                "-f", self.docker_compose_file,
                "run", "--rm",
                "-v", f"{input_jsonl.parent.absolute()}:/data/converted:ro",
                "-v", f"{output_features.parent.absolute()}:/data/output/evo2",
                "evo2",
                "--input", container_input,
                "--output", container_output,
                "--batch"  # Process as batch
            ]
            
            try:
                logger.info(f"Command: {' '.join(cmd)}")
                logger.info("(This may take several minutes...)")
                
                result = subprocess.run(cmd, check=True, cwd=str(self.project_root))
                logger.info(f"✓ Evo2 features extracted: {output_features}")
                return True
            
            except subprocess.CalledProcessError as e:
                logger.error(f"Docker service failed: {e}")
                return False
    
    def step3_enhance_expression(
        self,
        input_jsonl: Path,
        evo2_features: Optional[Path],
        output_enhanced: Path,
        mode: str = "model_enhanced"
    ) -> bool:
        """
        Step 3: Enhance expression estimates with Evo2 features.
        
        Uses the expression estimator module.
        """
        logger.info("=" * 60)
        logger.info("STEP 3: Enhancing Expression Estimates")
        logger.info("=" * 60)
        
        try:
            cmd = [
                "python", "scripts/enhance_expression_estimates.py",
                "--input", str(input_jsonl),
                "--output", str(output_enhanced),
                "--mode", mode
            ]
            
            if evo2_features and evo2_features.exists():
                cmd.extend(["--evo2-results", str(evo2_features)])
            
            logger.info(f"Command: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=str(self.project_root))
            logger.info(result.stdout)
            logger.info(f"✓ Enhanced dataset created: {output_enhanced}")
            return True
        
        except subprocess.CalledProcessError as e:
            logger.error(f"Enhancement failed: {e}")
            logger.error(e.stderr)
            return False
    
    def step4_train_surrogate(
        self,
        enhanced_data: Path,
        model_output: Path,
        force_docker: bool = True
    ) -> bool:
        """
        Step 4 (Optional): Train surrogate model with enhanced data using microservice.
        
        Note: Training always uses Docker by default to avoid sklearn dependency issues,
        even when --no-docker is specified for other steps.
        """
        logger.info("=" * 60)
        logger.info("STEP 4: Training Surrogate Model (Microservice)")
        logger.info("=" * 60)
        
        # Always use Docker for training unless explicitly disabled
        use_docker_for_training = force_docker or not self.use_docker
        
        if not use_docker_for_training:
            # Run locally without Docker (requires sklearn, lightgbm installed)
            logger.info("Running training locally (requires sklearn, lightgbm)")
            try:
                cmd = [
                    "python3", "-m", "codon_verifier.train_surrogate_multihost",
                    "--data", str(enhanced_data),
                    "--out", str(model_output),
                    "--mode", "unified"
                ]
                
                logger.info(f"Command: {' '.join(cmd)}")
                logger.info("(This may take 5-30 minutes depending on data size...)")
                
                result = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=str(self.project_root))
                logger.info(result.stdout)
                logger.info(f"✓ Surrogate model trained: {model_output}")
                return True
            
            except subprocess.CalledProcessError as e:
                logger.error(f"Training failed: {e}")
                logger.error(e.stderr)
                return False
        
        else:
            # Use training microservice via Docker Compose
            logger.info("Running training service via Docker Compose")
            if self.use_docker:
                logger.info("(Using Docker because --no-docker was NOT specified)")
            else:
                logger.info("(Using Docker for training even though --no-docker was specified)")
                logger.info("(This avoids sklearn dependency issues)")
            
            # Create training configuration JSON
            config_path = model_output.parent / "training_config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            training_config = {
                "task": "train_unified",
                "config": {
                    "data_paths": [f"/data/enhanced/{enhanced_data.name}"],
                    "output_path": f"/data/output/models/{model_output.name}",
                    "mode": "unified",
                    "data_config": {
                        "min_sequence_length": 10,
                        "max_sequence_length": 10000,
                        "filter_reviewed_only": False,
                        "balance_hosts": False
                    },
                    "surrogate_config": {
                        "quantile_hi": 0.9,
                        "test_size": 0.15
                    }
                },
                "metadata": {
                    "request_id": f"train_{int(time.time())}"
                }
            }
            
            with open(config_path, 'w') as f:
                json.dump(training_config, f, indent=2)
            
            logger.info(f"Training configuration saved to: {config_path}")
            
            # Build docker-compose command
            cmd = [
                "docker-compose",
                "-f", self.docker_compose_file,
                "run", "--rm",
                "-v", f"{enhanced_data.parent.absolute()}:/data/enhanced:ro",
                "-v", f"{model_output.parent.absolute()}:/data/output/models",
                "-v", f"{config_path.parent.absolute()}:/data/config",
                "training",
                "--input", f"/data/config/{config_path.name}"
            ]
            
            try:
                logger.info(f"Command: {' '.join(cmd)}")
                logger.info("(This may take 5-30 minutes depending on data size...)")
                
                result = subprocess.run(cmd, check=True, cwd=str(self.project_root))
                logger.info(f"✓ Surrogate model trained: {model_output}")
                
                # Clean up config file
                config_path.unlink()
                
                return True
            
            except subprocess.CalledProcessError as e:
                logger.error(f"Docker training service failed: {e}")
                return False
    
    def run_full_pipeline(
        self,
        input_tsv: Path,
        output_dir: Path,
        use_evo2: bool = True,
        train_surrogate: bool = False,
        limit_records: Optional[int] = None
    ) -> Dict:
        """
        Run the complete pipeline.
        
        Args:
            input_tsv: Input TSV file
            output_dir: Output directory
            use_evo2: Whether to use Evo2 microservice
            train_surrogate: Whether to train surrogate model
            limit_records: Optional limit for testing
            
        Returns:
            Results dictionary
        """
        logger.info("")
        logger.info("="  * 60)
        logger.info("MICROSERVICE EXPRESSION ENHANCEMENT PIPELINE")
        logger.info("=" * 60)
        logger.info(f"Input: {input_tsv}")
        logger.info(f"Output Directory: {output_dir}")
        logger.info(f"Use Evo2: {use_evo2}")
        logger.info(f"Train Surrogate: {train_surrogate}")
        if limit_records:
            logger.info(f"Limit: {limit_records} records (testing mode)")
        logger.info("=" * 60)
        logger.info("")
        
        start_time = time.time()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            "status": "in_progress",
            "steps": {},
            "total_time_s": 0.0
        }
        
        # Step 1: Convert TSV to JSONL
        jsonl_path = output_dir / f"{input_tsv.stem}.jsonl"
        step1_success = self.step1_convert_tsv_to_jsonl(input_tsv, jsonl_path)
        results["steps"]["1_convert"] = {
            "success": step1_success,
            "output": str(jsonl_path)
        }
        
        if not step1_success:
            results["status"] = "failed"
            return results
        
        # Step 2: Extract Evo2 features (optional)
        features_path = None
        if use_evo2:
            features_path = output_dir / f"{input_tsv.stem}_evo2_features.json"
            step2_success = self.step2_extract_evo2_features(
                jsonl_path,
                features_path,
                limit=limit_records
            )
            results["steps"]["2_evo2_features"] = {
                "success": step2_success,
                "output": str(features_path) if step2_success else None
            }
            
            if not step2_success:
                logger.warning("Evo2 feature extraction failed, falling back to metadata-only mode")
                features_path = None
        
        # Step 3: Enhance expression estimates
        mode = "model_enhanced" if (use_evo2 and features_path) else "metadata_only"
        enhanced_path = output_dir / f"{input_tsv.stem}_enhanced.jsonl"
        step3_success = self.step3_enhance_expression(
            jsonl_path,
            features_path,
            enhanced_path,
            mode=mode
        )
        results["steps"]["3_enhance_expression"] = {
            "success": step3_success,
            "output": str(enhanced_path),
            "mode": mode
        }
        
        if not step3_success:
            results["status"] = "failed"
            return results
        
        # Step 4: Train surrogate (optional)
        if train_surrogate:
            model_path = output_dir / "models" / f"{input_tsv.stem}_surrogate.pkl"
            model_path.parent.mkdir(exist_ok=True)
            step4_success = self.step4_train_surrogate(enhanced_path, model_path)
            results["steps"]["4_train_surrogate"] = {
                "success": step4_success,
                "output": str(model_path) if step4_success else None
            }
        
        # Final results
        results["total_time_s"] = time.time() - start_time
        results["status"] = "success"
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Total time: {results['total_time_s']:.2f}s")
        logger.info("")
        logger.info("Output files:")
        for step_name, step_data in results["steps"].items():
            if step_data.get("success") and step_data.get("output"):
                logger.info(f"  {step_name}: {step_data['output']}")
        logger.info("")
        logger.info("Next steps:")
        logger.info(f"  1. Review enhanced data: {enhanced_path}")
        if train_surrogate and results["steps"].get("4_train_surrogate", {}).get("success"):
            model_path = results["steps"]["4_train_surrogate"]["output"]
            logger.info(f"  2. Test surrogate model: {model_path}")
        logger.info("  3. Use in production optimization pipeline")
        logger.info("=" * 60)
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description='Microservice-Based Expression Enhancement Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with Evo2 microservice
  python scripts/microservice_enhance_expression.py \\
    --input data/input/Ec.tsv \\
    --output-dir data/enhanced/ecoli \\
    --evo2-service

  # Metadata-only mode (no Evo2)
  python scripts/microservice_enhance_expression.py \\
    --input data/input/Ec.tsv \\
    --output-dir data/enhanced/ecoli_baseline

  # With surrogate model training
  python scripts/microservice_enhance_expression.py \\
    --input data/input/Ec.tsv \\
    --output-dir data/enhanced/ecoli \\
    --evo2-service \\
    --train-surrogate

  # Testing with limited records
  python scripts/microservice_enhance_expression.py \\
    --input data/input/Ec.tsv \\
    --output-dir data/enhanced/test \\
    --evo2-service \\
    --limit 1000
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input TSV file path'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for all generated files'
    )
    parser.add_argument(
        '--evo2-service',
        action='store_true',
        help='Use Evo2 microservice for feature extraction'
    )
    parser.add_argument(
        '--no-docker',
        action='store_true',
        help='Run locally without Docker (for development)'
    )
    parser.add_argument(
        '--train-surrogate',
        action='store_true',
        help='Train surrogate model after enhancement'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of records (for testing)'
    )
    
    args = parser.parse_args()
    
    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    
    # Create pipeline
    pipeline = MicroservicePipeline(use_docker=not args.no_docker)
    
    # Run pipeline
    try:
        results = pipeline.run_full_pipeline(
            input_tsv=input_path,
            output_dir=output_dir,
            use_evo2=args.evo2_service,
            train_surrogate=args.train_surrogate,
            limit_records=args.limit
        )
        
        # Save results summary
        summary_path = output_dir / "pipeline_results.json"
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Pipeline summary saved to: {summary_path}")
        
        if results["status"] != "success":
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

