#!/usr/bin/env python3
"""
Complete Feature Generation Pipeline

This script directly converts TSV to the final enhanced expression JSONL file,
combining all steps into a single streamlined process:
1. Convert TSV to JSONL (using data_converter)
2. Generate structure features (lite)
3. Generate MSA features (lite) 
4. Extract Evo2 features
5. Enhance expression estimates
6. Output final training-ready JSONL

Usage:
    python scripts/generate_complete_features.py \\
        --input data/2025_bio-os_data/dataset/Ec.tsv \\
        --output data/enhanced/Ec_complete.jsonl \\
        --use-docker

Author: CodonVerifier Team
Date: 2025-01-27
"""

import argparse
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CompleteFeaturePipeline:
    """Streamlined pipeline for complete feature generation"""
    
    def __init__(
        self,
        use_docker: bool = True,
        project_root: Optional[Path] = None
    ):
        self.use_docker = use_docker
        
        # Save project root directory (absolute path)
        if project_root is None:
            # Assume script is in scripts/ directory
            self.project_root = Path(__file__).parent.parent.resolve()
        else:
            self.project_root = Path(project_root).resolve()
        
        self.stats = {
            'start_time': time.time(),
            'steps_completed': []
        }
    
    def step1_convert_tsv_to_jsonl(
        self,
        tsv_path: str,
        output_jsonl: str,
        limit: Optional[int] = None
    ) -> bool:
        """Step 1: Convert TSV to JSONL using data_converter"""
        logger.info("=" * 60)
        logger.info("STEP 1: Converting TSV to JSONL")
        logger.info("=" * 60)
        
        try:
            cmd = [
                "python", "-m", "codon_verifier.data_converter",
                "--input", tsv_path,
                "--output", output_jsonl
            ]
            if limit:
                cmd.extend(["--max-records", str(limit)])
            
            logger.info(f"Command: {' '.join(cmd)}")
            
            # Use Popen for real-time output streaming
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(self.project_root)
            )
            
            # Stream output in real-time
            for line in process.stdout:
                print(line, end='', flush=True)
            
            returncode = process.wait()
            
            if returncode != 0:
                logger.error(f"Conversion failed with code {returncode}")
                return False
            
            logger.info(f"✓ JSONL dataset created: {output_jsonl}")
            self.stats['steps_completed'].append('convert')
            return True
        
        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            return False
    
    def step2_generate_structure_features(
        self,
        input_jsonl: str,
        output_json: str,
        limit: Optional[int] = None
    ) -> bool:
        """Step 2: Generate structure features using lite service"""
        logger.info("=" * 60)
        logger.info("STEP 2: Generating Structure Features")
        logger.info("=" * 60)
        
        if self.use_docker:
            # Docker command
            data_dir = (self.project_root / 'data').resolve()
            input_rel = Path(input_jsonl).resolve().relative_to(data_dir)
            output_rel = Path(output_json).resolve().relative_to(data_dir)
            cmd = [
                'docker-compose', '-f', 'docker-compose.microservices.yml',
                'run', '--rm',
                '-v', f'{data_dir}:/data',
                'structure_features_lite',
                '--input', f'/data/{input_rel}',
                '--output', f'/data/{output_rel}'
            ]
            if limit:
                cmd.extend(['--limit', str(limit)])
        else:
            # Local command
            cmd = [
                sys.executable,
                'services/structure_features_lite/app.py',
                '--input', input_jsonl,
                '--output', output_json
            ]
            if limit:
                cmd.extend(['--limit', str(limit)])
        
        logger.info(f"Running: {' '.join(cmd)}")
        
        # Use Popen for real-time output streaming
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(self.project_root)
        )
        
        # Stream output in real-time
        for line in process.stdout:
            print(line, end='', flush=True)
        
        returncode = process.wait()
        
        if returncode != 0:
            logger.error(f"Structure feature generation failed with code {returncode}")
            return False
        
        logger.info("✓ Structure features generated")
        self.stats['steps_completed'].append('structure')
        return True
    
    def step3_generate_msa_features(
        self,
        input_jsonl: str,
        output_json: str,
        limit: Optional[int] = None
    ) -> bool:
        """Step 3: Generate MSA features using lite service"""
        logger.info("=" * 60)
        logger.info("STEP 3: Generating MSA Features")
        logger.info("=" * 60)
        
        if self.use_docker:
            # Docker command
            data_dir = (self.project_root / 'data').resolve()
            input_rel = Path(input_jsonl).resolve().relative_to(data_dir)
            output_rel = Path(output_json).resolve().relative_to(data_dir)
            cmd = [
                'docker-compose', '-f', 'docker-compose.microservices.yml',
                'run', '--rm',
                '-v', f'{data_dir}:/data',
                'msa_features_lite',
                '--input', f'/data/{input_rel}',
                '--output', f'/data/{output_rel}'
            ]
            if limit:
                cmd.extend(['--limit', str(limit)])
        else:
            # Local command
            cmd = [
                sys.executable,
                'services/msa_features_lite/app.py',
                '--input', input_jsonl,
                '--output', output_json
            ]
            if limit:
                cmd.extend(['--limit', str(limit)])
        
        logger.info(f"Running: {' '.join(cmd)}")
        
        # Use Popen for real-time output streaming
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(self.project_root)
        )
        
        # Stream output in real-time
        for line in process.stdout:
            print(line, end='', flush=True)
        
        returncode = process.wait()
        
        if returncode != 0:
            logger.error(f"MSA feature generation failed with code {returncode}")
            return False
        
        logger.info("✓ MSA features generated")
        self.stats['steps_completed'].append('msa')
        return True
    
    def step4_integrate_features(
        self,
        input_jsonl: str,
        structure_json: str,
        msa_json: str,
        output_jsonl: str
    ) -> bool:
        """Step 4: Integrate all features"""
        logger.info("=" * 60)
        logger.info("STEP 4: Integrating All Features")
        logger.info("=" * 60)
        
        if self.use_docker:
            # Docker command
            data_dir = (self.project_root / 'data').resolve()
            input_rel = Path(input_jsonl).resolve().relative_to(data_dir)
            struct_rel = Path(structure_json).resolve().relative_to(data_dir)
            msa_rel = Path(msa_json).resolve().relative_to(data_dir)
            out_rel = Path(output_jsonl).resolve().relative_to(data_dir)
            cmd = [
                'docker-compose', '-f', 'docker-compose.microservices.yml',
                'run', '--rm',
                '-v', f'{data_dir}:/data',
                'feature_integrator',
                '--input', f'/data/{input_rel}',
                '--structure-features', f'/data/{struct_rel}',
                '--msa-features', f'/data/{msa_rel}',
                '--output', f'/data/{out_rel}'
            ]
        else:
            # Local command
            cmd = [
                sys.executable,
                'services/feature_integrator/app.py',
                '--input', input_jsonl,
                '--structure-features', structure_json,
                '--msa-features', msa_json,
                '--output', output_jsonl
            ]
        
        logger.info(f"Running: {' '.join(cmd)}")
        
        # Use Popen for real-time output streaming
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(self.project_root)
        )
        
        # Stream output in real-time
        for line in process.stdout:
            print(line, end='', flush=True)
        
        returncode = process.wait()
        
        if returncode != 0:
            logger.error(f"Feature integration failed with code {returncode}")
            return False
        
        logger.info("✓ Features integrated")
        self.stats['steps_completed'].append('integrate')
        return True
    
    def step5_extract_evo2_features(
        self,
        input_jsonl: str,
        output_json: str
    ) -> bool:
        """Step 5: Extract Evo2 features"""
        logger.info("=" * 60)
        logger.info("STEP 5: Extracting Evo2 Features")
        logger.info("=" * 60)
        
        try:
            # Use the existing Evo2 extraction logic
            # Pass data-relative paths when using Docker so the evo2 helper can
            # map them into the container volume correctly.
            data_dir = (self.project_root / 'data').resolve()
            input_arg = str(Path(input_jsonl).resolve())
            output_arg = str(Path(output_json).resolve())
            if self.use_docker:
                try:
                    rel_in = Path(input_jsonl).resolve().relative_to(data_dir)
                    rel_out = Path(output_json).resolve().relative_to(data_dir)
                    input_arg = f"data/{rel_in.as_posix()}"
                    output_arg = f"data/{rel_out.as_posix()}"
                except Exception:
                    # Fallback to absolute if relative fails; the extractor will handle errors
                    pass
            cmd = [
                "python", "-c",
                f"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from scripts.train_with_all_features import EnhancedFeatureExtractor

extractor = EnhancedFeatureExtractor(host='E_coli')
success = extractor.extract_evo2_features_if_needed(
    input_jsonl='{input_arg}',
    output_json='{output_arg}',
    use_docker={str(self.use_docker)}
)
sys.exit(0 if success else 1)
"""
            ]
            
            logger.info("Running Evo2 feature extraction...")
            logger.info("⏱️  This may take a long time for large datasets...")
            
            # Use Popen for real-time output streaming
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(self.project_root)
            )
            
            # Stream output in real-time
            for line in process.stdout:
                print(line, end='', flush=True)
            
            # Wait for completion
            returncode = process.wait()
            
            if returncode != 0:
                logger.error(f"Evo2 feature extraction failed with code {returncode}")
                return False
            
            logger.info(f"✓ Evo2 features extracted: {output_json}")
            self.stats['steps_completed'].append('evo2')
            return True
        
        except Exception as e:
            logger.error(f"Evo2 feature extraction failed: {e}")
            return False
    
    def step6_enhance_expression(
        self,
        input_jsonl: str,
        evo2_features: str,
        output_jsonl: str
    ) -> bool:
        """Step 6: Enhance expression estimates"""
        logger.info("=" * 60)
        logger.info("STEP 6: Enhancing Expression Estimates")
        logger.info("=" * 60)
        
        try:
            cmd = [
                "python", "scripts/enhance_expression_estimates.py",
                "--input", input_jsonl,
                "--output", output_jsonl,
                "--evo2-results", evo2_features,
                "--mode", "model_enhanced"
            ]
            
            logger.info(f"Command: {' '.join(cmd)}")
            
            # Use Popen for real-time output streaming
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(self.project_root)
            )
            
            # Stream output in real-time
            for line in process.stdout:
                print(line, end='', flush=True)
            
            returncode = process.wait()
            
            if returncode != 0:
                logger.error(f"Expression enhancement failed with code {returncode}")
                return False
            
            logger.info(f"✓ Enhanced expression data created: {output_jsonl}")
            self.stats['steps_completed'].append('enhance_expression')
            return True
        
        except Exception as e:
            logger.error(f"Expression enhancement failed: {e}")
            return False
    
    def run_complete_pipeline(
        self,
        input_tsv: str,
        output_jsonl: str,
        limit: Optional[int] = None
    ) -> Dict:
        """
        Run the complete pipeline from TSV to final enhanced JSONL
        
        Args:
            input_tsv: Input TSV file path
            output_jsonl: Output JSONL file path
            limit: Optional limit on records for testing
        
        Returns:
            Results dictionary
        """
        logger.info("\n" + "=" * 80)
        logger.info("COMPLETE FEATURE GENERATION PIPELINE")
        logger.info("=" * 80)
        logger.info(f"Input: {input_tsv}")
        logger.info(f"Output: {output_jsonl}")
        logger.info(f"Mode: {'Docker' if self.use_docker else 'Local'}")
        if limit:
            logger.info(f"Limit: {limit} records (testing mode)")
        logger.info("=" * 80 + "\n")
        
        # Create output directory
        output_dir = Path(output_jsonl).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use a project-local temp directory under data/ to satisfy Docker volume mapping
        data_dir = self.project_root / 'data'
        temp_path = data_dir / f"temp_complete_{int(time.time())}"
        temp_path.mkdir(parents=True, exist_ok=True)

        # Intermediate files (all under data/... so relative_to('data') works)
        base_jsonl = temp_path / "base.jsonl"
        structure_json = temp_path / "structure.json"
        msa_json = temp_path / "msa.json"
        integrated_jsonl = temp_path / "integrated.jsonl"
        evo2_json = temp_path / "evo2_features.json"
        
        try:
            # Step 1: Convert TSV to JSONL
            if not self.step1_convert_tsv_to_jsonl(input_tsv, str(base_jsonl), limit=limit):
                return {'success': False, 'error': 'TSV conversion failed'}
            
            # Step 2: Generate structure features
            if not self.step2_generate_structure_features(str(base_jsonl), str(structure_json), limit=limit):
                return {'success': False, 'error': 'Structure feature generation failed'}
            
            # Step 3: Generate MSA features
            if not self.step3_generate_msa_features(str(base_jsonl), str(msa_json), limit=limit):
                return {'success': False, 'error': 'MSA feature generation failed'}
            
            # Step 4: Integrate features
            if not self.step4_integrate_features(str(base_jsonl), str(structure_json), str(msa_json), str(integrated_jsonl)):
                return {'success': False, 'error': 'Feature integration failed'}
            
            # Step 5: Extract Evo2 features
            if not self.step5_extract_evo2_features(str(integrated_jsonl), str(evo2_json)):
                return {'success': False, 'error': 'Evo2 feature extraction failed'}
            
            # Step 6: Enhance expression estimates
            if not self.step6_enhance_expression(str(integrated_jsonl), str(evo2_json), output_jsonl):
                return {'success': False, 'error': 'Expression enhancement failed'}
            
            # Final statistics
            elapsed = time.time() - self.stats['start_time']
            
            logger.info("\n" + "=" * 80)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("=" * 80)
            logger.info(f"Output file: {output_jsonl}")
            logger.info(f"Steps completed: {', '.join(self.stats['steps_completed'])}")
            logger.info(f"Total time: {elapsed:.1f}s")
            logger.info("=" * 80 + "\n")
            
            return {
                'success': True,
                'output_file': output_jsonl,
                'steps_completed': self.stats['steps_completed'],
                'elapsed_time': elapsed
            }
        
        except Exception as e:
            logger.error(f"\n✗ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'steps_completed': self.stats['steps_completed']
            }


def main():
    parser = argparse.ArgumentParser(
        description="Complete Feature Generation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (Docker)
  python scripts/generate_complete_features.py \\
    --input data/2025_bio-os_data/dataset/Ec.tsv \\
    --output data/enhanced/Ec_complete.jsonl \\
    --use-docker

  # Local execution (no Docker)
  python scripts/generate_complete_features.py \\
    --input data/2025_bio-os_data/dataset/Ec.tsv \\
    --output data/enhanced/Ec_complete.jsonl \\
    --no-docker

  # Testing with limited records
  python scripts/generate_complete_features.py \\
    --input data/2025_bio-os_data/dataset/Ec.tsv \\
    --output data/enhanced/Ec_test.jsonl \\
    --limit 100 \\
    --use-docker
"""
    )
    
    parser.add_argument('--input', required=True,
                       help="Input TSV file")
    parser.add_argument('--output', required=True,
                       help="Output JSONL file")
    parser.add_argument('--limit', type=int,
                       help="Limit number of records (for testing)")
    parser.add_argument('--use-docker', action='store_true', default=True,
                       help="Use Docker services (default: True)")
    parser.add_argument('--no-docker', action='store_true',
                       help="Run locally without Docker")
    
    args = parser.parse_args()
    
    # Handle Docker flag
    use_docker = args.use_docker and not args.no_docker
    
    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    # Create pipeline
    pipeline = CompleteFeaturePipeline(use_docker=use_docker)
    
    # Run pipeline
    result = pipeline.run_complete_pipeline(
        input_tsv=str(input_path),
        output_jsonl=args.output,
        limit=args.limit
    )
    
    if result['success']:
        logger.info("🎉 Pipeline completed successfully!")
        logger.info(f"Final output: {result['output_file']}")
    else:
        logger.error(f"❌ Pipeline failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == '__main__':
    main()
