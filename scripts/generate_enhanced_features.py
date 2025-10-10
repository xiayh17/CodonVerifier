#!/usr/bin/env python3
"""
Generate Enhanced Features Pipeline

End-to-end pipeline for generating enhanced features from raw data:
1. Convert TSV to JSONL (if needed)
2. Generate structure features (lite)
3. Generate MSA features (lite)
4. Integrate all features
5. Output training-ready JSONL

Usage:
    python scripts/generate_enhanced_features.py \\
        --input data/2025_bio-os_data/dataset/Ec.tsv \\
        --output data/enhanced/Ec_with_features.jsonl \\
        --mode lite

Author: CodonVerifier Team
Date: 2025-10-05
"""

import argparse
import json
import logging
import sys
import subprocess
from pathlib import Path
import time
from typing import Dict, Optional

# Note: pandas is imported lazily only when needed (non-Docker mode)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedFeaturePipeline:
    """Pipeline for generating enhanced features"""
    
    def __init__(
        self,
        working_dir: str = 'data/temp',
        use_docker: bool = False,
        project_root: Optional[Path] = None
    ):
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)
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
    
    def convert_tsv_to_jsonl(
        self,
        tsv_path: str,
        output_jsonl: str,
        limit: Optional[int] = None
    ) -> int:
        """
        Convert TSV to JSONL format
        
        Returns:
            Number of records converted
        """
        logger.info("="*60)
        logger.info("Step 1: Converting TSV to JSONL")
        logger.info("="*60)
        
        if self.use_docker:
            # Use Docker for TSV conversion
            logger.info("Using Docker for TSV conversion...")
            
            # Create conversion script
            conversion_script = '''
import pandas as pd
import json
import sys

tsv_path = sys.argv[1]
output_path = sys.argv[2]
limit = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] != 'None' else None

df = pd.read_csv(tsv_path, sep="\\t")
print(f"Loaded {len(df)} records from {tsv_path}")

if limit:
    df = df.head(limit)
    print(f"Limited to {len(df)} records")

records = []
for idx, row in df.iterrows():
    # Handle different column naming conventions
    protein_id = row.get('protein_id', row.get('Entry', row.get('id', f'protein_{idx}')))
    sequence = row.get('sequence', row.get('RefSeq_nn', row.get('dna', '')))
    protein_aa = row.get('protein_aa', row.get('RefSeq_aa', row.get('aa', '')))
    host = row.get('host', row.get('Organism', 'E_coli'))
    
    # Expression - default to 1.0 if not provided
    expression = float(row.get('expression', row.get('expr', 1.0)))
    
    record = {
        'protein_id': protein_id,
        'sequence': sequence if pd.notna(sequence) else '',
        'protein_aa': protein_aa if pd.notna(protein_aa) else '',
        'host': host if pd.notna(host) else 'E_coli',
        'expression': expression,
        'metadata': {
            'promoter': row.get('promoter', 'unknown'),
            'rbs': row.get('rbs', 'unknown'),
            'subcellular_location': row.get('Subcellular location [CC]', ''),
            'conditions': {'temperature': row.get('temperature', 37.0)}
        }
    }
    records.append(record)

with open(output_path, 'w') as f:
    for record in records:
        f.write(json.dumps(record) + "\\n")

print(f"Converted {len(records)} records to JSONL")
'''
            
            # Write script to temp file
            script_path = self.working_dir / 'convert_tsv.py'
            with open(script_path, 'w') as f:
                f.write(conversion_script)
            
            # Run in Docker
            limit_str = str(limit) if limit else 'None'
            cmd = [
                'docker', 'run', '--rm',
                '-v', f'{self.project_root}:/workspace',
                '-w', '/workspace',
                'python:3.10-slim',
                'bash', '-c',
                f'pip install -q pandas && python {script_path} {tsv_path} {output_jsonl} {limit_str}'
            ]
            
            logger.info("Running TSV conversion in Docker...")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.project_root))
            
            if result.returncode != 0:
                logger.error(f"Docker conversion failed:")
                logger.error(result.stderr)
                raise RuntimeError("TSV conversion failed")
            
            logger.info(result.stdout)
            
            # Count records
            n_records = 0
            with open(output_jsonl, 'r') as f:
                n_records = sum(1 for _ in f)
            
            logger.info(f"✓ Converted {n_records} records using Docker")
            self.stats['steps_completed'].append('convert')
            
            return n_records
            
        else:
            # Local execution - requires pandas
            try:
                import pandas as pd
            except ImportError:
                raise ImportError(
                    "pandas is required for TSV conversion. "
                    "Install with: pip install pandas\n"
                    "Or use Docker mode with --use-docker"
                )
            
            df = pd.read_csv(tsv_path, sep='\t')
            logger.info(f"Loaded {len(df)} records from {tsv_path}")
            
            if limit:
                df = df.head(limit)
                logger.info(f"Limited to {len(df)} records")
            
            # Convert to JSONL
            records = []
            for idx, row in df.iterrows():
                # Handle different column naming conventions
                protein_id = row.get('protein_id', row.get('Entry', row.get('id', f'protein_{idx}')))
                sequence = row.get('sequence', row.get('RefSeq_nn', row.get('dna', '')))
                protein_aa = row.get('protein_aa', row.get('RefSeq_aa', row.get('aa', '')))
                host = row.get('host', row.get('Organism', 'E_coli'))
                
                # Expression - default to 1.0 if not provided
                expression = float(row.get('expression', row.get('expr', 1.0)))
                
                record = {
                    'protein_id': protein_id,
                    'sequence': sequence if pd.notna(sequence) else '',
                    'protein_aa': protein_aa if pd.notna(protein_aa) else '',
                    'host': host if pd.notna(host) else 'E_coli',
                    'expression': expression,
                    'metadata': {
                        'promoter': row.get('promoter', 'unknown'),
                        'rbs': row.get('rbs', 'unknown'),
                        'subcellular_location': row.get('Subcellular location [CC]', ''),
                        'conditions': {
                            'temperature': row.get('temperature', 37.0)
                        }
                    }
                }
                records.append(record)
            
            # Write JSONL
            with open(output_jsonl, 'w') as f:
                for record in records:
                    f.write(json.dumps(record) + '\n')
            
            logger.info(f"✓ Wrote {len(records)} records to {output_jsonl}")
            self.stats['steps_completed'].append('convert')
            
            return len(records)
    
    def generate_structure_features(
        self,
        input_jsonl: str,
        output_json: str,
        limit: Optional[int] = None
    ):
        """Generate structure features using lite service"""
        logger.info("\n" + "="*60)
        logger.info("Step 2: Generating Structure Features (Lite)")
        logger.info("="*60)
        
        if self.use_docker:
            # Docker command
            data_dir = self.project_root / 'data'
            cmd = [
                'docker-compose', '-f', 'docker-compose.microservices.yml',
                'run', '--rm',
                '-v', f'{data_dir}:/data',
                'structure_features_lite',
                '--input', f'/data/{Path(input_jsonl).relative_to("data")}',
                '--output', f'/data/{Path(output_json).relative_to("data")}'
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
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.project_root))
        
        if result.returncode != 0:
            logger.error(f"Structure feature generation failed:")
            logger.error(result.stderr)
            raise RuntimeError("Structure feature generation failed")
        
        logger.info(result.stdout)
        logger.info("✓ Structure features generated")
        self.stats['steps_completed'].append('structure')
    
    def generate_msa_features(
        self,
        input_jsonl: str,
        output_json: str,
        limit: Optional[int] = None
    ):
        """Generate MSA features using lite service"""
        logger.info("\n" + "="*60)
        logger.info("Step 3: Generating MSA Features (Lite)")
        logger.info("="*60)
        
        if self.use_docker:
            # Docker command
            data_dir = self.project_root / 'data'
            cmd = [
                'docker-compose', '-f', 'docker-compose.microservices.yml',
                'run', '--rm',
                '-v', f'{data_dir}:/data',
                'msa_features_lite',
                '--input', f'/data/{Path(input_jsonl).relative_to("data")}',
                '--output', f'/data/{Path(output_json).relative_to("data")}'
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
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.project_root))
        
        if result.returncode != 0:
            logger.error(f"MSA feature generation failed:")
            logger.error(result.stderr)
            raise RuntimeError("MSA feature generation failed")
        
        logger.info(result.stdout)
        logger.info("✓ MSA features generated")
        self.stats['steps_completed'].append('msa')
    
    def integrate_features(
        self,
        input_jsonl: str,
        structure_json: str,
        msa_json: str,
        output_jsonl: str,
        limit: Optional[int] = None
    ):
        """Integrate all features"""
        logger.info("\n" + "="*60)
        logger.info("Step 4: Integrating All Features")
        logger.info("="*60)
        
        if self.use_docker:
            # Docker command
            data_dir = self.project_root / 'data'
            cmd = [
                'docker-compose', '-f', 'docker-compose.microservices.yml',
                'run', '--rm',
                '-v', f'{data_dir}:/data',
                'feature_integrator',
                '--input', f'/data/{Path(input_jsonl).relative_to("data")}',
                '--structure-features', f'/data/{Path(structure_json).relative_to("data")}',
                '--msa-features', f'/data/{Path(msa_json).relative_to("data")}',
                '--output', f'/data/{Path(output_jsonl).relative_to("data")}'
            ]
            if limit:
                cmd.extend(['--limit', str(limit)])
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
            if limit:
                cmd.extend(['--limit', str(limit)])
        
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.project_root))
        
        if result.returncode != 0:
            logger.error(f"Feature integration failed:")
            logger.error(result.stderr)
            raise RuntimeError("Feature integration failed")
        
        logger.info(result.stdout)
        logger.info("✓ Features integrated")
        self.stats['steps_completed'].append('integrate')
    
    def run_full_pipeline(
        self,
        input_path: str,
        output_path: str,
        limit: Optional[int] = None
    ) -> Dict:
        """
        Run complete pipeline
        
        Args:
            input_path: Input file (TSV or JSONL)
            output_path: Output JSONL path
            limit: Optional limit on records
        
        Returns:
            Statistics dictionary
        """
        logger.info("\n" + "="*80)
        logger.info("ENHANCED FEATURE GENERATION PIPELINE")
        logger.info("="*80)
        logger.info(f"Input: {input_path}")
        logger.info(f"Output: {output_path}")
        logger.info(f"Mode: {'Docker' if self.use_docker else 'Local'}")
        if limit:
            logger.info(f"Limit: {limit} records")
        logger.info("="*80 + "\n")
        
        # Determine input format
        input_path = Path(input_path)
        is_tsv = input_path.suffix.lower() in ['.tsv', '.txt', '.csv']
        
        # Intermediate files
        if is_tsv:
            base_jsonl = self.working_dir / f"{input_path.stem}_base.jsonl"
        else:
            base_jsonl = input_path
        
        structure_json = self.working_dir / f"{input_path.stem}_structure.json"
        msa_json = self.working_dir / f"{input_path.stem}_msa.json"
        
        try:
            # Step 1: Convert TSV if needed
            if is_tsv:
                n_records = self.convert_tsv_to_jsonl(
                    str(input_path),
                    str(base_jsonl),
                    limit=limit
                )
            else:
                logger.info("Input is already JSONL, skipping conversion")
                n_records = sum(1 for _ in open(base_jsonl))
                logger.info(f"Found {n_records} records")
            
            # Step 2: Generate structure features
            self.generate_structure_features(
                str(base_jsonl),
                str(structure_json),
                limit=limit
            )
            
            # Step 3: Generate MSA features
            self.generate_msa_features(
                str(base_jsonl),
                str(msa_json),
                limit=limit
            )
            
            # Step 4: Integrate all features
            self.integrate_features(
                str(base_jsonl),
                str(structure_json),
                str(msa_json),
                output_path,
                limit=limit
            )
            
            # Final statistics
            elapsed = time.time() - self.stats['start_time']
            
            logger.info("\n" + "="*80)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("="*80)
            logger.info(f"Output file: {output_path}")
            logger.info(f"Total records: {n_records if not limit else limit}")
            logger.info(f"Steps completed: {', '.join(self.stats['steps_completed'])}")
            logger.info(f"Total time: {elapsed:.1f}s")
            logger.info(f"Average time per record: {elapsed/(n_records if not limit else limit):.2f}s")
            logger.info("="*80 + "\n")
            
            return {
                'success': True,
                'output_file': output_path,
                'n_records': n_records if not limit else limit,
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
        description="Enhanced Feature Generation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (local)
  python scripts/generate_enhanced_features.py \\
    --input data/2025_bio-os_data/dataset/Ec.tsv \\
    --output data/enhanced/Ec_with_features.jsonl

  # Test with limited records
  python scripts/generate_enhanced_features.py \\
    --input data/2025_bio-os_data/dataset/Ec.tsv \\
    --output data/enhanced/Ec_test.jsonl \\
    --limit 100

  # Using Docker (recommended for production)
  python scripts/generate_enhanced_features.py \\
    --input data/2025_bio-os_data/dataset/Ec.tsv \\
    --output data/enhanced/Ec_with_features.jsonl \\
    --use-docker
"""
    )
    
    parser.add_argument('--input', required=True,
                       help="Input file (TSV or JSONL)")
    parser.add_argument('--output', required=True,
                       help="Output JSONL file")
    parser.add_argument('--limit', type=int,
                       help="Limit number of records (for testing)")
    parser.add_argument('--working-dir', default='data/temp',
                       help="Working directory for intermediate files")
    parser.add_argument('--use-docker', action='store_true',
                       help="Use Docker services (default: local)")
    parser.add_argument('--mode', default='lite',
                       choices=['lite', 'full'],
                       help="Feature generation mode (only 'lite' implemented)")
    
    args = parser.parse_args()
    
    if args.mode != 'lite':
        logger.warning(f"Mode '{args.mode}' not yet implemented, using 'lite'")
    
    # Create output directory
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run pipeline
    pipeline = EnhancedFeaturePipeline(
        working_dir=args.working_dir,
        use_docker=args.use_docker
    )
    
    result = pipeline.run_full_pipeline(
        input_path=args.input,
        output_path=args.output,
        limit=args.limit
    )
    
    sys.exit(0 if result['success'] else 1)


if __name__ == '__main__':
    main()
