#!/usr/bin/env python3
"""
Train Enhanced Model with All Features

Integrates:
1. Enhanced structural features (structure_features_lite)
2. Enhanced evolutionary features (msa_features_lite)
3. Enhanced contextual features (feature_integrator)
4. Evo2 sequence quality features
5. Original codon optimization features (90-dim)

Supports:
- Deep Ensembles
- Conformal Prediction
- Multi-host training
- Transfer learning

Author: CodonVerifier Team
Date: 2025-10-05
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import subprocess
import os

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent))

from codon_verifier.surrogate import (
    build_feature_vector,
    SurrogateModel,
    SurrogateConfig
)
from codon_verifier.hosts.tables import HOST_TABLES
from codon_verifier.model_ensemble import (
    DeepEnsemble,
    ConformalPredictor,
    EnsembleConfig,
    ConformalConfig,
    train_ensemble_with_conformal
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedFeatureExtractor:
    """Extract and combine all types of features"""
    
    def __init__(self, host: str = "E_coli", project_root: Optional[Path] = None):
        self.host = host
        
        # Get codon tables for host
        if host not in HOST_TABLES:
            raise ValueError(f"Unknown host: {host}")
        self.usage, self.trna_w = HOST_TABLES[host]
        
        # Save project root directory (absolute path)
        if project_root is None:
            # Assume script is in scripts/ directory
            self.project_root = Path(__file__).parent.parent.resolve()
        else:
            self.project_root = Path(project_root).resolve()
    
    def extract_evo2_features_if_needed(
        self,
        input_jsonl: str,
        output_json: str,
        use_docker: bool = False
    ) -> bool:
        """
        Extract Evo2 features if not already present
        
        Returns:
            True if successful or features already exist
        """
        output_path = Path(output_json)
        
        # Check if already exists
        if output_path.exists():
            logger.info(f"Evo2 features already exist: {output_json}")
            return True
        
        logger.info("Extracting Evo2 features...")
        
        try:
            if use_docker:
                # Use Docker Evo2 service
                data_dir = self.project_root / 'data'
                cmd = [
                    'docker-compose', '-f', 'docker-compose.microservices.yml',
                    'run', '--rm',
                    '-v', f'{data_dir}:/data',
                    'evo2',
                    '--input', f'/data/{Path(input_jsonl).relative_to("data")}',
                    '--output', f'/data/{Path(output_json).relative_to("data")}',
                    '--mode', 'features'
                ]
            else:
                # Use local Evo2 script
                cmd = [
                    sys.executable,
                    'services/evo2/app_enhanced.py',
                    '--input', input_jsonl,
                    '--output', output_json,
                    '--mode', 'features'
                ]
            
            logger.info(f"Running: {' '.join(cmd[:8])}...")
            logger.info("⏱️  This may take a long time for large datasets (estimated: ~1s per sequence)")
            
            # Run with real-time output for progress monitoring
            # Increase timeout for large datasets (7200s = 2 hours)
            import signal
            try:
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
                
                # Wait with timeout
                try:
                    returncode = process.wait(timeout=7200)
                except subprocess.TimeoutExpired:
                    process.kill()
                    logger.error("Evo2 extraction timed out after 2 hours")
                    return False
                
                if returncode != 0:
                    logger.error(f"Evo2 extraction failed with code {returncode}")
                    return False
                
                logger.info("✓ Evo2 features extracted")
                return True
            except Exception as e:
                logger.error(f"Error during Evo2 extraction: {e}")
                return False
            
        except Exception as e:
            logger.error(f"Failed to extract Evo2 features: {e}")
            return False
    
    def load_evo2_features(self, evo2_json: str, records: Optional[List[Dict]] = None) -> Dict[str, Dict]:
        """Load Evo2 features from JSON file, supporting multiple schemas.

        Supports:
          - Schema A: [{"protein_id": str, "features": {...}}]
          - Schema B (evo2 service results): [{"status": "success", "output": {...}}]
            In this case, if `records` are provided, align by index and map to protein_id.
        """
        evo2_features_by_protein: Dict[str, Dict] = {}

        if not Path(evo2_json).exists():
            logger.warning(f"Evo2 features file not found: {evo2_json}")
            return evo2_features_by_protein

        try:
            with open(evo2_json, 'r') as f:
                data = json.load(f)

            if not isinstance(data, list) or len(data) == 0:
                logger.warning("Evo2 features file is empty or not a list")
                return evo2_features_by_protein

            first = data[0] if isinstance(data, list) else {}

            # Schema A: items contain protein_id and nested features
            if isinstance(first, dict) and ('protein_id' in first or 'features' in first):
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    protein_id = item.get('protein_id')
                    # features may be nested or at top-level in some variants
                    features = item.get('features') if 'features' in item else {
                        k: v for k, v in item.items() if k not in ('protein_id',)
                    }
                    if protein_id and isinstance(features, dict):
                        evo2_features_by_protein[protein_id] = features
                logger.info(f"Loaded Evo2 features for {len(evo2_features_by_protein)} proteins (schema A)")

            # Schema B: evo2 service results list
            elif isinstance(first, dict) and ('status' in first and 'output' in first):
                if records is None:
                    logger.warning("Evo2 result schema detected but no records provided for alignment; skipping Evo2 features")
                    return evo2_features_by_protein

                aligned_count = 0
                mismatch_count = 0
                # Align by index position to records
                for idx, result in enumerate(data):
                    if not isinstance(result, dict):
                        mismatch_count += 1
                        continue
                    if result.get('status') != 'success':
                        mismatch_count += 1
                        continue
                    output = result.get('output', {})
                    if not isinstance(output, dict):
                        mismatch_count += 1
                        continue
                    if idx >= len(records):
                        mismatch_count += 1
                        continue
                    protein_id = records[idx].get('protein_id')
                    if not protein_id:
                        mismatch_count += 1
                        continue

                    # Extract only the Evo2 feature fields we actually use downstream
                    selected = {}
                    for fname in (
                        'avg_confidence', 'max_confidence', 'min_confidence', 'std_confidence',
                        'avg_loglik', 'perplexity', 'gc_content', 'codon_entropy'
                    ):
                        if fname in output:
                            try:
                                selected[fname] = float(output[fname])
                            except Exception:
                                pass

                    if selected:
                        evo2_features_by_protein[protein_id] = selected
                        aligned_count += 1
                    else:
                        mismatch_count += 1

                logger.info(
                    f"Loaded Evo2 features for {aligned_count} proteins (schema B), mismatches: {mismatch_count}"
                )

            else:
                logger.warning("Unrecognized Evo2 features schema; no features loaded")

        except Exception as e:
            logger.error(f"Failed to load Evo2 features: {e}")

        return evo2_features_by_protein
    
    def build_combined_feature_vector(
        self,
        record: Dict,
        evo2_features: Optional[Dict] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Build combined feature vector from all sources
        
        Args:
            record: JSONL record with sequence, extra_features, etc.
            evo2_features: Dictionary of Evo2 features keyed by protein_id
        
        Returns:
            (feature_vector, feature_names)
        """
        dna = record.get('sequence', '')
        protein_id = record.get('protein_id', '')
        
        # 1. Base codon features (90-dim)
        extra_features = record.get('extra_features', {})
        base_vec, base_keys = build_feature_vector(
            dna, 
            self.usage, 
            trna_w=self.trna_w,
            extra_features=extra_features
        )
        
        # 2. Evo2 features (if available)
        evo2_vec = []
        evo2_keys = []
        
        if evo2_features and protein_id in evo2_features:
            evo2_data = evo2_features[protein_id]
            
            # Extract key Evo2 features
            evo2_feature_names = [
                'avg_confidence',
                'max_confidence',
                'min_confidence',
                'std_confidence',
                'avg_loglik',
                'perplexity',
                'gc_content',
                'codon_entropy'
            ]
            
            for fname in evo2_feature_names:
                value = evo2_data.get(fname, 0.0)
                evo2_vec.append(float(value))
                evo2_keys.append(f'evo2_{fname}')
        
        # Combine all features
        if evo2_vec:
            combined_vec = np.concatenate([base_vec, np.array(evo2_vec)])
            combined_keys = base_keys + evo2_keys
        else:
            combined_vec = base_vec
            combined_keys = base_keys
        
        return combined_vec, combined_keys
    
    def load_and_prepare_data(
        self,
        jsonl_path: str,
        evo2_json: Optional[str] = None,
        max_samples: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[Dict]]:
        """
        Load data from JSONL and prepare feature matrix
        
        Returns:
            (X, y, feature_keys, records)
        """
        logger.info(f"Loading data from {jsonl_path}...")
        
        # Load records first (needed for aligning Evo2 results if required)
        records = []
        with open(jsonl_path, 'r') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                
                try:
                    record = json.loads(line.strip())
                    records.append(record)
                except Exception as e:
                    logger.warning(f"Failed to parse line {i}: {e}")
        
        logger.info(f"Loaded {len(records)} records")
        
        # Load Evo2 features if provided (support multiple schemas; align by index to records if needed)
        evo2_features = {}
        if evo2_json:
            evo2_features = self.load_evo2_features(evo2_json, records=records)
        
        # Build feature matrix
        X_list = []
        y_list = []
        feature_keys = None
        valid_records = []
        
        for i, record in enumerate(records):
            try:
                # Check for required fields
                dna = record.get('sequence', '')
                if not dna or len(dna) < 30:
                    continue
                
                # Get expression value
                expr = record.get('expression', {})
                if isinstance(expr, dict):
                    y_val = float(expr.get('value', 0))
                else:
                    y_val = float(expr)
                
                if y_val <= 0:
                    continue
                
                # Build feature vector
                vec, keys = self.build_combined_feature_vector(record, evo2_features)
                
                X_list.append(vec)
                y_list.append(y_val)
                valid_records.append(record)
                
                if feature_keys is None:
                    feature_keys = keys
                
                if (i + 1) % 1000 == 0:
                    logger.info(f"Processed {i + 1}/{len(records)} records...")
            
            except Exception as e:
                logger.warning(f"Failed to process record {i}: {e}")
                continue
        
        if not X_list:
            raise ValueError("No valid records found!")
        
        X = np.vstack(X_list)
        y = np.array(y_list)
        
        logger.info(f"✓ Prepared {len(y)} valid samples")
        logger.info(f"✓ Total features: {len(feature_keys)}")
        
        # Feature breakdown
        struct_count = sum(1 for k in feature_keys if k.startswith('struct_'))
        evo_count = sum(1 for k in feature_keys if k.startswith('evo_'))
        ctx_count = sum(1 for k in feature_keys if k.startswith('ctx_'))
        evo2_count = sum(1 for k in feature_keys if k.startswith('evo2_'))
        base_count = len(feature_keys) - struct_count - evo_count - ctx_count - evo2_count
        
        logger.info(f"  Feature breakdown:")
        logger.info(f"    - Base codon features: {base_count}")
        logger.info(f"    - Structural features: {struct_count}")
        logger.info(f"    - Evolutionary features: {evo_count}")
        logger.info(f"    - Contextual features: {ctx_count}")
        logger.info(f"    - Evo2 features: {evo2_count}")
        
        return X, y, feature_keys, valid_records


def train_basic_model(
    X: np.ndarray,
    y: np.ndarray,
    feature_keys: List[str],
    output_path: str,
    config: Optional[SurrogateConfig] = None
) -> Dict[str, Any]:
    """Train a basic surrogate model"""
    logger.info("\n" + "="*60)
    logger.info("Training Basic Surrogate Model")
    logger.info("="*60)
    
    if config is None:
        config = SurrogateConfig()
    
    model = SurrogateModel(feature_keys=feature_keys, cfg=config)
    metrics = model.fit(X, y)
    
    # Save model
    logger.info(f"Saving model to {output_path}")
    model.save(output_path)
    
    metrics['model_path'] = output_path
    metrics['n_samples'] = int(len(y))
    metrics['n_features'] = int(X.shape[1])
    
    return metrics


def train_ensemble_model(
    X: np.ndarray,
    y: np.ndarray,
    feature_keys: List[str],
    output_path: str,
    n_models: int = 5,
    config: Optional[SurrogateConfig] = None
) -> Dict[str, Any]:
    """Train a deep ensemble"""
    logger.info("\n" + "="*60)
    logger.info(f"Training Deep Ensemble ({n_models} models)")
    logger.info("="*60)
    
    if config is None:
        config = SurrogateConfig()
    
    ensemble_cfg = EnsembleConfig(
        n_models=n_models,
        surrogate_config=config
    )
    
    ensemble = DeepEnsemble(
        feature_keys=feature_keys,
        cfg=ensemble_cfg
    )
    
    metrics = ensemble.train(X, y)
    
    # Save ensemble
    logger.info(f"Saving ensemble to {output_path}")
    ensemble.save(output_path)
    
    metrics['model_path'] = output_path
    metrics['n_samples'] = int(len(y))
    metrics['n_features'] = int(X.shape[1])
    metrics['n_ensemble_models'] = n_models
    
    return metrics


def train_ensemble_with_conformal_wrapper(
    X: np.ndarray,
    y: np.ndarray,
    feature_keys: List[str],
    output_dir: str,
    n_models: int = 5,
    config: Optional[SurrogateConfig] = None
) -> Dict[str, Any]:
    """Train ensemble + conformal prediction"""
    logger.info("\n" + "="*60)
    logger.info(f"Training Deep Ensemble + Conformal Prediction")
    logger.info("="*60)
    
    if config is None:
        config = SurrogateConfig()
    
    ensemble_cfg = EnsembleConfig(
        n_models=n_models,
        surrogate_config=config
    )
    
    conformal_cfg = ConformalConfig(
        alpha=0.1  # 90% prediction intervals
    )
    
    # Train
    ensemble, conformal, metrics = train_ensemble_with_conformal(
        X, y, feature_keys,
        ensemble_cfg=ensemble_cfg,
        conformal_cfg=conformal_cfg,
        cal_ratio=0.2
    )
    
    # Save both
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ensemble_path = output_dir / 'ensemble.pkl'
    conformal_path = output_dir / 'conformal.pkl'
    
    logger.info(f"Saving ensemble to {ensemble_path}")
    ensemble.save(str(ensemble_path))
    
    logger.info(f"Saving conformal predictor to {conformal_path}")
    # Use joblib for conformal predictor
    import joblib
    joblib.dump(conformal, str(conformal_path))
    
    metrics['ensemble_path'] = str(ensemble_path)
    metrics['conformal_path'] = str(conformal_path)
    metrics['n_samples'] = int(len(y))
    metrics['n_features'] = int(X.shape[1])
    metrics['n_ensemble_models'] = n_models
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train Enhanced Model with All Features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Basic model with enhanced features
  python scripts/train_with_all_features.py \\
    --data data/test/Ec_10_v2.jsonl \\
    --output models/enhanced/ecoli_basic.pkl \\
    --host E_coli

  # Add Evo2 features
  python scripts/train_with_all_features.py \\
    --data data/enhanced/Ec_full.jsonl \\
    --evo2-features data/enhanced/Ec_evo2_features.json \\
    --output models/enhanced/ecoli_with_evo2.pkl

  # Deep Ensemble
  python scripts/train_with_all_features.py \\
    --data data/enhanced/Ec_full.jsonl \\
    --evo2-features data/enhanced/Ec_evo2_features.json \\
    --output models/enhanced/ecoli_ensemble.pkl \\
    --use-ensemble \\
    --n-models 5

  # Ensemble + Conformal Prediction
  python scripts/train_with_all_features.py \\
    --data data/enhanced/Ec_full.jsonl \\
    --evo2-features data/enhanced/Ec_evo2_features.json \\
    --output-dir models/enhanced/ecoli_full \\
    --use-ensemble \\
    --use-conformal \\
    --n-models 5

  # Extract Evo2 features on-the-fly
  python scripts/train_with_all_features.py \\
    --data data/enhanced/Ec_full.jsonl \\
    --extract-evo2 \\
    --output models/enhanced/ecoli_auto.pkl
        """
    )
    
    # Data
    parser.add_argument('--data', required=True,
                       help="Path to enhanced features JSONL file")
    parser.add_argument('--host', default='E_coli',
                       help="Host organism (default: E_coli)")
    parser.add_argument('--max-samples', type=int,
                       help="Maximum samples to use (for testing)")
    
    # Evo2 features
    parser.add_argument('--evo2-features',
                       help="Path to Evo2 features JSON file")
    parser.add_argument('--extract-evo2', action='store_true',
                       help="Extract Evo2 features automatically")
    parser.add_argument('--use-docker', action='store_true',
                       help="Use Docker for Evo2 extraction")
    
    # Output
    parser.add_argument('--output',
                       help="Output model path (for basic/ensemble)")
    parser.add_argument('--output-dir',
                       help="Output directory (for ensemble+conformal)")
    parser.add_argument('--output-metrics',
                       help="Output metrics JSON path")
    
    # Model type
    parser.add_argument('--use-ensemble', action='store_true',
                       help="Train deep ensemble")
    parser.add_argument('--use-conformal', action='store_true',
                       help="Use conformal prediction (requires --use-ensemble)")
    parser.add_argument('--n-models', type=int, default=5,
                       help="Number of models in ensemble (default: 5)")
    
    # Model configuration
    parser.add_argument('--n-estimators', type=int, default=400,
                       help="Number of boosting rounds (default: 400)")
    parser.add_argument('--learning-rate', type=float, default=0.05,
                       help="Learning rate (default: 0.05)")
    parser.add_argument('--max-depth', type=int, default=5,
                       help="Maximum tree depth (default: 5)")
    parser.add_argument('--use-log-transform', action='store_true',
                       help="Use log1p transform for targets")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.use_conformal and not args.use_ensemble:
        parser.error("--use-conformal requires --use-ensemble")
    
    if args.use_conformal and not args.output_dir:
        parser.error("--use-conformal requires --output-dir")
    
    if not args.use_conformal and not args.output:
        parser.error("Either --output or --output-dir (with --use-conformal) is required")
    
    try:
        start_time = time.time()
        
        logger.info("="*60)
        logger.info("ENHANCED MODEL TRAINING PIPELINE")
        logger.info("="*60)
        logger.info(f"Data: {args.data}")
        logger.info(f"Host: {args.host}")
        logger.info(f"Model type: {'Ensemble+Conformal' if args.use_conformal else 'Ensemble' if args.use_ensemble else 'Basic'}")
        logger.info("="*60)
        
        # Initialize feature extractor
        extractor = EnhancedFeatureExtractor(host=args.host)
        
        # Extract Evo2 features if requested
        evo2_json = args.evo2_features
        if args.extract_evo2:
            evo2_json = str(Path(args.data).parent / f"{Path(args.data).stem}_evo2_features.json")
            success = extractor.extract_evo2_features_if_needed(
                args.data,
                evo2_json,
                use_docker=args.use_docker
            )
            if not success:
                logger.warning("Failed to extract Evo2 features, proceeding without them")
                evo2_json = None
        
        # Load and prepare data
        X, y, feature_keys, records = extractor.load_and_prepare_data(
            args.data,
            evo2_json=evo2_json,
            max_samples=args.max_samples
        )
        
        # Configure model
        config = SurrogateConfig(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            use_log_transform=args.use_log_transform
        )
        
        # Train model
        if args.use_conformal:
            metrics = train_ensemble_with_conformal_wrapper(
                X, y, feature_keys,
                output_dir=args.output_dir,
                n_models=args.n_models,
                config=config
            )
        elif args.use_ensemble:
            metrics = train_ensemble_model(
                X, y, feature_keys,
                output_path=args.output,
                n_models=args.n_models,
                config=config
            )
        else:
            metrics = train_basic_model(
                X, y, feature_keys,
                output_path=args.output,
                config=config
            )
        
        # Add timing
        metrics['total_time_s'] = time.time() - start_time
        metrics['host'] = args.host
        
        # Save metrics
        if args.output_metrics:
            logger.info(f"\nSaving metrics to {args.output_metrics}")
            with open(args.output_metrics, 'w') as f:
                json.dump(metrics, f, indent=2)
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info(f"Samples: {metrics['n_samples']}")
        logger.info(f"Features: {metrics['n_features']}")
        logger.info(f"R² (mean): {metrics.get('r2_mu', 'N/A')}")
        logger.info(f"MAE (mean): {metrics.get('mae_mu', 'N/A')}")
        logger.info(f"Total time: {metrics['total_time_s']:.1f}s")
        if args.use_conformal:
            logger.info(f"Output directory: {args.output_dir}")
        else:
            logger.info(f"Model path: {args.output}")
        logger.info("="*60)
        
        sys.exit(0)
    
    except Exception as e:
        logger.error(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
