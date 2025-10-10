"""
Enhanced surrogate training with multi-host support and data mixing strategies.

This script extends the basic surrogate training to:
1. Support multiple host organisms
2. Train host-specific or unified models
3. Use advanced data loading and mixing strategies
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import logging

from codon_verifier.surrogate import (
    build_feature_vector,
    SurrogateModel,
    SurrogateConfig,
)
from codon_verifier.hosts.tables import get_host_tables, HOST_TABLES
from codon_verifier.data_loader import DataLoader, DataConfig, create_train_val_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def build_dataset_multihost(
    records: List[dict],
    host_tables: Dict[str, tuple]
) -> tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build dataset with proper host-specific codon usage tables.
    
    Args:
        records: List of data records
        host_tables: Dictionary mapping host names to (usage, trna) tuples
    
    Returns:
        Tuple of (X, y, feature_keys)
    """
    X = []
    y = []
    feat_keys = None
    
    for i, record in enumerate(records):
        try:
            dna = record["sequence"]
            host = record.get("host", "E_coli")
            extra = record.get("extra_features")
            
            # Get host-specific tables
            if host not in host_tables:
                logger.warning(f"Unknown host {host} in record {i}, using E_coli")
                host = "E_coli"
            
            usage, trna_w = host_tables[host]
            
            # Build features
            vec, keys = build_feature_vector(dna, usage, trna_w=trna_w, extra_features=extra)
            X.append(vec)
            
            if feat_keys is None:
                feat_keys = keys
            
            # Extract expression value
            expr = record.get("expression", {})
            if isinstance(expr, dict):
                y_val = float(expr.get("value", 0))
            else:
                y_val = float(expr)
            
            y.append(y_val)
            
        except Exception as e:
            logger.error(f"Error processing record {i}: {e}")
            continue
    
    if not X:
        raise ValueError("No valid records processed")
    
    X = np.vstack(X)
    y = np.array(y, dtype=float)
    
    logger.info(f"Built dataset: X shape {X.shape}, y shape {y.shape}")
    
    return X, y, feat_keys


def train_unified_model(
    data_paths: List[str],
    output_model_path: str,
    data_config: Optional[DataConfig] = None,
    surrogate_config: Optional[SurrogateConfig] = None,
    target_hosts: Optional[List[str]] = None,
    max_samples: Optional[int] = None
) -> Dict[str, Any]:
    """
    Train a unified model across multiple hosts.
    
    Args:
        data_paths: List of JSONL file paths
        output_model_path: Path to save the trained model
        data_config: Configuration for data loading
        surrogate_config: Configuration for surrogate model
        target_hosts: Optional list of hosts to include
        max_samples: Optional maximum number of samples
    
    Returns:
        Training metrics
    """
    # Load and mix data
    logger.info("Loading and mixing multi-host data...")
    loader = DataLoader(data_config)
    
    target_host_set = set(target_hosts) if target_hosts else None
    records = loader.load_and_mix(data_paths, target_hosts=target_host_set, total_samples=max_samples)
    
    if not records:
        raise ValueError("No records loaded")
    
    logger.info(f"Loaded {len(records)} records")
    
    # Prepare host tables
    host_tables = HOST_TABLES
    
    # Build dataset
    logger.info("Building feature dataset...")
    X, y, feat_keys = build_dataset_multihost(records, host_tables)
    
    # Train model
    logger.info("Training surrogate model...")
    model = SurrogateModel(feature_keys=feat_keys, cfg=surrogate_config)
    metrics = model.fit(X, y)
    
    # Save model
    logger.info(f"Saving model to {output_model_path}")
    model.save(output_model_path)
    
    # Add metadata
    metrics["model_path"] = output_model_path
    metrics["n_samples"] = int(len(y))
    metrics["n_features"] = int(X.shape[1])
    
    # Host distribution
    from collections import Counter
    host_dist = Counter(r.get("host", "unknown") for r in records)
    metrics["host_distribution"] = dict(host_dist)
    
    return metrics


def train_host_specific_models(
    data_paths: List[str],
    output_dir: str,
    data_config: Optional[DataConfig] = None,
    surrogate_config: Optional[SurrogateConfig] = None,
    target_hosts: Optional[List[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Train separate models for each host organism.
    
    Args:
        data_paths: List of JSONL file paths
        output_dir: Directory to save trained models
        data_config: Configuration for data loading
        surrogate_config: Configuration for surrogate model
        target_hosts: Optional list of hosts to train models for
    
    Returns:
        Dictionary mapping host to training metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    logger.info("Loading multi-host data...")
    loader = DataLoader(data_config)
    
    target_host_set = set(target_hosts) if target_hosts else None
    host_data = loader.load_multi_host(data_paths, target_hosts=target_host_set)
    
    if not host_data:
        raise ValueError("No data loaded")
    
    # Train model for each host
    all_metrics = {}
    host_tables = HOST_TABLES
    
    for host, records in host_data.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Training model for {host} ({len(records)} samples)")
        logger.info(f"{'='*60}")
        
        try:
            # Get host-specific tables
            if host not in host_tables:
                logger.warning(f"No codon table for {host}, skipping")
                continue
            
            usage, trna_w = host_tables[host]
            
            # Build dataset
            X = []
            y = []
            feat_keys = None
            
            for record in records:
                dna = record["sequence"]
                extra = record.get("extra_features")
                vec, keys = build_feature_vector(dna, usage, trna_w=trna_w, extra_features=extra)
                X.append(vec)
                if feat_keys is None:
                    feat_keys = keys
                
                expr = record.get("expression", {})
                y_val = float(expr.get("value", 0) if isinstance(expr, dict) else expr)
                y.append(y_val)
            
            X = np.vstack(X)
            y = np.array(y, dtype=float)
            
            # Train model
            model = SurrogateModel(feature_keys=feat_keys, cfg=surrogate_config)
            metrics = model.fit(X, y)
            
            # Save model
            output_path = os.path.join(output_dir, f"{host}_surrogate.pkl")
            model.save(output_path)
            
            # Record metrics
            metrics["model_path"] = output_path
            metrics["n_samples"] = int(len(y))
            metrics["host"] = host
            all_metrics[host] = metrics
            
            logger.info(f"Model saved: {output_path}")
            logger.info(f"Metrics: {json.dumps(metrics, indent=2)}")
            
        except Exception as e:
            logger.error(f"Failed to train model for {host}: {e}")
            continue
    
    return all_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train surrogate models with multi-host support"
    )
    
    # Input/output
    parser.add_argument(
        "--data",
        nargs="+",
        required=True,
        help="Path(s) to JSONL dataset file(s)"
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output path (file for unified model, directory for host-specific)"
    )
    
    # Training mode
    parser.add_argument(
        "--mode",
        choices=["unified", "host-specific"],
        default="unified",
        help="Training mode: unified model or host-specific models"
    )
    
    # Host selection
    parser.add_argument(
        "--hosts",
        nargs="+",
        default=None,
        help="Specific hosts to include (default: all)"
    )
    
    # Data configuration
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum total samples to use"
    )
    parser.add_argument(
        "--max-per-host",
        type=int,
        default=None,
        help="Maximum samples per host"
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=30,
        help="Minimum sequence length"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=3000,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--reviewed-only",
        action="store_true",
        help="Only use reviewed (SwissProt) entries"
    )
    parser.add_argument(
        "--balance-hosts",
        action="store_true",
        help="Balance samples across hosts"
    )
    
    # Model configuration
    parser.add_argument(
        "--quantile-hi",
        type=float,
        default=0.84,
        help="Upper quantile for uncertainty estimation"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="Test set fraction"
    )
    
    args = parser.parse_args()
    
    # Build configurations
    data_config = DataConfig(
        max_samples_per_host=args.max_per_host,
        min_sequence_length=args.min_length,
        max_sequence_length=args.max_length,
        filter_reviewed_only=args.reviewed_only,
        balance_hosts=args.balance_hosts,
    )
    
    surrogate_config = SurrogateConfig(
        quantile_hi=args.quantile_hi,
        test_size=args.test_size,
    )
    
    # Train models
    if args.mode == "unified":
        logger.info("Training unified multi-host model...")
        metrics = train_unified_model(
            args.data,
            args.out,
            data_config=data_config,
            surrogate_config=surrogate_config,
            target_hosts=args.hosts,
            max_samples=args.max_samples
        )
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(json.dumps(metrics, indent=2))
        
    else:  # host-specific
        logger.info("Training host-specific models...")
        all_metrics = train_host_specific_models(
            args.data,
            args.out,
            data_config=data_config,
            surrogate_config=surrogate_config,
            target_hosts=args.hosts
        )
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        for host, metrics in all_metrics.items():
            print(f"\n{host}:")
            print(json.dumps(metrics, indent=2))
    
    return 0


if __name__ == "__main__":
    exit(main())

