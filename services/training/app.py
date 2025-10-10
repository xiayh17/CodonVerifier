#!/usr/bin/env python3
"""
Training Service - Surrogate Model Training Microservice

This service handles model training tasks in the microservices architecture.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('training-service')

# Import training modules
sys.path.insert(0, '/app')
from codon_verifier.train_surrogate_multihost import (
    train_unified_model,
    train_host_specific_models
)
from codon_verifier.data_loader import DataConfig
from codon_verifier.surrogate import SurrogateConfig


def process_training_task(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process a training task"""
    start_time = time.time()
    
    try:
        task_type = task_data.get('task', 'train_unified')
        config = task_data.get('config', {})
        
        logger.info(f"Processing training task: {task_type}")
        logger.info(f"Configuration: {json.dumps(config, indent=2)}")
        
        # Extract configuration
        data_paths = config.get('data_paths', [])
        output_path = config.get('output_path', '/data/output/models/model.pkl')
        mode = config.get('mode', 'unified')
        
        # Data configuration
        data_config_dict = config.get('data_config', {})
        data_config = DataConfig(
            max_samples_per_host=data_config_dict.get('max_samples_per_host'),
            min_sequence_length=data_config_dict.get('min_sequence_length', 10),
            max_sequence_length=data_config_dict.get('max_sequence_length', 10000),
            filter_reviewed_only=data_config_dict.get('filter_reviewed_only', False),
            balance_hosts=data_config_dict.get('balance_hosts', False),
        )
        
        # Surrogate configuration
        surrogate_config_dict = config.get('surrogate_config', {})
        surrogate_config = SurrogateConfig(
            quantile_hi=surrogate_config_dict.get('quantile_hi', 0.9),
            test_size=surrogate_config_dict.get('test_size', 0.15),
        )
        
        # Training parameters
        target_hosts = config.get('target_hosts')
        max_samples = config.get('max_samples')
        
        # Validate data paths
        for path in data_paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"Data file not found: {path}")
        
        # Train model based on mode
        if mode == 'unified':
            logger.info("Training unified multi-host model...")
            metrics = train_unified_model(
                data_paths=data_paths,
                output_model_path=output_path,
                data_config=data_config,
                surrogate_config=surrogate_config,
                target_hosts=target_hosts,
                max_samples=max_samples
            )
        elif mode == 'host-specific':
            logger.info("Training host-specific models...")
            output_dir = str(Path(output_path).parent)
            metrics = train_host_specific_models(
                data_paths=data_paths,
                output_dir=output_dir,
                data_config=data_config,
                surrogate_config=surrogate_config,
                target_hosts=target_hosts
            )
        else:
            raise ValueError(f"Unknown training mode: {mode}")
        
        # Prepare result
        result = {
            "task": task_type,
            "status": "success",
            "output": {
                "mode": mode,
                "model_path": output_path if mode == 'unified' else output_dir,
                "metrics": metrics,
            },
            "metadata": {
                "request_id": task_data.get('metadata', {}).get('request_id', 'unknown'),
                "processing_time_seconds": round(time.time() - start_time, 2),
                "service": "training",
                "version": "1.0.0"
            }
        }
        
        logger.info(f"✓ Training completed successfully in {result['metadata']['processing_time_seconds']}s")
        return result
        
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        return {
            "task": task_data.get('task', 'unknown'),
            "status": "error",
            "error": str(e),
            "metadata": {
                "request_id": task_data.get('metadata', {}).get('request_id', 'unknown'),
                "processing_time_seconds": round(time.time() - start_time, 2),
                "service": "training",
                "version": "1.0.0"
            }
        }


def main():
    parser = argparse.ArgumentParser(description='Training Service - Surrogate Model Training')
    parser.add_argument('--input', type=str, required=True,
                       help='Input JSON file with training configuration')
    parser.add_argument('--output', type=str,
                       help='Output JSON file path (default: auto-generated in /data/output/)')
    
    args = parser.parse_args()
    
    # Read input configuration
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input configuration file not found: {input_path}")
        sys.exit(1)
    
    logger.info(f"Reading training configuration from: {input_path}")
    
    with open(input_path, 'r') as f:
        task_data = json.load(f)
    
    # Process training task
    result = process_training_task(task_data)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = Path('/data/output/training')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{input_path.stem}_result.json"
    
    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"✓ Training results written to: {output_path}")
    
    # Exit with appropriate code
    sys.exit(0 if result['status'] == 'success' else 1)


if __name__ == '__main__':
    main()

