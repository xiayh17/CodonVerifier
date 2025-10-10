"""
Utility functions for the training service
"""

from typing import Dict, Any, List
from pathlib import Path


def validate_training_config(config: Dict[str, Any]) -> bool:
    """
    Validate training configuration
    
    Args:
        config: Training configuration dictionary
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = ['data_paths', 'output_path']
    
    if not all(field in config for field in required_fields):
        return False
    
    if not isinstance(config['data_paths'], list) or not config['data_paths']:
        return False
    
    return True


def create_default_config() -> Dict[str, Any]:
    """
    Create a default training configuration
    
    Returns:
        Default configuration dictionary
    """
    return {
        "task": "train_unified",
        "config": {
            "data_paths": ["/data/converted/merged_dataset.jsonl"],
            "output_path": "/data/output/models/unified_model.pkl",
            "mode": "unified",
            "data_config": {
                "max_samples_per_host": None,
                "min_sequence_length": 10,
                "max_sequence_length": 10000,
                "filter_reviewed_only": False,
                "balance_hosts": True
            },
            "surrogate_config": {
                "quantile_hi": 0.9,
                "test_size": 0.15
            },
            "target_hosts": None,
            "max_samples": None
        },
        "metadata": {
            "request_id": "training_001",
            "description": "Unified multi-host model training"
        }
    }


def format_metrics(metrics: Dict[str, Any]) -> str:
    """
    Format training metrics for display
    
    Args:
        metrics: Metrics dictionary
        
    Returns:
        Formatted string
    """
    lines = []
    lines.append("Training Metrics:")
    lines.append("-" * 50)
    
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"  {key}: {value:.4f}")
        else:
            lines.append(f"  {key}: {value}")
    
    return "\n".join(lines)

