"""Utility functions for Evo2 service"""

import json
from pathlib import Path
from typing import Dict, Any, List


def validate_input(data: Dict[str, Any]) -> bool:
    """Validate input JSON structure"""
    required_fields = ['task', 'input']
    return all(field in data for field in required_fields)


def load_batch(input_file: Path) -> List[Dict[str, Any]]:
    """Load batch of tasks from file"""
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    else:
        return [data]


def save_results(results: List[Dict[str, Any]], output_file: Path):
    """Save results to JSON file"""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

