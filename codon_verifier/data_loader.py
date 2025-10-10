"""
Advanced data loading and mixing strategies for codon optimization training.

This module provides:
- Multi-host data loading
- Smart data mixing and augmentation
- Stratified sampling by organism
- Data quality filtering
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for data loading and mixing."""
    # Sampling strategy
    max_samples_per_host: Optional[int] = None
    min_sequence_length: int = 30
    max_sequence_length: int = 3000
    
    # Quality filters
    filter_reviewed_only: bool = False
    min_expression_value: Optional[float] = None
    exclude_low_confidence: bool = True
    
    # Data augmentation
    augment_reverse_complement: bool = False
    augment_synonym_swap: float = 0.0  # Probability of swapping synonymous codons
    
    # Mixing strategy
    balance_hosts: bool = True
    host_weights: Optional[Dict[str, float]] = None
    
    # Random seed
    random_seed: int = 42


class DataLoader:
    """
    Smart data loader with multi-host support and mixing strategies.
    """
    
    def __init__(self, config: Optional[DataConfig] = None):
        self.config = config or DataConfig()
        random.seed(self.config.random_seed)
    
    def load_jsonl(self, path: str) -> List[Dict]:
        """Load JSONL file."""
        records = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line: {e}")
        return records
    
    def filter_record(self, record: Dict) -> bool:
        """
        Apply quality filters to a record.
        
        Returns True if record should be kept.
        """
        # Sequence length check
        seq_len = len(record.get("sequence", ""))
        if seq_len < self.config.min_sequence_length:
            return False
        if seq_len > self.config.max_sequence_length:
            return False
        
        # Reviewed status
        if self.config.filter_reviewed_only:
            if not record.get("extra_features", {}).get("reviewed", False):
                return False
        
        # Expression value
        if self.config.min_expression_value is not None:
            expr = record.get("expression", {})
            if isinstance(expr, dict):
                value = expr.get("value", 0)
            else:
                value = expr
            if value < self.config.min_expression_value:
                return False
        
        # Confidence level
        if self.config.exclude_low_confidence:
            expr = record.get("expression", {})
            if isinstance(expr, dict):
                confidence = expr.get("confidence", "high")
                if confidence == "low":
                    return False
        
        return True
    
    def load_multi_host(
        self,
        file_paths: List[str],
        target_hosts: Optional[Set[str]] = None
    ) -> Dict[str, List[Dict]]:
        """
        Load multiple JSONL files and organize by host.
        
        Args:
            file_paths: List of JSONL file paths
            target_hosts: If provided, only load these hosts
        
        Returns:
            Dictionary mapping host name to list of records
        """
        host_data = {}
        
        for path in file_paths:
            logger.info(f"Loading {path}...")
            records = self.load_jsonl(path)
            
            for record in records:
                # Apply filters
                if not self.filter_record(record):
                    continue
                
                host = record.get("host", "unknown")
                
                # Check target hosts
                if target_hosts and host not in target_hosts:
                    continue
                
                if host not in host_data:
                    host_data[host] = []
                
                host_data[host].append(record)
        
        # Log statistics
        for host, records in host_data.items():
            logger.info(f"  {host}: {len(records)} records")
        
        return host_data
    
    def sample_balanced(
        self,
        host_data: Dict[str, List[Dict]],
        total_samples: Optional[int] = None
    ) -> List[Dict]:
        """
        Sample data with balanced representation from each host.
        
        Args:
            host_data: Dictionary mapping host to records
            total_samples: Total number of samples to return
        
        Returns:
            List of sampled records
        """
        if not host_data:
            return []
        
        hosts = list(host_data.keys())
        
        # Apply host weights if provided
        if self.config.host_weights:
            weights = [self.config.host_weights.get(h, 1.0) for h in hosts]
        else:
            weights = [1.0] * len(hosts)
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Determine samples per host
        if total_samples is None:
            # Use all data, but balance by taking min across hosts
            if self.config.balance_hosts:
                min_count = min(len(host_data[h]) for h in hosts)
                samples_per_host = {h: min_count for h in hosts}
            else:
                samples_per_host = {h: len(host_data[h]) for h in hosts}
        else:
            # Distribute samples according to weights
            samples_per_host = {}
            remaining = total_samples
            for i, host in enumerate(hosts):
                if i == len(hosts) - 1:
                    # Last host gets remaining samples
                    n_samples = remaining
                else:
                    n_samples = int(total_samples * weights[i])
                    remaining -= n_samples
                
                # Cap by available data
                max_available = len(host_data[host])
                if self.config.max_samples_per_host:
                    max_available = min(max_available, self.config.max_samples_per_host)
                
                samples_per_host[host] = min(n_samples, max_available)
        
        # Sample from each host
        sampled_records = []
        for host in hosts:
            n_samples = samples_per_host[host]
            records = host_data[host]
            
            if n_samples >= len(records):
                sampled = records
            else:
                sampled = random.sample(records, n_samples)
            
            sampled_records.extend(sampled)
            logger.info(f"Sampled {len(sampled)} records from {host}")
        
        # Shuffle
        random.shuffle(sampled_records)
        
        return sampled_records
    
    def augment_data(self, records: List[Dict]) -> List[Dict]:
        """
        Apply data augmentation strategies.
        
        Note: This is a placeholder for more sophisticated augmentation.
        Currently just returns the original records.
        """
        # Future: implement reverse complement augmentation,
        # synonymous codon swapping, etc.
        return records
    
    def load_and_mix(
        self,
        file_paths: List[str],
        target_hosts: Optional[Set[str]] = None,
        total_samples: Optional[int] = None
    ) -> List[Dict]:
        """
        Complete pipeline: load, filter, sample, and augment.
        
        Args:
            file_paths: List of JSONL file paths
            target_hosts: Optional set of hosts to include
            total_samples: Optional total number of samples
        
        Returns:
            Final list of records ready for training
        """
        logger.info("Loading multi-host data...")
        host_data = self.load_multi_host(file_paths, target_hosts)
        
        logger.info("Sampling balanced dataset...")
        sampled = self.sample_balanced(host_data, total_samples)
        
        logger.info("Applying augmentation...")
        augmented = self.augment_data(sampled)
        
        logger.info(f"Final dataset: {len(augmented)} records")
        return augmented
    
    def save_jsonl(self, records: List[Dict], output_path: str):
        """Save records to JSONL file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        logger.info(f"Saved {len(records)} records to {output_path}")


def split_by_host(
    records: List[Dict]
) -> Dict[str, List[Dict]]:
    """
    Split a list of records by host organism.
    
    Args:
        records: List of data records
    
    Returns:
        Dictionary mapping host name to records
    """
    host_data = {}
    for record in records:
        host = record.get("host", "unknown")
        if host not in host_data:
            host_data[host] = []
        host_data[host].append(record)
    return host_data


def create_train_val_split(
    records: List[Dict],
    val_fraction: float = 0.15,
    stratify_by_host: bool = True,
    random_seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """
    Create train/validation split with optional stratification by host.
    
    Args:
        records: List of data records
        val_fraction: Fraction of data for validation
        stratify_by_host: If True, maintain host distribution in both splits
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_records, val_records)
    """
    random.seed(random_seed)
    
    if not stratify_by_host:
        # Simple random split
        records_shuffled = records.copy()
        random.shuffle(records_shuffled)
        n_val = int(len(records) * val_fraction)
        return records_shuffled[n_val:], records_shuffled[:n_val]
    
    # Stratified split by host
    host_data = split_by_host(records)
    train_records = []
    val_records = []
    
    for host, host_records in host_data.items():
        shuffled = host_records.copy()
        random.shuffle(shuffled)
        n_val = int(len(shuffled) * val_fraction)
        
        val_records.extend(shuffled[:n_val])
        train_records.extend(shuffled[n_val:])
    
    # Shuffle again
    random.shuffle(train_records)
    random.shuffle(val_records)
    
    logger.info(f"Split: {len(train_records)} train, {len(val_records)} val")
    
    return train_records, val_records


def merge_datasets(
    dataset_paths: List[str],
    output_path: str,
    config: Optional[DataConfig] = None
) -> Dict[str, int]:
    """
    Merge multiple JSONL datasets into one with smart filtering and balancing.
    
    Args:
        dataset_paths: List of paths to JSONL files
        output_path: Output path for merged dataset
        config: DataConfig for filtering and sampling
    
    Returns:
        Statistics dictionary
    """
    loader = DataLoader(config)
    records = loader.load_and_mix(dataset_paths)
    loader.save_jsonl(records, output_path)
    
    host_counts = {}
    for record in records:
        host = record.get("host", "unknown")
        host_counts[host] = host_counts.get(host, 0) + 1
    
    stats = {
        "total_records": len(records),
        "host_counts": host_counts,
    }
    
    return stats

