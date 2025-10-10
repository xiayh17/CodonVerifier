"""Utility functions for Codon Verifier service"""

import json
from pathlib import Path
from typing import Dict, Any, List
from Bio.Seq import Seq


def validate_dna_sequence(sequence: str) -> bool:
    """Validate DNA sequence"""
    valid_bases = set('ATCG')
    return all(base.upper() in valid_bases for base in sequence)


def calculate_gc_content(sequence: str) -> float:
    """Calculate GC content of sequence"""
    if not sequence:
        return 0.0
    
    gc_count = sum(1 for base in sequence.upper() if base in 'GC')
    return gc_count / len(sequence)


def has_start_codon(sequence: str) -> bool:
    """Check if sequence starts with ATG"""
    return sequence.upper().startswith('ATG')


def has_stop_codon(sequence: str) -> bool:
    """Check if sequence ends with stop codon"""
    stop_codons = ['TAA', 'TAG', 'TGA']
    return any(sequence.upper().endswith(codon) for codon in stop_codons)

