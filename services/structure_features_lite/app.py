#!/usr/bin/env python3
"""
Structure Features Lite Service

Lightweight approximation of protein structure features without requiring
AlphaFold or ESMFold. Provides fast estimates based on sequence properties.

Features:
- pLDDT approximation (confidence scores)
- Disorder/flexibility predictions
- Secondary structure estimates
- SASA approximation

Author: CodonVerifier Team
Date: 2025-10-05
"""

import json
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class StructuralFeatures:
    """Approximated structural features"""
    plddt_mean: float = 75.0
    plddt_min: float = 60.0
    plddt_max: float = 90.0
    plddt_std: float = 10.0
    plddt_q25: float = 70.0
    plddt_q75: float = 85.0
    disorder_ratio: float = 0.15
    flexible_ratio: float = 0.05
    sasa_mean: float = 0.0
    sasa_total: float = 0.0
    sasa_polar_ratio: float = 0.0
    helix_ratio: float = 0.33
    sheet_ratio: float = 0.25
    coil_ratio: float = 0.42
    has_signal_peptide: float = 0.0
    has_transmembrane: float = 0.0
    tm_helix_count: float = 0.0


class StructureFeaturesLite:
    """
    Lightweight structure prediction approximation
    
    Uses amino acid composition, hydrophobicity, and sequence patterns
    to estimate structural properties without running AlphaFold.
    """
    
    # Amino acid properties
    HYDROPHOBICITY = {
        'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
        'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
        'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
        'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
    }
    
    DISORDER_PROPENSITY = {
        'A': 0.06, 'R': 0.18, 'N': 0.15, 'D': 0.15, 'C': 0.02,
        'Q': 0.12, 'E': 0.14, 'G': 0.16, 'H': 0.08, 'I': 0.03,
        'L': 0.05, 'K': 0.19, 'M': 0.05, 'F': 0.04, 'P': 0.22,
        'S': 0.14, 'T': 0.11, 'W': 0.04, 'Y': 0.07, 'V': 0.03
    }
    
    HELIX_PROPENSITY = {
        'A': 1.45, 'R': 0.79, 'N': 0.73, 'D': 0.98, 'C': 0.77,
        'Q': 1.17, 'E': 1.53, 'G': 0.53, 'H': 1.24, 'I': 1.00,
        'L': 1.34, 'K': 1.07, 'M': 1.20, 'F': 1.12, 'P': 0.59,
        'S': 0.79, 'T': 0.82, 'W': 1.14, 'Y': 0.61, 'V': 1.14
    }
    
    SHEET_PROPENSITY = {
        'A': 0.97, 'R': 0.90, 'N': 0.65, 'D': 0.80, 'C': 1.30,
        'Q': 1.23, 'E': 0.26, 'G': 0.81, 'H': 0.71, 'I': 1.60,
        'L': 1.22, 'K': 0.74, 'M': 1.67, 'F': 1.28, 'P': 0.62,
        'S': 0.72, 'T': 1.20, 'W': 1.19, 'Y': 1.29, 'V': 1.65
    }
    
    def __init__(self):
        self.stats = {
            'processed': 0,
            'errors': 0
        }
    
    def predict_structure(self, aa_sequence: str, protein_id: str = None) -> StructuralFeatures:
        """
        Predict approximate structural features from amino acid sequence
        
        Args:
            aa_sequence: Amino acid sequence
            protein_id: Optional protein identifier for logging
        
        Returns:
            StructuralFeatures object with approximated values
        """
        if not aa_sequence:
            return StructuralFeatures()
        
        # Calculate basic properties
        length = len(aa_sequence)
        
        # Disorder prediction
        disorder_score = self._calculate_disorder(aa_sequence)
        
        # pLDDT approximation (inverse of disorder)
        # Well-ordered proteins have high pLDDT (80-95)
        # Disordered regions have low pLDDT (50-70)
        base_plddt = 90 - (disorder_score * 40)  # Scale disorder to pLDDT range
        
        # Add sequence complexity bonus
        complexity = self._calculate_complexity(aa_sequence)
        plddt_mean = base_plddt + (complexity * 5)
        plddt_mean = max(50, min(95, plddt_mean))  # Clamp to realistic range
        
        # Estimate pLDDT distribution
        plddt_std = 8 + (disorder_score * 15)
        plddt_min = max(30, plddt_mean - 2 * plddt_std)
        plddt_max = min(100, plddt_mean + plddt_std)
        plddt_q25 = plddt_mean - 0.67 * plddt_std
        plddt_q75 = plddt_mean + 0.67 * plddt_std
        
        # Disorder and flexibility ratios
        disorder_ratio = disorder_score
        flexible_ratio = disorder_score * 0.4  # Flexible regions are subset of disordered
        
        # Secondary structure prediction
        helix_ratio, sheet_ratio, coil_ratio = self._predict_secondary_structure(aa_sequence)
        
        # Signal peptide and transmembrane detection
        has_signal = self._detect_signal_peptide(aa_sequence)
        has_tm, tm_count = self._detect_transmembrane(aa_sequence)
        
        # SASA approximation (surface area)
        sasa_mean, sasa_total, polar_ratio = self._approximate_sasa(aa_sequence)
        
        features = StructuralFeatures(
            plddt_mean=float(plddt_mean),
            plddt_min=float(plddt_min),
            plddt_max=float(plddt_max),
            plddt_std=float(plddt_std),
            plddt_q25=float(plddt_q25),
            plddt_q75=float(plddt_q75),
            disorder_ratio=float(disorder_ratio),
            flexible_ratio=float(flexible_ratio),
            helix_ratio=float(helix_ratio),
            sheet_ratio=float(sheet_ratio),
            coil_ratio=float(coil_ratio),
            sasa_mean=float(sasa_mean),
            sasa_total=float(sasa_total),
            sasa_polar_ratio=float(polar_ratio),
            has_signal_peptide=float(has_signal),
            has_transmembrane=float(has_tm),
            tm_helix_count=float(tm_count)
        )
        
        self.stats['processed'] += 1
        
        return features
    
    def _calculate_disorder(self, sequence: str) -> float:
        """Calculate disorder propensity score (0-1)"""
        if not sequence:
            return 0.5
        
        disorder_sum = sum(
            self.DISORDER_PROPENSITY.get(aa, 0.1) 
            for aa in sequence
        )
        disorder_score = disorder_sum / len(sequence)
        
        # Normalize to 0-1 range
        disorder_score = min(1.0, max(0.0, disorder_score))
        
        return disorder_score
    
    def _calculate_complexity(self, sequence: str) -> float:
        """Calculate sequence complexity (0-1), higher = more complex"""
        if not sequence:
            return 0.5
        
        # Count unique amino acids
        unique_aa = len(set(sequence))
        max_unique = 20
        
        # Calculate entropy
        aa_counts = {}
        for aa in sequence:
            aa_counts[aa] = aa_counts.get(aa, 0) + 1
        
        entropy = 0
        for count in aa_counts.values():
            p = count / len(sequence)
            if p > 0:
                entropy -= p * math.log2(p)
        
        max_entropy = math.log2(20)
        
        # Combine metrics
        complexity = (unique_aa / max_unique + entropy / max_entropy) / 2
        
        return complexity
    
    def _predict_secondary_structure(self, sequence: str) -> tuple:
        """Predict approximate secondary structure ratios"""
        if not sequence:
            return 0.33, 0.25, 0.42
        
        # Calculate propensities
        helix_score = sum(self.HELIX_PROPENSITY.get(aa, 1.0) for aa in sequence) / len(sequence)
        sheet_score = sum(self.SHEET_PROPENSITY.get(aa, 1.0) for aa in sequence) / len(sequence)
        
        # Normalize to ratios (coil is remainder)
        total = helix_score + sheet_score + 1.0  # +1 for coil baseline
        
        helix_ratio = helix_score / total
        sheet_ratio = sheet_score / total
        coil_ratio = 1.0 / total
        
        # Ensure they sum to 1
        total_ratio = helix_ratio + sheet_ratio + coil_ratio
        helix_ratio /= total_ratio
        sheet_ratio /= total_ratio
        coil_ratio /= total_ratio
        
        return helix_ratio, sheet_ratio, coil_ratio
    
    def _detect_signal_peptide(self, sequence: str) -> bool:
        """Detect potential signal peptide (N-terminal)"""
        if len(sequence) < 20:
            return False
        
        # Simple heuristic: hydrophobic N-terminus
        n_term = sequence[:20]
        hydrophobic_count = sum(1 for aa in n_term if self.HYDROPHOBICITY.get(aa, 0) > 2.0)
        
        # Signal peptides typically have >50% hydrophobic residues
        return hydrophobic_count / len(n_term) > 0.5
    
    def _detect_transmembrane(self, sequence: str) -> tuple:
        """Detect potential transmembrane helices"""
        if len(sequence) < 20:
            return False, 0
        
        # Scan for hydrophobic stretches (typical TM helix is ~20 aa)
        window_size = 20
        tm_count = 0
        
        for i in range(len(sequence) - window_size + 1):
            window = sequence[i:i+window_size]
            hydrophobic_count = sum(
                1 for aa in window 
                if self.HYDROPHOBICITY.get(aa, 0) > 2.5
            )
            
            # TM helices are >70% hydrophobic
            if hydrophobic_count / window_size > 0.7:
                tm_count += 1
                # Skip ahead to avoid counting same helix multiple times
                i += window_size
        
        has_tm = tm_count > 0
        
        return has_tm, tm_count
    
    def _approximate_sasa(self, sequence: str) -> tuple:
        """Approximate solvent accessible surface area"""
        if not sequence:
            return 0.0, 0.0, 0.0
        
        # Rough approximation: ~120 Å² per residue on average
        sasa_per_residue = 120.0
        sasa_total = len(sequence) * sasa_per_residue
        sasa_mean = sasa_per_residue
        
        # Count polar vs nonpolar residues
        polar_aa = set('RNDQEHKSTY')
        polar_count = sum(1 for aa in sequence if aa in polar_aa)
        polar_ratio = polar_count / len(sequence)
        
        return sasa_mean, sasa_total, polar_ratio


def process_jsonl(
    input_path: str,
    output_path: str,
    limit: Optional[int] = None
) -> Dict:
    """
    Process JSONL file and add structural features
    
    Args:
        input_path: Input JSONL file
        output_path: Output JSON file
        limit: Optional limit on number of records
    
    Returns:
        Statistics dictionary
    """
    predictor = StructureFeaturesLite()
    results = []
    
    logger.info(f"Processing {input_path}...")
    
    with open(input_path, 'r') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            
            try:
                record = json.loads(line.strip())
                
                # Extract protein AA sequence
                protein_aa = record.get('protein_aa', '')
                protein_id = record.get('protein_id', f'protein_{i}')
                
                if not protein_aa:
                    logger.warning(f"No protein_aa for {protein_id}, skipping")
                    continue
                
                # Predict features
                features = predictor.predict_structure(protein_aa, protein_id)
                
                # Add structural features to original record
                record['structure_features'] = asdict(features)
                # Ensure protein_id is preserved for downstream integration
                record['protein_id'] = protein_id
                results.append(record)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1} proteins...")
            
            except Exception as e:
                logger.error(f"Error processing record {i}: {e}")
                predictor.stats['errors'] += 1
                continue
    
    # Write results as JSONL (one JSON object per line)
    logger.info(f"Writing {len(results)} results to {output_path}")
    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    # Statistics
    stats = {
        'total_processed': predictor.stats['processed'],
        'total_errors': predictor.stats['errors'],
        'output_file': output_path,
        'output_count': len(results)
    }
    
    logger.info(f"Statistics: {stats}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Structure Features Lite Service - Fast structural feature approximation"
    )
    
    parser.add_argument('--input', required=True, help="Input JSONL file")
    parser.add_argument('--output', required=True, help="Output JSON file")
    parser.add_argument('--limit', type=int, help="Limit number of records (for testing)")
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Process
    try:
        stats = process_jsonl(args.input, args.output, args.limit)
        
        logger.info("✓ Structure features generation completed successfully!")
        logger.info(f"  Processed: {stats['total_processed']}")
        logger.info(f"  Errors: {stats['total_errors']}")
        logger.info(f"  Output: {stats['output_file']}")
        
        sys.exit(0)
    
    except Exception as e:
        logger.error(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
