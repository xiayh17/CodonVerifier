#!/usr/bin/env python3
"""
MSA Features Lite Service

Lightweight approximation of evolutionary features without requiring
actual MSA generation (MMseqs2/JackHMMER). Provides fast estimates
based on sequence properties and conservation heuristics.

Features:
- MSA depth approximation
- Conservation scores
- Evolutionary entropy
- Protein family indicators

Author: CodonVerifier Team
Date: 2025-10-05
"""

import json
import argparse
import logging
import sys
import math
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from collections import Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class EvolutionaryFeatures:
    """Approximated evolutionary features"""
    msa_depth: float = 100.0
    msa_effective_depth: float = 80.0
    msa_coverage: float = 0.95
    conservation_mean: float = 0.7
    conservation_min: float = 0.4
    conservation_max: float = 0.95
    conservation_entropy_mean: float = 0.5
    coevolution_score: float = 0.5
    contact_density: float = 0.3
    pfam_count: float = 0.0
    domain_count: float = 0.0


class MSAFeaturesLite:
    """
    Lightweight MSA and conservation approximation
    
    Uses sequence composition, complexity, and patterns to estimate
    evolutionary features without running MSA tools.
    """
    
    # Amino acid frequencies in well-conserved proteins
    TYPICAL_AA_FREQ = {
        'A': 0.082, 'R': 0.057, 'N': 0.043, 'D': 0.054, 'C': 0.014,
        'Q': 0.039, 'E': 0.063, 'G': 0.072, 'H': 0.023, 'I': 0.053,
        'L': 0.096, 'K': 0.058, 'M': 0.024, 'F': 0.039, 'P': 0.050,
        'S': 0.069, 'T': 0.056, 'W': 0.013, 'Y': 0.032, 'V': 0.067
    }
    
    # Conservation scores for different amino acid types
    CONSERVATION_WEIGHT = {
        'A': 0.7, 'R': 0.8, 'N': 0.7, 'D': 0.8, 'C': 0.95,  # Cys highly conserved
        'Q': 0.7, 'E': 0.8, 'G': 0.85, 'H': 0.75, 'I': 0.7,
        'L': 0.65, 'K': 0.75, 'M': 0.75, 'F': 0.75, 'P': 0.85,  # Pro important for structure
        'S': 0.7, 'T': 0.7, 'W': 0.9,  # Trp highly conserved
        'Y': 0.75, 'V': 0.65
    }
    
    def __init__(self):
        self.stats = {
            'processed': 0,
            'errors': 0
        }
    
    def predict_evolutionary_features(
        self, 
        aa_sequence: str,
        protein_id: str = None
    ) -> EvolutionaryFeatures:
        """
        Predict approximate evolutionary features from amino acid sequence
        
        Args:
            aa_sequence: Amino acid sequence
            protein_id: Optional protein identifier for logging
        
        Returns:
            EvolutionaryFeatures object with approximated values
        """
        if not aa_sequence:
            return EvolutionaryFeatures()
        
        length = len(aa_sequence)
        
        # Conservation prediction
        conservation_mean = self._estimate_conservation(aa_sequence)
        conservation_min = max(0.2, conservation_mean - 0.3)
        conservation_max = min(1.0, conservation_mean + 0.25)
        
        # Entropy calculation
        entropy = self._calculate_entropy(aa_sequence)
        conservation_entropy = 1.0 - (entropy / math.log2(20))  # Normalize to 0-1
        
        # MSA depth estimation (based on sequence complexity)
        msa_depth = self._estimate_msa_depth(aa_sequence)
        msa_effective_depth = msa_depth * 0.8  # Assume 80% effective
        msa_coverage = 0.95  # High coverage default
        
        # Co-evolution approximation
        coevolution_score = self._estimate_coevolution(aa_sequence)
        
        # Contact density (rough estimate based on length)
        contact_density = self._estimate_contact_density(length)
        
        # Domain detection (simple heuristic)
        pfam_count, domain_count = self._detect_domains(aa_sequence)
        
        features = EvolutionaryFeatures(
            msa_depth=float(msa_depth),
            msa_effective_depth=float(msa_effective_depth),
            msa_coverage=float(msa_coverage),
            conservation_mean=float(conservation_mean),
            conservation_min=float(conservation_min),
            conservation_max=float(conservation_max),
            conservation_entropy_mean=float(conservation_entropy),
            coevolution_score=float(coevolution_score),
            contact_density=float(contact_density),
            pfam_count=float(pfam_count),
            domain_count=float(domain_count)
        )
        
        self.stats['processed'] += 1
        
        return features
    
    def _estimate_conservation(self, sequence: str) -> float:
        """
        Estimate average conservation score (0-1, higher = more conserved)
        
        Based on:
        - Amino acid types (Cys, Trp, Pro more conserved)
        - Composition similarity to typical proteins
        - Low-complexity regions (less conserved)
        """
        if not sequence:
            return 0.5
        
        # Component 1: Weighted by amino acid conservation tendency
        weighted_conservation = sum(
            self.CONSERVATION_WEIGHT.get(aa, 0.7) 
            for aa in sequence
        ) / len(sequence)
        
        # Component 2: Composition similarity to typical proteins
        aa_freq = Counter(sequence)
        composition_similarity = 0
        for aa, freq in aa_freq.items():
            observed = freq / len(sequence)
            expected = self.TYPICAL_AA_FREQ.get(aa, 0.05)
            # Penalize large deviations
            composition_similarity += 1 - min(1.0, abs(observed - expected) / expected)
        composition_similarity /= 20  # Normalize
        
        # Component 3: Sequence complexity (higher complexity = more conserved)
        complexity = self._calculate_complexity(sequence)
        
        # Combine components
        conservation = (
            weighted_conservation * 0.5 +
            composition_similarity * 0.3 +
            complexity * 0.2
        )
        
        # Clamp to realistic range
        conservation = max(0.4, min(0.95, conservation))
        
        return conservation
    
    def _calculate_entropy(self, sequence: str) -> float:
        """Calculate Shannon entropy of amino acid distribution"""
        if not sequence:
            return 0.0
        
        aa_counts = Counter(sequence)
        entropy = 0.0
        
        for count in aa_counts.values():
            p = count / len(sequence)
            if p > 0:
                entropy -= p * math.log2(p)
        
        return entropy
    
    def _calculate_complexity(self, sequence: str) -> float:
        """Calculate sequence complexity (0-1)"""
        if not sequence:
            return 0.5
        
        # Number of unique amino acids
        unique_aa = len(set(sequence))
        
        # Entropy-based complexity
        entropy = self._calculate_entropy(sequence)
        max_entropy = math.log2(20)
        
        complexity = (unique_aa / 20 + entropy / max_entropy) / 2
        
        return complexity
    
    def _estimate_msa_depth(self, sequence: str) -> float:
        """
        Estimate MSA depth based on sequence properties
        
        Heuristics:
        - Higher complexity → more homologs → deeper MSA
        - Longer sequences → potentially deeper MSA
        - Conserved residues → evolutionary constrained → deeper MSA
        """
        if not sequence:
            return 50.0
        
        length = len(sequence)
        complexity = self._calculate_complexity(sequence)
        conservation = self._estimate_conservation(sequence)
        
        # Base MSA depth
        base_depth = 100
        
        # Length factor (longer proteins often have more homologs)
        length_factor = 1.0 + math.log10(length / 100) * 0.2
        
        # Complexity factor (more complex = more homologs)
        complexity_factor = 0.5 + complexity
        
        # Conservation factor (highly conserved = important = more homologs)
        conservation_factor = 0.7 + conservation * 0.6
        
        msa_depth = base_depth * length_factor * complexity_factor * conservation_factor
        
        # Add some variation
        msa_depth = max(20, min(500, msa_depth))
        
        return msa_depth
    
    def _estimate_coevolution(self, sequence: str) -> float:
        """
        Estimate co-evolution signal (0-1)
        
        Approximated by sequence patterns that suggest functional constraints
        """
        if len(sequence) < 10:
            return 0.3
        
        # Look for conserved pairs (simple approximation)
        # In reality, this requires MSA and direct coupling analysis
        
        # Check for charged residue pairs (often co-evolve for stability)
        charged_pos = {'R': [], 'K': [], 'D': [], 'E': []}
        for i, aa in enumerate(sequence):
            if aa in charged_pos:
                charged_pos[aa].append(i)
        
        # Count potential salt bridges (within ~10 residues)
        pair_count = 0
        for pos_aa in ['R', 'K']:
            for neg_aa in ['D', 'E']:
                for p1 in charged_pos[pos_aa]:
                    for p2 in charged_pos[neg_aa]:
                        if abs(p1 - p2) <= 10:
                            pair_count += 1
        
        # Normalize by sequence length
        coevolution = min(1.0, pair_count / (len(sequence) * 0.1))
        
        # Base score
        coevolution = 0.3 + coevolution * 0.4
        
        return coevolution
    
    def _estimate_contact_density(self, length: int) -> float:
        """
        Estimate residue contact density
        
        Shorter proteins tend to have higher contact density
        """
        if length < 50:
            return 0.5
        elif length < 200:
            return 0.35
        else:
            return 0.25
    
    def _detect_domains(self, sequence: str) -> tuple:
        """
        Detect potential protein domains (very rough heuristic)
        
        Returns:
            (pfam_count, domain_count)
        """
        length = len(sequence)
        
        # Very simple heuristic: assume 1 domain per ~150 residues
        domain_count = max(1, length // 150)
        
        # Pfam count (conserved domains) - assume some fraction
        pfam_count = max(0, domain_count - 1) if length > 100 else 0
        
        return pfam_count, domain_count


def process_jsonl(
    input_path: str,
    output_path: str,
    limit: Optional[int] = None
) -> Dict:
    """
    Process JSONL file and add MSA/evolutionary features
    
    Args:
        input_path: Input JSONL file
        output_path: Output JSON file
        limit: Optional limit on number of records
    
    Returns:
        Statistics dictionary
    """
    predictor = MSAFeaturesLite()
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
                features = predictor.predict_evolutionary_features(protein_aa, protein_id)
                
                # Add MSA features to original record
                record['msa_features'] = asdict(features)
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
        description="MSA Features Lite Service - Fast evolutionary feature approximation"
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
        
        logger.info("✓ MSA features generation completed successfully!")
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
