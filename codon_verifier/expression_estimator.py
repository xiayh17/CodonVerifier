#!/usr/bin/env python3
"""
Enhanced Expression Level Estimator

This module provides advanced expression level estimation by combining:
1. Metadata-based heuristics (reviewed status, subcellular location, length)
2. Nucleotide language model features (Evo2 confidence scores, likelihood)
3. Sequence-based features (codon usage, GC content)

The estimator can work in multiple modes:
- metadata_only: Original heuristic approach (baseline)
- model_enhanced: Incorporates Evo2 model outputs when available
- full: Uses all available features

Author: Codon Verifier Team
Date: 2025-10-04
"""

import logging
from typing import Dict, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class ExpressionEstimator:
    """
    Enhanced expression level estimator with multi-source feature integration.
    """
    
    def __init__(self, mode: str = "model_enhanced"):
        """
        Initialize estimator.
        
        Args:
            mode: Estimation mode ("metadata_only", "model_enhanced", "full")
        """
        self.mode = mode
        logger.info(f"Initialized ExpressionEstimator in '{mode}' mode")
    
    def estimate(
        self,
        # Metadata features
        reviewed: str,
        subcellular_location: str,
        protein_length: int,
        organism: str,
        # Model features (optional)
        model_features: Optional[Dict[str, float]] = None,
        # Sequence features (optional)
        sequence: Optional[str] = None
    ) -> Tuple[float, str]:
        """
        Estimate expression level using available features.
        
        Args:
            reviewed: SwissProt review status ("reviewed" or "unreviewed")
            subcellular_location: Protein subcellular localization
            protein_length: Amino acid length
            organism: Source organism
            model_features: Optional dict with keys:
                - avg_confidence: Average Evo2 confidence score (0-1)
                - max_confidence: Max Evo2 confidence score
                - min_confidence: Min Evo2 confidence score  
                - avg_loglik: Average log-likelihood from Evo2
                - perplexity: Sequence perplexity from Evo2
            sequence: Optional DNA sequence for direct feature extraction
            
        Returns:
            (expression_value, confidence_level) tuple
        """
        # Start with metadata-based baseline
        base_score, base_confidence = self._estimate_from_metadata(
            reviewed, subcellular_location, protein_length, organism
        )
        
        # If model features not available or mode is metadata_only, return baseline
        if self.mode == "metadata_only" or model_features is None:
            return base_score, base_confidence
        
        # Enhance with model features
        model_score, model_confidence = self._enhance_with_model(
            base_score, base_confidence, model_features
        )
        
        # Optionally enhance with sequence features
        if self.mode == "full" and sequence is not None:
            final_score, final_confidence = self._enhance_with_sequence(
                model_score, model_confidence, sequence
            )
            return final_score, final_confidence
        
        return model_score, model_confidence
    
    def _estimate_from_metadata(
        self,
        reviewed: str,
        subcellular_location: str,
        protein_length: int,
        organism: str
    ) -> Tuple[float, str]:
        """
        Original metadata-based heuristic estimation.
        
        This provides the baseline estimate based on biological knowledge.
        """
        # Base score
        score = 50.0
        confidence = "low"
        
        # Reviewed proteins from SwissProt tend to be better characterized
        if reviewed.lower() == "reviewed":
            score += 20.0
            confidence = "medium"
        
        # Subcellular location hints at expression level
        if subcellular_location:
            location_lower = subcellular_location.lower()
            # Cytoplasmic proteins often have higher expression
            if "cytoplasm" in location_lower or "cytosol" in location_lower:
                score += 15.0
            # Ribosomal proteins typically highly expressed
            if "ribosome" in location_lower:
                score += 30.0
                confidence = "medium"
            # Membrane proteins may have lower expression
            if "membrane" in location_lower:
                score -= 10.0
            # Secreted proteins
            if "secreted" in location_lower or "extracellular" in location_lower:
                score -= 5.0
        
        # Protein length factor (very long or very short may be less expressed)
        if 100 <= protein_length <= 500:
            score += 10.0
        elif protein_length < 50 or protein_length > 1000:
            score -= 10.0
        
        # Ensure non-negative
        score = max(10.0, score)
        
        return score, confidence
    
    def _enhance_with_model(
        self,
        base_score: float,
        base_confidence: str,
        model_features: Dict[str, float]
    ) -> Tuple[float, str]:
        """
        Enhance expression estimate using Evo2 model outputs.
        
        The intuition:
        - High confidence scores → sequence is well-formed → likely expressible
        - High average log-likelihood → sequence follows natural patterns
        - Low perplexity → sequence is predictable → stable expression
        
        Args:
            base_score: Baseline score from metadata
            base_confidence: Baseline confidence level
            model_features: Model output features
            
        Returns:
            Enhanced (score, confidence) tuple
        """
        score = base_score
        confidence = base_confidence
        
        # Extract model features
        avg_conf = model_features.get("avg_confidence", None)
        max_conf = model_features.get("max_confidence", None)
        min_conf = model_features.get("min_confidence", None)
        avg_loglik = model_features.get("avg_loglik", None)
        perplexity = model_features.get("perplexity", None)
        
        # Weight for model contribution (can be tuned)
        model_weight = 0.3  # Model contributes 30% to the final score
        
        # 1. Confidence score adjustment
        if avg_conf is not None:
            # High confidence (>0.9) indicates well-formed sequence → boost expression
            # Low confidence (<0.7) indicates problematic sequence → reduce expression
            if avg_conf > 0.9:
                conf_boost = 15.0 * (avg_conf - 0.9) / 0.1  # Up to +15
                score += conf_boost * model_weight
                if avg_conf > 0.95:
                    confidence = "high"
            elif avg_conf < 0.7:
                conf_penalty = 15.0 * (0.7 - avg_conf) / 0.3  # Up to -15
                score -= conf_penalty * model_weight
            
            # Consistency check: low variance in confidence is good
            if min_conf is not None and max_conf is not None:
                conf_range = max_conf - min_conf
                if conf_range < 0.1:  # Very consistent
                    score += 5.0 * model_weight
                    if confidence == "medium":
                        confidence = "high"
        
        # 2. Log-likelihood adjustment
        if avg_loglik is not None:
            # Normalize log-likelihood (typically ranges from -5 to 0 for good sequences)
            # Higher (closer to 0) is better
            if avg_loglik > -2.0:
                loglik_boost = 10.0 * (avg_loglik + 2.0) / 2.0  # Up to +10
                score += loglik_boost * model_weight
            elif avg_loglik < -4.0:
                loglik_penalty = 10.0 * (-4.0 - avg_loglik) / 2.0  # Up to -10
                score -= loglik_penalty * model_weight
        
        # 3. Perplexity adjustment
        if perplexity is not None:
            # Lower perplexity means more predictable/natural sequence
            # Good sequences: perplexity 2-10
            # Bad sequences: perplexity >50
            if perplexity < 10.0:
                perp_boost = 10.0 * (10.0 - perplexity) / 8.0  # Up to +10
                score += perp_boost * model_weight
            elif perplexity > 30.0:
                perp_penalty = 10.0 * (perplexity - 30.0) / 20.0  # Up to -10
                score -= min(perp_penalty * model_weight, 10.0)
        
        # Ensure valid range
        score = max(10.0, min(100.0, score))
        
        # Format debug message with optional values
        conf_str = f"{avg_conf:.3f}" if avg_conf is not None else "N/A"
        loglik_str = f"{avg_loglik:.3f}" if avg_loglik is not None else "N/A"
        perp_str = f"{perplexity:.2f}" if perplexity is not None else "N/A"
        
        logger.debug(
            f"Model enhancement: {base_score:.1f} → {score:.1f} "
            f"(conf={conf_str}, loglik={loglik_str}, perp={perp_str})"
        )
        
        return score, confidence
    
    def _enhance_with_sequence(
        self,
        base_score: float,
        base_confidence: str,
        sequence: str
    ) -> Tuple[float, str]:
        """
        Enhance with direct sequence features (GC content, codon usage).
        
        This is optional and provides additional refinement.
        """
        score = base_score
        confidence = base_confidence
        
        # GC content
        gc_count = sequence.count('G') + sequence.count('C')
        gc_content = gc_count / len(sequence) if len(sequence) > 0 else 0.5
        
        # Optimal GC content is 0.40-0.60 for E. coli
        if 0.40 <= gc_content <= 0.60:
            score += 5.0
        elif gc_content < 0.30 or gc_content > 0.70:
            score -= 5.0
        
        # Check for extreme patterns (poly-A, poly-T runs)
        max_homopolymer = self._max_homopolymer_length(sequence)
        if max_homopolymer > 8:
            score -= 10.0  # Long homopolymer runs reduce expression
        
        # Ensure valid range
        score = max(10.0, min(100.0, score))
        
        return score, confidence
    
    @staticmethod
    def _max_homopolymer_length(sequence: str) -> int:
        """Find maximum homopolymer run length in sequence."""
        if not sequence:
            return 0
        
        max_len = 1
        current_len = 1
        
        for i in range(1, len(sequence)):
            if sequence[i] == sequence[i-1]:
                current_len += 1
                max_len = max(max_len, current_len)
            else:
                current_len = 1
        
        return max_len


def load_evo2_features(
    evo2_result_path: str,
    index_mapping: Optional[Dict[str, int]] = None
) -> Dict[int, Dict[str, float]]:
    """
    Load Evo2 model features from result JSON file.
    
    Args:
        evo2_result_path: Path to merged_dataset_result.json
        index_mapping: Optional mapping from sequence ID to index in results
        
    Returns:
        Dictionary mapping record index to model features:
        {
            0: {"avg_confidence": 0.92, "max_confidence": 0.95, ...},
            1: {"avg_confidence": 0.88, "max_confidence": 0.90, ...},
            ...
        }
    """
    import json
    
    logger.info(f"Loading Evo2 features from {evo2_result_path}")
    
    try:
        with open(evo2_result_path, 'r') as f:
            results = json.load(f)
        
        features_by_index = {}
        
        for idx, result in enumerate(results):
            if result.get("status") != "success":
                logger.warning(f"Skipping record {idx}: status={result.get('status')}")
                continue
            
            output = result.get("output", {})
            
            # Extract features from Evo2 output
            # Support both old format (confidence_scores array) and new format (direct metrics)
            features = {}
            
            conf_scores = output.get("confidence_scores", [])
            if conf_scores:
                # Old format with confidence_scores array
                features["avg_confidence"] = float(np.mean(conf_scores))
                features["max_confidence"] = float(np.max(conf_scores))
                features["min_confidence"] = float(np.min(conf_scores))
                features["std_confidence"] = float(np.std(conf_scores))
            
            # New format: direct metrics from Evo2 service
            if "loglik" in output:
                features["loglik"] = float(output["loglik"])
            if "avg_loglik" in output:
                features["avg_loglik"] = float(output["avg_loglik"])
            if "perplexity" in output:
                features["perplexity"] = float(output["perplexity"])
            if "geom" in output:
                features["geom"] = float(output["geom"])
            if "gc_content" in output:
                features["gc_content"] = float(output["gc_content"])
            if "codon_entropy" in output:
                features["codon_entropy"] = float(output["codon_entropy"])
            
            # Only add if we extracted at least some features
            if features:
                features_by_index[idx] = features
            else:
                logger.warning(f"No Evo2 features extracted for record {idx}")
        
        logger.info(f"Loaded features for {len(features_by_index)} records")
        return features_by_index
    
    except Exception as e:
        logger.error(f"Failed to load Evo2 features: {e}")
        return {}


def estimate_expression_enhanced(
    reviewed: str,
    subcellular_location: str,
    protein_length: int,
    organism: str,
    model_features: Optional[Dict[str, float]] = None,
    sequence: Optional[str] = None,
    mode: str = "model_enhanced"
) -> Tuple[float, str]:
    """
    Convenience function for enhanced expression estimation.
    
    This is a drop-in replacement for the original estimate_expression_from_metadata
    with additional model enhancement capabilities.
    
    Args:
        reviewed: SwissProt review status
        subcellular_location: Protein localization
        protein_length: Amino acid length
        organism: Source organism
        model_features: Optional Evo2 model features
        sequence: Optional DNA sequence
        mode: Estimation mode
        
    Returns:
        (expression_value, confidence_level) tuple
        
    Example:
        >>> # Baseline mode (same as original)
        >>> expr, conf = estimate_expression_enhanced(
        ...     "reviewed", "cytoplasm", 250, "E. coli",
        ...     mode="metadata_only"
        ... )
        
        >>> # Enhanced with Evo2 features
        >>> model_feats = {
        ...     "avg_confidence": 0.92,
        ...     "avg_loglik": -1.5,
        ...     "perplexity": 8.2
        ... }
        >>> expr, conf = estimate_expression_enhanced(
        ...     "reviewed", "cytoplasm", 250, "E. coli",
        ...     model_features=model_feats,
        ...     mode="model_enhanced"
        ... )
    """
    estimator = ExpressionEstimator(mode=mode)
    return estimator.estimate(
        reviewed=reviewed,
        subcellular_location=subcellular_location,
        protein_length=protein_length,
        organism=organism,
        model_features=model_features,
        sequence=sequence
    )


# Backward compatibility: keep original function signature
def estimate_expression_from_metadata(
    reviewed: str,
    subcellular_location: str,
    protein_length: int,
    organism: str
) -> Tuple[float, str]:
    """
    Original metadata-based heuristic estimation (for backward compatibility).
    
    Use estimate_expression_enhanced() for model-enhanced estimation.
    """
    return estimate_expression_enhanced(
        reviewed=reviewed,
        subcellular_location=subcellular_location,
        protein_length=protein_length,
        organism=organism,
        model_features=None,
        mode="metadata_only"
    )

