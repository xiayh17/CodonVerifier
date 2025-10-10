"""
Enhanced Feature Engineering Module for Surrogate Model Training

This module provides advanced feature extractors including:
- Structural features (AlphaFold pLDDT, SASA, secondary structure)
- Evolutionary features (MSA depth, conservation scores)
- Context features (promoter, RBS, vector information)
"""

from __future__ import annotations
import os
import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import warnings

try:
    from Bio.PDB import PDBParser, DSSP, SASA
    from Bio.PDB.Polypeptide import PPBuilder
    _HAS_BIOPYTHON = True
except ImportError:
    _HAS_BIOPYTHON = False
    warnings.warn("BioPython not available. Structural features will be limited.")


@dataclass
class StructuralFeatures:
    """Structural features from AlphaFold/ESM/other structure predictors"""
    # AlphaFold pLDDT (confidence score)
    plddt_mean: float = 0.0
    plddt_min: float = 0.0
    plddt_max: float = 0.0
    plddt_std: float = 0.0
    plddt_q25: float = 0.0
    plddt_q75: float = 0.0
    
    # SASA (Solvent Accessible Surface Area)
    sasa_mean: float = 0.0
    sasa_total: float = 0.0
    sasa_polar_ratio: float = 0.0
    
    # Secondary structure composition
    helix_ratio: float = 0.0
    sheet_ratio: float = 0.0
    coil_ratio: float = 0.0
    
    # Disorder/flexibility
    disorder_ratio: float = 0.0  # pLDDT < 70 regions
    flexible_ratio: float = 0.0  # pLDDT < 50 regions
    
    # Transmembrane/signal peptide (from external predictors)
    has_signal_peptide: float = 0.0
    has_transmembrane: float = 0.0
    tm_helix_count: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class EvolutionaryFeatures:
    """Evolutionary features from MSA/conservation analysis"""
    # MSA statistics
    msa_depth: float = 0.0  # Number of sequences in MSA
    msa_effective_depth: float = 0.0  # Effective sequences after clustering
    msa_coverage: float = 0.0  # Fraction of positions covered
    
    # Conservation scores (averaged over sequence)
    conservation_mean: float = 0.0
    conservation_min: float = 0.0
    conservation_max: float = 0.0
    conservation_entropy_mean: float = 0.0
    
    # Co-evolution/contact prediction
    coevolution_score: float = 0.0
    contact_density: float = 0.0
    
    # Family/domain information
    pfam_count: float = 0.0
    domain_count: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class ContextFeatures:
    """Context features: promoter, RBS, vector, expression conditions"""
    # Promoter encoding (one-hot will be handled separately)
    promoter_strength: float = 0.5  # normalized 0-1
    promoter_type: str = "unknown"
    
    # RBS/Kozak
    rbs_strength: float = 0.5
    rbs_spacing: float = 0.0  # Distance from start codon
    kozak_score: float = 0.0  # For eukaryotes
    
    # Vector/plasmid
    vector_copy_number: float = 1.0  # log-scale normalized
    has_selection_marker: float = 0.0
    
    # Expression conditions
    temperature: float = 37.0  # C
    inducer_concentration: float = 0.0  # normalized
    growth_phase: str = "log"  # log/stationary/etc
    
    # Localization
    localization: str = "cytoplasm"  # cytoplasm/periplasm/secreted/membrane
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_numeric_dict(self) -> Dict[str, float]:
        """Convert to numeric features only (for ML model input)"""
        return {
            "promoter_strength": self.promoter_strength,
            "rbs_strength": self.rbs_strength,
            "rbs_spacing": self.rbs_spacing,
            "kozak_score": self.kozak_score,
            "vector_copy_number": self.vector_copy_number,
            "has_selection_marker": self.has_selection_marker,
            "temperature_norm": (self.temperature - 37.0) / 10.0,  # normalized
            "inducer_concentration": self.inducer_concentration,
        }


class StructuralFeatureExtractor:
    """Extract structural features from PDB files or AlphaFold predictions"""
    
    def __init__(self):
        if not _HAS_BIOPYTHON:
            warnings.warn("BioPython required for full structural feature extraction")
    
    def extract_from_pdb(self, pdb_path: str) -> StructuralFeatures:
        """Extract features from PDB file (e.g., AlphaFold output)"""
        if not _HAS_BIOPYTHON:
            return StructuralFeatures()
        
        features = StructuralFeatures()
        
        try:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("protein", pdb_path)
            
            # Extract pLDDT from B-factor column (AlphaFold convention)
            plddt_scores = []
            for model in structure:
                for chain in model:
                    for residue in chain:
                        for atom in residue:
                            if atom.name == "CA":  # Only C-alpha
                                plddt_scores.append(atom.bfactor)
            
            if plddt_scores:
                plddt_arr = np.array(plddt_scores)
                features.plddt_mean = float(np.mean(plddt_arr))
                features.plddt_min = float(np.min(plddt_arr))
                features.plddt_max = float(np.max(plddt_arr))
                features.plddt_std = float(np.std(plddt_arr))
                features.plddt_q25 = float(np.percentile(plddt_arr, 25))
                features.plddt_q75 = float(np.percentile(plddt_arr, 75))
                
                # Disorder/flexibility
                features.disorder_ratio = float(np.mean(plddt_arr < 70))
                features.flexible_ratio = float(np.mean(plddt_arr < 50))
            
            # Try to compute SASA if possible
            try:
                # This requires DSSP to be installed
                # For now, provide placeholder
                features.sasa_mean = 0.0
                features.sasa_total = 0.0
            except Exception:
                pass
            
        except Exception as e:
            warnings.warn(f"Failed to extract features from PDB: {e}")
        
        return features
    
    def extract_from_alphafold_json(self, json_path: str) -> StructuralFeatures:
        """Extract features from AlphaFold JSON output"""
        features = StructuralFeatures()
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            if 'plddt' in data:
                plddt_arr = np.array(data['plddt'])
                features.plddt_mean = float(np.mean(plddt_arr))
                features.plddt_min = float(np.min(plddt_arr))
                features.plddt_max = float(np.max(plddt_arr))
                features.plddt_std = float(np.std(plddt_arr))
                features.plddt_q25 = float(np.percentile(plddt_arr, 25))
                features.plddt_q75 = float(np.percentile(plddt_arr, 75))
                features.disorder_ratio = float(np.mean(plddt_arr < 70))
                features.flexible_ratio = float(np.mean(plddt_arr < 50))
            
            # Secondary structure if available
            if 'secondary_structure' in data:
                ss = data['secondary_structure']
                total = len(ss)
                if total > 0:
                    features.helix_ratio = ss.count('H') / total
                    features.sheet_ratio = ss.count('E') / total
                    features.coil_ratio = ss.count('C') / total
        
        except Exception as e:
            warnings.warn(f"Failed to extract features from JSON: {e}")
        
        return features
    
    def extract_from_dict(self, feature_dict: Dict[str, Any]) -> StructuralFeatures:
        """Extract from pre-computed feature dictionary"""
        features = StructuralFeatures()
        for key, value in feature_dict.items():
            if hasattr(features, key):
                setattr(features, key, float(value))
        return features


class EvolutionaryFeatureExtractor:
    """Extract evolutionary features from MSA/conservation data"""
    
    def extract_from_msa_file(self, msa_path: str) -> EvolutionaryFeatures:
        """Extract features from MSA file (A3M, FASTA, etc.)"""
        features = EvolutionaryFeatures()
        
        try:
            # Simple MSA depth counting
            with open(msa_path, 'r') as f:
                lines = f.readlines()
            
            # Count sequences (lines starting with '>')
            seq_count = sum(1 for line in lines if line.startswith('>'))
            features.msa_depth = float(seq_count)
            features.msa_effective_depth = float(seq_count)  # Simplified
            
        except Exception as e:
            warnings.warn(f"Failed to extract MSA features: {e}")
        
        return features
    
    def extract_from_conservation_scores(self, scores: List[float]) -> EvolutionaryFeatures:
        """Extract features from per-position conservation scores"""
        features = EvolutionaryFeatures()
        
        if scores:
            score_arr = np.array(scores)
            features.conservation_mean = float(np.mean(score_arr))
            features.conservation_min = float(np.min(score_arr))
            features.conservation_max = float(np.max(score_arr))
            
            # Entropy-based conservation
            # Higher entropy = less conserved
            hist, _ = np.histogram(score_arr, bins=10, density=True)
            hist = hist[hist > 0]
            if len(hist) > 0:
                features.conservation_entropy_mean = float(-np.sum(hist * np.log(hist)))
        
        return features
    
    def extract_from_dict(self, feature_dict: Dict[str, Any]) -> EvolutionaryFeatures:
        """Extract from pre-computed feature dictionary"""
        features = EvolutionaryFeatures()
        for key, value in feature_dict.items():
            if hasattr(features, key):
                setattr(features, key, float(value))
        return features


class ContextFeatureExtractor:
    """Extract context features from experimental metadata"""
    
    # Known promoter strengths (normalized 0-1, relative to strong constitutive)
    PROMOTER_STRENGTHS = {
        "T7": 1.0,
        "lacUV5": 0.8,
        "tac": 0.85,
        "trc": 0.75,
        "araBAD": 0.7,
        "AOX1": 0.9,  # P. pastoris
        "GAL1": 0.85,  # S. cerevisiae
        "TEF1": 0.7,
        "CMV": 0.9,  # Mammalian
        "unknown": 0.5,
    }
    
    # RBS strengths (relative)
    RBS_STRENGTHS = {
        "strong": 1.0,
        "medium": 0.6,
        "weak": 0.3,
        "BBa_B0034": 1.0,
        "BBa_B0032": 0.6,
        "BBa_B0030": 0.3,
        "unknown": 0.5,
    }
    
    def extract_from_metadata(self, metadata: Dict[str, Any]) -> ContextFeatures:
        """Extract context features from experiment metadata"""
        features = ContextFeatures()
        
        # Promoter
        promoter = metadata.get("promoter", "unknown")
        features.promoter_type = promoter
        features.promoter_strength = self.PROMOTER_STRENGTHS.get(promoter, 0.5)
        
        # RBS
        rbs = metadata.get("rbs", "unknown")
        features.rbs_strength = self.RBS_STRENGTHS.get(rbs, 0.5)
        features.rbs_spacing = float(metadata.get("rbs_spacing", 8.0))
        
        # Vector
        copy_number = metadata.get("copy_number", 1)
        features.vector_copy_number = np.log1p(float(copy_number))
        features.has_selection_marker = float(metadata.get("has_marker", 0))
        
        # Conditions
        conditions = metadata.get("conditions", {})
        features.temperature = float(conditions.get("temperature", 37.0))
        features.inducer_concentration = float(conditions.get("inducer_conc", 0.0))
        features.growth_phase = conditions.get("growth_phase", "log")
        
        # Localization
        features.localization = metadata.get("localization", "cytoplasm")
        
        return features


class EnhancedFeatureBundle:
    """Bundle all enhanced features together"""
    
    def __init__(
        self,
        structural: Optional[StructuralFeatures] = None,
        evolutionary: Optional[EvolutionaryFeatures] = None,
        context: Optional[ContextFeatures] = None
    ):
        self.structural = structural or StructuralFeatures()
        self.evolutionary = evolutionary or EvolutionaryFeatures()
        self.context = context or ContextFeatures()
    
    def to_feature_dict(self) -> Dict[str, float]:
        """Convert to flat dictionary of numeric features for ML model"""
        features = {}
        
        # Add structural features
        for key, value in self.structural.to_dict().items():
            features[f"struct_{key}"] = value
        
        # Add evolutionary features
        for key, value in self.evolutionary.to_dict().items():
            features[f"evo_{key}"] = value
        
        # Add context features (numeric only)
        for key, value in self.context.to_numeric_dict().items():
            features[f"ctx_{key}"] = value
        
        return features
    
    @classmethod
    def from_files(
        cls,
        pdb_path: Optional[str] = None,
        alphafold_json: Optional[str] = None,
        msa_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        feature_dict: Optional[Dict[str, Any]] = None
    ) -> "EnhancedFeatureBundle":
        """
        Create feature bundle from various file sources
        
        Args:
            pdb_path: Path to PDB file (AlphaFold output)
            alphafold_json: Path to AlphaFold JSON with pLDDT scores
            msa_path: Path to MSA file
            metadata: Experiment metadata dictionary
            feature_dict: Pre-computed features dictionary
        """
        # Extract structural features
        structural = StructuralFeatures()
        if pdb_path and os.path.exists(pdb_path):
            extractor = StructuralFeatureExtractor()
            structural = extractor.extract_from_pdb(pdb_path)
        elif alphafold_json and os.path.exists(alphafold_json):
            extractor = StructuralFeatureExtractor()
            structural = extractor.extract_from_alphafold_json(alphafold_json)
        elif feature_dict:
            extractor = StructuralFeatureExtractor()
            structural = extractor.extract_from_dict(feature_dict)
        
        # Extract evolutionary features
        evolutionary = EvolutionaryFeatures()
        if msa_path and os.path.exists(msa_path):
            extractor = EvolutionaryFeatureExtractor()
            evolutionary = extractor.extract_from_msa_file(msa_path)
        elif feature_dict:
            extractor = EvolutionaryFeatureExtractor()
            evolutionary = extractor.extract_from_dict(feature_dict)
        
        # Extract context features
        context = ContextFeatures()
        if metadata:
            extractor = ContextFeatureExtractor()
            context = extractor.extract_from_metadata(metadata)
        
        return cls(structural=structural, evolutionary=evolutionary, context=context)
    
    def save_to_json(self, path: str):
        """Save feature bundle to JSON file"""
        data = {
            "structural": self.structural.to_dict(),
            "evolutionary": self.evolutionary.to_dict(),
            "context": self.context.to_dict(),
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load_from_json(cls, path: str) -> "EnhancedFeatureBundle":
        """Load feature bundle from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        structural = StructuralFeatures(**data.get("structural", {}))
        evolutionary = EvolutionaryFeatures(**data.get("evolutionary", {}))
        context = ContextFeatures(**data.get("context", {}))
        
        return cls(structural=structural, evolutionary=evolutionary, context=context)


def merge_enhanced_features(
    base_features: Dict[str, float],
    enhanced_bundle: Optional[EnhancedFeatureBundle] = None
) -> Dict[str, float]:
    """
    Merge base codon features with enhanced features
    
    Args:
        base_features: Base features from build_feature_vector()
        enhanced_bundle: Enhanced features bundle
    
    Returns:
        Merged feature dictionary
    """
    merged = dict(base_features)
    
    if enhanced_bundle:
        enhanced_dict = enhanced_bundle.to_feature_dict()
        merged.update(enhanced_dict)
    
    return merged

