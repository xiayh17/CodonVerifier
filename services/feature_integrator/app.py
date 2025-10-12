#!/usr/bin/env python3
"""
Feature Integrator Service

Integrates all enhanced features (structure, evolutionary, context) into
a unified format ready for training. Handles:
- Feature merging from multiple sources
- Context feature extraction from metadata
- Missing value imputation
- Feature validation and quality checks

Author: CodonVerifier Team
Date: 2025-10-05
"""

import json
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ContextFeatureExtractor:
    """Extract context features from metadata"""
    
    # Promoter strength database
    PROMOTER_STRENGTHS = {
        'T7': 1.0,
        'lacUV5': 0.8,
        'tac': 0.85,
        'trc': 0.75,
        'araBAD': 0.7,
        'AOX1': 0.9,  # P. pastoris
        'GAL1': 0.85,  # S. cerevisiae
        'TEF1': 0.7,
        'CMV': 0.9,  # Mammalian
        'EF1A': 0.85,
        'SV40': 0.7,
        'unknown': 0.5,
        'default': 0.5
    }
    
    # RBS strength database
    RBS_STRENGTHS = {
        'strong': 1.0,
        'medium': 0.6,
        'weak': 0.3,
        'BBa_B0034': 1.0,
        'BBa_B0032': 0.6,
        'BBa_B0030': 0.3,
        'optimal': 1.0,
        'suboptimal': 0.5,
        'unknown': 0.5,
        'default': 0.5
    }
    
    def extract_from_metadata(self, metadata: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract numeric context features from metadata
        
        Returns:
            Dictionary of context features with ctx_ prefix
        """
        features = {}
        
        # Promoter features
        promoter = metadata.get('promoter', 'unknown')
        promoter_str = str(promoter).strip()
        features['ctx_promoter_strength'] = self.PROMOTER_STRENGTHS.get(
            promoter_str, 
            self.PROMOTER_STRENGTHS['default']
        )
        
        # RBS/Kozak features
        rbs = metadata.get('rbs', 'unknown')
        rbs_str = str(rbs).strip()
        features['ctx_rbs_strength'] = self.RBS_STRENGTHS.get(
            rbs_str,
            self.RBS_STRENGTHS['default']
        )
        
        features['ctx_rbs_spacing'] = float(metadata.get('rbs_spacing', 8.0))
        features['ctx_kozak_score'] = float(metadata.get('kozak_score', 0.0))
        
        # Vector features
        copy_number = metadata.get('copy_number', 1)
        features['ctx_vector_copy_number'] = float(copy_number)
        features['ctx_has_selection_marker'] = float(metadata.get('has_marker', 0))
        
        # Expression conditions
        conditions = metadata.get('conditions', {})
        if isinstance(conditions, dict):
            temp = conditions.get('temperature', 37.0)
            features['ctx_temperature_norm'] = (float(temp) - 37.0) / 10.0  # Normalize around 37C
            features['ctx_inducer_concentration'] = float(conditions.get('inducer_conc', 0.0))
        else:
            features['ctx_temperature_norm'] = 0.0
            features['ctx_inducer_concentration'] = 0.0
        
        # Growth phase encoding
        growth_phase = metadata.get('growth_phase', 'log')
        phase_encoding = {
            'log': 1.0, 'exponential': 1.0,
            'stationary': 0.5,
            'lag': 0.3
        }
        features['ctx_growth_phase_score'] = phase_encoding.get(growth_phase, 0.8)
        
        # Localization
        localization = metadata.get('localization', 'cytoplasm')
        loc_encoding = {
            'cytoplasm': 1.0,
            'periplasm': 0.8,
            'membrane': 0.6,
            'secreted': 0.7,
            'extracellular': 0.7
        }
        features['ctx_localization_score'] = loc_encoding.get(localization, 1.0)
        
        return features


class FeatureIntegrator:
    """
    Integrate all enhanced features into training-ready format
    """
    
    def __init__(self):
        self.context_extractor = ContextFeatureExtractor()
        self.stats = {
            'processed': 0,
            'with_structure': 0,
            'with_msa': 0,
            'with_context': 0,
            'with_codon_features': 0,
            'errors': 0
        }
    
    def extract_codon_features(self, sequence: str, host: str = 'E_coli') -> Dict[str, float]:
        """
        Extract codon usage features from DNA sequence
        
        Args:
            sequence: DNA sequence
            host: Host organism for codon usage tables
            
        Returns:
            Dictionary of codon features with codon_ prefix
        """
        features = {}
        
        try:
            from codon_verifier.metrics import (
                cai, tai, fop, gc_content, codon_pair_bias_score,
                codon_pair_score, cpg_upa_content, rare_codon_runs,
                homopolymers
            )
            from codon_verifier.hosts.tables import get_host_tables
            from codon_verifier.codon_utils import validate_cds
            
            # Validate sequence
            valid, msg = validate_cds(sequence)
            if not valid:
                logger.warning(f"Invalid CDS: {msg}")
                return {}
            
            # Get host tables
            usage_table, trna_weights, cpb_table = get_host_tables(host, include_cpb=True)
            
            # Calculate codon usage metrics
            features['codon_cai'] = cai(sequence, usage_table)
            features['codon_tai'] = tai(sequence, trna_weights)
            features['codon_fop'] = fop(sequence, usage_table)
            features['codon_gc'] = gc_content(sequence)
            
            # Codon pair metrics
            features['codon_cpb'] = codon_pair_bias_score(sequence, cpb_table)
            features['codon_cps'] = codon_pair_score(sequence, usage_table)
            
            # Dinucleotide analysis
            dinuc_stats = cpg_upa_content(sequence)
            features['codon_cpg_count'] = float(dinuc_stats['cpg_count'])
            features['codon_cpg_freq'] = dinuc_stats['cpg_freq']
            features['codon_cpg_obs_exp'] = dinuc_stats['cpg_obs_exp']
            features['codon_upa_count'] = float(dinuc_stats['upa_count'])
            features['codon_upa_freq'] = dinuc_stats['upa_freq']
            features['codon_upa_obs_exp'] = dinuc_stats['upa_obs_exp']
            
            # Rare codon runs
            rare_runs = rare_codon_runs(sequence, usage_table)
            features['codon_rare_runs'] = float(len(rare_runs))
            features['codon_rare_run_total_len'] = float(sum(length for _, length in rare_runs))
            
            # Homopolymers
            homos = homopolymers(sequence, min_len=6)
            features['codon_homopolymers'] = float(len(homos))
            features['codon_homopoly_total_len'] = float(sum(length for _, _, length in homos))
            
        except Exception as e:
            logger.warning(f"Error extracting codon features: {e}")
            
        return features
    
    def integrate_features(
        self,
        base_record: Dict,
        structure_features: Optional[Dict] = None,
        msa_features: Optional[Dict] = None
    ) -> Dict:
        """
        Integrate all features into a single record
        
        Args:
            base_record: Base record with sequence, expression, metadata
            structure_features: Optional structural features
            msa_features: Optional MSA/evolutionary features
        
        Returns:
            Integrated record ready for training
        """
        # Start with base record
        integrated = {
            'sequence': base_record.get('sequence', ''),
            'protein_aa': base_record.get('protein_aa', ''),
            'host': base_record.get('host', 'E_coli'),
            'expression': base_record.get('expression', 0.0)
        }
        
        # Ensure expression is in correct format
        if isinstance(integrated['expression'], (int, float)):
            integrated['expression'] = {
                'value': float(integrated['expression']),
                'unit': base_record.get('expression_unit', 'RFU'),
                'assay': base_record.get('assay', 'bulk_fluor')
            }
        
        # Initialize extra_features
        extra_features = {}
        
        # Add structural features
        if structure_features:
            for key, value in structure_features.items():
                extra_features[f'struct_{key}'] = float(value)
            self.stats['with_structure'] += 1
        
        # Add MSA/evolutionary features
        if msa_features:
            for key, value in msa_features.items():
                extra_features[f'evo_{key}'] = float(value)
            self.stats['with_msa'] += 1
        
        # Add context features from metadata
        metadata = base_record.get('metadata', {})
        if metadata:
            context_features = self.context_extractor.extract_from_metadata(metadata)
            extra_features.update(context_features)
            self.stats['with_context'] += 1
        
        # Add codon usage features
        sequence = integrated.get('sequence', '')
        host = integrated.get('host', 'E_coli')
        if sequence:
            codon_features = self.extract_codon_features(sequence, host)
            if codon_features:
                extra_features.update(codon_features)
                self.stats['with_codon_features'] += 1
        
        # Add extra_features to integrated record
        integrated['extra_features'] = extra_features
        
        # Preserve metadata for reference
        integrated['metadata'] = metadata
        
        # Add protein_id if available
        if 'protein_id' in base_record:
            integrated['protein_id'] = base_record['protein_id']
        
        self.stats['processed'] += 1
        
        return integrated
    
    def process_files(
        self,
        input_jsonl: str,
        structure_json: Optional[str] = None,
        msa_json: Optional[str] = None,
        output_jsonl: str = None,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        Process input files and integrate features
        
        Args:
            input_jsonl: Base input JSONL file
            structure_json: Optional structure features JSON
            msa_json: Optional MSA features JSON
            output_jsonl: Optional output JSONL path
            limit: Optional limit on records
        
        Returns:
            List of integrated records
        """
        logger.info("Loading input files...")
        
        # Load base records
        base_records = []
        with open(input_jsonl, 'r') as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                try:
                    record = json.loads(line.strip())
                    base_records.append(record)
                except Exception as e:
                    logger.error(f"Error loading base record {i}: {e}")
        
        logger.info(f"Loaded {len(base_records)} base records")
        
        # Load structure features (keyed by protein_id) - support both JSON and JSONL
        structure_dict = {}
        if structure_json and Path(structure_json).exists():
            with open(structure_json, 'r') as f:
                content = f.read().strip()
                # Try JSONL first (each line is a JSON object)
                if content and not content.startswith('['):
                    # JSONL format
                    f.seek(0)
                    for line in f:
                        try:
                            item = json.loads(line.strip())
                            protein_id = item.get('protein_id')
                            if protein_id:
                                structure_dict[protein_id] = item.get('structure_features', {})
                        except:
                            continue
                else:
                    # JSON array format
                    f.seek(0)
                    structure_data = json.load(f)
                    for item in structure_data:
                        protein_id = item.get('protein_id')
                        if protein_id:
                            structure_dict[protein_id] = item.get('structure_features', {})
            logger.info(f"Loaded structure features for {len(structure_dict)} proteins")
        
        # Load MSA features (keyed by protein_id) - support both JSON and JSONL
        msa_dict = {}
        if msa_json and Path(msa_json).exists():
            with open(msa_json, 'r') as f:
                content = f.read().strip()
                # Try JSONL first
                if content and not content.startswith('['):
                    # JSONL format
                    f.seek(0)
                    for line in f:
                        try:
                            item = json.loads(line.strip())
                            protein_id = item.get('protein_id')
                            if protein_id:
                                msa_dict[protein_id] = item.get('msa_features', {})
                        except:
                            continue
                else:
                    # JSON array format
                    f.seek(0)
                    msa_data = json.load(f)
                    for item in msa_data:
                        protein_id = item.get('protein_id')
                        if protein_id:
                            msa_dict[protein_id] = item.get('msa_features', {})
            logger.info(f"Loaded MSA features for {len(msa_dict)} proteins")
        
        # Integrate features
        logger.info("Integrating features...")
        integrated_records = []
        
        for i, record in enumerate(base_records):
            try:
                protein_id = record.get('protein_id', f'protein_{i}')
                
                # Get features
                structure_feats = structure_dict.get(protein_id)
                msa_feats = msa_dict.get(protein_id)
                
                # Integrate
                integrated = self.integrate_features(
                    record,
                    structure_features=structure_feats,
                    msa_features=msa_feats
                )
                
                integrated_records.append(integrated)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Integrated {i + 1}/{len(base_records)} records...")
            
            except Exception as e:
                logger.error(f"Error integrating record {i}: {e}")
                self.stats['errors'] += 1
                continue
        
        # Write output
        if output_jsonl:
            logger.info(f"Writing {len(integrated_records)} records to {output_jsonl}")
            with open(output_jsonl, 'w') as f:
                for record in integrated_records:
                    f.write(json.dumps(record) + '\n')
        
        # Log statistics
        logger.info(f"Integration statistics:")
        logger.info(f"  Total processed: {self.stats['processed']}")
        logger.info(f"  With structure features: {self.stats['with_structure']}")
        logger.info(f"  With MSA features: {self.stats['with_msa']}")
        logger.info(f"  With context features: {self.stats['with_context']}")
        logger.info(f"  With codon features: {self.stats['with_codon_features']}")
        logger.info(f"  Errors: {self.stats['errors']}")
        
        return integrated_records


def main():
    parser = argparse.ArgumentParser(
        description="Feature Integrator - Merge all enhanced features into training format"
    )
    
    parser.add_argument('--input', required=True, 
                       help="Input JSONL file (base records)")
    parser.add_argument('--structure-features', 
                       help="Structure features JSON file")
    parser.add_argument('--msa-features', 
                       help="MSA features JSON file")
    parser.add_argument('--output', required=True, 
                       help="Output JSONL file (training-ready)")
    parser.add_argument('--limit', type=int, 
                       help="Limit number of records (for testing)")
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Process
    try:
        integrator = FeatureIntegrator()
        
        integrated_records = integrator.process_files(
            input_jsonl=args.input,
            structure_json=args.structure_features,
            msa_json=args.msa_features,
            output_jsonl=args.output,
            limit=args.limit
        )
        
        logger.info("✓ Feature integration completed successfully!")
        logger.info(f"  Output: {args.output}")
        logger.info(f"  Total records: {len(integrated_records)}")
        
        # Sample output
        if integrated_records:
            sample = integrated_records[0]
            logger.info(f"\nSample integrated record:")
            logger.info(f"  Protein ID: {sample.get('protein_id', 'N/A')}")
            logger.info(f"  Host: {sample.get('host')}")
            logger.info(f"  Expression: {sample['expression'].get('value')}")
            logger.info(f"  Extra features count: {len(sample.get('extra_features', {}))}")
            if sample.get('extra_features'):
                logger.info(f"  Feature keys (first 5): {list(sample['extra_features'].keys())[:5]}")
        
        sys.exit(0)
    
    except Exception as e:
        logger.error(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
