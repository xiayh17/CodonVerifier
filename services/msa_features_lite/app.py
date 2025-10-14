#!/usr/bin/env python3
"""
MSA Features Service (Lite + Real)

Provides both lightweight approximation and real MSA-based evolutionary features.

Modes:
- Lite (default): Fast approximation based on sequence properties
- Real (--use-mmseqs2): Actual MSA generation using MMseqs2

Features:
- MSA depth (real homolog count or approximation)
- Conservation scores (from alignment or heuristics)
- Evolutionary entropy
- Protein family indicators

Author: CodonVerifier Team
Date: 2025-10-12
"""

import json
import argparse
import logging
import sys
import math
import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
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


class RealMSAGenerator:
    """
    Real MSA generation using MMseqs2
    """
    
    def __init__(
        self,
        database: str = "/data/mmseqs_db/uniref50",
        threads: int = 8,
        evalue: float = 1e-3,
        min_seq_id: float = 0.3,
        coverage: float = 0.5,
        use_gpu: bool = False,
        gpu_id: int = 0
    ):
        self.database = database
        self.threads = threads
        self.evalue = evalue
        self.min_seq_id = min_seq_id
        self.coverage = coverage
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.mmseqs_available = self._check_mmseqs2()
        self.database_available = False
        self.gpu_available = False
        
        if self.mmseqs_available:
            self.database_available = self._check_database()
            if not self.database_available:
                logger.warning("MMseqs2 database not available, will use Lite fallback")
            
            # Check GPU availability if requested
            if self.use_gpu:
                self.gpu_available = self._check_gpu()
                if self.gpu_available:
                    logger.info(f"GPU acceleration enabled (GPU {self.gpu_id})")
                else:
                    logger.warning("GPU requested but not available, using CPU")
                    self.use_gpu = False
        else:
            logger.warning("MMseqs2 not available, will use Lite fallback")
    
    def _check_mmseqs2(self) -> bool:
        """Check if MMseqs2 is installed"""
        try:
            result = subprocess.run(
                ['mmseqs', 'version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                logger.info(f"Found MMseqs2: {result.stdout.strip()}")
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return False
    
    def _check_database(self) -> bool:
        """Check if MMseqs2 database exists and is valid"""
        if not os.path.exists(self.database):
            logger.error(f"Database not found: {self.database}")
            return False
        
        # Check if it's a valid MMseqs2 database by checking for required files
        required_files = [
            f"{self.database}.dbtype",
            f"{self.database}.index",
            f"{self.database}.lookup"
        ]
        
        for file_path in required_files:
            if not os.path.exists(file_path):
                logger.error(f"Database file missing: {file_path}")
                return False
        
        # Try to read database info using view command
        try:
            result = subprocess.run(
                ['mmseqs', 'view', self.database],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                logger.info(f"Database valid: {self.database}")
                return True
            else:
                logger.error(f"Database invalid: {result.stderr}")
                return False
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.error(f"Database check failed: {e}")
            return False
    
    def _check_gpu(self) -> bool:
        """Check if GPU is available for MMseqs2"""
        try:
            # Check if nvidia-smi is available
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                gpu_name = result.stdout.strip()
                logger.info(f"Found GPU: {gpu_name}")
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        # Try to check CUDA availability
        try:
            result = subprocess.run(
                ['mmseqs', 'search', '--help'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if '--gpu' in result.stdout:
                logger.info("MMseqs2 supports GPU, but no GPU detected")
            return False
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def generate_msa_features_batch(
        self,
        records: List[Dict],
        work_dir: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Generate MSA features for a batch of records
        
        Returns:
            Dict mapping protein_id -> msa_features
        """
        if not self.mmseqs_available or not self.database_available:
            logger.warning("MMseqs2 or database not available, using Lite fallback")
            return {}
        
        # Smart GPU/CPU selection based on batch size
        batch_size = len(records)
        use_gpu_for_this_batch = self.use_gpu and self.gpu_available
        
        # For small batches, CPU is often faster due to GPU initialization overhead
        if batch_size < 50 and use_gpu_for_this_batch:
            logger.info(f"Small batch size ({batch_size}), using CPU for better performance")
            use_gpu_for_this_batch = False
        
        # Write sequences to FASTA
        fasta_file = os.path.join(work_dir, "queries.fasta")
        with open(fasta_file, 'w') as f:
            for rec in records:
                protein_id = rec.get('protein_id', 'unknown')
                protein_aa = rec.get('protein_aa', '')
                if protein_aa:
                    f.write(f">{protein_id}\n{protein_aa}\n")
        
        # Run MMseqs2 search with appropriate GPU setting
        original_use_gpu = self.use_gpu
        self.use_gpu = use_gpu_for_this_batch
        
        try:
            alignment_file = self._run_mmseqs2_search(fasta_file, work_dir)
        finally:
            # Restore original GPU setting
            self.use_gpu = original_use_gpu
        
        if not alignment_file:
            logger.warning("MMseqs2 search failed, using Lite fallback")
            return {}
        
        # Compute features from alignment
        results = {}
        for rec in records:
            protein_id = rec.get('protein_id', 'unknown')
            features = self._compute_features_from_alignment(protein_id, alignment_file)
            if features:
                results[protein_id] = features
        
        return results
    
    def _run_mmseqs2_search(self, query_fasta: str, output_dir: str) -> Optional[str]:
        """Run MMseqs2 search"""
        try:
            # Create query database
            query_db = os.path.join(output_dir, "query_db")
            subprocess.run(
                ['mmseqs', 'createdb', query_fasta, query_db],
                check=True,
                capture_output=True
            )
            
            # Run search
            result_db = os.path.join(output_dir, "result_db")
            tmp_dir = os.path.join(output_dir, "tmp")
            os.makedirs(tmp_dir, exist_ok=True)
            
            # Build MMseqs2 search command with conservative parameters
            search_cmd = [
                'mmseqs', 'search',
                query_db, self.database, result_db, tmp_dir,
                '--threads', str(self.threads),
                '-e', str(self.evalue),
                '--min-seq-id', str(self.min_seq_id),
                '-c', str(self.coverage),
                '--alignment-mode', '3',
                '--max-seqs', '1000',  # 限制搜索结果数量
                '-s', '7.5'  # 降低敏感度以提高速度
            ]
            
            # Add GPU support if available
            if self.use_gpu and self.gpu_available:
                # GPU-specific optimizations
                search_cmd.extend([
                    '--gpu', str(self.gpu_id),
                    '--gpu-memory', '8192',  # 限制GPU显存使用
                    '--batch-size', '32'     # GPU批次大小
                ])
                logger.info(f"Using GPU {self.gpu_id} for MMseqs2 search with GPU optimizations")
            
            # Run MMseqs2 search with timeout
            timeout_duration = 600 if self.use_gpu else 300  # GPU需要更长初始化时间
            try:
                logger.info(f"Running MMseqs2 search with {timeout_duration}s timeout...")
                result = subprocess.run(
                    search_cmd, 
                    check=True, 
                    capture_output=True, 
                    text=True,
                    timeout=timeout_duration
                )
                logger.info("MMseqs2 search completed successfully")
            except subprocess.TimeoutExpired:
                logger.error(f"MMseqs2 search timed out after {timeout_duration} seconds")
                logger.warning("This may be due to GPU initialization overhead or large database size")
                logger.warning("Consider using a smaller database (e.g., Swiss-Prot) for testing")
                return None
            except subprocess.CalledProcessError as e:
                logger.error(f"MMseqs2 search failed: {e}")
                logger.error(f"Command: {' '.join(search_cmd)}")
                logger.error(f"Stderr: {e.stderr}")
                if self.use_gpu:
                    logger.warning("GPU search failed, this may be due to:")
                    logger.warning("1. MMseqs2 not compiled with GPU support")
                    logger.warning("2. CUDA drivers not properly installed")
                    logger.warning("3. GPU memory insufficient")
                return None
            
            # Convert to TSV
            result_tsv = os.path.join(output_dir, "result.tsv")
            subprocess.run([
                'mmseqs', 'convertalis',
                query_db, self.database, result_db, result_tsv,
                '--format-output', 'query,target,pident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits'
            ], check=True, capture_output=True)
            
            return result_tsv
            
        except subprocess.CalledProcessError as e:
            logger.error(f"MMseqs2 failed: {e}")
            return None
    
    def _compute_features_from_alignment(
        self,
        protein_id: str,
        alignment_file: str
    ) -> Optional[Dict[str, float]]:
        """Compute MSA features from alignment results"""
        try:
            hits = []
            with open(alignment_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 12 and parts[0] == protein_id:
                        hits.append({
                            'pident': float(parts[2]),
                            'alnlen': int(parts[3]),
                            'evalue': float(parts[10]),
                        })
            
            if not hits:
                return None
            
            # Compute features
            identities = [h['pident'] for h in hits]
            features = {
                'msa_depth': float(len(hits)),
                'msa_effective_depth': float(sum(1 for h in hits if h['pident'] >= 50)),
                'msa_coverage': min(1.0, sum(h['alnlen'] for h in hits) / len(hits) / 100.0),
                'conservation_mean': sum(identities) / len(identities) / 100.0,
                'conservation_min': min(identities) / 100.0,
                'conservation_max': max(identities) / 100.0,
                'conservation_entropy_mean': math.sqrt(
                    sum((i/100.0 - sum(identities)/len(identities)/100.0)**2 for i in identities) / len(identities)
                ),
                'coevolution_score': 0.5,
                'contact_density': 0.3,
                'pfam_count': 1.0,
                'domain_count': 1.0
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error computing features for {protein_id}: {e}")
            return None


def _load_real_msa_map(real_msa_jsonl: str) -> Dict[str, Dict[str, float]]:
    """
    Load real MSA features from a JSONL produced by the production pipeline.
    Strict requirements:
    - Each line must include 'protein_id'
    - Each line must include 'msa_features' (a dict of numeric features)

    Returns:
        Mapping from protein_id -> msa_features dict
    """
    msa_map: Dict[str, Dict[str, float]] = {}
    with open(real_msa_jsonl, 'r') as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            try:
                obj = json.loads(line.strip())
            except Exception as e:
                logger.warning(f"Real MSA jsonl line {i} parse error: {e}")
                continue
            protein_id = obj.get('protein_id')
            msa_features = obj.get('msa_features')
            if not protein_id or not isinstance(msa_features, dict):
                logger.warning(f"Real MSA jsonl line {i} missing protein_id or msa_features; skipping")
                continue
            msa_map[str(protein_id)] = {k: float(v) for k, v in msa_features.items() if isinstance(v, (int, float))}
    logger.info(f"Loaded real MSA features for {len(msa_map)} protein_ids from {real_msa_jsonl}")
    return msa_map


def process_jsonl(
    input_path: str,
    output_path: str,
    limit: Optional[int] = None,
    real_msa_jsonl: Optional[str] = None
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
    real_msa_map: Optional[Dict[str, Dict[str, float]]] = None
    if real_msa_jsonl:
        # Strict mode: use only real MSA features; no heuristic fallback
        real_msa_map = _load_real_msa_map(real_msa_jsonl)
    results = []
    
    logger.info(f"Processing {input_path}...")
    
    with open(input_path, 'r') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            
            try:
                record = json.loads(line.strip())
                
                protein_id = record.get('protein_id')
                if real_msa_map is not None:
                    # Strict join by protein_id; no fallback
                    if not protein_id:
                        logger.warning(f"Input record {i} missing protein_id with real_msa_jsonl provided; skipping")
                        predictor.stats['errors'] += 1
                        continue
                    feats = real_msa_map.get(str(protein_id))
                    if not feats:
                        logger.warning(f"No real MSA features found for protein_id={protein_id}; skipping")
                        predictor.stats['errors'] += 1
                        continue
                    record['msa_features'] = feats
                else:
                    # Heuristic (lite) path
                    protein_aa = record.get('protein_aa', '')
                    pid = protein_id or f'protein_{i}'
                    if not protein_aa:
                        logger.warning(f"No protein_aa for {pid}, skipping")
                        continue
                    features = predictor.predict_evolutionary_features(protein_aa, pid)
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
        description="MSA Features Service - Lite approximation or real MMseqs2-based MSA"
    )
    
    parser.add_argument('--input', required=True, help="Input JSONL file")
    parser.add_argument('--output', required=True, help="Output JSON file")
    parser.add_argument('--limit', type=int, help="Limit number of records (for testing)")
    parser.add_argument('--real-msa-jsonl', help="Path to real MSA JSONL. If provided, strictly join by protein_id and do not use heuristics.")
    parser.add_argument('--use-mmseqs2', action='store_true', help="Use real MMseqs2 MSA generation instead of Lite approximation")
    parser.add_argument('--database', default='/data/mmseqs_db/uniref50', help="MMseqs2 database path (for --use-mmseqs2)")
    parser.add_argument('--threads', type=int, default=8, help="Number of threads for MMseqs2")
    parser.add_argument('--batch-size', type=int, default=100, help="Batch size for MMseqs2 processing")
    parser.add_argument('--use-gpu', action='store_true', help="Use GPU acceleration for MMseqs2 (requires CUDA)")
    parser.add_argument('--gpu-id', type=int, default=0, help="GPU ID to use (default: 0)")
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Process
    try:
        if args.use_mmseqs2:
            # Use real MMseqs2 MSA generation
            logger.info("Mode: Real MSA (MMseqs2)")
            logger.info(f"Database: {args.database}")
            logger.info(f"Threads: {args.threads}")
            
            # Load records
            records = []
            with open(args.input, 'r') as f:
                for i, line in enumerate(f):
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                        records.append(rec)
                        if args.limit and len(records) >= args.limit:
                            break
                    except json.JSONDecodeError as e:
                        logger.warning(f"Line {i+1}: JSON decode error: {e}")
            
            logger.info(f"Loaded {len(records)} records")
            
            # Initialize MMseqs2 generator
            msa_gen = RealMSAGenerator(
                database=args.database,
                threads=args.threads,
                use_gpu=args.use_gpu,
                gpu_id=args.gpu_id
            )
            
            # Process in batches
            results = []
            batch_size = args.batch_size
            lite_fallback = MSAFeaturesLite()
            
            with tempfile.TemporaryDirectory(prefix='msa_') as work_dir:
                for i in range(0, len(records), batch_size):
                    batch = records[i:i + batch_size]
                    batch_num = i // batch_size + 1
                    total_batches = (len(records) + batch_size - 1) // batch_size
                    
                    logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} records)...")
                    
                    # Generate MSA features
                    batch_dir = os.path.join(work_dir, f"batch_{batch_num}")
                    os.makedirs(batch_dir, exist_ok=True)
                    
                    msa_features_map = msa_gen.generate_msa_features_batch(batch, batch_dir)
                    
                    # Add features to records
                    for rec in batch:
                        protein_id = rec.get('protein_id', 'unknown')
                        
                        if protein_id in msa_features_map:
                            rec['msa_features'] = msa_features_map[protein_id]
                        else:
                            # Fallback to Lite
                            protein_aa = rec.get('protein_aa', '')
                            if protein_aa:
                                features = lite_fallback.predict_evolutionary_features(protein_aa, protein_id)
                                rec['msa_features'] = asdict(features)
                        
                        results.append(rec)
            
            # Write results
            logger.info(f"Writing {len(results)} results to {args.output}")
            with open(args.output, 'w') as f:
                for result in results:
                    f.write(json.dumps(result) + '\n')
            
            logger.info("✓ MSA features generation completed successfully!")
            logger.info(f"  Processed: {len(results)}")
            logger.info(f"  Output: {args.output}")
            
        else:
            # Use Lite approximation or pre-computed MSA
            logger.info("Mode: Lite approximation" + (" + pre-computed MSA" if args.real_msa_jsonl else ""))
            
            stats = process_jsonl(
                args.input,
                args.output,
                args.limit,
                real_msa_jsonl=args.real_msa_jsonl
            )
            
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
