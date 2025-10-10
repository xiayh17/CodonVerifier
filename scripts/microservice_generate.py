#!/usr/bin/env python3
"""
Microservice-based sequence generation and scoring pipeline.

This script orchestrates:
1. Sequence generation (via CodonTransformer microservice or local generator)
2. Surrogate model scoring (via local inference or microservice)
3. Top-K selection based on reward scores

Usage:
    # Use local generator + local surrogate
    python3 scripts/microservice_generate.py \
      --aa MAAAAAAA \
      --host E_coli \
      --n 500 \
      --surrogate data/production/ecoli/models/Ec_surrogate.pkl \
      --top 100 \
      --output data/output/generated_sequences.json

    # Use microservice generator (requires Docker)
    python3 scripts/microservice_generate.py \
      --aa MAAAAAAA \
      --host E_coli \
      --n 500 \
      --surrogate data/production/ecoli/models/Ec_surrogate.pkl \
      --use-docker-generator \
      --top 100
"""

import argparse
import json
import logging
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from codon_verifier.generator import generate_candidates
from codon_verifier.hosts.tables import E_COLI_USAGE, E_COLI_TRNA, HUMAN_USAGE, HUMAN_TRNA
from codon_verifier.lm_features import combined_lm_features
from codon_verifier.reward import combine_reward
from codon_verifier.surrogate import load_and_predict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MicroserviceGenerator:
    """Orchestrate microservice-based sequence generation and scoring."""
    
    def __init__(
        self,
        docker_compose_file: str = "docker-compose.yml",
        use_docker_generator: bool = False
    ):
        self.docker_compose_file = Path(docker_compose_file)
        self.use_docker_generator = use_docker_generator
        
    def step1_generate_sequences(
        self,
        aa: str,
        host: str,
        n: int,
        source: str = "heuristic",
        motifs_forbidden: Optional[List[str]] = None,
        temperature: float = 1.0,
        top_k: int = 50,
        beam_size: int = 0,
        method: str = "transformer"
    ) -> List[str]:
        """
        Step 1: Generate candidate sequences.
        
        Args:
            aa: Amino acid sequence
            host: Host organism
            n: Number of candidates to generate
            source: Generation source (ct/policy/heuristic)
            motifs_forbidden: List of forbidden motifs
            temperature: Sampling temperature
            top_k: Top-k sampling
            beam_size: Beam search size
            method: CodonTransformer method if source=ct
            
        Returns:
            List of DNA sequences
        """
        logger.info(f"\n{'='*60}")
        logger.info("Step 1: Generate Candidate Sequences")
        logger.info(f"{'='*60}")
        logger.info(f"AA sequence: {aa}")
        logger.info(f"Host: {host}")
        logger.info(f"Target candidates: {n}")
        logger.info(f"Source: {source}")
        
        start_time = time.time()
        
        if self.use_docker_generator and source == "ct":
            # Use Docker CodonTransformer service
            logger.info("Using Docker CodonTransformer service")
            sequences = self._generate_via_docker(
                aa, host, n, method, temperature, top_k, beam_size, motifs_forbidden
            )
        else:
            # Use local generator
            logger.info("Using local generator")
            sequences = generate_candidates(
                aa=aa,
                host=host,
                n=n,
                source=source,
                motifs_forbidden=motifs_forbidden,
                temperature=temperature,
                top_k=top_k,
                beam_size=beam_size,
                method=method
            )
        
        elapsed = time.time() - start_time
        logger.info(f"✓ Generated {len(sequences)} sequences in {elapsed:.2f}s")
        
        return sequences
    
    def _generate_via_docker(
        self,
        aa: str,
        host: str,
        n: int,
        method: str,
        temperature: float,
        top_k: int,
        beam_size: int,
        motifs_forbidden: Optional[List[str]]
    ) -> List[str]:
        """Generate sequences via Docker CodonTransformer service (placeholder)."""
        logger.warning("Docker CodonTransformer service is not fully implemented yet")
        logger.warning("Falling back to local generator")
        
        return generate_candidates(
            aa=aa,
            host=host,
            n=n,
            source="heuristic",  # Fallback to heuristic
            motifs_forbidden=motifs_forbidden,
            temperature=temperature,
            top_k=top_k,
            beam_size=beam_size
        )
    
    def step2_score_with_surrogate(
        self,
        sequences: List[str],
        aa: str,
        host: str,
        surrogate_path: Optional[str],
        motifs_forbidden: Optional[List[str]] = None,
        w_surrogate: float = 1.0,
        w_rules: float = 1.0,
        lambda_uncertainty: float = 1.0
    ) -> List[Dict[str, Any]]:
        """
        Step 2: Score sequences with surrogate model and reward function.
        
        Args:
            sequences: List of DNA sequences
            aa: Amino acid sequence
            host: Host organism
            surrogate_path: Path to surrogate model .pkl
            motifs_forbidden: List of forbidden motifs
            w_surrogate: Weight for surrogate model
            w_rules: Weight for rules-based features
            lambda_uncertainty: Uncertainty penalty strength
            
        Returns:
            List of scored sequence records
        """
        logger.info(f"\n{'='*60}")
        logger.info("Step 2: Score Sequences with Surrogate Model")
        logger.info(f"{'='*60}")
        logger.info(f"Number of sequences: {len(sequences)}")
        logger.info(f"Surrogate model: {surrogate_path or 'None (rules only)'}")
        
        start_time = time.time()
        
        # Get host-specific tables
        if host.lower() in {"e_coli", "ecoli", "ec"}:
            usage, trna = E_COLI_USAGE, E_COLI_TRNA
        elif host.lower() in {"human", "homo_sapiens"}:
            usage, trna = HUMAN_USAGE, HUMAN_TRNA
        else:
            logger.warning(f"Unknown host {host}, using E. coli tables")
            usage, trna = E_COLI_USAGE, E_COLI_TRNA
        
        # Get surrogate predictions if model provided
        musig = None
        if surrogate_path:
            logger.info("Loading surrogate model and predicting...")
            preds = load_and_predict(
                surrogate_path, 
                sequences, 
                usage, 
                trna_w=trna, 
                extra=None
            )
            musig = [(p["mu"], p["sigma"]) for p in preds]
            logger.info(f"  Mean μ: {sum(p['mu'] for p in preds) / len(preds):.2f}")
            logger.info(f"  Mean σ: {sum(p['sigma'] for p in preds) / len(preds):.2f}")
        
        # Score each sequence
        results = []
        for i, dna in enumerate(sequences):
            # Get LM features
            lm_feats = combined_lm_features(dna, aa=aa, host=host)
            
            # Get surrogate predictions
            mu = sigma = None
            if musig is not None:
                mu, sigma = musig[i]
            
            # Combine reward
            reward_dict = combine_reward(
                dna=dna,
                usage=usage,
                surrogate_mu=mu,
                surrogate_sigma=sigma,
                trna_w=trna,
                cpb=None,
                motifs=motifs_forbidden or [],
                lm_features=lm_feats,
                extra_features=dict(lm_feats),
                w_surrogate=w_surrogate,
                w_rules=w_rules,
                lambda_uncertainty=lambda_uncertainty
            )
            
            # Build result record
            result = {
                "dna": dna,
                "reward": reward_dict.get("reward", 0.0),
                "mu": mu,
                "sigma": sigma,
                "cai": reward_dict.get("cai"),
                "gc": reward_dict.get("gc"),
                "motif_penalty": reward_dict.get("motif_penalty", 0.0)
            }
            results.append(result)
        
        elapsed = time.time() - start_time
        logger.info(f"✓ Scored {len(results)} sequences in {elapsed:.2f}s")
        
        return results
    
    def step3_select_top_k(
        self,
        scored_sequences: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Step 3: Select top-K sequences by reward.
        
        Args:
            scored_sequences: List of scored sequence records
            top_k: Number of top sequences to select
            
        Returns:
            List of top-K sequences
        """
        logger.info(f"\n{'='*60}")
        logger.info("Step 3: Select Top-K Sequences")
        logger.info(f"{'='*60}")
        logger.info(f"Total sequences: {len(scored_sequences)}")
        logger.info(f"Top-K: {top_k}")
        
        # Sort by reward (descending)
        sorted_seqs = sorted(
            scored_sequences,
            key=lambda x: x["reward"],
            reverse=True
        )
        
        # Take top-K
        top_sequences = sorted_seqs[:top_k]
        
        # Log statistics
        if top_sequences:
            logger.info(f"  Best reward: {top_sequences[0]['reward']:.3f}")
            logger.info(f"  Worst reward: {top_sequences[-1]['reward']:.3f}")
            logger.info(f"  Mean reward: {sum(s['reward'] for s in top_sequences) / len(top_sequences):.3f}")
        
        logger.info(f"✓ Selected top {len(top_sequences)} sequences")
        
        return top_sequences
    
    def run_pipeline(
        self,
        aa: str,
        host: str,
        n: int,
        top_k: int,
        surrogate_path: Optional[str] = None,
        output_path: Optional[str] = None,
        source: str = "heuristic",
        motifs_forbidden: Optional[List[str]] = None,
        temperature: float = 1.0,
        top_k_sampling: int = 50,
        beam_size: int = 0,
        method: str = "transformer",
        w_surrogate: float = 1.0,
        w_rules: float = 1.0,
        lambda_uncertainty: float = 1.0
    ) -> Dict[str, Any]:
        """
        Run the complete generation and scoring pipeline.
        
        Returns:
            Dictionary with pipeline results and metadata
        """
        logger.info(f"\n{'='*70}")
        logger.info("🚀 MICROSERVICE SEQUENCE GENERATION PIPELINE")
        logger.info(f"{'='*70}")
        
        pipeline_start = time.time()
        
        # Step 1: Generate sequences
        sequences = self.step1_generate_sequences(
            aa=aa,
            host=host,
            n=n,
            source=source,
            motifs_forbidden=motifs_forbidden,
            temperature=temperature,
            top_k=top_k_sampling,
            beam_size=beam_size,
            method=method
        )
        
        if not sequences:
            logger.error("❌ No sequences generated!")
            return {
                "status": "failed",
                "error": "No sequences generated",
                "total_time_s": time.time() - pipeline_start
            }
        
        # Step 2: Score sequences
        scored_sequences = self.step2_score_with_surrogate(
            sequences=sequences,
            aa=aa,
            host=host,
            surrogate_path=surrogate_path,
            motifs_forbidden=motifs_forbidden,
            w_surrogate=w_surrogate,
            w_rules=w_rules,
            lambda_uncertainty=lambda_uncertainty
        )
        
        # Step 3: Select top-K
        top_sequences = self.step3_select_top_k(
            scored_sequences=scored_sequences,
            top_k=top_k
        )
        
        pipeline_elapsed = time.time() - pipeline_start
        
        # Prepare results
        results = {
            "status": "success",
            "metadata": {
                "aa_sequence": aa,
                "host": host,
                "total_generated": len(sequences),
                "top_k": len(top_sequences),
                "surrogate_model": surrogate_path,
                "total_time_s": pipeline_elapsed
            },
            "top_sequences": top_sequences
        }
        
        # Save results if output path provided
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"\n✓ Results saved to: {output_file}")
        
        # Print summary
        logger.info(f"\n{'='*70}")
        logger.info("✅ PIPELINE COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"Generated: {len(sequences)} sequences")
        logger.info(f"Top-K: {len(top_sequences)} sequences")
        logger.info(f"Best reward: {top_sequences[0]['reward']:.3f}")
        logger.info(f"Total time: {pipeline_elapsed:.2f}s")
        logger.info(f"{'='*70}")
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Microservice-based sequence generation and scoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Local generator + local surrogate
  python3 scripts/microservice_generate.py \\
    --aa MAAAAAAA \\
    --host E_coli \\
    --n 500 \\
    --surrogate data/production/ecoli/models/Ec_surrogate.pkl \\
    --top 100

  # With uncertainty penalty
  python3 scripts/microservice_generate.py \\
    --aa MAAAAAAA \\
    --host E_coli \\
    --n 500 \\
    --surrogate data/production/ecoli/models/Ec_surrogate.pkl \\
    --top 100 \\
    --lambda-uncertainty 2.0
"""
    )
    
    # Required arguments
    parser.add_argument("--aa", required=True, 
                       help="Protein amino acid sequence")
    parser.add_argument("--host", default="E_coli",
                       help="Host organism (default: E_coli)")
    parser.add_argument("--n", type=int, default=500,
                       help="Number of candidates to generate (default: 500)")
    parser.add_argument("--top", type=int, default=100,
                       help="Number of top sequences to return (default: 100)")
    
    # Model and output
    parser.add_argument("--surrogate",
                       help="Path to surrogate model .pkl file")
    parser.add_argument("--output",
                       help="Output JSON file path")
    
    # Generation parameters
    parser.add_argument("--source", 
                       choices=["ct", "policy", "heuristic"],
                       default="heuristic",
                       help="Generation source (default: heuristic)")
    parser.add_argument("--method", default="transformer",
                       help="CodonTransformer method if source=ct")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Sampling temperature (default: 1.0)")
    parser.add_argument("--topk", type=int, default=50,
                       help="Top-k sampling (default: 50)")
    parser.add_argument("--beams", type=int, default=0,
                       help="Beam search size (default: 0)")
    parser.add_argument("--forbid", nargs="*", 
                       default=["GAATTC", "GGATCC"],
                       help="Forbidden motifs (default: GAATTC GGATCC)")
    
    # Scoring parameters
    parser.add_argument("--w-surrogate", type=float, default=1.0,
                       help="Weight for surrogate model (default: 1.0)")
    parser.add_argument("--w-rules", type=float, default=1.0,
                       help="Weight for rules-based features (default: 1.0)")
    parser.add_argument("--lambda-uncertainty", type=float, default=1.0,
                       help="Uncertainty penalty strength (default: 1.0)")
    
    # Microservice options
    parser.add_argument("--use-docker-generator", action="store_true",
                       help="Use Docker CodonTransformer service for generation")
    parser.add_argument("--docker-compose", default="docker-compose.yml",
                       help="Path to docker-compose file")
    
    args = parser.parse_args()
    
    # Validate surrogate path
    if args.surrogate and not Path(args.surrogate).exists():
        logger.error(f"Surrogate model not found: {args.surrogate}")
        sys.exit(1)
    
    # Create pipeline
    pipeline = MicroserviceGenerator(
        docker_compose_file=args.docker_compose,
        use_docker_generator=args.use_docker_generator
    )
    
    # Run pipeline
    results = pipeline.run_pipeline(
        aa=args.aa,
        host=args.host,
        n=args.n,
        top_k=args.top,
        surrogate_path=args.surrogate,
        output_path=args.output,
        source=args.source,
        motifs_forbidden=args.forbid,
        temperature=args.temperature,
        top_k_sampling=args.topk,
        beam_size=args.beams,
        method=args.method,
        w_surrogate=args.w_surrogate,
        w_rules=args.w_rules,
        lambda_uncertainty=args.lambda_uncertainty
    )
    
    # Print top 10 sequences
    if results["status"] == "success":
        logger.info(f"\n{'='*70}")
        logger.info("🏆 TOP 10 SEQUENCES")
        logger.info(f"{'='*70}")
        for i, seq in enumerate(results["top_sequences"][:10], 1):
            logger.info(f"{i:2d}. Reward: {seq['reward']:7.3f} | "
                       f"μ: {seq['mu']:6.2f} | σ: {seq['sigma']:5.2f} | "
                       f"GC: {seq['gc']:.3f}")
            logger.info(f"    DNA: {seq['dna'][:60]}...")


if __name__ == "__main__":
    main()

