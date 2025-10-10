"""Language-model style scoring helpers for DNA candidates.

The real system is expected to call Evo 2's nucleotide LM interface.  To make
integration possible without that dependency, this module implements a
lightweight fallback based on codon-usage statistics.  The public helper
functions expose the same keys that an external LM should provide so that the
rest of the codebase can remain unchanged when swapping in the real model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
import os
from functools import lru_cache
from typing import Dict, Optional

from .codon_utils import AA_TO_CODONS, CODON_TO_AA, chunk_codons, aa_from_dna
from .hosts import tables
from . import evo2_adapter


@dataclass
class LMScore:
    """Container for host/conditional LM statistics."""

    log_likelihood: float
    avg_log_likelihood: float
    perplexity: float
    geometric_mean_prob: float
    score: float

    def to_dict(self, prefix: str) -> Dict[str, float]:
        return {
            f"{prefix}_loglik": self.log_likelihood,
            f"{prefix}_avg_loglik": self.avg_log_likelihood,
            f"{prefix}_perplexity": self.perplexity,
            f"{prefix}_geom": self.geometric_mean_prob,
            f"{prefix}_score": self.score,
        }


def _host_usage(host: str) -> Dict[str, float]:
    key = host.strip().lower()
    if key in {"e_coli", "ecoli", "e.coli", "escherichia_coli"}:
        return tables.E_COLI_USAGE
    raise KeyError(f"Host '{host}' is not registered in lm_features")


def _codon_probabilities(usage: Dict[str, float], smoothing: float = 1e-6) -> Dict[str, float]:
    probs: Dict[str, float] = {}
    for aa, codons in AA_TO_CODONS.items():
        fam = [c for c in codons if c in usage]
        if not fam:
            # Uniform within family if usage missing
            for c in codons:
                probs[c] = 1.0 / max(1, len(codons))
            continue
        total = sum(max(smoothing, usage[c]) for c in fam)
        for c in codons:
            raw = max(smoothing, usage.get(c, smoothing))
            probs[c] = raw / total if total > 0 else 1.0 / max(1, len(codons))
    return probs


@lru_cache(maxsize=8)
def _cached_codon_probs(host: str) -> Dict[str, float]:
    usage = _host_usage(host)
    return _codon_probabilities(usage)


def _score_from_prob(prob: float) -> float:
    # Map [0,1] geometric mean probability to a smoother score in [0,1].
    prob = max(0.0, min(1.0, prob))
    # emphasise differences around mid-range while keeping extremes saturated
    return prob ** (1 / 3.0)


def _lm_stats_from_probs(dna: str, probs: Dict[str, float]) -> LMScore:
    codons = chunk_codons(dna)
    loglik = 0.0
    count = 0
    for codon in codons:
        if len(codon) != 3:
            continue
        aa = CODON_TO_AA.get(codon)
        if aa is None or aa == "*":
            continue
        p = max(1e-9, probs.get(codon, 1e-9))
        loglik += math.log(p)
        count += 1
    if count == 0:
        return LMScore(0.0, 0.0, 1.0, 1.0, 1.0)
    avg = loglik / count
    geom = math.exp(avg)
    ppl = math.exp(-avg)
    score = _score_from_prob(geom)
    return LMScore(loglik, avg, ppl, geom, score)


def score_nt_lm(dna: str, host: str = "E_coli") -> Dict[str, float]:
    """Return host-specific LM scores for a DNA candidate.

    If environment variable `USE_EVO2_LM` is set to a truthy value and Evo 2
    backend is available, use Evo 2 to compute nt-LM stats. Otherwise fall back
    to internal codon-usage proxy.
    """

    use_evo2 = os.getenv("USE_EVO2_LM", "").strip().lower() in {"1","true","yes","on"}
    if use_evo2 and evo2_adapter.is_available():
        stats = evo2_adapter.score_sequence(dna)
        # Map to host-prefixed keys for compatibility
        return {
            "lm_host_loglik": stats.get("loglik", 0.0),
            "lm_host_avg_loglik": stats.get("avg_loglik", 0.0),
            "lm_host_perplexity": stats.get("perplexity", 1.0),
            "lm_host_geom": stats.get("geom", 1.0),
            # Heuristic score transform consistent with proxy path
            "lm_host_score": _score_from_prob(stats.get("geom", 1.0)),
        }

    probs = _cached_codon_probs(host)
    host_score = _lm_stats_from_probs(dna, probs)
    return host_score.to_dict("lm_host")


def score_conditional_nt_lm(dna: str, aa: Optional[str] = None, host: str = "E_coli") -> Dict[str, float]:
    """Score DNA conditioned on the protein sequence."""

    host_dict = score_nt_lm(dna, host=host)
    if aa is None:
        return host_dict
    translated = aa_from_dna(dna)
    if translated != aa.strip().upper():
        penalised = dict(host_dict)
        penalised.update({
            "lm_cond_loglik": -1e6,
            "lm_cond_avg_loglik": -1e6,
            "lm_cond_perplexity": float("inf"),
            "lm_cond_geom": 0.0,
            "lm_cond_score": 0.0,
        })
        return penalised

    use_evo2 = os.getenv("USE_EVO2_LM", "").strip().lower() in {"1","true","yes","on"}
    if use_evo2 and evo2_adapter.is_available():
        stats = evo2_adapter.score_sequence(dna)
        cond = {
            "lm_cond_loglik": stats.get("loglik", 0.0),
            "lm_cond_avg_loglik": stats.get("avg_loglik", 0.0),
            "lm_cond_perplexity": stats.get("perplexity", 1.0),
            "lm_cond_geom": stats.get("geom", 1.0),
            "lm_cond_score": _score_from_prob(stats.get("geom", 1.0)),
        }
        out = dict(host_dict)
        out.update({k: v for k, v in cond.items() if k not in out})
        return out

    return {
        **host_dict,
        **_lm_stats_from_probs(dna, _cached_codon_probs(host)).to_dict("lm_cond"),
    }


def combined_lm_features(dna: str, aa: Optional[str] = None, host: str = "E_coli") -> Dict[str, float]:
    """Convenience wrapper that merges host and conditional LM scores."""

    host_feats = score_nt_lm(dna, host=host)
    if aa is None:
        return host_feats
    cond = score_conditional_nt_lm(dna, aa=aa, host=host)
    merged = dict(host_feats)
    for k, v in cond.items():
        if k not in merged:
            merged[k] = v
    return merged
