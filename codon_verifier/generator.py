from __future__ import annotations

"""
Unified candidate generation entry points with constraint filtering.

Sources supported:
- "ct": CodonTransformer adapter (transformer/HFC/BFC/URC)
- "policy": HostConditionalCodonPolicy sampling
- "heuristic": constrained_decode using usage table

This module only handles generation and constraint filtering. Scoring is done
via reward.combine_reward or other downstream components.
"""

from typing import List, Optional, Dict, Tuple

from .codon_utils import constrained_decode, validate_cds, aa_from_dna
from .hosts import tables
from .policy import HostConditionalCodonPolicy
from . import codontransformer_adapter as ct_adapter


def _filter_constraints(dna: str, aa: str, motifs_forbidden: Optional[List[str]]) -> bool:
    ok, _ = validate_cds(dna)
    if not ok:
        return False
    if aa_from_dna(dna) != aa.strip().upper():
        return False
    if motifs_forbidden:
        s = dna.upper()
        for m in motifs_forbidden:
            mm = m.upper().replace("U","T")
            if mm in s:
                return False
    return True


def generate_candidates(
    aa: str,
    host: str,
    n: int,
    source: str = "heuristic",
    motifs_forbidden: Optional[List[str]] = None,
    temperature: float = 1.0,
    top_k: int = 50,
    beam_size: int = 0,
    method: str = "transformer",
) -> List[str]:
    """Generate up to N candidates passing constraints, deduplicated."""

    aa = aa.strip().upper()
    usage = tables.E_COLI_USAGE if host in {"E_coli", "e_coli", "ECOLI"} else tables.E_COLI_USAGE

    candidates: List[str] = []
    seen = set()

    if source == "ct":
        # Use CodonTransformer adapter (may raise if transformer method unavailable)
        raw = ct_adapter.generate_sequences(
            aa=aa, host=host, n=max(1, n * 2), method=method,
            temperature=temperature, top_k=top_k, beam_size=beam_size
        )
        for dna in raw:
            if dna in seen:
                continue
            if _filter_constraints(dna, aa, motifs_forbidden):
                candidates.append(dna)
                seen.add(dna)
            if len(candidates) >= n:
                break
        return candidates

    if source == "policy":
        policy = HostConditionalCodonPolicy([host], init_usage=usage)
        # oversample to compensate for constraint filtering
        tries = max(1, n * 3)
        for _ in range(tries):
            dna, _ = policy.sample_sequence(aa, host, motifs_forbidden=motifs_forbidden, temperature=temperature)
            if dna in seen:
                continue
            if _filter_constraints(dna, aa, motifs_forbidden):
                candidates.append(dna)
                seen.add(dna)
            if len(candidates) >= n:
                break
        return candidates

    # heuristic
    tries = max(1, n * 3)
    for _ in range(tries):
        dna = constrained_decode(aa, usage, motifs_forbidden=motifs_forbidden, temperature=temperature)
        if dna in seen:
            continue
        if _filter_constraints(dna, aa, motifs_forbidden):
            candidates.append(dna)
            seen.add(dna)
        if len(candidates) >= n:
            break
    return candidates


