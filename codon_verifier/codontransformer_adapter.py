from __future__ import annotations

"""
Adapter for integrating a pretrained CodonTransformer-like generator.

Design goals:
- Optional dependency: if CodonTransformer is not installed, the adapter can
  still provide HFC/BFC/URC generation using local usage tables for quick
  fallback. The true transformer method will raise a clear error if unavailable.
- Minimal, stable API:
    - is_available() -> bool
    - generate_sequences(aa: str, host: str, n: int,
        method: str = "transformer",
        temperature: float = 1.0,
        top_k: int = 50,
        beam_size: int = 0) -> list[str]

This module does not impose any scoring/ranking; it only generates candidates.
Constraint filtering and downstream scoring should be handled by caller.
"""

from typing import List, Dict, Optional
import math
import random

from .codon_utils import AA_TO_CODONS
from .hosts import tables


_HAS_CT = False
try:
    # Replace with the real package name/module path if different in your env
    # e.g., from codon_transformer import CodonTransformer
    import CodonTransformer as _CT  # type: ignore

    _HAS_CT = True
except Exception:
    _HAS_CT = False


def is_available() -> bool:
    """Return True if a CodonTransformer implementation is importable."""
    return _HAS_CT


def _host_usage(host: str) -> Dict[str, float]:
    key = host.strip().lower()
    if key in {"e_coli", "ecoli", "e.coli", "escherichia_coli"}:
        return tables.E_COLI_USAGE
    raise KeyError(f"Host '{host}' is not registered in codontransformer_adapter")


def _hfc_sequence(aa: str, usage: Dict[str, float]) -> str:
    # High-frequency choice per amino acid family
    aa = aa.strip().upper()
    out: List[str] = []
    for i, a in enumerate(aa):
        if i == 0 and a == "M":
            out.append("ATG"); continue
        codons = AA_TO_CODONS.get(a, [])
        if not codons:
            continue
        best = max(codons, key=lambda c: usage.get(c, 0.0))
        out.append(best)
    return "".join(out)


def _bfc_sequence(aa: str, usage: Dict[str, float]) -> str:
    # Sample codons proportionally to usage within each family (background freq)
    aa = aa.strip().upper()
    out: List[str] = []
    for i, a in enumerate(aa):
        if i == 0 and a == "M":
            out.append("ATG"); continue
        codons = AA_TO_CODONS.get(a, [])
        if not codons:
            continue
        fam = [c for c in codons if c in usage]
        if not fam:
            fam = list(codons)
        total = sum(max(1e-9, usage.get(c, 1e-9)) for c in fam)
        r = random.random(); cum = 0.0
        chosen = fam[-1]
        for c in fam:
            p = max(1e-9, usage.get(c, 1e-9))/total if total > 0 else 1.0/len(fam)
            cum += p
            if r <= cum:
                chosen = c; break
        out.append(chosen)
    return "".join(out)


def _urc_sequence(aa: str) -> str:
    # Uniform random per family
    aa = aa.strip().upper()
    out: List[str] = []
    for i, a in enumerate(aa):
        if i == 0 and a == "M":
            out.append("ATG"); continue
        codons = AA_TO_CODONS.get(a, [])
        if not codons:
            continue
        out.append(random.choice(codons))
    return "".join(out)


def generate_sequences(
    aa: str,
    host: str,
    n: int,
    method: str = "transformer",
    temperature: float = 1.0,
    top_k: int = 50,
    beam_size: int = 0,
    seed: Optional[int] = None,
) -> List[str]:
    """
    Generate N DNA candidates for a protein AA sequence using the requested method.

    method:
      - "transformer": use CodonTransformer if available (may raise if missing)
      - "HFC": highest-frequency choice per family
      - "BFC": sample by background frequency
      - "URC": uniform random choice
    """
    if seed is not None:
        random.seed(seed)

    method_up = method.strip().upper()
    usage = _host_usage(host)

    if method_up == "TRANSFORMER":
        if not _HAS_CT:
            raise RuntimeError("CodonTransformer not installed; cannot use 'transformer' method.")
        # Placeholder: replace with real API calls for your environment.
        # Below is a pseudo-interface showing temperature/top_k/beam usage.
        # sequences = _CT.predict_dna_sequence(
        #     aa, host=host, num_return_sequences=n, temperature=temperature,
        #     top_k=top_k, num_beams=beam_size
        # )
        # return list(sequences)
        raise NotImplementedError("Wire CodonTransformer API here (predict_dna_sequence).")

    out: List[str] = []
    for _ in range(max(1, n)):
        if method_up == "HFC":
            out.append(_hfc_sequence(aa, usage))
        elif method_up == "BFC":
            out.append(_bfc_sequence(aa, usage))
        elif method_up == "URC":
            out.append(_urc_sequence(aa))
        else:
            # default fallback: BFC with temperature-like diversity via Dirichlet noise
            out.append(_bfc_sequence(aa, usage))
    return out


