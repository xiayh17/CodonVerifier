"""
Optional Evo 2 adapter for nt-LM scoring.

This module provides a tiny wrapper that, when available, calls Evo 2 to
compute token-level logits/probabilities for DNA sequences and derives
aggregate scores compatible with the keys expected by the rest of this repo.

Two backends are supported:
- Local Python package `evo2` for on-device inference.
- NVIDIA hosted API/NIM if environment variables are provided.

If neither backend is available, this module exposes `is_available() == False`
and calling its functions will raise.
"""
from __future__ import annotations

import os
import math
from typing import Dict, Optional, Tuple


_HAS_LOCAL = False
try:
    # Local inference backend (if installed)
    from evo2 import Evo2  # type: ignore

    _HAS_LOCAL = True
except Exception:
    _HAS_LOCAL = False


def _has_nim_env() -> bool:
    return bool(os.getenv("NVCF_RUN_KEY") or os.getenv("EVO2_NIM_URL"))


def is_available() -> bool:
    """Return True if any Evo 2 backend is available."""
    return _HAS_LOCAL or _has_nim_env()


def check_evo2_available() -> bool:
    """Compat shim for services expecting check_evo2_available()."""
    return is_available()


class _LazyLocalModel:
    _instance = None

    @classmethod
    def get(cls, model_name: str = "evo2_7b"):
        if cls._instance is None:
            # 检查是否有本地模型文件路径
            model_path = os.getenv("EVO2_MODEL_PATH")
            if model_path and os.path.exists(model_path):
                # 优先尝试带 model_path 的构造；不支持则回退
                try:
                    cls._instance = Evo2(model_name, model_path=model_path)
                except TypeError:
                    cls._instance = Evo2(model_name)
            else:
                # 使用默认的模型名称
                cls._instance = Evo2(model_name)
        return cls._instance


def _score_from_token_probs(probs: Tuple[float, ...]) -> Tuple[float, float, float, float]:
    """Aggregate per-token next-token probabilities into LM stats.

    Returns: (loglik, avg_loglik, perplexity, geometric_mean_prob)
    """
    eps = 1e-9
    loglik = 0.0
    count = 0
    for p in probs:
        p = max(eps, min(1.0, float(p)))
        loglik += math.log(p)
        count += 1
    if count == 0:
        return 0.0, 0.0, 1.0, 1.0
    avg = loglik / count
    geom = math.exp(avg)
    ppl = math.exp(-avg)
    return loglik, avg, ppl, geom


def score_sequence_local(dna: str, model_name: str = "evo2_7b") -> Dict[str, float]:
    """Score DNA with local Evo 2; returns host-agnostic stats dict.

    Uses Evo2 tokenizer to compute next-token probabilities for each position.
    """
    if not _HAS_LOCAL:
        raise RuntimeError("Evo2 local backend not available")
    model = _LazyLocalModel.get(model_name)
    import torch

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    input_ids = torch.tensor(
        model.tokenizer.tokenize(dna), dtype=torch.long
    ).unsqueeze(0).to(device)
    outputs, _ = model(input_ids)
    logits = outputs[0]  # (B, L, vocab)
    # Convert logits to per-position next-token probabilities of the actually observed token
    with torch.no_grad():
        # Shift logits to align with next-token prediction
        shifted_logits = logits[:, :-1, :]
        next_ids = input_ids[:, 1:]
        probs = torch.softmax(shifted_logits, dim=-1)
        # Gather probabilities of the true next token
        p_true = probs.gather(-1, next_ids.unsqueeze(-1)).squeeze(-1)
        seq_probs = tuple(float(x) for x in p_true[0])

    loglik, avg, ppl, geom = _score_from_token_probs(seq_probs)
    return {
        "loglik": loglik,
        "avg_loglik": avg,
        "perplexity": ppl,
        "geom": geom,
    }


def score_sequence_nim(dna: str) -> Dict[str, float]:
    """Score DNA using NVIDIA hosted API/NIM.

    Expects env vars:
      - NVCF_RUN_KEY: API key
      - EVO2_NIM_URL (optional): override default URL
    """
    import requests

    key = os.getenv("NVCF_RUN_KEY")
    if not key:
        raise RuntimeError("NVCF_RUN_KEY not set for Evo2 NIM backend")
    url = os.getenv("EVO2_NIM_URL", "https://health.api.nvidia.com/v1/biology/arc/evo2-40b/generate")
    # Request small number of continuation tokens and sampled probs
    r = requests.post(
        url=url,
        headers={"Authorization": f"Bearer {key}"},
        json={
            "sequence": dna,
            "num_tokens": 1,  # we only need probabilities for next token(s)
            "top_k": 0,
            "enable_sampled_probs": True,
        },
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()
    # The exact schema may provide token-level probabilities. Here we conservatively
    # look for a field `probs` or `sampled_probs` and aggregate if present.
    probs = []
    if isinstance(data, dict):
        if "probs" in data and isinstance(data["probs"], list):
            probs = [float(p) for p in data["probs"]]
        elif "sampled_probs" in data and isinstance(data["sampled_probs"], list):
            probs = [float(p) for p in data["sampled_probs"]]
    loglik, avg, ppl, geom = _score_from_token_probs(tuple(probs))
    return {
        "loglik": loglik,
        "avg_loglik": avg,
        "perplexity": ppl,
        "geom": geom,
    }


def score_sequence(dna: str, model_name: str = "evo2_7b") -> Dict[str, float]:
    """Unified entry: prefer local, fallback to NIM if configured."""
    if _HAS_LOCAL:
        return score_sequence_local(dna, model_name=model_name)
    if _has_nim_env():
        return score_sequence_nim(dna)
    raise RuntimeError("No Evo2 backend available")


