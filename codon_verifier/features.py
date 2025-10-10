
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple
import os, json, numpy as np

KYTE_DOOLITTLE = {
    "I": 4.5,"V": 4.2,"L": 3.8,"F": 2.8,"C": 2.5,"M": 1.9,"A": 1.8,"G": -0.4,"T": -0.7,"S": -0.8,
    "W": -0.9,"Y": -1.3,"P": -1.6,"H": -3.2,"E": -3.5,"Q": -3.5,"D": -3.5,"N": -3.5,"K": -3.9,"R": -4.5
}

@dataclass
class FeatureBundle:
    plDDT_mean: Optional[float] = None
    plDDT_min: Optional[float] = None
    tm_segments: Optional[int] = None
    disorder_fraction: Optional[float] = None
    esm_emb_dim: Optional[int] = None
    esm_emb_l2: Optional[float] = None
    msa_depth: Optional[float] = None
    conservation_mean: Optional[float] = None
    length: Optional[int] = None
    kd_hydropathy_mean: Optional[float] = None
    extras: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, float]:
        out = {}
        for k,v in asdict(self).items():
            if k=="extras" and isinstance(v, dict):
                out.update({f"extra_{ek}": ev for ek,ev in v.items()})
            elif v is not None:
                out[k] = float(v) if isinstance(v, (int,float, np.floating)) else v
        return out

def load_alphafold_plddt_from_pdb(pdb_path: str) -> Tuple[Optional[float], Optional[float]]:
    if not os.path.exists(pdb_path):
        return None, None
    vals = []
    with open(pdb_path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith(("ATOM","HETATM")):
                try:
                    bfac = float(line[60:66])
                    vals.append(bfac)
                except Exception:
                    continue
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.min(vals))

def load_npz_embeddings(npz_path: str) -> Tuple[Optional[int], Optional[float]]:
    if not os.path.exists(npz_path):
        return None, None
    data = np.load(npz_path)
    if "mean" in data:
        vec = data["mean"]
    elif "emb" in data:
        emb = data["emb"]
        vec = emb.mean(axis=0) if emb.ndim == 2 else emb
    else:
        key = list(data.keys())[0]
        vec = data[key]
        if hasattr(vec, "ndim") and vec.ndim > 1:
            vec = vec.mean(axis=0)
    dim = int(vec.shape[-1])
    l2 = float(np.linalg.norm(vec))
    return dim, l2

def load_json_features(json_path: str) -> Dict[str, float]:
    if not os.path.exists(json_path):
        return {}
    with open(json_path, "r") as f:
        obj = json.load(f)
    out = {}
    for k,v in obj.items():
        try:
            out[k] = float(v)
        except Exception:
            continue
    return out

def compute_sequence_props(aa: str) -> Tuple[int, Optional[float]]:
    aa = aa.strip().upper()
    lens = len(aa)
    if lens == 0:
        return 0, None
    vals = [KYTE_DOOLITTLE.get(a, 0.0) for a in aa]
    kd = sum(vals)/len(vals) if vals else None
    return lens, kd

def assemble_feature_bundle(
    aa: str,
    alphafold_pdb: Optional[str] = None,
    esm_npz: Optional[str] = None,
    evo_json: Optional[str] = None,
    extras_json: Optional[str] = None,
) -> FeatureBundle:
    pl_mean, pl_min = load_alphafold_plddt_from_pdb(alphafold_pdb) if alphafold_pdb else (None, None)
    dim, l2 = load_npz_embeddings(esm_npz) if esm_npz else (None, None)
    evo = load_json_features(evo_json) if evo_json else {}
    extra = load_json_features(extras_json) if extras_json else {}
    length, kd = compute_sequence_props(aa)
    fb = FeatureBundle(
        plDDT_mean=pl_mean, plDDT_min=pl_min,
        esm_emb_dim=dim, esm_emb_l2=l2,
        msa_depth=evo.get("msa_depth"), conservation_mean=evo.get("conservation_mean"),
        length=length, kd_hydropathy_mean=kd,
        extras=extra if extra else None
    )
    return fb
