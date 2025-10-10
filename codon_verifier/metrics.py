
from typing import Dict, List, Tuple, Optional
import math
from collections import defaultdict
import shutil, subprocess

from .codon_utils import chunk_codons, validate_cds, CODON_TO_AA, AA_TO_CODONS, relative_adaptiveness_from_usage

########################
# Core sequence metrics
########################

def gc_content(seq: str) -> float:
    s = seq.upper().replace("U","T")
    if not s:
        return 0.0
    g = s.count("G"); c = s.count("C")
    return (g + c) / len(s)

def sliding_gc(seq: str, window: int = 50, step: int = 10) -> List[float]:
    s = seq.upper().replace("U","T")
    out = []
    for i in range(0, max(1, len(s)-window+1), step):
        win = s[i:i+window]
        if len(win) < window:
            break
        out.append(gc_content(win))
    return out

def homopolymers(seq: str, min_len: int = 6) -> List[Tuple[str,int,int]]:
    s = seq.upper().replace("U","T")
    res = []
    i = 0
    while i < len(s):
        j = i+1
        while j < len(s) and s[j] == s[i]:
            j += 1
        L = j - i
        if L >= min_len:
            res.append((s[i], i, L))
        i = j
    return res

def tandem_repeats(seq: str, min_unit: int = 2, max_unit: int = 6, min_repeats: int = 3) -> List[Tuple[int,int,str]]:
    s = seq.upper().replace("U","T")
    out = []; n = len(s)
    for k in range(min_unit, max_unit+1):
        i = 0
        while i + k*min_repeats <= n:
            motif = s[i:i+k]
            if any(b not in "ACGT" for b in motif): 
                i += 1; continue
            rep = 1; j = i + k
            while j + k <= n and s[j:j+k] == motif:
                rep += 1; j += k
            if rep >= min_repeats:
                out.append((i, rep*k, motif))
                i = j
            else:
                i += 1
    return out

########################
# CAI / tAI
########################

def cai(dna: str, usage: Dict[str, float]) -> float:
    ok, msg = validate_cds(dna)
    if not ok:
        raise ValueError(f"Invalid CDS for CAI: {msg}")
    w = relative_adaptiveness_from_usage(usage)
    codons = chunk_codons(dna)
    logs = []
    for c in codons:
        if CODON_TO_AA.get(c,"*") == "*":
            continue
        wi = max(1e-9, w.get(c, 1e-3))  # avoid log(0)
        logs.append(math.log(wi))
    return math.exp(sum(logs) / len(logs)) if logs else 0.0

def tai(dna: str, trna_weights: Dict[str, float]) -> float:
    ok, msg = validate_cds(dna)
    if not ok:
        raise ValueError(f"Invalid CDS for tAI: {msg}")
    fam_max = defaultdict(float)
    for aa, codons in AA_TO_CODONS.items():
        for c in codons:
            fam_max[aa] = max(fam_max[aa], trna_weights.get(c, 0.0))
    codons = chunk_codons(dna)
    logs = []
    for c in codons:
        aa = CODON_TO_AA.get(c, None)
        if aa is None or aa == "*": 
            continue
        raw = trna_weights.get(c, 0.0)
        denom = fam_max[aa] if fam_max[aa] > 0 else 1.0
        wi = raw/denom if denom>0 else 1.0
        wi = max(1e-9, wi)
        logs.append(math.log(wi))
    return math.exp(sum(logs)/len(logs)) if logs else 0.0

########################
# Rare codons / codon-pair
########################

def rare_codon_runs(dna: str, usage: Dict[str,float], quantile: float = 0.2, min_run: int = 3) -> List[Tuple[int,int]]:
    w = relative_adaptiveness_from_usage(usage)
    codons = chunk_codons(dna)
    fam_w = {}
    for aa, cods in AA_TO_CODONS.items():
        vals = [w[c] for c in cods if c in w]
        if not vals:
            continue
        vals_sorted = sorted(vals)
        idx = max(0, min(len(vals_sorted)-1, int(quantile*len(vals_sorted))))
        fam_w[aa] = vals_sorted[idx]
    rare = []
    for i,c in enumerate(codons):
        aa = CODON_TO_AA.get(c, None)
        if aa is None or aa == "*": 
            rare.append(False); continue
        thr = fam_w.get(aa, 0.0)
        rare.append(w.get(c, 0.0) <= thr)
    runs = []
    i=0
    while i < len(rare):
        if rare[i]:
            j=i+1
            while j < len(rare) and rare[j]:
                j+=1
            L=j-i
            if L>=min_run:
                runs.append((i,L))
            i=j
        else:
            i+=1
    return runs

def codon_pair_bias_score(dna: str, cpb: Optional[Dict[str,float]] = None) -> float:
    if cpb is None: 
        return 0.0
    codons = chunk_codons(dna)
    s = 0.0; n=0
    for a,b in zip(codons, codons[1:]):
        key = f"{a}-{b}"
        if key in cpb:
            s += cpb[key]; n += 1
    return (s/n) if n>0 else 0.0

########################
# Forbidden motifs and diversity
########################

def find_forbidden_sites(seq: str, motifs: List[str]) -> List[Tuple[str,int]]:
    s = seq.upper().replace("U","T")
    hits = []
    for m in motifs:
        mm = m.upper().replace("U","T")
        start = 0
        while True:
            k = s.find(mm, start)
            if k == -1:
                break
            hits.append((mm, k))
            start = k + 1
    return hits

def sequence_identity(a: str, b: str) -> float:
    """
    Simple global identity for equal-length sequences [0,1]. If different lengths, compare up to min length.
    """
    a = a.upper().replace("U","T"); b = b.upper().replace("U","T")
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    matches = sum(1 for i in range(n) if a[i]==b[i])
    return matches/float(n)

def min_identity_to_set(dna: str, references: List[str]) -> float:
    if not references:
        return 0.0
    return min(sequence_identity(dna, r) for r in references)

########################
# Structural/extra feature terms
########################

def lm_feature_terms(lm_feats: dict) -> dict:
    """Normalise LM-derived metrics to interpretable [0,1] terms."""
    if not lm_feats:
        return {
            "lm_host_term": 0.0,
            "lm_cond_term": 0.0,
            "lm_host_geom": 0.0,
            "lm_cond_geom": 0.0,
            "lm_host_perplexity": float("nan"),
            "lm_cond_perplexity": float("nan"),
        }

    def _extract(prefix: str) -> Tuple[float, float, float]:
        geom = float(lm_feats.get(f"{prefix}_geom", 0.0) or 0.0)
        score = float(lm_feats.get(f"{prefix}_score", geom) or geom)
        ppl = float(lm_feats.get(f"{prefix}_perplexity", float("nan")) or float("nan"))
        return geom, score, ppl

    host_geom, host_score, host_ppl = _extract("lm_host")
    cond_geom, cond_score, cond_ppl = _extract("lm_cond")

    return {
        "lm_host_term": max(0.0, min(1.0, host_score)),
        "lm_cond_term": max(0.0, min(1.0, cond_score)),
        "lm_host_geom": host_geom,
        "lm_cond_geom": cond_geom,
        "lm_host_perplexity": host_ppl,
        "lm_cond_perplexity": cond_ppl,
    }


def extra_feature_terms(extra: dict) -> dict:
    """
    Map provided extra features (AlphaFold/ESM/Evo) to [0,1] and combine.
    """
    if not extra:
        return {"feat_struct_term": 0.0}
    # pLDDT
    pl = extra.get("plDDT_mean", None)
    plddt_term = max(0.0, min(1.0, (pl/100.0))) if isinstance(pl, (int,float)) else 0.0
    # MSA depth (log-scale)
    msa = extra.get("msa_depth", None)
    msa_term = 0.0
    if isinstance(msa, (int,float)):
        msa_term = max(0.0, min(1.0, math.log10(1.0+msa)/3.0))  # ~1k depth -> ~1
    # Conservation in [0,1]
    cons = extra.get("conservation_mean", None)
    cons_term = max(0.0, min(1.0, float(cons))) if isinstance(cons, (int,float)) else 0.0
    # Hydropathy (prefer not extremely hydrophobic)
    kd = extra.get("kd_hydropathy_mean", None)
    kd_term = 0.0
    if isinstance(kd, (int,float)):
        kd_term = max(0.0, min(1.0, 1.0 - max(0.0, kd - 2.0)/4.0))
    feat_struct_term = 0.4*plddt_term + 0.2*msa_term + 0.2*cons_term + 0.2*kd_term
    return {"feat_struct_term": feat_struct_term}

########################
# 5' mRNA structure proxy
########################

def five_prime_structure_proxy(dna: str, window_nt: int = 45) -> float:
    s = dna.upper().replace("U","T")
    if len(s) < 3 + window_nt:
        w = s[3:]
    else:
        w = s[3:3+window_nt]
    comp = str.maketrans("ACGT","TGCA")
    revcomp = w.translate(comp)[::-1]
    score = 0
    for L in (5,4,3):
        for i in range(0, max(0, len(w)-L+1)):
            sub = w[i:i+L]
            if revcomp.find(sub) != -1:
                score += (6-L)
    return -float(score)

########################
# ViennaRNA-based 5' ΔG (optional)
########################

def _rnalfold_available() -> bool:
    return shutil.which("RNAfold") is not None

def five_prime_dG_vienna(dna: str, window_nt: int = 45) -> Optional[float]:
    """
    Compute 5' window minimum free energy ΔG (kcal/mol) using ViennaRNA.
    Priority: Python bindings (RNA) if available; otherwise use RNAfold CLI.
    Returns None if neither backend is available or sequence is too short.
    """
    s = dna.upper().replace("U","T")
    if len(s) < 3:
        return None
    # Use region after start codon, matching proxy windowing
    region = s[3:3+window_nt] if len(s) > 3 else ""
    if not region:
        return None
    # Try Python bindings
    try:
        import RNA  # type: ignore
        fc = RNA.fold_compound(region.replace("T","U"))
        structure, mfe = fc.mfe()
        return float(mfe)
    except Exception:
        pass
    # Try CLI fallback
    if _rnalfold_available():
        try:
            proc = subprocess.run(
                ["RNAfold", "--noPS"],
                input=(region.replace("T","U") + "\n").encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            out = proc.stdout.decode("utf-8", errors="ignore").strip().splitlines()
            # Expected output: line1=sequence, line2=structure ( dG )
            if len(out) >= 2:
                line = out[1]
                # Parse energy inside parentheses, e.g., "... (-3.40)"
                l = line.rfind("(")
                r = line.rfind(")")
                if l != -1 and r != -1 and r > l:
                    val = line[l+1:r].strip()
                    return float(val)
        except Exception:
            return None
    return None

########################
# Aggregate rule score
########################

def rules_score(
    dna: str,
    usage: Dict[str,float],
    lm_features: Optional[dict] = None,
    extra_features: Optional[dict] = None,
    trna_w: Optional[Dict[str,float]] = None,
    cpb: Optional[Dict[str,float]] = None,
    motifs: Optional[List[str]] = None,
    weights: Optional[Dict[str,float]] = None,
    gc_target: Tuple[float,float] = (0.35, 0.65),
    window_gc: Tuple[int,float,float] = (50, 0.30, 0.70),
    rare_quantile: float = 0.2,
    rare_min_run: int = 3,
    homopoly_min: int = 6,
    use_vienna_dG: bool = True,
    dG_threshold: float = -5.0,
    dG_range: float = 10.0,
    diversity_refs: Optional[List[str]] = None,
    diversity_max_identity: float = 0.98,
) -> Dict[str, float]:
    """
    Combine rule-based metrics into a single total score.
    """
    if weights is None:
        weights = {
            "lm_host": 0.6,
            "lm_cond": 0.25,
            "cai": 1.0,
            "tai": 0.5,
            "gc": 0.5,
            "win_gc": 0.5,
            "struct5": 0.5,
            "struct5_dG": 0.5,
            "forbidden": -1.0,
            "rare_runs": -0.5,
            "homopoly": -0.3,
            "cpb": 0.2,
            "feat_struct": 0.3,
            "diversity": 0.3,
        }
    ok, msg = validate_cds(dna)
    if not ok:
        raise ValueError(f"Invalid CDS: {msg}")

    _cai = cai(dna, usage)
    _tai = tai(dna, trna_w) if trna_w is not None else 0.0
    _gc = gc_content(dna)
    gc_lo, gc_hi = gc_target
    gc_term = 1.0 - max(0.0, (gc_lo - _gc)/(gc_lo) if _gc < gc_lo else ( _gc - gc_hi )/(1.0-gc_hi) )
    gc_term = max(0.0, min(1.0, gc_term))

    win, wlo, whi = window_gc
    win_gcs = sliding_gc(dna, window=win, step=max(10, win//5))
    _win_gc = 1.0
    if win_gcs:
        bad = sum(1 for g in win_gcs if (g < wlo or g > whi))
        _win_gc = 1.0 - bad/len(win_gcs)

    _struct5 = five_prime_structure_proxy(dna)
    _dG = five_prime_dG_vienna(dna) if use_vienna_dG else None

    hits = find_forbidden_sites(dna, motifs or [])
    _forbidden = -float(len(hits))

    runs = rare_codon_runs(dna, usage, quantile=rare_quantile, min_run=rare_min_run)
    _rare = -float(sum(L for _,L in runs))

    homos = homopolymers(dna, min_len=homopoly_min)
    _hpoly = -float(sum(L for _,_,L in homos))

    _cpb = codon_pair_bias_score(dna, cpb)

    lm_terms = lm_feature_terms(lm_features or {})
    _extra = extra_feature_terms(extra_features or {})

    # Diversity term: penalize high identity to references (> threshold)
    div_term = 0.0
    if diversity_refs:
        min_id = min_identity_to_set(dna, diversity_refs)
        if min_id > diversity_max_identity:
            # Linear penalty from threshold to 1.0
            div_term = - (min_id - diversity_max_identity) / max(1e-6, 1.0 - diversity_max_identity)

    struct_norm = 1.0 / (1.0 + math.exp(-_struct5/3.0))
    # ΔG term: reward sequences whose ΔG is above threshold (less structured)
    dG_term = 0.0
    if _dG is not None:
        if _dG >= dG_threshold:
            dG_term = 1.0
        else:
            # Linear ramp down over dG_range (kcal/mol)
            dG_term = max(0.0, 1.0 - (dG_threshold - _dG)/max(1e-6, dG_range))
    total = (
        weights["lm_host"] * lm_terms["lm_host_term"] +
        weights["lm_cond"] * lm_terms["lm_cond_term"] +
        weights["cai"] * _cai +
        weights["tai"] * _tai +
        weights["gc"] * gc_term +
        weights["win_gc"] * _win_gc +
        weights["struct5"] * struct_norm +
        weights["struct5_dG"] * dG_term +
        weights["forbidden"] * (-_forbidden) +
        weights["rare_runs"] * (-_rare) +
        weights["homopoly"] * (-_hpoly) +
        weights["cpb"] * _cpb +
        weights["feat_struct"] * _extra.get("feat_struct_term", 0.0)
        + weights["diversity"] * div_term
    )
    return {
        "lm_host_term": lm_terms["lm_host_term"],
        "lm_cond_term": lm_terms["lm_cond_term"],
        "lm_host_geom": lm_terms["lm_host_geom"],
        "lm_cond_geom": lm_terms["lm_cond_geom"],
        "lm_host_perplexity": lm_terms["lm_host_perplexity"],
        "lm_cond_perplexity": lm_terms["lm_cond_perplexity"],
        "cai": _cai,
        "tai": _tai,
        "gc": _gc,
        "gc_term": gc_term,
        "win_gc_term": _win_gc,
        "struct5_proxy": _struct5,
        "dG_vienna": _dG if _dG is not None else float("nan"),
        "dG_term": dG_term,
        "forbidden_hits": len(hits),
        "rare_run_len": -_rare,
        "homopoly_len": -_hpoly,
        "cpb": _cpb,
        "feat_struct_term": _extra.get("feat_struct_term", 0.0),
        "diversity_term": div_term,
        "total_rules": total,
    }
