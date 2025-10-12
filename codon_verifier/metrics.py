
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

def fop(dna: str, usage: Dict[str, float]) -> float:
    """
    Calculate Frequency of Optimal Codons (FOP).
    
    FOP = (number of optimal codons) / (total codons excluding stop codons)
    
    An optimal codon is defined as the codon with highest usage frequency
    for each amino acid in the given usage table.
    
    Args:
        dna: DNA sequence (must be valid CDS)
        usage: Codon usage frequency table
        
    Returns:
        FOP score in range [0, 1]
    """
    ok, msg = validate_cds(dna)
    if not ok:
        raise ValueError(f"Invalid CDS for FOP: {msg}")
    
    # Identify optimal codon for each amino acid
    optimal_codons = {}
    for aa, codon_list in AA_TO_CODONS.items():
        if aa == "*":  # Skip stop codons
            continue
        max_usage = -1.0
        optimal = None
        for c in codon_list:
            if c in usage and usage[c] > max_usage:
                max_usage = usage[c]
                optimal = c
        if optimal:
            optimal_codons[aa] = optimal
    
    # Count optimal codons in sequence
    codons = chunk_codons(dna)
    optimal_count = 0
    total_count = 0
    
    for c in codons:
        aa = CODON_TO_AA.get(c, None)
        if aa is None or aa == "*":
            continue
        total_count += 1
        if aa in optimal_codons and c == optimal_codons[aa]:
            optimal_count += 1
    
    return optimal_count / total_count if total_count > 0 else 0.0

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
    """
    Calculate average Codon Pair Bias (CPB) score.
    
    CPB scores measure deviation from expected codon pair frequencies.
    Negative values indicate under-represented (disfavored) pairs.
    Positive values indicate over-represented (favored) pairs.
    
    Args:
        dna: DNA sequence
        cpb: Dictionary mapping "CODON1-CODON2" to CPB score
        
    Returns:
        Average CPB score across all pairs in sequence
    """
    if cpb is None: 
        return 0.0
    codons = chunk_codons(dna)
    s = 0.0; n=0
    for a,b in zip(codons, codons[1:]):
        key = f"{a}-{b}"
        if key in cpb:
            s += cpb[key]; n += 1
    return (s/n) if n>0 else 0.0

def codon_pair_score(dna: str, usage: Dict[str,float], codon_pair_freq: Optional[Dict[str,float]] = None) -> float:
    """
    Calculate Codon Pair Score (CPS).
    
    CPS = ln(F(AB) / (F(A) * F(B)))
    
    where F(AB) is observed frequency of codon pair AB,
    F(A) and F(B) are individual codon frequencies.
    
    If codon_pair_freq is not provided, calculates from expected frequencies
    based on codon usage (assumes independence, CPS = 0).
    
    Args:
        dna: DNA sequence (must be valid CDS)
        usage: Codon usage frequency table
        codon_pair_freq: Optional observed codon pair frequency table
        
    Returns:
        Average CPS across all codon pairs in sequence
    """
    ok, msg = validate_cds(dna)
    if not ok:
        raise ValueError(f"Invalid CDS for CPS: {msg}")
    
    codons = chunk_codons(dna)
    if len(codons) < 2:
        return 0.0
    
    # If no codon pair frequencies provided, CPS = 0 (independence assumption)
    if codon_pair_freq is None:
        return 0.0
    
    scores = []
    for a, b in zip(codons, codons[1:]):
        key = f"{a}-{b}"
        
        # Get observed pair frequency
        pair_freq = codon_pair_freq.get(key, None)
        if pair_freq is None:
            continue
            
        # Get individual codon frequencies
        freq_a = usage.get(a, 0.0)
        freq_b = usage.get(b, 0.0)
        
        # Calculate expected frequency (independence)
        expected = freq_a * freq_b
        
        # Avoid log(0) and division by zero
        if pair_freq > 0 and expected > 0:
            cps = math.log(pair_freq / expected)
            scores.append(cps)
    
    return sum(scores) / len(scores) if scores else 0.0

########################
# Dinucleotide statistics
########################

def count_dinucleotides(seq: str, dinucleotides: Optional[List[str]] = None) -> Dict[str, int]:
    """
    Count occurrences of specific dinucleotides in sequence.
    
    Args:
        seq: DNA/RNA sequence
        dinucleotides: List of dinucleotide patterns to count (e.g., ["CG", "TA"])
                      If None, counts all 16 possible dinucleotides
    
    Returns:
        Dictionary mapping dinucleotide to count
    """
    s = seq.upper().replace("U", "T")
    
    if dinucleotides is None:
        # Count all possible dinucleotides
        dinucleotides = [a+b for a in "ACGT" for b in "ACGT"]
    
    counts = {dn: 0 for dn in dinucleotides}
    
    for i in range(len(s) - 1):
        dinuc = s[i:i+2]
        if dinuc in counts:
            counts[dinuc] += 1
    
    return counts

def cpg_upa_content(seq: str) -> Dict[str, float]:
    """
    Calculate CpG and UpA (TpA in DNA) dinucleotide content.
    
    CpG islands and UpA dinucleotides are functionally important:
    - CpG: methylation sites, gene regulation (often under-represented in prokaryotes)
    - UpA (TpA): affects mRNA stability and translation (often avoided in optimized genes)
    
    Args:
        seq: DNA/RNA sequence
        
    Returns:
        Dictionary with:
        - cpg_count: Number of CG dinucleotides
        - upa_count: Number of TA dinucleotides  
        - cpg_freq: CG frequency (per 100 dinucleotides)
        - upa_freq: TA frequency (per 100 dinucleotides)
        - cpg_obs_exp: Observed/Expected ratio for CG
        - upa_obs_exp: Observed/Expected ratio for TA
    """
    s = seq.upper().replace("U", "T")
    
    if len(s) < 2:
        return {
            "cpg_count": 0, "upa_count": 0,
            "cpg_freq": 0.0, "upa_freq": 0.0,
            "cpg_obs_exp": 0.0, "upa_obs_exp": 0.0
        }
    
    # Count dinucleotides
    counts = count_dinucleotides(s, ["CG", "TA"])
    cpg_count = counts["CG"]
    upa_count = counts["TA"]
    
    # Count individual nucleotides
    c_count = s.count("C")
    g_count = s.count("G")
    t_count = s.count("T")
    a_count = s.count("A")
    
    total_dinuc = len(s) - 1
    
    # Calculate frequencies (per 100 dinucleotides)
    cpg_freq = (cpg_count / total_dinuc * 100) if total_dinuc > 0 else 0.0
    upa_freq = (upa_count / total_dinuc * 100) if total_dinuc > 0 else 0.0
    
    # Calculate observed/expected ratios
    # Expected frequency = P(C) * P(G) for independent nucleotides
    cpg_expected = (c_count / len(s)) * (g_count / len(s)) * total_dinuc if len(s) > 0 else 0.0
    upa_expected = (t_count / len(s)) * (a_count / len(s)) * total_dinuc if len(s) > 0 else 0.0
    
    cpg_obs_exp = cpg_count / cpg_expected if cpg_expected > 0 else 0.0
    upa_obs_exp = upa_count / upa_expected if upa_expected > 0 else 0.0
    
    return {
        "cpg_count": cpg_count,
        "upa_count": upa_count,
        "cpg_freq": cpg_freq,
        "upa_freq": upa_freq,
        "cpg_obs_exp": cpg_obs_exp,
        "upa_obs_exp": upa_obs_exp
    }

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
# 5' UTR / mRNA structure analysis with MFE
########################

def _rnalfold_available() -> bool:
    return shutil.which("RNAfold") is not None

def _compute_mfe_vienna(sequence: str, temperature: float = 37.0) -> Optional[float]:
    """
    Compute minimum free energy ΔG (kcal/mol) using ViennaRNA at specified temperature.
    Priority: Python bindings (RNA) if available; otherwise use RNAfold CLI.
    Returns None if neither backend is available or sequence is too short.
    """
    if not sequence or len(sequence) < 3:
        return None
    
    # Convert DNA to RNA
    rna_seq = sequence.replace("T", "U")
    
    # Try Python bindings first
    try:
        import RNA  # type: ignore
        fc = RNA.fold_compound(rna_seq)
        # Set temperature (in Celsius)
        fc.temperature = temperature
        structure, mfe = fc.mfe()
        return float(mfe)
    except Exception:
        pass
    
    # Try CLI fallback
    if _rnalfold_available():
        try:
            # RNAfold with temperature specification
            cmd = ["RNAfold", "--noPS", f"--temp={temperature}"]
            proc = subprocess.run(
                cmd,
                input=(rna_seq + "\n").encode("utf-8"),
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

def five_prime_utr_mfe_analysis(dna: str, utr5_len: int = 0, temperature: float = 37.0) -> dict:
    """
    Compute 5' UTR / mRNA structure MFE analysis with two-tier window logic.
    
    Args:
        dna: DNA sequence (CDS only, no UTR)
        utr5_len: Length of 5' UTR (0 if no UTR)
        temperature: Temperature in Celsius for folding (default 37°C)
    
    Returns:
        dict with keys: mfe_5p_dG, mfe_global_dG, mfe_5p_note
    """
    result = {
        "mfe_5p_dG": None,
        "mfe_global_dG": None, 
        "mfe_5p_note": None
    }
    
    s = dna.upper().replace("U", "T")
    if len(s) < 3:  # Need at least start codon
        return result
    
    # Global MFE calculation (entire CDS)
    result["mfe_global_dG"] = _compute_mfe_vienna(s, temperature)
    
    # 5' structure analysis with two-tier window logic
    if utr5_len >= 20:
        # Full window: [-20..+50] relative to start codon
        # Since we only have CDS, we take [+1..+50] and note the limitation
        if len(s) >= 51:  # Need at least 50nt after start codon
            region_5p = s[1:51]  # [+1..+50]
            result["mfe_5p_dG"] = _compute_mfe_vienna(region_5p, temperature)
            result["mfe_5p_note"] = "utr_available_but_cds_only"
        else:
            region_5p = s[1:]  # Whatever is available after start codon
            result["mfe_5p_dG"] = _compute_mfe_vienna(region_5p, temperature)
            result["mfe_5p_note"] = "utr_available_but_short_cds"
    else:
        # Fallback window: [+1..+50] 
        if len(s) >= 51:  # Need at least 50nt after start codon
            region_5p = s[1:51]  # [+1..+50]
            result["mfe_5p_dG"] = _compute_mfe_vienna(region_5p, temperature)
            result["mfe_5p_note"] = "no_utr_fallback"
        else:
            region_5p = s[1:]  # Whatever is available after start codon
            result["mfe_5p_dG"] = _compute_mfe_vienna(region_5p, temperature)
            result["mfe_5p_note"] = "no_utr_fallback_short"
    
    return result

def five_prime_structure_proxy(dna: str, window_nt: int = 45) -> float:
    """
    Legacy function for backward compatibility.
    Uses simple palindrome-based scoring for 5' structure.
    """
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

def five_prime_dG_vienna(dna: str, window_nt: int = 45) -> Optional[float]:
    """
    Legacy function for backward compatibility.
    Compute 5' window minimum free energy ΔG (kcal/mol) using ViennaRNA.
    """
    s = dna.upper().replace("U","T")
    if len(s) < 3:
        return None
    # Use region after start codon, matching proxy windowing
    region = s[3:3+window_nt] if len(s) > 3 else ""
    if not region:
        return None
    return _compute_mfe_vienna(region, temperature=37.0)

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
    codon_pair_freq: Optional[Dict[str,float]] = None,
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
    utr5_len: int = 0,
    temperature: float = 37.0,
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
            "fop": 0.8,
            "gc": 0.5,
            "win_gc": 0.5,
            "struct5": 0.5,
            "struct5_dG": 0.5,
            "forbidden": -1.0,
            "rare_runs": -0.5,
            "homopoly": -0.3,
            "cpb": 0.2,
            "cps": 0.2,
            "cpg_penalty": -0.3,
            "upa_penalty": -0.2,
            "feat_struct": 0.3,
            "diversity": 0.3,
        }
    ok, msg = validate_cds(dna)
    if not ok:
        raise ValueError(f"Invalid CDS: {msg}")

    _cai = cai(dna, usage)
    _tai = tai(dna, trna_w) if trna_w is not None else 0.0
    _fop = fop(dna, usage)
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
    
    # Enhanced UTR-aware MFE analysis
    mfe_analysis = five_prime_utr_mfe_analysis(dna, utr5_len=utr5_len, temperature=temperature)
    _dG = mfe_analysis["mfe_5p_dG"] if use_vienna_dG else None
    _dG_global = mfe_analysis["mfe_global_dG"]
    _mfe_note = mfe_analysis["mfe_5p_note"]

    hits = find_forbidden_sites(dna, motifs or [])
    _forbidden = -float(len(hits))

    runs = rare_codon_runs(dna, usage, quantile=rare_quantile, min_run=rare_min_run)
    _rare = -float(sum(L for _,L in runs))

    homos = homopolymers(dna, min_len=homopoly_min)
    _hpoly = -float(sum(L for _,_,L in homos))

    _cpb = codon_pair_bias_score(dna, cpb)
    _cps = codon_pair_score(dna, usage, codon_pair_freq)
    
    # CpG/UpA content analysis
    dinuc_stats = cpg_upa_content(dna)
    # Penalize high CpG and UpA content (commonly avoided in optimized genes)
    # Use observed/expected ratio: ratio > 1 means over-represented
    cpg_penalty = max(0.0, dinuc_stats["cpg_obs_exp"] - 1.0)  # Penalty if over-represented
    upa_penalty = max(0.0, dinuc_stats["upa_obs_exp"] - 1.0)  # Penalty if over-represented

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
        weights["fop"] * _fop +
        weights["gc"] * gc_term +
        weights["win_gc"] * _win_gc +
        weights["struct5"] * struct_norm +
        weights["struct5_dG"] * dG_term +
        weights["forbidden"] * (-_forbidden) +
        weights["rare_runs"] * (-_rare) +
        weights["homopoly"] * (-_hpoly) +
        weights["cpb"] * _cpb +
        weights["cps"] * _cps +
        weights["cpg_penalty"] * cpg_penalty +
        weights["upa_penalty"] * upa_penalty +
        weights["feat_struct"] * _extra.get("feat_struct_term", 0.0) +
        weights["diversity"] * div_term
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
        "fop": _fop,
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
        "cps": _cps,
        "cpg_count": dinuc_stats["cpg_count"],
        "cpg_freq": dinuc_stats["cpg_freq"],
        "cpg_obs_exp": dinuc_stats["cpg_obs_exp"],
        "upa_count": dinuc_stats["upa_count"],
        "upa_freq": dinuc_stats["upa_freq"],
        "upa_obs_exp": dinuc_stats["upa_obs_exp"],
        "cpg_penalty": cpg_penalty,
        "upa_penalty": upa_penalty,
        "feat_struct_term": _extra.get("feat_struct_term", 0.0),
        "diversity_term": div_term,
        "total_rules": total,
        # Enhanced UTR-aware MFE fields
        "mfe_5p_dG": _dG if _dG is not None else float("nan"),
        "mfe_global_dG": _dG_global if _dG_global is not None else float("nan"),
        "mfe_5p_note": _mfe_note,
    }
