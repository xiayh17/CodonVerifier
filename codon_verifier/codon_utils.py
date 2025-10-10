
from typing import Dict, List, Tuple, Optional

# Standard genetic code mapping AA -> codons (for 20 AAs + stop; STOP not used for scoring)
AA_TO_CODONS = {
    "F": ["TTT", "TTC"],
    "L": ["TTA", "TTG", "CTT", "CTC", "CTA", "CTG"],
    "I": ["ATT", "ATC", "ATA"],
    "M": ["ATG"],
    "V": ["GTT", "GTC", "GTA", "GTG"],
    "S": ["TCT", "TCC", "TCA", "TCG", "AGT", "AGC"],
    "P": ["CCT", "CCC", "CCA", "CCG"],
    "T": ["ACT", "ACC", "ACA", "ACG"],
    "A": ["GCT", "GCC", "GCA", "GCG"],
    "Y": ["TAT", "TAC"],
    "*": ["TAA", "TAG", "TGA"],
    "H": ["CAT", "CAC"],
    "Q": ["CAA", "CAG"],
    "N": ["AAT", "AAC"],
    "K": ["AAA", "AAG"],
    "D": ["GAT", "GAC"],
    "E": ["GAA", "GAG"],
    "C": ["TGT", "TGC"],
    "W": ["TGG"],
    "R": ["CGT", "CGC", "CGA", "CGG", "AGA", "AGG"],
    "G": ["GGT", "GGC", "GGA", "GGG"],
}

CODON_TO_AA = {}
for aa, codons in AA_TO_CODONS.items():
    for c in codons:
        CODON_TO_AA[c] = aa

def chunk_codons(dna: str) -> List[str]:
    dna = dna.upper().replace("U", "T")
    n = (len(dna) // 3) * 3
    return [dna[i:i+3] for i in range(0, n, 3)]

def validate_cds(dna: str) -> Tuple[bool, str]:
    s = dna.upper().replace("U","T")
    if len(s) % 3 != 0:
        return False, "Length not multiple of 3."
    if any(b not in "ACGT" for b in s):
        return False, "Non-ACGT characters found."
    codons = chunk_codons(s)
    if not codons:
        return False, "Empty CDS."
    if codons[0] != "ATG":
        return False, "CDS does not start with ATG."
    if CODON_TO_AA.get(codons[-1], "?") == "*":
        return False, "CDS ends with a STOP codon; exclude terminal stop for scoring."
    if any(CODON_TO_AA.get(c, "?") == "*" for c in codons):
        return False, "Internal STOP codon detected."
    return True, "OK"

def aa_from_dna(dna: str) -> str:
    codons = chunk_codons(dna)
    aa = []
    for c in codons:
        aa.append(CODON_TO_AA.get(c, "X"))
    return "".join(aa)

def sample_codon_for_aa(
    amino_acid: str,
    usage: Dict[str, float],
    temperature: float = 1.0,
    exclude_codons: Optional[List[str]] = None,
) -> str:
    codons = AA_TO_CODONS.get(amino_acid, [])
    if not codons:
        return "NNN"
    exclude = set(exclude_codons or [])
    candidates = [c for c in codons if c not in exclude]
    if not candidates:
        candidates = codons
    # Softmax over usage with temperature
    import math, random
    vals = [max(1e-6, usage.get(c, 1e-6)) for c in candidates]
    logs = [math.log(v) / max(1e-6, temperature) for v in vals]
    mx = max(logs)
    exps = [math.exp(x - mx) for x in logs]
    s = sum(exps)
    probs = [e/s for e in exps]
    r = random.random()
    cum = 0.0
    for c,p in zip(candidates, probs):
        cum += p
        if r <= cum:
            return c
    return candidates[-1]

def constrained_decode(
    protein_aa: str,
    usage: Dict[str, float],
    motifs_forbidden: Optional[List[str]] = None,
    max_attempts_per_pos: int = 5,
    temperature: float = 1.0,
) -> str:
    """
    Greedy left-to-right decoding with per-step resampling to avoid forbidden motifs.
    Does not append terminal STOP; starts with ATG for 'M' if first AA.
    """
    import random
    dna = []
    aa_seq = protein_aa.strip().upper()
    for i, aa in enumerate(aa_seq):
        attempts = 0
        chosen = None
        while attempts < max_attempts_per_pos:
            attempts += 1
            if i == 0 and aa == "M":
                cand = "ATG"
            else:
                cand = sample_codon_for_aa(aa, usage, temperature=temperature)
            trial = "".join(dna) + cand
            if not motifs_forbidden:
                chosen = cand; break
            bad = False
            for m in motifs_forbidden:
                mm = m.upper().replace("U","T")
                if mm in trial:
                    bad = True; break
            if not bad:
                chosen = cand; break
        if chosen is None:
            chosen = sample_codon_for_aa(aa, usage, temperature=temperature)
        dna.append(chosen)
    return "".join(dna)

def relative_adaptiveness_from_usage(usage: Dict[str, float]) -> Dict[str, float]:
    """
    Convert raw codon usage (frequency per codon within its synonymous family) to w_i for CAI:
    w_i = f_i / max(f_j for codons coding same AA). Missing codons get w=1.0 fallback.
    """
    w = {}
    for aa, codons in AA_TO_CODONS.items():
        fam = [c for c in codons if c in usage]
        if not fam:
            for c in codons:
                w[c] = 1.0
            continue
        maxf = max(usage[c] for c in fam)
        if maxf <= 0:
            for c in codons:
                w[c] = 1.0
            continue
        for c in codons:
            f = usage.get(c, 0.0)
            w[c] = (f / maxf) if maxf > 0 else 1.0
    return w
