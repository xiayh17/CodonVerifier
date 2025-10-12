
"""
Host-specific codon usage tables and tRNA weights.
Based on highly expressed genes from CoCoPUTs and Kazusa databases.
"""
from typing import Dict

# E. coli K-12 (highly expressed genes)
E_COLI_USAGE: Dict[str, float] = {
    "TTA": 0.05, "TTG": 0.13, "CTT": 0.10, "CTC": 0.10, "CTA": 0.04, "CTG": 0.58,
    "GGT": 0.35, "GGC": 0.37, "GGA": 0.16, "GGG": 0.12,
    "GCT": 0.18, "GCC": 0.27, "GCA": 0.23, "GCG": 0.32,
    "CGT": 0.36, "CGC": 0.36, "CGA": 0.07, "CGG": 0.07, "AGA": 0.07, "AGG": 0.07,
    "TCT": 0.18, "TCC": 0.22, "TCA": 0.15, "TCG": 0.12, "AGT": 0.16, "AGC": 0.17,
    "ACT": 0.22, "ACC": 0.43, "ACA": 0.20, "ACG": 0.15,
    "CCT": 0.17, "CCC": 0.17, "CCA": 0.23, "CCG": 0.43,
    "GTT": 0.29, "GTC": 0.21, "GTA": 0.17, "GTG": 0.33,
    "ATT": 0.48, "ATC": 0.39, "ATA": 0.13,
    "TTT": 0.57, "TTC": 0.43,
    "TAT": 0.58, "TAC": 0.42,
    "CAT": 0.57, "CAC": 0.43,
    "CAA": 0.34, "CAG": 0.66,
    "AAT": 0.46, "AAC": 0.54,
    "AAA": 0.76, "AAG": 0.24,
    "GAT": 0.63, "GAC": 0.37,
    "GAA": 0.68, "GAG": 0.32,
    "TGT": 0.46, "TGC": 0.54,
    "TGG": 1.0,
    "ATG": 1.0,
}

E_COLI_TRNA = {c: max(0.05, v) for c, v in E_COLI_USAGE.items()}

# E. coli Codon Pair Bias (CPB) scores
# Based on Coleman et al. (2008) Science and subsequent analyses
# Negative values = under-represented (disfavored), Positive = over-represented (favored)
# This is a curated subset of strongly biased pairs
E_COLI_CPB: Dict[str, float] = {
    # Strongly disfavored pairs (under-represented)
    "CGA-CGA": -0.45, "CGA-CGG": -0.38, "CGG-CGA": -0.42, "CGG-CGG": -0.40,
    "AGA-CGA": -0.35, "AGG-CGA": -0.33, "CGA-AGA": -0.36, "CGA-AGG": -0.34,
    "ATA-CGA": -0.32, "CGA-ATA": -0.30, "ATA-ATA": -0.28,
    "CTA-CTA": -0.25, "TTA-CTA": -0.22, "CTA-TTA": -0.23,
    
    # Moderately disfavored pairs
    "AGG-AGG": -0.20, "AGA-AGA": -0.18, "AGA-AGG": -0.19, "AGG-AGA": -0.19,
    "GCG-GCG": -0.15, "TCG-TCG": -0.14, "CCG-TCG": -0.13,
    "TTA-TTA": -0.17, "TTA-TTG": -0.12, "TTG-TTA": -0.12,
    
    # Favored pairs (over-represented) 
    "CTG-CTG": 0.35, "GAA-CTG": 0.28, "CTG-GAA": 0.30,
    "ACC-ACC": 0.25, "GCC-GCC": 0.22, "GCC-ACC": 0.20,
    "AAA-AAA": 0.18, "GAT-GAT": 0.16, "AAA-GAT": 0.15,
    "GTG-GTG": 0.20, "GGC-GGC": 0.18, "CGT-CGT": 0.17,
}


# Human (Homo sapiens) - highly expressed genes
HUMAN_USAGE: Dict[str, float] = {
    "TTA": 0.07, "TTG": 0.13, "CTT": 0.13, "CTC": 0.20, "CTA": 0.07, "CTG": 0.40,
    "GGT": 0.16, "GGC": 0.34, "GGA": 0.25, "GGG": 0.25,
    "GCT": 0.26, "GCC": 0.40, "GCA": 0.23, "GCG": 0.11,
    "CGT": 0.08, "CGC": 0.19, "CGA": 0.11, "CGG": 0.21, "AGA": 0.20, "AGG": 0.21,
    "TCT": 0.18, "TCC": 0.22, "TCA": 0.15, "TCG": 0.06, "AGT": 0.15, "AGC": 0.24,
    "ACT": 0.24, "ACC": 0.36, "ACA": 0.28, "ACG": 0.12,
    "CCT": 0.28, "CCC": 0.33, "CCA": 0.27, "CCG": 0.11,
    "GTT": 0.18, "GTC": 0.24, "GTA": 0.11, "GTG": 0.47,
    "ATT": 0.36, "ATC": 0.48, "ATA": 0.16,
    "TTT": 0.45, "TTC": 0.55,
    "TAT": 0.43, "TAC": 0.57,
    "CAT": 0.41, "CAC": 0.59,
    "CAA": 0.25, "CAG": 0.75,
    "AAT": 0.46, "AAC": 0.54,
    "AAA": 0.42, "AAG": 0.58,
    "GAT": 0.46, "GAC": 0.54,
    "GAA": 0.42, "GAG": 0.58,
    "TGT": 0.45, "TGC": 0.55,
    "TGG": 1.0,
    "ATG": 1.0,
}

HUMAN_TRNA = {c: max(0.05, v) for c, v in HUMAN_USAGE.items()}

# Human Codon Pair Bias (CPB) scores
# Based on analysis of highly expressed human genes
HUMAN_CPB: Dict[str, float] = {
    # Strongly disfavored pairs (CpG-containing and rare codon pairs)
    "CGA-CGA": -0.50, "CGG-CGG": -0.42, "CGA-CGG": -0.45, "CGG-CGA": -0.44,
    "TCG-TCG": -0.35, "GCG-GCG": -0.33, "CCG-CCG": -0.32,
    "TTA-TTA": -0.30, "CTA-CTA": -0.28, "TTA-CTA": -0.26, "CTA-TTA": -0.27,
    "ATA-ATA": -0.25, "GTA-GTA": -0.23,
    
    # Moderately disfavored pairs
    "AGA-AGA": -0.18, "TCG-ACG": -0.16, "ACG-TCG": -0.15,
    "GCG-ACG": -0.14, "CCG-GCG": -0.13,
    
    # Favored pairs (over-represented in highly expressed genes)
    "CTG-CTG": 0.40, "GAG-CTG": 0.32, "CTG-GAG": 0.33,
    "ACC-ACC": 0.28, "GCC-GCC": 0.26, "GCC-ACC": 0.24,
    "TCC-TCC": 0.22, "AGC-AGC": 0.21, "TCC-AGC": 0.19,
    "GTG-GTG": 0.25, "GGC-GGC": 0.23, "GTC-GTC": 0.20,
}


# Mouse (Mus musculus) - highly expressed genes
MOUSE_USAGE: Dict[str, float] = {
    "TTA": 0.06, "TTG": 0.12, "CTT": 0.12, "CTC": 0.20, "CTA": 0.07, "CTG": 0.43,
    "GGT": 0.17, "GGC": 0.33, "GGA": 0.26, "GGG": 0.24,
    "GCT": 0.27, "GCC": 0.41, "GCA": 0.22, "GCG": 0.10,
    "CGT": 0.09, "CGC": 0.17, "CGA": 0.11, "CGG": 0.20, "AGA": 0.21, "AGG": 0.22,
    "TCT": 0.19, "TCC": 0.22, "TCA": 0.14, "TCG": 0.06, "AGT": 0.15, "AGC": 0.24,
    "ACT": 0.24, "ACC": 0.37, "ACA": 0.27, "ACG": 0.12,
    "CCT": 0.29, "CCC": 0.32, "CCA": 0.28, "CCG": 0.11,
    "GTT": 0.17, "GTC": 0.23, "GTA": 0.11, "GTG": 0.49,
    "ATT": 0.35, "ATC": 0.49, "ATA": 0.16,
    "TTT": 0.43, "TTC": 0.57,
    "TAT": 0.43, "TAC": 0.57,
    "CAT": 0.40, "CAC": 0.60,
    "CAA": 0.25, "CAG": 0.75,
    "AAT": 0.44, "AAC": 0.56,
    "AAA": 0.40, "AAG": 0.60,
    "GAT": 0.45, "GAC": 0.55,
    "GAA": 0.41, "GAG": 0.59,
    "TGT": 0.44, "TGC": 0.56,
    "TGG": 1.0,
    "ATG": 1.0,
}

MOUSE_TRNA = {c: max(0.05, v) for c, v in MOUSE_USAGE.items()}

# Mouse CPB scores (similar to human with minor adjustments)
MOUSE_CPB: Dict[str, float] = {
    # Strongly disfavored pairs
    "CGA-CGA": -0.48, "CGG-CGG": -0.40, "CGA-CGG": -0.43, "CGG-CGA": -0.42,
    "TCG-TCG": -0.34, "GCG-GCG": -0.32, "CCG-CCG": -0.31,
    "TTA-TTA": -0.29, "CTA-CTA": -0.27, "ATA-ATA": -0.24,
    
    # Moderately disfavored pairs
    "AGA-AGA": -0.17, "TCG-ACG": -0.15, "GTA-GTA": -0.22,
    
    # Favored pairs
    "CTG-CTG": 0.42, "GAG-CTG": 0.33, "CTG-GAG": 0.34,
    "ACC-ACC": 0.29, "GCC-GCC": 0.27, "GCC-ACC": 0.25,
    "GTG-GTG": 0.26, "GGC-GGC": 0.24, "GTC-GTC": 0.21,
}


# Saccharomyces cerevisiae (baker's yeast) - highly expressed genes
S_CEREVISIAE_USAGE: Dict[str, float] = {
    "TTA": 0.26, "TTG": 0.27, "CTT": 0.12, "CTC": 0.06, "CTA": 0.13, "CTG": 0.11,
    "GGT": 0.47, "GGC": 0.19, "GGA": 0.22, "GGG": 0.12,
    "GCT": 0.38, "GCC": 0.22, "GCA": 0.29, "GCG": 0.11,
    "CGT": 0.14, "CGC": 0.06, "CGA": 0.07, "CGG": 0.04, "AGA": 0.48, "AGG": 0.21,
    "TCT": 0.26, "TCC": 0.16, "TCA": 0.21, "TCG": 0.10, "AGT": 0.16, "AGC": 0.11,
    "ACT": 0.35, "ACC": 0.22, "ACA": 0.30, "ACG": 0.14,
    "CCT": 0.31, "CCC": 0.15, "CCA": 0.42, "CCG": 0.12,
    "GTT": 0.39, "GTC": 0.21, "GTA": 0.21, "GTG": 0.19,
    "ATT": 0.46, "ATC": 0.26, "ATA": 0.27,
    "TTT": 0.59, "TTC": 0.41,
    "TAT": 0.56, "TAC": 0.44,
    "CAT": 0.64, "CAC": 0.36,
    "CAA": 0.69, "CAG": 0.31,
    "AAT": 0.59, "AAC": 0.41,
    "AAA": 0.58, "AAG": 0.42,
    "GAT": 0.65, "GAC": 0.35,
    "GAA": 0.70, "GAG": 0.30,
    "TGT": 0.63, "TGC": 0.37,
    "TGG": 1.0,
    "ATG": 1.0,
}

S_CEREVISIAE_TRNA = {c: max(0.05, v) for c, v in S_CEREVISIAE_USAGE.items()}

# S. cerevisiae CPB scores
S_CEREVISIAE_CPB: Dict[str, float] = {
    # Strongly disfavored pairs (rare codon combinations)
    "CGG-CGG": -0.55, "CGG-CGA": -0.50, "CGA-CGG": -0.52, "CGA-CGA": -0.48,
    "CGC-CGC": -0.40, "TCG-TCG": -0.38, "CCG-CCG": -0.36,
    "CTC-CTC": -0.35, "CTG-CTG": -0.32, "CTC-CTG": -0.30,
    
    # Moderately disfavored pairs
    "AGG-AGG": -0.25, "CGC-CGA": -0.28, "GCG-GCG": -0.22,
    
    # Favored pairs (frequently used in highly expressed genes)
    "TTA-TTA": 0.35, "TTG-TTG": 0.32, "TTA-TTG": 0.30, "TTG-TTA": 0.30,
    "GGT-GGT": 0.38, "GCT-GCT": 0.35, "GGT-GCT": 0.28,
    "AGA-AGA": 0.40, "ACT-ACT": 0.33, "AGA-ACT": 0.25,
    "CAA-CAA": 0.30, "GAA-GAA": 0.32, "GAT-GAT": 0.28,
}


# Pichia pastoris - highly expressed genes
P_PASTORIS_USAGE: Dict[str, float] = {
    "TTA": 0.18, "TTG": 0.40, "CTT": 0.19, "CTC": 0.07, "CTA": 0.08, "CTG": 0.08,
    "GGT": 0.42, "GGC": 0.18, "GGA": 0.28, "GGG": 0.12,
    "GCT": 0.42, "GCC": 0.20, "GCA": 0.28, "GCG": 0.10,
    "CGT": 0.18, "CGC": 0.06, "CGA": 0.09, "CGG": 0.04, "AGA": 0.53, "AGG": 0.10,
    "TCT": 0.28, "TCC": 0.14, "TCA": 0.23, "TCG": 0.10, "AGT": 0.16, "AGC": 0.09,
    "ACT": 0.38, "ACC": 0.20, "ACA": 0.32, "ACG": 0.10,
    "CCT": 0.33, "CCC": 0.13, "CCA": 0.42, "CCG": 0.12,
    "GTT": 0.41, "GTC": 0.19, "GTA": 0.22, "GTG": 0.18,
    "ATT": 0.47, "ATC": 0.25, "ATA": 0.28,
    "TTT": 0.57, "TTC": 0.43,
    "TAT": 0.56, "TAC": 0.44,
    "CAT": 0.62, "CAC": 0.38,
    "CAA": 0.65, "CAG": 0.35,
    "AAT": 0.57, "AAC": 0.43,
    "AAA": 0.60, "AAG": 0.40,
    "GAT": 0.62, "GAC": 0.38,
    "GAA": 0.68, "GAG": 0.32,
    "TGT": 0.60, "TGC": 0.40,
    "TGG": 1.0,
    "ATG": 1.0,
}

P_PASTORIS_TRNA = {c: max(0.05, v) for c, v in P_PASTORIS_USAGE.items()}

# P. pastoris CPB scores (similar pattern to S. cerevisiae)
P_PASTORIS_CPB: Dict[str, float] = {
    # Strongly disfavored pairs
    "CGG-CGG": -0.52, "CGA-CGA": -0.46, "CGG-CGA": -0.48, "CGA-CGG": -0.49,
    "CGC-CGC": -0.38, "TCG-TCG": -0.36, "CCG-CCG": -0.34,
    "CTC-CTC": -0.33, "CTG-CTG": -0.30, "CTA-CTA": -0.28,
    
    # Moderately disfavored pairs
    "AGG-AGG": -0.23, "GCG-GCG": -0.20, "ACG-ACG": -0.18,
    
    # Favored pairs
    "TTG-TTG": 0.45, "TTA-TTG": 0.38, "TTG-TTA": 0.40,
    "GGT-GGT": 0.40, "GCT-GCT": 0.37, "GGT-GCT": 0.30,
    "AGA-AGA": 0.42, "ACT-ACT": 0.35, "AGA-ACT": 0.28,
    "GAA-GAA": 0.33, "GAT-GAT": 0.30, "CAA-CAA": 0.28,
}


# Host selector dictionary (includes usage, tRNA, and CPB tables)
HOST_TABLES = {
    "E_coli": (E_COLI_USAGE, E_COLI_TRNA, E_COLI_CPB),
    "Human": (HUMAN_USAGE, HUMAN_TRNA, HUMAN_CPB),
    "Mouse": (MOUSE_USAGE, MOUSE_TRNA, MOUSE_CPB),
    "S_cerevisiae": (S_CEREVISIAE_USAGE, S_CEREVISIAE_TRNA, S_CEREVISIAE_CPB),
    "P_pastoris": (P_PASTORIS_USAGE, P_PASTORIS_TRNA, P_PASTORIS_CPB),
}


def get_host_tables(host: str, include_cpb: bool = True) -> tuple:
    """
    Get codon usage, tRNA weights, and optionally CPB tables for a given host.
    
    Args:
        host: Host organism name (E_coli, Human, Mouse, S_cerevisiae, P_pastoris)
        include_cpb: If True, returns (usage, trna, cpb). If False, returns (usage, trna)
    
    Returns:
        Tuple of (usage_table, trna_weights) or (usage_table, trna_weights, cpb_table)
    
    Raises:
        ValueError: If host is not recognized
    """
    if host not in HOST_TABLES:
        raise ValueError(
            f"Unknown host: {host}. Available hosts: {', '.join(HOST_TABLES.keys())}"
        )
    tables = HOST_TABLES[host]
    if include_cpb:
        return tables
    else:
        return tables[0], tables[1]
