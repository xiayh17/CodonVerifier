
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


# Host selector dictionary
HOST_TABLES = {
    "E_coli": (E_COLI_USAGE, E_COLI_TRNA),
    "Human": (HUMAN_USAGE, HUMAN_TRNA),
    "Mouse": (MOUSE_USAGE, MOUSE_TRNA),
    "S_cerevisiae": (S_CEREVISIAE_USAGE, S_CEREVISIAE_TRNA),
    "P_pastoris": (P_PASTORIS_USAGE, P_PASTORIS_TRNA),
}


def get_host_tables(host: str) -> tuple[Dict[str, float], Dict[str, float]]:
    """
    Get codon usage and tRNA tables for a given host.
    
    Args:
        host: Host organism name (E_coli, Human, Mouse, S_cerevisiae, P_pastoris)
    
    Returns:
        Tuple of (usage_table, trna_weights)
    
    Raises:
        ValueError: If host is not recognized
    """
    if host not in HOST_TABLES:
        raise ValueError(
            f"Unknown host: {host}. Available hosts: {', '.join(HOST_TABLES.keys())}"
        )
    return HOST_TABLES[host]
