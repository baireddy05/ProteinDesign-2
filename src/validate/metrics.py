"""CPU-cheap sequence validation metrics (no Rosetta/MD needed).

All metrics are sequence-given-fixed-backbone-graph:
- composition similarity to WT (cosine)
- mean Kyte-Doolittle hydrophobicity
- net charge at pH 7, molecular weight
- core packing: Pearson r(node degree, KD) — buried (high-degree)
  positions should prefer hydrophobic residues in globular domains
- charge clashes: contacting same-sign D/E or K/R pairs per 100 contacts
"""
import math
import torch

KD = {  # Kyte-Doolittle
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9,
    "A": 1.8, "G": -0.4, "T": -0.7, "S": -0.8, "W": -0.9, "Y": -1.3,
    "P": -1.6, "H": -3.2, "E": -3.5, "Q": -3.5, "D": -3.5, "N": -3.5,
    "K": -3.9, "R": -4.5,
}
MW = {  # average residue masses (Da, water subtracted); +18.015 added once
    "A": 71.08, "R": 156.19, "N": 114.10, "D": 115.09, "C": 103.14,
    "E": 129.12, "Q": 128.13, "G": 57.05, "H": 137.14, "I": 113.16,
    "L": 113.16, "K": 128.17, "M": 131.19, "F": 147.18, "P": 97.12,
    "S": 87.08, "T": 101.11, "W": 186.21, "Y": 163.18, "V": 99.13,
}
PKA = {"D": 3.9, "E": 4.3, "H": 6.0, "C": 8.3, "Y": 10.1, "K": 10.5, "R": 12.5}
NEG = set("DE")
POS = set("KR")


def composition(seq):
    v = torch.zeros(20)
    order = "ACDEFGHIKLMNPQRSTVWY"
    for a in seq:
        v[order.index(a)] += 1
    return v / len(seq)


def cosine(a, b):
    return float((a @ b) / (a.norm() * b.norm() + 1e-9))


def mean_kd(seq):
    return sum(KD[a] for a in seq) / len(seq)


def net_charge(seq, ph=7.0):
    # termini + Henderson-Hasselbalch over ionizable side chains
    q = 1.0 / (1.0 + 10 ** (ph - 8.0)) - 1.0 / (1.0 + 10 ** (3.1 - ph))
    for a in seq:
        if a in NEG:
            q -= 1.0 / (1.0 + 10 ** (PKA[a] - ph))
        elif a in ("K", "R"):
            q += 1.0 / (1.0 + 10 ** (ph - PKA[a]))
        elif a == "H":
            q += 1.0 / (1.0 + 10 ** (ph - PKA[a]))
        elif a in ("C", "Y"):
            q -= 1.0 / (1.0 + 10 ** (PKA[a] - ph))
    return q


def mol_weight(seq):
    return sum(MW[a] for a in seq) + 18.015


def core_packing_r(edge_index, num_nodes, seq):
    """Pearson r between peer-degree and KD hydrophobicity."""
    deg = torch.zeros(num_nodes)
    src, dst = edge_index
    peer = src != dst
    deg.index_add_(0, src[peer], torch.ones(int(peer.sum())))
    h = torch.tensor([KD[a] for a in seq])
    d = deg - deg.mean()
    h = h - h.mean()
    return float((d @ h) / (d.norm() * h.norm() + 1e-9))


def charge_clashes(edge_index, seq):
    """Same-sign charged contacts (D/E-D/E or K/R-K/R) per 100 peer edges."""
    src, dst = edge_index
    m = src < dst  # undirected once (self-loops excluded by src<dst)
    n = 0
    for i, j in zip(src[m].tolist(), dst[m].tolist()):
        a, b = seq[i], seq[j]
        if (a in NEG and b in NEG) or (a in POS and b in POS):
            n += 1
    denom = int(m.sum())
    return n / max(denom, 1) * 100.0
