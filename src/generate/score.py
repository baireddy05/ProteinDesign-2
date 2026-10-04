"""Sequence scoring: mean log-prob of each residue given the rest + graph.

Score(seq) = mean_i log p(aa_i | rest, graph)  (mask position i, forward)
MutScore = Score(mut) - Score(wt). Higher = more compatible.
"""
import torch

AA = "ACDEFGHIKLMNPQRSTVWY"
AA2I = {a: i for i, a in enumerate(AA)}
MASK = 20


@torch.no_grad()
def per_residue_logprob(model, seq, edge_index, edge_attr, device):
    """seq: [N] long (0-19). Returns [N] log-probs of true AAs."""
    model.eval()
    n = seq.size(0)
    out = torch.empty(n)
    base = seq.to(device).clone()
    for i in range(n):
        x = base.clone()
        x[i] = MASK
        lp = model(x, edge_index.to(device), edge_attr.to(device)).log_softmax(-1)
        out[i] = lp[i, base[i]].cpu()
    return out


@torch.no_grad()
def score_sequence(model, seq, edge_index, edge_attr, device):
    return float(per_residue_logprob(model, seq, edge_index, edge_attr, device).mean())


def parse_mut(s):
    """'A12G' -> (pos0, from_idx, to_idx)."""
    wt, num, mut = s[0], int(s[1:-1]), s[-1]
    return num - 1, AA2I[wt], AA2I[mut]


@torch.no_grad()
def score_mutations(model, wt_seq, edge_index, edge_attr, device, muts):
    """muts: list like ['A12G']. Returns {mut: dscore}."""
    wt = score_sequence(model, wt_seq, edge_index, edge_attr, device)
    out = {}
    for m in muts:
        pos, w, m_ = parse_mut(m)
        assert wt_seq[pos] == w, f"WT mismatch at {m}"
        seq = wt_seq.clone()
        seq[pos] = m_
        out[m] = score_sequence(model, seq, edge_index, edge_attr, device) - wt
    return wt, out
