"""Residual Edge Convolution and Aggregation (ECA) block.

Paper: Strokach et al. Cell Systems 2020, Fig 1A.
  e_ij update: g_theta([v_i || v_j || e_ij])  (2-layer MLP)
  v_i update:  sum_{j in N(i)} h_theta(e_ij)  (linear)
Both with residual + LayerNorm + ReLU.
"""
import torch
import torch.nn as nn


class ECABlock(nn.Module):
    def __init__(self, dim: int = 64, hidden_mult: int = 1):
        super().__init__()
        h = dim * hidden_mult
        self.g = nn.Sequential(
            nn.Linear(3 * dim, h),
            nn.ReLU(),
            nn.Linear(h, dim),
        )
        self.h = nn.Linear(dim, dim)
        self.norm_e = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        self.act = nn.ReLU()

    def forward(self, v, e, edge_index):
        """
        v: [N, D] node embeddings
        e: [E, D] edge embeddings
        edge_index: [2, E] with row=src(i), col=dst(j). Must contain
                    both directions for undirected graphs.
        """
        row, col = edge_index[0], edge_index[1]
        vi = v[row]  # [E, D]
        vj = v[col]  # [E, D]
        inp = torch.cat([vi, vj, e], dim=-1)  # [E, 3D]
        de = self.g(inp)  # [E, D]
        e_new = self.norm_e(e + self.act(de))

        he = self.h(e_new)  # [E, D]
        # aggregate to source nodes (row). Since graph is bidirectional,
        # this sums all incident edges per node.
        # NOTE: under AMP autocast he can be half while v stays float
        # (embedding outputs are not autocast) -> match he's dtype.
        agg = torch.zeros(v.size(0), he.size(1), device=v.device, dtype=he.dtype)
        agg.index_add_(0, row, he)
        # normalize by degree to keep scale stable (mean-ish, keeps residual safe)
        deg = torch.zeros(v.size(0), 1, device=v.device)
        deg.index_add_(0, row, torch.ones_like(row, dtype=v.dtype).unsqueeze(-1))
        agg = agg / deg.clamp(min=1.0)

        v_new = self.norm_v(v + self.act(agg))
        return v_new, e_new
