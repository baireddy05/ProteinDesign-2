"""ProteinSolver network: embed -> N x ECA -> linear head."""
import torch.nn as nn
from .eca import ECABlock


class ProteinSolver(nn.Module):
    def __init__(
        self,
        num_node_classes: int = 10,  # sudoku: 0=empty + 1-9 ; protein: 20 AA + mask = 21
        edge_in_dim: int = 1,        # sudoku: 1 const ; protein: 2 (dist, seq_sep)
        dim: int = 64,
        num_blocks: int = 4,
    ):
        super().__init__()
        self.dim = dim
        self.node_emb = nn.Embedding(num_node_classes, dim)
        self.edge_emb = nn.Linear(edge_in_dim, dim)
        self.blocks = nn.ModuleList([ECABlock(dim) for _ in range(num_blocks)])
        self.head = nn.Linear(dim, num_node_classes)

    def forward(self, x, edge_index, edge_attr):
        """
        x: [N] long node labels
        edge_index: [2, E]
        edge_attr: [E, edge_in_dim] float
        returns logits: [N, num_classes]
        """
        v = self.node_emb(x)            # [N, D]
        e = self.edge_emb(edge_attr)    # [E, D]
        for blk in self.blocks:
            v, e = blk(v, e, edge_index)
        return self.head(v)
