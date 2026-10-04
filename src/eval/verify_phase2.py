"""Phase 2 verification: build 4 reference graphs + protein forward pass."""
import time
import torch
from src.model.proteinsolver import ProteinSolver
from src.data.pdb_to_graph import build_graph

PDES = ["data/raw/1n5u.pdb", "data/raw/4z8j.pdb", "data/raw/4unu.pdb", "data/raw/1oc7.pdb"]

model = ProteinSolver(num_node_classes=21, edge_in_dim=2, dim=64, num_blocks=3)
model.eval()
for p in PDES:
    g = build_graph(p)
    t0 = time.time()
    with torch.no_grad():
        logits = model(g["x"], g["edge_index"], g["edge_attr"])
    dt = (time.time() - t0) * 1000
    assert logits.shape == (g["num_nodes"], 21), logits.shape
    assert g["edge_attr"].shape == (g["num_edges"], 2)
    print(f"{p}: N={g['num_nodes']} E={g['num_edges']} logits={tuple(logits.shape)} {dt:.0f}ms")
print("OK phase2")
