"""Mutation effect demo: score a few hand-picked 4Z8J mutants vs WT.

Usage:
  .\\venv311\\Scripts\\python.exe -m src.eval.demo_mutations
Slow on CPU (each sequence = N forwards); keep to a few mutants.
"""
import torch
from src.model.proteinsolver import ProteinSolver
from src.data.pdb_to_graph import build_graph
from src.generate.score import score_mutations, AA2I
import yaml

with open("configs/protein_local_cpu.yaml") as f:
    cfg = yaml.safe_load(f)
device = torch.device("cpu")
model = ProteinSolver(num_node_classes=21, edge_in_dim=2,
                      dim=cfg["model"]["dim"], num_blocks=cfg["model"]["blocks"])
d = torch.load(cfg["train"]["ckpt_dir"] + "/last.pt", map_location=device)
model.load_state_dict(d["model"]); model.eval()

g = build_graph("data/raw/4z8j.pdb")
wt = g["x"]
# surface-ish small->smallish swaps + one core hydrophobic swap (positions illustrative)
muts = []
seq = "".join("ACDEFGHIKLMNPQRSTVWY"[i] for i in wt.tolist())
for pos in [10, 30, 50]:
    w = seq[pos]
    m = "A" if w != "A" else "S"
    muts.append(f"{w}{pos + 1}{m}")
wt_s, out = score_mutations(model, wt, g["edge_index"], g["edge_attr"], device, muts)
print(f"WT score={wt_s:.3f}")
for m, s in out.items():
    print(f"{m}: dscore={s:+.3f} ({'tolerated' if s > -0.5 else 'disruptive?'})")
print("OK mutations")
