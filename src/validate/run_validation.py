"""In-silico validation run: generate designs for a PDB topology and compare
WT vs designs on network score + physicochemical / packing metrics.

Usage:
  .\\venv311\\Scripts\\python.exe -m src.validate.run_validation --config configs/protein_local_cpu.yaml \\
      --pdb data/raw/4z8j.pdb --num 3 --temperature 1.0
"""
import argparse
import torch
import yaml
from src.model.proteinsolver import ProteinSolver
from src.data.pdb_to_graph import build_graph
from src.generate.sample import gen_incremental
from src.generate.score import score_sequence
from src.validate import metrics as M

AA = "ACDEFGHIKLMNPQRSTVWY"


def row(name, seq, ei, n, wt_comp, model, g, device):
    idx = torch.tensor([[AA.index(a)] for a in seq], dtype=torch.long).squeeze(-1)
    sc = score_sequence(model, idx, g["edge_index"], g["edge_attr"], device)
    ide = sum(a == b for a, b in zip(seq, WT[0])) / len(seq) * 100.0 if name != "WT" else 100.0
    comp = M.composition(seq)
    return {
        "name": name,
        "id%": ide if name != "WT" else 100.0,
        "score": sc,
        "cos": 1.0 if name == "WT" else M.cosine(wt_comp, comp),
        "kd": M.mean_kd(seq),
        "chg": M.net_charge(seq),
        "mw": M.mol_weight(seq) / 1000.0,
        "core_r": M.core_packing_r(ei, n, seq),
        "clash": M.charge_clashes(ei, seq),
        "seq": seq,
    }


WT = [""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--num", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.ckpt:
        import os
        cfg["train"]["ckpt_dir"] = os.path.dirname(args.ckpt) or "."
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("use_cuda", True) else "cpu")
    m = cfg["model"]
    model = ProteinSolver(num_node_classes=21, edge_in_dim=2,
                          dim=m["dim"], num_blocks=m["blocks"]).to(device)
    d = torch.load(cfg["train"]["ckpt_dir"] + "/last.pt", map_location=device)
    model.load_state_dict(d["model"]); model.eval()

    g = build_graph(args.pdb)
    n = g["num_nodes"]
    ei = g["edge_index"].to(device)
    ea = g["edge_attr"].to(device)
    wt = "".join(AA[i] for i in g["x"].tolist())
    WT[0] = wt
    wt_comp = M.composition(wt)

    rows = [row("WT", wt, ei, n, wt_comp, model, g, device)]
    rng = torch.Generator().manual_seed(args.seed)
    for k in range(args.num):
        with torch.no_grad():
            out = gen_incremental(model, ei, ea, device, n, sample=True,
                                  temp=args.temperature, rng=rng)
        s = "".join(AA[i] for i in out.cpu().tolist())
        rows.append(row(f"D{k + 1}", s, ei, n, wt_comp, model, g, device))

    print(f"{'name':<5}{'id%':>6}{'score':>8}{'cos':>6}{'kd':>7}{'chg':>7}{'kDa':>7}{'core_r':>8}{'clash':>7}")
    for r in rows:
        print(f"{r['name']:<5}{r['id%']:>6.1f}{r['score']:>8.3f}{r['cos']:>6.3f}"
              f"{r['kd']:>7.3f}{r['chg']:>7.1f}{r['mw']:>7.2f}{r['core_r']:>8.3f}{r['clash']:>7.2f}")
    for r in rows:
        print(f">{r['name']}\n{r['seq']}")
    print("OK validation")


if __name__ == "__main__":
    main()
