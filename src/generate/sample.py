"""Sequence generation for a fixed topology graph.

Modes:
  oneshot     - 1 forward, argmax (or sample) every position. O(1).
  incremental - N forwards, each step fix most-confident position (argmax). O(N).
  sample      - incremental but SAMPLE residue at most-confident position (temperature).

Usage:
  .\\venv311\\Scripts\\python.exe -m src.generate.sample --config configs/protein_local_cpu.yaml \\
      --pdb data/raw/4z8j.pdb --mode sample --num 5 --temperature 1.0
Prints FASTA + identity to WT + network score.
"""
import argparse
import torch
import yaml
from src.model.proteinsolver import ProteinSolver
from src.data.pdb_to_graph import build_graph, MASK
from src.generate.score import score_sequence, AA2I

AA = "ACDEFGHIKLMNPQRSTVWY"


def load_model(cfg, device):
    m = cfg["model"]
    model = ProteinSolver(num_node_classes=21, edge_in_dim=2,
                          dim=m["dim"], num_blocks=m["blocks"]).to(device)
    d = torch.load(cfg["train"]["ckpt_dir"] + "/last.pt", map_location=device)
    model.load_state_dict(d["model"]); model.eval()
    return model, d["step"]


@torch.no_grad()
def gen_oneshot(model, ei, ea, device, n, sample=False, temp=1.0, rng=None):
    x = torch.full((n,), MASK, dtype=torch.long, device=device)
    logits = model(x, ei, ea)
    if sample:
        return torch.multinomial((logits / temp).softmax(-1), 1, generator=rng).squeeze(-1)
    return logits.argmax(-1)


@torch.no_grad()
def gen_incremental(model, ei, ea, device, n, sample=False, temp=1.0, rng=None):
    cur = torch.full((n,), MASK, dtype=torch.long, device=device)
    while bool((cur == MASK).any()):
        logits = model(cur, ei, ea)
        prob = (logits / temp).softmax(-1) if sample else logits.softmax(-1)
        conf, pred = prob.max(-1)
        conf = conf.masked_fill(cur != MASK, -1.0)
        j = int(conf.argmax())
        if sample:
            cur[j] = torch.multinomial(prob[j].unsqueeze(0), 1, generator=rng).squeeze()
        else:
            cur[j] = pred[j]
    return cur


def seq_id(a, b):
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--mode", choices=["oneshot", "incremental", "sample"], default="sample")
    ap.add_argument("--num", type=int, default=5)
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
    model, step = load_model(cfg, device)
    g = build_graph(args.pdb)
    n = g["num_nodes"]
    ei, ea = g["edge_index"].to(device), g["edge_attr"].to(device)
    wt = "".join(AA[i] for i in g["x"].tolist())
    wt_s = score_sequence(model, g["x"], g["edge_index"], g["edge_attr"], device)
    print(f"ckpt step {step} | WT len={n} score={wt_s:.3f}")
    print(f">WT\n{wt}")
    rng = torch.Generator().manual_seed(args.seed)
    for k in range(args.num):
        if args.mode == "oneshot":
            out = gen_oneshot(model, ei, ea, device, n, sample=False)
        elif args.mode == "incremental":
            out = gen_incremental(model, ei, ea, device, n, sample=False)
        else:
            out = gen_incremental(model, ei, ea, device, n, sample=True,
                                  temp=args.temperature, rng=rng)
        s = "".join(AA[i] for i in out.cpu().tolist())
        sc = score_sequence(model, out.cpu(), g["edge_index"], g["edge_attr"], device)
        print(f">gen{k + 1}_{args.mode} id={seq_id(s, wt)*100:.1f}% score={sc:.3f}\n{s}")


if __name__ == "__main__":
    main()
