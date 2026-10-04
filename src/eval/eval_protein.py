"""Protein recon eval: one-shot masked accuracy on holdout graphs.

Usage:
  .\\venv311\\Scripts\\python.exe -m src.eval.eval_protein --config configs/protein_local_cpu.yaml
"""
import argparse
import torch
import yaml
from src.model.proteinsolver import ProteinSolver
from src.data.protein_dataset import ProteinGraphDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("use_cuda", True) else "cpu")
    m = cfg["model"]
    model = ProteinSolver(num_node_classes=21, edge_in_dim=2,
                          dim=m["dim"], num_blocks=m["blocks"]).to(device)
    ckpt = cfg["train"]["ckpt_dir"] + "/last.pt"
    d = torch.load(ckpt, map_location=device)
    model.load_state_dict(d["model"]); model.eval()
    print(f"loaded {ckpt} @ step {d['step']}")

    ds = ProteinGraphDataset(cfg["train"]["cache_dir"],
                             mask_frac=cfg["train"].get("mask_frac", 0.5),
                             train=False)  # holdout via default 0.1
    tot_c = tot = 0
    with torch.no_grad():
        for i in range(len(ds)):
            it = ds[i]
            logits = model(it["x"].to(device), it["edge_index"].to(device),
                           it["edge_attr"].to(device))
            pred = logits.argmax(-1).cpu()
            mk = it["x"] == 20
            tot_c += (pred[mk] == it["y"][mk]).sum().item()
            tot += mk.sum().item()
    print(f"holdout one-shot recon acc: {tot_c/tot*100:.1f}% on {len(ds)} graphs ({tot} masked cells)")


if __name__ == "__main__":
    main()
