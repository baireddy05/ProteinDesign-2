"""Resumable Sudoku training — works on local CPU and Colab Free T4.

Usage:
  python -m src.train.train_sudoku --config configs/local_cpu.yaml --steps 200
Resumes from ckpt.last if present.
"""
import argparse, os, yaml, time, torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.model.proteinsolver import ProteinSolver
from src.data.sudoku import SudokuDataset, sudoku_edge_index


def load_config(p):
    import yaml
    with open(p) as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)

    torch.manual_seed(cfg.get("seed", 0))
    torch.set_num_threads(cfg.get("num_threads", 8))

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("use_cuda", True) else "cpu")
    print(f"device={device}")

    model = ProteinSolver(
        num_node_classes=10, edge_in_dim=1,
        dim=cfg["model"]["dim"], num_blocks=cfg["model"]["blocks"],
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and cfg["train"].get("amp", False)))
    ce = nn.CrossEntropyLoss()

    ckpt_dir = cfg["train"]["ckpt_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    last = os.path.join(ckpt_dir, "last.pt")
    step = 0
    if os.path.exists(last):
        d = torch.load(last, map_location=device)
        model.load_state_dict(d["model"]); opt.load_state_dict(d["opt"]); step = d["step"]
        print(f"resumed at step {step}")

    max_steps = args.steps or cfg["train"]["max_steps"]
    bs = cfg["train"]["batch_size"]
    ds = SudokuDataset(size=10_000_000, n_mask=cfg["train"].get("n_mask", 40))
    dl = DataLoader(ds, batch_size=bs, collate_fn=SudokuDataset.collate,
                    num_workers=0, shuffle=False)
    ei_base = sudoku_edge_index().to(device)
    ea_base = torch.ones(ei_base.size(1), 1, device=device)

    model.train()
    t0 = time.time()
    it = iter(dl)
    while step < max_steps:
        try:
            xs, ys = next(it)
        except StopIteration:
            it = iter(dl); xs, ys = next(it)
        xs, ys = xs.to(device), ys.to(device)
        B = xs.size(0)
        # expand edge index per graph in batch -> block diagonal
        # simple loop (B<=64, N=81) keeps code Colab-safe without pyg batching
        opt.zero_grad()
        loss_acc = 0
        for b in range(B):
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and cfg["train"].get("amp", False))):
                logits = model(xs[b], ei_base, ea_base)  # [81,10]
                mask = xs[b] == 0
                loss = ce(logits[mask], ys[b][mask])
            scaler.scale(loss / B).backward()
            loss_acc += loss.item()
        scaler.step(opt); scaler.update()
        step += 1
        if step % cfg["train"].get("log_every", 20) == 0:
            # masked accuracy quick estimate on last batch
            with torch.no_grad():
                logits = model(xs[0], ei_base, ea_base)
                pred = logits.argmax(-1)
                m = xs[0] == 0
                acc = (pred[m] == ys[0][m]).float().mean().item()
            print(f"step {step}/{max_steps} loss={loss_acc/B:.3f} acc_mask={acc:.3f} {time.time()-t0:.1f}s", flush=True)
        if step % cfg["train"].get("save_every", 200) == 0:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, last)
            print(f"saved {last}")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, last)
    print("done")


if __name__ == "__main__":
    main()
