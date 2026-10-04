"""Sudoku eval: one-shot vs incremental (most-confident-first) decoding.

Usage:
  .\\venv311\\Scripts\\python.exe -m src.eval.eval_sudoku --config configs/local_cpu.yaml --num 200
Loads checkpoints/<...>/last.pt, tests on fresh puzzles (different seed).
"""
import argparse, torch, yaml
from src.model.proteinsolver import ProteinSolver
from src.data.sudoku import SudokuDataset, sudoku_edge_index


def load_cfg(p):
    with open(p) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def eval_oneshot(model, puzzles, ei, ea, device):
    correct, total = 0, 0
    for x, y in puzzles:
        x, y = x.to(device), y.to(device)
        pred = model(x, ei, ea).argmax(-1)
        m = x == 0
        correct += (pred[m] == y[m]).sum().item()
        total += m.sum().item()
    return correct / max(total, 1)


@torch.no_grad()
def eval_incremental(model, puzzles, ei, ea, device):
    correct, total, solved = 0, 0, 0
    for x, y in puzzles:
        x = x.to(device).clone()
        y = y.to(device)
        cur = x.clone()
        mask = cur == 0
        while mask.any():
            logits = model(cur, ei, ea)  # [81,10]
            prob = logits.softmax(-1)
            conf, pred = prob.max(-1)  # [81]
            conf = conf.masked_fill(~mask, -1.0)
            j = int(conf.argmax())
            cur[j] = pred[j]
            mask = cur == 0
        correct += (cur[x == 0] == y[x == 0]).sum().item()
        total += (x == 0).sum().item()
        if (cur == y).all():
            solved += 1
    return correct / max(total, 1), solved / len(puzzles)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--num", type=int, default=200)
    ap.add_argument("--inc_num", type=int, default=50)
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("use_cuda", True) else "cpu")
    model = ProteinSolver(num_node_classes=10, edge_in_dim=1,
                          dim=cfg["model"]["dim"], num_blocks=cfg["model"]["blocks"]).to(device)
    ckpt = cfg["train"]["ckpt_dir"] + "/last.pt"
    d = torch.load(ckpt, map_location=device)
    model.load_state_dict(d["model"])
    model.eval()
    print(f"loaded {ckpt} @ step {d['step']}")
    ei = sudoku_edge_index().to(device)
    ea = torch.ones(ei.size(1), 1, device=device)

    ds = SudokuDataset(size=args.num, n_mask=cfg["train"].get("n_mask", 40), seed=999)
    puzzles = [ds[i] for i in range(args.num)]  # NOTE: single call per i; ds[i][0],ds[i][1] would draw twice
    acc1 = eval_oneshot(model, puzzles, ei, ea, device)
    print(f"one-shot masked acc: {acc1*100:.1f}% on {args.num} puzzles")

    ds2 = SudokuDataset(size=args.inc_num, n_mask=cfg["train"].get("n_mask", 40), seed=1234)
    puzzles2 = [ds2[i] for i in range(args.inc_num)]
    acc2, solved = eval_incremental(model, puzzles2, ei, ea, device)
    print(f"incremental masked acc: {acc2*100:.1f}%, fully solved: {solved*100:.1f}% on {args.inc_num} puzzles")


if __name__ == "__main__":
    main()
