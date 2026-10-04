"""Protein masked-sequence training: 1 graph/step, grad accumulation.

Usage:
  .\\venv311\\Scripts\\python.exe -m src.train.train_protein --config configs/protein_local_cpu.yaml
"""
import argparse, os, time, yaml
import torch
import torch.nn as nn
from src.model.proteinsolver import ProteinSolver
from src.data.protein_dataset import ProteinGraphDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    torch.manual_seed(cfg.get("seed", 0))
    torch.set_num_threads(cfg.get("num_threads", 8))
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("use_cuda", True) else "cpu")
    print(f"device={device}", flush=True)

    mcfg, tcfg = cfg["model"], cfg["train"]
    model = ProteinSolver(num_node_classes=21, edge_in_dim=2,
                          dim=mcfg["dim"], num_blocks=mcfg["blocks"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tcfg["lr"])
    use_amp = device.type == "cuda" and tcfg.get("amp", False)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ce = nn.CrossEntropyLoss()
    ckpt_dir = tcfg["ckpt_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    last = os.path.join(ckpt_dir, "last.pt")
    step = 0
    if os.path.exists(last):
        d = torch.load(last, map_location=device)
        model.load_state_dict(d["model"]); opt.load_state_dict(d["opt"]); step = d["step"]
        print(f"resumed at step {step}", flush=True)

    ds = ProteinGraphDataset(tcfg["cache_dir"], mask_frac=tcfg.get("mask_frac", 0.5))
    print(f"train graphs: {len(ds)}", flush=True)
    max_steps, accum = tcfg["max_steps"], tcfg.get("accum", 8)
    model.train()
    t0 = time.time()
    opt.zero_grad()
    while step < max_steps:
        item = ds[step % len(ds)]
        x = item["x"].to(device)
        y = item["y"].to(device)
        ei = item["edge_index"].to(device)
        ea = item["edge_attr"].to(device)
        logits = None
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(x, ei, ea)
            m = x == 20
            loss = ce(logits[m], y[m]) / accum
        scaler.scale(loss).backward()
        if (step + 1) % accum == 0:
            scaler.step(opt); scaler.update(); opt.zero_grad()
        step += 1
        if step % tcfg.get("log_every", 20) == 0:
            with torch.no_grad():
                pred = logits.argmax(-1)
                acc = (pred[m] == y[m]).float().mean().item()
            print(f"step {step}/{max_steps} loss={loss.item()*accum:.3f} "
                  f"acc_mask={acc:.3f} N={x.size(0)} {time.time()-t0:.1f}s", flush=True)
        if step % tcfg.get("save_every", 100) == 0:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, last)
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, last)
    print("done", flush=True)


if __name__ == "__main__":
    main()
