"""Download PDBs from ids.txt (skip existing) and build cached graphs.

Usage:
  python -m src.data.build_cache --config configs/protein_local_cpu.yaml
Keeps chains with min_len <= N <= max_len, up to max_chains.
Cache: <cache_dir>/<pdbid>_<chain>.pt
"""
import argparse, os, yaml
import requests
from src.data.pdb_to_graph import build_graph


def load_cfg(p):
    with open(p) as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_cfg(args.config)["train"]
    ids = [l.strip().upper() for l in open("data/raw/ids.txt") if l.strip()]
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs(cfg["cache_dir"], exist_ok=True)
    kept, skipped = [], 0
    for pid in ids:
        if len(kept) >= cfg["max_chains"]:
            break
        pdbp = f"data/raw/{pid.lower()}.pdb"
        if not os.path.exists(pdbp):
            try:
                r = requests.get(f"https://files.rcsb.org/download/{pid}.pdb", timeout=60)
                if r.status_code != 200 or "ATOM" not in r.text:
                    continue
                open(pdbp, "w").write(r.text)
            except Exception:
                continue
        out = os.path.join(cfg["cache_dir"], f"{pid.lower()}_A.pt")
        if os.path.exists(out):
            import torch
            g = torch.load(out, weights_only=False)
            if cfg["min_len"] <= g["num_nodes"] <= cfg["max_len"]:
                kept.append(out)
            continue
        try:
            g = build_graph(pdbp)
        except Exception:
            continue
        n = g["num_nodes"]
        if not (cfg["min_len"] <= n <= cfg["max_len"]):
            skipped += 1
            continue
        import torch
        torch.save(g, out)
        kept.append(out)
        if len(kept) % 25 == 0:
            print(f"kept {len(kept)} (skipped_len {skipped})", flush=True)
    print(f"done: kept={len(kept)} skipped_len={skipped} cache={cfg['cache_dir']}")


if __name__ == "__main__":
    main()
