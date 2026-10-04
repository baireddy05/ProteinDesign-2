"""PDB -> protein graph (nodes=residues, edges=CA pairs < cutoff).

MVP approximation: paper uses min heavy-atom distance; we use CA distance
(same 12A cutoff, same edge-feature semantics). Swap to all-atom later.

Node classes: 20 AA + mask(20) = 21. Masking applied by dataset, not here.
Edge feats [E,2]: [1/(1+d), min(|i-j|,32)/32].
Self-loops included (Sudoku convention: 1701 = 1620 + 81).

Usage:
  .\\venv311\\Scripts\\python.exe -m src.data.pdb_to_graph --pdb data/raw/1n5u.pdb
"""
import argparse
import torch
import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"
AA2I = {a: i for i, a in enumerate(AA)}
MASK = 20
RES3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "MSE": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y",
    "VAL": "V",
}
CUTOFF = 12.0
SEQ_CLIP = 32


def parse_chain(pdb_path, chain_id=None, model=1):
    """Returns (seq_str, coords[N,3] float32, chain_id used). Skips non-standard residues."""
    import biotite.structure.io.pdb as pdb_io
    f = pdb_io.PDBFile.read(pdb_path)
    s = f.get_structure(model=model)
    if chain_id is None:
        chains = sorted(set(s.chain_id.tolist()))
        # prefer longest protein chain
        best, best_n = chains[0], -1
        for c in chains:
            n = int(((s.chain_id == c) & (s.atom_name == "CA")).sum())
            if n > best_n:
                best, best_n = c, n
        chain_id = best
    s = s[s.chain_id == chain_id]
    # altloc filter (annotation name varies across biotite versions/files)
    try:
        alt = s.altloc_id
        keep = (alt == "") | (alt == "A")
        s = s[keep]
    except AttributeError:
        pass  # no altloc annotation present
    s = s[s.atom_name == "CA"]
    seq, xyz = [], []
    for res_name, coord in zip(s.res_name.tolist(), np.asarray(s.coord)):
        a = RES3TO1.get(res_name.strip().upper())
        if a is None:
            continue
        seq.append(a)
        xyz.append(coord)
    return "".join(seq), np.array(xyz, dtype=np.float32), chain_id


def build_graph(pdb_path, chain_id=None, cutoff=CUTOFF):
    seq, xyz, chain = parse_chain(pdb_path, chain_id)
    n = len(seq)
    if n == 0:
        raise ValueError(f"no residues parsed from {pdb_path}")
    x = torch.tensor([AA2I[a] for a in seq], dtype=torch.long)
    with torch.no_grad():
        d = torch.cdist(torch.from_numpy(xyz), torch.from_numpy(xyz))  # [N,N]
        adj = (d < cutoff)
        adj.fill_diagonal_(False)
        src, dst = torch.where(adj)
        # self-loops
        loop = torch.arange(n)
        src = torch.cat([src, loop]); dst = torch.cat([dst, loop])
        dd = d[src, dst]
        inv = 1.0 / (1.0 + dd)
        sep = torch.abs(src - dst).clamp(max=SEQ_CLIP).float() / SEQ_CLIP
        edge_attr = torch.stack([inv, sep], dim=-1)
        edge_index = torch.stack([src, dst], dim=0)
    return {
        "x": x, "edge_index": edge_index, "edge_attr": edge_attr,
        "seq": seq, "chain": chain, "xyz": torch.from_numpy(xyz),
        "num_nodes": n, "num_edges": int(edge_index.size(1)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--chain", default=None)
    ap.add_argument("--cutoff", type=float, default=CUTOFF)
    args = ap.parse_args()
    g = build_graph(args.pdb, args.chain, args.cutoff)
    n = g["num_nodes"]
    peers = g["num_edges"] - n
    print(f"{args.pdb} chain {g['chain']}: len={n} seq={g['seq'][:30]}...")
    print(f"edges: {g['num_edges']} total = {peers} peer + {n} self, "
          f"avg_degree={peers/max(n,1):.1f}")
    print(f"edge_attr range: inv_dist [{g['edge_attr'][:,0].min():.3f},{g['edge_attr'][:,0].max():.3f}], "
          f"seq [{g['edge_attr'][:,1].min():.3f},{g['edge_attr'][:,1].max():.3f}]")


if __name__ == "__main__":
    main()
