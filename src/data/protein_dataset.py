"""Protein graph dataset: cached graphs + fresh 50% masks each access."""
import glob, os, random
import torch
from torch.utils.data import Dataset
from src.data.pdb_to_graph import MASK


class ProteinGraphDataset(Dataset):
    def __init__(self, cache_dir, mask_frac=0.5, seed=0, holdout=0.1, train=True):
        files = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))
        if not files:
            raise RuntimeError(f"no cached graphs in {cache_dir}")
        rng = random.Random(seed)
        rng.shuffle(files)
        n_hold = max(1, int(len(files) * holdout)) if holdout else 0
        # shared split: same seed -> same order -> disjoint by construction
        self.files = files[n_hold:] if train else files[:n_hold]
        self.mask_frac = mask_frac
        self.rng = random.Random(seed + (1 if train else 2))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        g = torch.load(self.files[idx % len(self.files)], weights_only=False)
        y = g["x"]
        n = y.size(0)
        k = max(1, int(n * self.mask_frac))
        midx = self.rng.sample(range(n), k)
        x = y.clone()
        x[midx] = MASK
        return {"x": x, "y": y, "edge_index": g["edge_index"],
                "edge_attr": g["edge_attr"].float(), "mask": torch.tensor(midx)}
