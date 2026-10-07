"""Homolog-pair training dataset: template graph + aligned homolog sequence.

Split is at TEMPLATE level (all pairs of a template stay together) to avoid
homolog leakage across train/test. Loss/masking apply only to valid (aligned)
positions; '-' columns are never predicted.

Pair file JSONL: {"key": "<pdb>_<chain>:<homologAC>", "seq": "<template-len, '-'=gap>"}.
"""
import glob
import json
import os
import random
import torch
from torch.utils.data import Dataset
from src.data.pdb_to_graph import AA2I, MASK

AA = "ACDEFGHIKLMNPQRSTVWY"


class HomologPairDataset(Dataset):
    def __init__(self, pairs_path, cache_dir, mask_frac=0.5, seed=0,
                 holdout=0.1, train=True):
        self.cache = {}
        cfiles = {os.path.basename(p)[:-3].upper(): p
                  for p in glob.glob(os.path.join(cache_dir, "*.pt"))}
        by_t = {}
        with open(pairs_path) as f:
            for line in f:
                d = json.loads(line)
                tkey = d["key"].split(":")[0].upper()
                if tkey in cfiles:
                    by_t.setdefault(tkey, []).append(d["seq"])
        tmpls = sorted(by_t)
        if not tmpls:
            raise RuntimeError("no pairs with cached templates")
        rng = random.Random(seed)
        rng.shuffle(tmpls)
        n_hold = max(1, int(len(tmpls) * holdout)) if holdout else 0
        keep = tmpls[n_hold:] if train else tmpls[:n_hold]
        self.items = [(t, s) for t in keep for s in by_t[t]]
        self.files = cfiles
        self.mask_frac = mask_frac
        self.rng = random.Random(seed + (1 if train else 2))

    def __len__(self):
        return len(self.items)

    def _graph(self, tkey):
        if tkey not in self.cache:
            self.cache[tkey] = torch.load(self.files[tkey], weights_only=False)
        return self.cache[tkey]

    def __getitem__(self, idx):
        tkey, hseq = self.items[idx % len(self.items)]
        g = self._graph(tkey)
        n = g["x"].size(0)
        valid = [i for i, a in enumerate(hseq) if a in AA2I]
        k = max(1, int(len(valid) * self.mask_frac))
        midx = set(self.rng.sample(valid, k))
        x = torch.empty(n, dtype=torch.long)
        y = torch.empty(n, dtype=torch.long)
        for i, a in enumerate(hseq):
            if a not in AA2I:
                x[i] = MASK  # gap/unknown: never predicted
                y[i] = 0
            else:
                y[i] = AA2I[a]
                x[i] = MASK if i in midx else AA2I[a]
        return {"x": x, "y": y, "edge_index": g["edge_index"],
                "edge_attr": g["edge_attr"].float(),
                "loss_mask": torch.tensor([i in midx for i in range(n)])}
