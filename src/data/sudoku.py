"""Sudoku CSP as graph: 81 nodes, edges where cells share row/col/box.

On-the-fly generation: base pattern + shuffle, mask ~50%.
Fast enough for CPU; uniqueness not guaranteed (fine for prototype).
"""
import random
import torch
from torch.utils.data import Dataset

_EDGE_INDEX_CACHE = None


def base_grid():
    return [[(3 * (r % 3) + r // 3 + c) % 9 + 1 for c in range(9)] for r in range(9)]


def shuffle_grid(g):
    # shuffle bands, rows within bands, stacks, cols within stacks, digits, transpose
    import copy
    g = copy.deepcopy(g)
    # digit remap
    perm = list(range(1, 10))
    random.shuffle(perm)
    mp = {i + 1: perm[i] for i in range(9)}
    g = [[mp[v] for v in row] for row in g]
    # row bands
    bands = [g[i * 3:(i + 1) * 3] for i in range(3)]
    random.shuffle(bands)
    g = []
    for b in bands:
        random.shuffle(b)
        g.extend(b)
    # transpose randomly
    if random.random() < 0.5:
        g = [list(r) for r in zip(*g)]
    # col stacks (after possible transpose, same op on columns)
    cols = [list(c) for c in zip(*g)]
    stacks = [cols[i * 3:(i + 1) * 3] for i in range(3)]
    random.shuffle(stacks)
    cols = []
    for s in stacks:
        random.shuffle(s)
        cols.extend(s)
    g = [list(r) for r in zip(*cols)]
    return g


def same_constraint(a, b):
    ra, ca = divmod(a, 9)
    rb, cb = divmod(b, 9)
    if ra == rb or ca == cb:
        return True
    if (ra // 3, ca // 3) == (rb // 3, cb // 3):
        return True
    return False


def sudoku_edge_index(add_self_loops=True):
    """Directed edge index. 81*20=1620 peer edges + 81 self = 1701 (paper count)."""
    global _EDGE_INDEX_CACHE
    if _EDGE_INDEX_CACHE is not None:
        return _EDGE_INDEX_CACHE
    src, dst = [], []
    for i in range(81):
        for j in range(81):
            if i != j and same_constraint(i, j):
                src.append(i)
                dst.append(j)
    if add_self_loops:
        for i in range(81):
            src.append(i)
            dst.append(i)
    ei = torch.tensor([src, dst], dtype=torch.long)
    _EDGE_INDEX_CACHE = ei
    return ei


def num_directed_edges():
    return sudoku_edge_index().size(1)  # expect 1701


class SudokuDataset(Dataset):
    """Generates (input, target) pairs. input has n_mask cells set to 0."""

    def __init__(self, size=10000, n_mask=40, seed=0):
        self.size = size
        self.n_mask = n_mask
        self.rng = random.Random(seed)
        self.edge_index = sudoku_edge_index()
        self.edge_attr = torch.ones(self.edge_index.size(1), 1, dtype=torch.float32)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        g = shuffle_grid(base_grid())
        flat = [v for row in g for v in row]  # 81 ints 1-9
        y = torch.tensor(flat, dtype=torch.long)
        mask_idx = self.rng.sample(range(81), self.n_mask)
        x = y.clone()
        x[mask_idx] = 0  # 0 = empty token
        return x, y

    @staticmethod
    def collate(batch):
        xs = torch.stack([b[0] for b in batch])  # [B, 81]
        ys = torch.stack([b[1] for b in batch])
        return xs, ys
