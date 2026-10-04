"""Phase 3 sanity: dataset splits + one masked sample."""
from src.data.protein_dataset import ProteinGraphDataset

ds = ProteinGraphDataset("data/processed/graphs", train=True)
dh = ProteinGraphDataset("data/processed/graphs", train=False, holdout=0.1)
print("train", len(ds), "holdout", len(dh))
it = ds[0]
print("x", tuple(it["x"].shape), "masked", int((it["x"] == 20).sum()),
      "E", it["edge_index"].shape[1])
overlap = len(set(ds.files) & set(dh.files))
print("split overlap:", overlap)
assert overlap == 0
print("OK dataset")
