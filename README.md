# ProteinSolver Rebuild — CPU + Colab Free
Inverse protein folding as CSP with GNN. See project plan in session.

## Hardware
- Local: i5-13500H, 16GB, Iris Xe (CPU-only inference/dev)
- Train: Colab Free T4 (resumable, fp16)

## Structure
```
src/model/eca.py            # Residual Edge Convolution + Aggregation block
src/model/proteinsolver.py  # Full network (node/edge embed + N x ECA + head)
src/data/sudoku.py          # Sudoku CSP graph builder + on-the-fly dataset
src/data/pdb_to_graph.py    # PDB -> protein graph (Phase 2)
src/train/train_sudoku.py   # Resumable training loop (CPU + T4)
configs/local_cpu.yaml
configs/colab_free_t4.yaml
notebooks/colab_free.ipynb  # Thin launcher for Colab Free
```

## Quickstart (local CPU smoke test)
Use Python 3.11 (`py -3.11`) — torch 2.14 is broken on 3.14 (c10.dll init failure).
System python (3.14) and `venv_protein` cannot import torch; use `venv311`.
```
py -3.11 -m venv venv311
.\venv311\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\venv311\Scripts\python.exe -m pip install pyyaml tqdm numpy
.\venv311\Scripts\python.exe -m src.train.train_sudoku --config configs/local_cpu.yaml --steps 200
```

## Colab Free
1. Upload repo to GitHub or Drive.
2. Open `notebooks/colab_free.ipynb`, mount Drive.
3. Run — checkpoints go to `/content/drive/MyDrive/proteinsolver/ckpt/`.
4. Re-run same notebook to resume after preemption.
