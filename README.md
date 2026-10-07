# ProteinSolver Rebuild — Inverse Protein Folding as CSP with GNNs

Re-implementation of Strokach et al., Cell Systems 2020 ("Fast and Flexible Protein
Design Using Deep Graph Neural Networks"): protein topologies as constraint graphs,
a residual Edge-Convolution/Aggregation GNN trained by masked reconstruction,
used to score mutations and generate novel sequences for fixed backbones.

## Results

### Sudoku prototype (CSP validation)
| Model | One-shot | Incremental | Fully solved |
|---|---|---|---|
| Local CPU: 4x64-dim, 2k steps | 74.0% | 94.3% | 66% |
| Colab T4: 4x128-dim, 59k steps | **97.3%** | **97.3%** | **82%** |

Paper reference: 72% single-pass, ~90% iterative. Held-out seeds; inputs/targets
paired per draw (see `src/eval/eval_sudoku.py` note).

### Protein training (masked reconstruction, 50% mask)
| Model | Train graphs | Steps | Holdout recon |
|---|---|---|---|
| Local CPU: 3x64-dim | 245 (40–200 res) | 10k | 15.6% |
| Colab T4: 4x128-dim, fp16 | 2675 (40–200 res) | 60k | 45.4%* |

\* Optimistic: local holdout likely overlaps the Colab training set (same RCSB
query). Treat as train-fit signal, not a generalization number. Paper: ~22–32%.

### Generation + validation (4Z8J, 96 res, big model)
| | id% | score | core_r | clash |
|---|---|---|---|---|
| WT | 100 | −2.831 | 0.299 | 2.11 |
| D1 | 9.4 | −2.378 | 0.334 | 3.25 |
| D2 | 6.2 | −2.413 | 0.350 | 1.62 |
| D3 | 10.4 | −2.246 | **0.502** | 1.87 |

### Cross-fold check (1OC7, 364 res, big model)
| | id% | score | core_r | clash |
|---|---|---|---|---|
| WT | 100 | −2.873 | 0.237 | 1.20 |
| D1 | 8.2 | −2.567 | 0.386 | 1.62 |
| D2 | 8.0 | −2.489 | 0.423 | 1.94 |
| D3 | 7.7 | −2.517 | 0.464 | 2.31 |

Pattern holds across folds: designs outscore WT (within-model), beat WT on core
packing correlation, keep native-like composition and mass. Known weaknesses:
designs skew hydrophilic on 4Z8J and over-negative on 1OC7 (charge balance);
point-mutant Δscores from the small model were near-zero. Scores are
self-reported — packing/clash metrics (computed without the network) are the
independent corroboration. No wet-lab validation; no Rosetta/MD (Free-tier scope).

## Layout
```
src/model/eca.py, proteinsolver.py  # ECA block + network
src/data/sudoku.py                  # Sudoku CSP graphs + generator
src/data/pdb_to_graph.py            # PDB -> CA graph (<12A, [inv_dist, seq_sep])
src/data/fetch_ids.py, build_cache.py, protein_dataset.py
src/train/train_sudoku.py, train_protein.py   # resumable, AMP for CUDA
src/generate/score.py, sample.py    # scoring, one-shot/incremental/sample
src/validate/metrics.py, run_validation.py
src/eval/eval_sudoku.py, eval_protein.py, verify_*.py
src/serve/app.py                    # FastAPI: /health /wt /score /mutate /generate
configs/local_cpu.yaml, colab_free_t4.yaml
configs/protein_local_cpu.yaml, protein_colab_t4.yaml
notebooks/colab_free.ipynb          # T4 launcher (sudoku + protein scale-up)
```

## Quickstart (local, CPU)
Use Python 3.11 — torch 2.14 wheels are broken on 3.14 (`c10.dll` init failure).
```
py -3.11 -m venv venv311
.\venv311\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\venv311\Scripts\python.exe -m pip install pyyaml tqdm numpy biotite requests fastapi uvicorn httpx
.\venv311\Scripts\python.exe -m src.train.train_sudoku --config configs/local_cpu.yaml --steps 2000
.\venv311\Scripts\python.exe -m src.eval.eval_sudoku --config configs/local_cpu.yaml
.\venv311\Scripts\python.exe -m uvicorn src.serve.app:app --port 8000
```

## Colab Free (T4) flow
Notebook handles clone → install → train with Drive checkpoints; every stage
resumes after preemption. Protein scale-up: `fetch_ids --rows 6000` +
`build_cache` (~2675 graphs) → `train_protein` 60k steps (~15 min on T4) →
download `last.pt` → eval/generate locally with `--ckpt` (+ `--cache_dir`).

## Environment notes (hard-won)
- torch on Python 3.14 Windows fails DLL load; use 3.11.
- AMP fp16 needs the `dtype=he.dtype` accumulator in `ECABlock` (embeddings stay fp32).
- `ds[i][0], ds[i][1]` draws twice from the generator — always unpack once.
- `data/raw/`, `checkpoints/`, venvs are gitignored; `fetch_ids` creates its dirs.
