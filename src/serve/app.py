"""ProteinSolver inference API (CPU).

Run:
  .\\venv311\\Scripts\\python.exe -m uvicorn src.serve.app:app --port 8000
Endpoints use PDBs from data/raw/<id>.pdb with the local CPU checkpoint.
"""
import os
import torch
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from src.model.proteinsolver import ProteinSolver
from src.data.pdb_to_graph import build_graph
from src.generate.sample import gen_oneshot, gen_incremental
from src.generate.score import score_sequence, score_mutations
from src.validate import metrics as M

AA = "ACDEFGHIKLMNPQRSTVWY"
CFG_PATH = os.environ.get("PS_CONFIG", "configs/protein_local_cpu.yaml")
RAW_DIR = os.environ.get("PS_RAWDIR", "data/raw")

with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

DEVICE = torch.device("cpu")
MODEL = ProteinSolver(num_node_classes=21, edge_in_dim=2,
                      dim=CFG["model"]["dim"], num_blocks=CFG["model"]["blocks"]).to(DEVICE)
_CKPT = torch.load(CFG["train"]["ckpt_dir"] + "/last.pt", map_location=DEVICE)
MODEL.load_state_dict(_CKPT["model"])
MODEL.eval()
GRAPHS = {}


def get_graph(pdb):
    pdb = pdb.lower()
    if pdb not in GRAPHS:
        path = os.path.join(RAW_DIR, pdb + ".pdb")
        if not os.path.exists(path):
            raise HTTPException(404, f"unknown pdb '{pdb}' (need {path})")
        GRAPHS[pdb] = build_graph(path)
    return GRAPHS[pdb]


def seq_of(g):
    return "".join(AA[i] for i in g["x"].tolist())


app = FastAPI(title="ProteinSolver-mini")


@app.get("/health")
def health():
    return {"status": "ok", "ckpt_step": _CKPT["step"], "device": str(DEVICE)}


@app.get("/wt")
def wt(pdb: str = "4z8j"):
    g = get_graph(pdb)
    s = seq_of(g)
    return {"pdb": pdb, "len": len(s), "seq": s,
            "score": score_sequence(MODEL, g["x"], g["edge_index"], g["edge_attr"], DEVICE),
            "kd": M.mean_kd(s), "charge": M.net_charge(s),
            "core_r": M.core_packing_r(g["edge_index"], g["num_nodes"], s),
            "clash": M.charge_clashes(g["edge_index"], s)}


class ScoreReq(BaseModel):
    pdb: str = "4z8j"
    seq: str


@app.post("/score")
def score(req: ScoreReq):
    g = get_graph(req.pdb)
    s = req.seq.strip().upper()
    if len(s) != g["num_nodes"] or any(a not in AA for a in s):
        raise HTTPException(400, f"seq must be {g['num_nodes']} standard AAs")
    idx = torch.tensor([AA.index(a) for a in s])
    wt = seq_of(g)
    return {"pdb": req.pdb,
            "score": score_sequence(MODEL, idx, g["edge_index"], g["edge_attr"], DEVICE),
            "id_to_wt": sum(a == b for a, b in zip(s, wt)) / len(s) * 100.0,
            "kd": M.mean_kd(s), "charge": M.net_charge(s),
            "core_r": M.core_packing_r(g["edge_index"], g["num_nodes"], s),
            "clash": M.charge_clashes(g["edge_index"], s)}


class MutReq(BaseModel):
    pdb: str = "4z8j"
    muts: List[str]


@app.post("/mutate")
def mutate(req: MutReq):
    g = get_graph(req.pdb)
    try:
        wt_s, out = score_mutations(MODEL, g["x"], g["edge_index"], g["edge_attr"],
                                    DEVICE, req.muts)
    except (KeyError, ValueError, AssertionError) as e:
        raise HTTPException(400, f"bad mutation: {e}")
    return {"pdb": req.pdb, "wt_score": wt_s, "dscores": out}


class GenReq(BaseModel):
    pdb: str = "4z8j"
    mode: str = "sample"
    num: int = 1
    temperature: float = 1.0
    seed: int = 0


@app.post("/generate")
def generate(req: GenReq):
    if req.mode not in ("oneshot", "incremental", "sample") or not (1 <= req.num <= 5):
        raise HTTPException(400, "mode in {oneshot,incremental,sample}, 1<=num<=5")
    g = get_graph(req.pdb)
    n = g["num_nodes"]
    ei, ea = g["edge_index"].to(DEVICE), g["edge_attr"].to(DEVICE)
    wt = seq_of(g)
    rng = torch.Generator().manual_seed(req.seed)
    out = []
    with torch.no_grad():
        for _ in range(req.num):
            if req.mode == "oneshot":
                o = gen_oneshot(MODEL, ei, ea, DEVICE, n)
            else:
                o = gen_incremental(MODEL, ei, ea, DEVICE, n,
                                    sample=(req.mode == "sample"),
                                    temp=req.temperature, rng=rng)
            s = "".join(AA[i] for i in o.cpu().tolist())
            idx = o.cpu()
            out.append({"seq": s,
                        "id_to_wt": sum(a == b for a, b in zip(s, wt)) / len(s) * 100.0,
                        "score": score_sequence(MODEL, idx, g["edge_index"],
                                                g["edge_attr"], DEVICE)})
    return {"pdb": req.pdb, "mode": req.mode, "designs": out}
