"""Serve smoke test: /health + /wt (full scoring path on 4Z8J)."""
from fastapi.testclient import TestClient
from src.serve.app import app

c = TestClient(app)
h = c.get("/health").json()
print("health:", h)
assert h["status"] == "ok" and h["ckpt_step"] == 10000
w = c.get("/wt", params={"pdb": "4z8j"}).json()
print("wt len:", w["len"], "score:", round(w["score"], 3),
      "core_r:", round(w["core_r"], 3), "clash:", round(w["clash"], 2))
assert w["len"] == 96 and abs(w["score"] - (-2.682)) < 0.01
r = c.post("/score", json={"pdb": "4z8j", "seq": "X" * 95})
print("bad seq status:", r.status_code)
assert r.status_code == 400
print("OK serve")
