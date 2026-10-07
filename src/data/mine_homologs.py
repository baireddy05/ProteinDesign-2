"""Mine homolog training pairs: SIFTS PDB->UniProt, UniRef50 cluster members,
local-align to template chain, keep matched columns.

Output JSONL: {"key": "<pdb>_<chain>", "seq": "<template-length, '-'=unmapped>"}.
Resume-safe: skips keys already present in --out.

Usage (pilot, 10 templates x 30 homologs):
  .\\venv311\\Scripts\\python.exe -m src.data.mine_homologs --max_templates 10 --max_homologs 30
"""
import argparse, gzip, json, os, time
import requests

REST = "https://rest.uniprot.org"
SLEEP = 0.25


def sifts_map(path):
    mp = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("#") or line.startswith("PDB"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            key = (p[0].upper(), p[1])
            if key not in mp and p[2] != "?":
                mp[key] = p[2].split(".")[0]
    return mp


def uniref50_members(ac):
    time.sleep(SLEEP)
    r = requests.get(f"{REST}/uniref/search",
                     params={"query": f"uniprot_id:{ac}", "format": "json", "size": 10},
                     timeout=60)
    r.raise_for_status()
    ids = [x["id"] for x in r.json().get("results", []) if x["id"].startswith("UniRef50_")]
    if not ids:
        return []
    time.sleep(SLEEP)
    e = requests.get(f"{REST}/uniref/{ids[0]}", params={"format": "json"},
                     timeout=120).json()
    return [(m["accessions"][0], m.get("sequenceLength", 0)) for m in e.get("members", [])
            if m.get("accessions")]


def fetch_fasta(acs):
    out = {}
    for i in range(0, len(acs), 100):
        chunk = acs[i:i + 100]
        q = " OR ".join(f"accession:{a}" for a in chunk)
        time.sleep(SLEEP)
        # NOTE: GET, not POST — UniProt redirects POST and drops the form body
        r = requests.get(f"{REST}/uniprotkb/search",
                         params={"query": q, "format": "fasta", "size": 500},
                         timeout=120)
        r.raise_for_status()
        name, seq, out_entry = None, [], None
        for line in r.text.splitlines():
            if line.startswith(">"):
                if name:
                    out[name] = "".join(seq)
                # header: >db|ACCESSION|EntryName ... -> key on accession AND name
                parts = line[1:].split("|")
                acc = parts[1] if len(parts) > 2 else parts[0].split()[0]
                entry = parts[-1].split()[0] if len(parts) > 1 else acc
                name = acc
                out_entry = entry
                seq = []
            else:
                seq.append(line.strip())
        if name:
            out[name] = "".join(seq)
            if out_entry and out_entry != name:
                out[out_entry] = "".join(seq)
    return out


def map_to_template(tseq, hseq):
    """Local-align homolog to template. Returns template-length list (aa or None)."""
    from biotite.sequence import ProteinSequence
    from biotite.sequence.align import align_optimal, SubstitutionMatrix
    mat = SubstitutionMatrix.std_protein_matrix()
    # free end gaps + global: every template position lands in the trace
    # (local mode trims flanks and breaks the template-length invariant)
    ali = align_optimal(ProteinSequence(tseq), ProteinSequence(hseq), mat,
                        gap_penalty=(-10, -1), terminal_penalty=False, local=False)[0]
    gt, gh = ali.get_gapped_sequences()
    res, ti = [], 0
    for a, b in zip(str(gt), str(gh)):
        if a != "-":
            res.append(None if b == "-" else b)
            ti += 1
    assert len(res) == len(tseq), (len(res), len(tseq))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sifts", default="data/raw/pdb_chain_uniprot.tsv.gz")
    ap.add_argument("--cache", default="data/processed/graphs")
    ap.add_argument("--out", default="data/processed/homologs.jsonl")
    ap.add_argument("--max_templates", type=int, default=10)
    ap.add_argument("--max_homologs", type=int, default=30)
    ap.add_argument("--min_cov", type=float, default=0.6)
    args = ap.parse_args()

    import torch, glob
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                done.add(json.loads(line)["key"])
    print(f"resume: {len(done)} template-homolog pairs present")

    smap = sifts_map(args.sifts)
    files = sorted(glob.glob(os.path.join(args.cache, "*.pt")))[:]
    n_new = 0
    with open(args.out, "a") as out:
        for fp in files[:args.max_templates]:
            import re
            m = re.match(r"([0-9a-z]{4})_(.+)\.pt$", os.path.basename(fp))
            pdb, ch = m.group(1).upper(), m.group(2)
            tseq = None
            g = torch.load(fp, weights_only=False)
            from src.data.pdb_to_graph import AA
            tseq = "".join(AA[i] for i in g["x"].tolist())
            ac = smap.get((pdb, ch))
            if ac is None:
                print(f"{pdb}_{ch}: no UniProt AC, skip")
                continue
            try:
                mems = uniref50_members(ac)
            except Exception as e:
                print(f"{pdb}_{ch}: uniref fail {e}")
                continue
            lo, hi = int(len(tseq) * 0.5), int(len(tseq) * 2.0) + 50
            comp = [(a, ln) for a, ln in mems if lo <= ln <= hi]
            # members come similarity-ordered -> stride for identity spread
            st = max(1, len(comp) // args.max_homologs)
            mems = comp[::st][:args.max_homologs]
            if not mems:
                print(f"{pdb}_{ch}: no length-compatible members")
                continue
            try:
                fasta = fetch_fasta([a for a, _ in mems])
            except Exception as e:
                print(f"{pdb}_{ch}: fasta fail {e}")
                continue
            kept = 0
            n_lowcov = n_alignfail = 0
            for a, _ in mems:
                key = f"{pdb}_{ch}:{a}"
                if key in done or a not in fasta:
                    continue
                try:
                    mapped = map_to_template(tseq, fasta[a])
                except Exception as e:
                    n_alignfail += 1
                    continue
                cov = sum(1 for x in mapped if x) / len(tseq)
                if cov < args.min_cov:
                    n_lowcov += 1
                    continue
                out.write(json.dumps({"key": key,
                                      "seq": "".join(x or "-" for x in mapped)}) + "\n")
                done.add(key)
                kept += 1
                n_new += 1
            print(f"{pdb}_{ch} ({ac}): {len(mems)} members, kept {kept} "
                  f"(lowcov {n_lowcov}, alignfail {n_alignfail})", flush=True)
    print(f"done: +{n_new} pairs")


if __name__ == "__main__":
    main()
