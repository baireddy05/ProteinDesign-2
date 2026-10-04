"""Fetch training PDB IDs: RCSB search (X-ray, high-res, single protein entity),
fallback to a curated list of classic small globular domains if offline.

Usage:
  .\\venv311\\Scripts\\python.exe -m src.data.fetch_ids --out data/raw/ids.txt --rows 500
"""
import argparse

# Classic small (<150 res) well-behaved domains; doubles as offline fallback.
FALLBACK = """1UBQ 1CRN 1VII 2GB1 1PGB 1SHG 1TEN 1FNA 1MJC 2ACY 1APS 1RIS 1CSP
1SRL 1CKA 1FKB 1SNC 1LZ1 1HEL 1R69 1LMB 2CRO 1IFC 1TTF 1WIT 1QYS 1KUL 1E0L
1I6C 1C9O 1DWR 1PIN 1O6S 2F21 1PSF 1E0G 1BX7 1SUP 1OPA 2SN3 1KIT 1STN 1REX
1CEM 1E0N 1APS 2ACY 1MJC 1SHG 1CSP 1SRL 1TEN 1FNA""".split()


def rcsb_ids(rows=500):
    import requests
    q = {
        "query": {
            "type": "group", "logical_operator": "and",
            "nodes": [
                {"type": "terminal", "service": "text",
                 "parameters": {"attribute": "exptl.method", "operator": "exact_match",
                               "value": "X-RAY DIFFRACTION"}},
                {"type": "terminal", "service": "text",
                 "parameters": {"attribute": "rcsb_entry_info.resolution_combined",
                               "operator": "less", "value": 2.0}},
                {"type": "terminal", "service": "text",
                 "parameters": {"attribute": "rcsb_entry_info.polymer_entity_count_protein",
                               "operator": "equals", "value": 1}},
            ],
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": rows},
            "sort": [{"sort_by": "rcsb_entry_info.resolution_combined", "direction": "asc"}],
        },
    }
    r = requests.post("https://search.rcsb.org/rcsbsearch/v2/query", json=q, timeout=60)
    r.raise_for_status()
    return [h["identifier"] for h in r.json()["result_set"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/ids.txt")
    ap.add_argument("--rows", type=int, default=500)
    args = ap.parse_args()
    try:
        ids = rcsb_ids(args.rows)
        print(f"RCSB returned {len(ids)} ids")
    except Exception as e:
        print(f"RCSB query failed ({e}); using fallback list")
        ids = []
    ids = list(dict.fromkeys([i.upper() for i in ids] + FALLBACK))
    with open(args.out, "w") as f:
        f.write("\n".join(ids) + "\n")
    print(f"wrote {len(ids)} ids -> {args.out}")


if __name__ == "__main__":
    main()
