"""Build genome-wide (all ~18.5k DepMap genes) STRING and CORUM gene-gene
similarity caches, stored SPARSE as edge lists.

A dense 18531x18531 matrix is ~2.7 GB; instead we store nonzero pairs only as
parquet edge lists [entrez_i, entrez_j, weight]. Downstream code builds a
submatrix for any panel by filtering edges to the panel's entrez ids.

Outputs (all under data/processed/depmap/):
  entrez_symbol.parquet        entrez -> symbol for the full 18531-gene set
  corum_edges_all.parquet      [entrez_i, entrez_j, weight=1, source='corum']
  corum_membership_all.parquet [entrez, complex_id] long-form (pathway hedging)
  string_edges_all.parquet     [entrez_i, entrez_j, weight=score/1000]

Run: micromamba run -n geneal python scripts/build_graph_caches.py
"""
import gzip
import re
from itertools import combinations

import pandas as pd

BASE = "/cv/data/braid/andera29/projs/gene-al"
PROC = f"{BASE}/data/processed/depmap"
STRING_DIR = f"{BASE}/data/string_dl"
STRING_THRESHOLD = 0.4  # keep combined_score/1000 >= this to bound size

LABEL_PAT = re.compile(r"^(.*) \((\d+)\)$")


def load_panel():
    with open(f"{PROC}/panel_all.txt") as f:
        panel = [int(x) for x in f.read().split()]
    panel = list(dict.fromkeys(panel))  # dedupe, preserve order
    return panel


def build_entrez_symbol(panel):
    """Symbol for every panel entrez, parsed from gene_effect row labels
    ('SYM (entrez)'), which are the authoritative DepMap symbols (offline)."""
    ge = pd.read_parquet(f"{PROC}/gene_effect.parquet", columns=[])
    ent2sym = {}
    for lab in ge.index:
        m = LABEL_PAT.match(lab)
        if m:
            ent2sym[int(m.group(2))] = m.group(1)
    panel_set = set(panel)
    rows = [(e, ent2sym[e]) for e in panel if e in ent2sym]
    df = pd.DataFrame(rows, columns=["entrez", "symbol"])
    df.to_parquet(f"{PROC}/entrez_symbol.parquet", index=False)
    print(f"[entrez_symbol] {len(df)}/{len(panel)} panel genes mapped to a symbol")
    return df, {s: e for e, s in rows}, panel_set


def build_corum(sym2ent, panel_set):
    corum = pd.read_csv(f"{BASE}/data/corum_dl/humanComplexes.txt", sep="\t", dtype=str)
    edges = set()  # frozenset-like ordered tuple (i<j)
    membership = set()  # (entrez, complex_id)
    n_complexes = 0
    for _, row in corum.iterrows():
        raw = row.get("subunits_gene_name")
        if not isinstance(raw, str) or not raw.strip():
            continue
        cid = row["complex_id"]
        ents = set()
        for s in (x.strip() for x in raw.split(";")):
            e = sym2ent.get(s)
            if e is not None:  # already restricted to panel set
                ents.add(e)
        if not ents:
            continue
        n_complexes += 1
        for e in ents:
            membership.add((e, cid))
        for a, b in combinations(sorted(ents), 2):
            edges.add((a, b))

    edf = pd.DataFrame(sorted(edges), columns=["entrez_i", "entrez_j"])
    edf["weight"] = 1.0
    edf["source"] = "corum"
    edf.to_parquet(f"{PROC}/corum_edges_all.parquet", index=False)

    mdf = pd.DataFrame(sorted(membership), columns=["entrez", "complex_id"])
    mdf.to_parquet(f"{PROC}/corum_membership_all.parquet", index=False)

    covered = set(edf["entrez_i"]) | set(edf["entrez_j"])
    print(
        f"[corum] {len(edf)} edges; {len(covered)} genes covered; "
        f"{n_complexes} complexes with >=1 panel member; "
        f"membership rows={len(mdf)}"
    )
    return edf, mdf


def build_string(sym2ent, panel_set):
    info_path = f"{STRING_DIR}/9606.protein.info.v12.0.txt.gz"
    links_path = f"{STRING_DIR}/9606.protein.links.v12.0.txt.gz"

    # ENSP -> preferred symbol -> panel entrez (only keep panel-mapped proteins)
    ensp2ent = {}
    with gzip.open(info_path, "rt") as f:
        header = f.readline()
        cols = header.rstrip("\n").split("\t")
        try:
            i_id = cols.index("#string_protein_id")
        except ValueError:
            i_id = 0
        try:
            i_name = cols.index("preferred_name")
        except ValueError:
            i_name = 1
        for line in f:
            p = line.rstrip("\n").split("\t")
            sym = p[i_name]
            e = sym2ent.get(sym)
            if e is not None:
                ensp2ent[p[i_id]] = e
    print(f"[string] {len(ensp2ent)} ENSP ids map to a panel entrez")

    thr_score = int(round(STRING_THRESHOLD * 1000))
    edges = {}  # (i<j) -> max weight
    n_lines = 0
    with gzip.open(links_path, "rt") as f:
        f.readline()  # header: protein1 protein2 combined_score (space-sep)
        for line in f:
            n_lines += 1
            p1, p2, score = line.split()
            score = int(score)
            if score < thr_score:
                continue
            ea = ensp2ent.get(p1)
            if ea is None:
                continue
            eb = ensp2ent.get(p2)
            if eb is None or ea == eb:
                continue
            key = (ea, eb) if ea < eb else (eb, ea)
            w = score / 1000.0
            if w > edges.get(key, 0.0):
                edges[key] = w

    rows = [(a, b, w) for (a, b), w in edges.items()]
    edf = pd.DataFrame(rows, columns=["entrez_i", "entrez_j", "weight"])
    edf = edf.sort_values(["entrez_i", "entrez_j"]).reset_index(drop=True)
    edf.to_parquet(f"{PROC}/string_edges_all.parquet", index=False)
    covered = set(edf["entrez_i"]) | set(edf["entrez_j"])
    print(
        f"[string] scanned {n_lines} raw edges; {len(edf)} undirected edges "
        f"(score>={STRING_THRESHOLD}); {len(covered)} genes covered"
    )
    return edf


def main():
    panel = load_panel()
    print(f"panel size {len(panel)}")
    _, sym2ent, panel_set = build_entrez_symbol(panel)
    build_corum(sym2ent, panel_set)
    build_string(sym2ent, panel_set)
    print("DONE")


if __name__ == "__main__":
    main()
