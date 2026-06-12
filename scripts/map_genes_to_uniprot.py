# scripts/map_genes_to_uniprot.py
"""Entrez GeneID -> UniProtKB canonical accession + sequence, via UniProt REST.

Usage: python scripts/map_genes_to_uniprot.py --entrez-file <ids.txt> --out data/processed/depmap/uniprot_map.parquet
or:    python scripts/map_genes_to_uniprot.py --from-gene-effect data/processed/depmap/gene_effect.parquet --limit 500
"""
from __future__ import annotations
import argparse, time, io, re
from pathlib import Path
import requests
import pandas as pd
from geneal.data.depmap import parse_entrez, load_gene_effect

UNIPROT = "https://rest.uniprot.org"


def _get_with_retry(url: str, tries: int = 5, timeout: int = 60):
    """GET with a few retries — the UniProt endpoint occasionally drops SSL."""
    last = None
    for a in range(tries):
        try:
            return requests.get(url, timeout=timeout)
        except requests.exceptions.RequestException as e:
            last = e
            time.sleep(3)
    raise last


_LINK_RE = re.compile(r'<([^>]+)>\s*;\s*rel="next"')


def _next_link(link_header: str | None) -> str | None:
    """Extract the rel=\"next\" URL from a UniProt Link header, if present.

    The URL itself contains commas (fields=accession,sequence,reviewed), so we
    must match the <...>;rel="next" structure rather than splitting on commas.
    """
    if not link_header:
        return None
    m = _LINK_RE.search(link_header)
    return m.group(1) if m else None


def submit_idmapping(entrez_ids: list[int]) -> str:
    r = requests.post(f"{UNIPROT}/idmapping/run",
                      data={"from": "GeneID", "to": "UniProtKB",
                            "ids": ",".join(str(i) for i in entrez_ids)})
    r.raise_for_status()
    return r.json()["jobId"]


def poll(job_id: str, timeout: int = 600) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = requests.get(f"{UNIPROT}/idmapping/status/{job_id}")
        r.raise_for_status()
        d = r.json()
        if d.get("jobStatus") in (None, "FINISHED") or "results" in d:
            return
        time.sleep(3)
    raise TimeoutError(f"idmapping job {job_id} did not finish")


def fetch_results(job_id: str) -> pd.DataFrame:
    # stream reviewed canonical entries as TSV with sequence
    url = (f"{UNIPROT}/idmapping/uniprotkb/results/stream/{job_id}"
           "?format=tsv&fields=accession,sequence,reviewed&compressed=false")
    r = requests.get(url)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), sep="\t")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-gene-effect")
    ap.add_argument("--entrez-file")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="data/processed/depmap/uniprot_map.parquet")
    args = ap.parse_args()

    if args.from_gene_effect:
        ge = load_gene_effect(args.from_gene_effect)
        entrez = [parse_entrez(g) for g in ge.index]
    else:
        entrez = [int(x) for x in Path(args.entrez_file).read_text().split()]
    if args.limit:
        entrez = entrez[: args.limit]

    # UniProt id-mapping caps batch size; chunk to be safe
    frames = []
    CHUNK = 5000
    for i in range(0, len(entrez), CHUNK):
        chunk = entrez[i:i + CHUNK]
        # the API returns from->to pairs; we re-run mapping that also returns the
        # source id by using the per-id 'from' column available in JSON results.
        job = submit_idmapping(chunk); poll(job)
        # JSON results carry both 'from' (GeneID) and the UniProt entry.
        # A single GeneID can map to multiple UniProtKB entries, so the total
        # record count can exceed the number of input ids -> follow the
        # `next` Link header to page through ALL results.
        url = (f"{UNIPROT}/idmapping/uniprotkb/results/{job}"
               "?format=json&fields=accession,sequence,reviewed&size=500")
        rows = []
        while url:
            jr = _get_with_retry(url)
            jr.raise_for_status()
            data = jr.json()
            for rec in data.get("results", []):
                ent = int(rec["from"])
                to = rec["to"]
                acc = to["primaryAccession"]
                seq = to["sequence"]["value"]
                reviewed = to.get("entryType", "").lower().startswith("uniprotkb reviewed")
                rows.append({"entrez": ent, "accession": acc, "sequence": seq,
                             "reviewed": reviewed})
            url = _next_link(jr.headers.get("Link"))
            if url:
                time.sleep(0.5)
        frames.append(pd.DataFrame(rows))
        time.sleep(1)
    out = pd.concat(frames, ignore_index=True)
    # prefer reviewed (Swiss-Prot) canonical; keep first per entrez
    out = (out.sort_values("reviewed", ascending=False)
              .drop_duplicates("entrez", keep="first").reset_index(drop=True))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"mapped {out['entrez'].nunique()} / {len(set(entrez))} entrez ids -> {args.out}")


if __name__ == "__main__":
    main()
