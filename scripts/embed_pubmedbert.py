# scripts/embed_pubmedbert.py
"""Embed each panel gene's NCBI text (symbol + name + summary) with PubMedBERT.

Reproducible: fixed model (NeuML/pubmedbert-base-embeddings), deterministic
(eval mode). Writes an Entrez-indexed embedding parquet the geneal pipeline
consumes. Gene text is fetched from MyGene.info (cached to a pickle so reruns
are offline-stable)."""
from __future__ import annotations
import argparse, pickle
from pathlib import Path
import pandas as pd


def fetch_gene_text(entrez_ids, cache):
    cache = Path(cache)
    if cache.exists():
        return pickle.load(open(cache, "rb"))
    import mygene
    mg = mygene.MyGeneInfo()
    out = mg.querymany(entrez_ids, scopes="entrezgene",
                       fields="symbol,name,summary", species="human", verbose=False)
    txt = {}
    for r in out:
        e = r.get("query")
        parts = [r.get("symbol", ""), r.get("name", ""), r.get("summary", "")]
        t = ". ".join(p for p in parts if p)
        if e and t.strip():
            txt[int(e)] = t
    cache.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(txt, open(cache, "wb"))
    return txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/processed/depmap/panel_hvg.txt")
    ap.add_argument("--out", default="data/processed/embeddings/pubmedbert_hvg.parquet")
    ap.add_argument("--model", default="NeuML/pubmedbert-base-embeddings")
    ap.add_argument("--text-cache", default="data/processed/depmap/gene_text.pkl")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    entrez = [int(x) for x in Path(args.panel).read_text().split()]
    txt = fetch_gene_text(entrez, args.text_cache)
    ents = [e for e in entrez if e in txt]
    docs = [txt[e] for e in ents]
    print(f"genes with text: {len(ents)}/{len(entrez)}")
    model = SentenceTransformer(args.model)
    emb = model.encode(docs, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
    df = pd.DataFrame(emb, index=pd.Index(ents, name="entrez"))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out)
    print(f"embedded {len(df)} genes, dim={df.shape[1]} -> {args.out}")


if __name__ == "__main__":
    main()
