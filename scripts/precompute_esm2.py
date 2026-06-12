# scripts/precompute_esm2.py
"""Embed mapped protein sequences with ESM2, cache as genes(entrez) x dim parquet.

Usage: python scripts/precompute_esm2.py --map data/processed/depmap/uniprot_map_small.parquet \
         --out data/processed/embeddings/esm2_small.parquet --model esm2_t33_650M_UR50D
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from geneal.data.esm2_embed import embed_sequences


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="esm2_t33_650M_UR50D")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    m = pd.read_parquet(args.map)
    seqs = {str(int(r.entrez)): r.sequence for r in m.itertuples() if isinstance(r.sequence, str)}
    emb = embed_sequences(seqs, model_name=args.model, batch_size=args.batch_size)
    emb.index = emb.index.astype(int)
    emb.index.name = "entrez"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    emb.to_parquet(args.out)
    print(f"embedded {len(emb)} proteins, dim={emb.shape[1]} -> {args.out}")


if __name__ == "__main__":
    main()
