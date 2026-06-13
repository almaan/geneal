# scripts/diagnose_embeddings.py
"""Quantify whether an embedding carries (a) predictive signal for lethality and
(b) outcome-redundancy structure (for diversity). Compares an embedding+panel on
one cell line. See KDPP_FAILURE_ANALYSIS.md for why these two diagnostics matter."""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import Ridge
from geneal.data.depmap import load_gene_effect, build_cell_line_dataset
from geneal.models.surrogate import GPRSurrogate
from geneal.models.kernels import posterior_correlation


def _r2(yt, yp):
    return 1 - np.sum((yt - yp) ** 2) / np.sum((yt - yt.mean()) ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene-effect", default="data/processed/depmap/gene_effect.parquet")
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--cell-line", default="ACH-000147")
    ap.add_argument("--n-train", type=int, default=400)
    args = ap.parse_args()

    ge = load_gene_effect(args.gene_effect)
    emb = pd.read_parquet(args.embeddings)
    ds = build_cell_line_dataset(ge, emb, args.cell_line)
    X, y = ds.embeddings, ds.target
    n = len(y)
    rng = np.random.default_rng(0)
    idx = rng.permutation(n); ntr = min(args.n_train, n - 50)
    tr, te = idx[:ntr], idx[ntr:]

    gp = GPRSurrogate(n_iters=150).fit(X[tr], y[tr]); gm, _ = gp.predict(X[te])
    rd = Ridge(alpha=1.0).fit(X[tr], y[tr])
    print(f"=== embedding diagnostics: {args.embeddings} on {args.cell_line} ===")
    print(f"n={n} dim={X.shape[1]}  strongly-lethal frac (lethality>1): {(y>1).mean():.3f}")
    print(f"GP held-out R2 {_r2(y[te],gm):.4f}  corr {np.corrcoef(y[te],gm)[0,1]:.4f}")
    print(f"Ridge held-out R2 {_r2(y[te],rd.predict(X[te])):.4f}")

    D = squareform(pdist(X)); order = np.argsort(D, axis=1)
    nn_d = [np.abs(y[i] - y[order[i, 1:11]]).mean() for i in range(n)]
    rd_d = [np.abs(y[i] - y[rng.choice(n, 10)]).mean() for i in range(n)]
    ratio = float(np.mean(nn_d) / np.mean(rd_d))
    print(f"NN/random lethality-diff ratio {ratio:.3f}  (<<1 => outcome structure; ~1 => none)")

    surr = GPRSurrogate(n_iters=100).fit(X[idx[:50]], y[idx[:50]])
    cand = idx[50:]
    _, cov = surr.predict_cov(X[cand]); S = posterior_correlation(cov)
    off = np.abs(S[np.triu_indices_from(S, 1)])
    print(f"k-DPP kernel off-diag |S| mean {off.mean():.4f}  (near 0 => diversity term inert)")


if __name__ == "__main__":
    main()
