# geneal HVG Panel + ESM2-650M Upgrade — Implementation Plan (Plan 4)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or executing-plans. Steps use `- [ ]`.

**Goal:** Fix the two measured embedding/data bottlenecks: (1) the first-500-by-ID gene panel is 97% inert — replace with a ~2000-gene highly-variable / most-lethal panel (99% strongly-lethal-somewhere); (2) ESM2-35M predicts lethality weakly (held-out R²≈0.12) and embedding-neighbors aren't lethality-correlated (NN/random diff ratio≈0.95) — upgrade to ESM2-650M (dim 1280) on GPU. Then re-run the diagnostics + single-objective + dual experiments and compare. If both fail to lift predictive R² and the outcome-redundancy ratio, fall back to scPRINT (separate plan).

**Architecture:** A panel-selection script computes per-gene cross-cell-line variance + min-effect from the full DepMap matrix and writes a ~2000-gene HVG∪most-lethal panel (Entrez ids). The existing UniProt mapping (with pagination, already added) maps that panel; the existing ESM2 pipeline embeds it with `esm2_t33_650M_UR50D` on `device="cuda"` (B200, 182GB — verified available). The existing depmap adapter + Runner + metric panel + reports all consume the new embedding parquet unchanged (keyed by Entrez). Diagnostics re-run to quantify the lift.

**Tech Stack:** existing geneal stack; GPU verified (NVIDIA B200). No new deps.

**Measured baseline to beat (ESM2-35M, first-500 panel, ACH-000147):** GP held-out R²=0.12; NN/random lethality-diff ratio=0.95 (no outcome structure); 2.6% of panel strongly lethal; single-objective recall@50: greedy 0.44 / kdpp 0.37; k-DPP kernel near-diagonal (off-diag |S| mean 0.025).

**Honesty:** the point is to learn whether better representation + a non-inert panel change the conclusions. Do NOT tune to favor any method. Report the diagnostics (R², redundancy ratio, kernel off-diagonal magnitude) before and after, and whether the method ranking changes.

---

## File Structure

```
scripts/select_gene_panel.py     # CREATE: HVG ∪ most-lethal panel -> entrez list
scripts/diagnose_embeddings.py   # CREATE: held-out R², NN-redundancy ratio, kernel off-diag (reusable)
data/processed/depmap/panel_hvg.txt          # ARTIFACT: entrez ids
data/processed/depmap/uniprot_map_hvg.parquet # ARTIFACT
data/processed/embeddings/esm2_650m_hvg.parquet # ARTIFACT (genes x 1280)
tests/test_select_panel.py        # CREATE
```
(map_genes_to_uniprot.py + precompute_esm2.py + run_single_sweep.py + run_dual_experiment.py already exist and are reused with new --paths.)

---

## Task 1: gene-panel selection

**Files:** Create `scripts/select_gene_panel.py`, `tests/test_select_panel.py`.

- [ ] **Step 1: failing test `tests/test_select_panel.py`**

```python
# tests/test_select_panel.py
import numpy as np
import pandas as pd
from geneal.data.depmap import parse_entrez
import importlib.util, pathlib

# import the script module by path
_spec = importlib.util.spec_from_file_location(
    "select_gene_panel", pathlib.Path("scripts/select_gene_panel.py"))
sel = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(sel)


def test_select_panel_picks_variable_and_lethal(tmp_path):
    # 5 genes x 4 lines: g0 inert, g1 highly variable, g2 strongly lethal everywhere
    df = pd.DataFrame(
        {"L1": [0.0, -3.0, -4.0, 0.1, 0.0],
         "L2": [0.0, 3.0, -4.0, 0.0, 0.1],
         "L3": [0.0, -3.0, -4.0, 0.0, 0.0],
         "L4": [0.0, 3.0, -4.0, 0.1, 0.0]},
        index=pd.Index(["A (1)", "B (2)", "C (3)", "D (4)", "E (5)"], name="gene"))
    df.columns.name = "cell_line"
    p = tmp_path / "ge.parquet"; df.to_parquet(p)
    entrez = sel.select_panel(str(p), n_hvg=2, n_lethal=2)
    # g1 (entrez 2, highly variable) and g2 (entrez 3, strongly lethal) must be in
    assert 2 in entrez and 3 in entrez
    # g0 inert (entrez 1) should NOT dominate
    assert isinstance(entrez, list) and all(isinstance(e, int) for e in entrez)
```

- [ ] **Step 2: run, confirm fail.**

- [ ] **Step 3: implement `scripts/select_gene_panel.py`**

```python
# scripts/select_gene_panel.py
"""Select a gene panel = top-N highly-variable UNION top-N most-lethal genes,
from the full DepMap gene-effect matrix. Writes Entrez ids (one per line).

Rationale: the first-500-by-ID panel was 97% inert (no lethal signal). HVG +
most-lethal genes give a panel that is ~99% strongly-lethal in some cell line,
so top-k recovery is a meaningful task and methods can differentiate."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from geneal.data.depmap import load_gene_effect, parse_entrez


def select_panel(gene_effect_path: str, n_hvg: int = 2000,
                 n_lethal: int = 1000) -> list[int]:
    ge = load_gene_effect(gene_effect_path)
    eff = ge.to_numpy()
    gene_var = np.nanvar(eff, axis=1)
    gene_min = np.nanmin(eff, axis=1)  # most negative = most lethal somewhere
    hvg = np.argsort(gene_var)[::-1][:n_hvg]
    leth = np.argsort(gene_min)[:n_lethal]
    keep = np.union1d(hvg, leth)
    return [parse_entrez(ge.index[i]) for i in keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene-effect", default="data/processed/depmap/gene_effect.parquet")
    ap.add_argument("--n-hvg", type=int, default=2000)
    ap.add_argument("--n-lethal", type=int, default=1000)
    ap.add_argument("--out", default="data/processed/depmap/panel_hvg.txt")
    args = ap.parse_args()
    entrez = select_panel(args.gene_effect, args.n_hvg, args.n_lethal)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(str(e) for e in entrez))
    print(f"panel: {len(entrez)} entrez ids -> {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: run, confirm test passes.**
- [ ] **Step 5: run it for real:** `micromamba run -n geneal python scripts/select_gene_panel.py --n-hvg 2000 --n-lethal 1000`. Report panel size (~2364 expected).
- [ ] **Step 6: commit** `feat: HVG + most-lethal gene panel selection`.

---

## Task 2: map panel to UniProt + embed with ESM2-650M on GPU

This is a scripts-run task (the code exists). Produces the new embedding cache.

- [ ] **Step 1: map the panel to UniProt** (map_genes_to_uniprot.py already supports an entrez file + pagination):
```bash
micromamba run -n geneal python scripts/map_genes_to_uniprot.py \
  --entrez-file data/processed/depmap/panel_hvg.txt \
  --out data/processed/depmap/uniprot_map_hvg.parquet
```
Report mapped / total coverage. (~2364 ids; pagination required — the script has it.)

- [ ] **Step 2: embed with ESM2-650M on GPU.** precompute_esm2.py exposes --model and reads device via embed_sequences (which auto-uses cuda if available, but pass batch-size for the big model):
```bash
micromamba run -n geneal python scripts/precompute_esm2.py \
  --map data/processed/depmap/uniprot_map_hvg.parquet \
  --out data/processed/embeddings/esm2_650m_hvg.parquet \
  --model esm2_t33_650M_UR50D --batch-size 16
```
Report: # embedded, dim (1280), timing. NOTE: confirm embed_sequences uses GPU — `device=None` auto-selects cuda when available (verified B200 present). If it runs on CPU by mistake, pass an explicit device path through (check esm2_embed.embed_sequences signature — it accepts `device`; precompute_esm2 may need a --device passthrough; add it if missing, defaulting to auto). Long proteins truncate at max_len=1022.

- [ ] **Step 3: commit** any precompute_esm2.py/--device tweak (artifacts are gitignored): `feat: ESM2-650M GPU embedding of HVG panel`.

---

## Task 3: diagnostics — did the upgrade lift the signal?

**Files:** Create `scripts/diagnose_embeddings.py` (reusable; quantifies the two bottlenecks).

- [ ] **Step 1: write `scripts/diagnose_embeddings.py`** — given a gene_effect parquet, an embedding parquet, and a cell line: report (a) GP + Ridge held-out R² and correlation; (b) NN-redundancy ratio = mean|lethality diff to 10 embedding-NN| / mean|to 10 random|; (c) k-DPP kernel off-diagonal |S| magnitude on one AL round; (d) fraction of panel strongly lethal. Argparse: --embeddings, --cell-line, --gene-effect. Print a labeled table. (Reuse the diagnostic logic already prototyped in the session.)

```python
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
from geneal.models.acquisition import UCB
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
```

- [ ] **Step 2: run diagnostics on the NEW embedding+panel:**
```bash
micromamba run -n geneal python scripts/diagnose_embeddings.py \
  --embeddings data/processed/embeddings/esm2_650m_hvg.parquet --cell-line ACH-000147
```
- [ ] **Step 3: ALSO run it on the OLD embedding for a clean before/after:**
```bash
micromamba run -n geneal python scripts/diagnose_embeddings.py \
  --embeddings data/processed/embeddings/esm2_small.parquet --cell-line ACH-000147
```
- [ ] **Step 4: write `data/EMBEDDING_DIAGNOSTICS.md`** with the before/after table (R², redundancy ratio, off-diag |S|, lethal fraction) and an honest read: did 650M + HVG lift predictive R²? did it create any outcome-redundancy structure (ratio moving below ~0.9)? If the redundancy ratio is still ≈1, state plainly that protein-sequence embeddings don't encode knockout-effect similarity regardless of size, and scPRINT (expression-based) is the indicated next step.
- [ ] **Step 5: commit** `feat: embedding diagnostics + before/after ESM2-650M vs 35M`.

---

## Task 4: re-run experiments on the upgraded embedding+panel

- [ ] **Step 1: single-objective sweep** (4 lines, 2 seeds) on the new embedding:
```bash
micromamba run -n geneal python scripts/run_single_sweep.py \
  --embeddings data/processed/embeddings/esm2_650m_hvg.parquet \
  --n-cell-lines 4 --seeds 0 1 --k 50 --out-root res/runs_single_650m
```
Report the per-metric final mean±CI per method.

- [ ] **Step 2: dual selectivity** on the new embedding:
```bash
micromamba run -n geneal python scripts/run_dual_experiment.py \
  --embeddings data/processed/embeddings/esm2_650m_hvg.parquet \
  --seeds 0 1 --out-root res/runs_dual_650m
```
Report the differential-target per-metric mean±CI.

- [ ] **Step 3: append to `data/REAL_RESULT_PANEL.md`** a "650M + HVG panel" section: the new single + dual numbers vs the 35M/first-500 baseline, and an HONEST read — did better representation + a non-inert panel (a) raise absolute recall, (b) change the method ranking (does k-DPP now beat greedy on single-objective? does the selectivity flip strengthen?). Do NOT tune.
- [ ] **Step 4: commit** `experiment: re-run single + dual on ESM2-650M HVG panel`.

---

## Self-Review (completed during planning)

- **Spec coverage (user's ask "bigger ESM2, increase panel to most HVG, scPRINT if neither works"):** bigger ESM2 = 650M (Task 2) ✓; HVG panel (Task 1, ~2364 genes, 99% lethal-somewhere vs 19% before) ✓; scPRINT = explicit fallback gated on Task 3 diagnostics (out of scope here, separate plan) ✓.
- **Before/after rigor:** Task 3 runs diagnostics on BOTH old and new embeddings so the lift is measured, not assumed.
- **Reuse:** mapping + embedding + experiment scripts already exist; only panel-selection + diagnostics are new code (both small, tested where it matters). GPU verified (B200).
- **Honesty guards:** Tasks 3–4 forbid tuning; require reporting whether ranking changes; pre-commit to the scPRINT pivot if the redundancy ratio stays ≈1 (protein-sequence embeddings structurally can't encode knockout-effect similarity).
- **No placeholders:** all code complete. The one conditional (precompute_esm2 --device passthrough) is flagged with the fix.

## Out of scope
scPRINT embeddings (fallback, separate plan if Task 3 shows ESM2-650M still lacks predictive R² and outcome structure). Full 18k-gene embedding. Method development on the k-DPP kernel (separate — see KDPP_FAILURE_ANALYSIS.md directions).
