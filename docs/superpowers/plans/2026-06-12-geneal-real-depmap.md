# geneal Real DepMap + ESM2 — Implementation Plan (Plan 2 of 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Run the geneal AL framework (k-DPP + baselines) on REAL DepMap CRISPR lethality with ESM2 gene embeddings, producing the decisive experiment: does k-DPP beat quality-only greedy and diversity-only (CoreSet/TypiClust) on recovering the most-lethal knockouts in real cell lines? This is the MLCB 8-page differentiator AND the test of whether the k-DPP-vs-greedy moat survives on natural landscapes (it was fragile on synthetic).

**Architecture:** A DepMap adapter turns the curated `gene_effect.parquet` into a per-cell-line `Dataset` (target = lethality = −gene_effect, so higher = more lethal, matching the framework convention). An ESM2 embedding pipeline maps DepMap gene Entrez IDs → UniProt canonical protein → ESM2 mean-pooled embedding, cached to parquet. The existing Runner/Selection/Report stack then runs unchanged on the real `Dataset`. Everything is structured **subset-first**: a few hundred genes × 1–2 cell lines gives the decisive signal in minutes before scaling to all ~18k genes.

**Tech Stack:** existing geneal stack + `fair-esm` (installed), UniProt REST for gene→protein mapping. Data already on disk: `data/processed/depmap/gene_effect.parquet` (genes `SYMBOL (EntrezID)` × cell_lines `ACH-...`, 18531×1208), `data/raw/depmap/Model.csv` (cell-line metadata). See `data/DATA_STATUS.md`.

**Critical-path note (MLCB July 1):** Tasks 1–3 (adapter + a SMALL embedding cache + first real run) produce a publishable signal. Task 4 (full-genome embedding) is the scale-up; do it once the small result confirms the effect is worth the GPU time. Do NOT block the decisive experiment on embedding all 18k genes.

**Convention:** target = lethality = **−(Chronos gene_effect)**. Chronos: more negative effect = more essential/lethal → negated, higher = more lethal = better, consistent with the framework (acquisitions maximize). NaN gene-effect entries are dropped per cell line.

---

## File Structure

```
src/geneal/data/depmap.py            # CREATE: load curated parquet -> per-cell-line Dataset
src/geneal/data/esm2_embed.py        # CREATE: gene Entrez -> UniProt seq -> ESM2 embedding, cached
scripts/map_genes_to_uniprot.py      # CREATE: Entrez -> UniProt accession + canonical sequence (UniProt REST)
scripts/precompute_esm2.py           # CREATE: run esm2_embed over a gene list -> parquet cache
scripts/run_real_experiment.py       # CREATE: hydra entrypoint for a real-data run
conf/dataset/depmap.yaml             # CREATE
conf/config_real.yaml                # CREATE: real-data top-level config (all 6 methods)
tests/test_depmap.py                 # CREATE
tests/test_esm2_embed.py             # CREATE (uses smallest ESM2, 2-3 toy seqs)
data/processed/embeddings/esm2.parquet   # ARTIFACT (gitignored): genes x 320
data/processed/depmap/uniprot_map.parquet # ARTIFACT: entrez -> accession,sequence
```

---

## Task 1: DepMap adapter → per-cell-line Dataset

**Files:**
- Create: `src/geneal/data/depmap.py`
- Test: `tests/test_depmap.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_depmap.py
import numpy as np
import pandas as pd
import pytest
from geneal.data.depmap import load_gene_effect, build_cell_line_dataset, parse_entrez


def test_parse_entrez():
    assert parse_entrez("A1BG (1)") == 1
    assert parse_entrez("TP53 (7157)") == 7157


def _toy_effect(tmp_path):
    # genes x cell_lines, with a NaN to exercise dropping
    df = pd.DataFrame(
        {"ACH-001": [-2.0, 0.1, np.nan], "ACH-002": [-1.0, 0.0, -3.0]},
        index=pd.Index(["TP53 (7157)", "A1BG (1)", "EGFR (1956)"], name="gene"),
    )
    df.columns.name = "cell_line"
    p = tmp_path / "ge.parquet"
    df.to_parquet(p)
    return p


def test_load_gene_effect_roundtrip(tmp_path):
    p = _toy_effect(tmp_path)
    df = load_gene_effect(p)
    assert df.shape == (3, 2)
    assert df.index.name == "gene"


def test_build_dataset_target_is_negated_effect_and_drops_nan(tmp_path):
    p = _toy_effect(tmp_path)
    # embeddings keyed by entrez id; provide a tiny embedding table
    emb = pd.DataFrame(
        np.arange(9).reshape(3, 3).astype(float),
        index=pd.Index([7157, 1, 1956], name="entrez"),
    )
    ds = build_cell_line_dataset(load_gene_effect(p), emb, cell_line="ACH-001")
    # EGFR is NaN for ACH-001 -> dropped; 2 genes remain
    assert ds.n_genes == 2
    # target = -effect : TP53 effect -2.0 -> target +2.0 (most lethal -> highest)
    i = ds.gene_names.index("TP53 (7157)")
    assert np.isclose(ds.target[i], 2.0)
    # embedding row matches the gene's entrez id
    assert ds.embeddings.shape == (2, 3)


def test_build_dataset_requires_embedding_coverage(tmp_path):
    p = _toy_effect(tmp_path)
    emb = pd.DataFrame(np.zeros((1, 3)), index=pd.Index([7157], name="entrez"))
    # only TP53 has an embedding; A1BG dropped for missing embedding
    ds = build_cell_line_dataset(load_gene_effect(p), emb, cell_line="ACH-001")
    assert ds.n_genes == 1
    assert ds.gene_names == ["TP53 (7157)"]
```

- [ ] **Step 2: Run, confirm fail** (`ModuleNotFoundError: geneal.data.depmap`)

Run: `micromamba run -n geneal pytest tests/test_depmap.py -v`

- [ ] **Step 3: Implement `src/geneal/data/depmap.py`**

```python
# src/geneal/data/depmap.py
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd
from geneal.data.dataset import Dataset

_ENTREZ_RE = re.compile(r"\((\d+)\)\s*$")


def parse_entrez(gene_label: str) -> int:
    """Extract the Entrez id from a DepMap gene label like 'TP53 (7157)'."""
    m = _ENTREZ_RE.search(gene_label)
    if not m:
        raise ValueError(f"no entrez id in gene label: {gene_label!r}")
    return int(m.group(1))


def load_gene_effect(path) -> pd.DataFrame:
    """Load the curated genes x cell_lines Chronos gene-effect matrix."""
    return pd.read_parquet(path)


def build_cell_line_dataset(gene_effect: pd.DataFrame, embeddings: pd.DataFrame,
                            cell_line: str) -> Dataset:
    """Build a per-cell-line Dataset.

    gene_effect: genes (index 'SYMBOL (Entrez)') x cell_lines.
    embeddings:  rows indexed by Entrez id, columns = embedding dims.
    target = -gene_effect for the chosen cell line (higher = more lethal).
    Genes with NaN effect for this cell line OR no embedding are dropped.
    """
    if cell_line not in gene_effect.columns:
        raise KeyError(f"cell line {cell_line!r} not in gene-effect matrix")
    col = gene_effect[cell_line].dropna()
    rows, target, names = [], [], []
    emb_index = set(embeddings.index)
    for label, effect in col.items():
        ent = parse_entrez(label)
        if ent not in emb_index:
            continue
        rows.append(embeddings.loc[ent].to_numpy(dtype=float))
        target.append(-float(effect))  # lethality = -effect
        names.append(label)
    if not names:
        raise ValueError("no genes with both an effect and an embedding")
    return Dataset(embeddings=np.vstack(rows),
                   target=np.asarray(target), gene_names=names)
```

- [ ] **Step 4: Run, confirm 5 passed**

Run: `micromamba run -n geneal pytest tests/test_depmap.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/geneal/data/depmap.py tests/test_depmap.py
git commit -m "feat: add DepMap per-cell-line Dataset adapter"
```

---

## Task 2: ESM2 embedding pipeline (small-first)

**Files:**
- Create: `src/geneal/data/esm2_embed.py`
- Test: `tests/test_esm2_embed.py`

- [ ] **Step 1: Write the failing test** (uses the smallest ESM2; CPU; 2 toy sequences)

```python
# tests/test_esm2_embed.py
import numpy as np
import pytest
from geneal.data.esm2_embed import embed_sequences


def test_embed_sequences_shape_and_determinism():
    seqs = {"P1": "MKTAYIAKQR", "P2": "MQIFVKTLTG"}
    a = embed_sequences(seqs, model_name="esm2_t6_8M_UR50D")
    b = embed_sequences(seqs, model_name="esm2_t6_8M_UR50D")
    assert set(a.index) == {"P1", "P2"}
    assert a.shape[1] == 320  # esm2_t6_8M embedding dim
    # deterministic
    np.testing.assert_allclose(a.loc["P1"].to_numpy(), b.loc["P1"].to_numpy(), atol=1e-5)
    # different sequences -> different embeddings
    assert not np.allclose(a.loc["P1"].to_numpy(), a.loc["P2"].to_numpy())
```

- [ ] **Step 2: Run, confirm fail.** Run: `micromamba run -n geneal pytest tests/test_esm2_embed.py -v`

- [ ] **Step 3: Implement `src/geneal/data/esm2_embed.py`**

```python
# src/geneal/data/esm2_embed.py
from __future__ import annotations
import os
import numpy as np
import pandas as pd

# keep ESM checkpoints inside the project data dir (set before torch.hub use)
os.environ.setdefault("TORCH_HOME", os.path.join("data", "raw", "torch_hub"))

_MODEL_DIM = {"esm2_t6_8M_UR50D": 320, "esm2_t12_35M_UR50D": 480,
              "esm2_t30_150M_UR50D": 640, "esm2_t33_650M_UR50D": 1280}


def embed_sequences(sequences: dict[str, str], model_name: str = "esm2_t33_650M_UR50D",
                    batch_size: int = 8, device: str | None = None,
                    max_len: int = 1022) -> pd.DataFrame:
    """Mean-pooled ESM2 residue embeddings for {id: protein_sequence}.

    Returns a DataFrame indexed by id, columns = embedding dims. Sequences longer
    than max_len are truncated (ESM2 context limit). CPU by default; pass
    device='cuda' to use a GPU. Deterministic (eval mode, no dropout).
    """
    import torch
    import esm

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, alphabet = getattr(esm.pretrained, model_name)()
    model = model.eval().to(device)
    bc = alphabet.get_batch_converter()
    repr_layer = model.num_layers

    ids = list(sequences.keys())
    out: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for i in range(0, len(ids), batch_size):
            chunk = ids[i:i + batch_size]
            data = [(k, sequences[k][:max_len]) for k in chunk]
            _, _, toks = bc(data)
            toks = toks.to(device)
            res = model(toks, repr_layers=[repr_layer])["representations"][repr_layer]
            for j, k in enumerate(chunk):
                L = len(sequences[k][:max_len])
                # tokens 1..L are residues (0 is BOS); mean-pool over residues
                vec = res[j, 1:L + 1].mean(0).cpu().numpy()
                out[k] = vec
    return pd.DataFrame.from_dict(out, orient="index")
```

- [ ] **Step 4: Run, confirm 1 passed** (downloads the ~29MB small model once). Run: `micromamba run -n geneal pytest tests/test_esm2_embed.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/geneal/data/esm2_embed.py tests/test_esm2_embed.py
git commit -m "feat: add ESM2 sequence embedding pipeline"
```

---

## Task 3: Gene→UniProt mapping + SMALL embedding cache + first real run

This task produces the **decisive small-scale result**. It is a scripts+manual-run task, not a pure-TDD task — the deliverable is a real recall@k comparison, captured in a status note.

**Files:**
- Create: `scripts/map_genes_to_uniprot.py`
- Create: `scripts/precompute_esm2.py`
- Create: `scripts/run_real_experiment.py`
- Create: `conf/dataset/depmap.yaml`, `conf/config_real.yaml`

- [ ] **Step 1: Write `scripts/map_genes_to_uniprot.py`**

Maps Entrez ids → UniProt canonical accession + protein sequence via the UniProt REST ID-mapping API (no auth). Takes a list of Entrez ids, writes `data/processed/depmap/uniprot_map.parquet` (columns: entrez, accession, sequence). Use the idmapping run/poll/stream endpoints:

```python
# scripts/map_genes_to_uniprot.py
"""Entrez GeneID -> UniProtKB canonical accession + sequence, via UniProt REST.

Usage: python scripts/map_genes_to_uniprot.py --entrez-file <ids.txt> --out data/processed/depmap/uniprot_map.parquet
or:    python scripts/map_genes_to_uniprot.py --from-gene-effect data/processed/depmap/gene_effect.parquet --limit 500
"""
from __future__ import annotations
import argparse, time, io
from pathlib import Path
import requests
import pandas as pd
from geneal.data.depmap import parse_entrez, load_gene_effect

UNIPROT = "https://rest.uniprot.org"


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
        # JSON results carry both 'from' (GeneID) and the UniProt entry
        jr = requests.get(f"{UNIPROT}/idmapping/uniprotkb/results/{job}"
                          "?format=json&fields=accession,sequence,reviewed&size=500")
        jr.raise_for_status()
        data = jr.json()
        rows = []
        for rec in data.get("results", []):
            ent = int(rec["from"])
            to = rec["to"]
            acc = to["primaryAccession"]
            seq = to["sequence"]["value"]
            reviewed = to.get("entryType", "").lower().startswith("uniprotkb reviewed")
            rows.append({"entrez": ent, "accession": acc, "sequence": seq,
                         "reviewed": reviewed})
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
```

NOTE: the UniProt id-mapping JSON pagination uses a `next` link header for >500 results; for the small subset (`--limit 500`) one page suffices. If scaling to all genes (Task 4) the script must follow pagination — handle that in Task 4, not here. If the JSON `entryType` reviewed-detection or pagination misbehaves, report it; the canonical-sequence selection is the only part that matters.

- [ ] **Step 2: Run the mapping on a SMALL subset (first ~500 genes)**

```bash
micromamba run -n geneal python scripts/map_genes_to_uniprot.py \
  --from-gene-effect data/processed/depmap/gene_effect.parquet --limit 500 \
  --out data/processed/depmap/uniprot_map_small.parquet
```
Expected: prints "mapped N / 500 ..." with N a large fraction of 500. Report N. If the API errors or returns very few, report exactly what came back.

- [ ] **Step 3: Write `scripts/precompute_esm2.py`** (embeds the mapped sequences, caches by Entrez)

```python
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
```

- [ ] **Step 4: Run the small embedding** (use the small ESM2 model first for speed; upgrade model in Task 4)

```bash
micromamba run -n geneal python scripts/precompute_esm2.py \
  --map data/processed/depmap/uniprot_map_small.parquet \
  --out data/processed/embeddings/esm2_small.parquet --model esm2_t12_35M_UR50D
```
Expected: "embedded N proteins, dim=480 -> ...". Report N + timing.

- [ ] **Step 5: Pick the cell line(s).** Choose 1–2 well-characterized cancer cell lines from `data/raw/depmap/Model.csv` with low NaN rates. A robust default: pick the cell line column with the FEWEST NaNs among the embedded genes. Print the chosen ModelID(s) and their OncotreeLineage from Model.csv.

- [ ] **Step 6: Write `scripts/run_real_experiment.py` + configs, then RUN the decisive comparison.**

`conf/dataset/depmap.yaml`:
```yaml
gene_effect_path: data/processed/depmap/gene_effect.parquet
embedding_path: data/processed/embeddings/esm2_small.parquet
cell_line: ACH-000001   # overridden on CLI with the chosen line
```

`scripts/run_real_experiment.py`:
```python
# scripts/run_real_experiment.py
"""Run all methods on a real DepMap cell line and write a report + Pareto."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from geneal.data.depmap import load_gene_effect, build_cell_line_dataset
from geneal.experiment.objects import ObjectiveObject, DesignObject, Method, Experiment
from geneal.metrics.recall import RecallAtK
from geneal.models.acquisition import UCB, RandomAcquisition, GreedyMean
from geneal.models.selection import KDPP, TopQGreedy, GreedyFantasy, CoreSet, TypiClust
from geneal.models.surrogate import GPRSurrogate
from geneal.models.noise import GaussianNoise
from geneal.runner.runner import Runner
from geneal.runner.logging import write_run_dir
from geneal.report.report import build_report
from geneal.report.pareto import pareto_summary, pareto_plot_div


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene-effect", default="data/processed/depmap/gene_effect.parquet")
    ap.add_argument("--embeddings", default="data/processed/embeddings/esm2_small.parquet")
    ap.add_argument("--cell-line", required=True)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--n-initial", type=int, default=50)
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out-root", default="res/runs_real")
    args = ap.parse_args()

    ge = load_gene_effect(args.gene_effect)
    emb = pd.read_parquet(args.embeddings)
    ds = build_cell_line_dataset(ge, emb, cell_line=args.cell_line)
    print(f"cell line {args.cell_line}: {ds.n_genes} genes with embeddings")

    def gpr():
        return GPRSurrogate(n_iters=100)
    methods = [
        Method("kdpp", UCB(2.0), KDPP("greedy"), gpr()),
        Method("greedy", UCB(2.0), TopQGreedy(), gpr()),
        Method("fantasy", UCB(2.0), GreedyFantasy(), gpr()),
        Method("coreset", GreedyMean(), CoreSet(), gpr()),
        Method("typiclust", GreedyMean(), TypiClust(), gpr()),
        Method("random", RandomAcquisition(), TopQGreedy(), gpr()),
    ]
    exp = Experiment(ds, ObjectiveObject(RecallAtK(k=args.k), "maximize"),
                     DesignObject(n_rounds=args.rounds, batch_size=args.batch,
                                  n_initial=args.n_initial, seed=0),
                     GaussianNoise(args.sigma), methods)
    df = Runner().run(exp, seeds=args.seeds)
    run_dir = write_run_dir(out_root=args.out_root, df=df,
                            config=vars(args), extra_manifest={"cell_line": args.cell_line})
    build_report(pd.read_parquet(run_dir / "rounds.parquet"), run_dir / "report.html")
    summ = pareto_summary(df)
    (run_dir / "pareto.html").write_text(pareto_plot_div(summ))
    final = df[df["round"] == df["round"].max()].groupby("method")["metric"].mean().sort_values(ascending=False)
    print("=== final recall@%d ===" % args.k); print(final.to_string())
    print("=== pareto ==="); print(summ.to_string())
    print(f"run dir: {run_dir}")


if __name__ == "__main__":
    main()
```

Run it on the chosen cell line:
```bash
micromamba run -n geneal python scripts/run_real_experiment.py \
  --cell-line <CHOSEN_ACH_ID> --k 50 --rounds 10 --batch 10 --n-initial 50 --seeds 0 1 2
```

- [ ] **Step 7: Record the decisive result.** Write `data/REAL_RESULT_SMALL.md` with: cell line(s), #genes, the final recall@k per method, the Pareto summary, and an HONEST one-paragraph read — does k-DPP beat greedy on real data? does it beat the diversity-only baselines? This is the go/no-go signal for the 8-page framing. Do NOT tune parameters to force k-DPP to win; report what happens.

- [ ] **Step 8: Commit (scripts + configs + result note; NOT the data/embedding artifacts — they're gitignored)**

```bash
git add scripts/map_genes_to_uniprot.py scripts/precompute_esm2.py scripts/run_real_experiment.py conf/dataset/depmap.yaml
git add data/REAL_RESULT_SMALL.md 2>/dev/null || true
git commit -m "feat: real DepMap experiment pipeline + first small-scale result"
```

---

## Task 4: Scale up (full-genome embedding + multi-cell-line) — ONLY after Task 3 confirms the signal

Do this only if the small result is promising enough to warrant the GPU time and the 8-page push.

- [ ] **Step 1:** Extend `map_genes_to_uniprot.py` to follow UniProt result pagination (the `next` link) so it maps all ~18.5k Entrez ids, not just 500. Run on GPU node.
- [ ] **Step 2:** Run `precompute_esm2.py` over ALL mapped genes with the larger model `esm2_t33_650M_UR50D` (dim 1280) on GPU; cache to `data/processed/embeddings/esm2.parquet`. Report coverage (genes embedded / total) and timing.
- [ ] **Step 3:** Pick a panel of cell lines (e.g. 5–10 spanning lineages from Model.csv). Run `run_real_experiment.py` per cell line; aggregate recall@k and Pareto across the panel.
- [ ] **Step 4:** Write `data/REAL_RESULT_FULL.md` with the aggregated per-cell-line comparison + an honest read. This is the source for the MLCB Figure 2.
- [ ] **Step 5:** Commit scripts changes + result notes (not artifacts).

---

## Self-Review (completed during planning)

- **Spec coverage:** DepMap adapter (T1) ✓; ESM2 embeddings (T2) ✓; gene→protein mapping via UniProt/Entrez per the data-access memo (T3) ✓; real run of ALL methods incl k-DPP + CoreSet + TypiClust baselines (T3) ✓; honest decisive-result capture (T3 Step 7, T4 Step 4) ✓; subset-first to protect the July 1 critical path ✓.
- **Convention consistency:** target = −gene_effect everywhere (T1 implementation + test); embeddings keyed by Entrez id (T1, T2, T3 all agree); cell line chosen by lowest-NaN coverage (T3 Step 5).
- **Honesty guards:** T3 Step 7 and T4 Step 4 explicitly forbid tuning to force a k-DPP win — the real-data result is the open question the synthetic work could not settle.
- **No placeholders:** every code step is complete. The only flagged conditionals are UniProt JSON pagination (deferred to T4) and reviewed-entry detection (report if it misbehaves).
- **Artifacts gitignored:** all parquet caches live under data/ (gitignored); only scripts/configs/result-notes are committed.

## Out of scope
Selectivity / two-cell-line differential objective (separate extension, build after the single-objective real result lands). scPRINT embeddings (ESM2 is the chosen first path; scPRINT only if ESM2 underperforms and time allows).
```
