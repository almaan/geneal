# Reproducing geneal

Every stage is deterministic (explicit seeds, pinned params/models). Run stages
via `make <target>` (see `Makefile`) or the scripts directly. Data + embedding
artifacts land in `data/` (gitignored); experiment outputs + reports in `res/`
(gitignored). Result notes are tracked in `data/*.md`.

## Environments

- **`geneal`** (main): torch 2.11, gpytorch, pyro, scikit-learn, plotly, jinja2,
  sentence-transformers, mygene, requests. `micromamba run -n geneal <cmd>`.
- **`scprint`** (separate, REQUIRED only for scPRINT embeddings): scPRINT pins an
  older torch that conflicts with `geneal`. Create:
  `micromamba create -y -n scprint -c conda-forge python=3.11 && micromamba run -n scprint pip install scprint`.
- **GPU**: ESM2 embedding auto-uses CUDA if present (verified on a B200). Everything
  else is CPU-fine.

## Architecture: genome-wide embedding caches, panel = row-subset

Embeddings are computed **once over all ~18.5k DepMap genes** (Entrez-indexed) and
cached. A "panel" (e.g. the 2043-gene HVG demo set, or 5116-gene set) is just a
**row-subset** of a cache — selected at experiment time via `--panel <entrez-file>`.
**Resizing the panel never re-embeds.** Genome-wide caches:
`pubmedbert_all.parquet`, `esm2_650m_all.parquet`, scPRINT (44k-gene native weight),
and STRING/CORUM edge-list + membership tables. **Co-dependency is NOT used** (it is
label-adjacent — derived from the effect matrix — so not a deployable prior; it was
only a diagnostic probe). Build all caches with `make pubmedbert-all esm2-all graphs-all`;
demo subsets (`pubmedbert_hvg` etc.) are kept for quick runs.

## Pipeline order

| stage | make target | script | output |
|---|---|---|---|
| 1. DepMap data | (manual, see below) | download + curate | `data/processed/depmap/gene_effect.parquet` |
| 2. Gene panel | `make panel` | `select_gene_panel.py` | `panel_hvg.txt` (2043 genes, deterministic) |
| 3. UniProt map | `make uniprot` | `map_genes_to_uniprot.py` | `uniprot_map_hvg.parquet` |
| 4a. ESM2 emb | `make esm2-emb` | `precompute_esm2.py` (GPU) | `esm2_650m_hvg.parquet` |
| 4b. PubMedBERT emb | `make pubmedbert-emb` | `embed_pubmedbert.py` | `pubmedbert_hvg.parquet` |
| 4c. scPRINT emb | `make scprint-emb` | `extract_scprint_gene_emb.py` (scprint env) | `scprint_hvg.parquet` |
| 5. Diagnostics | `make diagnostics` / `make multiline` | `diagnose_embeddings.py` / `diagnose_multiline.py` | redundancy ratio, R² tables |
| 6. Graph validation | `make string` / `make corum` | `validate_string.py` / `validate_corum.py` | redundancy ratios |
| 7. Risk nomination | `make risk` | `run_risk_nomination.py` | `res/runs_risk/<ts>/report.html` |
| 7b. Large sweep | `make risk-large` | `launch_risk_sweep.sh` | sharded combined report |
| 8. Dual/selectivity | `make dual` | `run_dual_experiment.py` | report |
| **9. DEFAULT ablation** | **`make ablation`** | **`run_ablation.py`** | **`res/runs_ablation/<ts>/report.html`** |

### Default output — the two-analysis ablation (stage 9)

`make ablation` (or `sbatch jobs/ablation.sh`) is the **standing default report**.
It runs the full active-learning loop (8 rounds × batch 10) for every method and
produces ONE report with two deliberately-separate analyses:

- **A — safety vs efficacy.** One axis (the safety rule): `none` (naive greedy) →
  `truncation` (known-toxicity ceiling) → `ehvi` (dual-objective EHVI acquisition,
  learned-toxicity ceiling), plus `random`/`coreset`/`typiclust` baselines. All six
  plotted on the efficacy–toxicity tradeoff (method-points + per-gene cloud).
- **B — diversity / robustness.** The operators `none`/`cap` (CORUM per-pathway)/
  `kdpp` (STRING-similarity k-DPP) layered on two bases (`greedy` and the winning
  safety rule from A). Capping is a bolt-on-any-method hedge.

**Representations:** PubMedBERT embeddings predict efficacy/toxicity; CORUM gives
pathway membership (capping/concentration); STRING gives the k-DPP similarity `S`.
STRING is a *similarity*, never a prediction embedding. Default scale: 5k panel, 6
lines, 3 seeds. Full genome: `make ablation-fullgenome` (or
`sbatch --export=ALL,PANEL= jobs/ablation.sh`).

### Stage 1 — DepMap data (one-time)
The DepMap 26Q1 CRISPRGeneEffect + Model are fetched from the portal download
manifest (`https://depmap.org/portal/api/download/files`, pre-signed GCS links,
no auth; links expire — re-fetch manifest to re-download). Curated transposed to
genes×cell_lines at `data/processed/depmap/gene_effect.parquet`. See
`data/DATA_STATUS.md` for the exact recon.

### Stage 4c — scPRINT (separate env)
Weights: HuggingFace `jkobject/scPRINT` `medium-v1.5.ckpt` (no auth). Needs the
Entrez→Ensembl map `data/processed/depmap/entrez_ensembl.parquet` (built via
`mygene` during the PubMedBERT/scPRINT prep). See `data/SCPRINT_VERIFY.md`.

## Reports

The headline experiment (`make risk`) writes a detailed self-contained HTML to
`res/runs_risk/<run>/report.html`: efficacy–risk Pareto, per-metric error-bar
curves, mean±95%CI tables, and a per-cell-line breakdown. Regenerate a report
from any saved run parquet:
`micromamba run -n geneal python -m geneal.report.risk_report <run>/risk.parquet <run>/report.html`

## Seeds / determinism

All RNG is explicit `numpy.random.default_rng(seed)`; experiment seeds are the
`--seeds` arg (Makefile default `0 1 2 3`). The Runner uses common random numbers
(shared init set + noise per seed across methods). Same seeds + same inputs →
identical outputs (verified: `tests/test_runner.py::test_runner_is_deterministic`,
cross-process repro test).

## Key result docs (tracked)
- `data/EMBEDDING_DIAGNOSTICS.md` — FM embeddings (ESM2/scPRINT/PubMedBERT) + STRING/CORUM all lack outcome-redundancy; co-dependency works.
- `data/RISK_NOMINATION_RESULTS.md` — greedy concentration risk vs per-pathway-cap hedge frontier.
- `data/STRING_VALIDATION.md`, `data/CORUM_VALIDATION.md` — graph-prior validation.
- `docs/superpowers/plans/` — all implementation plans; `docs/superpowers/specs/` — the design + moat.
