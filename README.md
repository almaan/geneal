# geneal — Selective & Diverse Target Nomination

**Multi-Objective Active Learning and Portfolio Hedging for target discovery.**

Code and reproduction pipeline for *"Nominating Selective and Diverse Targets:
Multi-Objective Active Learning and Portfolio Hedging"* (MLCB). The framework treats
target nomination as a **selective multi-objective** problem — maximize efficacy in a
target cell population while minimizing activity in non-target populations — under a
fixed experimental budget, and then **hedges** the nominated cohort against
mechanistic redundancy using external biological knowledge.

Everything is evaluated on the **DepMap** CRISPR gene-effect resource, treated as an
offline oracle: ground-truth lethality is hidden and revealed only when the active
learning loop "assays" a batch of genes.

---

## What the method does

The pipeline runs an active-learning loop over a genome-wide candidate pool
(~18.5k genes), then nominates a final cohort of *K* genes. Two contributions:

**1. Selectivity during acquisition (Section A → Table 1).**
A Gaussian-process surrogate (Matérn-5/2; a joint multitask GP over efficacy +
non-target activity) drives acquisition. Selectivity is enforced either by a
multi-objective acquisition — **Monte-Carlo Expected Hypervolume Improvement (EHVI)**,
batched greedily with kriging-believer fantasies — or by a **threshold filter**
`a(x)·1[μ_tox(x) ≤ τ]` applied dynamically per round or post-hoc at nomination.
Baselines: random, density/cluster (TypiClust), informative-diverse (quality-weighted
k-DPP), and single-objective greedy (UCB).

**2. Hedging at nomination (Section B → Tables 2 / A.2).**
The final cohort is selected with a **quality-weighted k-DPP**,
`L_ij = e_i · S_ij · e_j` (efficacy × similarity × efficacy), maximized by greedy MAP.
The similarity `S` injects prior biology — **CORUM** protein-complex Jaccard overlap —
so redundant, same-complex picks are penalized. Portfolio metrics (concentration,
robustness, risk = wᵀSw, effective bets N_eff = K²/1ᵀS1) quantify the diversification,
evaluated on CORUM and re-checked on a held-out **STRING** graph.

---

## Repository layout

```
src/geneal/            the importable library (see "Package" below)
scripts/               pipeline entrypoints (data prep, run, aggregate, report)
jobs/                  SLURM job scripts (per-line shard + aggregation)
tests/                 pytest suite
slurm_env.template.sh  cluster/micromamba config template (copy -> slurm_env.sh)
README.md              this file
```

---

## Installation

```bash
micromamba create -y -n geneal -c conda-forge python=3.11
micromamba run -n geneal pip install -e ".[dev]"   # torch, gpytorch, scikit-learn, jinja2, mygene, networkx
```
Run anything with `micromamba run -n geneal <cmd>`. CPU is sufficient throughout.

---

## Data

Place the following under `data/` (all gitignored):

| file | source |
|---|---|
| `data/processed/depmap/gene_effect.parquet` | DepMap Public **25Q3** `CRISPRGeneEffect` (genes × cell lines, Chronos), row index `"SYMBOL (Entrez)"`. Download the CSV from [depmap.org](https://depmap.org) and save as parquet. |
| `data/corum_dl/humanComplexes.txt` | CORUM (mammalian protein complexes). |
| `data/string_dl/` | STRING human `protein.links` + `protein.aliases`. |

---

## Reproducing the results

### Step 1 — build caches

```bash
# CORUM + STRING edge lists and CORUM membership
micromamba run -n geneal python scripts/build_graph_caches.py

# gene list = ALL DepMap genes (genome-wide; no panel subsetting)
micromamba run -n geneal python -c "from geneal.data.depmap import load_gene_effect,parse_entrez; ge=load_gene_effect('data/processed/depmap/gene_effect.parquet'); open('data/processed/depmap/all_genes.txt','w').write('\n'.join(str(parse_entrez(g)) for g in ge.index))"

# PubMedBERT text embeddings (mygene.info annotations -> NeuML/pubmedbert-base-embeddings)
micromamba run -n geneal python scripts/embed_pubmedbert.py \
  --panel data/processed/depmap/all_genes.txt \
  --out   data/processed/embeddings/pubmedbert_all.parquet
```

### Step 2 — cluster config (SLURM only, one-time)

No cluster-specific values are hardcoded in the job scripts; the launcher reads them
from `slurm_env.sh`, created once from the committed template:

```bash
cp slurm_env.template.sh slurm_env.sh    # gitignored; never committed
```

Then edit `slurm_env.sh`:

| variable | meaning |
|---|---|
| `GENEAL_PARTITION` | SLURM partition (`sbatch --partition`) |
| `GENEAL_ACCOUNT` | SLURM account (`""` if your cluster has none) |
| `GENEAL_SHARD_TIME` / `_CPUS` / `_MEM` | per-line array-task walltime / cpus-per-task / mem-per-cpu |
| `GENEAL_AGG_TIME` / `_CPUS` / `_MEM` | aggregation-job resources |
| `MAMBA_EXE`, `MAMBA_ROOT_PREFIX` | micromamba binary + root prefix |
| `GENEAL_ENV` | conda env name (default `geneal`) |

The launcher `source`s this file, passes `--partition/--account/--time/--cpus-per-task/
--mem-per-cpu` to `sbatch`, and forwards the micromamba vars to the jobs via
`--export=ALL`. Override the path with `GENEAL_SLURM_ENV=/path/to/env.sh`.

### Step 3 — run the analysis (genome-wide, joint multitask GP)

**SLURM** (one array shard per target line + a dependent aggregation job):
```bash
CONTRASTS=3 CONTRAST_IDS="ACH-002462 ACH-001310 ACH-000133" JOINT=1 SEEDS="0 1" \
  PANEL_A=none PANEL_B=none TAG=mlcb \
  bash scripts/launch_ablation_sweep.sh 12
# -> res/runs_ablation/sweep_mlcb_<ts>/report.html
```
Launcher knobs: `SEEDS` (default `0 1`), `JOINT` (`1`=joint multitask GP),
`CONTRASTS`/`CONTRAST_IDS`, `PANEL_A`/`PANEL_B` (`none`=full genome),
`ANALYSES` (`both`/`a`/`b`), `TAG`, `EXPORTFIGS`.

**Without SLURM** (portable, no `slurm_env.sh` needed):
```bash
micromamba run -n geneal python scripts/run_analysis.py \
  --panel-a none --panel-b none --joint-gp \
  --n-cell-lines 12 --contrast-lines ACH-002462 ACH-001310 ACH-000133 \
  --seeds 0 1 --K 30 --n-initial 40 --n-rounds 8 --batch 10 --tau 0.5
```

**Fixed configuration.** 12 target lines (top-12 fewest-NaN), 3 non-cancerous contrast
lines, 2 seeds, K=30 nominees, 40 seed genes, 8 rounds × 10 genes (120 assayed),
τ=0.5 (absolute Chronos scale), joint multitask GP. Results pool over 12 × 2 × 3 = 72
settings.

### Step 4 — the report → manuscript tables

The HTML report at `res/runs_ablation/<run>/report.html` maps to:

| report | manuscript |
|---|---|
| Section A table | **Table 1** (selectivity) |
| Section B.1 (evaluated on CORUM) | **Table 2** (CORUM hedging) |
| Section B.2 (evaluated on held-out STRING) | **Table A.2** (STRING hedging) |

If Section A and Section B were run separately (`--analyses a` / `--analyses b`),
stitch them into one report without recomputation:
```bash
micromamba run -n geneal python scripts/merge_ab_report.py \
  --a-run <A_run_dir> --b-run <B_run_dir> --out <merged_dir>
```

---

## The `geneal` package

`src/geneal/` is the importable library the scripts orchestrate. Core modules on the
reproduction path:

| module | role |
|---|---|
| `data/depmap.py` | load the Chronos gene-effect matrix; parse Entrez ids |
| `data/selective.py` | per-line datasets, contrast/aggregate toxicity, CORUM membership, contrast-line ranking |
| `models/surrogate.py` | `GPRSurrogate` — exact GP (Matérn-5/2, gpytorch) |
| `models/multitask.py` | `MultiTaskGPR` — joint GP over (efficacy, non-target activity) |
| `models/multiobjective.py` | Pareto fronts, hypervolume, Monte-Carlo EHVI |
| `models/selection.py` | acquisition-time baselines: CoreSet, TypiClust, k-DPP (greedy MAP / exact) |
| `models/hedged_selection.py` | nomination-time hedging: top-K / per-pathway cap / quality-weighted k-DPP |
| `metrics/portfolio.py` | concentration, robustness, portfolio risk, effective bets |
| `runner/ablation.py` | the analysis engine: acquisition loop, two-stage nomination, evaluation, similarity builders |
| `runner/bivariate.py` | bivariate EHVI active-learning loop |
| `runner/gp_cv.py` | method-independent 5-fold GP fit quality (R², Spearman) |
| `report/ablation_report.py` | HTML + LaTeX report (Tables 1 / 2 / A.2, figures) |

The package also contains exploratory modules beyond the manuscript pipeline; the
table above is the reproduction path.

---

## Scripts

| script | purpose |
|---|---|
| `scripts/build_graph_caches.py` | CORUM + STRING edge lists and CORUM membership caches |
| `scripts/embed_pubmedbert.py` | PubMedBERT gene-text embeddings (via mygene.info) |
| `scripts/run_analysis.py` | the analysis engine (Sections A + B); `--analyses {both,a,b}` |
| `scripts/aggregate_ablation.py` | concatenate per-line shards → one combined report |
| `scripts/merge_ab_report.py` | stitch a separate Section-A run with a Section-B run |
| `scripts/launch_ablation_sweep.sh` | SLURM sweep launcher (sharded + aggregation) |
| `jobs/ablation_shard.sh` | per-line array task |
| `jobs/ablation_aggregate.sh` | dependent aggregation job |

---

## Tests

```bash
micromamba run -n geneal pytest
```

---

## Notes

- **τ = 0.5** is an absolute value on the Chronos scale (a knockout is "safe" if its
  non-target lethality is at/below 0.5), not a quantile.
- `data/`, `res/`, `logs/`, and `slurm_env.sh` are gitignored.
