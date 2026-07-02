# Reproducing the MLCB analysis

Minimal pipeline to reproduce **Table 1** (selectivity), **Table 2** (CORUM
hedging), **Table A.2** (STRING hedging) and the HTML report for
*"Nominating Selective and Diverse Targets: Multi-Objective Active Learning and
Portfolio Hedging."*

Everything is deterministic given the seeds/params below, with one documented
exception (the joint-GP note at the end). Data + outputs are gitignored (`data/`,
`res/`).

## 1. Environment

```bash
micromamba create -y -n geneal -c conda-forge python=3.11
micromamba run -n geneal pip install -e ".[dev]"      # torch, gpytorch, sklearn, jinja2, mygene, networkx
```
Run anything with `micromamba run -n geneal <cmd>`. CPU is sufficient.

## 2. Inputs (place under `data/`)

| file | source |
|---|---|
| `data/processed/depmap/gene_effect.parquet` | DepMap Public **25Q3** `CRISPRGeneEffect` (genes × cell lines, Chronos), row index `"SYMBOL (Entrez)"`. Download the CSV from depmap.org and save as parquet. |
| `data/corum_dl/humanComplexes.txt` | CORUM (mammalian protein complexes). |
| `data/string_dl/` | STRING human `protein.links` + `protein.aliases`. |

## 3. Build caches

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

## 4. Run the analysis (genome-wide, joint multitask GP)

**SLURM (sharded: one job per target line + dependent aggregation):**
```bash
CONTRASTS=3 CONTRAST_IDS="ACH-002462 ACH-001310 ACH-000133" JOINT=1 SEEDS="0 1" \
  PANEL_A=none PANEL_B=none TAG=mlcb \
  bash scripts/launch_ablation_sweep.sh 12
# -> res/runs_ablation/sweep_mlcb_<ts>/report.html
```

**Single process (no SLURM):**
```bash
micromamba run -n geneal python scripts/run_analysis.py \
  --panel-a none --panel-b none --joint-gp \
  --n-cell-lines 12 --contrast-lines ACH-002462 ACH-001310 ACH-000133 \
  --seeds 0 1 --K 30 --n-initial 40 --n-rounds 8 --batch 10 --tau 0.5
```

Report → manuscript mapping:
- **Section A** table → **Table 1**
- **Section B.1** (evaluated on CORUM) → **Table 2**
- **Section B.2** (evaluated on held-out STRING) → **Table A.2**

Fixed config: 12 target lines (top-12 fewest-NaN), 3 non-cancerous contrast lines,
2 seeds, K=30 nominees, 40 seed genes, 8 rounds × 10 genes (120 assayed),
τ=0.5 (absolute Chronos scale), joint multitask GP. Results pool over
12 × 2 × 3 = 72 settings.

`--analyses {both,a,b}` runs Section A only, Section B only, or both.

## 5. (Optional) merge a separate A run with a separate B run

Only if Section A and Section B were run in different jobs:
```bash
micromamba run -n geneal python scripts/merge_ab_report.py \
  --a-run <A_run_dir> --b-run <B_run_dir> --out <merged_dir>
```

## Notes

- **Reproducibility caveat:** the joint multitask GP (`MultiTaskGPR`) initializes
  its task-covariance from torch's global RNG, which the pipeline does not seed, so
  the joint-GP methods (EHVI and all filtered variants) vary slightly run-to-run.
  Seed torch inside `MultiTaskGPR.fit` for exact reproduction. Single-task methods
  (greedy / random / cluster / info-diverse) are fully deterministic.
- Tests: `micromamba run -n geneal pytest`.

## Components

```
scripts/build_graph_caches.py     CORUM + STRING caches
scripts/embed_pubmedbert.py       PubMedBERT gene embeddings
scripts/run_analysis.py           the analysis engine (Sections A + B)
scripts/aggregate_ablation.py     concatenate shards -> combined report
scripts/merge_ab_report.py        stitch a separate A run + B run
scripts/launch_ablation_sweep.sh  SLURM sweep launcher
jobs/ablation_shard.sh            per-line array task
jobs/ablation_aggregate.sh        dependent aggregation job
src/geneal/                       importable library (surrogate, acquisition,
                                  nomination, metrics, report)
```
