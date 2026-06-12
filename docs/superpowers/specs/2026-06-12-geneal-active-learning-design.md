# geneal — Active Learning for Gene Knockout Selection (v1 Design)

**Date:** 2026-06-12
**Status:** Approved design, pre-implementation
**Package:** `geneal`

## 1. Purpose

Build a modular machine-learning framework that uses **active learning (AL)** to
reach a target conclusion faster than a baseline sampling method, on molecular
data.

**v1 use case:** Given a single cancer cell line, use AL to choose which gene to
knock out next, racing a greedy/random baseline to recover the set of most-lethal
knockout genes. The "experiment" is virtual: the AL loop reveals withheld DepMap
gene-effect values (plus emulated measurement noise) instead of running real
experiments.

The framework must generalize beyond this use case — every problem-specific piece
is swappable. v1 is the first concrete example, not the final shape. If it works,
it is publishable.

### Scientific positioning (vs NAIAD, arxiv 2411.12010)

The closest prior work, NAIAD, shares the AL-over-rounds frame on CRISPR data and
argues for *adaptive gene embeddings that scale with training data*, using a
**greedy max-predicted-effect** acquisition. To have a defensible edge rather than
re-implementing NAIAD's easier single-gene sub-problem, geneal has two primary
contributions — one **method**, one **benchmark axis**:

#### Primary method: quality-weighted k-DPP batch acquisition

The core technical contribution. NAIAD acquires **greedily** (top single predicted
effect); a real knockout round buys ~q≈10 assays at once. Picking the q
highest-scoring genes wastes assays on redundant, near-identical knockouts. We want
a batch that is simultaneously **high-performing and diverse**: many high-scoring
genes, no two of which are near-duplicates in *outcome* space.

Formalize as a **quality-weighted Determinantal Point Process (k-DPP)**. Build a
kernel over candidates:

```
L_ij = q_i · S_ij · q_j
```

- `q_i` = **quality** of gene i = a pluggable acquisition score (UCB / EI / mean;
  default UCB). Reuses the existing acquisition layer — the DPP wraps it.
- `S_ij` = **outcome similarity** = the surrogate's posterior *correlation* between
  genes i and j, from the joint predictive covariance.

Select the size-q subset maximizing `det(L_B)`. The determinant is large only when
picks are both high-quality (large diagonal) and non-redundant (off-diagonal
correlation small → near-orthogonal rows). Two near-identical genes make their rows
collinear → determinant collapses → the DPP will not take both.

**Why this is the unification of "informative" and "diverse":**
`log det(posterior covariance of a batch)` is the Gaussian joint entropy = the
batch's **information gain**. So max-`det` IS an information-theoretic batch
acquisition, with outcome-covariance-driven diversity built in — not q independent
high-scorers (NAIAD greedy), not an input-space diversity heuristic (greedy +
fantasies). Greedy top-q, greedy+fantasies, and random selection are all baselines
the DPP must beat.

**Solver:** subset max-`det` is NP-hard exactly. Ship **greedy MAP** (add the gene
with largest marginal det-gain, q times — fast, near-optimal, deterministic) as the
default, with exact/DPP-sampling pluggable behind the same Selection interface.

**Surrogate requirement (new vs v1):** this needs the surrogate to expose a **joint
predictive covariance** (`predict_cov(X) -> (mean, cov)`), not just marginal std.
gpytorch GP provides it natively; the BNN provides it via the sample covariance of
posterior predictive draws.

**Evaluation — does it deliver diverse AND high-performing batches?**
At fixed total budget, per round: (1) recall@k vs rounds (primary — diversity
should recover the top-k set faster by not wasting assays on redundancy); (2) batch
quality = mean true effect of the selected batch; (3) batch diversity = mean
pairwise outcome-distance within the batch (or `det` of the selected sub-covariance).
Headline result: a **quality–diversity Pareto plot** — greedy top-q sits in the
high-quality/low-diversity corner, random selection in the high-diversity/low-quality
corner; the k-DPP should Pareto-dominate both and win on recall@k. NAIAD's greedy
acquisition lives in a single corner and cannot produce this frontier.

#### Benchmark axis: zero-shot foundation-model embeddings

Supporting, not the headline. How far does the loop get using FROZEN gene/protein
FM embeddings (scPRINT, ESM2) as the prior, with no perturbation-trained embeddings
(contra NAIAD's adaptive embeddings)? Embedding source is an experimental axis
(scPRINT vs ESM2). The AL *baseline* — the control the AL loop must beat — is
**random selection** (no surrogate). A random embedding is kept only as a
sanity/ablation: with signal-free embeddings, AL should not beat random selection.
(No PCA baseline — PCA needs a feature matrix to decompose; the embeddings are
already the only per-gene features we have.)

Held in reserve: cross-cell-line transfer via FM embeddings. NAIAD does gene
*pairs*; geneal v1 is single-gene — no combinatorial-novelty claim without
extending there.

### Self-evolving requirement

The project is intended to evolve under supervision of an LLM agent + code
harness. This demands: clear, pluggable success metrics; reproducible runs;
machine-readable structured logs; and verbose, parseable outputs.

## 2. Key Decisions (from brainstorming)

| Decision | Choice |
|---|---|
| Success metric | **Pluggable** (in Objective). Default = **recover top-k set** (recall@k curve vs rounds). |
| Batch acquisition | Each round acquires a batch of size `q`. Selection is **pluggable**: v1 ships **top-q greedy** + **greedy+fantasies** (baselines); the **primary method** is a **quality-weighted k-DPP** (`L_ij = q_i·S_ij·q_j`, greedy-MAP default + sampling pluggable). See §1. |
| Surrogate I/O | **Per-cell-line** model: `gene_embedding -> effect`. v1 surrogate exposes `predict(X)->(mean,std)`; the k-DPP method additionally needs **joint covariance** `predict_cov(X)->(mean,cov)` (gpytorch native; BNN via sample cov). Abstractions keep **cell-line-as-feature** open for later. |
| Embeddings | **Precompute + cache** offline (genes × dim matrix). Source behind an **adapter**. Heavy models stay out of the AL loop. |
| First embedding source | **scPRINT** first; fall back to **ESM2** if derivation is hard. Both acceptable. |
| Environment | **Fresh micromamba env `geneal`** (user creates/populates from provided package list). Not shared `genml`. |
| Noise model | **Pluggable**; start with **fixed homoscedastic Gaussian**. |
| Surrogate covariance default | **Matérn kernel** for the GPR surrogate. |
| Architecture style | **Hydra `_target_` instantiation** on plain classes behind small ABCs. No custom registry. |
| Repro / harness for v1 | Seeded reproducibility, structured run logs, HTML report with latex tables. Multi-seed via Hydra multirun. |
| Method comparison fairness | AL and baseline share the **same cell line, same initial set, and paired noise draws** (common random numbers). Only selection differs. |

## 3. Architecture & Components

Config-driven. Plain classes behind small ABCs/Protocols, instantiated by Hydra
`_target_`. One round = one AL step.

### Experiment (container, built from Hydra config)
- **DataObject** — wraps the DepMap CRISPRGeneEffect matrix (cell_lines × genes)
  and the gene-embedding matrix (genes × dim). Knows feature names (genes) and
  row names. Provides candidate genes and ground-truth effect for the selected
  cell line. Ingestion adapters: DepMap CSV → internal representation.
- **ObjectiveObject** — what to optimize and the **success metric** (pluggable;
  default recall@k). Optimization direction (lethality = most-negative gene
  effect → minimize, or equivalently maximize lethality). Evaluates the metric
  each round.
- **DesignObject** — `n_rounds`, `batch_size` (q), `n_initial`, `seed`.

### Model
- **Surrogate** — ABC: `fit(X, y)`, `predict(X) -> (mean, std)`. Impls: **GPR**
  (gpytorch, Matérn default), **BNN** (pyro). Not limited to these.
- **Acquisition** — ABC: `score(surrogate, candidates) -> scores`. Impls: EIG,
  UCB, EI, etc.
- **Selection** — ABC: `select(scores_or_surrogate, q) -> indices`. Impls:
  **top-q greedy**, **greedy+fantasies** (conditions surrogate on fantasized
  outcomes for batch diversity). Built to allow more sophisticated approaches.

### NoiseModel
- ABC: `apply(value) -> noisy_value`. Impls: none / **fixed Gaussian** (first) /
  heteroscedastic / DepMap-derived (later).

### EmbeddingAdapter
- ABC: `embed(genes) -> matrix`. Impls: **scPRINT** (first), ESM2, placeholder
  (random/PCA). Used **offline** by the precompute script — never in the AL loop.

### Runner
- Runs an Experiment for ≥1 method (AL vs greedy/random baseline) across N seeds.
- Logs every round: config, selections, revealed values, metric-so-far →
  structured run directory.
- Snapshots resolved config, seed, code git SHA, env name for reproducibility.
- Multi-seed via Hydra multirun.

### Report
- Reads a run directory → HTML report: recall@k curves, AL-vs-baseline plots
  (plotly visuals), dropdown latex tables for manuscript transfer.

## 4. Data Flow

### Offline (once)
1. Download DepMap CRISPRGeneEffect → `data/raw/depmap/`.
2. Curate → internal matrix (cell_lines × genes) → `data/processed/`.
3. Precompute embeddings: `EmbeddingAdapter.embed(genes)` →
   `data/processed/embeddings/<source>.parquet` (genes × dim). scPRINT first.

### Per run (Hydra config → Runner)
1. Build Experiment: load processed matrix + cached embeddings. Pick one cell
   line → ground-truth `effect[gene]`.
2. DesignObject seeds the RNG. Pick `n_initial` genes (random, seeded) → revealed
   set. Reveal = read withheld value + `NoiseModel.apply`.
3. **Round loop** (`n_rounds`):
   a. Surrogate `.fit(X_revealed_emb, y_revealed_noisy)`.
   b. Acquisition `.score` over unrevealed candidates.
   c. Selection `.select(..., q)` → q gene indices.
   d. Reveal those q (read withheld + noise) → add to revealed set.
   e. ObjectiveObject metric (recall@k) on current revealed set → log round record.
4. Repeat for each method (AL, baseline) × N seeds.
5. Runner writes the run directory: `config.yaml`, `rounds.parquet`
   (method, seed, round, selected, revealed_val, metric), `manifest.json`.

### Report
Reads run directory → aggregate over seeds (mean ± CI of recall@k) → HTML.

**Invariant:** AL and baseline see the same cell line, same initial set, and
paired noise draws (common random numbers). Only selection differs — a fair race.

## 5. Error Handling

- Config validated at build time (Hydra structured configs / dataclasses) — fail
  fast on bad impl target or missing param.
- Embedding precompute: genes missing an embedding are logged and dropped from
  the candidate set, recorded in the manifest (not silent).
- Surrogate fit failure (e.g. GPR non-convergence) is caught and logged per round;
  run is marked degraded, not crashed.

## 6. Testing

- Each ABC impl unit-tested with placeholder embedding + tiny synthetic effect
  matrix (no heavy deps).
- End-to-end smoke test: placeholder embedding → 2 rounds → asserts run dir
  exists and metric behaves sanely. Fast, CI-able.
- Determinism test: same seed + config → identical `rounds.parquet`.

## 7. Repository Layout

```
src/geneal/
  data/        depmap adapter, dataset, embedding adapters
  models/      surrogate (gpr, bnn), acquisition, selection, noise
  experiment/  data_object, objective, design, experiment builder
  runner/      runner, logging
  report/      html report, latex tables, plots
  metrics/     recall@k etc (pluggable)
conf/          hydra configs (experiment, model, data, ...)
scripts/       download_depmap.py, precompute_embeddings.py
tests/
data/          raw/ processed/  (gitignored)
CLAUDE.md
```

## 8. v1 Deliverables

- DepMap CRISPRGeneEffect downloaded + curated into `data/processed/`.
- scPRINT (or ESM2 fallback) gene embeddings precomputed + cached.
- Surrogates: GPR (Matérn) + BNN.
- Acquisition: EI / UCB / EIG.
- Selection: top-q greedy + greedy+fantasies.
- NoiseModel: fixed Gaussian.
- Metric: recall@k (pluggable).
- Runner: AL vs greedy/random baseline over N seeds, structured logs, manifest.
- HTML report with recall@k curves, AL-vs-baseline plots, dropdown latex tables.
- `CLAUDE.md` (light on exact module names — to be iterated).
- Tests + end-to-end smoke test.
- Fresh `geneal` micromamba env (user creates from provided package list:
  adds gpytorch + botorch for GPR/q-acquisition, scPRINT/fair-esm for embeddings,
  on top of torch/pyro/sklearn/hydra/jinja2/plotly/pandas).

## 9. Out of Scope (v1, but designed-for)

- Cell-line-as-feature surrogate (cross-cell-line transfer).
- RL framework extension.
- Heteroscedastic / DepMap-derived noise.
- Sophisticated batch acquisition beyond fantasies.
- On-the-fly embedding generation.
