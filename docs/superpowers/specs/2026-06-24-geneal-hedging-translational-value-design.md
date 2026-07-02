# Hedging Portfolio-Risk Analysis - Design

**Date:** 2026-06-24
**Status:** design approved in brainstorming; ready for implementation plan
**Target code:** the ablation harness - `scripts/run_ablation.py`,
`src/geneal/runner/ablation.py`, `src/geneal/report/ablation_report.py`.
Part 1 / Part 2 = the report's **Section A / Section B**.

## Problem

We have shown the mechanism hedge (quality-weighted k-DPP over an external
mechanism graph) lowers pathway concentration. We have NOT shown it is *useful* in
a way that resists the obvious criticism: the existing risk evidence
(`dropout_robustness`, dropout curve) is computed on the same CORUM structure the
hedge operates on, so it is close to self-referential.

This design adds a principled, portfolio-theory risk readout of a nominated
candidate set, reports it on the hedge's own graph (CORUM) AND an independent
graph (STRING), and adds a second nomination-quality variant - all inside the
existing Section B, with negligible extra compute (everything happens at the final
nomination stage).

## Reframing: selectivity across populations (naming only)

Exposition switches from "toxicity / safety" to selectivity across populations:

- **target population efficacy** (maximize) = lethality in the target line.
  Code today: `effb` (`-(gene_effect[target_line])`).
- **non-target population efficacy** (minimize) = lethality across the non-target
  population (population mean / contrast reference). Code today: `toxb`.
  Bold the "non": target vs **non**-target population.

The objective pair is handled by EHVI / hypervolume (Section A), never by scalar
subtraction at the objective level. The diversity operator (Section B) does need a
scalar quality `q` to temper the k-DPP - see "Nomination quality variants" below.
This is purely a relabel in prose, report copy, and column glosses; the underlying
arrays (`effb`, `toxb`) are unchanged.

## What already exists (do not rebuild)

Section B already computes, per `(base, operator, tox_source, line, seed)`:
- `concentration` (max single-complex fraction, CORUM), `robustness`
  (`dropout_robustness`, CORUM), `n_pathways` (CORUM), dropout curve `drop_i`
  (CORUM pathway-failure simulation)
- `mean/max efficacy`, `mean toxicity`, `hypervolume` (true (eff, -tox) plane)
- STRING pairwise spread (`_string_spread`: mean/max pairwise STRING similarity)
- pareto recall metrics

Operators in `DIVERSITY_OPS`: `none`, `kdpp_corum`, `kdpp_string` (k-DPP with
external CORUM / STRING similarity S). Bases in `B_BASES`: the safety-aware
acquisitions (greedy / truncation / EHVI). The k-DPP quality at nomination is
currently `q = predicted efficacy` (UCB/mean), with non-target lethality applied
only as a safety filter (`nominate()` in `ablation.py`).

## New work (3 additions to Section B)

### Addition 1 - Portfolio-risk metrics R and N_eff, on CORUM and STRING

Two scalar risk readouts of a nominated set, each computed on BOTH the CORUM and
the STRING similarity matrix (both already built per run as `S_by_src["corum"]`,
`S_by_src["string"]`, unit-diagonal, off-diagonal in [0,1]):

- **R (portfolio variance)** = `w^T S w` with equal weights `w = 1/K`:
  `R = 1/K + (1/K^2) * sum_{i!=j in pick} S_ij`.
  Markowitz equal-weight portfolio variance: if each gene carries some
  adverse-event risk correlated by S, R is the variance of portfolio loss. Diverse
  set -> off-diagonals ~0 -> R approx 1/K (floor, all-idiosyncratic); concentrated
  set -> large off-diagonals -> high R. Lower = better hedged.
- **N_eff (effective number of independent bets)** = `K^2 / (1^T S 1)` over the
  selected submatrix; range [1, K]. Interpretable companion ("none = 3 independent
  bets, kdpp = 18").

Both S matrices are already normalized to unit diagonal / [0,1] off-diagonal (the
existing `build_corum_S` returns Jaccard in [0,1] with unit diagonal;
`build_embedding_S` similarly; STRING sim is [0,1]). So R is comparable across the
two graphs without extra scaling.

Reported columns: `risk_corum`, `neff_corum`, `risk_string`, `neff_string`.

Note on the old discrete metrics: `concentration` / `robustness` / dropout curve
stay on CORUM (their natural discrete-complex home), and STRING already has
`_string_spread` (mean/max pairwise similarity). R and N_eff are the new
both-graph, threshold-free risk metrics; they do not require inventing a STRING
"module" partition. (If we later want concentration/robustness on STRING too, that
needs a STRING-graph community partition with an arbitrary threshold - deferred,
out of scope unless requested.)

### Addition 2 - Second nomination-quality result: q = efficacy vs q = efficacy - toxicity

Today the k-DPP quality is `q = predicted efficacy` (target population). Add a
SECOND result (not a second run) using `q = predicted efficacy - predicted
toxicity` (= predicted selectivity = target_pop - non-target_pop, the old
`sel_score`). Both reuse the SAME final fitted predictions (the joint GP, or the
efficacy+toxicity GPs already fit for the safety filter), so the only extra cost is
one more `select_idx` pass at the final stage - negligible.

Mechanism: `nominate()` gains a `quality` parameter, `"eff"` (current default) or
`"sel"`. When `"sel"`, `q = m_eff - m_tox` (predicted), with the same eligibility
pool/safety filter applied. The Section B loop calls `nominate(...)` once per
`quality` value and tags rows with a `quality` column (`"eff"` / `"sel"`).

This isolates the effect of feeding selectivity (not just efficacy) into the
diversity operator's quality, separately from the base acquisition.

### Addition 3 - Diversity barplot (per-complex gene counts)

For the headline Section B configs, emit the per-complex gene-count distribution of
each method's K picks so the report can render a barplot: greedy/none spikes into a
few complexes; k-DPP is flat. Stored as a tidy `bar_rows` -> `group_counts.parquet`
(columns: `base, operator, quality, tox_source, cell_line, seed, complex_id,
count`), restricted to a small set of headline configs (e.g. base=ehvi, operators
none vs kdpp_corum, both quality variants) to keep the file small.

## Part 1 (Section A) - add AUC-hypervolume over rounds

Section A (safety/efficacy bases incl. EHVI) needs NO external sources. The prose /
report copy adopts the target/non-target population naming. EHVI tops hypervolume
by construction (it optimizes expected HV; we evaluate realized HV) - the expected
upper bound, stated honestly, not the contribution.

One NEW metric: the area under the hypervolume-over-rounds curve (AUC-HV), the
EHVI-vs-baselines headline (how fast each acquisition accumulates hypervolume in
the (target population, -non-target population) plane). Today only a single
final-nominee HV exists; `round_rows` carries no per-round HV.

- Add per-round hypervolume to `round_rows`, in BOTH senses (reuse `hypervolume2d`):
  - `assayed_hypervolume` = HV of the revealed/assayed Pareto front at round r
    (the loop's exploration of the frontier - standard multi-objective BO metric).
  - `nom_hypervolume` = HV of the round-r NOMINATED set (decision quality per round).
- AUC-HV = normalized area under HV-vs-round (trapezoid / n_rounds), per
  `(method, line, seed)`, computed in the report from `rounds.parquet`.

Reporting:
- **Main Section A table:** keep the candidate-set HV (the existing final-nominee
  `hypervolume`).
- **Separate table:** AUC-HV, both senses - (1) revealed-front `assayed_hypervolume`
  AUC and (2) nominated-set `nom_hypervolume` AUC - per method, mean +/- 95% CI over
  lines and seeds. Plus a HV-over-rounds curve figure.

## Part 2 (Section B) - the contribution

Headline claim: the k-DPP hedge (selecting on CORUM) lowers R and raises N_eff vs
`none` not only on CORUM (its own graph) but ALSO on STRING (held-out, independent)
- so the hedge generalizes across mechanism graphs rather than gaming the metric it
optimizes. If STRING does not move, report it plainly (the hedge is graph-specific).
Secondary: the `q=sel` variant vs `q=eff` shows whether selectivity-aware quality
changes the efficacy/risk trade-off.

Figures (added to `ablation_report.py` Section B):
1. **Diversity barplot** - per-complex gene counts, none vs k-DPP (and the two
   quality variants), from `group_counts.parquet`.
2. **Risk table/plot** - R and N_eff per `(base, operator, quality)` x graph
   {CORUM, STRING}, mean +/- 95% CI over lines and seeds.

## Components

Reuse, untouched:
- `src/geneal/models/multiobjective.py` (`hypervolume2d`, `mc_ehvi`, `pareto_*`)
- `src/geneal/models/multitask.py::MultiTaskGPR` (joint GP, already wired via
  `--joint-gp`)
- `src/geneal/runner/ablation.py`: `build_corum_S`, `build_embedding_S`,
  `run_acquisition`, the STRING similarity builder, `_string_spread`, `dropout_curve`
- `src/geneal/metrics/portfolio.py`: `pathway_concentration`, `dropout_robustness`,
  `n_pathways_covered`

New pure functions (TDD), in `src/geneal/metrics/portfolio.py`:
- `portfolio_risk(picks, S) -> float`  (= w^T S w, equal weights)
- `effective_bets(picks, S) -> float`  (= K^2 / 1^T S 1)

Modified:
- `ablation.py::nominate(...)` - add `quality="eff"|"sel"` (default `"eff"`).
- `ablation.py::evaluate(...)` - add optional `risk_S: dict[str, ndarray]`
  (e.g. `{"corum": S_corum, "string": S_string}`); when given, add
  `risk_<src>` / `neff_<src>` columns.
- `run_ablation.py` Section A loop - add `assayed_hypervolume` and `nom_hypervolume`
  to `round_rows` (per-round HV via `hypervolume2d`).
- `run_ablation.py` Section B loop - call `nominate` for each `quality`; pass
  `risk_S={"corum":..., "string":...}` to `evaluate`; collect `group_counts` for
  headline configs; write `group_counts.parquet`; add `quality` to the row dict and
  the console summary.
- `ablation_report.py` - add the AUC-HV separate table (revealed + nominated) + HV-
  over-rounds curve (Section A); render the diversity barplot + R/N_eff table
  (Section B); adopt target/non-target naming in Section A/B copy.

## Data (all on disk, verified)

- `data/processed/depmap/gene_effect.parquet`
- `data/processed/depmap/corum_sim*.parquet`, CORUM membership cache
- `data/processed/depmap/string_sim.parquet` (2043x2043, [0,1])
- embeddings under `data/processed/embeddings/`

No external downloads. OpenTargets / DGIdb / druggability were considered and
rejected: the useful question (does pathway concentration carry correlated risk on
an independent structure) is answered directly by the CORUM-vs-STRING
generalization of R and N_eff, without a heavy fetch or a held-out-validation
framing the data cannot cleanly support.

## Out of scope

- The simulated correlated-pathway attrition / selective-hit oracle (redundant with
  `robustness`; not genuinely held-out - DepMap is the training signal).
- Herfindahl index (duplicates the concentration story).
- Cross-cell-line transfer (lethal genes are lethal in most lines - little
  cross-line selectivity signal).
- STRING-module concentration/robustness (needs an arbitrary community threshold;
  R/N_eff cover the both-graph risk continuously instead).
- Changing the AL-loop acquisitions (Section A already runs EHVI).

## Honest caveats

- R and N_eff are continuous, portfolio-theory generalizations of the existing
  `concentration` (max single-group fraction) - same family, but defined on the
  continuous S and tied to diversification math.
- EHVI tops hypervolume by construction; that is the expected upper bound, not the
  contribution. The contribution is the risk reduction (R, N_eff) of k-DPP and its
  generalization from CORUM to STRING.
- CORUM-hedged k-DPP only helps STRING-measured risk if CORUM and STRING share
  structure; the CORUM-vs-STRING comparison is exactly that test, reported either
  way (including a null result).
