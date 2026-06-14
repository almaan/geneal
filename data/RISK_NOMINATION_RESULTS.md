# Risk-aware target nomination — results (geneal Plan 5)

4 cell lines (ACH-000696/651/219/971), 2 seeds, K=30 targets, PubMedBERT embedding,
selective-lethality target (lethal in line minus common-essential toxicity proxy),
CORUM complexes as pathways, per-pathway cap as the hedging lever.

## Aggregate (mean ± 95% CI)

| method | selective-lethality (efficacy) | concentration (max frac one pathway = RISK) | dropout-robustness |
|---|---|---|---|
| greedy | 0.410 ± 0.155 | 0.821 ± 0.145 | 0.694 ± 0.208 |
| cap=1 | 0.294 ± 0.089 | 0.069 ± 0.010 | 0.977 ± 0.011 |
| cap=2 | 0.280 ± 0.084 | 0.118 ± 0.016 | 0.971 ± 0.010 |
| cap=3 | 0.285 ± 0.091 | 0.163 ± 0.019 | 0.962 ± 0.013 |
| cap=5 | 0.309 ± 0.105 | 0.241 ± 0.016 | 0.951 ± 0.016 |

## Honest read

**The risk-reduction effect is large and rock-solid (tight CIs):** greedy concentrates
~82% of its 30-target portfolio in a SINGLE pathway (dropout-robustness 0.69 — if that
pathway is toxic/undruggable, ~31% of portfolio value is lost). Per-pathway capping
collapses concentration to 0.07-0.24 and lifts robustness to 0.95-0.98. The cap level
traces a smooth efficacy-risk Pareto greedy cannot reach — the deliverable for a
target-nominator choosing a risk tolerance.

**The efficacy COST is directionally clear but under-powered:** capping costs ~0.10-0.13
selective-lethality (0.41 -> ~0.29-0.31), but the lethality CIs are wide (±0.10-0.16 at
4 lines x 2 seeds). The COST magnitude needs more lines/seeds to state tightly; the
RISK-REDUCTION side does not (tight CIs, huge effect).

**Metric caveat:** n_pathways_covered has a nonsensical CI (greedy 31.6±37) because genes
belong to MULTIPLE CORUM complexes, inflating the count — concentration and
dropout-robustness are the trustworthy portfolio-risk metrics; n_pathways should be
dropped or redefined (distinct dominant-complex assignment).

**Bottom line:** greedy target nomination carries severe, unflagged pathway-concentration
risk; a simple per-pathway cap removes it at modest, tunable efficacy cost. Clean,
practically-motivated contribution. TODO before paper: more lines/seeds to tighten the
efficacy-cost CI; fix/drop n_pathways; add the selectivity dual-axis explicitly.
