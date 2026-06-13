# geneal — metric-panel results (Plan 3)

Single-objective sweep: 4 cell lines (ACH-000147 Breast, ACH-000183 Lymphoid,
ACH-000067 CNS/Brain, ACH-000900 Lung), 2 seeds, 500 genes (ESM2-35M, dim 480),
k=50, 10 rounds, batch 10, n_initial 50, α=0.5, 20 nugget clusters.

## Single-objective — final-round mean ± 95% CI (across 4 lines × 2 seeds)

**recall@50** (true top-50 lethal recovered)
```
greedy     0.4375 ± 0.0409   ← best
fantasy    0.3950 ± 0.0740
kdpp       0.3700 ± 0.0687
typiclust  0.2850 ± 0.0231
coreset    0.2775 ± 0.0300
random     0.2700 ± 0.0181
```

**α-NDCG@50** (quality-weighted, redundancy-discounted coverage of lethal clusters)
```
coreset    0.2261 ± 0.1087   ← best (diversity-only methods win)
random     0.2047 ± 0.0844
typiclust  0.2014 ± 0.0911
kdpp       0.1736 ± 0.0785
fantasy    0.1714 ± 0.0772
greedy     0.1565 ± 0.0623
```

**diversity** (mean pairwise embedding distance of cumulative revealed set)
```
coreset    3.0063 ± 0.0146   ← best
random     2.7187 ± 0.0099
greedy     2.7156 ± 0.0450
typiclust  2.6497 ± 0.0002
fantasy    2.6331 ± 0.0968
kdpp       2.6085 ± 0.0821   ← LEAST diverse
```

**max_value** (best lethality found / global max)
```
greedy     0.9367 ± 0.0814   ← best
kdpp       0.9174 ± 0.1110
fantasy    0.9174 ± 0.1110
typiclust  0.7373 ± 0.0603
random     0.7187 ± 0.0612
coreset    0.7137 ± 0.1340
```

## Honest read (single-objective)

There is a clean **quality–coverage tradeoff** and **k-DPP does not bridge it**:
- Quality methods (greedy/kdpp/fantasy) win recall@50 and max_value (find the
  lethal genes); diversity-only methods (coreset/typiclust/random) win α-NDCG and
  diversity (cover distinct clusters).
- **k-DPP is worst-of-both:** it loses to greedy on recall (0.37 vs 0.44) and
  max_value, AND is the LEAST diverse method (2.61, below greedy) so it loses to
  coreset on α-NDCG and diversity too. On real DepMap the quality term q_i² in the
  k-DPP kernel dominates the determinant (the latent GP covariance over 500 real
  genes is near-uniform, so the diversity term barely fires) — k-DPP collapses
  toward a tighter-than-greedy exploitation. Answer to "does k-DPP win on α-NDCG
  or diversity where it ties on recall": NO — it is below greedy on diversity and
  below the diversity-only methods on α-NDCG.

The robust, defensible finding is NOT "k-DPP wins" but the **characterization**:
quality-driven AL with frozen FM embeddings recovers lethal knockouts (recall,
max_value) far better than diversity-based or random selection; diversity-based
selection conversely gives broader cluster coverage (α-NDCG, diversity) at the cost
of finding fewer truly-lethal genes. Batch outcome-diversity (k-DPP) does not help
either objective on this data.

## Dual / selectivity result

Differential target = lethality_A − lethality_B. Line A (efficacy) = ACH-000147
(Breast), line B (toxicity) = ACH-000183 (Lymphoid). 500 genes, 2 seeds, k=50,
10 rounds, batch 10.

**recall@50 (differential top-50)**
```
kdpp       0.3400 ± 0.0784   ← tied best
fantasy    0.3400 ± 0.0784   ← tied best
typiclust  0.3200 ± 0.0392
random     0.2600 ± 0.0784
coreset    0.2400 ± 0.0784
greedy     0.2200 ± 0.0392   ← WORST (collapses)
```

**max_value**
```
kdpp       1.0000 ± 0.0000   ← perfect, tied
fantasy    1.0000 ± 0.0000   ← perfect, tied
typiclust  0.9191 ± 0.1585
greedy     0.8535 ± 0.2871
random     0.7736 ± 0.4438
coreset    0.6070 ± 0.1959
```

**α-NDCG@50**: coreset 0.292 ≈ greedy 0.280 > random 0.254 > fantasy 0.237 ≈ kdpp 0.233 (close, wide CIs).
**diversity**: coreset 3.006 > others ~2.55–2.72.

## Honest read (dual / selectivity) — THE STORY FLIPS

On the harder **selectivity** objective, the single-objective ranking REVERSES:
greedy is the WORST method on recall (0.22) and mid on max_value, while
**k-DPP and fantasy win recall (0.34) and hit perfect max_value (1.0)**. The
differential target (two subtracted noisy lethality signals) is noisier and less
peaked than raw lethality, so pure greedy exploitation gets stuck and the
exploration/diversity in k-DPP/fantasy pays off. Answer to "does the dual
objective change the method ranking": YES, decisively — quality+diversity methods
beat greedy here, the opposite of the single-objective case.

**Defensible thesis (better than the original):** batch diversity does not help
for simple extreme recovery (greedy wins, single-objective), but becomes valuable
for the harder selectivity / efficacy-toxicity objective, where greedy
exploitation fails and quality+diversity acquisition wins.

**CAVEATS (do not overclaim):** 1 cell-line pair, 2 seeds, wide CIs — kdpp
0.34±0.078 vs greedy 0.22±0.039 (CIs nearly touch). Suggestive, NOT established.
k-DPP ties `fantasy` (the cheap quality+diversity proxy) on every metric, so
"k-DPP specifically" is not singled out over simpler diversity-aware quality
methods. Needs multiple line pairs + more seeds to confirm before it can anchor
a paper claim.

