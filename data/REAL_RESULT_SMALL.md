# Real DepMap Result — Small-Scale Decisive Experiment

First real recall@k comparison of k-DPP vs baselines on actual DepMap CRISPR
lethality. This is the go/no-go signal for the MLCB paper framing.

## Setup

- **Cell line:** ACH-000696 — **OVCAR-8** (OVCAR8), lineage **Ovary/Fallopian Tube**,
  Ovarian Epithelial Tumor. Chosen automatically as the cell-line column with the
  fewest NaNs among the embedded genes (0 NaNs of 500).
- **#genes in experiment:** **500** (genes with both an ESM2 embedding and a
  non-NaN Chronos gene-effect for this cell line).
- **Embedding model:** **esm2_t12_35M_UR50D** (dim 480), mean-pooled residue
  embeddings, CPU. 500/500 mapped proteins embedded in ~15 s.
- **Gene→protein mapping:** UniProt REST id-mapping (GeneID → UniProtKB),
  prefer-reviewed dedup. **500 / 500** entrez ids mapped (499/500 reviewed/Swiss-Prot).
  NOTE: the first 500 GeneIDs expand to 1003 UniProtKB records (isoforms +
  unreviewed entries), so single-page (size=500) fetch covered only 254 distinct
  ids; added `next`-link pagination to the script to map all 500. Full-genome
  pagination/scale remains Task 4.
- **Target convention:** lethality = −(Chronos gene_effect); higher = more lethal.
- **AL design:** RecallAtK(k=50), maximize; n_rounds=10, batch=10, n_initial=50,
  GaussianNoise(sigma=0.1); seeds 0 1 2; GPRSurrogate(n_iters=100) for all methods.
- Run dir: `res/runs_real/run`.

## Final recall@50 per method (verbatim, mean over seeds 0 1 2)

```
method
greedy       0.493333
fantasy      0.453333
kdpp         0.420000
coreset      0.320000
typiclust    0.266667
random       0.240000
```

## Pareto summary (verbatim)

```
      method  mean_quality  mean_diversity  final_recall
0    coreset      0.121009        3.072379      0.320000
1    fantasy      0.212150        2.325596      0.453333
2     greedy      0.195846        2.361971      0.493333
3       kdpp      0.194647        2.383142      0.420000
4     random      0.063592        2.762177      0.240000
5  typiclust      0.068056        2.738514      0.266667
```

## Honest read (go/no-go signal)

(a) **Does k-DPP beat quality-only greedy on real data? No.** On OVCAR-8,
plain quality-only greedy (TopQGreedy + UCB) is the single best method at
recall@50 = 0.493, ahead of fantasy (0.453) and k-DPP (0.420). k-DPP's added
batch diversity (mean_diversity 2.383 vs greedy's 2.362) is marginal and does
not translate into better recovery of the most-lethal knockouts — if anything it
costs ~7 recall points versus greedy here. This mirrors the synthetic finding
that the k-DPP-vs-greedy advantage is fragile; on this real landscape it does not
appear. **This is a negative result for the "k-DPP beats greedy" framing and
should be reported as such, not tuned away.**

(b) **Do k-DPP and the quality methods beat diversity-only CoreSet/TypiClust?
Yes, clearly.** All three quality-driven methods (greedy 0.493, fantasy 0.453,
k-DPP 0.420) beat CoreSet (0.320) and TypiClust (0.267). Pure embedding-space
diversity without a quality signal is markedly worse — so a quality/surrogate
signal genuinely matters; diversity alone is not enough. This is a real, positive
differentiator the paper can stand on.

(c) **Does k-DPP beat random? Yes.** k-DPP (0.420) clearly beats random (0.240),
as do all quality methods. The acquisition framework works on real data; the open
question is only the k-DPP-over-greedy increment, which this experiment does not
support.

**Bottom line / go-no-go:** The decisive small-scale signal is that *quality-aware
active learning robustly beats diversity-only and random selection on real DepMap
lethality, but k-DPP does not beat simple quality-only greedy here.* The headline
MLCB framing should therefore center on "quality-driven AL (surrogate + UCB) vs
diversity-only / random baselines on real essentiality landscapes," NOT on a
k-DPP-beats-greedy moat. Whether the k-DPP advantage emerges with a larger model
(esm2_t33_650M, dim 1280), more genes, or other cell lines/lineages is the
question for Task 4 — but on this single OVCAR-8 / 500-gene / 35M-param-embedding
configuration, k-DPP ties-to-loses against greedy. Reported faithfully; no
parameters were tuned to favor k-DPP.

### Caveats (do not over-read a single configuration)

- One cell line, 500 genes, smallest practical ESM2 (35M). The ordering could
  shift with a stronger embedding, more genes, or a different lineage — Task 4.
- 3 seeds; no formal significance test on the ~0.07 greedy−kdpp gap. The
  qualitative split (quality > diversity-only > random) is large and stable; the
  greedy-vs-kdpp ordering is the part that needs more cell lines to firm up.
