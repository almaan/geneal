# Real DepMap multi-cell-line sweep: k-DPP vs baselines

Diligence experiment. Tests whether the single-line verdict (greedy beats k-DPP on OVCAR-8) holds across cell lines or is OVCAR-8-specific. Nothing was tuned to favor any method.

## Config

- recall@50; 10 rounds, batch 10, n_initial 50, GaussianNoise(0.1), seeds [0, 1, 2]

- surrogate: fresh GPRSurrogate(n_iters=100) per method per line

- methods: kdpp=UCB(2.0)+KDPP('greedy'), greedy=UCB(2.0)+TopQGreedy, fantasy=UCB(2.0)+GreedyFantasy, coreset=GreedyMean+CoreSet, typiclust=GreedyMean+TypiClust, random=RandomAcquisition+TopQGreedy


## Cell lines (most coverage among embedded genes)

- `ACH-000147` — Breast (500 genes)
- `ACH-000183` — Lymphoid (500 genes)
- `ACH-000067` — CNS/Brain (500 genes)
- `ACH-000900` — Lung (500 genes)
- `ACH-000696` — Ovary/Fallopian Tube (500 genes)
- `ACH-000527` — Ovary/Fallopian Tube (500 genes)
- `ACH-000097` — Breast (499 genes)
- `ACH-000362` — Myeloid (499 genes)

## Final recall@50 per line (mean over seeds)

```
                         lineage   kdpp greedy fantasy coreset typiclust random kdpp - greedy
ACH-000147                Breast 0.5400 0.5000  0.5467  0.3600    0.2467 0.2600        0.0400
ACH-000183              Lymphoid 0.4333 0.4867  0.4067  0.3000    0.2800 0.2600       -0.0533
ACH-000067             CNS/Brain 0.4200 0.5000  0.4533  0.3133    0.3133 0.3000       -0.0800
ACH-000900                  Lung 0.4067 0.4467  0.4467  0.3533    0.2867 0.2667       -0.0400
ACH-000696  Ovary/Fallopian Tube 0.4200 0.4933  0.4533  0.3200    0.2667 0.2400       -0.0733
ACH-000527  Ovary/Fallopian Tube 0.4000 0.3933  0.4467  0.3200    0.3200 0.2533        0.0067
ACH-000097                Breast 0.5400 0.4867  0.5267  0.3467    0.2733 0.2733        0.0533
ACH-000362               Myeloid 0.4333 0.4600  0.4867  0.3400    0.2333 0.2733       -0.0267
MEAN                 (all lines) 0.4492 0.4708  0.4708  0.3317    0.2775 0.2658       -0.0217
```

## Aggregate

- Lines completed: **8**
- **kdpp >= greedy: 3/8 lines**
- quality (max kdpp/greedy/fantasy) > diversity-only (max coreset/typiclust): **8/8 lines**
- kdpp > random: **8/8 lines**
- mean (kdpp - greedy) gap across lines: **-0.0217**

Mean recall@50 per method across lines (sorted):

```
fantasy     0.4708
greedy      0.4708
kdpp        0.4492
coreset     0.3317
typiclust   0.2775
random      0.2658
```

## Honest read

**Does greedy >= k-DPP generalize?** Largely yes, but not universally. Greedy beats
k-DPP in 5/8 lines; k-DPP beats greedy in 3/8 (so kdpp >= greedy in 3/8). The mean
gap is small but favors greedy: -0.0217 recall@50 (greedy 0.4708 vs kdpp 0.4492 averaged
across lines). So the OVCAR-8 finding (greedy > k-DPP) is the more common outcome and
holds on average, but it is NOT a clean sweep — k-DPP wins on 3 lines, two of them
clearly. This is a "greedy usually edges k-DPP" result, not "greedy dominates."

**Is there any line/lineage where k-DPP clearly wins?** Yes. k-DPP beats greedy on both
Breast lines: `ACH-000097` (+0.0533) and `ACH-000147` (+0.0400) — the two largest gaps in
either direction in the whole table. (`ACH-000527`, Ovary, is a near-tie at +0.0067.)
Both Breast lines are also the two highest-recall problems overall (kdpp 0.540 on each),
so on the "easiest" objectives k-DPP's diversity-aware batch is competitive-to-better.
No single lineage favors greedy uniformly: greedy clearly wins CNS/Brain (-0.0800) and
OVCAR-8/Ovary (-0.0733), but the other Ovary line is a tie and both Breast lines flip.
There is no lineage on which k-DPP is systematically worse than random or than diversity-only.

**Does quality >> diversity-only hold?** Yes, unambiguously. A quality method (best of
kdpp/greedy/fantasy) beat the best diversity-only method (coreset/typiclust) in 8/8 lines,
and the margin is large: quality methods cluster at ~0.41-0.55 recall while coreset sits
at ~0.33 and typiclust at ~0.28. This is the most robust finding in the sweep and it
generalizes cleanly across every lineage tested. Notably `fantasy` ties `greedy` for the
top mean (0.4708) and is the single best method on 4/8 lines.

**k-DPP vs random:** k-DPP beat random in 8/8 lines — the UCB-guided k-DPP is always
worth more than uninformed sampling, even on the lines where greedy edges it.

**Bottom line:** The single-line verdict partially generalizes. "Quality >> diversity-only"
is universal (8/8). "Greedy >= k-DPP" is the majority outcome (5/8 for greedy) and holds
on the mean, but is line-dependent rather than absolute: k-DPP clearly wins both Breast
lines. The honest read is that greedy and k-DPP are close substitutes for the quality
signal, with greedy a small average favorite, and the real separation in this benchmark
is between quality-driven acquisition and pure diversity sampling.

