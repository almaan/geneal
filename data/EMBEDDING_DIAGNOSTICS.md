# Embedding diagnostics — before/after ESM2-650M + HVG panel (ACH-000147 Breast)

| diagnostic | OLD: ESM2-35M, first-500 panel | NEW: ESM2-650M, HVG panel | verdict |
|---|---|---|---|
| genes | 500 | 2015 | — |
| embedding dim | 480 | 1280 | — |
| strongly-lethal fraction (lethality>1) | 0.026 | **0.431** | ✅ panel fix worked |
| GP held-out R² | 0.116 | **0.095** | ❌ no improvement (slightly worse) |
| GP held-out corr | 0.389 | 0.328 | ❌ |
| Ridge held-out R² | 0.092 | 0.060 | ❌ |
| NN/random lethality-diff ratio | 0.953 | 0.890 | ~no structure either way |
| k-DPP kernel off-diag \|S\| | 0.025 | **0.0000** | ❌ even more diagonal |

## Honest read

**HVG panel fix: large success.** 43% of the panel is strongly lethal somewhere
vs 2.6% for first-500-by-ID. The top-k recovery task is now meaningful (not a
needle-in-haystack), so methods can differentiate and absolute recall should rise.

**ESM2-650M did NOT fix the embedding bottleneck — and this is conclusive.**
Going from 35M→650M (and 480→1280 dim, and 4× more genes) did NOT improve
predictive R² (0.095 vs 0.116, slightly worse) and the outcome-redundancy ratio
stayed ~0.89 (essentially no structure). The k-DPP diversity kernel is now
PERFECTLY diagonal (|S|=0.0000). Scaling the protein language model does not help
because it is the WRONG REPRESENTATION, not too small: ESM2 encodes protein
structure/family (its training objective), not cancer-context functional
essentiality, and encodes knockout-effect CORRELATION not at all.

**Decision (pre-registered: "scPRINT if neither works"):** neither bigger ESM2
nor (on the embedding axis) the panel change rescued predictive signal or outcome
structure. scPRINT (single-cell-expression-trained, encodes functional /
co-expression structure) is the indicated next representation. The HVG panel is
kept regardless — it is an unambiguous improvement.
