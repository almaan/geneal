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

## scPRINT (expression-based FM) — does NOT fix it either (2026-06-13)

scPRINT zero-shot gene embedding (medium-v1.5, freeze_embeddings=False so genuinely
learned, 256-dim), mapped Entrez->Ensembl (2043/2043), same HVG panel, ACH-000147:

| diagnostic | ESM2-650M | scPRINT |
|---|---|---|
| GP held-out R² | 0.095 | -0.004 |
| Ridge held-out R² | 0.060 | **0.146** |
| NN/random redundancy ratio | 0.890 | **0.889** |
| k-DPP kernel off-diag \|S\| | 0.0000 | 0.0000 |

**Conclusion — strong general negative result.** TWO foundation-model gene
embeddings of completely different modalities — protein-sequence (ESM2) and
single-cell-expression (scPRINT) — BOTH fail to encode knockout-outcome-correlation
structure: embedding-near genes are no more similar in lethality than random genes
(ratio ~0.89 for both), and the k-DPP diversity kernel is perfectly diagonal (|S|=0)
for both. Batch-diversity acquisition collapses to greedy regardless of FM embedding.
Note scPRINT has MORE linearly-decodable lethality signal (Ridge R² 0.146 > ESM2
0.060) but its GP R² is ~0 (kernel/scale mismatch in the 256-dim learned space) and
it still provides no pairwise outcome structure.

The bottleneck is FUNDAMENTAL to FM-embedding-based diversity AL for knockout
discovery, not a representation-choice detail. The fix is a structured similarity
source that is NOT an FM embedding: a GRN/pathway graph kernel, or DepMap's own
gene co-dependency (SVD of the gene×cell-line effect matrix). FM embeddings encode
what their pretraining saw (sequence family / expression context), not which gene
knockouts have correlated lethality.

## PubMedBERT (biomedical literature text) — best FM predictor, STILL no diversity structure (2026-06-13)

NeuML/pubmedbert-base-embeddings on per-gene NCBI summary text (symbol+name+summary;
2029/2043 have real NCBI summaries), 768-dim, same HVG panel, ACH-000147:

| diagnostic | ESM2-650M | scPRINT | PubMedBERT | co-dependency |
|---|---|---|---|---|
| GP held-out R² | 0.095 | -0.004 | **0.195** | 0.791 |
| redundancy ratio | 0.890 | 0.889 | 0.854 | 0.386 |
| k-DPP kernel off-diag \|S\| | 0.0000 | 0.0000 | **0.0000** | 0.0575 |

**PubMedBERT is the best PREDICTIVE FM embedding** (GP R² 0.195, ~2x ESM2) — literature
text about gene function carries more lethality signal than sequence or expression.
(Its Ridge R² is very negative: 768-dim text embeddings overfit OLS; GP is the valid read.)

**But it STILL has no outcome-redundancy structure** (ratio 0.854, k-DPP kernel |S|=0).
THREE foundation-model embeddings across THREE modalities — protein-sequence (ESM2),
single-cell-expression (scPRINT), biomedical-literature (PubMedBERT) — ALL give an inert
(|S|≈0) diversity kernel. The negative result is modality-general: no FM embedding encodes
knockout-outcome CORRELATION, only a perturbation-derived representation (co-dependency) does.
Note quality vs diversity are separable: PubMedBERT helps the QUALITY term (best R²) while
being useless for the DIVERSITY term.
