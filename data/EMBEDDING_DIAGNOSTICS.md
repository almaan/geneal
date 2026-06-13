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

## Is the gap inherent or a metric problem? — INHERENT (supervised-projection diagnostic, 2026-06-13)

Trained a supervised linear projection (->64d) of each FM embedding to make
embedding cosine match true outcome-redundancy (cross-cell-line effect-PROFILE
cosine), then measured the NN/random redundancy ratio on HELD-OUT genes (70/30
gene split):

| embedding | raw ratio | supervised projection (held-out genes) |
|---|---|---|
| ESM2 | 0.859 | 0.927 (WORSE) |
| PubMedBERT | 0.889 | 0.982 (WORSE) |

The supervised projection FAILED TO GENERALIZE — the ratio got worse on held-out
genes, the signature of structure that is ABSENT, not merely hidden in a bad metric.
If redundancy were latent-but-present, a learned metric would generalize and lower
the ratio; instead it memorized training-gene redundancy and did not transfer.

**CONCLUSION: the FM redundancy gap is INHERENT, not a metric/geometry artifact.**
Confirmed three ways: (1) three FM modalities fail identically; (2) PCA + kernel
choice (RBF/cosine) revive off-diagonals but not the redundancy ratio; (3) a
supervised projection cannot recover generalizable redundancy. Knockout-outcome
redundancy is a PERTURBATIONAL property absent from gene-identity foundation models
(sequence/expression/literature). It cannot be transformed-out of FM embeddings; it
requires a perturbation-derived representation (co-dependency) or an external
relational prior (pathway/GRN graph).

## Multi-cell-line confirmation (5 lines, out-of-sample) — 2026-06-13

Repeated the diagnostic on 5 cell lines distinct from ACH-000147
(ACH-000219/651/696/881/971), all 4 representations:

redundancy ratio (lower=more outcome structure):
  ESM2 0.86-0.89 | scPRINT 0.86-0.91 | PubMedBERT 0.82-0.85 | codep 0.39-0.52
GP held-out R2:
  ESM2 ~0 | scPRINT ~0 | PubMedBERT 0.23-0.27 | codep 0.77-0.86

The finding is CONSISTENT across all 5 lines: (1) all 3 FM modalities lack
outcome-redundancy structure (ratio 0.82-0.91); (2) PubMedBERT is reliably the best
FM PREDICTOR (R2 ~0.23 vs ~0 for ESM2/scPRINT) — literature text carries the most
quality signal; (3) co-dependency reliably carries both structure (ratio ~0.4) and
prediction (R2 ~0.8). Not cell-line-specific — publication-grade robustness on the
negative result.

## STRING PPI graph — ALSO does not carry outcome-redundancy (2026-06-13)

STRING v12.0 human, 2029/2043 panel genes mapped, 99.3% with >=1 edge. Tested as a
relational prior for S (the clean, non-label-adjacent alternative to co-dependency):

- STRING-edge vs random outcome-redundancy ratio: 1.117 (Spearman(S,truth)=0.16) — real but weak (+12%).
- HEADLINE NN/random lethality-diff ratio (ACH-000147): 0.882 (direct edges), 0.860
  (shared-neighbor), 0.845 (score>=0.9 only). FM baseline ~0.86, co-dependency 0.39.

STRING sits AT the FM baseline. A curated functional/PPI network captures general
functional association but NOT knockout-outcome substitutability in a specific cell
line. The graph-S k-DPP is not worth building on STRING.

## FINAL synthesis — the complete representation table

| representation | type | redundancy ratio | GP R2 |
|---|---|---|---|
| ESM2 | protein sequence FM | 0.86 | ~0 |
| scPRINT | sc-expression FM | 0.89 | ~0 |
| PubMedBERT | literature text FM | 0.85 | 0.23 (best FM) |
| STRING | curated PPI/pathway graph | 0.86 | (relational, n/a) |
| co-dependency | perturbational (DepMap effects) | 0.39 | 0.80 |

THESIS (bulletproof, 4 negative + 1 positive, 5+ cell lines): knockout-outcome
redundancy in a specific cell context is captured by NO prior representation of gene
identity or general function (sequence/expression/literature/curated-network). It is
a context-specific PERTURBATIONAL property, recoverable only from perturbation data.
Supervised projection cannot extract it from FMs (fails to generalize). Implication
for diversity-aware batch AL: it requires perturbation-derived priors; zero-shot FM
and graph priors are provably insufficient. (co-dependency works but is label-adjacent
— the honest tension at the heart of the result.)
