# CORUM as a gene-gene similarity prior for geneal — validation

**Date:** 2026-06-13 · **Cell line:** ACH-000147 · **Panel:** 2043 HVG Entrez genes
**Verdict headline:** CORUM **does NOT** carry knockout-outcome-redundancy better than STRING on the decisive geneal metric (NN/random lethality-diff ratio **0.892** vs STRING 0.86 vs co-dependency 0.39 — *higher = worse*). **BUT** CORUM is a usable **mechanism-diversity** signal: it is far more concentrated than STRING (top-50 lethal genes collapse into two dominant complex-blocks of 17 and 16 genes rather than STRING's ~13 scattered components), and its cross-cell-line profile-cosine ratio is **1.206 > 1**, so co-complex members *are* mildly more outcome-correlated than random — just not enough to beat the random NN baseline.

---

## Step 1 — Download

- **Method:** The old `mips.helmholtz-muenchen.de/corum/download/releases/...` static paths are dead (site is now a single-page app; the legacy `.zip` URL returns a 423-byte HTML shell, and `mips.gsf.de` no longer resolves). The current CORUM exposes a **FastAPI backend**. Discovered the route from the JS bundle and OpenAPI spec:
  - `GET https://mips.helmholtz-muenchen.de/fastapi-corum/public/file/info` → lists files (`file_id`, `formats`)
  - `GET https://mips.helmholtz-muenchen.de/fastapi-corum/public/file/download_current_file?file_id=human&file_format=txt` → the data
  - Required `curl -k` (server's intermediate cert chain does not verify in this env; flagged for honesty — content integrity confirmed separately, see below).
- **Version:** CORUM **5.3** (release date 2026-04-14, from `/public/releases/current`).
- **File:** `humanComplexes.txt`, 6.27 MB, tab-separated. Parsed cleanly to **5628 distinct human complexes** (all `complex_id` numeric → no field-shift corruption; all 5628 rows have non-empty `subunits_gene_name`). The 7812 raw lines vs 5628 rows is from legitimate embedded newlines in free-text description columns, handled correctly by the CSV parser.
- Member genes taken from the `subunits_gene_name` column (semicolon-separated HGNC symbols), mapped to panel Entrez via the symbols parsed from DepMap's `'SYM (Entrez)'` labels.

## Step 2 — CORUM similarity over the panel

- `S_corum[i,j] = 1` iff genes i,j co-occur in ≥1 CORUM complex (binary), saved to **`data/processed/depmap/corum_sim.parquet`** (Entrez-indexed, panel order, 2043×2043).
- Weighted version (count of shared complexes) saved to **`data/processed/depmap/corum_sim_weighted.parquet`**.
- **Coverage:** **1333 / 2043** panel genes are in ≥1 CORUM complex = **65.2%**.
- **Density:** **18 305** co-complexed gene-pairs out of 2 085 903 possible = **0.00878** (0.88% of pairs). Sparse, as expected for curated physical complexes.

## Step 3 — Does CORUM carry outcome-redundancy?

Lethality `y = -gene_effect` in ACH-000147; 2015/2043 panel genes have a valid effect value.

### HEADLINE — NN/random lethality-diff ratio (lower = better; encodes substitutability)

For each gene: mean `|y_i - y_j|` to its CORUM co-complex partners vs to 10 random panel genes.

| neighbor source | NN/random ratio | note |
|---|---|---|
| FM embeddings (ESM2/scPRINT/PubMedBERT) | 0.86 | prior — no signal |
| STRING (direct / shared-neighbor) | 0.86–0.88 | ≈ FM |
| **CORUM 5.3 co-complex** | **0.892** | nn=0.7293, rand=0.8177 |
| co-dependency (DepMap-derived) | 0.39 | label-adjacent, strong |

- **1225** genes had ≥1 CORUM partner among the 2015 with lethality; **790 excluded** (no CORUM partner in panel).
- **Result: 0.892 — essentially the same null-zone as STRING (0.86) and the FM embeddings, and nowhere near co-dependency (0.39).** CORUM partners are *not* meaningfully more knockout-redundant in this single cell line than random genes. The signal is in the right direction (ratio < 1) but its magnitude is indistinguishable from STRING's.

### Profile-cosine across OTHER cell lines (ratio > 1 = captures redundancy)

Mean cosine of full lethality profiles (1207 cell lines, excluding ACH-000147) for CORUM co-complexed pairs vs random pairs:

- co-complex mean **0.8911**, random mean **0.7389**, **ratio = 1.206** (n = 18 303 pairs).
- **Positive signal:** co-complex members *are* more outcome-correlated than random across the cell-line panel. But note the random baseline is already high (0.74) because gene-effect profiles share strong global structure (essentiality), so a 1.21× lift is modest — consistent with the headline ratio sitting near the null. This is the same pattern STRING showed.

## Step 4 — Mechanism-density of top-50 lethal genes

Top-50 lethal genes in ACH-000147, grouped by CORUM co-membership (connected components):

- **41 / 50** are in ≥1 CORUM complex.
- **18 connected components**, with sizes **[17, 16, 2, 1, 1, 1, ...]** — i.e. **two large complex-blocks of 17 and 16 genes** dominate, then 15 singletons.
- **36 CORUM complexes contain ≥2 of the top-50** lethal genes.
- **Compare STRING:** STRING split the top-50 into **~13 components** with no comparably dominant block.
- **Takeaway:** CORUM is *more concentrated* — the two big blocks almost certainly capture the ribosome / core machinery driving pan-lethal essentiality. For a portfolio "don't pick many genes from one complex" rule, CORUM gives cleaner, larger, mechanistically-named buckets than STRING's diffuse graph components.

## Verdict (honest)

1. **Outcome-redundancy (use a): NO improvement over STRING.** NN/random ratio 0.892 ≈ STRING 0.86 ≈ FM 0.86, all far from co-dependency's 0.39. Physical complex co-membership does **not** predict single-cell-line knockout substitutability any better than STRING does. This reconfirms the diagnostics' core finding: **knockout-outcome redundancy is a perturbational property** that curated/sequence/text priors don't carry — only DepMap-derived co-dependency does.
2. **Mechanism-diversity (use b): YES, usable and arguably better than STRING.** Profile-cosine lift 1.21×, 65% coverage, and a much more concentrated top-50 structure (two 16–17-gene complex blocks vs STRING's 13 scattered components) make CORUM a clean, interpretable mechanism-bucketing prior for portfolio selection. It is **denser within true complexes** (named, curated) than STRING's continuous graph, which is exactly what a "one pick per mechanism" diversity constraint wants.
3. **Recommendation:** Do not use CORUM as the outcome-redundancy kernel — it fails the same way STRING does. Do consider it for the mechanism-diversity / portfolio-deduplication objective, where its curated complex blocks are a strong, low-noise prior. Co-dependency remains the only signal that captures actual knockout redundancy.

## Artifacts
- `data/processed/depmap/corum_sim.parquet` — binary co-membership, 2043×2043, Entrez-indexed.
- `data/processed/depmap/corum_sim_weighted.parquet` — shared-complex counts.
- `data/corum_dl/humanComplexes.txt` — raw CORUM 5.3 download.
- `data/corum_dl/build_validate.py`, `data/corum_dl/results.json` — reproducible build + metrics.

## CORUM mechanism-k-DPP β-sweep (2026-06-13) — same sparsity wall as STRING

q=predicted lethality (UCB, normalized), S=CORUM binary co-complex, K=30, β-temper sweep:
- ACH-000147: greedy 1.19/13complexes; all β(0.5-8): 0.82/30c
- ACH-000881: greedy 1.26/17c;        all β: 1.10/30c

β INERT, jumps straight to 30/30 distinct complexes at fixed sub-greedy lethality.
CORUM density 0.88% (sparser than STRING's 1.85%@0.7) — among 2000 candidates the
greedy-MAP trivially finds 30 mutually-non-co-complexed genes, so diversity is FREE,
no tradeoff to tune. THIRD confirmation of the same structural fact (FM input-NN,
STRING, CORUM): a BINARY/sparse "same-mechanism" graph over a pathway-spread candidate
pool cannot create a tunable quality-diversity frontier.

KEY DISTINCTION that emerged: co-dependency was CONTINUOUS (every pair has a value,
|S| off-diag 0.057) and gave structure; binary graphs (STRING>thr, CORUM) are sparse
and collapse. The tradeoff likely needs EITHER (a) candidates concentrated in few
large complexes (dense regime — untested 'dense-complex stress test'), OR (b) a
continuous similarity, not binary co-membership. Open question, not yet resolved.

## Continuous similarity (STRING raw scores) — STILL no tradeoff; the root cause is now clear (2026-06-13)

Continuous (un-thresholded) STRING-S, β-temper sweep: β still inert (ACH-000147 0.69->0.75
across β=0.5..16; ACH-000881 flat 0.84). 

ROOT CAUSE (final, general): for the quality-temper β to trace a lethality-vs-diversity
frontier, the HIGH-LETHALITY genes must be densely similar TO EACH OTHER in S — only then
does taking many lethal genes force redundancy the DPP trades against. In ALL identity/
function similarities (3 FM modalities, STRING binary+continuous, CORUM binary+continuous),
high-lethality genes are NOT densely inter-similar — they spread across S — so the greedy-MAP
always finds ~30 high-q mutually-dissimilar genes and diversity is free (β inert).
Co-dependency is the ONLY exception and NOT by coincidence: lethal genes are densely
co-dependent because correlated essentiality is exactly what co-dependency measures. The
property that makes a similarity yield a tunable quality-diversity tradeoff (lethal genes
clustering within it) IS the perturbational property no identity/function representation has.
9 similarity sources/forms tested; all fail for this one root reason. This is intrinsic,
not a graph-choice or binary-vs-continuous detail.
