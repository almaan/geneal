# Why k-DPP collapses to greedy on real DepMap — mechanism + method-development directions

Diagnosed empirically (ACH-000147 Breast, 500 genes, ESM2-35M, one AL round
fit on 50 genes, 450 candidates). Reproduce: see the diagnostic snippet in the
session / `scripts/` analysis.

## The mechanism

k-DPP selects the size-q subset maximizing `det(L_B)` where
`L = diag(q) · S · diag(q)`, q = quality (UCB score), S = GP posterior outcome
correlation. On real data this kernel is **nearly diagonal**:

| quantity | measured | implication |
|---|---|---|
| off-diagonal \|S_ij\| | mean **0.025**, max 0.99 | candidates are essentially outcome-INDEPENDENT |
| posterior std range | 0.169–0.193 | near-HOMOSCEDASTIC; model equally (un)certain everywhere |
| L diag / \|L offdiag\| | **42×** | det(L_B) ≈ ∏ diagonals |

Because off-diagonals ≈ 0, `det(L_B) ≈ ∏_i q_i²`, so k-DPP reduces to **"take the
top-q candidates by quality-SQUARED."** Squaring sharpens the ranking toward the
extreme top, so k-DPP picks an even MORE concentrated batch than linear greedy —
which is exactly why it measured as the LEAST diverse method in the panel and
lost recall to greedy.

## Why the off-diagonals vanish (root cause)

`S_ij` is the GP posterior outcome correlation. With a Matérn kernel over 320-d
frozen ESM2 embeddings and only ~50 training points, the posterior over 450
candidates is near-independent: two genes' predicted lethality co-vary only if
their embeddings are nearly identical, which real genes seldom are. The
biological redundancy diversity SHOULD exploit — "knock out X or Y, same pathway,
correlated effect" — is NOT represented in the frozen-embedding GP posterior. The
model has no pathway structure, so its covariance cannot encode redundancy. The
diversity term is therefore measuring the wrong thing (input-embedding proximity
filtered through a near-flat GP), not biological outcome redundancy.

## Why it nonetheless helped on the selectivity (dual) target

The differential target lethality_A − lethality_B subtracts two noisy signals →
less peaked, higher and more variable posterior std → off-diagonals and the
exploration term matter more → greedy exploitation gets stuck and
diversity/exploration (k-DPP, fantasy) wins. The k-DPP advantage tracks posterior
uncertainty: it appears exactly where the surrogate is uncertain and the target
is hard, and vanishes where the target is peaked and the model is confident.

## Method-development directions (this is the constructive part)

1. **Move diversity to a structured space.** Outcome-covariance from a
   frozen-embedding GP is near-diagonal; replace/augment S with a kernel that has
   real structure — a GRN / pathway-graph Laplacian kernel, or direct
   embedding-space redundancy (what CoreSet/TypiClust use, and they DO get
   diversity). The DPP machinery is fine; the similarity it consumes is wrong.
2. **Stop quality-squared over-sharpening.** Use q_i (not q_i²) or a tempered
   weight `exp(α·score)` with tunable α so the kernel doesn't collapse onto the
   top by construction. This is the standard k-DPP quality knob.
3. **Fix the surrogate, not the acquisition.** The GP posterior is
   near-homoscedastic, so UCB's exploration term is ~constant — the surrogate is
   the bottleneck. A deep-kernel GP (learn a warping on top of frozen ESM2) or a
   genuinely heteroscedastic model would restore a usable uncertainty/covariance
   signal for any diversity-aware acquisition.
4. **Target the regime where diversity pays.** Empirically the win is on noisy /
   differential / selectivity objectives. A method that ADAPTS the
   quality–diversity balance to posterior uncertainty (diversify more when the
   model is uncertain, exploit when confident) would unify the single-objective
   and selectivity results.

## Paper implication

This is itself a publishable characterization: **batch outcome-diversity (DPP)
fails for frozen-FM-embedding active learning on peaked targets because the GP
posterior covariance is near-diagonal — the embedding carries no outcome-redundancy
structure — and helps only on harder, noisier (selectivity) targets where the
posterior is uncertain.** It motivates structured-similarity or deep-kernel
surrogates as the fix. Honest negative-with-mechanism + a clear method path.
