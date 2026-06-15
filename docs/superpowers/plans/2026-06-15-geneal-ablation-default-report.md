# geneal Default Ablation Report — Implementation Plan (Plan 7)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or executing-plans. Steps use `- [ ]`.

**Goal:** Build the project's \emph{default} output: TWO clean analyses, each answering
one objective, comparing our solutions against naive and prior-work baselines.
\textbf{We deliberately do NOT entangle them into one $3\times3$ grid} — safety and
diversity are separate questions and are reported separately.

\textbf{Two analyses:}
\begin{enumerate}
\item \textbf{Safety vs.\ efficacy} — pick targets lethal to the cancer cell but not broadly toxic. Axis = \emph{safety only}.
\item \textbf{Diversity / robustness} — spread picks across pathways. Capping and k-DPP are \emph{operators layered on top of} a base nomination, not grid cells.
\end{enumerate}

---

### Analysis A — safety vs.\ efficacy

One axis, the \textbf{safety rule}:
\begin{itemize}
\item \texttt{none}: greedy efficacy (top-K predicted efficacy).
\item \texttt{truncation}: max predicted efficacy s.t.\ predicted toxicity $\le\tau$.
\item \texttt{ehvi}: dual-objective AL — EHVI acquisition over efficacy+toxicity GPs (toxicity \emph{learned}), constrained nomination.
\end{itemize}
Plus 3 baselines: \texttt{random}, \texttt{coreset}, \texttt{typiclust}.
\textbf{6 methods.} All plotted in the single efficacy--toxicity tradeoff
(method-points with error bars + per-gene cloud with picks overlaid).
Acquisition: safety \texttt{none}/\texttt{truncation} share a greedy-efficacy AL loop
(toxicity known a priori = common-essential); \texttt{ehvi} runs an EHVI loop (toxicity
learned); baselines run their own acquisition. \textbf{No diversity operator here.}

### Analysis B — diversity / robustness

The diversity operators applied at nomination, layered on TWO bases (per user):
\begin{itemize}
\item Bases: \texttt{greedy} (naive top-K efficacy) and \texttt{best-safety} (the winning safety method from Analysis A).
\item Operator: \texttt{none} $\to$ \texttt{cap} (per-pathway, CORUM membership) $\to$ \texttt{kdpp} (quality-weighted k-DPP, \textbf{STRING-derived similarity} $S$).
\end{itemize}
$2\times3 = 6$ nomination variants. Metrics: pathway concentration, dropout robustness,
n\_pathways (distinct CORUM complexes), $\alpha$-NDCG — and efficacy retained (operators
must de-concentrate \emph{without} tanking efficacy). \textbf{Capping is the bolt-on-any-method
demo}: it needs only the graph (STRING similarity / CORUM membership), so it composes
with any base.

\textbf{Why STRING here and not as a prediction embedding.} STRING is a network, not a
per-gene feature vector — it yields a \emph{similarity} $S_{ij}$ (combined score), not an
embedding to regress on. So PubMedBERT stays the efficacy/toxicity predictor throughout;
STRING similarity + CORUM membership supply the \emph{diversity structure} (k-DPP $S$,
capping). They are complementary, not competing.

---

**Efficiency.** Methods sharing an acquisition share one AL loop per (line, seed). We run
\textbf{4 distinct acquisitions} (random, greedy-efficacy, coreset, typiclust) plus EHVI
once per (line, seed), then apply each analysis's nomination rules to the revealed sets.
EHVI is the only expensive acquisition (vectorized MC + candidate shortlist).

**Scale (default, tunable).** 5k HVG panel, 6 cell lines, 3 seeds, \textbf{8 AL rounds
$\times$ batch 10} (n\_initial $\sim$40, $\sim$120 genes assayed). Full genome via
\texttt{--panel} off; this is an experiment option, not a different report.

**Embedding (efficacy prediction) = PubMedBERT (default).** CORUM provides pathway
membership (capping / concentration / robustness); STRING combined-score provides the
k-DPP similarity $S$. Both genome-wide caches. STRING/CORUM are \emph{diversity structure},
never a prediction embedding — see the note in Analysis B.

---

## File structure
```
src/geneal/runner/ablation.py        # CREATE: acquisitions + AL loop + nominate + evaluate
src/geneal/report/ablation_report.py # CREATE: grid heatmaps, method table, efficacy-toxicity tradeoff (points + per-gene), diversity diagram
scripts/run_ablation.py              # CREATE: CLI (panel/lines/seeds/K/tau/...), runs grid+baselines, writes report
jobs/ablation.sh                     # CREATE: SLURM sbatch (braid, thread-pinned)
tests/test_ablation.py               # CREATE
```

---

## Task 1: ablation engine

**Files:** `src/geneal/runner/ablation.py`, `tests/test_ablation.py`.

Provide:
\begin{itemize}
\item \texttt{acquire(kind, X, eff, tox, revealed, q, rng, surr\_factory, ehvi\_samples, shortlist)} $\to$ list of newly-revealed indices for one round, for \texttt{kind} in \{random, greedy, coreset, typiclust, ehvi\}. greedy/ehvi fit GP(s) on revealed labels; coreset/typiclust use embedding geometry; ehvi uses \texttt{mc\_ehvi} greedy batch with fantasies (reuse \texttt{geneal.models.multiobjective}).
\item \texttt{run\_acquisition(kind, X, eff, tox, n\_init, rounds, q, seed, ...)} $\to$ final revealed indices (the AL loop). Common random numbers: initial set + noise seeded from \texttt{seed}.
\item \texttt{nominate(revealed, X, eff, tox, membership, K, safety, diversity, tau, S=None, ...)} $\to$ K nominated indices. Fits final efficacy GP (and toxicity GP if safety=ehvi); predicts; applies safety filter (none / predicted-tox$\le\tau$) then diversity selection (none=top-K by predicted efficacy / cap=HedgedSelect on \texttt{membership} / kdpp=KDPP over eligible using STRING similarity \texttt{S}). Returns indices. Analysis A calls with diversity=none; Analysis B sweeps diversity over a fixed safety base.
\item \texttt{evaluate(pick, eff, tox, membership)} $\to$ dict of TRUE-value metrics: mean\_efficacy, max\_efficacy, mean\_toxicity, concentration, robustness, n\_pathways (distinct CORUM complexes), alpha\_ndcg.
\end{itemize}

- [ ] **Step 1: failing test** `tests/test_ablation.py` — on a toy separable problem (efficacy~dim0, toxicity~dim1): (i) each acquisition kind returns $q$ distinct unrevealed indices; (ii) \texttt{run\_acquisition} grows the revealed set by $q$/round and is deterministic given seed; (iii) \texttt{nominate} with safety=truncation returns only genes with (true, for the test) toxicity below a generous $\tau$; (iv) safety=none vs cap diversity changes the pick; (v) \texttt{evaluate} returns all metric keys with finite values.
- [ ] **Step 2:** run, confirm fail.
- [ ] **Step 3:** implement `ablation.py`. Reuse: \texttt{GPRSurrogate}; \texttt{UCB}/\texttt{RandomAcquisition}/\texttt{GreedyMean}; \texttt{TopQGreedy}/\texttt{CoreSet}/\texttt{TypiClust}/\texttt{KDPP}/\texttt{HedgedSelect}; \texttt{mc\_ehvi}/\texttt{pareto\_front}; \texttt{portfolio} metrics; \texttt{AlphaNDCG}. Toxicity ground truth = common-essential (known); EHVI acquisition + safety=ehvi additionally LEARN a toxicity GP from revealed toxicity labels.
- [ ] **Step 4:** run, confirm pass.
- [ ] **Step 5:** commit `feat: ablation engine (acquisitions + AL loop + nominate + evaluate)`.

---

## Task 2: ablation experiment script

**Files:** `scripts/run_ablation.py`.

- [ ] **Step 1:** CLI args: `--embeddings` (default genome-wide PubMedBERT cache), `--string` (STRING combined-score edges for the k-DPP $S$), `--corum` (CORUM membership for capping), `--panel` (default 5k panel; omit for full genome), `--n-cell-lines 6`, `--seeds 0 1 2`, `--K 30`, `--tau 0.5`, `--n-initial`, `--n-rounds 8`, `--batch 10`, `--cap 2`, `--ehvi-samples 32`, `--shortlist 150`, `--out-root res/runs_ablation`. Build the STRING similarity $S$ once (dense sub-block over the panel; sparse-graph → cosine/combined-score fallback to identity for genes absent from STRING).
- [ ] **Step 2:** For each (line, seed): run the distinct acquisitions ONCE (cache revealed sets), then build BOTH analyses' method rows.
  \begin{itemize}
  \item \textbf{Analysis A} (6): \texttt{greedy}(safety=none), \texttt{truncation}, \texttt{ehvi} + baselines \texttt{random}/\texttt{coreset}/\texttt{typiclust}. All diversity=none.
  \item \textbf{Analysis B} (6): bases $\in$ \{\texttt{greedy}, \texttt{best-safety}\} $\times$ operator $\in$ \{\texttt{none}, \texttt{cap}, \texttt{kdpp}\}. \texttt{best-safety} = the winning safety method from A (resolved after A's metrics are in; default \texttt{truncation} if tie). k-DPP uses STRING $S$; cap uses CORUM membership.
  \end{itemize}
  Record per (analysis, method, line, seed) the evaluate() metrics, tagged with analysis + axis labels. Also save per-gene (efficacy, toxicity, picked-by) for the tradeoff scatter (rep line, seed 0).
- [ ] **Step 3:** write parquet(s) + call the report builder.
- [ ] **Step 4:** smoke-run small (2 lines, 1 seed, few rounds); confirm 12 methods present + report builds.
- [ ] **Step 5:** commit `feat: ablation experiment script (3x3 grid + baselines, full AL loop)`.

---

## Task 3: default ablation report

**Files:** `src/geneal/report/ablation_report.py`.

Elegant self-contained HTML, TWO clearly-separated sections (not a grid):
- [ ] **Header + glossary**: define the two analyses, the safety axis, the diversity operators, the metrics, and the PubMedBERT(predict)/STRING+CORUM(diversity-structure) split.
- [ ] **Section A — safety vs.\ efficacy**: (a) the 6 methods as points at (mean toxicity, mean efficacy) with x/y 95\%CI error bars, Pareto-better = up-left; (b) per-gene cloud with each safety method's picks overlaid; (c) method table (mean+max efficacy, mean toxicity, $\pm$CI). Auto headline: which safety rule wins the frontier.
- [ ] **Section B — diversity / robustness**: $2\times3$ small-multiples (rows=base \{greedy, best-safety\}, cols=operator \{none, cap, kdpp\}) of concentration + robustness; a grouped bar of concentration / robustness / n\_pathways / $\alpha$-NDCG; and an efficacy-vs-concentration scatter showing operators de-concentrate at little efficacy cost. Auto headline: best operator per base, efficacy retained.
- [ ] commit `feat: default two-analysis report (safety-efficacy tradeoff + diversity/robustness)`.

---

## Task 4: SLURM job + make target + set as default

- [ ] `jobs/ablation.sh`: sbatch (braid/account, micromamba, thread-pinned), runs `scripts/run_ablation.py` with the default args; `EMB`/`PANEL`/`NLINES`/`SEEDS` overridable; full genome via `PANEL=` empty.
- [ ] `Makefile`: `ablation` target (5k default) + `ablation-fullgenome`; mark in REPRODUCE.md as the default output.
- [ ] Run the default (5k, 6 lines, 3 seeds) via sbatch; verify the report.
- [ ] commit `feat: ablation SLURM job + make target; set as default report`.

---

## Self-review
- TWO separate analyses, not one 3$\times$3 grid (per user): A = safety axis vs efficacy (6 methods incl. naive + prior-work baselines); B = diversity operators \{none,cap,kdpp\} layered on TWO bases \{greedy, best-safety\} = 6 variants.
- Capping framed as a bolt-on-any-method operator (graph-only: CORUM membership / STRING $S$); not a grid cell.
- STRING used as a \emph{similarity} for k-DPP $S$, never a prediction embedding; PubMedBERT predicts throughout (per user correction).
- Full AL loop per method; acquisitions shared across cells (greedy/random/coreset/typiclust + EHVI per (line,seed), only EHVI expensive).
- Section A efficacy--toxicity tradeoff: all 6 methods (points) + per-gene overlay (both, per user).
- Metrics: efficacy(mean+max), toxicity, concentration, robustness, distinct-pathways, $\alpha$-NDCG (recall@k excluded).
- Panel is an experiment option (5k default / full genome), same report.
- Reuses every existing component; only the ablation engine + report are new.
