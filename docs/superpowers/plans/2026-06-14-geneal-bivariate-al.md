# geneal Bivariate Efficacy–Toxicity Active Learning — Implementation Plan (Plan 6)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]`.

**Goal:** Treat BOTH efficacy and toxicity as unknown and explore them jointly with a multi-objective active-learning loop (Expected Hypervolume Improvement, EHVI). Each assay reveals a gene's effect in the target line (efficacy) AND in a toxicity reference (toxicity), so the loop explores two axes simultaneously and nominates selective targets (lethal in target, safe in reference) under JOINT uncertainty. Three conditions, compared:
1. **known-tox** — toxicity known a priori (common-essential); single-objective AL on efficacy + constraint. The baseline comparison.
2. **learned-contrast** — toxicity = lethality in a FIXED normal-ish proxy cell line (different domain); predicted by a 2nd surrogate, explored.
3. **learned-population** — toxicity = mean lethality across the population; predicted, explored.

**Why this matters:** real target nomination doesn't know toxicity up front — you discover efficacy and toxicity together. The contribution is a principled bivariate AL loop (EHVI over the efficacy↑/toxicity↓ Pareto front) and a head-to-head against the a-priori-known-toxicity ceiling: how much do you lose by having to LEARN toxicity vs knowing it?

**Architecture:** Two independent GP surrogates (efficacy in line A; toxicity = lethality in fixed line B, or population mean). A bivariate AL loop acquires a size-q batch each round by greedy EHVI over the two posteriors, reveals both labels (with noise), refits both, repeats. Nomination/eval each round on the (efficacy↑, toxicity↓) objective space: dominated hypervolume of the revealed selective set, and recall of the true selective-top-k. Reuses GPRSurrogate, NoiseModel, the DepMap adapter, and the report styling.

**Tech stack:** existing geneal stack. EHVI via Monte-Carlo over the two GP posteriors (robust, simple) with a greedy-fantasy batch. No new deps.

**Convention:** efficacy = −effect in line A (higher=better); toxicity = −effect in line B / population (higher=more toxic). Selectivity objective space = maximize efficacy, minimize toxicity. Reference point for hypervolume = (min efficacy, max toxicity) margins.

---

## File Structure
```
src/geneal/models/multiobjective.py   # CREATE: pareto_front, hypervolume2d, mc_ehvi
src/geneal/runner/bivariate.py         # CREATE: BivariateALRunner (two surrogates, EHVI batch loop)
src/geneal/report/bivariate_report.py  # CREATE: hypervolume-over-rounds + 2D objective scatter + known-vs-learned
scripts/run_bivariate_al.py            # CREATE: 3 conditions, metrics, report
tests/test_multiobjective.py
tests/test_bivariate.py
```

---

## Task 1: Multi-objective primitives (Pareto, hypervolume, EHVI)

**Files:** Create `src/geneal/models/multiobjective.py`; Test `tests/test_multiobjective.py`.

Objectives are framed as MAXIMIZE both coordinates (pass efficacy and −toxicity). 2-D only.

- [ ] **Step 1: failing test**

```python
# tests/test_multiobjective.py
import numpy as np
from geneal.models.multiobjective import pareto_front, hypervolume2d, mc_ehvi


def test_pareto_front_2d():
    # points (maximize both); (2,2) dominates (1,1); (3,0),(0,3) non-dominated
    P = np.array([[1, 1], [2, 2], [3, 0], [0, 3], [2, 1]])
    idx = pareto_front(P)
    assert set(idx) == {1, 2, 3}  # (2,2),(3,0),(0,3)


def test_hypervolume2d_monotone():
    ref = np.array([0.0, 0.0])
    hv1 = hypervolume2d(np.array([[2, 2]]), ref)
    assert np.isclose(hv1, 4.0)
    hv2 = hypervolume2d(np.array([[2, 2], [3, 1]]), ref)  # adds region
    assert hv2 > hv1


def test_mc_ehvi_prefers_improving_candidate():
    rng = np.random.default_rng(0)
    ref = np.array([0.0, 0.0])
    front = np.array([[2.0, 2.0]])
    # cand A: mean (3,3) clearly extends the front; cand B: mean (1,1) dominated
    mean = np.array([[3.0, 3.0], [1.0, 1.0]])
    std = np.array([[0.2, 0.2], [0.2, 0.2]])
    ehvi = mc_ehvi(mean, std, front, ref, rng, n_samples=200)
    assert ehvi[0] > ehvi[1]
    assert ehvi[1] >= 0.0
```

- [ ] **Step 2: run, confirm fail.**

- [ ] **Step 3: implement `src/geneal/models/multiobjective.py`**

```python
# src/geneal/models/multiobjective.py
from __future__ import annotations
import numpy as np


def pareto_front(P: np.ndarray) -> list[int]:
    """Indices of non-dominated rows (maximize both columns). 2-D."""
    P = np.asarray(P, float)
    n = len(P)
    keep = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if j == i:
                continue
            if (P[j, 0] >= P[i, 0] and P[j, 1] >= P[i, 1] and
                    (P[j, 0] > P[i, 0] or P[j, 1] > P[i, 1])):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return keep


def hypervolume2d(P: np.ndarray, ref: np.ndarray) -> float:
    """Dominated hypervolume (area) of point set P above reference ref (maximize
    both). Only non-dominated points above ref contribute."""
    P = np.asarray(P, float)
    if len(P) == 0:
        return 0.0
    pf = P[pareto_front(P)]
    pf = pf[(pf[:, 0] > ref[0]) & (pf[:, 1] > ref[1])]
    if len(pf) == 0:
        return 0.0
    # sort by x descending; sweep
    pf = pf[np.argsort(-pf[:, 0])]
    area = 0.0
    prev_y = ref[1]
    for x, y in pf:
        if y > prev_y:
            area += (x - ref[0]) * (y - prev_y)
            prev_y = y
    return float(area)


def mc_ehvi(mean, std, front, ref, rng, n_samples: int = 128) -> np.ndarray:
    """Monte-Carlo Expected Hypervolume Improvement per candidate (maximize both
    objectives). mean/std: (n_cand, 2) independent-Gaussian posteriors. front:
    current Pareto set (m, 2). Returns EHVI >= 0 per candidate."""
    mean = np.asarray(mean, float); std = np.asarray(std, float)
    front = np.asarray(front, float).reshape(-1, 2)
    base = hypervolume2d(front, ref)
    n = len(mean)
    out = np.zeros(n)
    for s in range(n_samples):
        draw = mean + std * rng.standard_normal(mean.shape)  # (n,2)
        for i in range(n):
            hv = hypervolume2d(np.vstack([front, draw[i]]), ref)
            out[i] += max(hv - base, 0.0)
    return out / n_samples
```

- [ ] **Step 4: run, confirm 3 passed.**
- [ ] **Step 5: commit** `feat: add multi-objective primitives (Pareto front, 2D hypervolume, MC-EHVI)`.

---

## Task 2: Bivariate AL runner (two surrogates, EHVI batch loop)

**Files:** Create `src/geneal/runner/bivariate.py`; Test `tests/test_bivariate.py`.

- [ ] **Step 1: failing test**

```python
# tests/test_bivariate.py
import numpy as np
from geneal.runner.bivariate import BivariateALRunner
from geneal.models.surrogate import GPRSurrogate
from geneal.models.noise import GaussianNoise


def _toy():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((120, 5))
    # efficacy correlated with dim0; toxicity with dim1 (so they're separable)
    eff = X[:, 0] + 0.1 * rng.standard_normal(120)
    tox = X[:, 1] + 0.1 * rng.standard_normal(120)
    return X, eff, tox


def test_bivariate_loop_runs_and_grows_hypervolume():
    X, eff, tox = _toy()
    r = BivariateALRunner(lambda: GPRSurrogate(n_iters=40), GaussianNoise(0.01))
    hist = r.run(X, eff, tox, n_initial=15, n_rounds=4, batch_size=5,
                 seed=0)
    assert len(hist) == 5  # round 0..4
    hv = [h["hypervolume"] for h in hist]
    assert hv[-1] >= hv[0] - 1e-9      # hypervolume non-decreasing overall
    assert all(set(h) >= {"round", "hypervolume", "n_revealed"} for h in hist)


def test_bivariate_deterministic():
    X, eff, tox = _toy()
    def run():
        return BivariateALRunner(lambda: GPRSurrogate(n_iters=40), GaussianNoise(0.01)
                                 ).run(X, eff, tox, 15, 3, 5, seed=1)
    a, b = run(), run()
    assert [h["hypervolume"] for h in a] == [h["hypervolume"] for h in b]
```

- [ ] **Step 2: run, confirm fail.**

- [ ] **Step 3: implement `src/geneal/runner/bivariate.py`**

```python
# src/geneal/runner/bivariate.py
from __future__ import annotations
import numpy as np
from geneal.models.multiobjective import pareto_front, hypervolume2d, mc_ehvi


class BivariateALRunner:
    """Active learning over TWO objectives (efficacy↑, toxicity↓), both unknown.
    Each acquisition reveals both labels (one assay profiles both contexts).
    Batch acquisition = greedy EHVI over the two GP posteriors. Toxicity is
    minimized, so we maximize (efficacy, −toxicity) in objective space."""

    def __init__(self, surrogate_factory, noise, ehvi_samples: int = 64):
        self.make = surrogate_factory
        self.noise = noise
        self.ehvi_samples = ehvi_samples

    def run(self, X, efficacy, toxicity, n_initial, n_rounds, batch_size, seed):
        X = np.asarray(X, float)
        eff = np.asarray(efficacy, float); tox = np.asarray(toxicity, float)
        n = len(eff)
        rng = np.random.default_rng(seed)
        noise_e = self.noise.draw(n, np.random.default_rng((seed, 1)))
        noise_t = self.noise.draw(n, np.random.default_rng((seed, 2)))
        acq_rng = np.random.default_rng((seed, 99))
        # objective space: maximize (eff, -tox). reference = slightly below worst.
        ref = np.array([eff.min() - 0.1 * (np.ptp(eff) + 1e-9),
                        -(tox.max()) - 0.1 * (np.ptp(tox) + 1e-9)])

        revealed = list(rng.permutation(n)[:n_initial])
        def obs(idx):
            return np.array([eff[idx] + noise_e[idx], -(tox[idx] + noise_t[idx])])

        def record(rnd):
            pts = np.array([[eff[i], -tox[i]] for i in revealed])
            return {"round": rnd, "n_revealed": len(revealed),
                    "hypervolume": hypervolume2d(pts, ref)}

        hist = [record(0)]
        for r in range(1, n_rounds + 1):
            ye = np.array([eff[i] + noise_e[i] for i in revealed])
            yt = np.array([tox[i] + noise_t[i] for i in revealed])
            se = self.make().fit(X[revealed], ye)
            st = self.make().fit(X[revealed], yt)
            cand = [i for i in range(n) if i not in set(revealed)]
            if not cand:
                break
            me, sde = se.predict(X[cand]); mt, sdt = st.predict(X[cand])
            mean = np.column_stack([me, -mt])     # maximize eff, -tox
            std = np.column_stack([sde, sdt])
            # current front from revealed objective points
            pts = np.array([[eff[i], -tox[i]] for i in revealed])
            front = pts[pareto_front(pts)]
            # greedy-EHVI batch with fantasies (add picked mean to the front)
            picked_local = []
            cur_front = front.copy()
            avail = list(range(len(cand)))
            for _ in range(min(batch_size, len(cand))):
                ehvi = mc_ehvi(mean[avail], std[avail], cur_front, ref,
                               acq_rng, self.ehvi_samples)
                j = avail[int(np.argmax(ehvi))]
                picked_local.append(j); avail.remove(j)
                cur_front = np.vstack([cur_front, mean[j]])  # fantasize at mean
            for j in picked_local:
                revealed.append(cand[j])
            hist.append(record(r))
        return hist
```

- [ ] **Step 4: run, confirm 2 passed** (may take ~30s; GP fits + MC-EHVI).
- [ ] **Step 5: commit** `feat: add bivariate EHVI active-learning runner`.

---

## Task 3: Experiment — 3 conditions on real DepMap + report

**Files:** Create `scripts/run_bivariate_al.py`, `src/geneal/report/bivariate_report.py`.

- [ ] **Step 1: write `scripts/run_bivariate_al.py`**. For chosen target lines, embedding (genome-wide cache subset by `--panel`):
  - **toxicity reference**: `--tox-mode {known, contrast, population}`. known = common-essential (a-priori, NOT explored — single-objective AL on efficacy then constrain). contrast = lethality in a FIXED proxy line `--contrast-line` (default: a high-coverage line from a distinct lineage; print it). population = mean lethality across all lines.
  - For learned modes: run `BivariateALRunner` on (efficacy in target line, toxicity = reference). Track hypervolume + true selective-top-k recall over rounds.
  - For known mode: efficacy-only AL loop, toxicity known; same metrics for comparison.
  - argparse: --embeddings, --panel, --cell-lines/--n-cell-lines, --contrast-line, --tox-mode (run all three by default), --n-initial, --n-rounds, --batch, --seeds, --out-root.
  - Save per-round metrics parquet + build the report.

- [ ] **Step 2: write `src/geneal/report/bivariate_report.py`** — elegant HTML:
  - **hypervolume-over-rounds** curve, one line per condition (known / contrast / population), error bars over seeds. Shows how fast each discovers the efficacy-toxicity frontier.
  - **selective-recall-over-rounds** (recall of true selective-top-k).
  - **2D objective scatter** (efficacy vs −toxicity) of the final revealed set, Pareto front highlighted, for one rep line.
  - **known-vs-learned gap** summary: final hypervolume / recall, known (ceiling) vs learned-contrast vs learned-population, mean±CI. Glossary explaining EHVI, hypervolume, the toxicity modes.

- [ ] **Step 3: run** on ~4 lines, 2 seeds:
```bash
micromamba run -n geneal python scripts/run_bivariate_al.py \
  --embeddings data/processed/embeddings/pubmedbert_all.parquet \
  --panel data/processed/depmap/panel_hvg.txt --n-cell-lines 4 --seeds 0 1 \
  --n-rounds 8 --batch 10
```
Report hypervolume + recall per condition.

- [ ] **Step 4: write `data/BIVARIATE_RESULTS.md`** — honest read: does learning toxicity (contrast/population) approach the known-toxicity ceiling? how much exploration cost? which toxicity reference (contrast vs population) is easier to learn? Does EHVI explore the frontier faster than a scalarized baseline (optional add)? Do NOT tune to flatter.
- [ ] **Step 5: commit** `experiment: bivariate efficacy-toxicity AL (known vs learned-contrast vs learned-population)`.

---

## Self-Review (completed during planning)
- **Covers the ask:** toxicity treated as unknown + explored jointly with efficacy (Task 2 two-surrogate EHVI loop) ✓; two learned options — contrast line (fixed normal-ish proxy) + population (Task 3 --tox-mode) ✓; a-priori known-toxicity kept as comparison ✓; multi-objective Pareto/EHVI acquisition (chosen) ✓; fixed proxy line (chosen) ✓.
- **EHVI tractability:** Monte-Carlo EHVI (Task 1) — simple, correct, batch via greedy-fantasy. Exact analytic 2-D EHVI is a future optimization, not needed for these candidate-set sizes.
- **Determinism:** seeded RNG for noise + acquisition; test asserts reproducibility.
- **Reuse:** GPRSurrogate, GaussianNoise, DepMap adapter, report styling. Only multiobjective + bivariate runner are new logic.
- **Honesty guards:** Task 4 forbids tuning; the key question (cost of learning toxicity vs knowing it) is reported as-is.

## Out of scope
Analytic EHVI; >2 objectives; the pathway-hedging cap (orthogonal — can layer on the nominated set later). Real normal-tissue toxicity (DepMap has none; fixed proxy line + population are the stand-ins, documented).
