# geneal Metrics + Report + Dual-Case Overhaul — Implementation Plan (Plan 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Replace the single-metric evaluation with a panel of evaluation metrics (recall@k, max-lethal-value, cumulative-set diversity, α-NDCG), rewrite the HTML report to be elegant with stage-labeled error-bar plots and per-metric stage tables (model rows × round columns, mean±CI), add a dual-cell-line selectivity (efficacy–toxicity) experiment with a 2D scatter visualization, and speed the experiment harness via process parallelism + an optional GPU GP. Validate on a small, tunable config (4 cell lines × 2 seeds).

**Architecture:** Evaluation is separated from acquisition. The Runner already drives acquisition via the surrogate+acquisition+selection; we add an **evaluation-metric panel** computed each round on the *cumulative revealed set* (in acquisition order, for rank-sensitive α-NDCG), recorded as extra columns. A new differential `Dataset` builder supports the dual case by setting `target = lethality_A − lethality_B`; the entire existing single-objective stack then runs on it unchanged. A parallel sweep harness fans the independent (cell-line × method × seed) runs across processes. The report module is rewritten to consume the richer per-round/per-metric DataFrame.

**Tech Stack:** existing geneal stack (numpy, scipy, gpytorch, pandas, plotly, jinja2, pytest). Process parallelism via `concurrent.futures.ProcessPoolExecutor` (stdlib). GPU via gpytorch `.to(device)`.

**Conventions:** target = lethality = −(Chronos effect), higher = better; metrics evaluated on the cumulative revealed set per round; "stage"/"round" used interchangeably in plots/tables; all new experiment knobs (n_cell_lines, n_seeds, k, rounds, batch, n_initial, alpha, n_clusters) are CLI/config params with the requested defaults (4 lines, 2 seeds).

---

## File Structure

```
src/geneal/metrics/panel.py          # CREATE: MaxValue, CumulativeDiversity, AlphaNDCG + evaluate_panel
src/geneal/metrics/recall.py         # (exists, unchanged)
src/geneal/runner/runner.py          # MODIFY: record an eval-metric panel + cumulative acquisition order
src/geneal/models/surrogate.py       # MODIFY: GPRSurrogate device='cpu'|'cuda' option
src/geneal/data/dual.py              # CREATE: build_differential_dataset (selectivity target A-B)
src/geneal/runner/sweep.py           # CREATE: parallel sweep over (cell_line, method, seed)
src/geneal/report/report2.py         # CREATE: elegant report — error-bar stage plots + per-metric stage tables
src/geneal/report/dual_plot.py       # CREATE: 2D lethality_A vs lethality_B scatter
scripts/run_single_sweep.py          # CREATE: single-objective panel sweep (4 lines x 2 seeds, tunable)
scripts/run_dual_experiment.py       # CREATE: dual selectivity experiment + 2D scatter
tests/test_panel.py                  # CREATE
tests/test_dual.py                   # CREATE
tests/test_report2.py                # CREATE
tests/test_sweep.py                  # CREATE
```

---

## Task 1: Evaluation-metric panel (max-value, diversity, α-NDCG)

**Files:** Create `src/geneal/metrics/panel.py`; Test `tests/test_panel.py`.

All panel metrics share one signature so the Runner can call them uniformly:
`evaluate(revealed_order: list[int], target: np.ndarray, embeddings: np.ndarray) -> float`
where `revealed_order` is the cumulative list of revealed gene indices **in acquisition order** (rank matters for α-NDCG; the others are order-insensitive).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_panel.py
import numpy as np
from geneal.metrics.panel import MaxValue, CumulativeDiversity, AlphaNDCG, evaluate_panel


def test_maxvalue_is_best_revealed_over_global_max():
    target = np.array([1.0, 5.0, 2.0, 4.0])  # global max 5.0
    m = MaxValue()
    assert m.evaluate([0, 2], target, None) == 2.0 / 5.0   # best revealed 2.0
    assert m.evaluate([0, 1], target, None) == 1.0          # found the max
    assert m.name == "max_value"


def test_maxvalue_handles_nonpositive_globalmax():
    target = np.array([-3.0, -1.0, -2.0])  # global max -1.0
    # normalized by max(|global max|, eps); best revealed -3 -> negative ratio ok
    v = MaxValue().evaluate([0], target, None)
    assert np.isfinite(v)


def test_cumulative_diversity_mean_pairwise():
    emb = np.array([[0.0, 0.0], [3.0, 4.0], [0.0, 0.0]])  # 0&2 identical, 1 far
    d = CumulativeDiversity()
    assert d.evaluate([0, 2], None, emb) == 0.0      # two identical -> 0
    assert np.isclose(d.evaluate([0, 1], None, emb), 5.0)  # dist (0,0)-(3,4)=5
    assert d.name == "diversity"


def test_alpha_ndcg_penalizes_same_cluster_redundancy():
    # 4 genes: 0,1 in a high-lethal cluster A; 2 in lethal cluster B; 3 inert.
    # embeddings make clusters obvious; lethality high for 0,1,2.
    emb = np.array([[0.0, 0.0], [0.05, 0.0], [10.0, 0.0], [20.0, 0.0]])
    target = np.array([5.0, 5.0, 5.0, 0.0])
    m = AlphaNDCG(k=2, alpha=0.5, n_clusters=3, seed=0)
    # picking one-from-A then B (diverse) should score >= picking both from A
    diverse = m.evaluate([0, 2], target, emb)
    redundant = m.evaluate([0, 1], target, emb)
    assert diverse >= redundant
    assert 0.0 <= diverse <= 1.0 and 0.0 <= redundant <= 1.0
    assert m.name == "alpha_ndcg@2"


def test_evaluate_panel_returns_named_dict():
    emb = np.random.default_rng(0).standard_normal((10, 3))
    target = np.arange(10.0)
    panel = [MaxValue(), CumulativeDiversity(), AlphaNDCG(k=5, n_clusters=3, seed=0)]
    out = evaluate_panel(panel, [0, 1, 2], target, emb)
    assert set(out) == {"max_value", "diversity", "alpha_ndcg@5"}
    assert all(np.isfinite(v) for v in out.values())
```

- [ ] **Step 2: Run, confirm fail** (`ModuleNotFoundError: geneal.metrics.panel`).

- [ ] **Step 3: Implement `src/geneal/metrics/panel.py`**

```python
# src/geneal/metrics/panel.py
from __future__ import annotations
from typing import Sequence
import numpy as np


class MaxValue:
    """Best (highest) true target among revealed genes, normalized by the global
    max so it lands in a comparable range. The 'best lethal value found so far'
    curve — the classic simple-regret-style readout."""
    @property
    def name(self) -> str:
        return "max_value"

    def evaluate(self, revealed_order, target, embeddings) -> float:
        target = np.asarray(target)
        if len(revealed_order) == 0:
            return 0.0
        best = float(np.max(target[list(revealed_order)]))
        denom = max(abs(float(np.max(target))), 1e-8)
        return best / denom


class CumulativeDiversity:
    """Mean pairwise Euclidean distance of the cumulative revealed set in
    embedding space. How spread the collected genes are over time."""
    @property
    def name(self) -> str:
        return "diversity"

    def evaluate(self, revealed_order, target, embeddings) -> float:
        idx = list(revealed_order)
        if len(idx) < 2:
            return 0.0
        X = np.asarray(embeddings)[idx]
        n = len(idx)
        total, cnt = 0.0, 0
        for i in range(n):
            diff = X[i + 1:] - X[i]
            if len(diff):
                total += float(np.sum(np.linalg.norm(diff, axis=1)))
                cnt += len(diff)
        return total / cnt if cnt else 0.0


class AlphaNDCG:
    """alpha-NDCG over the acquisition-ordered revealed list (Clarke et al. 2008).

    Nuggets = k-means clusters of gene embeddings; a gene 'covers' its cluster
    with graded relevance = max(0, lethality). The gain of revealing a gene is
    discounted by (1-alpha)^(# previously-revealed genes in the same cluster), so
    piling into one lethal cluster is penalized and covering many distinct lethal
    clusters is rewarded. Normalized by a greedy 'ideal' ordering -> [0, 1].

    Clusters are computed once per (embeddings) call from ALL genes so the nugget
    structure is fixed; this is a fast k-means (numpy)."""
    def __init__(self, k: int = 50, alpha: float = 0.5, n_clusters: int = 20,
                 seed: int = 0) -> None:
        self.k = k
        self.alpha = alpha
        self.n_clusters = n_clusters
        self.seed = seed

    @property
    def name(self) -> str:
        return f"alpha_ndcg@{self.k}"

    def _labels(self, embeddings):
        X = np.asarray(embeddings, dtype=float)
        n = X.shape[0]
        kk = min(self.n_clusters, n)
        rng = np.random.default_rng(self.seed)
        centers = X[rng.choice(n, size=kk, replace=False)].copy()
        labels = np.zeros(n, dtype=int)
        for _ in range(25):
            d = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
            new = d.argmin(1)
            if np.array_equal(new, labels):
                break
            labels = new
            for c in range(kk):
                pts = X[labels == c]
                if len(pts):
                    centers[c] = pts.mean(0)
        return labels

    def _dcg(self, order, labels, rel):
        seen: dict[int, int] = {}
        dcg = 0.0
        for i, g in enumerate(order[: self.k]):
            c = int(labels[g])
            r = seen.get(c, 0)
            gain = rel[g] * ((1.0 - self.alpha) ** r)
            dcg += gain / np.log2(i + 2)
            seen[c] = r + 1
        return dcg

    def evaluate(self, revealed_order, target, embeddings) -> float:
        order = list(revealed_order)
        if not order:
            return 0.0
        labels = self._labels(embeddings)
        rel = np.clip(np.asarray(target, dtype=float), 0.0, None)
        actual = self._dcg(order, labels, rel)
        # ideal: greedily order the revealed genes to maximize alpha-discounted gain
        ideal_order, seen, pool = [], {}, list(order)
        while pool and len(ideal_order) < self.k:
            best_g, best_gain = None, -np.inf
            for g in pool:
                c = int(labels[g])
                gain = rel[g] * ((1.0 - self.alpha) ** seen.get(c, 0))
                if gain > best_gain:
                    best_gain, best_g = gain, g
            ideal_order.append(best_g)
            seen[int(labels[best_g])] = seen.get(int(labels[best_g]), 0) + 1
            pool.remove(best_g)
        ideal = self._dcg(ideal_order, labels, rel)
        return float(actual / ideal) if ideal > 0 else 0.0


def evaluate_panel(panel, revealed_order, target, embeddings) -> dict:
    """Evaluate every metric in the panel on the cumulative revealed set."""
    return {m.name: float(m.evaluate(revealed_order, target, embeddings))
            for m in panel}
```

- [ ] **Step 4: Run, confirm 5 passed.**
- [ ] **Step 5: Commit** `feat: add evaluation-metric panel (max-value, diversity, alpha-NDCG)`.

---

## Task 2: Runner records the metric panel + acquisition order

**Files:** Modify `src/geneal/runner/runner.py`; Test `tests/test_runner_panel.py`.

**Context:** the Runner records one `metric` (the Objective) + batch diagnostics. Add: (a) it tracks the cumulative revealed indices in acquisition order; (b) each round it evaluates an optional `eval_panel` (list of panel metrics) on that order, recording one column per metric (`eval_<name>`); recall@k stays as `metric`. The panel is passed to `Runner.run(experiment, seeds, eval_panel=None)`; when None, behavior is unchanged.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_runner_panel.py
import numpy as np
from geneal.runner.runner import Runner
from geneal.experiment.objects import ObjectiveObject, DesignObject, Method, Experiment
from geneal.metrics.recall import RecallAtK
from geneal.metrics.panel import MaxValue, CumulativeDiversity, AlphaNDCG
from geneal.models.acquisition import UCB
from geneal.models.selection import TopQGreedy
from geneal.models.surrogate import GPRSurrogate
from geneal.models.noise import GaussianNoise
from geneal.data.dataset import make_synthetic


def test_runner_records_eval_panel_columns():
    ds = make_synthetic(n_genes=60, dim=5, seed=0, noise_sd=0.05)
    exp = Experiment(
        dataset=ds,
        objective=ObjectiveObject(metric=RecallAtK(k=10), direction="maximize"),
        design=DesignObject(n_rounds=3, batch_size=5, n_initial=10, seed=0),
        noise=GaussianNoise(sigma=0.05),
        methods=[Method("al", UCB(2.0), TopQGreedy(), GPRSurrogate(n_iters=50))],
    )
    panel = [MaxValue(), CumulativeDiversity(), AlphaNDCG(k=10, n_clusters=5, seed=0)]
    df = Runner().run(exp, seeds=[0], eval_panel=panel)
    for col in ["eval_max_value", "eval_diversity", "eval_alpha_ndcg@10"]:
        assert col in df.columns
    # max_value should be non-decreasing over rounds (cumulative best)
    s = df.sort_values("round")["eval_max_value"].to_numpy()
    assert np.all(np.diff(s) >= -1e-9)


def test_runner_without_panel_unchanged():
    ds = make_synthetic(n_genes=40, dim=4, seed=0)
    exp = Experiment(ds, ObjectiveObject(RecallAtK(k=5), "maximize"),
                     DesignObject(n_rounds=2, batch_size=4, n_initial=8, seed=0),
                     GaussianNoise(0.05),
                     [Method("al", UCB(2.0), TopQGreedy(), GPRSurrogate(n_iters=40))])
    df = Runner().run(exp, seeds=[0])
    assert "metric" in df.columns
    assert not any(c.startswith("eval_") for c in df.columns)
```

- [ ] **Step 2: Run, confirm fail.**

- [ ] **Step 3: Modify `runner.py`.** Read the current file first. Changes:
  (a) `run(self, experiment, seeds, eval_panel=None)` — thread `eval_panel` into `_run_one`.
  (b) In `_run_one`, maintain `revealed` already in acquisition order (it is: init set then appended `sel`s). Pass the current `revealed` list and `eval_panel` to `_record`.
  (c) In `_record`, after the existing fields, if `eval_panel` is provided, compute `evaluate_panel(eval_panel, revealed, target, ds.embeddings)` and merge each as `eval_<name>`.
  Add import: `from geneal.metrics.panel import evaluate_panel`. Signature of `_record` gains `eval_panel=None`. Round-0 also records the panel (on the initial revealed set).

  Concretely, `_record` becomes:
```python
    @staticmethod
    def _record(method, seed, rnd, revealed, metric, target, ds,
                batch=None, eval_panel=None) -> dict:
        rec = {
            "method": method.name, "seed": seed, "round": rnd,
            "n_revealed": len(revealed),
            "metric": metric.evaluate(revealed, target),
            "metric_name": metric.name,
            "batch_quality": (batch_quality(batch, target)
                              if batch is not None else float("nan")),
            "batch_diversity": (batch_diversity(batch, ds.embeddings)
                                if batch is not None else float("nan")),
        }
        if eval_panel:
            for name, val in evaluate_panel(eval_panel, list(revealed),
                                            target, ds.embeddings).items():
                rec[f"eval_{name}"] = val
        return rec
```
  and the two call sites pass `eval_panel=eval_panel` (round 0 with `batch=None`, in-loop with `batch=sel`). `_run_one` gains an `eval_panel` parameter; `run` passes it through the method loop.

- [ ] **Step 4: Run, confirm 2 passed.**
- [ ] **Step 5: Confirm no regression:** `pytest tests/test_runner.py tests/test_runner_diagnostics.py -v` (expect 5 + 1).
- [ ] **Step 6: Commit** `feat: Runner records evaluation-metric panel per round`.

---

## Task 3: GPU option on GPR + parallel sweep harness

**Files:** Modify `src/geneal/models/surrogate.py` (device option); Create `src/geneal/runner/sweep.py`; Test `tests/test_sweep.py`.

**Honesty note for the implementer:** at ~500 genes exact-GP fits are small; a GPU often does NOT beat CPU here due to transfer/launch overhead, and process-parallelism across (line×method×seed) is the larger, more reliable win. Wire the GPU option (it matters at full-genome scale) but do not claim a speedup you haven't measured — the sweep parallelism is the headline.

- [ ] **Step 1: Add a `device` option to `GPRSurrogate`.** Read the current class. Add `device: str = "cpu"` to `__init__` (store it), and in `fit`/`predict`/`predict_cov` move the model and tensors to `self.device` (e.g. `tx = torch.tensor(...).to(self.device)`, `model = self._build(...)` then `.to(self.device)`, and `.cpu().numpy()` on the way out). `clone()` must pass `device` through. Default `"cpu"` keeps all existing tests valid. Add a test that `GPRSurrogate(device="cpu")` fits/predicts as before (a GPU test is skipped if `torch.cuda.is_available()` is False).

```python
# add to tests/test_sweep.py (device part)
import numpy as np, pytest
from geneal.models.surrogate import GPRSurrogate
from geneal.data.dataset import make_synthetic

def test_gpr_device_cpu_default_ok():
    ds = make_synthetic(n_genes=60, dim=4, seed=0)
    surr = GPRSurrogate(n_iters=50, device="cpu").fit(ds.embeddings[:40], ds.target[:40])
    m, s = surr.predict(ds.embeddings[40:])
    assert m.shape == (20,) and np.all(s >= 0)
    assert surr.clone().device == "cpu"
```

- [ ] **Step 2: Write the parallel sweep + its test.** `sweep.py` exposes:
`run_sweep(build_experiment, cell_lines, seeds, eval_panel=None, max_workers=None) -> pd.DataFrame`
where `build_experiment(cell_line) -> Experiment` is a picklable callable constructing the per-cell-line Experiment, and each (cell_line) Experiment is run via `Runner().run(exp, seeds, eval_panel)` in a separate process; results are concatenated with a `cell_line` column added. Use `concurrent.futures.ProcessPoolExecutor`. Because Experiments contain surrogate objects, parallelize at the **cell-line** granularity (one task per cell line, each running all methods×seeds) — this keeps tasks coarse and picklable via the `build_experiment` factory (pass the factory + cell_line, build inside the worker).

```python
# tests/test_sweep.py (sweep part — append to the file)
import pandas as pd
from geneal.runner.sweep import run_sweep
from geneal.experiment.objects import ObjectiveObject, DesignObject, Method, Experiment
from geneal.metrics.recall import RecallAtK
from geneal.models.acquisition import UCB, RandomAcquisition
from geneal.models.selection import TopQGreedy
from geneal.models.surrogate import GPRSurrogate
from geneal.models.noise import GaussianNoise
from geneal.data.dataset import make_synthetic


def _build(cell_line):
    # toy: cell_line is just a seed offset into synthetic data
    ds = make_synthetic(n_genes=50, dim=4, seed=int(cell_line), noise_sd=0.05)
    return Experiment(
        ds, ObjectiveObject(RecallAtK(k=5), "maximize"),
        DesignObject(n_rounds=2, batch_size=4, n_initial=8, seed=0),
        GaussianNoise(0.05),
        [Method("al", UCB(2.0), TopQGreedy(), GPRSurrogate(n_iters=40)),
         Method("random", RandomAcquisition(), TopQGreedy(), GPRSurrogate(n_iters=40))])


def test_run_sweep_parallel_has_all_cells_and_methods():
    df = run_sweep(_build, cell_lines=["0", "1", "2"], seeds=[0, 1], max_workers=2)
    assert set(df["cell_line"]) == {"0", "1", "2"}
    assert set(df["method"]) == {"al", "random"}
    # 3 cells * 2 methods * 2 seeds * (2 rounds + round0=3) = 36 rows
    assert len(df) == 36
```

```python
# src/geneal/runner/sweep.py
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from geneal.runner.runner import Runner


def _run_one_cell(args):
    build_experiment, cell_line, seeds, eval_panel = args
    exp = build_experiment(cell_line)
    df = Runner().run(exp, seeds=seeds, eval_panel=eval_panel)
    df["cell_line"] = cell_line
    return df


def run_sweep(build_experiment, cell_lines, seeds, eval_panel=None,
              max_workers=None) -> pd.DataFrame:
    """Run build_experiment(cell_line) across cell lines in parallel processes.

    build_experiment must be a top-level (picklable) callable returning an
    Experiment. Each process runs all methods x seeds for one cell line.
    """
    tasks = [(build_experiment, cl, seeds, eval_panel) for cl in cell_lines]
    frames = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for df in ex.map(_run_one_cell, tasks):
            frames.append(df)
    return pd.concat(frames, ignore_index=True)
```

- [ ] **Step 3: Run, confirm sweep + device tests pass** (`pytest tests/test_sweep.py -v`). Note: ProcessPoolExecutor under pytest needs the `_build`/`_run_one_cell` callables to be module-top-level (they are) so they pickle.
- [ ] **Step 4: Confirm no regression on surrogate/predict_cov/kdpp tests.**
- [ ] **Step 5: Commit** `feat: GPU option on GPR + parallel cell-line sweep harness`.

---

## Task 4: Dual / selectivity dataset (efficacy–toxicity)

**Files:** Create `src/geneal/data/dual.py`; Test `tests/test_dual.py`.

**Context:** the dual case selects knockouts lethal in cancer line A but safe in normal-proxy line B. `target = lethality_A − lethality_B = (−effect_A) − (−effect_B) = effect_B − effect_A`. Higher = more selectively lethal to A. Genes need an effect in BOTH lines and an embedding. The existing single-objective stack then runs unchanged on this Dataset; recall@k, max-value, diversity, α-NDCG all apply to the differential target. We also keep per-line raw lethality around for the 2D scatter (returned alongside).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dual.py
import numpy as np
import pandas as pd
from geneal.data.dual import build_differential_dataset


def _toy():
    ge = pd.DataFrame(
        {"ACH-A": [-3.0, 0.0, -1.0], "ACH-B": [0.0, -2.0, -1.0]},
        index=pd.Index(["TP53 (7157)", "A1BG (1)", "EGFR (1956)"], name="gene"))
    ge.columns.name = "cell_line"
    emb = pd.DataFrame(np.eye(3), index=pd.Index([7157, 1, 1956], name="entrez"))
    return ge, emb


def test_differential_target_is_lethalityA_minus_lethalityB():
    ge, emb = _toy()
    ds, aux = build_differential_dataset(ge, emb, line_a="ACH-A", line_b="ACH-B")
    # lethality = -effect; differential = (-effA) - (-effB) = effB - effA
    # TP53: effA -3, effB 0 -> lethalityA 3, lethalityB 0 -> diff +3 (selective to A)
    i = ds.gene_names.index("TP53 (7157)")
    assert np.isclose(ds.target[i], 3.0)
    # A1BG: effA 0, effB -2 -> lethalityA 0, lethalityB 2 -> diff -2 (toxic to B)
    j = ds.gene_names.index("A1BG (1)")
    assert np.isclose(ds.target[j], -2.0)
    # aux carries per-line lethality for the 2D scatter
    assert "lethality_a" in aux and "lethality_b" in aux
    assert len(aux["lethality_a"]) == ds.n_genes


def test_dual_drops_genes_missing_either_line_or_embedding():
    ge, emb = _toy()
    ge.loc["EGFR (1956)", "ACH-B"] = np.nan  # missing in B -> dropped
    ds, aux = build_differential_dataset(ge, emb, line_a="ACH-A", line_b="ACH-B")
    assert "EGFR (1956)" not in ds.gene_names
    assert ds.n_genes == 2
```

- [ ] **Step 2: Run, confirm fail.**

- [ ] **Step 3: Implement `src/geneal/data/dual.py`**

```python
# src/geneal/data/dual.py
from __future__ import annotations
import numpy as np
import pandas as pd
from geneal.data.dataset import Dataset
from geneal.data.depmap import parse_entrez


def build_differential_dataset(gene_effect: pd.DataFrame, embeddings: pd.DataFrame,
                               line_a: str, line_b: str):
    """Selectivity (efficacy-toxicity) dataset: target = lethality_A - lethality_B.

    lethality = -(Chronos effect). target high => lethal in A, safe in B. Genes
    need a non-NaN effect in BOTH lines and an embedding (by Entrez id).
    Returns (Dataset, aux) where aux has per-gene 'lethality_a'/'lethality_b'
    arrays (aligned to ds.gene_names) for the 2D scatter visualization.
    """
    for ln in (line_a, line_b):
        if ln not in gene_effect.columns:
            raise KeyError(f"cell line {ln!r} not in gene-effect matrix")
    sub = gene_effect[[line_a, line_b]].dropna()
    emb_index = set(embeddings.index)
    rows, target, names, leth_a, leth_b = [], [], [], [], []
    for label, r in sub.iterrows():
        ent = parse_entrez(label)
        if ent not in emb_index:
            continue
        la = -float(r[line_a]); lb = -float(r[line_b])
        rows.append(embeddings.loc[ent].to_numpy(dtype=float))
        target.append(la - lb); names.append(label)
        leth_a.append(la); leth_b.append(lb)
    if not names:
        raise ValueError("no genes with effects in both lines and an embedding")
    ds = Dataset(embeddings=np.vstack(rows), target=np.asarray(target),
                 gene_names=names)
    aux = {"lethality_a": np.asarray(leth_a), "lethality_b": np.asarray(leth_b),
           "line_a": line_a, "line_b": line_b}
    return ds, aux
```

- [ ] **Step 4: Run, confirm 2 passed.**
- [ ] **Step 5: Commit** `feat: add dual-cell-line selectivity (efficacy-toxicity) dataset`.

---

## Task 5: Elegant report — error-bar stage plots, per-metric stage tables, dual scatter

**Files:** Create `src/geneal/report/report2.py`, `src/geneal/report/dual_plot.py`; Test `tests/test_report2.py`.

**Design requirements (from the user):** pretty/elegant; plots clearly show the STAGE (round) on the x-axis; use **error bars (95% CI), NOT shaded envelopes**; one curve per method per metric. For EACH metric, a **stage table**: rows = methods, columns = rounds, each cell = `mean ± ci`. Dual case: a 2D scatter (lethality_A vs lethality_B) with the true selective-top-k highlighted and each method's selected genes marked.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_report2.py
import numpy as np
import pandas as pd
from geneal.report.report2 import (metric_columns, stage_table, build_report2)
from geneal.report.dual_plot import dual_scatter_div


def _df():
    rows = []
    for method in ("kdpp", "greedy"):
        for seed in (0, 1):
            for rnd in range(3):
                rows.append({"method": method, "seed": seed, "round": rnd,
                             "metric": 0.1 * rnd, "metric_name": "recall@50",
                             "eval_max_value": 0.2 * rnd + 0.1,
                             "eval_diversity": 1.0 + rnd,
                             "eval_alpha_ndcg@50": 0.3 * rnd})
    return pd.DataFrame(rows)


def test_metric_columns_discovers_eval_and_recall():
    cols = metric_columns(_df())
    assert "metric" in cols  # recall
    assert "eval_max_value" in cols and "eval_alpha_ndcg@50" in cols


def test_stage_table_model_rows_round_cols_mean_ci():
    t = stage_table(_df(), "eval_max_value")
    # rows = methods, columns = rounds
    assert set(t.index) == {"kdpp", "greedy"}
    assert list(t.columns) == [0, 1, 2]
    # cells are 'mean ± ci' strings
    assert "±" in t.loc["kdpp", 0]


def test_build_report2_writes_rich_html(tmp_path):
    out = tmp_path / "report.html"
    build_report2(_df(), out, title="geneal single-objective")
    html = out.read_text()
    assert "<html" in html.lower()
    assert "recall@50" in html
    assert "max_value" in html and "alpha_ndcg@50" in html and "diversity" in html
    # stage tables present (one per metric) and error-bar plots embedded
    assert "±" in html
    assert "plotly" in html.lower() or "<div" in html.lower()


def test_dual_scatter_div_is_html():
    rng = np.random.default_rng(0)
    aux = {"lethality_a": rng.standard_normal(50), "lethality_b": rng.standard_normal(50),
           "line_a": "ACH-A", "line_b": "ACH-B"}
    target = aux["lethality_a"] - aux["lethality_b"]
    selected = {"kdpp": [0, 1, 2, 3], "greedy": [4, 5, 6, 7]}
    div = dual_scatter_div(aux, target, top_k=10, selected_by_method=selected)
    assert isinstance(div, str) and ("<div" in div.lower() or "plotly" in div.lower())
```

- [ ] **Step 2: Run, confirm fail.**

- [ ] **Step 3: Implement `src/geneal/report/report2.py`**

```python
# src/geneal/report/report2.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from jinja2 import Environment, BaseLoader

_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>{{ title }}</title>
<style>
  :root { --fg:#1a1a2e; --muted:#6b7280; --line:#e5e7eb; --accent:#3b4cca; }
  body { font-family: 'Inter', system-ui, sans-serif; color: var(--fg);
         margin: 0; padding: 2.5rem 3rem; background: #fafafb; line-height: 1.5; }
  h1 { font-weight: 700; letter-spacing: -0.02em; margin-bottom: .25rem; }
  .sub { color: var(--muted); margin-bottom: 2rem; }
  h2 { margin-top: 2.5rem; font-weight: 650; border-bottom: 2px solid var(--accent);
       display: inline-block; padding-bottom: 2px; }
  .card { background:#fff; border:1px solid var(--line); border-radius:14px;
          padding:1.25rem 1.5rem; margin:1rem 0; box-shadow:0 1px 3px rgba(0,0,0,.04); }
  table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
  th, td { border-bottom: 1px solid var(--line); padding: 7px 12px; text-align: right; }
  th:first-child, td:first-child { text-align: left; font-weight: 600; }
  thead th { color: var(--muted); font-weight: 600; border-bottom: 2px solid var(--line); }
  caption { text-align:left; font-weight:650; margin-bottom:.5rem; color:var(--fg); }
  .metric-note { color: var(--muted); font-size: .9rem; margin:.2rem 0 .8rem; }
</style></head><body>
  <h1>{{ title }}</h1>
  <div class="sub">{{ methods|length }} methods · {{ n_seeds }} seeds · {{ n_rounds }} rounds
       · metrics: {{ metric_labels|join(', ') }}</div>
  {% if dual_div %}<h2>Selectivity landscape</h2><div class="card">{{ dual_div|safe }}</div>{% endif %}
  {% for m in metrics %}
    <h2>{{ m.label }}</h2>
    <div class="metric-note">{{ m.desc }}</div>
    <div class="card">{{ m.plot|safe }}</div>
    <div class="card"><table>
      <caption>{{ m.label }} by stage — mean ± 95% CI (rows: model, cols: round)</caption>
      <thead><tr><th>model</th>{% for c in m.cols %}<th>{{ c }}</th>{% endfor %}</tr></thead>
      <tbody>{% for r in m.rows %}<tr><td>{{ r.name }}</td>
        {% for v in r.cells %}<td>{{ v }}</td>{% endfor %}</tr>{% endfor %}</tbody>
    </table></div>
  {% endfor %}
</body></html>"""

_METRIC_DESC = {
    "metric": "Fraction of the true top-k most-lethal genes recovered (recall@k).",
    "eval_max_value": "Best true lethality found so far, normalized by the global max.",
    "eval_diversity": "Mean pairwise embedding distance of the cumulative revealed set.",
}


def metric_columns(df: pd.DataFrame) -> list[str]:
    cols = ["metric"] + sorted(c for c in df.columns if c.startswith("eval_"))
    return [c for c in cols if c in df.columns]


def _label(col: str, df: pd.DataFrame) -> str:
    if col == "metric":
        return str(df["metric_name"].iloc[0]) if "metric_name" in df else "recall"
    return col[len("eval_"):]


def _mean_ci(series: pd.Series) -> tuple[float, float]:
    x = series.to_numpy(dtype=float)
    n = len(x)
    mean = float(np.mean(x))
    if n < 2:
        return mean, 0.0
    se = float(np.std(x, ddof=1)) / np.sqrt(n)
    return mean, 1.96 * se


def stage_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    methods = sorted(df["method"].unique())
    rounds = sorted(df["round"].unique())
    data = {}
    for r in rounds:
        cells = []
        for m in methods:
            sub = df[(df["method"] == m) & (df["round"] == r)][col]
            mean, ci = _mean_ci(sub)
            cells.append(f"{mean:.3f} ± {ci:.3f}")
        data[r] = cells
    return pd.DataFrame(data, index=methods)


def _error_bar_plot(df: pd.DataFrame, col: str, label: str) -> str:
    fig = go.Figure()
    rounds = sorted(df["round"].unique())
    for m in sorted(df["method"].unique()):
        means, cis = [], []
        for r in rounds:
            mean, ci = _mean_ci(df[(df["method"] == m) & (df["round"] == r)][col])
            means.append(mean); cis.append(ci)
        fig.add_trace(go.Scatter(
            x=rounds, y=means, mode="lines+markers", name=m,
            error_y=dict(type="data", array=cis, visible=True, thickness=1.2, width=3)))
    fig.update_layout(template="simple_white", xaxis_title="stage (round)",
                      yaxis_title=label, legend_title="model",
                      margin=dict(l=60, r=20, t=10, b=50), height=420)
    fig.update_xaxes(dtick=1)
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def build_report2(df: pd.DataFrame, out_path, title: str = "geneal report",
                  dual_div: str | None = None) -> Path:
    cols = metric_columns(df)
    metrics = []
    for c in cols:
        label = _label(c, df)
        t = stage_table(df, c)
        metrics.append({
            "label": label,
            "desc": _METRIC_DESC.get(c, "alpha-NDCG: quality-weighted, redundancy-discounted coverage." if "alpha" in c else ""),
            "plot": _error_bar_plot(df, c, label),
            "cols": list(t.columns),
            "rows": [{"name": idx, "cells": list(t.loc[idx])} for idx in t.index],
        })
    env = Environment(loader=BaseLoader())
    html = env.from_string(_TEMPLATE).render(
        title=title, metrics=metrics,
        metric_labels=[m["label"] for m in metrics],
        methods=sorted(df["method"].unique()),
        n_seeds=df["seed"].nunique(), n_rounds=df["round"].nunique(),
        dual_div=dual_div)
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    return out_path
```

- [ ] **Step 4: Implement `src/geneal/report/dual_plot.py`**

```python
# src/geneal/report/dual_plot.py
from __future__ import annotations
import numpy as np
import plotly.graph_objects as go


def dual_scatter_div(aux: dict, target: np.ndarray, top_k: int,
                     selected_by_method: dict[str, list[int]]) -> str:
    """2D scatter of lethality_A (x) vs lethality_B (y). The true selective
    top-k (highest target = lethality_A - lethality_B) is highlighted; each
    method's selected genes are overlaid as distinct markers."""
    la = np.asarray(aux["lethality_a"]); lb = np.asarray(aux["lethality_b"])
    target = np.asarray(target)
    k = min(top_k, len(target))
    true_top = set(np.argsort(target)[::-1][:k].tolist())
    is_top = np.array([i in true_top for i in range(len(target))])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=la[~is_top], y=lb[~is_top], mode="markers", name="genes",
        marker=dict(size=5, color="#cbd5e1"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=la[is_top], y=lb[is_top], mode="markers",
        name=f"true selective top-{k}",
        marker=dict(size=9, color="#ef4444", symbol="star",
                    line=dict(width=0.5, color="#7f1d1d"))))
    symbols = ["circle-open", "diamond-open", "square-open", "x", "cross"]
    for s, (mname, idx) in zip(symbols, selected_by_method.items()):
        idx = list(idx)
        fig.add_trace(go.Scatter(
            x=la[idx], y=lb[idx], mode="markers", name=f"selected: {mname}",
            marker=dict(size=12, symbol=s, line=dict(width=1.5))))
    # diagonal: equally lethal in both (no selectivity)
    lo = float(min(la.min(), lb.min())); hi = float(max(la.max(), lb.max()))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                             line=dict(dash="dot", color="#9ca3af"),
                             name="no selectivity", hoverinfo="skip"))
    fig.update_layout(template="simple_white",
                      xaxis_title=f"lethality in {aux.get('line_a','A')} (efficacy)",
                      yaxis_title=f"lethality in {aux.get('line_b','B')} (toxicity)",
                      legend_title="", height=560,
                      margin=dict(l=60, r=20, t=10, b=50))
    return fig.to_html(full_html=False, include_plotlyjs="cdn")
```

- [ ] **Step 5: Run, confirm 4 passed.**
- [ ] **Step 6: Commit** `feat: elegant report v2 (error-bar stage plots, per-metric stage tables) + dual scatter`.

---

## Task 6: Run the experiments (single panel + dual) and produce reports

**Files:** Create `scripts/run_single_sweep.py`, `scripts/run_dual_experiment.py`. This task produces the actual results + reports; capture them honestly.

**Defaults (all CLI-tunable):** 4 cell lines, 2 seeds, k=50, rounds=10, batch=10, n_initial=50, alpha=0.5, n_clusters=20, all 6 methods. Reuse the existing `data/processed/embeddings/esm2_small.parquet` + `gene_effect.parquet`.

- [ ] **Step 1: Write `scripts/run_single_sweep.py`** — selects the 4 lowest-NaN cell lines among embedded genes; defines a top-level `build_experiment(cell_line)` factory (picklable) building the 6-method Experiment with the eval panel `[MaxValue(), CumulativeDiversity(), AlphaNDCG(k, alpha, n_clusters)]`; calls `run_sweep(build_experiment, lines, seeds, eval_panel, max_workers)`; writes the combined parquet + `build_report2(df, ..., title=...)` per the aggregate (across cell lines) AND optionally per cell line. Print, per metric, the final-round mean±CI per method. All knobs via argparse with the defaults above.

- [ ] **Step 2: Run it.**
```bash
micromamba run -n geneal python scripts/run_single_sweep.py \
  --n-cell-lines 4 --seeds 0 1 --k 50 --rounds 10 --batch 10 --n-initial 50 \
  --alpha 0.5 --n-clusters 20 --out-root res/runs_single_panel
```
Report the final-round per-metric mean±CI table and the report path.

- [ ] **Step 3: Write `scripts/run_dual_experiment.py`** — picks a cancer line A and a contrasting line B (default: two of the chosen lines from different lineages; expose `--line-a/--line-b`); builds the differential Dataset + aux via `build_differential_dataset`; runs all 6 methods over 2 seeds with the eval panel on the differential target; records, per method, the set of genes selected by the final round (for the scatter — use seed 0's revealed set); builds the report with `build_report2(..., dual_div=dual_scatter_div(aux, ds.target, top_k=k, selected_by_method=...))`.

- [ ] **Step 4: Run it.**
```bash
micromamba run -n geneal python scripts/run_dual_experiment.py \
  --line-a <ACH_cancer> --line-b <ACH_contrast> --k 50 --rounds 10 --batch 10 \
  --n-initial 50 --seeds 0 1 --out-root res/runs_dual
```
Report the differential-target per-metric final mean±CI per method + the report path.

- [ ] **Step 5: Write `data/REAL_RESULT_PANEL.md`** capturing, HONESTLY: the single-objective panel results (all 4 metrics, final-round mean±CI per method) and the dual/selectivity results, with a one-paragraph read per case. Specifically address: does any method (esp. k-DPP) win on **α-NDCG or diversity** even where it ties/loses on recall? does the dual/selectivity objective change the method ranking vs the single objective? Do NOT tune to favor any method.

- [ ] **Step 6: Commit** scripts + result note (artifacts gitignored): `experiment: single-panel + dual selectivity results with metric panel and v2 report`.

---

## Self-Review (completed during planning)

- **Spec coverage (user's asks):** keep recall@k ✓ (Task 2 `metric`); max-lethal-value ✓ (Task 1 MaxValue); diversity-alone ✓ (Task 1 CumulativeDiversity, cumulative revealed set); α-NDCG with embedding-cluster nuggets ✓ (Task 1 AlphaNDCG); report heavily revised + elegant ✓ (Task 5 report2, styled); stage clearly shown ✓ (x-axis "stage (round)", dtick=1); error bars not envelopes ✓ (`error_y` not fill); per-metric stage tables, model rows × round cols, mean±CI ✓ (Task 5 stage_table); dual case + visualization ✓ (Tasks 4+5 dual_plot 2D scatter); 4 lines × 2 seeds, tunable ✓ (Task 6 argparse defaults); parallelize ✓ (Task 3 sweep) + GPU ✓ (Task 3 device, with honest caveat that CPU-parallel is the real win at this scale).
- **Acquisition unchanged:** the eval panel is purely for recording/reporting; acquisition still uses surrogate+UCB. α-NDCG/diversity do NOT leak into selection. (Confirm in Task 2 — panel is computed in `_record` only.)
- **α-NDCG ranking:** uses the cumulative revealed list in ACQUISITION order (round order; within-round in the selection's returned order), which is the meaningful ranking for an AL method. Documented in panel.py.
- **Type/interface consistency:** panel metrics share `evaluate(revealed_order, target, embeddings)`; recall@k keeps its own 2-arg signature (it's the Objective, recorded as `metric`, not part of the panel) — these are deliberately different and not mixed.
- **No placeholders:** every code step is complete. GPU path is wired but explicitly not claimed as a speedup at 500 genes.

## Out of scope
Full-genome (18k) embeddings + 650M ESM2 (Plan 2 Task 4, separate). Significance testing beyond 95% CI. scPRINT embeddings.
```
