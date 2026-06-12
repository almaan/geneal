# geneal k-DPP Batch Acquisition Method — Implementation Plan (Plan 1.5 of 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the core methodological contribution — a quality-weighted k-DPP batch acquisition — plus the diversity-only baselines it must beat (CoreSet, TypiClust), and a quality–diversity Pareto evaluation. All on synthetic data, producing the guaranteed "Figure 1" for the MLCB submission.

**Architecture:** Add a joint predictive covariance to the Surrogate interface (`predict_cov`). Add new Selection strategies behind the existing `Selection` protocol: `KDPP` (the method), `CoreSet` and `TypiClust` (IterPert-style diversity-only baselines). Add a Pareto/diagnostics evaluation that measures, per round and per method, recall@k + batch quality + batch diversity. The Runner and Report from v1 are reused; the only Runner change is passing the surrogate (which now exposes `predict_cov`) — selection strategies that need the joint covariance call it themselves.

**Tech Stack:** Existing geneal stack (numpy, scipy, gpytorch, pyro, pandas, plotly, pytest). No new dependencies. The k-DPP greedy-MAP and sampling use numpy linear algebra only.

**The moat this builds (from the design spec §1):** quality×diversity coupling for top-k *extreme recovery*. The 2×2: NAIAD = quality-only (greedy top-q), IterPert = diversity-only (CoreSet/TypiClust), geneal = both (k-DPP). The headline empirical claim is Pareto-dominance of k-DPP over both failure modes on recall@k. This plan produces exactly that comparison on synthetic data.

**Conventions (carried from v1):** target = higher-is-better; all acquisitions maximize; RNG is an explicit `numpy.random.Generator`; arrays are numpy. The `Selection.select` signature already carries `surrogate`, `X_train`, `y_train` (added during v1 code review).

---

## File Structure

```
src/geneal/interfaces.py                 # MODIFY: add predict_cov to Surrogate protocol
src/geneal/models/surrogate.py           # MODIFY: GPRSurrogate.predict_cov, BNNSurrogate.predict_cov
src/geneal/models/kernels.py             # CREATE: posterior_correlation(cov) helper
src/geneal/models/selection.py           # MODIFY: add KDPP, CoreSet, TypiClust
src/geneal/metrics/diagnostics.py        # CREATE: batch_quality, batch_diversity
src/geneal/report/pareto.py              # CREATE: quality-diversity Pareto aggregation + plot
conf/method/al_kdpp.yaml                 # CREATE
conf/method/baseline_coreset.yaml        # CREATE
conf/method/baseline_typiclust.yaml      # CREATE
tests/test_predict_cov.py                # CREATE
tests/test_kernels.py                    # CREATE
tests/test_kdpp.py                       # CREATE
tests/test_coreset_typiclust.py          # CREATE
tests/test_diagnostics.py                # CREATE
tests/test_pareto.py                     # CREATE
tests/test_kdpp_beats_baselines.py       # CREATE: the headline empirical assertion (synthetic)
```

---

## Task 1: Joint predictive covariance on the Surrogate interface

**Files:**
- Modify: `src/geneal/interfaces.py`
- Modify: `src/geneal/models/surrogate.py`
- Test: `tests/test_predict_cov.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_predict_cov.py
import numpy as np
from geneal.data.dataset import make_synthetic
from geneal.models.surrogate import GPRSurrogate, BNNSurrogate


def _xy():
    ds = make_synthetic(n_genes=120, dim=5, seed=2, noise_sd=0.05)
    return ds.embeddings[:80], ds.target[:80], ds.embeddings[80:]


def test_gpr_predict_cov_shapes_and_psd():
    Xtr, ytr, Xte = _xy()
    surr = GPRSurrogate(n_iters=80).fit(Xtr, ytr)
    mean, cov = surr.predict_cov(Xte)
    n = len(Xte)
    assert mean.shape == (n,)
    assert cov.shape == (n, n)
    # symmetric
    assert np.allclose(cov, cov.T, atol=1e-4)
    # diagonal matches marginal variance from predict (std**2), loosely
    _, std = surr.predict(Xte)
    assert np.allclose(np.diag(cov), std**2, rtol=0.2, atol=1e-3)
    # PSD up to small numerical noise
    eig = np.linalg.eigvalsh((cov + cov.T) / 2)
    assert eig.min() > -1e-4


def test_bnn_predict_cov_shapes():
    Xtr, ytr, Xte = _xy()
    surr = BNNSurrogate(hidden=16, n_steps=150, seed=0).fit(Xtr, ytr)
    mean, cov = surr.predict_cov(Xte)
    n = len(Xte)
    assert mean.shape == (n,)
    assert cov.shape == (n, n)
    assert np.allclose(cov, cov.T, atol=1e-4)
```

- [ ] **Step 2: Run to confirm failure**

Run: `micromamba run -n geneal pytest tests/test_predict_cov.py -v`
Expected: FAIL — `AttributeError: 'GPRSurrogate' object has no attribute 'predict_cov'`

- [ ] **Step 3: Add `predict_cov` to the Surrogate protocol in `interfaces.py`**

Add this method to the `Surrogate` Protocol (after `predict`):

```python
    def predict_cov(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...  # (mean, full covariance)
```

So the `Surrogate` Protocol body reads:

```python
@runtime_checkable
class Surrogate(Protocol):
    """Maps gene embeddings -> predicted target with uncertainty."""
    def fit(self, X: np.ndarray, y: np.ndarray) -> "Surrogate": ...
    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...  # (mean, std)
    def predict_cov(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...  # (mean, full covariance)
    def clone(self) -> "Surrogate": ...
```

- [ ] **Step 4: Implement `GPRSurrogate.predict_cov`**

In `src/geneal/models/surrogate.py`, add this method to `GPRSurrogate` (right after `predict`). It returns the full joint posterior covariance on the original target scale (variance scales by `_y_std**2`):

```python
    def predict_cov(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import torch
        import gpytorch
        if self._model is None:
            raise RuntimeError("GPRSurrogate.predict_cov called before fit")
        X = np.asarray(X, dtype=np.float64)
        tx = torch.tensor((X - self._x_mean) / self._x_std, dtype=torch.float32)
        self._model.eval()
        self._likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            post = self._likelihood(self._model(tx))
            mean = post.mean.numpy()
            cov = post.covariance_matrix.numpy()
        # de-standardize: target = z * y_std + y_mean  ->  cov scales by y_std**2
        return mean * self._y_std + self._y_mean, cov * (self._y_std ** 2)
```

- [ ] **Step 5: Implement `BNNSurrogate.predict_cov`**

Add this method to `BNNSurrogate` (after `predict`). It reuses the posterior predictive draws and takes their sample covariance:

```python
    def predict_cov(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import torch
        import pyro
        from pyro.infer import Predictive
        if self._guide is None:
            raise RuntimeError("BNNSurrogate.predict_cov called before fit")
        X = np.asarray(X, dtype=np.float64)
        xt = torch.tensor((X - self._x_mean) / self._x_std, dtype=torch.float32)
        pred = Predictive(self._model, guide=self._guide,
                          num_samples=self.n_predict, return_sites=["obs"])
        with torch.no_grad():
            obs = pred(xt)["obs"].detach().numpy()  # (n_predict, n), standardized
        mean = obs.mean(0) * self._y_std + self._y_mean
        cov = np.cov(obs, rowvar=False) * (self._y_std ** 2)  # (n, n)
        return mean, np.atleast_2d(cov)
```

- [ ] **Step 6: Run to confirm pass**

Run: `micromamba run -n geneal pytest tests/test_predict_cov.py -v`
Expected: PASS (2 passed). May take ~10-20s.

- [ ] **Step 7: Confirm no regression on existing surrogate tests**

Run: `micromamba run -n geneal pytest tests/test_surrogate.py -v`
Expected: PASS (3 passed).

- [ ] **Step 8: Commit**

```bash
git add src/geneal/interfaces.py src/geneal/models/surrogate.py tests/test_predict_cov.py
git commit -m "feat: add joint predictive covariance (predict_cov) to surrogates"
```

---

## Task 2: Posterior correlation kernel helper

**Files:**
- Create: `src/geneal/models/kernels.py`
- Test: `tests/test_kernels.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_kernels.py
import numpy as np
from geneal.models.kernels import posterior_correlation


def test_correlation_diag_is_one():
    cov = np.array([[4.0, 1.0], [1.0, 9.0]])
    corr = posterior_correlation(cov)
    assert np.allclose(np.diag(corr), 1.0)


def test_correlation_values():
    cov = np.array([[4.0, 2.0], [2.0, 9.0]])
    corr = posterior_correlation(cov)
    # off-diagonal = 2 / (2*3) = 1/3
    assert np.isclose(corr[0, 1], 1.0 / 3.0)
    assert np.allclose(corr, corr.T)


def test_zero_variance_safe():
    # a candidate with ~0 posterior variance must not produce nan/inf
    cov = np.array([[0.0, 0.0], [0.0, 4.0]])
    corr = posterior_correlation(cov)
    assert np.all(np.isfinite(corr))
    assert np.isclose(corr[1, 1], 1.0)
```

- [ ] **Step 2: Run to confirm failure**

Run: `micromamba run -n geneal pytest tests/test_kernels.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'geneal.models.kernels'`

- [ ] **Step 3: Implement `src/geneal/models/kernels.py`**

```python
# src/geneal/models/kernels.py
from __future__ import annotations
import numpy as np


def posterior_correlation(cov: np.ndarray) -> np.ndarray:
    """Correlation matrix from a covariance matrix.

    S_ij = cov_ij / (sqrt(cov_ii) * sqrt(cov_jj)). Zero-variance rows/cols are
    treated as self-correlation 1 and zero cross-correlation (they carry no
    diversity information). Output is symmetric with unit diagonal.
    """
    cov = np.asarray(cov, dtype=float)
    d = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    denom = np.outer(d, d)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(denom > 0, cov / denom, 0.0)
    n = cov.shape[0]
    corr[np.diag_indices(n)] = 1.0
    # clip numerical drift into [-1, 1]
    return np.clip((corr + corr.T) / 2, -1.0, 1.0)
```

- [ ] **Step 4: Run to confirm pass**

Run: `micromamba run -n geneal pytest tests/test_kernels.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/geneal/models/kernels.py tests/test_kernels.py
git commit -m "feat: add posterior_correlation kernel helper"
```

---

## Task 3: k-DPP selection strategy (the method)

**Files:**
- Modify: `src/geneal/models/selection.py`
- Test: `tests/test_kdpp.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_kdpp.py
import numpy as np
from geneal.models.selection import KDPP
from geneal.models.acquisition import GreedyMean, UCB
from geneal.data.dataset import make_synthetic
from geneal.models.surrogate import GPRSurrogate


def test_kdpp_returns_q_distinct_absolute_idx():
    ds = make_synthetic(n_genes=40, dim=4, seed=1)
    Xtr, ytr = ds.embeddings[:10], ds.target[:10]
    surr = GPRSurrogate(n_iters=60).fit(Xtr, ytr)
    cand = list(range(10, 40))
    Xc = ds.embeddings[cand]
    mean, std = surr.predict(Xc)
    sel = KDPP(solver="greedy").select(
        candidate_idx=cand, X_candidates=Xc, mean=mean, std=std,
        best=float(ytr.max()), q=5, rng=np.random.default_rng(0),
        surrogate=surr, acquisition=UCB(beta=1.0), X_train=Xtr, y_train=ytr)
    assert len(sel) == 5
    assert len(set(sel)) == 5
    assert set(sel).issubset(set(cand))


def test_kdpp_avoids_duplicate_directions():
    # Two candidates with identical embeddings (perfectly correlated outcomes)
    # and high quality: the DPP must NOT pick both before picking a distinct one.
    # Build candidates: idx0 and idx1 identical; idx2 distinct. q=2 must include idx2.
    rng = np.random.default_rng(0)
    base = rng.standard_normal((1, 4))
    Xc = np.vstack([base, base, rng.standard_normal((1, 4))])  # 0,1 identical; 2 distinct
    ds_x = np.vstack([Xc, rng.standard_normal((8, 4))])
    ds_y = ds_x @ rng.standard_normal(4)
    surr = GPRSurrogate(n_iters=60).fit(ds_x[3:], ds_y[3:])
    cand = [100, 101, 102]
    mean, std = surr.predict(Xc)
    # force high, near-equal quality so diversity is the deciding factor
    sel = KDPP(solver="greedy").select(
        candidate_idx=cand, X_candidates=Xc, mean=mean, std=std,
        best=0.0, q=2, rng=np.random.default_rng(0),
        surrogate=surr, acquisition=GreedyMean(), X_train=ds_x[3:], y_train=ds_y[3:])
    assert 102 in sel  # the distinct candidate must be chosen over the duplicate pair
    assert not (100 in sel and 101 in sel)  # never both identical ones


def test_kdpp_sampling_solver_runs_and_is_seeded():
    ds = make_synthetic(n_genes=30, dim=4, seed=3)
    Xtr, ytr = ds.embeddings[:8], ds.target[:8]
    surr = GPRSurrogate(n_iters=50).fit(Xtr, ytr)
    cand = list(range(8, 30)); Xc = ds.embeddings[cand]
    mean, std = surr.predict(Xc)
    kw = dict(candidate_idx=cand, X_candidates=Xc, mean=mean, std=std, best=0.0,
              q=4, surrogate=surr, acquisition=UCB(1.0), X_train=Xtr, y_train=ytr)
    a = KDPP(solver="sampling").select(rng=np.random.default_rng(7), **kw)
    b = KDPP(solver="sampling").select(rng=np.random.default_rng(7), **kw)
    assert len(a) == 4 and len(set(a)) == 4
    assert a == b  # same seed -> same sample
```

- [ ] **Step 2: Run to confirm failure**

Run: `micromamba run -n geneal pytest tests/test_kdpp.py -v`
Expected: FAIL — `ImportError: cannot import name 'KDPP'`

- [ ] **Step 3: Implement `KDPP` in `src/geneal/models/selection.py`**

Add at the top of the file (with the existing `import numpy as np`):

```python
from geneal.models.kernels import posterior_correlation
```

Append this class to `src/geneal/models/selection.py`:

```python
class KDPP:
    """Quality-weighted k-DPP batch acquisition (the geneal method).

    Builds an L-ensemble kernel over candidates:  L = diag(q) S diag(q),
    where q_i = acquisition score (quality) and S_ij = surrogate posterior
    correlation (outcome similarity). Selecting the size-q subset that maximizes
    det(L_B) yields a batch that is simultaneously high-quality (large diagonal)
    and non-redundant (near-orthogonal rows). Because
    log det(posterior cov of a batch) is the Gaussian joint entropy, max-det is
    an information-theoretic batch acquisition with outcome-diversity built in.

    solver:
      "greedy"   -> greedy MAP: repeatedly add the candidate with the largest
                    marginal gain in log det(L_B). Deterministic, near-optimal.
      "sampling" -> exact k-DPP sampling via the eigendecomposition of L.
    """

    def __init__(self, solver: str = "greedy", jitter: float = 1e-9) -> None:
        if solver not in ("greedy", "sampling"):
            raise ValueError("solver must be 'greedy' or 'sampling'")
        self.solver = solver
        self.jitter = jitter

    def _build_L(self, *, mean, std, best, rng, surrogate, acquisition,
                 X_candidates):
        # quality term q_i from the acquisition (non-negative; shift if needed)
        q = np.asarray(acquisition.score(mean, std, best, rng), dtype=float)
        q = q - q.min() + 1e-6 if q.min() < 0 else q + 1e-6
        # outcome-similarity term S from the surrogate's joint posterior covariance
        _, cov = surrogate.predict_cov(np.asarray(X_candidates))
        S = posterior_correlation(cov)
        L = (q[:, None] * S) * q[None, :]
        # symmetrize + jitter for numerical PSD
        L = (L + L.T) / 2 + self.jitter * np.eye(L.shape[0])
        return L

    def select(self, *, candidate_idx, X_candidates, mean, std, best, q, rng,
               surrogate, acquisition, X_train, y_train) -> list[int]:
        candidate_idx = list(candidate_idx)
        n = len(candidate_idx)
        q = min(q, n)
        L = self._build_L(mean=mean, std=std, best=best, rng=rng,
                          surrogate=surrogate, acquisition=acquisition,
                          X_candidates=X_candidates)
        if self.solver == "greedy":
            local = _greedy_map_logdet(L, q)
        else:
            local = _sample_kdpp(L, q, rng)
        return [candidate_idx[i] for i in local]
```

Also append these two module-level helpers to `selection.py`:

```python
def _greedy_map_logdet(L: np.ndarray, k: int) -> list[int]:
    """Greedy MAP for max log det(L_S), |S|=k. O(k^2 n). Cholesky-style update."""
    n = L.shape[0]
    selected: list[int] = []
    # c[j] = squared norm contribution; d2[j] = current marginal gain (det increment)
    d2 = np.diag(L).astype(float).copy()
    c = np.zeros((n, n))
    for _ in range(min(k, n)):
        j = int(np.argmax(np.where(d2 > 0, d2, -np.inf)))
        if d2[j] <= 0:
            # remaining candidates are linearly dependent; pad with best leftover
            leftover = [i for i in range(n) if i not in selected]
            selected.extend(leftover[: k - len(selected)])
            break
        selected.append(j)
        if len(selected) == min(k, n):
            break
        # update marginal gains (incremental Cholesky of L over selected set)
        ci = (L[:, j] - c[:, :len(selected) - 1] @ c[j, :len(selected) - 1]) / np.sqrt(d2[j])
        c[:, len(selected) - 1] = ci
        d2 = d2 - ci ** 2
        d2[selected] = -np.inf
    return selected[:k]


def _sample_kdpp(L: np.ndarray, k: int, rng) -> list[int]:
    """Exact k-DPP sampling via eigendecomposition (Kulesza & Taskar)."""
    vals, vecs = np.linalg.eigh(L)
    vals = np.clip(vals, 0.0, None)
    n = L.shape[0]
    k = min(k, n)
    # elementary symmetric polynomials e[l, m] for k-DPP eigenvalue selection
    E = np.zeros((k + 1, n + 1))
    E[0, :] = 1.0
    for l in range(1, k + 1):
        for m in range(1, n + 1):
            E[l, m] = E[l, m - 1] + vals[m - 1] * E[l - 1, m - 1]
    # select k eigenvectors
    chosen_vecs = []
    l = k
    for m in range(n, 0, -1):
        if l == 0:
            break
        if E[l, m] <= 0:
            continue
        if rng.random() < vals[m - 1] * E[l - 1, m - 1] / E[l, m]:
            chosen_vecs.append(m - 1)
            l -= 1
    V = vecs[:, chosen_vecs]  # (n, k)
    # sample k items from the selected subspace
    selected: list[int] = []
    for _ in range(len(chosen_vecs)):
        probs = (V ** 2).sum(axis=1)
        probs = probs / probs.sum()
        i = int(rng.choice(n, p=probs))
        selected.append(i)
        # project V to the subspace orthogonal to e_i
        if V.shape[1] == 1:
            V = np.zeros((n, 0))
            continue
        col = np.argmax(np.abs(V[i, :]))
        Vj = V[:, col].copy()
        V = np.delete(V, col, axis=1)
        V = V - np.outer(Vj, V[i, :] / Vj[i])
        # re-orthonormalize
        Q, _ = np.linalg.qr(V)
        V = Q
    return sorted(set(selected))[:k] if selected else list(range(k))
```

NOTE on the sampling solver: exact k-DPP sampling is fiddly. The test only requires it to (a) return q distinct indices and (b) be deterministic given the seed. If the orthogonalization above proves numerically unstable in the test, simplify `_sample_kdpp` to a **greedy stochastic** fallback: draw the first item proportional to `diag(L)`, then iteratively draw proportional to the marginal det-gain — still seeded and distinct. The greedy MAP solver is the one used in experiments; sampling is a secondary option. Do not block the task on exact-sampling perfection — if unstable, implement the seeded greedy-stochastic version and note it in the commit.

- [ ] **Step 4: Run to confirm pass**

Run: `micromamba run -n geneal pytest tests/test_kdpp.py -v`
Expected: PASS (3 passed). If `test_kdpp_sampling_solver_runs_and_is_seeded` is unstable, apply the greedy-stochastic fallback described above until it passes.

- [ ] **Step 5: Confirm no regression**

Run: `micromamba run -n geneal pytest tests/test_selection.py -v`
Expected: PASS (4 passed).

- [ ] **Step 6: Commit**

```bash
git add src/geneal/models/selection.py tests/test_kdpp.py
git commit -m "feat: add quality-weighted k-DPP batch selection (greedy MAP + sampling)"
```

---

## Task 4: CoreSet and TypiClust baselines (IterPert-style diversity-only)

**Files:**
- Modify: `src/geneal/models/selection.py`
- Test: `tests/test_coreset_typiclust.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_coreset_typiclust.py
import numpy as np
from geneal.models.selection import CoreSet, TypiClust
from geneal.models.acquisition import GreedyMean
from geneal.data.dataset import make_synthetic
from geneal.models.surrogate import GPRSurrogate


def _setup(n_genes=40, n_init=10, dim=4, seed=1):
    ds = make_synthetic(n_genes=n_genes, dim=dim, seed=seed)
    Xtr, ytr = ds.embeddings[:n_init], ds.target[:n_init]
    surr = GPRSurrogate(n_iters=50).fit(Xtr, ytr)
    cand = list(range(n_init, n_genes))
    Xc = ds.embeddings[cand]
    mean, std = surr.predict(Xc)
    return dict(candidate_idx=cand, X_candidates=Xc, mean=mean, std=std,
               best=float(ytr.max()), surrogate=surr, acquisition=GreedyMean(),
               X_train=Xtr, y_train=ytr)


def test_coreset_returns_q_distinct():
    kw = _setup()
    sel = CoreSet().select(q=5, rng=np.random.default_rng(0), **kw)
    assert len(sel) == 5 and len(set(sel)) == 5
    assert set(sel).issubset(set(kw["candidate_idx"]))


def test_coreset_picks_spread_points():
    # CoreSet (farthest-point) on a line should pick near the extremes first.
    X = np.linspace(0, 1, 11).reshape(-1, 1)
    cand = list(range(11))
    sel = CoreSet().select(
        candidate_idx=cand, X_candidates=X, mean=np.zeros(11), std=np.zeros(11),
        best=0.0, q=2, rng=np.random.default_rng(0), surrogate=None,
        acquisition=GreedyMean(), X_train=np.array([[0.5]]), y_train=np.array([0.0]))
    # the two farthest-spread points are the endpoints 0 and 10
    assert set(sel) == {0, 10} or set(sel) == {10, 0}


def test_typiclust_returns_q_distinct():
    kw = _setup()
    sel = TypiClust().select(q=5, rng=np.random.default_rng(0), **kw)
    assert len(sel) == 5 and len(set(sel)) == 5
    assert set(sel).issubset(set(kw["candidate_idx"]))
```

- [ ] **Step 2: Run to confirm failure**

Run: `micromamba run -n geneal pytest tests/test_coreset_typiclust.py -v`
Expected: FAIL — `ImportError: cannot import name 'CoreSet'`

- [ ] **Step 3: Implement `CoreSet` and `TypiClust` in `selection.py`**

These select in the **embedding/input space** (this is what IterPert's selection rules do — diversity over the representation, quality-free). Append to `selection.py`:

```python
class CoreSet:
    """Greedy farthest-point (k-center) selection in embedding space.

    Diversity-only baseline representing IterPert's "greedy distance
    maximization" selection rule. Quality (acquisition) is ignored — each pick is
    the candidate maximally far (Euclidean, in embedding space) from the already
    chosen set, seeded by the existing training points.
    """
    def select(self, *, candidate_idx, X_candidates, mean, std, best, q, rng,
               surrogate, acquisition, X_train, y_train) -> list[int]:
        candidate_idx = list(candidate_idx)
        Xc = np.asarray(X_candidates, dtype=float)
        n = len(candidate_idx)
        q = min(q, n)
        # distance to the nearest already-selected/training point
        anchors = np.asarray(X_train, dtype=float)
        if anchors.ndim == 1:
            anchors = anchors.reshape(1, -1)
        min_d = np.full(n, np.inf)
        if len(anchors):
            min_d = np.min(np.linalg.norm(Xc[:, None, :] - anchors[None, :, :], axis=2), axis=1)
        chosen: list[int] = []
        for _ in range(q):
            j = int(np.argmax(min_d))
            chosen.append(j)
            min_d = np.minimum(min_d, np.linalg.norm(Xc - Xc[j], axis=1))
            min_d[chosen] = -np.inf
        return [candidate_idx[j] for j in chosen]


class TypiClust:
    """Typicality-in-clusters selection (Hacohen et al.), IterPert's best baseline.

    Cluster candidates (k-means, k=q) in embedding space; from each cluster pick
    the most "typical" point (highest local density = smallest mean distance to
    its K nearest neighbours within the cluster). Diversity-only, quality-free.
    """
    def __init__(self, n_neighbors: int = 5) -> None:
        self.n_neighbors = n_neighbors

    def select(self, *, candidate_idx, X_candidates, mean, std, best, q, rng,
               surrogate, acquisition, X_train, y_train) -> list[int]:
        candidate_idx = list(candidate_idx)
        Xc = np.asarray(X_candidates, dtype=float)
        n = len(candidate_idx)
        q = min(q, n)
        labels = _kmeans_labels(Xc, q, rng)
        chosen: list[int] = []
        for c in range(q):
            members = np.where(labels == c)[0]
            if len(members) == 0:
                continue
            chosen.append(int(members[_most_typical(Xc[members], self.n_neighbors)]))
        # if some clusters were empty, top up with farthest-point picks
        if len(chosen) < q:
            remaining = [i for i in range(n) if i not in chosen]
            chosen.extend(remaining[: q - len(chosen)])
        return [candidate_idx[j] for j in chosen[:q]]
```

Append these helpers to `selection.py`:

```python
def _kmeans_labels(X: np.ndarray, k: int, rng, n_iter: int = 25) -> np.ndarray:
    """Minimal seeded k-means (Lloyd). Returns cluster label per row."""
    n = X.shape[0]
    k = min(k, n)
    centers = X[rng.choice(n, size=k, replace=False)].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(n_iter):
        d = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        new = d.argmin(axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for c in range(k):
            pts = X[labels == c]
            if len(pts):
                centers[c] = pts.mean(0)
    return labels


def _most_typical(X: np.ndarray, n_neighbors: int) -> int:
    """Index of the densest point: smallest mean distance to its K nearest."""
    m = X.shape[0]
    if m == 1:
        return 0
    K = min(n_neighbors, m - 1)
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    knn_mean = np.sort(D, axis=1)[:, :K].mean(axis=1)
    return int(np.argmin(knn_mean))
```

- [ ] **Step 4: Run to confirm pass**

Run: `micromamba run -n geneal pytest tests/test_coreset_typiclust.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/geneal/models/selection.py tests/test_coreset_typiclust.py
git commit -m "feat: add CoreSet and TypiClust diversity-only baselines (IterPert-style)"
```

---

## Task 5: Batch diagnostics (quality + diversity per round)

**Files:**
- Create: `src/geneal/metrics/diagnostics.py`
- Test: `tests/test_diagnostics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_diagnostics.py
import numpy as np
from geneal.metrics.diagnostics import batch_quality, batch_diversity


def test_batch_quality_is_mean_true_effect():
    target = np.array([1.0, 2.0, 3.0, 4.0])
    assert batch_quality([1, 3], target) == np.mean([2.0, 4.0])


def test_batch_diversity_zero_for_identical_rows():
    X = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    assert batch_diversity([0, 1, 2], X) == 0.0


def test_batch_diversity_positive_for_spread():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert batch_diversity([0, 1, 2], X) > 0.0


def test_batch_diversity_singleton_is_zero():
    X = np.array([[0.0, 0.0], [1.0, 0.0]])
    assert batch_diversity([0], X) == 0.0
```

- [ ] **Step 2: Run to confirm failure**

Run: `micromamba run -n geneal pytest tests/test_diagnostics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'geneal.metrics.diagnostics'`

- [ ] **Step 3: Implement `src/geneal/metrics/diagnostics.py`**

```python
# src/geneal/metrics/diagnostics.py
from __future__ import annotations
from typing import Sequence
import numpy as np


def batch_quality(batch_idx: Sequence[int], target: np.ndarray) -> float:
    """Mean true target (higher = better) of the genes selected in a batch."""
    batch_idx = list(batch_idx)
    if not batch_idx:
        return 0.0
    return float(np.mean(np.asarray(target)[batch_idx]))


def batch_diversity(batch_idx: Sequence[int], embeddings: np.ndarray) -> float:
    """Mean pairwise Euclidean distance among the batch in embedding space.

    A simple, model-free diversity readout for the quality-diversity Pareto plot.
    Zero for a singleton or identical points; larger for a more spread batch.
    """
    batch_idx = list(batch_idx)
    if len(batch_idx) < 2:
        return 0.0
    X = np.asarray(embeddings)[batch_idx]
    n = len(batch_idx)
    total, cnt = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            total += float(np.linalg.norm(X[i] - X[j]))
            cnt += 1
    return total / cnt if cnt else 0.0
```

- [ ] **Step 4: Run to confirm pass**

Run: `micromamba run -n geneal pytest tests/test_diagnostics.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/geneal/metrics/diagnostics.py tests/test_diagnostics.py
git commit -m "feat: add batch quality and diversity diagnostics"
```

---

## Task 6: Record per-batch diagnostics in the Runner

**Files:**
- Modify: `src/geneal/runner/runner.py`
- Test: `tests/test_runner_diagnostics.py`

**Context:** the Runner currently records `metric` per round. For the Pareto plot we also need, per round (r >= 1), the quality and diversity of the *batch acquired that round*. Add `batch_quality` and `batch_diversity` columns. Round 0 (initial set, no acquired batch) gets `NaN` for these.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_runner_diagnostics.py
import numpy as np
import pandas as pd
from geneal.runner.runner import Runner
from geneal.experiment.objects import ObjectiveObject, DesignObject, Method, Experiment
from geneal.metrics.recall import RecallAtK
from geneal.models.acquisition import UCB
from geneal.models.selection import TopQGreedy
from geneal.models.surrogate import GPRSurrogate
from geneal.models.noise import GaussianNoise
from geneal.data.dataset import make_synthetic


def test_runner_records_batch_diagnostics():
    ds = make_synthetic(n_genes=60, dim=5, seed=0, noise_sd=0.05)
    exp = Experiment(
        dataset=ds,
        objective=ObjectiveObject(metric=RecallAtK(k=10), direction="maximize"),
        design=DesignObject(n_rounds=3, batch_size=5, n_initial=10, seed=0),
        noise=GaussianNoise(sigma=0.05),
        methods=[Method("al_ucb", UCB(2.0), TopQGreedy(), GPRSurrogate(n_iters=60))],
    )
    df = Runner().run(exp, seeds=[0])
    assert {"batch_quality", "batch_diversity"}.issubset(df.columns)
    # round 0 has no acquired batch -> NaN
    assert np.isnan(df[df["round"] == 0]["batch_quality"].iloc[0])
    # rounds >= 1 have finite diagnostics
    later = df[df["round"] >= 1]
    assert later["batch_quality"].notna().all()
    assert later["batch_diversity"].notna().all()
```

- [ ] **Step 2: Run to confirm failure**

Run: `micromamba run -n geneal pytest tests/test_runner_diagnostics.py -v`
Expected: FAIL — KeyError / assertion on missing columns.

- [ ] **Step 3: Modify `runner.py`**

Add the import at the top:

```python
from geneal.metrics.diagnostics import batch_quality, batch_diversity
```

Change `_record` to accept optional batch diagnostics, and have the round loop compute them for the acquired batch. Replace the `_record` static method and its call sites:

```python
    @staticmethod
    def _record(method, seed, rnd, revealed, metric, target, ds,
                batch=None) -> dict:
        return {
            "method": method.name,
            "seed": seed,
            "round": rnd,
            "n_revealed": len(revealed),
            "metric": metric.evaluate(revealed, target),
            "metric_name": metric.name,
            "batch_quality": (batch_quality(batch, target)
                              if batch is not None else float("nan")),
            "batch_diversity": (batch_diversity(batch, ds.embeddings)
                                if batch is not None else float("nan")),
        }
```

Update the round-0 record call (no batch):

```python
        out = [self._record(method, seed, 0, revealed, metric, ds.target, ds)]
```

And the in-loop record call (pass the acquired batch `sel`):

```python
            out.append(self._record(method, seed, r, revealed, metric,
                                    ds.target, ds, batch=sel))
```

- [ ] **Step 4: Run to confirm pass**

Run: `micromamba run -n geneal pytest tests/test_runner_diagnostics.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Confirm no regression on the existing runner tests**

Run: `micromamba run -n geneal pytest tests/test_runner.py -v`
Expected: PASS (5 passed) — existing tests only assert `>=` on columns, so the new columns don't break them.

- [ ] **Step 6: Commit**

```bash
git add src/geneal/runner/runner.py tests/test_runner_diagnostics.py
git commit -m "feat: record per-batch quality and diversity diagnostics in Runner"
```

---

## Task 7: Quality–diversity Pareto aggregation + plot

**Files:**
- Create: `src/geneal/report/pareto.py`
- Test: `tests/test_pareto.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pareto.py
import numpy as np
import pandas as pd
from geneal.report.pareto import pareto_summary, pareto_plot_div


def _df():
    # two methods, two seeds, rounds 0..2; round 0 has NaN diagnostics
    rows = []
    for method, (qual, div) in {"kdpp": (3.0, 2.0), "greedy": (3.0, 0.2),
                                "coreset": (0.5, 2.5)}.items():
        for seed in (0, 1):
            for rnd in range(3):
                rows.append({
                    "method": method, "seed": seed, "round": rnd,
                    "metric": 0.1 * rnd + (0.2 if method == "kdpp" else 0.0),
                    "metric_name": "recall@10",
                    "batch_quality": float("nan") if rnd == 0 else qual,
                    "batch_diversity": float("nan") if rnd == 0 else div,
                })
    return pd.DataFrame(rows)


def test_pareto_summary_one_row_per_method():
    s = pareto_summary(_df())
    assert set(s["method"]) == {"kdpp", "greedy", "coreset"}
    assert {"mean_quality", "mean_diversity", "final_recall"}.issubset(s.columns)
    # NaN round-0 excluded from quality/diversity means
    kd = s[s["method"] == "kdpp"].iloc[0]
    assert np.isclose(kd["mean_quality"], 3.0)
    assert np.isclose(kd["mean_diversity"], 2.0)


def test_pareto_plot_div_is_html():
    div = pareto_plot_div(pareto_summary(_df()))
    assert isinstance(div, str)
    assert "plotly" in div.lower() or "<div" in div.lower()
```

- [ ] **Step 2: Run to confirm failure**

Run: `micromamba run -n geneal pytest tests/test_pareto.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'geneal.report.pareto'`

- [ ] **Step 3: Implement `src/geneal/report/pareto.py`**

```python
# src/geneal/report/pareto.py
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def pareto_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per method: mean batch quality, mean batch diversity, final recall.

    Quality/diversity averaged over all acquired batches (round >= 1, NaN round-0
    rows dropped); final_recall is the mean metric at the last round across seeds.
    """
    acq = df[df["round"] >= 1]
    g = acq.groupby("method").agg(
        mean_quality=("batch_quality", "mean"),
        mean_diversity=("batch_diversity", "mean"),
    ).reset_index()
    last = df["round"].max()
    fin = (df[df["round"] == last].groupby("method")["metric"].mean()
           .reset_index().rename(columns={"metric": "final_recall"}))
    return g.merge(fin, on="method")


def pareto_plot_div(summary: pd.DataFrame) -> str:
    """Scatter of mean diversity (x) vs mean quality (y); marker size = final recall."""
    fig = go.Figure()
    sizes = summary["final_recall"].to_numpy()
    sizes = 10 + 30 * (sizes - sizes.min()) / (np.ptp(sizes) + 1e-9)
    for _, row in summary.iterrows():
        fig.add_trace(go.Scatter(
            x=[row["mean_diversity"]], y=[row["mean_quality"]],
            mode="markers+text", text=[row["method"]], textposition="top center",
            marker=dict(size=float(10 + 30 * row["final_recall"])),
            name=f"{row['method']} (recall={row['final_recall']:.2f})"))
    fig.update_layout(xaxis_title="batch diversity (mean pairwise dist)",
                      yaxis_title="batch quality (mean true effect)",
                      template="simple_white",
                      title="Quality–diversity Pareto (marker size = final recall@k)")
    return fig.to_html(full_html=False, include_plotlyjs="cdn")
```

- [ ] **Step 4: Run to confirm pass**

Run: `micromamba run -n geneal pytest tests/test_pareto.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/geneal/report/pareto.py tests/test_pareto.py
git commit -m "feat: add quality-diversity Pareto summary and plot"
```

---

## Task 8: Hydra configs for the new methods

**Files:**
- Create: `conf/method/al_kdpp.yaml`, `conf/method/baseline_coreset.yaml`, `conf/method/baseline_typiclust.yaml`

- [ ] **Step 1: Create the three config files**

`conf/method/al_kdpp.yaml`:
```yaml
name: al_kdpp
acquisition: { _target_: geneal.models.acquisition.UCB, beta: 2.0 }
selection:   { _target_: geneal.models.selection.KDPP, solver: greedy }
surrogate:   { _target_: geneal.models.surrogate.GPRSurrogate }
```

`conf/method/baseline_coreset.yaml`:
```yaml
name: baseline_coreset
acquisition: { _target_: geneal.models.acquisition.GreedyMean }
selection:   { _target_: geneal.models.selection.CoreSet }
surrogate:   { _target_: geneal.models.surrogate.GPRSurrogate }
```

`conf/method/baseline_typiclust.yaml`:
```yaml
name: baseline_typiclust
acquisition: { _target_: geneal.models.acquisition.GreedyMean }
selection:   { _target_: geneal.models.selection.TypiClust }
surrogate:   { _target_: geneal.models.surrogate.GPRSurrogate }
```

- [ ] **Step 2: Verify they instantiate**

Run:
```bash
micromamba run -n geneal python -c "
from hydra import initialize, compose
from hydra.utils import instantiate
with initialize(version_base=None, config_path='../conf/method'):
    for name in ['al_kdpp','baseline_coreset','baseline_typiclust']:
        cfg = compose(config_name=name)
        s = instantiate(cfg.selection); a = instantiate(cfg.acquisition)
        print(name, type(s).__name__, type(a).__name__)
"
```
Expected: prints the three method names with their selection/acquisition class names, no error.

- [ ] **Step 3: Commit**

```bash
git add conf/method/al_kdpp.yaml conf/method/baseline_coreset.yaml conf/method/baseline_typiclust.yaml
git commit -m "feat: add hydra configs for k-DPP and diversity-only baselines"
```

---

## Task 9: Headline empirical test — k-DPP beats both failure modes on synthetic

**Files:**
- Test: `tests/test_kdpp_beats_baselines.py`

**Context:** This is the plan's payoff and a guard against regressions in the method. It encodes the moat's central empirical claim on a controlled synthetic problem where the structure is known: when the top-k genes form a few tight clusters in embedding space, quality-only greedy wastes the batch inside one cluster (redundant) and diversity-only coreset wanders off the high-value region — k-DPP should recover the top-k set fastest. Keep the problem small enough to run in well under a minute but structured enough that the effect appears. If the effect is real but noisy, average over a few seeds (the test does).

- [ ] **Step 1: Write the test**

```python
# tests/test_kdpp_beats_baselines.py
"""Headline claim: on a clustered top-k synthetic problem, k-DPP recovers the
true top-k set at least as fast as quality-only (greedy) AND diversity-only
(coreset) baselines, averaged over seeds. This is the moat made falsifiable."""
import numpy as np
import pandas as pd
from geneal.data.dataset import Dataset
from geneal.experiment.objects import ObjectiveObject, DesignObject, Method, Experiment
from geneal.metrics.recall import RecallAtK
from geneal.models.acquisition import UCB, GreedyMean
from geneal.models.selection import KDPP, TopQGreedy, CoreSet
from geneal.models.surrogate import GPRSurrogate
from geneal.models.noise import GaussianNoise
from geneal.runner.runner import Runner


def _clustered_dataset(seed=0):
    """Top-k genes live in a few tight clusters: greedy will over-sample one
    cluster (redundant), coreset will spread off the high-value region."""
    rng = np.random.default_rng(seed)
    dim = 6
    # 4 cluster centers; a few are 'lethal' (high target), most are not
    n_clusters = 8
    centers = rng.standard_normal((n_clusters, dim)) * 3
    cluster_value = rng.standard_normal(n_clusters)
    cluster_value[:2] += 6.0  # two clusters are strongly lethal
    embeddings, target, names = [], [], []
    gid = 0
    for c in range(n_clusters):
        for _ in range(20):
            embeddings.append(centers[c] + rng.standard_normal(dim) * 0.2)
            target.append(cluster_value[c] + rng.normal(0, 0.1))
            names.append(f"GENE_{gid:04d}"); gid += 1
    return Dataset(np.array(embeddings), np.array(target), names)


def _run(selection, acquisition, seeds):
    ds = _clustered_dataset(seed=0)
    exp = Experiment(
        dataset=ds,
        objective=ObjectiveObject(metric=RecallAtK(k=20), direction="maximize"),
        design=DesignObject(n_rounds=5, batch_size=8, n_initial=16, seed=0),
        noise=GaussianNoise(sigma=0.1),
        methods=[Method("m", acquisition, selection, GPRSurrogate(n_iters=60))],
    )
    df = Runner().run(exp, seeds=seeds)
    last = df["round"].max()
    return df[df["round"] == last]["metric"].mean()


def test_kdpp_at_least_matches_both_baselines():
    seeds = [0, 1, 2]
    kdpp = _run(KDPP(solver="greedy"), UCB(beta=2.0), seeds)
    greedy = _run(TopQGreedy(), UCB(beta=2.0), seeds)
    coreset = _run(CoreSet(), GreedyMean(), seeds)
    # k-DPP should not lose to either failure mode (small tolerance for noise)
    assert kdpp >= greedy - 0.05, f"kdpp {kdpp} < greedy {greedy}"
    assert kdpp >= coreset - 0.05, f"kdpp {kdpp} < coreset {coreset}"
```

- [ ] **Step 2: Run the test**

Run: `micromamba run -n geneal pytest tests/test_kdpp_beats_baselines.py -v`
Expected: PASS. If k-DPP does NOT match or beat both baselines, DO NOT weaken the assertion. Instead report it as a finding — it means either (a) the synthetic problem isn't structured enough to expose the effect (make clusters tighter / top-k more concentrated / batch larger), or (b) the method has a bug. Investigate in that order and report which. The whole thesis rests on this effect being real, so a genuine failure here is important signal, not a test to relax.

- [ ] **Step 3: Commit**

```bash
git add tests/test_kdpp_beats_baselines.py
git commit -m "test: headline claim — k-DPP matches/beats quality-only and diversity-only on clustered synthetic"
```

---

## Task 10: Full suite + end-to-end with all methods

**Files:** none new — verification.

- [ ] **Step 1: Run the entire test suite**

Run: `micromamba run -n geneal pytest`
Expected: all tests pass (v1's 37 + the new ones). Report the total.

- [ ] **Step 2: Run the real entrypoint with all five methods on synthetic and confirm a report is produced**

First add the new methods to a scratch run via Hydra overrides (the default `conf/config.yaml` `methods` list can be overridden on the CLI, or temporarily run with the method-group configs). Simplest: a one-off python check that builds an Experiment with all five methods and writes a report including the Pareto plot. Create NOTHING permanent — run inline:

```bash
micromamba run -n geneal python -c "
import pandas as pd
from geneal.data.dataset import make_synthetic
from geneal.experiment.objects import *
from geneal.metrics.recall import RecallAtK
from geneal.models.acquisition import UCB, RandomAcquisition, GreedyMean
from geneal.models.selection import KDPP, TopQGreedy, GreedyFantasy, CoreSet, TypiClust
from geneal.models.surrogate import GPRSurrogate
from geneal.models.noise import GaussianNoise
from geneal.runner.runner import Runner
from geneal.report.pareto import pareto_summary
ds = make_synthetic(n_genes=120, dim=6, seed=0, noise_sd=0.1)
methods = [
  Method('kdpp', UCB(2.0), KDPP('greedy'), GPRSurrogate(n_iters=60)),
  Method('greedy', UCB(2.0), TopQGreedy(), GPRSurrogate(n_iters=60)),
  Method('fantasy', UCB(2.0), GreedyFantasy(), GPRSurrogate(n_iters=60)),
  Method('coreset', GreedyMean(), CoreSet(), GPRSurrogate(n_iters=60)),
  Method('typiclust', GreedyMean(), TypiClust(), GPRSurrogate(n_iters=60)),
  Method('random', RandomAcquisition(), TopQGreedy(), GPRSurrogate(n_iters=60)),
]
exp = Experiment(ds, ObjectiveObject(RecallAtK(k=15),'maximize'),
                 DesignObject(n_rounds=4, batch_size=8, n_initial=16, seed=0),
                 GaussianNoise(0.1), methods)
df = Runner().run(exp, seeds=[0,1])
print(df.groupby('method')['metric'].last())
print(pareto_summary(df).to_string())
"
```
Expected: prints final recall per method and the Pareto summary table, no errors. (This is a smoke check of integration, not a scientific result.)

- [ ] **Step 3: Report**

Report the full-suite total and the smoke-check output. No commit needed (no files changed).

---

## Self-Review (completed during planning)

- **Spec coverage:** k-DPP method (Task 3) ✓; `predict_cov` surrogate requirement (Task 1) ✓; posterior-correlation S_ij (Task 2) ✓; quality-gating via pluggable acquisition q_i (Task 3, `_build_L`) ✓; greedy-MAP + sampling solvers (Task 3) ✓; CoreSet + TypiClust required baselines (Task 4) ✓; batch quality/diversity diagnostics (Tasks 5–6) ✓; quality–diversity Pareto (Task 7) ✓; headline Pareto-dominance claim as a test (Task 9) ✓; Hydra configs (Task 8) ✓.
- **Composability hook:** `KDPP._build_L` derives S from `surrogate.predict_cov`; the spec's "S_ij interface should accept an external similarity kernel" is a noted future extension, NOT built here (YAGNI for MLCB) — flagged so it isn't mistaken for a gap.
- **Selectivity extension:** NOT built here (per decision: single-objective is the guaranteed core). `q_i` is already a pluggable acquisition score, so a future differential acquisition (UCB on μ_A−μ_B) drops in without touching KDPP. No structural blocker.
- **Placeholder scan:** every code step is complete; the only conditional is the k-DPP sampling fallback (Task 3 Step 3), which is fully specified.
- **Type consistency:** all new selectors use the exact `select(*, candidate_idx, X_candidates, mean, std, best, q, rng, surrogate, acquisition, X_train, y_train)` signature from `interfaces.py` (updated in v1 review). `predict_cov` returns `(mean, cov)` consistently across GPR/BNN and is consumed identically in `KDPP._build_L`.

## Out of scope (Plan 2 / later)
Real DepMap data + ESM2/scPRINT embeddings (Plan 2, data track running in parallel). Selectivity/two-cell-line differential objective (design-compatible, build after core lands). External prior-fused kernel for S_ij (future axis).
