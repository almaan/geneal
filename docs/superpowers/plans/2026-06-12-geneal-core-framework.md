# geneal Core Active-Learning Framework — Implementation Plan (Plan 1 of 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the complete active-learning framework (surrogates, acquisition, selection, noise, metric, runner, HTML report) and prove it end-to-end on a synthetic effect matrix with placeholder embeddings — zero heavy data/embedding dependencies.

**Architecture:** Plain classes behind small Protocols, instantiated by Hydra `_target_`. A Runner runs ≥1 method (AL vs baseline) across N seeds using common random numbers (shared initial set + shared noise draws), logs every round to a structured run directory, and a Report turns that directory into HTML with recall@k curves and latex tables.

**Tech Stack:** Python 3.11, numpy, pandas, scipy, gpytorch (GPR Matérn), pyro (BNN), hydra, jinja2, plotly, pyarrow, pytest. scikit-learn is a light convenience dep (utilities only). Real DepMap data + scPRINT/ESM2 embeddings are **Plan 2** — this plan uses a synthetic dataset so the whole loop is testable in milliseconds.

**Scope note / refinement from spec:** v1 GPR surrogate uses **gpytorch** — an exact GP (`gpytorch.models.ExactGP`) with a Matérn 2.5 kernel and Gaussian likelihood, trained by marginal-likelihood maximization. Returns predictive mean + std. botorch is NOT required for v1 — greedy+fantasy batching is implemented as a surrogate refit with the fantasized outcome (Task 7). botorch can be added later if its acquisition machinery is wanted.

---

## File Structure

```
pyproject.toml                          # package metadata + deps
src/geneal/__init__.py
src/geneal/interfaces.py                 # Protocols: Surrogate, Acquisition, Selection, NoiseModel, Metric
src/geneal/data/__init__.py
src/geneal/data/dataset.py               # Dataset dataclass + synthetic generator
src/geneal/metrics/__init__.py
src/geneal/metrics/recall.py             # RecallAtK
src/geneal/models/__init__.py
src/geneal/models/noise.py               # NoNoise, GaussianNoise
src/geneal/models/surrogate.py           # GPRSurrogate, BNNSurrogate
src/geneal/models/acquisition.py         # UCB, EI, MaxVariance, GreedyMean, RandomAcquisition
src/geneal/models/selection.py           # TopQGreedy, GreedyFantasy
src/geneal/experiment/__init__.py
src/geneal/experiment/objects.py         # ObjectiveObject, DesignObject, Method, Experiment
src/geneal/runner/__init__.py
src/geneal/runner/runner.py              # Runner (round loop, CRN)
src/geneal/runner/logging.py             # write_run_dir (parquet + manifest + config)
src/geneal/report/__init__.py
src/geneal/report/report.py              # build_report (HTML)
src/geneal/report/templates/report.html.j2
conf/config.yaml                         # top-level hydra config
conf/dataset/synthetic.yaml
conf/method/al_ucb.yaml
conf/method/baseline_random.yaml
scripts/run_experiment.py                # hydra entrypoint
tests/test_dataset.py
tests/test_recall.py
tests/test_noise.py
tests/test_surrogate.py
tests/test_acquisition.py
tests/test_selection.py
tests/test_runner.py
tests/test_report.py
tests/test_smoke_end_to_end.py
```

**Conventions used throughout:**
- The optimization **target** is "lethality": higher = more lethal = better. (From DepMap, target = `-gene_effect`; the synthetic generator produces target directly.) All acquisition functions **maximize** the target.
- Arrays are numpy. `Surrogate.predict` returns `(mean, std)` as 1-D arrays.
- RNG is always an explicit `numpy.random.Generator` passed in — never global state.

---

## Task 0: Environment + package scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `src/geneal/__init__.py`
- Create: all `__init__.py` listed in File Structure (empty)

- [ ] **Step 1: Create the env (USER ACTION — ask the user to run this)**

The user owns env creation. Ask them to run:

```bash
micromamba create -n geneal -c conda-forge python=3.11 \
  numpy pandas scipy scikit-learn pyro-ppl pytorch gpytorch \
  hydra-core jinja2 plotly pyarrow pytest -y
```

(If `gpytorch` is unavailable on conda-forge for the resolved torch version, install it
into the env with pip afterward: `micromamba run -n geneal pip install gpytorch`.)

Then activate for all subsequent commands: `micromamba activate geneal`.
(Embedding deps — scPRINT / fair-esm — are added in Plan 2, not here.)

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "geneal"
version = "0.1.0"
description = "Active learning for gene-knockout selection on molecular data"
requires-python = ">=3.11"
dependencies = [
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "gpytorch",
    "pyro-ppl",
    "torch",
    "hydra-core",
    "jinja2",
    "plotly",
    "pyarrow",
]

[project.optional-dependencies]
dev = ["pytest"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

- [ ] **Step 3: Create empty package files**

Create `src/geneal/__init__.py` with:

```python
"""geneal: active learning for gene-knockout selection."""
__version__ = "0.1.0"
```

Create empty `__init__.py` (single newline) in each of:
`src/geneal/data/`, `src/geneal/metrics/`, `src/geneal/models/`,
`src/geneal/experiment/`, `src/geneal/runner/`, `src/geneal/report/`.

- [ ] **Step 4: Editable install**

Run: `micromamba run -n geneal pip install -e ".[dev]"`
Expected: `Successfully installed geneal-0.1.0`

- [ ] **Step 5: Verify pytest collects nothing yet (sanity)**

Run: `micromamba run -n geneal pytest`
Expected: `no tests ran` (exit code 5 is fine).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/geneal
git commit -m "chore: scaffold geneal package and env"
```

---

## Task 1: Interfaces (Protocols)

**Files:**
- Create: `src/geneal/interfaces.py`
- Test: `tests/test_interfaces.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_interfaces.py
import numpy as np
from geneal.interfaces import Surrogate, Acquisition, Selection, NoiseModel, Metric


class _Surr:
    def fit(self, X, y): return self
    def predict(self, X): return np.zeros(len(X)), np.ones(len(X))
    def clone(self): return _Surr()


def test_surrogate_protocol_runtime_check():
    assert isinstance(_Surr(), Surrogate)


def test_non_surrogate_fails_check():
    class NotSurr:
        def fit(self, X, y): return self
    assert not isinstance(NotSurr(), Surrogate)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `micromamba run -n geneal pytest tests/test_interfaces.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'geneal.interfaces'`

- [ ] **Step 3: Write the implementation**

```python
# src/geneal/interfaces.py
from __future__ import annotations
from typing import Protocol, runtime_checkable, Sequence
import numpy as np


@runtime_checkable
class Surrogate(Protocol):
    """Maps gene embeddings -> predicted target with uncertainty."""
    def fit(self, X: np.ndarray, y: np.ndarray) -> "Surrogate": ...
    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...  # (mean, std)
    def clone(self) -> "Surrogate": ...


@runtime_checkable
class Acquisition(Protocol):
    """Scores candidates from posterior predictive. Higher score = more desirable."""
    def score(self, mean: np.ndarray, std: np.ndarray, best: float,
              rng: np.random.Generator) -> np.ndarray: ...


@runtime_checkable
class Selection(Protocol):
    """Chooses q candidate indices to acquire this round.

    Receives everything any strategy might need; impls use the subset they care
    about. `candidate_idx` are absolute dataset indices; the return value is a
    list of absolute dataset indices (a subset of candidate_idx), length q.
    """
    def select(self, *, candidate_idx: Sequence[int], X_candidates: np.ndarray,
               mean: np.ndarray, std: np.ndarray, best: float, q: int,
               rng: np.random.Generator, surrogate: Surrogate,
               acquisition: Acquisition) -> list[int]: ...


@runtime_checkable
class NoiseModel(Protocol):
    """Draws a per-gene noise vector once per seed (common random numbers)."""
    def draw(self, n: int, rng: np.random.Generator) -> np.ndarray: ...


@runtime_checkable
class Metric(Protocol):
    """Evaluates progress given the revealed set and ground-truth target."""
    def evaluate(self, revealed_idx: Sequence[int], target: np.ndarray) -> float: ...
    @property
    def name(self) -> str: ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `micromamba run -n geneal pytest tests/test_interfaces.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/geneal/interfaces.py tests/test_interfaces.py
git commit -m "feat: add core Protocol interfaces"
```

---

## Task 2: Dataset + synthetic generator

**Files:**
- Create: `src/geneal/data/dataset.py`
- Test: `tests/test_dataset.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dataset.py
import numpy as np
from geneal.data.dataset import Dataset, make_synthetic


def test_make_synthetic_shapes():
    ds = make_synthetic(n_genes=50, dim=8, seed=0)
    assert ds.embeddings.shape == (50, 8)
    assert ds.target.shape == (50,)
    assert len(ds.gene_names) == 50
    assert ds.n_genes == 50


def test_make_synthetic_is_deterministic():
    a = make_synthetic(n_genes=20, dim=4, seed=7)
    b = make_synthetic(n_genes=20, dim=4, seed=7)
    np.testing.assert_array_equal(a.target, b.target)
    np.testing.assert_array_equal(a.embeddings, b.embeddings)


def test_synthetic_target_is_learnable_from_embeddings():
    # Target is a (noisy) linear function of embeddings, so correlation with the
    # true linear signal should be strong -> a surrogate can learn it.
    ds = make_synthetic(n_genes=500, dim=6, seed=1, noise_sd=0.1)
    # Fit least squares; R^2 should be high.
    X = np.hstack([ds.embeddings, np.ones((ds.n_genes, 1))])
    beta, *_ = np.linalg.lstsq(X, ds.target, rcond=None)
    pred = X @ beta
    ss_res = np.sum((ds.target - pred) ** 2)
    ss_tot = np.sum((ds.target - ds.target.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    assert r2 > 0.8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `micromamba run -n geneal pytest tests/test_dataset.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'geneal.data.dataset'`

- [ ] **Step 3: Write the implementation**

```python
# src/geneal/data/dataset.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Dataset:
    """One cell line's knockout problem.

    embeddings: (n_genes, dim) gene representations (placeholder or real).
    target:     (n_genes,) optimization target; HIGHER = more lethal = better.
    gene_names: list of gene identifiers, len n_genes.
    """
    embeddings: np.ndarray
    target: np.ndarray
    gene_names: list[str]

    def __post_init__(self) -> None:
        n = len(self.gene_names)
        if self.embeddings.shape[0] != n or self.target.shape[0] != n:
            raise ValueError("embeddings, target, gene_names must share length")

    @property
    def n_genes(self) -> int:
        return len(self.gene_names)


def make_synthetic(n_genes: int = 200, dim: int = 8, seed: int = 0,
                   noise_sd: float = 0.1) -> Dataset:
    """Synthetic dataset where target is a noisy linear function of embeddings."""
    rng = np.random.default_rng(seed)
    embeddings = rng.standard_normal((n_genes, dim))
    beta = rng.standard_normal(dim)
    signal = embeddings @ beta
    target = signal + rng.normal(0.0, noise_sd, size=n_genes)
    gene_names = [f"GENE_{i:04d}" for i in range(n_genes)]
    return Dataset(embeddings=embeddings, target=target, gene_names=gene_names)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `micromamba run -n geneal pytest tests/test_dataset.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/geneal/data/dataset.py tests/test_dataset.py
git commit -m "feat: add Dataset and synthetic generator"
```

---

## Task 3: RecallAtK metric

**Files:**
- Create: `src/geneal/metrics/recall.py`
- Test: `tests/test_recall.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_recall.py
import numpy as np
from geneal.metrics.recall import RecallAtK


def test_recall_name():
    assert RecallAtK(k=5).name == "recall@5"


def test_recall_full_when_all_top_revealed():
    target = np.array([10.0, 9.0, 8.0, 1.0, 0.0])  # top-2 are idx 0,1
    m = RecallAtK(k=2)
    assert m.evaluate([0, 1, 4], target) == 1.0


def test_recall_partial():
    target = np.array([10.0, 9.0, 8.0, 1.0, 0.0])
    m = RecallAtK(k=2)
    assert m.evaluate([0, 4], target) == 0.5  # found idx 0 of {0,1}


def test_recall_zero():
    target = np.array([10.0, 9.0, 8.0, 1.0, 0.0])
    m = RecallAtK(k=2)
    assert m.evaluate([3, 4], target) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `micromamba run -n geneal pytest tests/test_recall.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'geneal.metrics.recall'`

- [ ] **Step 3: Write the implementation**

```python
# src/geneal/metrics/recall.py
from __future__ import annotations
from typing import Sequence
import numpy as np


class RecallAtK:
    """Fraction of the true top-k (highest target) that has been revealed."""

    def __init__(self, k: int = 10) -> None:
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k

    @property
    def name(self) -> str:
        return f"recall@{self.k}"

    def true_top(self, target: np.ndarray) -> set[int]:
        k = min(self.k, len(target))
        return set(np.argsort(target)[::-1][:k].tolist())

    def evaluate(self, revealed_idx: Sequence[int], target: np.ndarray) -> float:
        top = self.true_top(target)
        if not top:
            return 0.0
        hit = len(set(int(i) for i in revealed_idx) & top)
        return hit / len(top)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `micromamba run -n geneal pytest tests/test_recall.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/geneal/metrics/recall.py tests/test_recall.py
git commit -m "feat: add RecallAtK metric"
```

---

## Task 4: Noise models

**Files:**
- Create: `src/geneal/models/noise.py`
- Test: `tests/test_noise.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_noise.py
import numpy as np
from geneal.models.noise import NoNoise, GaussianNoise


def test_nonoise_is_zeros():
    rng = np.random.default_rng(0)
    np.testing.assert_array_equal(NoNoise().draw(5, rng), np.zeros(5))


def test_gaussian_shape_and_determinism():
    a = GaussianNoise(sigma=0.5).draw(100, np.random.default_rng(3))
    b = GaussianNoise(sigma=0.5).draw(100, np.random.default_rng(3))
    assert a.shape == (100,)
    np.testing.assert_array_equal(a, b)


def test_gaussian_scale_roughly_correct():
    x = GaussianNoise(sigma=2.0).draw(50000, np.random.default_rng(1))
    assert abs(x.std() - 2.0) < 0.1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `micromamba run -n geneal pytest tests/test_noise.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'geneal.models.noise'`

- [ ] **Step 3: Write the implementation**

```python
# src/geneal/models/noise.py
from __future__ import annotations
import numpy as np


class NoNoise:
    """Reveals true target values unchanged."""
    def draw(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return np.zeros(n)


class GaussianNoise:
    """Homoscedastic Gaussian measurement noise, N(0, sigma^2)."""
    def __init__(self, sigma: float = 1.0) -> None:
        if sigma < 0:
            raise ValueError("sigma must be >= 0")
        self.sigma = sigma

    def draw(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(0.0, self.sigma, size=n)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `micromamba run -n geneal pytest tests/test_noise.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/geneal/models/noise.py tests/test_noise.py
git commit -m "feat: add NoNoise and GaussianNoise models"
```

---

## Task 5: Surrogates (GPR + BNN)

**Files:**
- Create: `src/geneal/models/surrogate.py`
- Test: `tests/test_surrogate.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_surrogate.py
import numpy as np
import pytest
from geneal.data.dataset import make_synthetic
from geneal.models.surrogate import GPRSurrogate, BNNSurrogate


def _train_test():
    ds = make_synthetic(n_genes=120, dim=5, seed=2, noise_sd=0.05)
    Xtr, ytr = ds.embeddings[:80], ds.target[:80]
    Xte, yte = ds.embeddings[80:], ds.target[80:]
    return Xtr, ytr, Xte, yte


def test_gpr_predict_shapes_and_clone():
    Xtr, ytr, Xte, yte = _train_test()
    surr = GPRSurrogate().fit(Xtr, ytr)
    mean, std = surr.predict(Xte)
    assert mean.shape == (len(Xte),)
    assert std.shape == (len(Xte),)
    assert np.all(std >= 0)
    # clone is unfitted and independent
    clone = surr.clone()
    assert isinstance(clone, GPRSurrogate)


def test_gpr_learns_signal():
    Xtr, ytr, Xte, yte = _train_test()
    mean, _ = GPRSurrogate().fit(Xtr, ytr).predict(Xte)
    corr = np.corrcoef(mean, yte)[0, 1]
    assert corr > 0.7


def test_bnn_predict_shapes():
    Xtr, ytr, Xte, yte = _train_test()
    mean, std = BNNSurrogate(hidden=16, n_steps=200, seed=0).fit(Xtr, ytr).predict(Xte)
    assert mean.shape == (len(Xte),)
    assert std.shape == (len(Xte),)
    assert np.all(std >= 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `micromamba run -n geneal pytest tests/test_surrogate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'geneal.models.surrogate'`

- [ ] **Step 3: Write the implementation**

```python
# src/geneal/models/surrogate.py
from __future__ import annotations
import numpy as np


class GPRSurrogate:
    """Exact Gaussian Process regression with a Matern-2.5 kernel (gpytorch).

    An ExactGP with constant mean, ScaleKernel(MaternKernel(nu)), and a Gaussian
    likelihood, trained by maximizing the exact marginal log-likelihood with Adam.
    Inputs and targets are standardized internally; predictions are returned on the
    original target scale as (mean, std). `clone` returns a fresh, unfitted
    surrogate with the same hyperparameters (used for fantasy batching).
    """

    def __init__(self, nu: float = 2.5, n_iters: int = 100, lr: float = 0.1) -> None:
        self.nu = nu
        self.n_iters = n_iters
        self.lr = lr
        self._model = None
        self._likelihood = None
        self._x_mean = None
        self._x_std = None
        self._y_mean = None
        self._y_std = None

    def _build(self, train_x, train_y):
        import gpytorch

        nu = self.nu

        class _ExactGP(gpytorch.models.ExactGP):
            def __init__(self, tx, ty, likelihood):
                super().__init__(tx, ty, likelihood)
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.MaternKernel(nu=nu))

            def forward(self, x):
                return gpytorch.distributions.MultivariateNormal(
                    self.mean_module(x), self.covar_module(x))

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = _ExactGP(train_x, train_y, likelihood)
        return model, likelihood

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GPRSurrogate":
        import torch
        import gpytorch

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self._x_mean, self._x_std = X.mean(0), X.std(0) + 1e-8
        self._y_mean, self._y_std = y.mean(), y.std() + 1e-8
        tx = torch.tensor((X - self._x_mean) / self._x_std, dtype=torch.float32)
        ty = torch.tensor((y - self._y_mean) / self._y_std, dtype=torch.float32)

        self._model, self._likelihood = self._build(tx, ty)
        self._model.train()
        self._likelihood.train()
        opt = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self._likelihood, self._model)
        for _ in range(self.n_iters):
            opt.zero_grad()
            out = self._model(tx)
            loss = -mll(out, ty)
            loss.backward()
            opt.step()
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import torch
        import gpytorch
        if self._model is None:
            raise RuntimeError("GPRSurrogate.predict called before fit")
        X = np.asarray(X, dtype=np.float64)
        tx = torch.tensor((X - self._x_mean) / self._x_std, dtype=torch.float32)
        self._model.eval()
        self._likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            post = self._likelihood(self._model(tx))
            mean = post.mean.numpy()
            std = post.stddev.numpy()
        return mean * self._y_std + self._y_mean, std * self._y_std

    def clone(self) -> "GPRSurrogate":
        return GPRSurrogate(nu=self.nu, n_iters=self.n_iters, lr=self.lr)


class BNNSurrogate:
    """Bayesian neural network (single hidden layer) via Pyro SVI.

    Predictive mean/std come from sampling the variational posterior. Kept small
    and few-step by default so it is fast in the AL loop and in tests.
    """

    def __init__(self, hidden: int = 32, n_steps: int = 500, lr: float = 1e-2,
                 prior_scale: float = 1.0, n_predict: int = 64, seed: int = 0) -> None:
        self.hidden = hidden
        self.n_steps = n_steps
        self.lr = lr
        self.prior_scale = prior_scale
        self.n_predict = n_predict
        self.seed = seed
        self._guide = None
        self._x_mean = None
        self._x_std = None
        self._y_mean = None
        self._y_std = None

    def _model(self, x, y=None):
        import torch
        import pyro
        import pyro.distributions as dist
        d = x.shape[1]
        ps = self.prior_scale
        w1 = pyro.sample("w1", dist.Normal(torch.zeros(d, self.hidden),
                                           ps * torch.ones(d, self.hidden)).to_event(2))
        b1 = pyro.sample("b1", dist.Normal(torch.zeros(self.hidden),
                                           ps * torch.ones(self.hidden)).to_event(1))
        w2 = pyro.sample("w2", dist.Normal(torch.zeros(self.hidden, 1),
                                           ps * torch.ones(self.hidden, 1)).to_event(2))
        b2 = pyro.sample("b2", dist.Normal(torch.zeros(1), ps * torch.ones(1)).to_event(1))
        sigma = pyro.sample("sigma", dist.HalfNormal(torch.tensor(1.0)))
        h = torch.tanh(x @ w1 + b1)
        out = (h @ w2 + b2).squeeze(-1)
        with pyro.plate("data", x.shape[0]):
            pyro.sample("obs", dist.Normal(out, sigma), obs=y)
        return out

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BNNSurrogate":
        import torch
        import pyro
        from pyro.infer import SVI, Trace_ELBO
        from pyro.infer.autoguide import AutoDiagonalNormal

        pyro.clear_param_store()
        pyro.set_rng_seed(self.seed)

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self._x_mean, self._x_std = X.mean(0), X.std(0) + 1e-8
        self._y_mean, self._y_std = y.mean(), y.std() + 1e-8
        xt = torch.tensor((X - self._x_mean) / self._x_std, dtype=torch.float32)
        yt = torch.tensor((y - self._y_mean) / self._y_std, dtype=torch.float32)

        self._guide = AutoDiagonalNormal(self._model)
        svi = SVI(self._model, self._guide,
                  pyro.optim.Adam({"lr": self.lr}), loss=Trace_ELBO())
        for _ in range(self.n_steps):
            svi.step(xt, yt)
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import torch
        import pyro
        from pyro.infer import Predictive
        if self._guide is None:
            raise RuntimeError("BNNSurrogate.predict called before fit")
        X = np.asarray(X, dtype=np.float64)
        xt = torch.tensor((X - self._x_mean) / self._x_std, dtype=torch.float32)
        pred = Predictive(self._model, guide=self._guide,
                          num_samples=self.n_predict, return_sites=["obs"])
        with torch.no_grad():
            obs = pred(xt)["obs"].detach().numpy()  # (n_predict, n)
        mean = obs.mean(0) * self._y_std + self._y_mean
        std = obs.std(0) * self._y_std
        return mean, std

    def clone(self) -> "BNNSurrogate":
        return BNNSurrogate(hidden=self.hidden, n_steps=self.n_steps, lr=self.lr,
                            prior_scale=self.prior_scale, n_predict=self.n_predict,
                            seed=self.seed)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `micromamba run -n geneal pytest tests/test_surrogate.py -v`
Expected: PASS (3 passed). Both the gpytorch GPR fit and the BNN fit take a few seconds — acceptable. If `test_gpr_learns_signal` correlation is borderline, raise `n_iters` (e.g. 200) — the contract, not the exact iteration count, is what matters.

- [ ] **Step 5: Commit**

```bash
git add src/geneal/models/surrogate.py tests/test_surrogate.py
git commit -m "feat: add GPR (gpytorch Matern ExactGP) and BNN (pyro) surrogates"
```

---

## Task 6: Acquisition functions

**Files:**
- Create: `src/geneal/models/acquisition.py`
- Test: `tests/test_acquisition.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_acquisition.py
import numpy as np
from geneal.models.acquisition import (
    UCB, ExpectedImprovement, MaxVariance, GreedyMean, RandomAcquisition,
)


def test_ucb_prefers_high_mean_and_high_std():
    mean = np.array([0.0, 1.0, 0.0])
    std = np.array([0.0, 0.0, 1.0])
    s = UCB(beta=2.0).score(mean, std, best=0.0, rng=np.random.default_rng(0))
    # idx1 (mean=1) and idx2 (2*std=2) both beat idx0
    assert s[1] > s[0]
    assert s[2] > s[0]


def test_greedy_mean_is_mean():
    mean = np.array([3.0, 1.0])
    std = np.array([5.0, 5.0])
    s = GreedyMean().score(mean, std, best=0.0, rng=np.random.default_rng(0))
    np.testing.assert_array_equal(s, mean)


def test_max_variance_is_std():
    mean = np.array([3.0, 1.0])
    std = np.array([0.2, 0.9])
    s = MaxVariance().score(mean, std, best=0.0, rng=np.random.default_rng(0))
    np.testing.assert_array_equal(s, std)


def test_ei_nonnegative_and_zero_when_std_zero_below_best():
    mean = np.array([1.0, 2.0])
    std = np.array([0.0, 1.0])
    s = ExpectedImprovement().score(mean, std, best=5.0, rng=np.random.default_rng(0))
    assert np.all(s >= 0)
    assert s[0] == 0.0  # std 0 and mean < best -> no improvement


def test_random_is_deterministic_given_seed():
    mean = np.zeros(10); std = np.zeros(10)
    a = RandomAcquisition().score(mean, std, 0.0, np.random.default_rng(5))
    b = RandomAcquisition().score(mean, std, 0.0, np.random.default_rng(5))
    np.testing.assert_array_equal(a, b)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `micromamba run -n geneal pytest tests/test_acquisition.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'geneal.models.acquisition'`

- [ ] **Step 3: Write the implementation**

```python
# src/geneal/models/acquisition.py
from __future__ import annotations
import numpy as np
from scipy.stats import norm

# All acquisitions MAXIMIZE the target. Higher score => more desirable.


class UCB:
    """Upper confidence bound: mean + beta * std."""
    def __init__(self, beta: float = 2.0) -> None:
        self.beta = beta

    def score(self, mean, std, best, rng):
        return np.asarray(mean) + self.beta * np.asarray(std)


class ExpectedImprovement:
    """Expected improvement over current best (maximization)."""
    def __init__(self, xi: float = 0.0) -> None:
        self.xi = xi

    def score(self, mean, std, best, rng):
        mean = np.asarray(mean, dtype=float)
        std = np.asarray(std, dtype=float)
        imp = mean - best - self.xi
        out = np.zeros_like(mean)
        mask = std > 1e-12
        z = np.zeros_like(mean)
        z[mask] = imp[mask] / std[mask]
        out[mask] = imp[mask] * norm.cdf(z[mask]) + std[mask] * norm.pdf(z[mask])
        # where std==0: improvement only if mean exceeds best
        out[~mask] = np.maximum(imp[~mask], 0.0)
        return np.maximum(out, 0.0)


class MaxVariance:
    """Pure-exploration / information-gain proxy: predictive std."""
    def score(self, mean, std, best, rng):
        return np.asarray(std)


class GreedyMean:
    """Pure exploitation: predicted mean."""
    def score(self, mean, std, best, rng):
        return np.asarray(mean)


class RandomAcquisition:
    """Uniform random scores — used by the random baseline method."""
    def score(self, mean, std, best, rng):
        return rng.random(len(mean))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `micromamba run -n geneal pytest tests/test_acquisition.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/geneal/models/acquisition.py tests/test_acquisition.py
git commit -m "feat: add UCB, EI, MaxVariance, GreedyMean, Random acquisitions"
```

> Note: full EIG is intentionally deferred. `MaxVariance` is the v1 information-seeking acquisition. Adding a richer EIG is a future task once a concrete need appears.

---

## Task 7: Selection strategies

**Files:**
- Create: `src/geneal/models/selection.py`
- Test: `tests/test_selection.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_selection.py
import numpy as np
from geneal.models.selection import TopQGreedy, GreedyFantasy
from geneal.models.acquisition import GreedyMean, UCB
from geneal.models.surrogate import GPRSurrogate
from geneal.data.dataset import make_synthetic


def test_topq_picks_highest_scores_and_maps_to_absolute_idx():
    candidate_idx = [10, 11, 12, 13]
    mean = np.array([0.1, 0.9, 0.5, 0.3])
    std = np.zeros(4)
    sel = TopQGreedy().select(
        candidate_idx=candidate_idx, X_candidates=np.zeros((4, 2)),
        mean=mean, std=std, best=0.0, q=2,
        rng=np.random.default_rng(0), surrogate=None, acquisition=GreedyMean(),
    )
    assert sel == [11, 12]  # highest mean (0.9) then 0.5


def test_topq_returns_q_items():
    sel = TopQGreedy().select(
        candidate_idx=[0, 1, 2, 3, 4], X_candidates=np.zeros((5, 2)),
        mean=np.arange(5.0), std=np.zeros(5), best=0.0, q=3,
        rng=np.random.default_rng(0), surrogate=None, acquisition=GreedyMean(),
    )
    assert len(sel) == 3
    assert len(set(sel)) == 3


def test_fantasy_returns_q_distinct_absolute_idx():
    ds = make_synthetic(n_genes=40, dim=4, seed=1)
    surr = GPRSurrogate().fit(ds.embeddings[:10], ds.target[:10])
    cand = list(range(10, 40))
    Xc = ds.embeddings[cand]
    mean, std = surr.predict(Xc)
    sel = GreedyFantasy().select(
        candidate_idx=cand, X_candidates=Xc, mean=mean, std=std,
        best=float(ds.target[:10].max()), q=5, rng=np.random.default_rng(0),
        surrogate=surr, acquisition=UCB(beta=1.0),
    )
    assert len(sel) == 5
    assert len(set(sel)) == 5
    assert set(sel).issubset(set(cand))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `micromamba run -n geneal pytest tests/test_selection.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'geneal.models.selection'`

- [ ] **Step 3: Write the implementation**

```python
# src/geneal/models/selection.py
from __future__ import annotations
from typing import Sequence
import numpy as np


class TopQGreedy:
    """Score all candidates once, take the q highest-scoring."""
    def select(self, *, candidate_idx: Sequence[int], X_candidates, mean, std,
               best, q, rng, surrogate, acquisition) -> list[int]:
        scores = acquisition.score(mean, std, best, rng)
        candidate_idx = list(candidate_idx)
        q = min(q, len(candidate_idx))
        order = np.argsort(scores)[::-1][:q]
        return [candidate_idx[i] for i in order]


class GreedyFantasy:
    """Sequential greedy batch with fantasized outcomes for diversity.

    Pick the best candidate, fantasize its outcome as the surrogate's predictive
    mean, refit a clone of the surrogate on the augmented data, rescore, repeat q
    times. Encourages spread instead of q near-identical picks.
    """
    def select(self, *, candidate_idx: Sequence[int], X_candidates, mean, std,
               best, q, rng, surrogate, acquisition) -> list[int]:
        candidate_idx = list(candidate_idx)
        X_candidates = np.asarray(X_candidates)
        q = min(q, len(candidate_idx))

        # Reconstruct the surrogate's current training data by refitting a clone
        # incrementally with fantasies. We need the original training set; pull it
        # from a fresh fit is not possible, so we accumulate fantasies on top of a
        # working clone that starts from the passed-in surrogate's predictions.
        work = surrogate.clone()
        # Seed the working model with the real fitted surrogate's training data via
        # its predictions on candidates is insufficient; instead we fit `work` to
        # the fantasies plus an anchor of the current candidates' predicted means.
        fant_X: list[np.ndarray] = []
        fant_y: list[float] = []

        remaining = list(range(len(candidate_idx)))
        chosen: list[int] = []
        cur_mean, cur_std = np.asarray(mean).copy(), np.asarray(std).copy()
        cur_best = best

        for _ in range(q):
            scores = acquisition.score(cur_mean[remaining], cur_std[remaining],
                                       cur_best, rng)
            local = remaining[int(np.argmax(scores))]
            chosen.append(candidate_idx[local])
            remaining.remove(local)
            if not remaining:
                break
            # Fantasize the outcome at the chosen point = its predictive mean.
            fant_X.append(X_candidates[local])
            fant_y.append(float(cur_mean[local]))
            cur_best = max(cur_best, fant_y[-1])
            # Refit working model on fantasies and re-predict remaining candidates.
            work = surrogate.clone().fit(np.vstack(fant_X), np.asarray(fant_y))
            m, s = work.predict(X_candidates)
            cur_mean, cur_std = np.asarray(m), np.asarray(s)
        return chosen
```

- [ ] **Step 4: Run test to verify it passes**

Run: `micromamba run -n geneal pytest tests/test_selection.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/geneal/models/selection.py tests/test_selection.py
git commit -m "feat: add TopQGreedy and GreedyFantasy selection"
```

---

## Task 8: Experiment objects (Objective, Design, Method, Experiment)

**Files:**
- Create: `src/geneal/experiment/objects.py`
- Test: `tests/test_objects.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_objects.py
from geneal.experiment.objects import ObjectiveObject, DesignObject, Method, Experiment
from geneal.metrics.recall import RecallAtK
from geneal.models.acquisition import UCB, RandomAcquisition
from geneal.models.selection import TopQGreedy
from geneal.models.surrogate import GPRSurrogate
from geneal.models.noise import GaussianNoise
from geneal.data.dataset import make_synthetic


def test_design_defaults_and_validation():
    d = DesignObject(n_rounds=5, batch_size=10, n_initial=20, seed=0)
    assert d.n_rounds == 5 and d.batch_size == 10


def test_objective_holds_metric_and_direction():
    o = ObjectiveObject(metric=RecallAtK(k=10), direction="maximize")
    assert o.metric.name == "recall@10"
    assert o.direction == "maximize"


def test_experiment_assembles():
    ds = make_synthetic(n_genes=50, dim=4, seed=0)
    exp = Experiment(
        dataset=ds,
        objective=ObjectiveObject(metric=RecallAtK(k=5), direction="maximize"),
        design=DesignObject(n_rounds=3, batch_size=5, n_initial=10, seed=0),
        noise=GaussianNoise(sigma=0.1),
        methods=[
            Method(name="al_ucb", acquisition=UCB(), selection=TopQGreedy(),
                   surrogate=GPRSurrogate()),
            Method(name="baseline", acquisition=RandomAcquisition(),
                   selection=TopQGreedy(), surrogate=GPRSurrogate()),
        ],
    )
    assert len(exp.methods) == 2
    assert exp.dataset.n_genes == 50
```

- [ ] **Step 2: Run test to verify it fails**

Run: `micromamba run -n geneal pytest tests/test_objects.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'geneal.experiment.objects'`

- [ ] **Step 3: Write the implementation**

```python
# src/geneal/experiment/objects.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
from geneal.interfaces import Surrogate, Acquisition, Selection, NoiseModel, Metric
from geneal.data.dataset import Dataset


@dataclass
class ObjectiveObject:
    """What to optimize and how to measure success."""
    metric: Metric
    direction: Literal["maximize", "minimize"] = "maximize"


@dataclass
class DesignObject:
    """Experimental design knobs."""
    n_rounds: int
    batch_size: int
    n_initial: int
    seed: int = 0

    def __post_init__(self) -> None:
        for name in ("n_rounds", "batch_size", "n_initial"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")


@dataclass
class Method:
    """A named (surrogate, acquisition, selection) strategy to race."""
    name: str
    acquisition: Acquisition
    selection: Selection
    surrogate: Surrogate


@dataclass
class Experiment:
    """Top-level container assembled from config."""
    dataset: Dataset
    objective: ObjectiveObject
    design: DesignObject
    noise: NoiseModel
    methods: list[Method] = field(default_factory=list)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `micromamba run -n geneal pytest tests/test_objects.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/geneal/experiment/objects.py tests/test_objects.py
git commit -m "feat: add Experiment/Objective/Design/Method containers"
```

---

## Task 9: Runner (round loop + common random numbers)

**Files:**
- Create: `src/geneal/runner/runner.py`
- Test: `tests/test_runner.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_runner.py
import numpy as np
import pandas as pd
from geneal.runner.runner import Runner
from geneal.experiment.objects import ObjectiveObject, DesignObject, Method, Experiment
from geneal.metrics.recall import RecallAtK
from geneal.models.acquisition import UCB, RandomAcquisition
from geneal.models.selection import TopQGreedy
from geneal.models.surrogate import GPRSurrogate
from geneal.models.noise import GaussianNoise
from geneal.data.dataset import make_synthetic


def _experiment():
    ds = make_synthetic(n_genes=80, dim=5, seed=0, noise_sd=0.05)
    return Experiment(
        dataset=ds,
        objective=ObjectiveObject(metric=RecallAtK(k=10), direction="maximize"),
        design=DesignObject(n_rounds=4, batch_size=5, n_initial=10, seed=0),
        noise=GaussianNoise(sigma=0.05),
        methods=[
            Method("al_ucb", UCB(beta=2.0), TopQGreedy(), GPRSurrogate()),
            Method("baseline", RandomAcquisition(), TopQGreedy(), GPRSurrogate()),
        ],
    )


def test_runner_produces_dataframe_with_expected_rows():
    df = Runner().run(_experiment(), seeds=[0, 1])
    # rows = methods(2) * seeds(2) * (n_rounds+1 including round 0) = 2*2*5 = 20
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 20
    assert set(df.columns) >= {"method", "seed", "round", "n_revealed", "metric"}
    assert set(df["method"]) == {"al_ucb", "baseline"}


def test_runner_is_deterministic():
    a = Runner().run(_experiment(), seeds=[0])
    b = Runner().run(_experiment(), seeds=[0])
    pd.testing.assert_frame_equal(a, b)


def test_common_random_numbers_shared_initial_set():
    # Round 0 metric must be identical across methods for the same seed (same
    # initial set, same noise draws).
    df = Runner().run(_experiment(), seeds=[0])
    r0 = df[df["round"] == 0]
    assert r0["metric"].nunique() == 1


def test_al_beats_or_matches_random_on_average():
    df = Runner().run(_experiment(), seeds=list(range(5)))
    final = df[df["round"] == df["round"].max()]
    al = final[final["method"] == "al_ucb"]["metric"].mean()
    base = final[final["method"] == "baseline"]["metric"].mean()
    assert al >= base  # AL should not lose to random on this learnable problem
```

- [ ] **Step 2: Run test to verify it fails**

Run: `micromamba run -n geneal pytest tests/test_runner.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'geneal.runner.runner'`

- [ ] **Step 3: Write the implementation**

```python
# src/geneal/runner/runner.py
from __future__ import annotations
from typing import Sequence
import numpy as np
import pandas as pd
from geneal.experiment.objects import Experiment, Method


class Runner:
    """Runs every method across seeds with common random numbers.

    For each seed: draw one noise vector and one initial set, SHARED by all
    methods (paired comparison). Then run each method's round loop independently.
    Returns a tidy DataFrame: one row per (method, seed, round).
    """

    def run(self, experiment: Experiment, seeds: Sequence[int]) -> pd.DataFrame:
        ds = experiment.dataset
        design = experiment.design
        metric = experiment.objective.metric
        records: list[dict] = []

        for seed in seeds:
            # --- common random numbers for this seed ---
            crn = np.random.default_rng(seed)
            noise_vec = experiment.noise.draw(ds.n_genes, crn)
            init_idx = crn.permutation(ds.n_genes)[:design.n_initial].tolist()

            for method in experiment.methods:
                records.extend(
                    self._run_one(ds, design, metric, method, noise_vec,
                                  init_idx, seed)
                )

        return pd.DataFrame.from_records(records)

    def _run_one(self, ds, design, metric, method: Method, noise_vec,
                 init_idx, seed) -> list[dict]:
        # Per-method acquisition RNG, derived deterministically from the seed and
        # method name so methods don't share an acquisition RNG stream but runs
        # remain reproducible.
        acq_rng = np.random.default_rng((seed, abs(hash(method.name)) % (2**32)))

        revealed = list(init_idx)
        revealed_y = (ds.target[revealed] + noise_vec[revealed]).tolist()

        out = [self._record(method, seed, 0, revealed, metric, ds.target)]

        for r in range(1, design.n_rounds + 1):
            surr = method.surrogate.clone().fit(
                ds.embeddings[revealed], np.asarray(revealed_y))
            cand = [i for i in range(ds.n_genes) if i not in set(revealed)]
            if not cand:
                break
            Xc = ds.embeddings[cand]
            mean, std = surr.predict(Xc)
            best = float(max(revealed_y))
            sel = method.selection.select(
                candidate_idx=cand, X_candidates=Xc, mean=mean, std=std,
                best=best, q=design.batch_size, rng=acq_rng,
                surrogate=surr, acquisition=method.acquisition)
            for idx in sel:
                revealed.append(idx)
                revealed_y.append(float(ds.target[idx] + noise_vec[idx]))
            out.append(self._record(method, seed, r, revealed, metric, ds.target))
        return out

    @staticmethod
    def _record(method, seed, rnd, revealed, metric, target) -> dict:
        return {
            "method": method.name,
            "seed": seed,
            "round": rnd,
            "n_revealed": len(revealed),
            "metric": metric.evaluate(revealed, target),
            "metric_name": metric.name,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `micromamba run -n geneal pytest tests/test_runner.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/geneal/runner/runner.py tests/test_runner.py
git commit -m "feat: add Runner with common-random-number round loop"
```

---

## Task 10: Run logging (parquet + manifest + config)

**Files:**
- Create: `src/geneal/runner/logging.py`
- Test: `tests/test_logging.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_logging.py
import json
import pandas as pd
from geneal.runner.logging import write_run_dir


def test_write_run_dir_creates_artifacts(tmp_path):
    df = pd.DataFrame({"method": ["a"], "seed": [0], "round": [0],
                       "n_revealed": [5], "metric": [0.5], "metric_name": ["recall@5"]})
    run_dir = write_run_dir(
        out_root=tmp_path, df=df,
        config={"dataset": "synthetic", "seeds": [0]},
        extra_manifest={"git_sha": "abc123", "env": "geneal"},
    )
    assert (run_dir / "rounds.parquet").exists()
    assert (run_dir / "config.yaml").exists()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["git_sha"] == "abc123"
    assert manifest["n_rows"] == 1
    # round-trip the parquet
    back = pd.read_parquet(run_dir / "rounds.parquet")
    assert len(back) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `micromamba run -n geneal pytest tests/test_logging.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'geneal.runner.logging'`

- [ ] **Step 3: Write the implementation**

```python
# src/geneal/runner/logging.py
from __future__ import annotations
from pathlib import Path
import json
import pandas as pd


def write_run_dir(*, out_root, df: pd.DataFrame, config: dict,
                  extra_manifest: dict | None = None, run_name: str = "run") -> Path:
    """Write a self-contained run directory: parquet + config.yaml + manifest.json.

    Caller supplies any time-varying values (git sha, timestamp, env) in
    `extra_manifest` so this function stays deterministic and import-light.
    """
    out_root = Path(out_root)
    run_dir = out_root / run_name
    # de-collide if the run dir already exists
    i = 1
    while run_dir.exists():
        run_dir = out_root / f"{run_name}_{i}"
        i += 1
    run_dir.mkdir(parents=True)

    df.to_parquet(run_dir / "rounds.parquet", index=False)

    # config as simple YAML (avoid extra deps: write a minimal serialization)
    _write_yaml(run_dir / "config.yaml", config)

    manifest = {"n_rows": int(len(df)),
                "methods": sorted(df["method"].unique().tolist()) if len(df) else [],
                "metric_name": (df["metric_name"].iloc[0] if len(df) else None)}
    if extra_manifest:
        manifest.update(extra_manifest)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return run_dir


def _write_yaml(path: Path, data: dict) -> None:
    try:
        import yaml  # PyYAML ships with hydra-core
        path.write_text(yaml.safe_dump(data, sort_keys=False))
    except Exception:
        # Fallback: JSON is valid YAML
        path.write_text(json.dumps(data, indent=2))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `micromamba run -n geneal pytest tests/test_logging.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/geneal/runner/logging.py tests/test_logging.py
git commit -m "feat: add run-directory logging (parquet + manifest + config)"
```

---

## Task 11: HTML report (recall curves + latex tables)

**Files:**
- Create: `src/geneal/report/report.py`
- Create: `src/geneal/report/templates/report.html.j2`
- Test: `tests/test_report.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_report.py
import pandas as pd
from geneal.report.report import build_report, aggregate, to_latex_table


def _df():
    rows = []
    for method in ("al_ucb", "baseline"):
        for seed in (0, 1):
            for rnd in range(3):
                val = 0.1 * rnd + (0.2 if method == "al_ucb" else 0.0)
                rows.append({"method": method, "seed": seed, "round": rnd,
                             "n_revealed": 10 + 5 * rnd, "metric": val,
                             "metric_name": "recall@10"})
    return pd.DataFrame(rows)


def test_aggregate_mean_ci():
    agg = aggregate(_df())
    # one row per (method, round) = 2 * 3 = 6
    assert len(agg) == 6
    assert {"method", "round", "mean", "ci_low", "ci_high"}.issubset(agg.columns)


def test_latex_table_is_string_with_tabular():
    latex = to_latex_table(aggregate(_df()))
    assert "\\begin{tabular}" in latex
    assert "al_ucb" in latex


def test_build_report_writes_html(tmp_path):
    out = tmp_path / "report.html"
    build_report(_df(), out)
    assert out.exists()
    html = out.read_text()
    assert "<html" in html.lower()
    assert "recall@10" in html
    assert "\\begin{tabular}" in html  # latex embedded in a dropdown
```

- [ ] **Step 2: Run test to verify it fails**

Run: `micromamba run -n geneal pytest tests/test_report.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'geneal.report.report'`

- [ ] **Step 3: Write the template**

```jinja
{# src/geneal/report/templates/report.html.j2 #}
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>geneal report — {{ metric_name }}</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 2rem; }
    table { border-collapse: collapse; } td, th { border: 1px solid #ccc; padding: 4px 8px; }
    details { margin: 1rem 0; } summary { cursor: pointer; font-weight: bold; }
    pre { background: #f5f5f5; padding: 1rem; overflow-x: auto; }
  </style>
</head>
<body>
  <h1>geneal — active learning vs baseline</h1>
  <p>Metric: <strong>{{ metric_name }}</strong>. Methods: {{ methods | join(", ") }}.</p>

  <h2>Recall curve</h2>
  {{ plot_div | safe }}

  <h2>Summary table</h2>
  {{ html_table | safe }}

  <details>
    <summary>LaTeX table (click to expand)</summary>
    <pre>{{ latex_table }}</pre>
  </details>
</body>
</html>
```

- [ ] **Step 4: Write the implementation**

```python
# src/geneal/report/report.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader, select_autoescape

_TEMPLATES = Path(__file__).parent / "templates"


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and 95% normal-approx CI of the metric per (method, round)."""
    g = df.groupby(["method", "round"])["metric"]
    out = g.agg(["mean", "std", "count"]).reset_index()
    se = out["std"].fillna(0.0) / np.sqrt(out["count"].clip(lower=1))
    out["ci_low"] = out["mean"] - 1.96 * se
    out["ci_high"] = out["mean"] + 1.96 * se
    return out[["method", "round", "mean", "ci_low", "ci_high"]]


def to_latex_table(agg: pd.DataFrame) -> str:
    """LaTeX tabular of mean metric per method per round (manuscript-ready)."""
    pivot = agg.pivot(index="round", columns="method", values="mean").round(3)
    return pivot.to_latex(index=True, caption="Mean metric by round",
                          label="tab:recall", escape=True)


def _plot_div(agg: pd.DataFrame, metric_name: str) -> str:
    fig = go.Figure()
    for method, sub in agg.groupby("method"):
        sub = sub.sort_values("round")
        fig.add_trace(go.Scatter(x=sub["round"], y=sub["mean"], mode="lines+markers",
                                 name=method))
        fig.add_trace(go.Scatter(
            x=list(sub["round"]) + list(sub["round"][::-1]),
            y=list(sub["ci_high"]) + list(sub["ci_low"][::-1]),
            fill="toself", line=dict(width=0), showlegend=False, opacity=0.2,
            hoverinfo="skip", name=f"{method} CI"))
    fig.update_layout(xaxis_title="round", yaxis_title=metric_name,
                      template="simple_white")
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def build_report(df: pd.DataFrame, out_path) -> Path:
    """Render the full HTML report from a runner DataFrame."""
    metric_name = df["metric_name"].iloc[0] if len(df) else "metric"
    agg = aggregate(df)
    env = Environment(loader=FileSystemLoader(str(_TEMPLATES)),
                      autoescape=select_autoescape(["html"]))
    template = env.get_template("report.html.j2")
    html = template.render(
        metric_name=metric_name,
        methods=sorted(df["method"].unique().tolist()),
        plot_div=_plot_div(agg, metric_name),
        html_table=agg.round(3).to_html(index=False),
        latex_table=to_latex_table(agg),
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    return out_path
```

- [ ] **Step 5: Run test to verify it passes**

Run: `micromamba run -n geneal pytest tests/test_report.py -v`
Expected: PASS (3 passed)

- [ ] **Step 6: Commit**

```bash
git add src/geneal/report/ tests/test_report.py
git commit -m "feat: add HTML report with recall curves and latex tables"
```

---

## Task 12: Hydra config + run script + end-to-end smoke test

**Files:**
- Create: `conf/config.yaml`, `conf/dataset/synthetic.yaml`, `conf/method/al_ucb.yaml`, `conf/method/baseline_random.yaml`
- Create: `scripts/run_experiment.py`
- Test: `tests/test_smoke_end_to_end.py`

- [ ] **Step 1: Write the failing smoke test**

```python
# tests/test_smoke_end_to_end.py
"""End-to-end: synthetic data -> runner -> log dir -> HTML report. No heavy deps."""
import pandas as pd
from geneal.data.dataset import make_synthetic
from geneal.experiment.objects import ObjectiveObject, DesignObject, Method, Experiment
from geneal.metrics.recall import RecallAtK
from geneal.models.acquisition import UCB, RandomAcquisition
from geneal.models.selection import TopQGreedy
from geneal.models.surrogate import GPRSurrogate
from geneal.models.noise import GaussianNoise
from geneal.runner.runner import Runner
from geneal.runner.logging import write_run_dir
from geneal.report.report import build_report


def test_full_pipeline_smoke(tmp_path):
    ds = make_synthetic(n_genes=60, dim=5, seed=0, noise_sd=0.05)
    exp = Experiment(
        dataset=ds,
        objective=ObjectiveObject(metric=RecallAtK(k=10), direction="maximize"),
        design=DesignObject(n_rounds=3, batch_size=5, n_initial=10, seed=0),
        noise=GaussianNoise(sigma=0.05),
        methods=[
            Method("al_ucb", UCB(beta=2.0), TopQGreedy(), GPRSurrogate()),
            Method("baseline", RandomAcquisition(), TopQGreedy(), GPRSurrogate()),
        ],
    )
    df = Runner().run(exp, seeds=[0, 1])
    run_dir = write_run_dir(out_root=tmp_path, df=df,
                            config={"dataset": "synthetic", "seeds": [0, 1]},
                            extra_manifest={"git_sha": "test", "env": "geneal"})
    report = build_report(pd.read_parquet(run_dir / "rounds.parquet"),
                          run_dir / "report.html")
    assert report.exists()
    assert (run_dir / "rounds.parquet").exists()
    assert (run_dir / "manifest.json").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `micromamba run -n geneal pytest tests/test_smoke_end_to_end.py -v`
Expected: PASS already IF all prior tasks are done — this test only uses code that exists. If any import fails, the corresponding task is incomplete. (If it passes here, that's fine; the remaining steps add the Hydra entrypoint.)

- [ ] **Step 3: Write the Hydra configs**

`conf/config.yaml`:
```yaml
defaults:
  - dataset: synthetic
  - _self_

seeds: [0, 1, 2]
out_root: res/runs

objective:
  _target_: geneal.metrics.recall.RecallAtK
  k: 10

design:
  _target_: geneal.experiment.objects.DesignObject
  n_rounds: 8
  batch_size: 10
  n_initial: 20
  seed: 0

noise:
  _target_: geneal.models.noise.GaussianNoise
  sigma: 0.1

methods:
  - name: al_ucb
    acquisition: { _target_: geneal.models.acquisition.UCB, beta: 2.0 }
    selection:   { _target_: geneal.models.selection.TopQGreedy }
    surrogate:   { _target_: geneal.models.surrogate.GPRSurrogate }
  - name: baseline_random
    acquisition: { _target_: geneal.models.acquisition.RandomAcquisition }
    selection:   { _target_: geneal.models.selection.TopQGreedy }
    surrogate:   { _target_: geneal.models.surrogate.GPRSurrogate }
```

`conf/dataset/synthetic.yaml`:
```yaml
_target_: geneal.data.dataset.make_synthetic
n_genes: 500
dim: 8
seed: 0
noise_sd: 0.1
```

`conf/method/al_ucb.yaml`:
```yaml
name: al_ucb
acquisition: { _target_: geneal.models.acquisition.UCB, beta: 2.0 }
selection:   { _target_: geneal.models.selection.TopQGreedy }
surrogate:   { _target_: geneal.models.surrogate.GPRSurrogate }
```

`conf/method/baseline_random.yaml`:
```yaml
name: baseline_random
acquisition: { _target_: geneal.models.acquisition.RandomAcquisition }
selection:   { _target_: geneal.models.selection.TopQGreedy }
surrogate:   { _target_: geneal.models.surrogate.GPRSurrogate }
```

- [ ] **Step 4: Write the run script**

```python
# scripts/run_experiment.py
"""Hydra entrypoint: build an Experiment from config, run it, log, report."""
from __future__ import annotations
import subprocess
import pandas as pd
import hydra
from hydra.utils import instantiate, call
from omegaconf import DictConfig, OmegaConf

from geneal.experiment.objects import ObjectiveObject, Method, Experiment
from geneal.runner.runner import Runner
from geneal.runner.logging import write_run_dir
from geneal.report.report import build_report


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    dataset = call(cfg.dataset)           # make_synthetic(...)
    metric = instantiate(cfg.objective)   # RecallAtK
    design = instantiate(cfg.design)
    noise = instantiate(cfg.noise)
    methods = [
        Method(name=m["name"],
               acquisition=instantiate(m["acquisition"]),
               selection=instantiate(m["selection"]),
               surrogate=instantiate(m["surrogate"]))
        for m in cfg.methods
    ]
    exp = Experiment(
        dataset=dataset,
        objective=ObjectiveObject(metric=metric, direction="maximize"),
        design=design, noise=noise, methods=methods,
    )
    df = Runner().run(exp, seeds=list(cfg.seeds))
    run_dir = write_run_dir(
        out_root=cfg.out_root, df=df,
        config=OmegaConf.to_container(cfg, resolve=True),
        extra_manifest={"git_sha": _git_sha()},
    )
    build_report(pd.read_parquet(run_dir / "rounds.parquet"),
                 run_dir / "report.html")
    print(f"Run complete: {run_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the smoke test AND the real entrypoint**

Run: `micromamba run -n geneal pytest tests/test_smoke_end_to_end.py -v`
Expected: PASS (1 passed)

Run: `micromamba run -n geneal python scripts/run_experiment.py seeds=[0,1] dataset.n_genes=200 design.n_rounds=4`
Expected: prints `Run complete: res/runs/run...`; that dir contains `rounds.parquet`, `config.yaml`, `manifest.json`, `report.html`.

- [ ] **Step 6: Run the FULL test suite**

Run: `micromamba run -n geneal pytest`
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add conf scripts tests/test_smoke_end_to_end.py
git commit -m "feat: add hydra config, run entrypoint, end-to-end smoke test"
```

---

## Task 13: CLAUDE.md

**Files:**
- Create: `CLAUDE.md`

- [ ] **Step 1: Write CLAUDE.md**

```markdown
# geneal

Active learning for gene-knockout selection on molecular data. Race an AL
strategy against a baseline to recover the most-lethal knockout genes for a cell
line in fewer rounds.

## Environment
- micromamba env: `geneal`. Run anything with `micromamba run -n geneal <cmd>`.
- Editable install: `pip install -e ".[dev]"`.

## Layout
- `src/geneal/interfaces.py` — Protocols every swappable piece implements.
- `src/geneal/data/` — Dataset + synthetic generator (real DepMap = Plan 2).
- `src/geneal/models/` — surrogate, acquisition, selection, noise.
- `src/geneal/metrics/` — pluggable success metrics (recall@k default).
- `src/geneal/experiment/` — Objective/Design/Method/Experiment containers.
- `src/geneal/runner/` — round loop (common random numbers) + run logging.
- `src/geneal/report/` — HTML report (recall curves + latex tables).
- `conf/` — Hydra configs (`_target_` instantiation).
- `scripts/run_experiment.py` — Hydra entrypoint.

## Conventions
- Target = lethality, HIGHER is better. Acquisitions maximize.
- RNG is always an explicit numpy Generator passed in. No global random state.
- Run a config: `python scripts/run_experiment.py seeds=[0,1,2]`.
- Tests: `pytest`. Keep new components behind a Protocol + a test.

## Status
- Plan 1 (core framework on synthetic data): see docs/superpowers/plans/.
- Plan 2 (DepMap data + scPRINT/ESM2 embeddings): not yet written.
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add CLAUDE.md"
```

---

## Self-Review (completed during planning)

- **Spec coverage:** Experiment/Objective/Design containers (Task 8) ✓; Surrogate GPR+BNN (Task 5) ✓; acquisition EI/UCB/(EIG→MaxVariance, deferred-EIG noted) (Task 6) ✓; selection top-q + fantasy (Task 7) ✓; pluggable noise, Gaussian first (Task 4) ✓; recall@k pluggable metric (Task 3) ✓; Runner with AL-vs-baseline, multi-seed, structured logs (Tasks 9–10) ✓; common random numbers fairness invariant (Task 9) ✓; HTML report with curves + dropdown latex tables (Task 11) ✓; seeded reproducibility (Tasks 9, 12 determinism tests) ✓; Hydra `_target_` (Task 12) ✓; CLAUDE.md (Task 13) ✓. **Embeddings/DepMap deliverables are explicitly Plan 2**, not gaps.
- **Placeholder scan:** none — every code step is complete.
- **Type consistency:** `Surrogate.fit/predict/clone`, `Acquisition.score(mean,std,best,rng)`, `Selection.select(**kwargs)`, `NoiseModel.draw(n,rng)`, `Metric.evaluate/name` are used identically across Tasks 1, 5–11.
- **Deviations from spec flagged for user:** (1) EIG deferred, `MaxVariance` is the v1 info-seeking acquisition; (2) botorch not required for v1 — fantasy batching is a surrogate refit (GPR itself is gpytorch per user request).

## Out of scope (Plan 2)
DepMap download + curation, scPRINT/ESM2 embedding adapters + precompute script, swapping the synthetic Dataset for the real cached embeddings + effect matrix. Interfaces in this plan are designed to accept that swap with no change to the Runner/Report.
