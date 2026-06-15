# src/geneal/models/multitask.py
"""Multitask (coregionalized) Gaussian process for jointly modeling two
correlated objectives -- efficacy and toxicity.

When efficacy and toxicity are correlated (empirically ~0.8 in DepMap), modeling
them JOINTLY lets each output borrow strength from the other: the (more plentiful,
acquisition-targeted) efficacy labels sharpen the toxicity prediction, which is
exactly the regime where the learned-safety methods (trunc_pred, EHVI) struggle.

Implementation: an ExactGP with an intrinsic-coregionalization kernel
  K((x,t),(x',t')) = k_data(x,x') * B[t,t'],
where k_data is Matern-5/2 over the shared inputs and B is a learned task
covariance (gpytorch MultitaskKernel, rank r). Both outputs are observed at every
revealed gene (block design), so a standard MultitaskGP fits.

API mirrors GPRSurrogate but is multi-output:
  fit(X, Y)            Y shape (n, 2)
  predict(X)           -> (mean (n,2), std (n,2))            [diagonal, cheap]
  predict_taskcov(X)   -> (mean (n,2), cov (n,2,2))          [per-point task cov]
"""
from __future__ import annotations
import numpy as np


class MultiTaskGPR:
    def __init__(self, n_iters: int = 120, lr: float = 0.1, rank: int = 1,
                 nu: float = 2.5, num_tasks: int = 2, device: str = "cpu") -> None:
        self.n_iters = n_iters
        self.lr = lr
        self.rank = rank
        self.nu = nu
        self.num_tasks = num_tasks
        self.device = device
        self._model = None
        self._likelihood = None
        self._x_mean = self._x_std = None
        self._y_mean = self._y_std = None

    def _build(self, tx, ty):
        import gpytorch
        nu, num_tasks, rank = self.nu, self.num_tasks, self.rank

        class _MTGP(gpytorch.models.ExactGP):
            def __init__(self, tx, ty, likelihood):
                super().__init__(tx, ty, likelihood)
                self.mean_module = gpytorch.means.MultitaskMean(
                    gpytorch.means.ConstantMean(), num_tasks=num_tasks)
                self.covar_module = gpytorch.kernels.MultitaskKernel(
                    gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=nu)),
                    num_tasks=num_tasks, rank=rank)

            def forward(self, x):
                return gpytorch.distributions.MultitaskMultivariateNormal(
                    self.mean_module(x), self.covar_module(x))

        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
        return _MTGP(tx, ty, likelihood), likelihood

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "MultiTaskGPR":
        import torch, gpytorch
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if Y.ndim == 1:
            Y = Y[:, None]
        self.num_tasks = Y.shape[1]
        self._x_mean, self._x_std = X.mean(0), X.std(0) + 1e-8
        self._y_mean, self._y_std = Y.mean(0), Y.std(0) + 1e-8
        tx = torch.tensor((X - self._x_mean) / self._x_std, dtype=torch.float32, device=self.device)
        ty = torch.tensor((Y - self._y_mean) / self._y_std, dtype=torch.float32, device=self.device)
        self._model, self._likelihood = self._build(tx, ty)
        self._model.train(); self._likelihood.train()
        opt = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self._likelihood, self._model)
        for _ in range(self.n_iters):
            opt.zero_grad()
            loss = -mll(self._model(tx), ty)
            loss.backward(); opt.step()
        self._model.eval(); self._likelihood.eval()
        return self

    def _posterior(self, X):
        import torch, gpytorch
        X = np.asarray(X, dtype=np.float64)
        tx = torch.tensor((X - self._x_mean) / self._x_std, dtype=torch.float32, device=self.device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            return self._likelihood(self._model(tx))

    def predict(self, X):
        """(mean (n,T), std (n,T)) on the original scale (diagonal only)."""
        import numpy as _np
        post = self._posterior(X)
        m = post.mean.detach().cpu().numpy()                 # (n, T) standardized
        v = post.variance.detach().cpu().numpy()             # (n, T) standardized
        mean = m * self._y_std + self._y_mean
        std = _np.sqrt(_np.clip(v, 0, None)) * self._y_std
        return mean, std

    def predict_taskcov(self, X):
        """(mean (n,T), cov (n,T,T)) on the original scale -- the per-input task
        covariance (incl. cross-task), used for correlated EHVI sampling."""
        import numpy as _np
        post = self._posterior(X)
        m = post.mean.detach().cpu().numpy()                                  # (n,T)
        T = self.num_tasks
        n = m.shape[0]
        full = post.covariance_matrix.detach().cpu().numpy()                  # (n*T, n*T), interleaved
        # gpytorch MultitaskMVN is interleaved: index = i*T + t
        cov = _np.empty((n, T, T))
        for i in range(n):
            blk = full[i * T:(i + 1) * T, i * T:(i + 1) * T]
            cov[i] = blk
        # rescale standardized cov back to original units: cov_ts * ys_t * ys_s
        scale = _np.outer(self._y_std, self._y_std)
        cov = cov * scale[None, :, :]
        mean = m * self._y_std + self._y_mean
        return mean, cov

    def clone(self) -> "MultiTaskGPR":
        return MultiTaskGPR(n_iters=self.n_iters, lr=self.lr, rank=self.rank,
                            nu=self.nu, num_tasks=self.num_tasks, device=self.device)
