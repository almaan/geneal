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

    def predict_cov(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Joint LATENT posterior covariance (the f-posterior, without observation
        noise) on the original target scale. The latent covariance is the right
        object for an outcome-diversity kernel: adding the likelihood noise to the
        diagonal but not the off-diagonals would shrink every correlation toward 0
        and artificially weaken the anti-redundancy signal. Uses the exact
        predictive covariance (not the LOVE/fast_pred_var Lanczos approximation),
        which can be inconsistent between diagonal and off-diagonal and break PSD.
        """
        import torch
        if self._model is None:
            raise RuntimeError("GPRSurrogate.predict_cov called before fit")
        X = np.asarray(X, dtype=np.float64)
        tx = torch.tensor((X - self._x_mean) / self._x_std, dtype=torch.float32)
        self._model.eval()
        self._likelihood.eval()
        with torch.no_grad():
            post = self._model(tx)  # latent f-posterior, no observation noise
            mean = post.mean.numpy()
            cov = post.covariance_matrix.numpy()
        return mean * self._y_std + self._y_mean, cov * (self._y_std ** 2)

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
        pyro.deterministic("latent", out)  # noise-free function value, for predict_cov
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

    def predict_cov(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Joint LATENT posterior covariance (sample covariance of the noise-free
        function values), not the observation `obs` site. The latent covariance is
        the right object for an outcome-diversity kernel — including observation
        noise would inflate the diagonal relative to the off-diagonals and shrink
        every correlation toward 0.
        """
        import torch
        import pyro
        from pyro.infer import Predictive
        if self._guide is None:
            raise RuntimeError("BNNSurrogate.predict_cov called before fit")
        X = np.asarray(X, dtype=np.float64)
        xt = torch.tensor((X - self._x_mean) / self._x_std, dtype=torch.float32)
        pred = Predictive(self._model, guide=self._guide,
                          num_samples=self.n_predict, return_sites=["latent"])
        with torch.no_grad():
            lat = pred(xt)["latent"].detach().numpy()  # (n_predict, n), standardized
        lat = lat.reshape(self.n_predict, -1)
        mean = lat.mean(0) * self._y_std + self._y_mean
        cov = np.cov(lat, rowvar=False) * (self._y_std ** 2)
        return mean, np.atleast_2d(cov)

    def clone(self) -> "BNNSurrogate":
        return BNNSurrogate(hidden=self.hidden, n_steps=self.n_steps, lr=self.lr,
                            prior_scale=self.prior_scale, n_predict=self.n_predict,
                            seed=self.seed)
