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
