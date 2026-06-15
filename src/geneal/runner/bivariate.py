# src/geneal/runner/bivariate.py
from __future__ import annotations
import numpy as np
from geneal.models.multiobjective import pareto_front, hypervolume2d, mc_ehvi


class BivariateALRunner:
    """Active learning over TWO objectives (efficacy↑, toxicity↓), both unknown.
    Each acquisition reveals both labels (one assay profiles both contexts).
    Batch acquisition = greedy EHVI over the two GP posteriors. Toxicity is
    minimized, so we maximize (efficacy, −toxicity) in objective space."""

    def __init__(self, surrogate_factory, noise, ehvi_samples: int = 64,
                 shortlist: int = 200, joint_factory=None):
        self.make = surrogate_factory
        self.noise = noise
        self.ehvi_samples = ehvi_samples
        self.shortlist = shortlist   # run EHVI only on the top-N most promising candidates
        self.joint_factory = joint_factory  # if set: one multitask GP + correlated EHVI

    def run(self, X, efficacy, toxicity, n_initial, n_rounds, batch_size, seed,
            tox_known=False, eff_known=False):
        """tox_known / eff_known: treat that objective as an ORACLE (true value,
        zero uncertainty). tox_known only = the a-priori-toxicity condition (learn
        efficacy, know toxicity). BOTH = full-information oracle = absolute
        hypervolume ceiling. Default both False = learn both."""
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
                    "hypervolume": hypervolume2d(pts, ref),
                    "revealed": list(revealed)}

        hist = [record(0)]
        for r in range(1, n_rounds + 1):
            cand = [i for i in range(n) if i not in set(revealed)]
            if not cand:
                break
            joint_gp = None
            if self.joint_factory is not None and not (eff_known or tox_known):
                # ONE multitask GP over (efficacy, toxicity): correlated posterior.
                Ye = np.array([eff[i] + noise_e[i] for i in revealed])
                Yt = np.array([tox[i] + noise_t[i] for i in revealed])
                joint_gp = self.joint_factory().fit(X[revealed], np.column_stack([Ye, Yt]))
                m2, s2 = joint_gp.predict(X[cand])
                me, sde = m2[:, 0], s2[:, 0]
                mt, sdt = m2[:, 1], s2[:, 1]
            else:
                if eff_known:
                    me = eff[cand]; sde = np.zeros(len(cand))    # oracle efficacy
                else:
                    ye = np.array([eff[i] + noise_e[i] for i in revealed])
                    me, sde = self.make().fit(X[revealed], ye).predict(X[cand])
                if tox_known:
                    mt = tox[cand]; sdt = np.zeros(len(cand))    # oracle toxicity
                else:
                    yt = np.array([tox[i] + noise_t[i] for i in revealed])
                    mt, sdt = self.make().fit(X[revealed], yt).predict(X[cand])
            mean = np.column_stack([me, -mt])     # maximize eff, -tox
            std = np.column_stack([sde, sdt])
            # shortlist: run the (slow) MC-EHVI only on the most promising
            # candidates by OPTIMISTIC selectivity (UCB on both objectives).
            if self.shortlist and len(cand) > self.shortlist:
                opt = (mean + std).sum(axis=1)    # optimistic eff + (-tox)
                keep = np.argsort(opt)[::-1][:self.shortlist]
                cand = [cand[i] for i in keep]
                mean, std = mean[keep], std[keep]
            # correlated EHVI: per-candidate covariance in (eff, -tox) space
            cov = None
            if joint_gp is not None:
                _, tcov = joint_gp.predict_taskcov(X[cand])       # (m,2,2) in (eff,tox)
                D = np.array([[1.0, 0.0], [0.0, -1.0]])           # flip tox -> -tox
                cov = np.einsum("ij,mjk,kl->mil", D, tcov, D)
            # current front from revealed objective points
            pts = np.array([[eff[i], -tox[i]] for i in revealed])
            front = pts[pareto_front(pts)]
            # greedy-EHVI batch with fantasies (add picked mean to the front)
            picked_local = []
            cur_front = front.copy()
            avail = list(range(len(cand)))
            for _ in range(min(batch_size, len(cand))):
                cov_av = cov[avail] if cov is not None else None
                ehvi = mc_ehvi(mean[avail], std[avail], cur_front, ref,
                               acq_rng, self.ehvi_samples, cov=cov_av)
                j = avail[int(np.argmax(ehvi))]
                picked_local.append(j); avail.remove(j)
                cur_front = np.vstack([cur_front, mean[j]])  # fantasize at mean
            for j in picked_local:
                revealed.append(cand[j])
            hist.append(record(r))
        return hist
