# src/geneal/runner/runner.py
from __future__ import annotations
from typing import Sequence
import hashlib
import numpy as np
import pandas as pd
from geneal.experiment.objects import Experiment, Method
from geneal.metrics.diagnostics import batch_quality, batch_diversity


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
        if design.n_initial >= ds.n_genes:
            raise ValueError(
                f"n_initial ({design.n_initial}) must be < n_genes "
                f"({ds.n_genes}); otherwise the initial set reveals everything "
                "and there is nothing left to acquire.")
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
        # remain reproducible. Uses a stable (non-salted) hash of the name so runs
        # reproduce across processes, not just within one (builtin hash() is
        # salted per-process via PYTHONHASHSEED).
        name_hash = int.from_bytes(
            hashlib.sha256(method.name.encode()).digest()[:4], "big")
        acq_rng = np.random.default_rng((seed, name_hash))

        revealed = list(init_idx)
        revealed_y = (ds.target[revealed] + noise_vec[revealed]).tolist()

        out = [self._record(method, seed, 0, revealed, metric, ds.target, ds)]

        for r in range(1, design.n_rounds + 1):
            X_train = ds.embeddings[revealed]
            y_train = np.asarray(revealed_y)
            surr = method.surrogate.clone().fit(X_train, y_train)
            seen = set(revealed)
            cand = [i for i in range(ds.n_genes) if i not in seen]
            if not cand:
                break
            Xc = ds.embeddings[cand]
            mean, std = surr.predict(Xc)
            best = float(max(revealed_y))
            sel = method.selection.select(
                candidate_idx=cand, X_candidates=Xc, mean=mean, std=std,
                best=best, q=design.batch_size, rng=acq_rng,
                surrogate=surr, acquisition=method.acquisition,
                X_train=X_train, y_train=y_train)
            for idx in sel:
                revealed.append(idx)
                revealed_y.append(float(ds.target[idx] + noise_vec[idx]))
            out.append(self._record(method, seed, r, revealed, metric,
                                    ds.target, ds, batch=sel))
        return out

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
