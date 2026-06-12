# src/geneal/runner/runner.py
from __future__ import annotations
from typing import Sequence
import hashlib
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
        # remain reproducible. Uses a stable (non-salted) hash of the name so runs
        # reproduce across processes, not just within one (builtin hash() is
        # salted per-process via PYTHONHASHSEED).
        name_hash = int.from_bytes(
            hashlib.sha256(method.name.encode()).digest()[:4], "big")
        acq_rng = np.random.default_rng((seed, name_hash))

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
