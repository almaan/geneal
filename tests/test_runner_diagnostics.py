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
    assert np.isnan(df[df["round"] == 0]["batch_quality"].iloc[0])
    later = df[df["round"] >= 1]
    assert later["batch_quality"].notna().all()
    assert later["batch_diversity"].notna().all()
