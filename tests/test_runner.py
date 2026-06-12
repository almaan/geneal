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
