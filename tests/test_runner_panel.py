# tests/test_runner_panel.py
import numpy as np
from geneal.runner.runner import Runner
from geneal.experiment.objects import ObjectiveObject, DesignObject, Method, Experiment
from geneal.metrics.recall import RecallAtK
from geneal.metrics.panel import MaxValue, CumulativeDiversity, AlphaNDCG
from geneal.models.acquisition import UCB
from geneal.models.selection import TopQGreedy
from geneal.models.surrogate import GPRSurrogate
from geneal.models.noise import GaussianNoise
from geneal.data.dataset import make_synthetic


def test_runner_records_eval_panel_columns():
    ds = make_synthetic(n_genes=60, dim=5, seed=0, noise_sd=0.05)
    exp = Experiment(
        dataset=ds,
        objective=ObjectiveObject(metric=RecallAtK(k=10), direction="maximize"),
        design=DesignObject(n_rounds=3, batch_size=5, n_initial=10, seed=0),
        noise=GaussianNoise(sigma=0.05),
        methods=[Method("al", UCB(2.0), TopQGreedy(), GPRSurrogate(n_iters=50))],
    )
    panel = [MaxValue(), CumulativeDiversity(), AlphaNDCG(k=10, n_clusters=5, seed=0)]
    df = Runner().run(exp, seeds=[0], eval_panel=panel)
    for col in ["eval_max_value", "eval_diversity", "eval_alpha_ndcg@10"]:
        assert col in df.columns
    # max_value should be non-decreasing over rounds (cumulative best)
    s = df.sort_values("round")["eval_max_value"].to_numpy()
    assert np.all(np.diff(s) >= -1e-9)


def test_runner_without_panel_unchanged():
    ds = make_synthetic(n_genes=40, dim=4, seed=0)
    exp = Experiment(ds, ObjectiveObject(RecallAtK(k=5), "maximize"),
                     DesignObject(n_rounds=2, batch_size=4, n_initial=8, seed=0),
                     GaussianNoise(0.05),
                     [Method("al", UCB(2.0), TopQGreedy(), GPRSurrogate(n_iters=40))])
    df = Runner().run(exp, seeds=[0])
    assert "metric" in df.columns
    assert not any(c.startswith("eval_") for c in df.columns)
