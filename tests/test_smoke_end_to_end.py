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
