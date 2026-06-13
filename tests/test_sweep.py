# add to tests/test_sweep.py (device part)
import numpy as np, pytest
from geneal.models.surrogate import GPRSurrogate
from geneal.data.dataset import make_synthetic

def test_gpr_device_cpu_default_ok():
    ds = make_synthetic(n_genes=60, dim=4, seed=0)
    surr = GPRSurrogate(n_iters=50, device="cpu").fit(ds.embeddings[:40], ds.target[:40])
    m, s = surr.predict(ds.embeddings[40:])
    assert m.shape == (20,) and np.all(s >= 0)
    assert surr.clone().device == "cpu"


# tests/test_sweep.py (sweep part — append to the file)
import pandas as pd
from geneal.runner.sweep import run_sweep
from geneal.experiment.objects import ObjectiveObject, DesignObject, Method, Experiment
from geneal.metrics.recall import RecallAtK
from geneal.models.acquisition import UCB, RandomAcquisition
from geneal.models.selection import TopQGreedy
from geneal.models.surrogate import GPRSurrogate
from geneal.models.noise import GaussianNoise
from geneal.data.dataset import make_synthetic


def _build(cell_line):
    # toy: cell_line is just a seed offset into synthetic data
    ds = make_synthetic(n_genes=50, dim=4, seed=int(cell_line), noise_sd=0.05)
    return Experiment(
        ds, ObjectiveObject(RecallAtK(k=5), "maximize"),
        DesignObject(n_rounds=2, batch_size=4, n_initial=8, seed=0),
        GaussianNoise(0.05),
        [Method("al", UCB(2.0), TopQGreedy(), GPRSurrogate(n_iters=40)),
         Method("random", RandomAcquisition(), TopQGreedy(), GPRSurrogate(n_iters=40))])


def test_run_sweep_parallel_has_all_cells_and_methods():
    df = run_sweep(_build, cell_lines=["0", "1", "2"], seeds=[0, 1], max_workers=2)
    assert set(df["cell_line"]) == {"0", "1", "2"}
    assert set(df["method"]) == {"al", "random"}
    # 3 cells * 2 methods * 2 seeds * (2 rounds + round0=3) = 36 rows
    assert len(df) == 36
