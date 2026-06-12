# tests/test_objects.py
from geneal.experiment.objects import ObjectiveObject, DesignObject, Method, Experiment
from geneal.metrics.recall import RecallAtK
from geneal.models.acquisition import UCB, RandomAcquisition
from geneal.models.selection import TopQGreedy
from geneal.models.surrogate import GPRSurrogate
from geneal.models.noise import GaussianNoise
from geneal.data.dataset import make_synthetic


def test_design_defaults_and_validation():
    d = DesignObject(n_rounds=5, batch_size=10, n_initial=20, seed=0)
    assert d.n_rounds == 5 and d.batch_size == 10


def test_objective_holds_metric_and_direction():
    o = ObjectiveObject(metric=RecallAtK(k=10), direction="maximize")
    assert o.metric.name == "recall@10"
    assert o.direction == "maximize"


def test_experiment_assembles():
    ds = make_synthetic(n_genes=50, dim=4, seed=0)
    exp = Experiment(
        dataset=ds,
        objective=ObjectiveObject(metric=RecallAtK(k=5), direction="maximize"),
        design=DesignObject(n_rounds=3, batch_size=5, n_initial=10, seed=0),
        noise=GaussianNoise(sigma=0.1),
        methods=[
            Method(name="al_ucb", acquisition=UCB(), selection=TopQGreedy(),
                   surrogate=GPRSurrogate()),
            Method(name="baseline", acquisition=RandomAcquisition(),
                   selection=TopQGreedy(), surrogate=GPRSurrogate()),
        ],
    )
    assert len(exp.methods) == 2
    assert exp.dataset.n_genes == 50
