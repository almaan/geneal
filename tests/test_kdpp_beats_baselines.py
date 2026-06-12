# tests/test_kdpp_beats_baselines.py
"""What the k-DPP method robustly delivers on synthetic top-k recovery — stated
honestly after fixing the surrogate to return the LATENT (noise-free) covariance.

Findings (verified across multiple regimes; see geneal-scientific-positioning
memory for the full reality check):

  ROBUST: diversity-only selection (CoreSet, the IterPert-style rule) is
  catastrophic on a top-k *extreme recovery* objective — it spends most of the
  budget on the inert tail. k-DPP (and any quality-aware method) beats it by a
  wide margin everywhere. This is the defensible half of the contribution and
  the part this test asserts.

  OPEN / NOT ROBUST on synthetic: k-DPP vs quality-only greedy. With the correct
  latent kernel, quality-greedy is already near-optimal for top-k recovery on
  these synthetic landscapes (UCB's mean+beta*sigma already injects some spread),
  and outcome-diversity adds little — k-DPP only edges greedy in a narrow regime
  (many separated reward modes + tight per-round budget). So we assert only that
  k-DPP is NEVER MUCH WORSE than greedy (it is a safe quality+diversity method),
  NOT that it dominates. Whether outcome-diversity helps on real essentiality
  landscapes is the open question for the real-DepMap experiment.

This test therefore encodes the *honest* claims: k-DPP >> diversity-only, and
k-DPP is not beaten by greedy beyond tolerance. It does NOT assert k-DPP > greedy.
"""
import numpy as np
from geneal.data.dataset import Dataset
from geneal.experiment.objects import ObjectiveObject, DesignObject, Method, Experiment
from geneal.metrics.recall import RecallAtK
from geneal.models.acquisition import UCB, GreedyMean
from geneal.models.selection import KDPP, TopQGreedy, CoreSet
from geneal.models.surrogate import GPRSurrogate
from geneal.models.noise import GaussianNoise
from geneal.runner.runner import Runner


def _clustered_dataset(seed=0):
    """Rewards spread across several separated clusters — the regime where batch
    composition matters most: a quality-only batch can pile into one lethal mode
    and miss others, a diversity-only batch wanders off the lethal modes entirely."""
    rng = np.random.default_rng(seed)
    dim = 6
    n_clusters = 12
    centers = rng.standard_normal((n_clusters, dim)) * 3
    cluster_value = rng.standard_normal(n_clusters)
    cluster_value[:8] += 6.0  # many lethal clusters -> reward is spread, not concentrated
    embeddings, target, names = [], [], []
    gid = 0
    for c in range(n_clusters):
        for _ in range(20):
            embeddings.append(centers[c] + rng.standard_normal(dim) * 0.15)
            target.append(cluster_value[c] + rng.normal(0, 0.1))
            names.append(f"GENE_{gid:04d}"); gid += 1
    return Dataset(np.array(embeddings), np.array(target), names)


def _run(selection, acquisition, seeds):
    ds = _clustered_dataset(seed=0)
    exp = Experiment(
        dataset=ds,
        objective=ObjectiveObject(metric=RecallAtK(k=80), direction="maximize"),
        design=DesignObject(n_rounds=8, batch_size=6, n_initial=16, seed=0),
        noise=GaussianNoise(sigma=0.1),
        methods=[Method("m", acquisition, selection, GPRSurrogate(n_iters=60))],
    )
    df = Runner().run(exp, seeds=seeds)
    last = df["round"].max()
    return df[df["round"] == last]["metric"].mean()


def test_kdpp_beats_diversity_only_and_is_not_worse_than_greedy():
    seeds = [0, 1, 2]
    kdpp = _run(KDPP(solver="greedy"), UCB(beta=2.0), seeds)
    greedy = _run(TopQGreedy(), UCB(beta=2.0), seeds)
    coreset = _run(CoreSet(), GreedyMean(), seeds)
    print(f"\nkdpp={kdpp:.4f} greedy={greedy:.4f} coreset={coreset:.4f}")
    # ROBUST claim: quality-aware k-DPP decisively beats diversity-only selection.
    assert kdpp >= coreset + 0.10, f"kdpp {kdpp} not clearly > coreset {coreset}"
    # SAFETY claim: k-DPP is never much worse than quality-only greedy.
    assert kdpp >= greedy - 0.07, f"kdpp {kdpp} << greedy {greedy}"
