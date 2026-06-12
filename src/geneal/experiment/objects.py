# src/geneal/experiment/objects.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
from geneal.interfaces import Surrogate, Acquisition, Selection, NoiseModel, Metric
from geneal.data.dataset import Dataset


@dataclass
class ObjectiveObject:
    """What to optimize and how to measure success."""
    metric: Metric
    direction: Literal["maximize", "minimize"] = "maximize"


@dataclass
class DesignObject:
    """Experimental design knobs."""
    n_rounds: int
    batch_size: int
    n_initial: int
    seed: int = 0

    def __post_init__(self) -> None:
        for name in ("n_rounds", "batch_size", "n_initial"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")


@dataclass
class Method:
    """A named (surrogate, acquisition, selection) strategy to race."""
    name: str
    acquisition: Acquisition
    selection: Selection
    surrogate: Surrogate


@dataclass
class Experiment:
    """Top-level container assembled from config."""
    dataset: Dataset
    objective: ObjectiveObject
    design: DesignObject
    noise: NoiseModel
    methods: list[Method] = field(default_factory=list)
