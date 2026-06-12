# src/geneal/data/dataset.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Dataset:
    """One cell line's knockout problem.

    embeddings: (n_genes, dim) gene representations (placeholder or real).
    target:     (n_genes,) optimization target; HIGHER = more lethal = better.
    gene_names: list of gene identifiers, len n_genes.
    """
    embeddings: np.ndarray
    target: np.ndarray
    gene_names: list[str]

    def __post_init__(self) -> None:
        n = len(self.gene_names)
        if self.embeddings.shape[0] != n or self.target.shape[0] != n:
            raise ValueError("embeddings, target, gene_names must share length")

    @property
    def n_genes(self) -> int:
        return len(self.gene_names)


def make_synthetic(n_genes: int = 200, dim: int = 8, seed: int = 0,
                   noise_sd: float = 0.1) -> Dataset:
    """Synthetic dataset where target is a noisy linear function of embeddings."""
    rng = np.random.default_rng(seed)
    embeddings = rng.standard_normal((n_genes, dim))
    beta = rng.standard_normal(dim)
    signal = embeddings @ beta
    target = signal + rng.normal(0.0, noise_sd, size=n_genes)
    gene_names = [f"GENE_{i:04d}" for i in range(n_genes)]
    return Dataset(embeddings=embeddings, target=target, gene_names=gene_names)
