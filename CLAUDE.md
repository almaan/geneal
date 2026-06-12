# geneal

Active learning for gene-knockout selection on molecular data. Race an AL
strategy against a baseline to recover the most-lethal knockout genes for a cell
line in fewer rounds.

## Environment
- micromamba env: `geneal`. Run anything with `micromamba run -n geneal <cmd>`.
- Editable install: `pip install -e ".[dev]"`.
- GPR uses gpytorch; BNN uses pyro.

## Layout
- `src/geneal/interfaces.py` — Protocols every swappable piece implements.
- `src/geneal/data/` — Dataset + synthetic generator (real DepMap = Plan 2).
- `src/geneal/models/` — surrogate (gpytorch GPR, pyro BNN), acquisition, selection, noise.
- `src/geneal/metrics/` — pluggable success metrics (recall@k default).
- `src/geneal/experiment/` — Objective/Design/Method/Experiment containers.
- `src/geneal/runner/` — round loop (common random numbers) + run logging.
- `src/geneal/report/` — HTML report (recall curves + latex tables).
- `conf/` — Hydra configs (`_target_` instantiation).
- `scripts/run_experiment.py` — Hydra entrypoint.

## Conventions
- Target = lethality, HIGHER is better. Acquisitions maximize.
- RNG is always an explicit numpy Generator passed in. No global random state.
- Per-method acquisition RNG seeds from a stable sha256 of the method name, so
  runs reproduce across processes (not just within one).
- Run a config: `python scripts/run_experiment.py seeds=[0,1,2]`.
- Tests: `pytest`. Keep new components behind a Protocol + a test.

## Status
- Plan 1 (core framework on synthetic data): docs/superpowers/plans/2026-06-12-geneal-core-framework.md — implemented.
- Plan 2 (DepMap data + scPRINT/ESM2 embeddings): not yet written. Interfaces
  designed to accept the swap with no change to Runner/Report.
