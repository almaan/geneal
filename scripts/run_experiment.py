# scripts/run_experiment.py
"""Hydra entrypoint: build an Experiment from config, run it, log, report."""
from __future__ import annotations
import subprocess
import pandas as pd
import hydra
from hydra.utils import instantiate, call
from omegaconf import DictConfig, OmegaConf

from geneal.experiment.objects import ObjectiveObject, Method, Experiment
from geneal.runner.runner import Runner
from geneal.runner.logging import write_run_dir
from geneal.report.report import build_report


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    dataset = call(cfg.dataset)           # make_synthetic(...)
    metric = instantiate(cfg.objective)   # RecallAtK
    design = instantiate(cfg.design)
    noise = instantiate(cfg.noise)
    methods = [
        Method(name=m["name"],
               acquisition=instantiate(m["acquisition"]),
               selection=instantiate(m["selection"]),
               surrogate=instantiate(m["surrogate"]))
        for m in cfg.methods
    ]
    exp = Experiment(
        dataset=dataset,
        objective=ObjectiveObject(metric=metric, direction="maximize"),
        design=design, noise=noise, methods=methods,
    )
    df = Runner().run(exp, seeds=list(cfg.seeds))
    run_dir = write_run_dir(
        out_root=cfg.out_root, df=df,
        config=OmegaConf.to_container(cfg, resolve=True),
        extra_manifest={"git_sha": _git_sha()},
    )
    build_report(pd.read_parquet(run_dir / "rounds.parquet"),
                 run_dir / "report.html")
    print(f"Run complete: {run_dir}")


if __name__ == "__main__":
    main()
