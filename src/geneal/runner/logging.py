# src/geneal/runner/logging.py
from __future__ import annotations
from pathlib import Path
import json
import pandas as pd


def write_run_dir(*, out_root, df: pd.DataFrame, config: dict,
                  extra_manifest: dict | None = None, run_name: str = "run") -> Path:
    """Write a self-contained run directory: parquet + config.yaml + manifest.json.

    Caller supplies any time-varying values (git sha, timestamp, env) in
    `extra_manifest` so this function stays deterministic and import-light.
    """
    out_root = Path(out_root)
    run_dir = out_root / run_name
    # de-collide if the run dir already exists
    i = 1
    while run_dir.exists():
        run_dir = out_root / f"{run_name}_{i}"
        i += 1
    run_dir.mkdir(parents=True)

    df.to_parquet(run_dir / "rounds.parquet", index=False)

    # config as simple YAML (avoid extra deps: write a minimal serialization)
    _write_yaml(run_dir / "config.yaml", config)

    manifest = {"n_rows": int(len(df)),
                "methods": sorted(df["method"].unique().tolist()) if len(df) else [],
                "metric_name": (df["metric_name"].iloc[0] if len(df) else None)}
    if extra_manifest:
        manifest.update(extra_manifest)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return run_dir


def _write_yaml(path: Path, data: dict) -> None:
    try:
        import yaml  # PyYAML ships with hydra-core
        path.write_text(yaml.safe_dump(data, sort_keys=False))
    except Exception:
        # Fallback: JSON is valid YAML
        path.write_text(json.dumps(data, indent=2))
