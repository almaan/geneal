# tests/test_logging.py
import json
import pandas as pd
from geneal.runner.logging import write_run_dir


def test_write_run_dir_creates_artifacts(tmp_path):
    df = pd.DataFrame({"method": ["a"], "seed": [0], "round": [0],
                       "n_revealed": [5], "metric": [0.5], "metric_name": ["recall@5"]})
    run_dir = write_run_dir(
        out_root=tmp_path, df=df,
        config={"dataset": "synthetic", "seeds": [0]},
        extra_manifest={"git_sha": "abc123", "env": "geneal"},
    )
    assert (run_dir / "rounds.parquet").exists()
    assert (run_dir / "config.yaml").exists()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["git_sha"] == "abc123"
    assert manifest["n_rows"] == 1
    # round-trip the parquet
    back = pd.read_parquet(run_dir / "rounds.parquet")
    assert len(back) == 1
