# src/geneal/runner/sweep.py
from __future__ import annotations
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from geneal.runner.runner import Runner


def _run_one_cell(args):
    build_experiment, cell_line, seeds, eval_panel = args
    exp = build_experiment(cell_line)
    df = Runner().run(exp, seeds=seeds, eval_panel=eval_panel)
    df["cell_line"] = cell_line
    return df


def run_sweep(build_experiment, cell_lines, seeds, eval_panel=None,
              max_workers=None) -> pd.DataFrame:
    """Run build_experiment(cell_line) across cell lines in parallel processes.

    build_experiment must be a top-level (picklable) callable returning an
    Experiment. Each process runs all methods x seeds for one cell line.
    """
    tasks = [(build_experiment, cl, seeds, eval_panel) for cl in cell_lines]
    frames = []
    # 'spawn' (not the Linux default 'fork') because the surrogate uses torch
    # autograd; fork-based workers inherit autograd's threading state and torch
    # raises "Unable to handle autograd's threading in combination with
    # fork-based multiprocessing" once autograd has run in the parent process.
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
        for df in ex.map(_run_one_cell, tasks):
            frames.append(df)
    return pd.concat(frames, ignore_index=True)
