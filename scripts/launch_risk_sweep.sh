#!/usr/bin/env bash
# Sharded large-scale risk-nomination sweep.
# Splits a large set of DepMap cell lines across N parallel workers (each runs
# run_risk_nomination.py on its line-shard for ALL seeds), then aggregates every
# shard's parquet into one combined result + mean+/-CI table + figures.
#
# Usage:
#   scripts/launch_risk_sweep.sh [N_LINES] [N_SEEDS] [N_SHARDS] [K]
# Defaults: 40 lines, 6 seeds, 8 shards, K=30
#
# Env: must be run so that `micromamba run -n geneal` works. Writes to
# res/runs_risk/launch_<timestamp>/.
set -euo pipefail

N_LINES="${1:-40}"
N_SEEDS="${2:-6}"
N_SHARDS="${3:-8}"
K="${4:-30}"
# Embedding cache (override with EMB=...). Default = genome-wide PubMedBERT cache
# (no panel subset => full ~18.5k genes). Set PANEL=<file> to subset.
EMB="${EMB:-data/processed/embeddings/pubmedbert_all.parquet}"
PANEL_ARG=""; [ -n "${PANEL:-}" ] && PANEL_ARG="--panel ${PANEL}"
GE="data/processed/depmap/gene_effect.parquet"
MM="micromamba run -n geneal"

TS="$(date +%Y%m%d-%H%M%S)"
ROOT="res/runs_risk/launch_${TS}"
mkdir -p "$ROOT"
SEEDS="$(seq 0 $((N_SEEDS-1)) | tr '\n' ' ')"

echo "launch: ${N_LINES} lines x ${N_SEEDS} seeds across ${N_SHARDS} shards, K=${K} -> ${ROOT}"

# 1. Pick the N_LINES lowest-NaN cell lines (among embedded genes) and shard them.
$MM python - "$GE" "$EMB" "$N_LINES" "$N_SHARDS" "$ROOT" <<'PY'
import sys, json
import pandas as pd
from geneal.data.depmap import load_gene_effect, parse_entrez
ge_path, emb_path, n_lines, n_shards, root = sys.argv[1:6]
n_lines, n_shards = int(n_lines), int(n_shards)
ge = load_gene_effect(ge_path)
emb = pd.read_parquet(emb_path)
embset = set(emb.index)
labs = [g for g in ge.index if parse_entrez(g) in embset]
lines = ge.loc[labs].isna().sum(0).sort_values().index[:n_lines].tolist()
shards = [lines[i::n_shards] for i in range(n_shards)]  # round-robin balance
for i, s in enumerate(shards):
    open(f"{root}/shard_{i}.lines", "w").write(" ".join(s))
print(f"selected {len(lines)} lines, sharded into {n_shards}")
PY

# 2. Launch one worker per shard in parallel.
pids=()
for ((i=0; i<N_SHARDS; i++)); do
    LINES_FILE="${ROOT}/shard_${i}.lines"
    [ -s "$LINES_FILE" ] || { echo "shard $i empty, skip"; continue; }
    LINES="$(cat "$LINES_FILE")"
    (
        $MM python scripts/run_risk_nomination.py \
            --gene-effect "$GE" --embeddings "$EMB" $PANEL_ARG \
            --cell-lines $LINES --seeds $SEEDS --K "$K" \
            --out-root "$ROOT" --run-name "shard_${i}" \
            >"${ROOT}/shard_${i}.log" 2>&1
    ) &
    pids+=($!)
    echo "  shard $i launched (pid ${pids[-1]}, $(echo $LINES | wc -w) lines)"
done

# 3. Wait for all shards; report any failures.
fail=0
for p in "${pids[@]}"; do
    if ! wait "$p"; then echo "WARNING: a shard (pid $p) failed"; fail=1; fi
done
echo "all shards done (fail=$fail)"

# 4. Aggregate every shard parquet -> combined table + figures.
$MM python - "$ROOT" "$K" <<'PY'
import sys, glob
import numpy as np, pandas as pd
root, K = sys.argv[1], int(sys.argv[2])
parts = glob.glob(f"{root}/shard_*/risk.parquet")
if not parts:
    print("NO shard parquets found"); sys.exit(1)
df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
df.to_parquet(f"{root}/combined.parquet")
n_lines = df["cell_line"].nunique(); n_seeds = df["seed"].nunique()
def ci(x):
    x=np.asarray(x,float); n=len(x); m=float(x.mean())
    return m, (0.0 if n<2 else 1.96*float(x.std(ddof=1))/np.sqrt(n))
caps=sorted({int(m.split('cap')[1]) for m in df.method.unique() if 'cap' in m})
order=[m for m in (["efficacy","constrained"]+[f"constrained_cap{c}" for c in caps]) if m in set(df.method)]
metrics=[c for c in ["mean_efficacy","max_efficacy","mean_toxicity","concentration","robustness"] if c in df.columns]
print(f"\n=== COMBINED: {n_lines} lines x {n_seeds} seeds (mean +/- 95% CI) ===")
for met in metrics:
    print(f"\n[{met}]")
    for meth in order:
        s=df[df.method==meth][met]
        if len(s): mn,c=ci(s); print(f"  {meth:18s} {mn:.3f} +/- {c:.3f}")
# detailed combined report
try:
    from geneal.report.risk_report import build_risk_report
    build_risk_report(df, f"{root}/report.html")
    print(f"report -> {root}/report.html")
except Exception as e:
    print("report skipped:", e)
print(f"combined -> {root}/combined.parquet")
PY

echo "DONE: ${ROOT}"
