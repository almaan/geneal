#!/usr/bin/env bash
# Sharded ablation sweep: one SLURM job per target cell line (parallel), then a
# dependent aggregation job that concatenates the shards and builds ONE report.
# Wall time ~ a single line (~20-40 min) instead of the sum over lines.
#
# Usage:
#   scripts/launch_ablation_sweep.sh [N_LINES]
#   JOINT=1 scripts/launch_ablation_sweep.sh 12          # multitask GP
#   SEEDS="0 1 2" TAG=indep scripts/launch_ablation_sweep.sh 12
# Env: SEEDS (default "0 1"), JOINT (empty=independent), TAG, EXPORTFIGS.
set -euo pipefail
N_LINES="${1:-12}"
SEEDS="${SEEDS:-0 1}"; JOINT="${JOINT:-}"; TAG="${TAG:-${JOINT:+joint}${JOINT:-indep}}"
PANEL_A="${PANEL_A:-none}"; PANEL_B="${PANEL_B:-data/processed/depmap/panel_5k.txt}"
EMB="${EMB:-data/processed/embeddings/pubmedbert_all.parquet}"
GE="data/processed/depmap/gene_effect.parquet"
TS="$(date +%Y%m%d-%H%M%S)"; ROOT="res/runs_ablation/sweep_${TAG}_${TS}"; mkdir -p "$ROOT" logs

# 1. Deterministic target lines + contrast line -> ROOT/lines.txt, ROOT/contrast.txt
micromamba run -n geneal python - "$GE" "$EMB" "$N_LINES" "$ROOT" <<'PY'
import sys
import pandas as pd
from geneal.data.depmap import load_gene_effect, parse_entrez
from geneal.data.selective import rank_contrast_lines
ge_path, emb_path, n_lines, root = sys.argv[1:5]
n_lines = int(n_lines)
ge = load_gene_effect(ge_path); emb = pd.read_parquet(emb_path)
embset = set(emb.index)
labs = [g for g in ge.index if parse_entrez(g) in embset]
contrast = rank_contrast_lines(ge, n=10)[0]
ranked = ge.loc[labs].isna().sum(axis=0).sort_values().index.tolist()
lines = [c for c in ranked if c != contrast][:n_lines]
open(f"{root}/lines.txt", "w").write("\n".join(lines) + "\n")
open(f"{root}/contrast.txt", "w").write(contrast)
print(f"{len(lines)} lines, contrast={contrast}")
PY
CONTRAST="$(cat "$ROOT/contrast.txt")"
NL="$(grep -c . "$ROOT/lines.txt")"
echo "sweep -> $ROOT  ($NL lines x [$SEEDS] seeds, JOINT='${JOINT}', contrast=$CONTRAST)"

# 2. One array task per line (parallel).
ARRAY_ID=$(sbatch --parsable --array="0-$((NL-1))" \
    --export=ALL,ROOT="$ROOT",SEEDS="$SEEDS",JOINT="$JOINT",CONTRAST="$CONTRAST",EMB="$EMB",PANEL_A="$PANEL_A",PANEL_B="$PANEL_B" \
    jobs/ablation_shard.sh)
echo "shard array job: $ARRAY_ID"

# 3. Aggregation, runs only if all shards succeed.
AGG_ID=$(sbatch --parsable --dependency="afterok:${ARRAY_ID}" \
    --export=ALL,ROOT="$ROOT" jobs/ablation_aggregate.sh)
echo "aggregation job: $AGG_ID (afterok:$ARRAY_ID)"
echo "report will land at: $ROOT/report.html"
