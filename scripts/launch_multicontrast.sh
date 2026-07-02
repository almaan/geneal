#!/usr/bin/env bash
# Sharded full-genome multi-contrast EHVI: one array task per target cell line,
# then a dependent aggregation that concatenates the shards, copies the result
# into an existing ablation run dir, and rebuilds that report (adds the MC tables).
#
# Usage:
#   SWEEP_DIR=res/runs_ablation/sweep_joint3c_... \
#   CONTRAST="ACH-001360 ACH-000608 ACH-000934" \
#   scripts/launch_multicontrast.sh [N_LINES]
set -euo pipefail
N_LINES="${1:-12}"
SEEDS="${SEEDS:-0 1}"
SWEEP_DIR="${SWEEP_DIR:?set SWEEP_DIR (ablation run to update)}"
CONTRAST="${CONTRAST:?set CONTRAST (space-separated ModelIDs)}"
EMB="${EMB:-data/processed/embeddings/pubmedbert_all.parquet}"
GE="data/processed/depmap/gene_effect.parquet"
TS="$(date +%Y%m%d-%H%M%S)"; ROOT="res/runs_multicontrast/sharded_${TS}"; mkdir -p "$ROOT" logs

# deterministic target lines (lowest-NaN, excluding the contrasts) -> ROOT/lines.txt
micromamba run -n geneal python - "$GE" "$EMB" "$N_LINES" "$ROOT" "$CONTRAST" <<'PY'
import sys
import pandas as pd
from geneal.data.depmap import load_gene_effect, parse_entrez
ge_path, emb_path, n_lines, root, contrast = sys.argv[1:6]
n_lines = int(n_lines); cset = set(contrast.split())
ge = load_gene_effect(ge_path); emb = pd.read_parquet(emb_path)
labs = [g for g in ge.index if parse_entrez(g) in set(emb.index)]
ranked = ge.loc[labs].isna().sum(axis=0).sort_values().index.tolist()
lines = [c for c in ranked if c not in cset][:n_lines]
open(f"{root}/lines.txt", "w").write("\n".join(lines) + "\n")
print(f"{len(lines)} lines, contrasts={sorted(cset)}")
PY
NL="$(grep -c . "$ROOT/lines.txt")"
echo "multicontrast sharded -> $ROOT  ($NL lines x [$SEEDS] seeds, contrasts=[$CONTRAST]) -> updates $SWEEP_DIR"

# persist the reproducible launch command next to the report it updates
cat > "$SWEEP_DIR/launch_multicontrast.sh" <<EOF
#!/usr/bin/env bash
# Reproduces the multi-contrast EHVI addition for $SWEEP_DIR.
SWEEP_DIR="${SWEEP_DIR}" CONTRAST="${CONTRAST}" SEEDS="${SEEDS}" \\
  bash scripts/launch_multicontrast.sh ${N_LINES}
EOF
chmod +x "$SWEEP_DIR/launch_multicontrast.sh"

DEP="${DEP:-}"   # optional: afterok:<jobid> to chain after an ablation sweep
ARRAY_ID=$(sbatch --parsable ${DEP:+--dependency=afterok:$DEP} --array="0-$((NL-1))" \
    --export=ALL,ROOT="$ROOT",CONTRAST="$CONTRAST",SEEDS="$SEEDS" \
    jobs/multicontrast_shard.sh)
echo "shard array job: $ARRAY_ID"
AGG_ID=$(sbatch --parsable --dependency="afterok:${ARRAY_ID}" \
    --export=ALL,ROOT="$ROOT",SWEEP_DIR="$SWEEP_DIR" jobs/multicontrast_aggregate.sh)
echo "aggregation job: $AGG_ID (afterok:$ARRAY_ID)"
echo "report will update at: $SWEEP_DIR/report.html"
