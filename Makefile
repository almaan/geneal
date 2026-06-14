# geneal — reproducible pipeline.
# Every target is deterministic: explicit seeds, pinned params, fixed inputs.
# Run any stage with `make <target>`. Data/embedding artifacts land in data/
# (gitignored); experiment outputs in res/ (gitignored). Results notes are in
# data/*.md (tracked via -f). See REPRODUCE.md for the full pipeline + env notes.
#
# Env: the main env is `geneal` (torch 2.11, gpytorch, pyro, sklearn, plotly).
# scPRINT embeddings require a SEPARATE env `scprint` (conflicting torch) — see
# the scprint-emb target + REPRODUCE.md. GPU (CUDA) used automatically for ESM2.

MM      := micromamba run -n geneal
GE      := data/processed/depmap/gene_effect.parquet
PANEL   := data/processed/depmap/panel_hvg.txt
EMB     := data/processed/embeddings/pubmedbert_hvg.parquet
SEEDS   := 0 1 2 3
K       := 30

.PHONY: help test data panel uniprot esm2-emb pubmedbert-emb scprint-emb \
        diagnostics multiline string corum risk risk-large dual all clean-res

help:
	@echo "geneal pipeline targets (see REPRODUCE.md):"
	@echo "  test           - run the full pytest suite"
	@echo "  data           - download + curate DepMap (scripts/... ; see REPRODUCE.md)"
	@echo "  panel          - select the 2043-gene HVG panel (deterministic)"
	@echo "  uniprot        - map panel Entrez -> UniProt sequences"
	@echo "  esm2-emb       - ESM2-650M embeddings (GPU)"
	@echo "  pubmedbert-emb - PubMedBERT gene-text embeddings"
	@echo "  scprint-emb    - scPRINT embeddings (SEPARATE env, see REPRODUCE.md)"
	@echo "  diagnostics    - embedding redundancy/R2 diagnostic (1 line)"
	@echo "  multiline      - redundancy diagnostic across 5 cell lines"
	@echo "  string / corum - graph similarity validation"
	@echo "  risk           - risk-aware nomination (12 lines x 4 seeds)"
	@echo "  risk-large     - sharded large sweep (scripts/launch_risk_sweep.sh)"
	@echo "  dual           - selectivity/dual-line experiment"

test:
	$(MM) pytest -q

# --- data + panel (deterministic given the DepMap release) ---
panel:
	$(MM) python scripts/select_gene_panel.py --n-hvg 2000 --n-lethal 1000 --out $(PANEL)

uniprot:
	$(MM) python scripts/map_genes_to_uniprot.py --entrez-file $(PANEL) \
	  --out data/processed/depmap/uniprot_map_hvg.parquet

# --- embeddings (GPU auto for ESM2; pinned models) ---
esm2-emb:
	$(MM) python scripts/precompute_esm2.py \
	  --map data/processed/depmap/uniprot_map_hvg.parquet \
	  --out data/processed/embeddings/esm2_650m_hvg.parquet \
	  --model esm2_t33_650M_UR50D --batch-size 16

pubmedbert-emb:
	$(MM) python scripts/embed_pubmedbert.py --panel $(PANEL) \
	  --out data/processed/embeddings/pubmedbert_hvg.parquet

scprint-emb:
	@echo "scPRINT needs a SEPARATE env (conflicting torch). See REPRODUCE.md."
	micromamba run -n scprint python scripts/extract_scprint_gene_emb.py \
	  --panel-map data/processed/depmap/entrez_ensembl.parquet \
	  --out data/processed/embeddings/scprint_hvg.parquet

# --- diagnostics (deterministic, seed 0 inside the scripts) ---
diagnostics:
	$(MM) python scripts/diagnose_embeddings.py --embeddings $(EMB) --cell-line ACH-000147

multiline:
	$(MM) python scripts/diagnose_multiline.py

string:
	$(MM) python scripts/validate_string.py

corum:
	$(MM) python scripts/validate_corum.py

# --- headline experiment: risk-aware nomination ---
risk:
	$(MM) python scripts/run_risk_nomination.py --n-cell-lines 12 --seeds $(SEEDS) \
	  --K $(K) --embeddings $(EMB)

risk-large:
	bash scripts/launch_risk_sweep.sh 40 6 8 $(K)

dual:
	$(MM) python scripts/run_dual_experiment.py --seeds $(SEEDS)

all: test panel diagnostics multiline risk
	@echo "core pipeline complete"

clean-res:
	rm -rf res/runs_risk/* res/runs_single_650m/* res/runs_dual/*
