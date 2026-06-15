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
ALLGENES:= data/processed/depmap/panel_all.txt        # full genome (18.5k entrez)
PANEL   := data/processed/depmap/panel_hvg.txt         # demo panel (2043) for subsetting
EMB_ALL := data/processed/embeddings/pubmedbert_all.parquet  # genome-wide cache
EMB     := data/processed/embeddings/pubmedbert_hvg.parquet  # demo subset
SEEDS   := 0 1 2 3
K       := 30

# ARCHITECTURE: embeddings are GENOME-WIDE caches (entrez-indexed). A "panel" is
# just a row-subset (pass --panel <entrez-file> to experiments). Resizing the
# panel never re-embeds. Genome-wide caches: pubmedbert_all, esm2_650m_all,
# scprint (44k native), string_edges_all + corum_*_all. Co-dependency is NOT used.

.PHONY: help test data panel uniprot esm2-emb pubmedbert-emb scprint-emb \
        diagnostics multiline string corum risk risk-large dual all clean-res \
        ablation ablation-fullgenome

help:
	@echo "geneal pipeline targets (see REPRODUCE.md):"
	@echo "  ablation       - DEFAULT OUTPUT: two-analysis ablation (5k panel)"
	@echo "  ablation-fullgenome - same, full ~18.5k-gene genome"
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

# --- GENOME-WIDE embedding caches (precompute ONCE; panels subset them) ---
allgenes:
	$(MM) python -c "from geneal.data.depmap import load_gene_effect,parse_entrez; ge=load_gene_effect('$(GE)'); open('$(ALLGENES)','w').write(chr(10).join(str(parse_entrez(g)) for g in ge.index))"

uniprot-all:
	$(MM) python scripts/map_genes_to_uniprot.py --entrez-file $(ALLGENES) \
	  --out data/processed/depmap/uniprot_map_all.parquet

esm2-all:        ## ESM2-650M, all genes, GPU (long, one-time)
	$(MM) python scripts/precompute_esm2.py \
	  --map data/processed/depmap/uniprot_map_all.parquet \
	  --out data/processed/embeddings/esm2_650m_all.parquet \
	  --model esm2_t33_650M_UR50D --batch-size 16

pubmedbert-all:  ## PubMedBERT, all genes (one-time)
	$(MM) python scripts/embed_pubmedbert.py --panel $(ALLGENES) \
	  --out data/processed/embeddings/pubmedbert_all.parquet \
	  --text-cache data/processed/depmap/gene_text_all.pkl

graphs-all:      ## STRING + CORUM full edge lists + membership (one-time)
	$(MM) python scripts/build_graph_caches.py

scprint-emb:
	@echo "scPRINT needs a SEPARATE env (conflicting torch). See REPRODUCE.md."
	micromamba run -n scprint python scripts/extract_scprint_gene_emb.py \
	  --out data/processed/embeddings/scprint_gene_emb_raw.parquet

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
# Uses the genome-wide PubMedBERT cache, subset to a panel via --panel.
risk:
	$(MM) python scripts/run_risk_nomination.py --n-cell-lines 12 --seeds $(SEEDS) \
	  --K $(K) --embeddings $(EMB_ALL) --panel $(PANEL) --out-root res/runs_risk

risk-5k:         ## risk nomination on the 5k panel (genome-wide cache subset)
	$(MM) python scripts/run_risk_nomination.py --n-cell-lines 12 --seeds $(SEEDS) \
	  --K $(K) --embeddings $(EMB_ALL) --panel data/processed/depmap/panel_5k.txt \
	  --out-root res/runs_risk_5k

risk-large:
	bash scripts/launch_risk_sweep.sh 40 6 8 $(K)

# --- DEFAULT OUTPUT: two-analysis ablation (safety-vs-efficacy + diversity) ---
# A) safety axis (none/truncation/ehvi) + random/coreset/typiclust on the
#    efficacy-toxicity tradeoff; B) diversity operators (none/cap/kdpp) on two
#    bases. PubMedBERT predicts; CORUM=pathways (cap), STRING=k-DPP similarity.
# 5k panel, 6 lines, 3 seeds, AL 8x10. SLURM: `sbatch jobs/ablation.sh`.
ablation:        ## DEFAULT report: two-analysis ablation on the 5k panel
	$(MM) python scripts/run_ablation.py --embeddings $(EMB_ALL) \
	  --panel data/processed/depmap/panel_5k.txt \
	  --n-cell-lines 6 --seeds 0 1 2 --K $(K) --n-rounds 8 --batch 10 \
	  --out-root res/runs_ablation

ablation-fullgenome:  ## same ablation on the full ~18.5k-gene genome
	$(MM) python scripts/run_ablation.py --embeddings $(EMB_ALL) --panel none \
	  --n-cell-lines 6 --seeds 0 1 2 --K $(K) --n-rounds 8 --batch 10 \
	  --out-root res/runs_ablation_fullgenome

dual:
	$(MM) python scripts/run_dual_experiment.py --seeds $(SEEDS)

all: test panel diagnostics multiline risk
	@echo "core pipeline complete"

clean-res:
	rm -rf res/runs_risk/* res/runs_single_650m/* res/runs_dual/*
