"""Extract scPRINT zero-shot (context-independent) gene input embedding.

Loads the medium-v1.5 checkpoint, pulls gene_encoder.embedding.weight
(rows = model.genes, Ensembl IDs), saves to parquet, and -- if the raw
precomputed ESM2 init mmap is available -- computes per-gene cosine
similarity between the TRAINED embedding and the raw init to decide
whether scPRINT learned something distinct from raw ESM2.

Run: micromamba run -n scprint python scripts/extract_scprint_gene_emb.py
"""
import os, sys, numpy as np, pandas as pd

CKPT = "data/raw/scprint/medium-v1.5.ckpt"
OUT = "data/processed/embeddings/scprint_gene_emb_raw.parquet"
MMAP = "data/raw/scprint/gene_emb_init.mmap"

os.makedirs(os.path.dirname(OUT), exist_ok=True)

# --- Load via the official scPrint API ---
loaded_via = None
genes = None
W = None
try:
    from scprint import scPrint
    model = scPrint.load_from_checkpoint(CKPT, precpt_gene_emb=None, transformer="normal")
    model.eval()
    genes = list(model.genes)
    # key name verified from ckpt: gene_encoder.embedding.weight (singular)
    enc = model.gene_encoder
    emb = getattr(enc, "embedding", None) or getattr(enc, "embeddings", None)
    W = emb.weight.detach().cpu().float().numpy()
    loaded_via = "scPrint.load_from_checkpoint"
    print("[ok] loaded via scPrint API; d_model=%d n_genes=%d" % (W.shape[1], W.shape[0]))
except Exception as e:
    print("[warn] scPrint API load failed: %r" % e)
    print("[info] falling back to raw torch state_dict read")
    import torch
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]; hp = ck["hyper_parameters"]
    W = sd["gene_encoder.embedding.weight"].detach().cpu().float().numpy()
    genes = list(hp["genes"])
    loaded_via = "raw_state_dict"

assert W.shape[0] == len(genes), (W.shape, len(genes))
df = pd.DataFrame(W, index=pd.Index(genes, name="ensembl_gene_id"))
df.columns = [f"d{i}" for i in range(W.shape[1])]
df.to_parquet(OUT)
print("[ok] saved %s shape=%s loaded_via=%s" % (OUT, W.shape, loaded_via))

# --- structure stats ---
norms = np.linalg.norm(W, axis=1)
print("row-norm mean=%.4f std=%.4f (normalized? %s)" %
      (norms.mean(), norms.std(), "yes" if norms.std() < 1e-4 else "no"))

# --- DECISIVE: compare trained emb vs raw ESM2 init mmap if present & shape-recoverable ---
if os.path.exists(MMAP) and os.path.getsize(MMAP) > 0:
    nbytes = os.path.getsize(MMAP)
    G, D = W.shape
    cand = []
    for dt in (np.float32, np.float16, np.float64):
        if nbytes % (G * np.dtype(dt).itemsize) == 0:
            cand.append((dt, nbytes // (G * np.dtype(dt).itemsize)))
    print("[mmap] bytes=%d  candidate (dtype, inferred_dim) for G=%d: %s" % (nbytes, G, cand))
    init = None
    for dt, dim in cand:
        try:
            arr = np.memmap(MMAP, dtype=dt, mode="r").reshape(G, dim)
            init = np.asarray(arr, dtype=np.float32)
            print("[mmap] read as dtype=%s dim=%d" % (np.dtype(dt).name, dim))
            break
        except Exception as ee:
            print("[mmap] reshape failed for %s: %r" % (dt, ee))
    if init is not None:
        # init dim (e.g. ESM 1280/640) differs from d_model (256): scPRINT pools
        # init -> d_model via AdaptiveAvgPool1d. Replicate that to compare like-for-like.
        if init.shape[1] != D:
            import torch
            pooled = torch.nn.functional.adaptive_avg_pool1d(
                torch.from_numpy(init).unsqueeze(1), D).squeeze(1).numpy()
            print("[mmap] pooled init %s -> %s to match d_model" % (init.shape, pooled.shape))
            init_cmp = pooled
        else:
            init_cmp = init
        a = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)
        b = init_cmp / (np.linalg.norm(init_cmp, axis=1, keepdims=True) + 1e-9)
        cos = (a * b).sum(1)
        print("=== TRAINED vs RAW-INIT per-gene cosine ===")
        print("mean=%.4f median=%.4f std=%.4f min=%.4f max=%.4f frac>0.99=%.3f" %
              (cos.mean(), np.median(cos), cos.std(), cos.min(), cos.max(),
               float((cos > 0.99).mean())))
        verdict = "FROZEN==ESM2 (BAD)" if cos.mean() > 0.99 else "LEARNED/DISTINCT (GOOD)"
        print("VERDICT:", verdict)
    else:
        print("[mmap] could not infer layout; skipping strict cosine check")
else:
    print("[mmap] init file absent; strict cosine check skipped")
