# src/geneal/data/esm2_embed.py
from __future__ import annotations
import os
import numpy as np
import pandas as pd

os.environ.setdefault("TORCH_HOME", os.path.join("data", "raw", "torch_hub"))

_MODEL_DIM = {"esm2_t6_8M_UR50D": 320, "esm2_t12_35M_UR50D": 480,
              "esm2_t30_150M_UR50D": 640, "esm2_t33_650M_UR50D": 1280}


def embed_sequences(sequences: dict[str, str], model_name: str = "esm2_t33_650M_UR50D",
                    batch_size: int = 8, device: str | None = None,
                    max_len: int = 1022) -> pd.DataFrame:
    """Mean-pooled ESM2 residue embeddings for {id: protein_sequence}.

    Returns a DataFrame indexed by id, columns = embedding dims. Sequences longer
    than max_len are truncated (ESM2 context limit). CPU by default; pass
    device='cuda' to use a GPU. Deterministic (eval mode, no dropout).
    """
    import torch
    import esm

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, alphabet = getattr(esm.pretrained, model_name)()
    model = model.eval().to(device)
    bc = alphabet.get_batch_converter()
    repr_layer = model.num_layers

    ids = list(sequences.keys())
    out: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for i in range(0, len(ids), batch_size):
            chunk = ids[i:i + batch_size]
            data = [(k, sequences[k][:max_len]) for k in chunk]
            _, _, toks = bc(data)
            toks = toks.to(device)
            res = model(toks, repr_layers=[repr_layer])["representations"][repr_layer]
            for j, k in enumerate(chunk):
                L = len(sequences[k][:max_len])
                vec = res[j, 1:L + 1].mean(0).cpu().numpy()
                out[k] = vec
    return pd.DataFrame.from_dict(out, orient="index")
