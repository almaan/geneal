# tests/test_esm2_embed.py
import numpy as np
import pytest
from geneal.data.esm2_embed import embed_sequences


def test_embed_sequences_shape_and_determinism():
    seqs = {"P1": "MKTAYIAKQR", "P2": "MQIFVKTLTG"}
    a = embed_sequences(seqs, model_name="esm2_t6_8M_UR50D")
    b = embed_sequences(seqs, model_name="esm2_t6_8M_UR50D")
    assert set(a.index) == {"P1", "P2"}
    assert a.shape[1] == 320
    np.testing.assert_allclose(a.loc["P1"].to_numpy(), b.loc["P1"].to_numpy(), atol=1e-5)
    assert not np.allclose(a.loc["P1"].to_numpy(), a.loc["P2"].to_numpy())
