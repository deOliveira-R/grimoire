from __future__ import annotations

import numpy as np

from grimoire.embed.base import l2_normalize, serialize_float32


def test_normalize_unit_norm() -> None:
    v = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 1.0, 1.0, 1.0]], dtype=object)
    # Each row treated independently; use a proper 2D array with a single shape:
    a = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    out = l2_normalize(a)
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms[0], 1.0)
    # Zero row is left as zero rather than NaN.
    assert np.allclose(norms[1], 0.0)
    _ = v  # silence unused-var


def test_serialize_roundtrip() -> None:
    v = np.array([1.5, -2.5, 3.0], dtype=np.float32)
    blob = serialize_float32(v)
    assert isinstance(blob, bytes)
    assert len(blob) == 4 * 3
    # Deserialize.
    arr = np.frombuffer(blob, dtype=np.float32)
    assert np.allclose(arr, v)


def test_stub_embedder_deterministic() -> None:
    from tests.support.stub_embedder import StubEmbedder

    e = StubEmbedder(dim=8)
    a = e.encode(["alpha", "beta"])
    b = e.encode(["alpha", "beta"])
    assert a.shape == (2, 8)
    assert np.allclose(a, b)
    # Different inputs produce different vectors (except with astronomical collision prob).
    c = e.encode(["gamma"])
    assert not np.allclose(a[0], c[0])
