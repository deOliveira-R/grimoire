"""Embedder protocol and vector helpers.

Vectors are L2-normalized before insertion into sqlite-vec so that the default
L2 distance gives the same top-k ordering as cosine — this avoids touching the
vec0 schema (plan §10.5 forbids unilateral schema changes)."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    """Minimal contract for an item- or chunk-embedding model."""

    dim: int

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an (N, dim) float32 array."""
        ...


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """L2-normalize each row; zero rows stay zero (avoid NaN)."""
    x = x.astype(np.float32, copy=False)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    # Avoid division by zero — zero vectors legitimately stay at the origin.
    safe = np.where(norms == 0, 1.0, norms)
    return (x / safe).astype(np.float32)


def serialize_float32(v: np.ndarray) -> bytes:
    """Pack a 1D vector as little-endian float32 — the wire format sqlite-vec expects."""
    if v.dtype != np.float32:
        v = v.astype(np.float32)
    return v.tobytes()
