from __future__ import annotations

import hashlib

import numpy as np


class StubEmbedder:
    """Deterministic, cheap embedder for tests. Hash of input seeds an RNG,
    so repeated calls return the same vector without downloading any model."""

    def __init__(self, dim: int, fixed: dict[str, np.ndarray] | None = None) -> None:
        self.dim = dim
        self._fixed = {k: v.astype(np.float32) for k, v in (fixed or {}).items()}

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            if t in self._fixed:
                out[i] = self._fixed[t]
                continue
            seed = int.from_bytes(hashlib.sha256(t.encode()).digest()[:8], "big") % (2**32)
            rng = np.random.default_rng(seed)
            out[i] = rng.standard_normal(self.dim).astype(np.float32)
        return out
