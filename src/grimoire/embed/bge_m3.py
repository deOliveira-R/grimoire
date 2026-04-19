"""BGE-M3 chunk embedder. 1024-dim. 8k context."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from grimoire.embed.base import l2_normalize

log = logging.getLogger(__name__)

MODEL_NAME = "BAAI/bge-m3"
DIM = 1024


class BGEM3Embedder:
    dim: int = DIM

    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None) -> None:
        self._model_name = model_name
        self._device = device
        self._model: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "BGE-M3 requires the 'ml' extra. Install with pip install 'grimoire[ml]'"
            ) from exc

        log.info("loading BGE-M3 model: %s", self._model_name)
        self._model = SentenceTransformer(self._model_name, device=self._device)

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        self._load()
        arr = self._model.encode(
            texts,
            batch_size=8,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        ).astype(np.float32)
        return l2_normalize(arr)
