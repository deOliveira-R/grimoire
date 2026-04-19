"""SPECTER2 paper embedder. 768-dim. Expects ``"title [SEP] abstract"`` strings.

The transformers/torch imports are deferred to first use so that importing the
package (for CLI help, ingest-only workflows, etc.) never triggers a torch load.
The ``ml`` extra pulls the weights on first encode."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from grimoire.embed.base import l2_normalize

log = logging.getLogger(__name__)

MODEL_NAME = "allenai/specter2_base"
DIM = 768


class Specter2Embedder:
    dim: int = DIM

    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None) -> None:
        self._model_name = model_name
        self._device = device
        self._model: Any = None
        self._tokenizer: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "SPECTER2 requires the 'ml' extra. Install with pip install 'grimoire[ml]'"
            ) from exc

        log.info("loading SPECTER2 model: %s", self._model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModel.from_pretrained(self._model_name)
        self._model.eval()
        if self._device:
            self._model.to(self._device)
        self._torch = torch

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        self._load()
        torch = self._torch
        inputs = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
            return_token_type_ids=False,
        )
        if self._device:
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model(**inputs)
        # CLS-token pooling — standard for SPECTER2.
        cls = out.last_hidden_state[:, 0, :].cpu().numpy().astype(np.float32)
        return l2_normalize(cls)


def format_item_text(title: str, abstract: str | None, body_fallback: str | None = None) -> str:
    """Plan §8: SPECTER2 expects title + [SEP] + abstract. When abstract is empty,
    fall back to the first 500 words of the body."""
    filler = (abstract or "").strip()
    if not filler and body_fallback:
        filler = " ".join(body_fallback.split()[:500])
    return f"{title.strip()} [SEP] {filler}"
