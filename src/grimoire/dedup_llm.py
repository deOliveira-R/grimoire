"""LLM judge for the ambiguous dedup band (0.90 ≤ sim < 0.97).

Returns:
  - 'same'      — the two items describe the same publication
  - 'related'   — different publications but tightly coupled (same study, erratum, etc.)
  - 'different' — semantically close but distinct papers

Returns None when ANTHROPIC_API_KEY is not configured or the call fails.
Callers treat None the same as 'different' for safety (plan §3: false positives
matter more than false negatives)."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal

from grimoire.config import settings
from grimoire.models import Metadata

log = logging.getLogger(__name__)


Verdict = Literal["same", "related", "different"]

_SYSTEM_PROMPT = """You are judging whether two bibliographic records describe the SAME publication.

You will receive two records, each with title, abstract, authors, and identifiers.

Classify as one of:
- "same": the records describe the same work (e.g. preprint and published version,
   OCR'd copy vs. clean copy, minor wording variants). Identical DOIs, arXiv IDs,
   or closely-matched title+abstract+author-set almost always fall here.
- "related": the records describe different but tightly coupled works — an
   erratum, a follow-up paper, or a later edition of the same underlying study.
- "different": the records are semantically close (same topic, possibly shared
   authors) but describe distinct contributions.

Be strict: false positives for "same" are worse than false negatives. When in
doubt, prefer "related" or "different".

Reply with ONLY a JSON object of the shape {"verdict": "same" | "related" | "different"}.
No prose, no markdown fences."""


def judge(candidate: Metadata, existing: Metadata) -> Verdict | None:
    if not settings.anthropic_api_key:
        return None

    try:
        import anthropic
    except ImportError:  # pragma: no cover - anthropic is in the ingest extra
        log.warning("anthropic SDK not installed; LLM judge disabled")
        return None

    try:
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        message = client.messages.create(
            model=settings.llm_model,
            max_tokens=64,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _format_pair(candidate, existing)}],
        )
    except Exception as exc:
        log.warning("LLM judge call failed: %s", exc)
        return None

    text = "".join(
        block.text
        for block in message.content
        if getattr(block, "type", None) == "text" and hasattr(block, "text")
    )
    return _parse_verdict(text)


def _format_pair(a: Metadata, b: Metadata) -> str:
    return f"Record A:\n{_format_one(a)}\n\nRecord B:\n{_format_one(b)}"


def _format_one(m: Metadata) -> str:
    author_line = ", ".join(
        (f"{x.family_name}, {x.given_name}" if x.given_name else x.family_name) for x in m.authors
    )
    parts = [
        f"  title: {m.title or '(missing)'}",
        f"  authors: {author_line or '(missing)'}",
        f"  year: {m.publication_year if m.publication_year else '(missing)'}",
        f"  doi: {m.doi or '(none)'}",
        f"  arxiv: {m.arxiv_id or '(none)'}",
        f"  abstract: {_trim(m.abstract, 1200) if m.abstract else '(missing)'}",
    ]
    return "\n".join(parts)


def _trim(s: str, n: int) -> str:
    s = s.strip()
    return s if len(s) <= n else s[:n] + " …"


_VERDICT_RE = re.compile(r'"verdict"\s*:\s*"(same|related|different)"', re.IGNORECASE)


def _parse_verdict(text: str) -> Verdict | None:
    m = _VERDICT_RE.search(text)
    if m:
        v = m.group(1).lower()
        if v in {"same", "related", "different"}:
            return v  # type: ignore[return-value]
    # Last-ditch JSON attempt
    try:
        blob: Any = json.loads(text)
    except Exception:
        return None
    if isinstance(blob, dict):
        v2 = str(blob.get("verdict", "")).lower()
        if v2 in {"same", "related", "different"}:
            return v2  # type: ignore[return-value]
    return None
