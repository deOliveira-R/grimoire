"""Sentence-aware chunker with page tracking and a sliding overlap window.

Input is a list of ``(page_number, text)`` tuples; output is a flat list of
``Chunk`` objects tagged with the page that the chunk's first token came from.
Paragraph-awareness is approximated by keeping sentences intact — a chunk
never cuts mid-sentence."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Naive but adequate for research-paper prose. Accepts ., !, ? followed by
# whitespace. Doesn't try to be clever about abbreviations.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True, slots=True)
class Chunk:
    page: int | None
    chunk_index: int
    text: str


def chunk_pages(
    pages: list[tuple[int, str]],
    target_words: int = 400,
    overlap_words: int = 40,
) -> list[Chunk]:
    """Pack sentences into target-sized chunks with word-level overlap.

    ``target_words`` ≈ 512 BPE tokens (plan §6 Phase 2). ``overlap_words`` ≈
    50 tokens, enough for semantic continuity without wasting embedding calls."""
    if target_words <= 0:
        raise ValueError("target_words must be > 0")
    if overlap_words < 0 or overlap_words >= target_words:
        raise ValueError("overlap_words must be in [0, target_words)")

    # Build (page, [words]) per sentence. Sentences longer than target_words
    # (common in stripped-PDF blobs without sentence punctuation) get sliced
    # so the packer never gets a "sentence" it can't fit.
    sentences: list[tuple[int, list[str]]] = []
    for page, text in pages:
        if not text:
            continue
        for raw in _SENTENCE_SPLIT.split(text):
            words = raw.split()
            if not words:
                continue
            if len(words) <= target_words:
                sentences.append((page, words))
            else:
                for off in range(0, len(words), target_words):
                    sentences.append((page, words[off : off + target_words]))

    if not sentences:
        return []

    chunks: list[Chunk] = []
    idx = 0
    i = 0
    while i < len(sentences):
        buf: list[str] = []
        first_page = sentences[i][0]
        j = i
        while j < len(sentences) and len(buf) < target_words:
            buf.extend(sentences[j][1])
            j += 1

        if not buf:
            break
        chunks.append(Chunk(page=first_page, chunk_index=idx, text=" ".join(buf)))
        idx += 1

        if j >= len(sentences):
            break

        # Slide: back up over sentences to accumulate ~overlap_words of overlap,
        # then advance at least one sentence so we don't loop forever.
        back_total = 0
        k = j
        while k > i + 1 and back_total < overlap_words:
            k -= 1
            back_total += len(sentences[k][1])
        i = max(k, i + 1)

    return chunks
