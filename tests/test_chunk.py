from __future__ import annotations

from itertools import pairwise

from grimoire.chunk import Chunk, chunk_pages


def _words(s: str) -> int:
    return len(s.split())


def test_single_short_page_becomes_one_chunk() -> None:
    chunks = chunk_pages([(1, "One sentence only.")], target_words=400, overlap_words=40)
    assert len(chunks) == 1
    assert chunks[0].page == 1
    assert chunks[0].chunk_index == 0


def test_chunks_respect_target_word_count() -> None:
    # 1000-word stream, target 200, overlap 20
    long = " ".join(f"w{i}" for i in range(1000))
    chunks = chunk_pages([(1, long)], target_words=200, overlap_words=20)
    # Expect several chunks, each near target size, monotonically indexed.
    assert len(chunks) > 1
    for c in chunks:
        assert _words(c.text) <= 260  # small slack for sentence-aware boundary
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_overlap_between_adjacent_chunks() -> None:
    text = " ".join(f"w{i}." for i in range(500))  # 500 mini-sentences
    chunks = chunk_pages([(1, text)], target_words=100, overlap_words=20)
    # Consecutive chunks should share some suffix/prefix words.
    for a, b in pairwise(chunks):
        a_tail = set(a.text.split()[-30:])
        b_head = set(b.text.split()[:30])
        assert a_tail & b_head, "expected word overlap between adjacent chunks"


def test_page_tracking_across_pages() -> None:
    pages = [
        (1, " ".join(f"p1w{i}" for i in range(150))),
        (2, " ".join(f"p2w{i}" for i in range(150))),
        (3, " ".join(f"p3w{i}" for i in range(150))),
    ]
    chunks = chunk_pages(pages, target_words=100, overlap_words=10)
    seen_pages = {c.page for c in chunks}
    assert seen_pages == {1, 2, 3}
    # First chunk starts on page 1.
    assert chunks[0].page == 1


def test_empty_input() -> None:
    assert chunk_pages([], target_words=400, overlap_words=40) == []
    assert chunk_pages([(1, "")], target_words=400, overlap_words=40) == []


def test_chunk_is_stable() -> None:
    text = " ".join(f"word{i}" for i in range(500))
    a = chunk_pages([(1, text)], target_words=100, overlap_words=20)
    b = chunk_pages([(1, text)], target_words=100, overlap_words=20)
    assert [(c.page, c.chunk_index, c.text) for c in a] == [
        (c.page, c.chunk_index, c.text) for c in b
    ]


def test_chunk_dataclass_fields() -> None:
    c = Chunk(page=1, chunk_index=0, text="hello")
    assert c.page == 1 and c.chunk_index == 0 and c.text == "hello"
