"""Chapter-boundary extraction for books (plan §6 Phase 6).

Two strategies, tried in order:
  * PDFs: pymupdf's TOC (doc.get_toc()) — we take top-level entries as
    chapters and compute page ranges.
  * EPUBs: the EPUB spine (reading order) combined with NCX / nav TOC titles.

When neither source yields structure, returns ``None`` — the caller keeps
the book as a single item rather than guessing at chapters. Heuristic
chapter detection (regex "Chapter N" on body text, GROBID processFulltext)
is deferred to v2; on reflowable books without bookmarks the rewards are
thin and the false-positive rate is high."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ChapterSpec:
    """One chapter's boundary inside a parent book file.

    For PDFs ``start_page`` / ``end_page`` are 1-indexed and inclusive; for
    EPUBs they're ``None`` since EPUBs don't have stable page numbers.
    ``spine_href`` points at the underlying EPUB item for later re-extraction."""

    index: int  # 0-based position in the book (chapter order)
    title: str
    start_page: int | None = None
    end_page: int | None = None
    spine_href: str | None = None


def detect(path: Path) -> list[ChapterSpec] | None:
    """Return chapter boundaries, or ``None`` if nothing detectable.

    The CAS stores files as ``{hash}`` (no extension), so we sniff magic
    bytes and fall back to the suffix — both call sites (ingest-time with
    an original ``.pdf``/``.epub`` path and post-ingest through the CAS)
    work transparently."""
    kind = _detect_kind(path)
    if kind == "pdf":
        return _pdf_chapters(path)
    if kind == "epub":
        return _epub_chapters(path)
    return None


def _detect_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix == ".epub":
        return "epub"
    try:
        with path.open("rb") as fh:
            head = fh.read(4)
    except OSError:
        return "unknown"
    if head[:4] == b"%PDF":
        return "pdf"
    if head[:2] == b"PK":  # ZIP magic — EPUBs are ZIP containers.
        return "epub"
    return "unknown"


# ---------- PDF --------------------------------------------------------------


def _pdf_chapters(path: Path) -> list[ChapterSpec] | None:
    import pymupdf

    with pymupdf.open(path) as doc:
        toc = doc.get_toc()  # [[level, title, page], ...]
        if not toc:
            return None
        page_count = doc.page_count

    top_level = [entry for entry in toc if entry[0] == 1]
    # Single top-level entry or empty top level → there's nothing to split:
    # the whole PDF is effectively one chapter (or one document).
    if len(top_level) < 2:
        return None

    specs: list[ChapterSpec] = []
    for i, entry in enumerate(top_level):
        _, title, start = entry
        # pymupdf pages are 1-indexed but occasionally emit 0 for unknown; clamp.
        start = max(1, int(start))
        if i + 1 < len(top_level):
            end = max(start, int(top_level[i + 1][2]) - 1)
        else:
            end = page_count
        specs.append(
            ChapterSpec(
                index=i,
                title=(title or f"Chapter {i + 1}").strip(),
                start_page=start,
                end_page=end,
            )
        )
    return specs


# ---------- EPUB -------------------------------------------------------------


def _epub_chapters(path: Path) -> list[ChapterSpec] | None:
    import ebooklib
    from ebooklib import epub

    book = epub.read_epub(str(path))

    # Titles come from the NCX TOC; the spine gives the reading order. If the
    # spine has fewer than 2 content documents, splitting is pointless.
    spine_items: list[tuple[str, ebooklib.epub.EpubHtml]] = []
    for item_ref in book.spine:
        # Spine entries are ``(idref, linear)`` tuples — ignore nav entries
        # (which are typically ``idref == 'nav'`` and contain the TOC itself).
        idref = item_ref[0] if isinstance(item_ref, tuple) else item_ref
        if idref == "nav":
            continue
        item = book.get_item_with_id(idref)
        if item is None or item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue
        spine_items.append((idref, item))

    if len(spine_items) < 2:
        return None

    titles_by_href = _epub_toc_titles(book.toc)

    specs: list[ChapterSpec] = []
    for i, (_, item) in enumerate(spine_items):
        href = item.get_name()
        title = titles_by_href.get(href) or _fallback_title(item) or f"Chapter {i + 1}"
        specs.append(ChapterSpec(index=i, title=title.strip(), spine_href=href))
    return specs


def _epub_toc_titles(toc: object) -> dict[str, str]:
    """Walk the EPUB TOC tree to build a ``{href: title}`` map.

    EPUB TOC entries are either ``epub.Link`` objects or nested ``(Section,
    [entries])`` tuples. ``Link.href`` may include a fragment ``#anchor``
    which we strip so it matches the spine item path."""
    from ebooklib import epub

    out: dict[str, str] = {}

    def _walk(node: object) -> None:
        if isinstance(node, tuple) and len(node) == 2:
            _, children = node
            for c in children:
                _walk(c)
        elif isinstance(node, list):
            for c in node:
                _walk(c)
        elif isinstance(node, epub.Link):
            href = node.href.split("#", 1)[0]
            if href and href not in out:
                out[href] = node.title or ""

    _walk(toc)
    return out


def _fallback_title(item: object) -> str | None:
    """Pull the first <h1>/<h2>/<title> from the chapter body as a fallback."""
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(item.get_content(), "html.parser")  # type: ignore[attr-defined]
        for tag in ("h1", "h2", "title"):
            el = soup.find(tag)
            if el and el.get_text(strip=True):
                return el.get_text(strip=True)
    except Exception:
        return None
    return None


# ---------- text slicing (used by the indexer) --------------------------------


def pdf_chapter_text(path: Path, start_page: int, end_page: int) -> list[tuple[int, str]]:
    """Return ``[(page_number, text), ...]`` for the given PDF page range.

    Page numbers are preserved so chunk metadata still points at the right
    page in the underlying book."""
    import pymupdf

    out: list[tuple[int, str]] = []
    with pymupdf.open(path) as doc:
        lo = max(1, start_page)
        hi = min(doc.page_count, end_page)
        for pageno in range(lo, hi + 1):
            text = doc.load_page(pageno - 1).get_text()
            if text and text.strip():
                out.append((pageno, text))
    return out


def epub_chapter_text(path: Path, spine_href: str) -> str:
    """Return the text of a single EPUB spine entry."""
    from bs4 import BeautifulSoup
    from ebooklib import epub

    book = epub.read_epub(str(path))
    for item in book.get_items():
        if item.get_name() == spine_href:
            soup = BeautifulSoup(item.get_content(), "html.parser")
            return soup.get_text(separator="\n")
    return ""
