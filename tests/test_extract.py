from __future__ import annotations

from pathlib import Path

from grimoire.extract import epub as epub_extract
from grimoire.extract import pdf as pdf_extract


class TestPDF:
    def test_extract_text(self, pdf_with_doi: Path) -> None:
        text = pdf_extract.extract_text(pdf_with_doi)
        assert "A Nice Paper" in text
        assert "10.1234/example.2024" in text

    def test_extract_first_page(self, pdf_with_doi: Path) -> None:
        first = pdf_extract.extract_first_page(pdf_with_doi)
        assert "A Nice Paper" in first

    def test_page_count(self, pdf_with_doi: Path) -> None:
        assert pdf_extract.page_count(pdf_with_doi) == 1


class TestEPUB:
    def test_extract_text(self, sample_epub: Path) -> None:
        text = epub_extract.extract_text(sample_epub)
        assert "Hello world" in text
        assert "Chapter 1" in text

    def test_extract_metadata(self, sample_epub: Path) -> None:
        meta = epub_extract.extract_metadata(sample_epub)
        assert meta["title"] == "Sample Book"
        assert "Jane Doe" in meta["creators"]
