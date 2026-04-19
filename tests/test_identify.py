from __future__ import annotations

from grimoire.identify import (
    extract_arxiv_ids,
    extract_dois,
    extract_isbns,
    identify,
    is_valid_isbn,
    normalize_isbn,
)


class TestDOI:
    def test_plain_doi(self) -> None:
        assert "10.1234/example.2024" in extract_dois("see 10.1234/example.2024 for details")

    def test_doi_url(self) -> None:
        dois = extract_dois("https://doi.org/10.1038/s41586-024-12345-6 ok")
        assert "10.1038/s41586-024-12345-6" in dois

    def test_doi_prefix(self) -> None:
        dois = extract_dois("doi: 10.1016/j.nuclengdes.2023.112345")
        assert "10.1016/j.nuclengdes.2023.112345" in dois

    def test_trailing_punctuation_stripped(self) -> None:
        dois = extract_dois("See 10.1234/foo.bar.")
        assert "10.1234/foo.bar" in dois

    def test_no_doi(self) -> None:
        assert extract_dois("no identifier here") == []

    def test_dedup(self) -> None:
        dois = extract_dois("10.1234/x and again 10.1234/x")
        assert dois.count("10.1234/x") == 1


class TestArxiv:
    def test_new_style(self) -> None:
        assert extract_arxiv_ids("preprint arXiv:2401.01234") == ["2401.01234"]

    def test_new_style_with_version(self) -> None:
        ids = extract_arxiv_ids("see arXiv:2401.01234v2 for revision")
        assert ids == ["2401.01234"]

    def test_old_style(self) -> None:
        assert extract_arxiv_ids("hep-th arXiv:hep-th/9901001") == ["hep-th/9901001"]

    def test_url(self) -> None:
        assert extract_arxiv_ids("https://arxiv.org/abs/2401.01234") == ["2401.01234"]

    def test_no_arxiv(self) -> None:
        assert extract_arxiv_ids("nothing") == []


class TestISBN:
    def test_isbn13_valid(self) -> None:
        assert is_valid_isbn("9780131101630")

    def test_isbn10_valid(self) -> None:
        assert is_valid_isbn("0131101633")

    def test_isbn_with_dashes(self) -> None:
        assert is_valid_isbn("978-0-13-110163-0")

    def test_isbn_invalid_checksum(self) -> None:
        assert not is_valid_isbn("9780131101631")

    def test_normalize_to_isbn13(self) -> None:
        assert normalize_isbn("0131101633") == "9780131101630"
        assert normalize_isbn("978-0-13-110163-0") == "9780131101630"

    def test_extract_with_prefix(self) -> None:
        isbns = extract_isbns("ISBN: 978-0-13-110163-0")
        assert "9780131101630" in isbns

    def test_rejects_random_digits(self) -> None:
        assert extract_isbns("phone 415-555-1234 more") == []


class TestIdentifyFacade:
    def test_identify_returns_all(self) -> None:
        text = "Title. doi: 10.1234/x. arXiv:2401.00001. ISBN 978-0-13-110163-0"
        result = identify(text)
        assert result.dois == ["10.1234/x"]
        assert result.arxiv_ids == ["2401.00001"]
        assert result.isbns == ["9780131101630"]

    def test_empty(self) -> None:
        result = identify("")
        assert result.dois == [] and result.arxiv_ids == [] and result.isbns == []
