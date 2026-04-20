"""GROBID TEI parser unit tests. No network — we feed synthetic TEI-XML."""

from __future__ import annotations

import pytest

from grimoire.extract.grobid import parse_tei

SAMPLE_TEI = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt>
        <title level="a" type="main">Boron dilution transients in pressurized water reactors</title>
      </titleStmt>
      <sourceDesc>
        <biblStruct>
          <analytic>
            <author>
              <persName>
                <forename type="first">Alice</forename>
                <surname>Smith</surname>
              </persName>
            </author>
            <author>
              <persName>
                <forename type="first">Bob</forename>
                <forename type="middle">J.</forename>
                <surname>Doe</surname>
              </persName>
            </author>
            <idno type="DOI">10.1016/j.nucengdes.2022.111111</idno>
          </analytic>
          <monogr>
            <title level="j">Nuclear Engineering and Design</title>
            <imprint>
              <date type="published" when="2022-03-15">2022</date>
            </imprint>
          </monogr>
        </biblStruct>
      </sourceDesc>
    </fileDesc>
    <profileDesc>
      <abstract>
        <p>We study boron dilution transients using a three-dimensional model.</p>
        <p>Results show that mixing effects dominate below 1 m/s flow velocity.</p>
      </abstract>
    </profileDesc>
  </teiHeader>
</TEI>"""


def test_parse_full_tei() -> None:
    md = parse_tei(SAMPLE_TEI)
    assert md is not None
    assert md.title == "Boron dilution transients in pressurized water reactors"
    assert "boron dilution transients" in (md.abstract or "").lower()
    assert "mixing effects" in (md.abstract or "").lower()
    assert md.publication_year == 2022
    assert md.doi == "10.1016/j.nucengdes.2022.111111"
    assert md.venue == "Nuclear Engineering and Design"
    assert md.source == "grobid"
    assert md.confidence == 0.85
    assert [a.family_name for a in md.authors] == ["Smith", "Doe"]
    assert md.authors[1].given_name == "Bob J."


def test_parse_with_only_title() -> None:
    xml = """<?xml version="1.0"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt><title>Just a title</title></titleStmt>
      <sourceDesc><biblStruct/></sourceDesc>
    </fileDesc>
  </teiHeader>
</TEI>"""
    md = parse_tei(xml)
    assert md is not None
    assert md.title == "Just a title"
    assert md.abstract is None
    assert md.authors == []


def test_parse_empty_header_returns_none() -> None:
    xml = """<?xml version="1.0"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt/>
      <sourceDesc><biblStruct/></sourceDesc>
    </fileDesc>
  </teiHeader>
</TEI>"""
    md = parse_tei(xml)
    assert md is None


def test_parse_invalid_xml() -> None:
    assert parse_tei("<<<not xml>>>") is None


def test_parse_doi_from_bare_idno_type_variants() -> None:
    xml = """<?xml version="1.0"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt><title>X</title></titleStmt>
      <sourceDesc>
        <biblStruct>
          <analytic>
            <idno type="doi">10.1234/lowercased</idno>
          </analytic>
        </biblStruct>
      </sourceDesc>
    </fileDesc>
  </teiHeader>
</TEI>"""
    md = parse_tei(xml)
    assert md is not None
    assert md.doi == "10.1234/lowercased"


def test_parse_no_idno_but_doi_in_text() -> None:
    """GROBID sometimes leaves the DOI loose in header text; we regex-rescue it."""
    xml = """<?xml version="1.0"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt><title>A paper</title></titleStmt>
      <sourceDesc>
        <biblStruct>
          <note>see 10.1038/s41586-024-99999-9 for details.</note>
        </biblStruct>
      </sourceDesc>
    </fileDesc>
  </teiHeader>
</TEI>"""
    md = parse_tei(xml)
    assert md is not None
    assert md.doi == "10.1038/s41586-024-99999-9"


def test_extract_header_no_url_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    from grimoire.extract import grobid

    monkeypatch.setattr("grimoire.config.settings.grobid_url", None)
    from pathlib import Path

    assert grobid.extract_header(Path("/does/not/matter.pdf")) is None
