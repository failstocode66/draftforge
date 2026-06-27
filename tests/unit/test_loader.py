from pathlib import Path

import pytest

from draftforge.ingest.loader import (
    EmptyDocumentError,
    load_document,
    load_document_meta,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_loads_txt():
    text = load_document(FIXTURES / "sample.txt")
    assert "Float therapy" in text
    assert "Epsom salt" in text


def test_loads_md_raw():
    text = load_document(FIXTURES / "sample.md")
    # Raw markdown is preserved (heading marker, bold, bullets all intact).
    assert "# Float Therapy Benefits" in text
    assert "**sensory-deprivation tank**" in text
    assert "- Deep relaxation" in text


def test_loads_pdf_with_extractable_text():
    text = load_document(FIXTURES / "sample.pdf")
    assert "Float therapy" in text
    assert "Epsom-salt water" in text


def test_missing_file_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_document(FIXTURES / "does_not_exist.txt")


def test_pdf_with_no_text_raises_empty_document_error():
    with pytest.raises(EmptyDocumentError):
        load_document(FIXTURES / "no_text.pdf")


def test_whitespace_only_txt_raises_empty_document_error(tmp_path):
    blank = tmp_path / "blank.txt"
    blank.write_text("   \n\t  \n", encoding="utf-8")
    with pytest.raises(EmptyDocumentError):
        load_document(blank)


def test_accepts_string_path(tmp_path):
    p = tmp_path / "note.txt"
    p.write_text("hello world", encoding="utf-8")
    assert load_document(str(p)) == "hello world"


def test_load_document_meta_has_parameter_type_hints():
    # P1a polish: the public loader entrypoints carry parameter type hints.
    import typing

    hints = typing.get_type_hints(load_document_meta)
    assert "path" in hints
    assert "max_chars" in hints
    assert hints["return"] is not None


def test_truncation_meta_flag_set_when_over_max_chars(tmp_path):
    big = tmp_path / "big.txt"
    big.write_text("x" * 500, encoding="utf-8")

    meta = load_document_meta(big, max_chars=100)
    assert meta.truncated is True
    assert len(meta.text) == 100
    assert meta.original_length == 500


def test_truncation_meta_flag_clear_when_under_max_chars(tmp_path):
    small = tmp_path / "small.txt"
    small.write_text("x" * 50, encoding="utf-8")

    meta = load_document_meta(small, max_chars=100)
    assert meta.truncated is False
    assert meta.text == "x" * 50
    assert meta.original_length == 50


def test_load_document_facade_returns_truncated_text(tmp_path):
    # load_document is the str-returning facade over load_document_meta: when the
    # source exceeds max_chars it returns the CAPPED text (not the full content),
    # and the meta path reports truncated=True for the same input.
    big = tmp_path / "big.txt"
    big.write_text("z" * 500, encoding="utf-8")

    text = load_document(big, max_chars=120)
    assert isinstance(text, str)
    assert len(text) == 120  # actually capped, not the full 500
    assert text == "z" * 120

    meta = load_document_meta(big, max_chars=120)
    assert meta.truncated is True
    assert meta.text == text  # facade returns exactly the meta's (capped) text
