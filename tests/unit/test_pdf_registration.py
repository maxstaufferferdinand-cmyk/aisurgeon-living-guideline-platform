"""Deterministic technical PDF registration tests."""

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pypdf import PdfWriter

from aisurgeon.extraction.pdf_registration import (
    PdfRegistrationError,
    deterministic_source_id,
    register_pdf,
    sha256_file,
)


@pytest.fixture
def synthetic_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "synthetic.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.add_blank_page(width=72, height=72)
    with path.open("wb") as stream:
        writer.write(stream)
    return path


def test_sha256_is_correct_and_streaming(synthetic_pdf: Path) -> None:
    expected = hashlib.sha256(synthetic_pdf.read_bytes()).hexdigest()
    assert sha256_file(synthetic_pdf) == expected


def test_pdf_registration_captures_technical_metadata(synthetic_pdf: Path) -> None:
    timestamp = datetime(2026, 7, 14, tzinfo=UTC)
    registration = register_pdf(
        synthetic_pdf,
        worker_id="worker-test",
        registered_at=timestamp,
    )
    assert registration.original_filename == "synthetic.pdf"
    assert registration.resolved_local_path == synthetic_pdf.resolve()
    assert registration.page_count == 2
    assert registration.pdf_version is not None
    assert registration.file_size_bytes == synthetic_pdf.stat().st_size
    assert registration.encrypted is False
    assert registration.registration_timestamp_utc == timestamp


def test_source_id_is_deterministic_and_hash_based(synthetic_pdf: Path) -> None:
    first = register_pdf(synthetic_pdf, worker_id="one")
    second = register_pdf(synthetic_pdf, worker_id="two")
    assert first.source_id == second.source_id
    assert first.source_id == deterministic_source_id(first.sha256)
    assert synthetic_pdf.stem not in first.source_id


def test_explicit_source_id_is_preserved(synthetic_pdf: Path) -> None:
    registration = register_pdf(
        synthetic_pdf,
        worker_id="worker-test",
        source_id="OWNER-SOURCE-001",
    )
    assert registration.source_id == "OWNER-SOURCE-001"


def test_encrypted_pdf_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "encrypted.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.encrypt("synthetic-password")
    with path.open("wb") as stream:
        writer.write(stream)
    registration = register_pdf(path, worker_id="worker-test")
    assert registration.encrypted is True
    assert registration.page_count is None


@pytest.mark.parametrize(
    ("filename", "content"),
    [("invalid.pdf", b"not a PDF"), ("wrong.txt", b"%PDF-1.4\n")],
)
def test_invalid_pdf_is_rejected(tmp_path: Path, filename: str, content: bytes) -> None:
    path = tmp_path / filename
    path.write_bytes(content)
    with pytest.raises(PdfRegistrationError):
        register_pdf(path, worker_id="worker-test")

