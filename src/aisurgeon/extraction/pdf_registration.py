"""Deterministic, non-semantic local PDF registration."""

import hashlib
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

from pydantic import BaseModel, ConfigDict, Field
from pypdf import PdfReader
from pypdf.errors import FileNotDecryptedError, PdfReadError

PDF_REGISTRATION_SCHEMA_VERSION = "pdf_registration_v1"
SOURCE_ID_VERSION = "pdf-v1"
_SOURCE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_PDF_HEADER_PATTERN = re.compile(br"%PDF-(\d\.\d)")


class PdfRegistrationError(ValueError):
    """Raised when a local file cannot be safely registered as a PDF."""


class PdfRegistration(BaseModel):
    """Technical metadata for one immutable local PDF fingerprint."""

    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(min_length=1)
    original_filename: str = Field(min_length=1)
    resolved_local_path: Path
    file_size_bytes: int = Field(ge=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    page_count: int | None = Field(default=None, ge=1)
    pdf_version: str | None = None
    encrypted: bool
    registration_timestamp_utc: datetime
    worker_id: str = Field(min_length=1)
    registration_schema_version: str = PDF_REGISTRATION_SCHEMA_VERSION


def sha256_stream(stream: BinaryIO, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a binary stream without loading it into memory."""
    digest = hashlib.sha256()
    while chunk := stream.read(chunk_size):
        digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 hash for a file."""
    with path.open("rb") as stream:
        return sha256_stream(stream)


def deterministic_source_id(pdf_sha256: str) -> str:
    """Create a versioned identity from the canonical PDF fingerprint."""
    if not re.fullmatch(r"[0-9a-f]{64}", pdf_sha256):
        raise PdfRegistrationError("PDF-SHA-256 ist ungültig.")
    return f"{SOURCE_ID_VERSION}-{pdf_sha256[:20]}"


def _read_pdf_version(path: Path) -> str | None:
    with path.open("rb") as stream:
        header = stream.read(1024)
    match = _PDF_HEADER_PATTERN.search(header)
    return match.group(1).decode("ascii") if match else None


def register_pdf(
    path: Path,
    *,
    worker_id: str,
    source_id: str | None = None,
    registered_at: datetime | None = None,
) -> PdfRegistration:
    """Register technical PDF metadata without extracting semantic text."""
    if not worker_id.strip():
        raise PdfRegistrationError("Worker-ID fehlt.")
    if not path.exists():
        raise PdfRegistrationError("PDF-Datei existiert nicht.")
    if not path.is_file():
        raise PdfRegistrationError("PDF-Pfad ist keine reguläre Datei.")
    if path.suffix.lower() != ".pdf":
        raise PdfRegistrationError("Dateiendung muss .pdf sein.")
    pdf_version = _read_pdf_version(path)
    if pdf_version is None:
        raise PdfRegistrationError("PDF-Signatur ist ungültig.")

    resolved = path.resolve(strict=True)
    fingerprint = sha256_file(resolved)
    selected_source_id = source_id or deterministic_source_id(fingerprint)
    if not _SOURCE_ID_PATTERN.fullmatch(selected_source_id):
        raise PdfRegistrationError("source_id enthält unzulässige Zeichen.")

    try:
        reader = PdfReader(resolved, strict=True)
        encrypted = reader.is_encrypted
        try:
            page_count = len(reader.pages)
        except FileNotDecryptedError:
            page_count = None
    except (PdfReadError, OSError, ValueError) as exc:
        raise PdfRegistrationError("PDF-Struktur ist ungültig oder nicht lesbar.") from exc

    return PdfRegistration(
        source_id=selected_source_id,
        original_filename=resolved.name,
        resolved_local_path=resolved,
        file_size_bytes=resolved.stat().st_size,
        sha256=fingerprint,
        page_count=page_count,
        pdf_version=pdf_version,
        encrypted=encrypted,
        registration_timestamp_utc=registered_at or datetime.now(UTC),
        worker_id=worker_id.strip(),
    )
