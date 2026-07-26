"""Deterministic local PDF preflight for canonical transcription v3."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pypdf import PdfReader

from aisurgeon.extraction.canonical.outputs import write_json, write_jsonl
from aisurgeon.extraction.pdf_registration import register_pdf

PDF_PREFLIGHT_SCHEMA_VERSION = "pdf_preflight_v1"


class PdfPagePreflight(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = PDF_PREFLIGHT_SCHEMA_VERSION
    source_id: str
    page_number: int = Field(ge=1)
    width: float
    height: float
    rotation: int
    text_layer_character_count: int = Field(ge=0)
    image_count: int = Field(ge=0)
    image_heavy: bool
    text_heavy: bool
    approximate_page_density_score: float = Field(ge=0)
    obvious_blank_page: bool
    technical_warnings: list[str] = Field(default_factory=list)


class PdfPreflight(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = PDF_PREFLIGHT_SCHEMA_VERSION
    source_id: str
    pdf_sha256: str
    file_size_bytes: int
    pdf_version: str | None
    encrypted: bool
    page_count: int
    technical_warnings: list[str] = Field(default_factory=list)


def _safe_extract_text(page: Any) -> str:
    try:
        return page.extract_text() or ""
    except Exception:
        return ""


def _image_count(page: Any) -> int:
    try:
        xobjects = page.get("/Resources", {}).get("/XObject", {})
        return len(xobjects) if hasattr(xobjects, "__len__") else 0
    except Exception:
        return 0


def run_pdf_preflight(
    *,
    pdf_path: Path,
    source_id: str,
    worker_id: str,
    output_dir: Path | None = None,
) -> tuple[PdfPreflight, list[PdfPagePreflight]]:
    """Collect non-semantic technical PDF facts and optionally persist them."""
    registration = register_pdf(pdf_path, worker_id=worker_id, source_id=source_id)
    if registration.page_count is None:
        raise ValueError("Encrypted PDF page count is unavailable")
    reader = PdfReader(registration.resolved_local_path, strict=True)
    pages: list[PdfPagePreflight] = []
    for index, page in enumerate(reader.pages, start=1):
        media = page.mediabox
        width = float(media.width)
        height = float(media.height)
        text = _safe_extract_text(page)
        chars = len(text.strip())
        images = _image_count(page)
        area = max(width * height, 1.0)
        density = round(chars / (area / 1000.0), 4)
        warnings: list[str] = []
        if not text and images:
            warnings.append("no_text_layer_with_images")
        if not text and not images:
            warnings.append("no_text_or_images_detected")
        pages.append(
            PdfPagePreflight(
                source_id=registration.source_id,
                page_number=index,
                width=width,
                height=height,
                rotation=int(page.get("/Rotate", 0) or 0),
                text_layer_character_count=chars,
                image_count=images,
                image_heavy=images > 0 and chars < 200,
                text_heavy=chars >= 800,
                approximate_page_density_score=density,
                obvious_blank_page=chars == 0 and images == 0,
                technical_warnings=warnings,
            )
        )
    preflight = PdfPreflight(
        source_id=registration.source_id,
        pdf_sha256=registration.sha256,
        file_size_bytes=registration.file_size_bytes,
        pdf_version=registration.pdf_version,
        encrypted=registration.encrypted,
        page_count=registration.page_count,
        technical_warnings=[],
    )
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "pdf_preflight.json", preflight)
        write_jsonl(output_dir / "page_preflight.jsonl", pages)
    return preflight, pages
