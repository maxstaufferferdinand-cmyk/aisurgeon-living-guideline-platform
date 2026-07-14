"""Test isolation from all real AISurgeon environment variables."""

import pytest
from pypdf import PdfWriter


@pytest.fixture(autouse=True)
def isolate_aisurgeon_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "AISURGEON_WORKER_ID",
        "AISURGEON_DATA_ROOT",
        "AISURGEON_PDF_SOURCE_DIR",
        "AISURGEON_RUNS_DIR",
        "AISURGEON_CACHE_DIR",
        "AISURGEON_EXPORTS_DIR",
        "AISURGEON_LOGS_DIR",
        "GEMINI_API_KEY",
        "OPENAI_API_KEY",
        "NCBI_API_KEY",
        "NCBI_EMAIL",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def synthetic_pdf(tmp_path):
    """Create a tiny synthetic PDF without using any guideline fixture."""
    path = tmp_path / "synthetic.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.add_blank_page(width=72, height=72)
    with path.open("wb") as stream:
        writer.write(stream)
    return path
