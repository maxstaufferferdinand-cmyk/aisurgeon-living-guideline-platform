"""Test isolation from all real AISurgeon environment variables."""

import pytest


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

