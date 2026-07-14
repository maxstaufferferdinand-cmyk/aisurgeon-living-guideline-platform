"""Tests for typed and secret-safe settings."""

from pathlib import Path

from pydantic import SecretStr

from aisurgeon.config import Settings


def test_settings_are_typed_and_derive_paths(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "pdfs"
    settings = Settings(
        AISURGEON_WORKER_ID="worker-test",
        AISURGEON_DATA_ROOT=tmp_path,
        AISURGEON_PDF_SOURCE_DIR=pdf_dir,
        GEMINI_API_KEY="dummy-secret",
    )

    assert settings.data_root == tmp_path
    assert settings.pdf_source_dir == pdf_dir
    assert settings.runs_dir == tmp_path / "runs"
    assert isinstance(settings.gemini_api_key, SecretStr)
    assert "dummy-secret" not in repr(settings)


def test_explicit_output_path_overrides_data_root(tmp_path: Path) -> None:
    override = tmp_path / "special-runs"
    settings = Settings(
        AISURGEON_DATA_ROOT=tmp_path / "root",
        AISURGEON_RUNS_DIR=override,
    )
    assert settings.runs_dir == override


def test_temporary_env_file_is_explicitly_loaded(tmp_path: Path) -> None:
    env_file = tmp_path / "isolated.env"
    env_file.write_text(
        "AISURGEON_WORKER_ID=env-worker\nAISURGEON_DATA_ROOT=/tmp/isolated-root\n",
        encoding="utf-8",
    )
    settings = Settings.from_env_file(env_file)
    assert settings.worker_id == "env-worker"

