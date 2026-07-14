"""Isolated CLI tests using only temporary local configuration."""

import os
from pathlib import Path

from typer.testing import CliRunner

from aisurgeon.cli.app import app

runner = CliRunner()


def _write_env(path: Path, data_root: Path, pdf_dir: Path, *, worker: str = "test-worker") -> None:
    path.write_text(
        f"AISURGEON_WORKER_ID={worker}\n"
        f"AISURGEON_DATA_ROOT={data_root}\n"
        f"AISURGEON_PDF_SOURCE_DIR={pdf_dir}\n",
        encoding="utf-8",
    )


def test_help_works_without_configuration() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "config-check" in result.output
    assert "setup-local" in result.output


def test_config_check_reports_only_credential_state(tmp_path: Path) -> None:
    data_root = tmp_path / "root"
    pdf_dir = tmp_path / "pdfs"
    output_dirs = tuple(data_root / name for name in ("runs", "cache", "exports", "logs"))
    for directory in (data_root, pdf_dir, *output_dirs):
        directory.mkdir(parents=True, exist_ok=True)
    env_file = tmp_path / "check.env"
    secret = "PREFIX_super_dummy_value_SUFFIX"
    _write_env(env_file, data_root, pdf_dir)
    with env_file.open("a", encoding="utf-8") as stream:
        stream.write(f"GEMINI_API_KEY={secret}\n")

    result = runner.invoke(app, ["config-check", "--env-file", str(env_file)])

    assert result.exit_code == 0
    assert "GEMINI_API_KEY: gesetzt" in result.output
    assert "OPENAI_API_KEY: fehlt" in result.output
    for fragment in (secret, "PREFIX_super", "value_SUFFIX"):
        assert fragment not in result.output


def test_explicit_relative_dotenv_path_works(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "root"
    pdf_dir = tmp_path / "pdfs"
    for directory in (
        data_root,
        pdf_dir,
        *(data_root / name for name in ("runs", "cache", "exports", "logs")),
    ):
        directory.mkdir(parents=True, exist_ok=True)
    _write_env(tmp_path / ".env", data_root, pdf_dir)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["config-check", "--env-file", ".env"])
    assert result.exit_code == 0
    assert "Lokale Grundkonfiguration ist gültig" in result.output


def test_config_check_detects_missing_worker_id(tmp_path: Path) -> None:
    env_file = tmp_path / "missing-worker.env"
    env_file.write_text(f"AISURGEON_DATA_ROOT={tmp_path}\n", encoding="utf-8")
    result = runner.invoke(app, ["config-check", "--env-file", str(env_file)])
    assert result.exit_code == 2
    assert "Erforderliche lokale Konfiguration fehlt" in result.output


def test_config_check_detects_missing_pdf_directory(tmp_path: Path) -> None:
    data_root = tmp_path / "root"
    data_root.mkdir()
    for name in ("runs", "cache", "exports", "logs"):
        (data_root / name).mkdir()
    env_file = tmp_path / "missing-pdf.env"
    _write_env(env_file, data_root, tmp_path / "absent")
    result = runner.invoke(app, ["config-check", "--env-file", str(env_file)])
    assert result.exit_code == 3
    assert "Mindestens ein lokaler Pfad ist ungültig" in result.output


def test_config_check_detects_access_failure(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "root"
    pdf_dir = tmp_path / "pdfs"
    output_dirs = tuple(data_root / name for name in ("runs", "cache", "exports", "logs"))
    for directory in (data_root, pdf_dir, *output_dirs):
        directory.mkdir(parents=True, exist_ok=True)
    env_file = tmp_path / "access.env"
    _write_env(env_file, data_root, pdf_dir)
    real_access = os.access

    def controlled_access(path: os.PathLike[str] | str, mode: int) -> bool:
        if Path(path) == pdf_dir and mode == os.R_OK:
            return False
        return real_access(path, mode)

    monkeypatch.setattr(os, "access", controlled_access)
    result = runner.invoke(app, ["config-check", "--env-file", str(env_file)])
    assert result.exit_code == 3
    assert "nicht lesbar" in result.output


def test_setup_local_is_idempotent_and_preserves_files(tmp_path: Path) -> None:
    data_root = tmp_path / "root"
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    env_file = tmp_path / "setup.env"
    _write_env(env_file, data_root, pdf_dir)

    first = runner.invoke(app, ["setup-local", "--env-file", str(env_file)])
    sentinel = data_root / "runs" / "keep.txt"
    sentinel.write_text("unchanged", encoding="utf-8")
    original_env = env_file.read_bytes()
    second = runner.invoke(app, ["setup-local", "--env-file", str(env_file)])

    assert first.exit_code == second.exit_code == 0
    assert all((data_root / name).is_dir() for name in ("runs", "cache", "exports", "logs"))
    assert sentinel.read_text(encoding="utf-8") == "unchanged"
    assert env_file.read_bytes() == original_env


def test_create_env_if_missing_is_safe_and_arguments_win(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    env_file = tmp_path / "new-local.env"
    data_root = tmp_path / "argument-root"

    result = runner.invoke(
        app,
        [
            "setup-local",
            "--worker-id",
            "argument-worker",
            "--data-root",
            str(data_root),
            "--pdf-source-dir",
            str(pdf_dir),
            "--env-file",
            str(env_file),
            "--create-env-if-missing",
        ],
    )

    assert result.exit_code == 0
    content = env_file.read_text(encoding="utf-8")
    assert "AISURGEON_WORKER_ID=argument-worker" in content
    assert "GEMINI_API_KEY=\n" in content
    assert "OPENAI_API_KEY=\n" in content
    if os.name == "posix":
        assert env_file.stat().st_mode & 0o077 == 0

    before = env_file.read_bytes()
    again = runner.invoke(
        app,
        [
            "setup-local",
            "--worker-id",
            "other-worker",
            "--data-root",
            str(data_root),
            "--pdf-source-dir",
            str(pdf_dir),
            "--env-file",
            str(env_file),
            "--create-env-if-missing",
        ],
    )
    assert again.exit_code == 0
    assert env_file.read_bytes() == before
    assert "Worker-ID: other-worker" in again.output
