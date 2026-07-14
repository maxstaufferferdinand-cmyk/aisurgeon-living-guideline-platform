"""Isolated canonical extraction CLI dry-run tests."""

from pathlib import Path

from typer.testing import CliRunner

from aisurgeon.cli.app import app

runner = CliRunner()


def test_extract_guideline_dry_run_has_no_network(tmp_path: Path, synthetic_pdf: Path) -> None:
    env_file = tmp_path / "isolated.env"
    env_file.write_text(
        "AISURGEON_WORKER_ID=synthetic-worker\nGEMINI_API_KEY=dummy-never-used\n",
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        [
            "extract-guideline",
            "--pdf",
            str(synthetic_pdf),
            "--source-id",
            "SYNTHETIC-TWO-COLUMN",
            "--output-root",
            str(tmp_path / "runs"),
            "--env-file",
            str(env_file),
            "--pages-per-job",
            "1",
            "--overlap-pages",
            "1",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert "kein Upload und kein Gemini-API-Aufruf" in result.output
    assert "dummy-never-used" not in result.output
    assert list((tmp_path / "runs").rglob("extraction_plan.json"))
