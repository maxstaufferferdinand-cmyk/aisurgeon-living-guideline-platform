"""Isolated canonical extraction CLI dry-run tests."""

import json
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
    plans = list((tmp_path / "runs").rglob("extraction_plan.json"))
    assert len(plans) == 1
    plan = json.loads(plans[0].read_text(encoding="utf-8"))
    assert plan["document_map_schema_version"] == "document_map_v1"
    assert plan["canonical_extraction_schema_version"] == "canonical_extraction_v2"
    assert plan["document_map_prompt_version"] == "gemini_document_map_v1"
    assert plan["formal_items_prompt_version"] == "gemini_formal_items_comments_v2"
    assert "SYNTHETIC-TWO-COLUMN" in plans[0].parent.name
    assert "gemini_formal_items_comments_v2" in plans[0].parent.name
