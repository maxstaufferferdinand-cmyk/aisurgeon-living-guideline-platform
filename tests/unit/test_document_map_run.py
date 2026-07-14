"""Dry-run, Git gate, run-ID, and secret-safe audit tests."""

import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import SecretStr
from typer.testing import CliRunner

from aisurgeon.cli.app import app
from aisurgeon.extraction.gemini import document_map as orchestration
from aisurgeon.extraction.gemini.errors import GeminiConfigurationError
from aisurgeon.extraction.gemini.models import (
    DocumentMap,
    GeminiDocumentMapResult,
    RemoteFileMetadata,
)
from aisurgeon.extraction.pdf_registration import register_pdf

runner = CliRunner()


def test_run_id_contains_required_components() -> None:
    run_id = orchestration.make_run_id(
        timestamp=datetime(2026, 7, 14, 12, 30, tzinfo=UTC),
        worker_id="worker-01",
        source_id="pdf-v1-abc",
        pdf_sha256="a" * 64,
    )
    assert re.fullmatch(r"20260714T123000000000Z_worker-01_pdf-v1-abc_aaaaaaaa", run_id)


def test_dry_run_never_constructs_gemini_client(
    tmp_path: Path, synthetic_pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(orchestration, "git_metadata", lambda root: ("a" * 40, "test", True))

    def forbidden_client(**kwargs):
        raise AssertionError("network client must not be constructed")

    manifest, run_dir = orchestration.run_document_map(
        pdf_path=synthetic_pdf,
        worker_id="worker-test",
        output_root=tmp_path / "runs",
        api_key=SecretStr("dummy-key"),
        dry_run=True,
        project_root=Path.cwd(),
        client_factory=forbidden_client,
    )
    assert manifest.status == "dry_run"
    assert (run_dir / "run_manifest.json").is_file()
    assert not (run_dir / "remote_file_metadata.json").exists()


def test_dirty_worktree_gate_blocks_live_run_before_client(
    tmp_path: Path, synthetic_pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(orchestration, "git_metadata", lambda root: ("a" * 40, "test", True))
    with pytest.raises(GeminiConfigurationError, match="Dirty Worktree"):
        orchestration.run_document_map(
            pdf_path=synthetic_pdf,
            worker_id="worker-test",
            output_root=tmp_path / "runs",
            api_key=SecretStr("dummy-key"),
            dry_run=False,
            project_root=Path.cwd(),
            client_factory=lambda **kwargs: pytest.fail("client constructed"),
        )


def test_cli_dry_run_uses_only_explicit_temporary_env(
    tmp_path: Path, synthetic_pdf: Path
) -> None:
    env_file = tmp_path / "isolated.env"
    output_root = tmp_path / "runs"
    env_file.write_text(
        "AISURGEON_WORKER_ID=dry-run-worker\nGEMINI_API_KEY=dummy-only\n",
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        [
            "gemini-document-map",
            "--pdf",
            str(synthetic_pdf),
            "--env-file",
            str(env_file),
            "--output-root",
            str(output_root),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert "kein Upload und kein Gemini-API-Aufruf" in result.output
    assert "dummy-only" not in result.output


class SuccessfulGateway:
    def __init__(self, **kwargs) -> None:
        self.last_remote_metadata = RemoteFileMetadata(
            status="ACTIVE", remote_file_deleted=True
        )

    def create_document_map(self, *, source_id: str, **kwargs) -> GeminiDocumentMapResult:
        value = {
            "schema_version": "document_map_v1",
            "source_id": source_id,
            "document_title": None,
            "issuing_organization": None,
            "guideline_identifier": None,
            "guideline_class": None,
            "language": None,
            "publication_year": None,
            "version_text": None,
            "validity_status": None,
            "declared_page_count": 2,
            "detected_document_layout": "synthetic",
            "column_layout": "single",
            "recurring_header_footer_description": None,
            "front_matter_page_ranges": [],
            "table_of_contents_page_ranges": [],
            "clinical_main_body_page_ranges": [{"page_start": 1, "page_end": 2}],
            "bibliography_page_ranges": [],
            "appendix_page_ranges": [],
            "recommendation_or_statement_patterns": [],
            "comment_or_rationale_patterns": [],
            "native_grading_systems": [],
            "detected_formal_item_types": ["synthetic type"],
            "detected_table_inventory": [],
            "detected_algorithm_inventory": [],
            "detected_decision_tree_inventory": [],
            "uncertain_regions": [],
            "warnings": [],
        }
        document_map = DocumentMap.model_validate(value)
        return GeminiDocumentMapResult(
            document_map=document_map,
            raw_json=json.dumps(value),
            remote_file_metadata=self.last_remote_metadata,
            token_usage={"total_tokens": 10},
        )


class PageCountMismatchGateway(SuccessfulGateway):
    def create_document_map(self, *, source_id: str, **kwargs) -> GeminiDocumentMapResult:
        result = super().create_document_map(source_id=source_id, **kwargs)
        result.document_map.declared_page_count = 99
        result.raw_json = result.document_map.model_dump_json()
        return result


def test_live_audit_outputs_never_contain_api_key(
    tmp_path: Path, synthetic_pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "PREFIX_ultra_dummy_secret_SUFFIX"
    monkeypatch.setattr(orchestration, "git_metadata", lambda root: ("a" * 40, "test", False))
    manifest, run_dir = orchestration.run_document_map(
        pdf_path=synthetic_pdf,
        worker_id="worker-test",
        output_root=tmp_path / "runs",
        api_key=SecretStr(secret),
        dry_run=False,
        project_root=Path.cwd(),
        client_factory=SuccessfulGateway,
    )
    assert manifest.status == "succeeded"
    combined = "\n".join(
        path.read_text(encoding="utf-8") for path in run_dir.rglob("*") if path.is_file()
    )
    for fragment in (secret, "PREFIX_ultra", "secret_SUFFIX"):
        assert fragment not in combined


def test_page_count_warning_does_not_set_validation_failed_status(
    tmp_path: Path, synthetic_pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(orchestration, "git_metadata", lambda root: ("a" * 40, "test", False))
    manifest, run_dir = orchestration.run_document_map(
        pdf_path=synthetic_pdf,
        worker_id="worker-test",
        output_root=tmp_path / "runs",
        api_key=SecretStr("dummy-key"),
        dry_run=False,
        project_root=Path.cwd(),
        client_factory=PageCountMismatchGateway,
    )
    report = json.loads((run_dir / "validation_report.json").read_text(encoding="utf-8"))
    assert manifest.status == "succeeded"
    assert report["valid"] is True
    assert report["review_required"] is True
    assert any(issue["code"] == "page_count_mismatch" for issue in report["issues"])


def test_pdf_register_writes_outside_repository(tmp_path: Path, synthetic_pdf: Path) -> None:
    env_file = tmp_path / "register.env"
    env_file.write_text("AISURGEON_WORKER_ID=register-worker\n", encoding="utf-8")
    output_dir = tmp_path / "registration"
    result = runner.invoke(
        app,
        [
            "pdf-register",
            "--pdf",
            str(synthetic_pdf),
            "--env-file",
            str(env_file),
            "--output-dir",
            str(output_dir),
        ],
    )
    assert result.exit_code == 0
    written = json.loads((output_dir / "pdf_registration.json").read_text(encoding="utf-8"))
    assert written["sha256"] == register_pdf(
        synthetic_pdf, worker_id="register-worker"
    ).sha256
