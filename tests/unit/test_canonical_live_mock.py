"""Fully mocked one-upload canonical orchestration test."""

import json
from pathlib import Path
from types import SimpleNamespace

from pydantic import SecretStr

from aisurgeon.extraction.canonical import pipeline
from aisurgeon.extraction.canonical.models import (
    ExtractionBatch,
    VisualObjectBatch,
)
from aisurgeon.extraction.gemini.models import DocumentMap, RemoteFileMetadata


class FakeGateway:
    uploads = 0
    deletes = 0

    def __init__(self, **kwargs) -> None:
        self.last_remote_metadata = RemoteFileMetadata(
            status="not_uploaded", remote_file_deleted=False
        )

    def upload_pdf(self, pdf_path: Path):
        type(self).uploads += 1
        return SimpleNamespace(name="files/fake", uri="mock://fake")

    def request_structured(self, *, model, prompt: str, **kwargs):
        if model is DocumentMap:
            value = DocumentMap(
                schema_version="document_map_v1",
                source_id="SOURCE",
                declared_page_count=2,
                detected_document_layout="two-column synthetic",
                column_layout="two-column",
                clinical_main_body_page_ranges=[{"page_start": 1, "page_end": 2}],
                detected_formal_item_types=["Empfehlung"],
            )
        elif model is ExtractionBatch:
            value = ExtractionBatch(
                formal_items=[
                    {
                        "source_id": "SOURCE",
                        "extraction_batch_id": "clinical-0001-0002",
                        "item_type": "recommendation",
                        "original_number": "1.1",
                        "exact_original_text": "Exakter synthetischer Originaltext.",
                        "page_start": 1,
                        "page_end": 1,
                        "extraction_confidence": 1,
                    }
                ]
            )
        elif model is VisualObjectBatch:
            value = VisualObjectBatch()
        else:
            raise AssertionError(model)
        return value, value.model_dump_json(), {"total_tokens": 1}

    def delete_remote(self, remote) -> bool:
        type(self).deletes += 1
        return True


def test_live_pipeline_reuses_one_upload_and_writes_outputs(
    tmp_path: Path, synthetic_pdf: Path, monkeypatch
) -> None:
    FakeGateway.uploads = FakeGateway.deletes = 0
    monkeypatch.setattr(pipeline, "git_metadata", lambda root: ("a" * 40, "test", False))
    status, run_dir = pipeline.run_live_extraction(
        pdf_path=synthetic_pdf,
        worker_id="worker",
        source_id="SOURCE",
        output_root=tmp_path / "runs",
        api_key=SecretStr("dummy-never-output"),
        project_root=Path.cwd(),
        client_factory=FakeGateway,
    )
    assert status == "completed"
    assert FakeGateway.uploads == FakeGateway.deletes == 1
    formal = json.loads((run_dir / "formal_items.jsonl").read_text(encoding="utf-8"))
    assert formal["exact_original_text"] == "Exakter synthetischer Originaltext."
    assert (run_dir / "review_findings.xlsx").is_file()
