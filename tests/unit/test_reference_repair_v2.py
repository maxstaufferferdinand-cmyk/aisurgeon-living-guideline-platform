import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar

import pytest
from pydantic import SecretStr
from pypdf import PdfReader, PdfWriter

from aisurgeon.search.pubmed.query import sha256_text
from aisurgeon.synthesis.reference_repair import (
    OriginalReferenceRepairV2PageJobOutput,
    bibliography_page_plan_v2,
    load_reference_repair_v2_model_config,
    page_job_fingerprint_v2,
    reject_bulk_partial_reference_response,
    reject_document_map_versions_in_repair_fingerprint,
    run_reference_repair_v2_and_rebuild,
    stitch_page_job_references_v2,
    validate_complete_reference_sequence_v2,
    validate_page_job_output_v2,
    write_pdf_slice,
)

EN_DASH = "\u2013"


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _pdf(path: Path, pages: int = 72) -> None:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=72, height=72)
    with path.open("wb") as stream:
        writer.write(stream)


def _article(pmid: str) -> dict:
    return {
        "schema_version": "pubmed_fetch_v1",
        "pmid": pmid,
        "doi": None,
        "title": f"New Trial {pmid}",
        "abstract": "Abstract",
        "authors": ["A Autor"],
        "journal": "Journal",
        "publication_year": 2025,
        "publication_types": ["Randomized Controlled Trial"],
        "mesh_terms": ["Humans"],
        "has_abstract": True,
    }


def _block() -> dict:
    return {
        "source_id": "SRC",
        "formal_item_id": "F1",
        "sequence_number": 1,
        "original_item_number": "1",
        "source_native_item_type": "Empfehlung",
        "section_path": ["Kapitel"],
        "exact_original_item_text": "Originaltext [1].",
        "exact_original_comments": [f"Kommentar mit [46{EN_DASH}64]."],
        "new_evidence_de": "Neue Evidenz PMID 35324483.",
        "aisurgeon_evidence_class": "C",
        "conclusion_de": "Schlussfolgerung PMID 35324483.",
        "decision": "rationale_updated",
        "updated_item_text_de": "Originaltext.",
        "used_direct_pmids": ["35324483"],
        "used_indirect_pmids": [],
        "used_context_pmids": [],
        "old_reference_ids": [],
        "new_reference_ids": [],
        "review_required": False,
        "review_notes": [],
    }


def _runs(tmp_path: Path, *, bibliography_start: int = 1, bibliography_end: int = 20):
    base = tmp_path / "input"
    extraction = base / "extract"
    search = base / "search"
    fetch = base / "fetch"
    mapping = base / "mapping"
    synthesis = base / "synthesis"
    failed = base / "failed-reference"
    for path in (extraction, search, fetch, mapping, synthesis, failed):
        path.mkdir(parents=True)
    pdf = base / "source.pdf"
    _pdf(pdf, pages=max(72, bibliography_end))
    _json(
        extraction / "document_map.validated.json",
        {
            "source_id": "SRC",
            "bibliography_page_ranges": [
                {"page_start": bibliography_start, "page_end": bibliography_end}
            ],
        },
    )
    _json(extraction / "extraction_manifest.json", {"source_id": "SRC", "status": "completed"})
    _jsonl(extraction / "references.jsonl", [])
    _jsonl(fetch / "pubmed_articles.jsonl", [_article("35324483")])
    _json(fetch / "pubmed_fetch_manifest.json", {"source_id": "SRC", "status": "completed"})
    _jsonl(synthesis / "updated_guideline_blocks.jsonl", [_block()])
    _json(synthesis / "synthesis_summary.json", {"processed_formal_items": 1})
    _json(
        synthesis / "synthesis_manifest.json",
        {
            "source_id": "SRC",
            "status": "completed_with_review",
            "input_runs": [str(extraction), str(search), str(fetch), str(mapping)],
        },
    )
    _json(failed / "reference_rebuild_manifest.json", {"source_id": "SRC", "status": "failed"})
    return pdf, extraction, synthesis, failed


def _page_output(page: int, numbers: list[int], *, complete: bool = True) -> dict:
    return {
        "schema_version": "original_reference_repair_v2",
        "source_id": "SRC",
        "job_id": f"page_{page:04d}",
        "primary_original_pdf_page": page,
        "context_original_pdf_pages": [],
        "first_reference_number_observed": min(numbers) if numbers else None,
        "last_reference_number_observed": max(numbers) if numbers else None,
        "observed_reference_numbers": numbers,
        "references": [
            {
                "original_reference_number": number,
                "exact_reference_text": f"Reference {number}. Journal 2020; 1: 1-2.",
                "page_start": page,
                "page_end": page,
                "column_start": "left" if index < len(numbers) / 2 else "right",
                "column_end": "left" if index < len(numbers) / 2 else "right",
                "continuation_from_previous_page": False,
                "continuation_to_next_page": False,
                "extraction_confidence": 0.99,
                "review_required": False,
                "review_notes": [],
            }
            for index, number in enumerate(numbers)
        ],
        "page_complete": complete,
        "review_required": False,
        "review_notes": [],
    }


class FakeV2Client:
    calls: ClassVar[list[Path]] = []
    skip_number: ClassVar[int | None] = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def request_page_job(self, *, slice_pdf: Path, prompt: str, source_id: str):
        self.calls.append(slice_pdf)
        assert len(PdfReader(str(slice_pdf)).pages) <= 3
        primary = int(prompt.split("primary_original_pdf_page: ", 1)[1].splitlines()[0])
        start = (primary - 1) * 36 + 1
        numbers = [n for n in range(start, start + 36) if n != self.skip_number]
        payload = _page_output(primary, numbers)
        output = OriginalReferenceRepairV2PageJobOutput.model_validate(payload)
        return output, payload, {
            "finish_reason": "STOP",
            "usage": {"total_tokens": 1},
            "response_candidate_count": 1,
            "response_size": len(json.dumps(payload)),
            "attempt_count": 1,
        }


def test_bulk_20_page_response_with_25_references_is_rejected() -> None:
    with pytest.raises(ValueError, match="25 of 720"):
        reject_bulk_partial_reference_response({"references": [{} for _ in range(25)]})


def test_page_plan_creates_20_jobs_and_physical_slices(tmp_path: Path) -> None:
    pdf = tmp_path / "source.pdf"
    _pdf(pdf, pages=72)
    plan = bibliography_page_plan_v2(
        document_map={"bibliography_page_ranges": [{"page_start": 52, "page_end": 71}]},
        pdf_page_count=72,
        pages_per_job=1,
        context_pages=1,
    )
    assert plan["job_count"] == 20
    assert plan["page_conversion_rule"] == "document_map_pages_are_one_based_physical_pdf_pages"
    first = plan["jobs"][0]
    assert first["primary_original_pdf_page"] == 52
    assert [row["role"] for row in first["slice_page_map"]] == ["primary", "next_context"]
    slice_pdf = tmp_path / "slice.pdf"
    write_pdf_slice(pdf, plan["jobs"][8]["slice_page_map"], slice_pdf)
    assert len(PdfReader(str(slice_pdf)).pages) == 3


def test_page_job_validation_flags_gaps_and_incomplete_pages() -> None:
    output = OriginalReferenceRepairV2PageJobOutput.model_validate(_page_output(54, [46, 48]))
    status, findings = validate_page_job_output_v2(output)
    assert status == "completed_with_review"
    assert any(row["issue_code"] == "visible_sequence_gap" for row in findings)
    incomplete = OriginalReferenceRepairV2PageJobOutput.model_validate(
        _page_output(54, [46], complete=False)
    )
    status, findings = validate_page_job_output_v2(incomplete)
    assert status == "failed"
    assert any(row["issue_code"] == "page_incomplete" for row in findings)


def test_stitching_continuation_and_sequence_gate() -> None:
    left = _page_output(54, [61])
    right = _page_output(55, [61, 62])
    left["references"][0]["exact_reference_text"] = "Reference 61 starts"
    left["references"][0]["continuation_to_next_page"] = True
    right["references"][0]["exact_reference_text"] = "and ends."
    right["references"][0]["continuation_from_previous_page"] = True
    stitched, findings = stitch_page_job_references_v2(
        [
            OriginalReferenceRepairV2PageJobOutput.model_validate(left),
            OriginalReferenceRepairV2PageJobOutput.model_validate(right),
        ]
    )
    assert not findings
    by_number = {row["original_reference_number"]: row for row in stitched}
    assert by_number["61"]["exact_reference_text"] == "Reference 61 starts and ends."
    assert validate_complete_reference_sequence_v2(stitched)["complete"] is False


def test_720_complete_allows_rebuild_and_719_blocks_docx(tmp_path: Path) -> None:
    pdf, extraction, synthesis, failed = _runs(tmp_path, bibliography_start=1, bibliography_end=20)
    FakeV2Client.calls = []
    FakeV2Client.skip_number = None
    run = run_reference_repair_v2_and_rebuild(
        pdf=pdf,
        extraction_run=extraction,
        synthesis_run=synthesis,
        failed_reference_run=failed,
        output_root=tmp_path / "out",
        api_key=SecretStr("secret-never-output"),
        client_factory=FakeV2Client,
        now=lambda: datetime(2026, 7, 19, tzinfo=UTC),
    )
    manifest = json.loads((run / "reference_repair_v2_manifest.json").read_text())
    assert manifest["status"] in {"completed", "completed_with_review"}
    assert manifest["page_job_count"] == 20
    assert len(FakeV2Client.calls) == 20
    rebuild = Path(manifest["reference_rebuild_run"])
    docx = rebuild / "AISurgeon_Aktualisierte_Leitlinie_GERD_EoE_2026_references_repaired_v2.docx"
    assert docx.is_file()
    with zipfile.ZipFile(docx) as archive:
        xml = archive.read("word/document.xml").decode("utf-8")
    assert "[N1]" in xml
    assert "PMID 35324483" not in xml.split("Literaturverzeichnis", 1)[0]

    pdf2, extraction2, synthesis2, failed2 = _runs(
        tmp_path / "second", bibliography_start=1, bibliography_end=20
    )
    FakeV2Client.skip_number = 720
    with pytest.raises(RuntimeError, match="Reference repair v2 failed"):
        run_reference_repair_v2_and_rebuild(
            pdf=pdf2,
            extraction_run=extraction2,
            synthesis_run=synthesis2,
            failed_reference_run=failed2,
            output_root=tmp_path / "out2",
            api_key=SecretStr("secret-never-output"),
            client_factory=FakeV2Client,
            now=lambda: datetime(2026, 7, 19, tzinfo=UTC),
        )
    failed_manifest_path = next(
        (tmp_path / "out2").glob("reference-repair-v2-*/reference_repair_v2_manifest.json")
    )
    failed_manifest = json.loads(failed_manifest_path.read_text())
    assert failed_manifest["status"] == "failed"
    assert failed_manifest["final_integrity"]["missing_reference_numbers"] == [720]
    assert not list(failed_manifest_path.parent.glob("**/*references_repaired_v2.docx"))


def test_fingerprint_versions_are_dedicated_and_deterministic(tmp_path: Path) -> None:
    config = load_reference_repair_v2_model_config()
    assert "document_map" not in json.dumps(config)
    with pytest.raises(ValueError, match="Document-map"):
        reject_document_map_versions_in_repair_fingerprint(
            {"prompt_version": "gemini_document_map_v1"}
        )
    job = {
        "primary_original_pdf_page": 52,
        "context_original_pdf_pages": [53],
        "slice_page_map": [
            {"slice_page_index": 1, "original_pdf_page_number": 52, "role": "primary"}
        ],
    }
    fingerprint_1 = page_job_fingerprint_v2(
        source_id="SRC",
        pdf_hash="pdf",
        job=job,
        slice_pdf_hash="slice",
        model_config=config,
        prompt_hash=sha256_text("prompt"),
    )
    fingerprint_2 = page_job_fingerprint_v2(
        source_id="SRC",
        pdf_hash="pdf",
        job=job,
        slice_pdf_hash="slice",
        model_config=config,
        prompt_hash=sha256_text("prompt"),
    )
    assert fingerprint_1 == fingerprint_2


def test_resume_reuses_compatible_page_checkpoints(tmp_path: Path) -> None:
    pdf, extraction, synthesis, failed = _runs(tmp_path, bibliography_start=1, bibliography_end=20)
    FakeV2Client.calls = []
    FakeV2Client.skip_number = None
    run = run_reference_repair_v2_and_rebuild(
        pdf=pdf,
        extraction_run=extraction,
        synthesis_run=synthesis,
        failed_reference_run=failed,
        output_root=tmp_path / "out",
        api_key=SecretStr("secret"),
        client_factory=FakeV2Client,
        now=lambda: datetime(2026, 7, 19, tzinfo=UTC),
    )
    FakeV2Client.calls = []
    resumed = run_reference_repair_v2_and_rebuild(
        pdf=pdf,
        extraction_run=extraction,
        synthesis_run=synthesis,
        failed_reference_run=failed,
        output_root=tmp_path / "out",
        api_key=SecretStr("secret"),
        resume_run=run,
        client_factory=FakeV2Client,
        now=lambda: datetime(2026, 7, 19, tzinfo=UTC),
    )
    assert resumed == run
    assert FakeV2Client.calls == []
    fingerprint = json.loads((run / "checkpoint_fingerprint.json").read_text())
    assert "document_map_v1" not in json.dumps(fingerprint)
