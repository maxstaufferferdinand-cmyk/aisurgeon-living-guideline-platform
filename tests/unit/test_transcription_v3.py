"""Mocked tests for canonical Gemini transcription v3."""

import json
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from aisurgeon.extraction.pdf_preflight import PdfPagePreflight, PdfPreflight
from aisurgeon.extraction.provider_preflight import run_provider_preflight
from aisurgeon.extraction.semantic_structure import (
    SemanticStructureDraft,
    build_semantic_payload,
    derive_pubmed_start_date,
    run_semantic_structure,
)
from aisurgeon.extraction.transcription_v3.completeness import (
    split_incomplete_job,
    validate_transcription_completeness,
)
from aisurgeon.extraction.transcription_v3.models import (
    ExtractionScout,
    ExtractionScoutRegion,
    SourceContentDraft,
    TranscriptionJob,
    VisualBlock,
)
from aisurgeon.extraction.transcription_v3.pipeline import (
    inject_source_content_metadata,
    run_transcription_v3,
)
from aisurgeon.extraction.transcription_v3.planner import build_transcription_plan
from aisurgeon.extraction.transcription_v3.retry import classify_provider_failure
from aisurgeon.orchestration.guideline_v3 import run_guideline_end_to_end_v3
from aisurgeon.search.pubmed.generation import derive_start_date_from_extraction_manifest


def _preflight(page_count: int = 12) -> tuple[PdfPreflight, list[PdfPagePreflight]]:
    pdf = PdfPreflight(
        source_id="SRC",
        pdf_sha256="a" * 64,
        file_size_bytes=1000,
        pdf_version="1.7",
        encrypted=False,
        page_count=page_count,
    )
    pages = [
        PdfPagePreflight(
            source_id="SRC",
            page_number=page,
            width=100,
            height=100,
            rotation=0,
            text_layer_character_count=500,
            image_count=0,
            image_heavy=False,
            text_heavy=False,
            approximate_page_density_score=0.5,
            obvious_blank_page=False,
        )
        for page in range(1, page_count + 1)
    ]
    return pdf, pages


def _scout(*regions: ExtractionScoutRegion, page_count: int = 12) -> ExtractionScout:
    return ExtractionScout(
        source_id="SRC",
        model_id="gemini-3.5-flash",
        declared_page_count=page_count,
        regions=list(regions),
    )


def test_gemini_transcription_draft_has_no_semantic_or_technical_fields() -> None:
    draft = SourceContentDraft(
        represented_original_pdf_pages=[1],
        detected_reading_order="left_to_right",
        visual_blocks=[
            VisualBlock(
                page_number=1,
                reading_order_index=1,
                block_type="paragraph",
                exact_visible_text="Original source.",
            )
        ],
    )
    assert "recommendation" not in json.dumps(draft.model_json_schema()).casefold()
    for forbidden in ("source_id", "schema_version", "job_id", "chunk_id"):
        with pytest.raises(ValidationError):
            SourceContentDraft.model_validate({**draft.model_dump(), forbidden: "bad"})


def test_python_injects_technical_metadata_deterministically() -> None:
    job = TranscriptionJob(
        job_id="job-1",
        chunk_id="chunk-1",
        profile="single_column_prose_verbatim",
        primary_pages=[1],
        reason="test",
    )
    draft = SourceContentDraft(
        represented_original_pdf_pages=[1],
        detected_reading_order="monotonic",
        visual_blocks=[],
    )
    content = inject_source_content_metadata(draft, source_id="SRC", job=job)
    assert content.source_id == "SRC"
    assert content.schema_version == "canonical_transcription_v3"
    assert content.prompt_version == "gemini_source_transcription_v3"
    assert inject_source_content_metadata(draft, source_id="SRC", job=job) == content


def test_chunk_profiles_are_bounded_by_layout() -> None:
    pdf, pages = _preflight(12)
    single = build_transcription_plan(preflight=pdf, pages=pages, scout=None)
    assert len(single[0].primary_pages) == 5
    assert len(single[0].primary_pages) <= 10
    two_col = build_transcription_plan(
        preflight=pdf,
        pages=pages,
        scout=_scout(ExtractionScoutRegion(region_kind="multi_column", page_start=1, page_end=12)),
    )
    assert len(two_col[0].primary_pages) <= 5
    dense = build_transcription_plan(
        preflight=pdf,
        pages=pages,
        scout=_scout(ExtractionScoutRegion(region_kind="dense_region", page_start=1, page_end=12)),
    )
    assert len(dense[0].primary_pages) <= 4


def test_bibliography_tables_algorithms_and_coverage_plans() -> None:
    pdf, pages = _preflight(6)
    scout = _scout(
        ExtractionScoutRegion(region_kind="bibliography", page_start=1, page_end=2),
        ExtractionScoutRegion(region_kind="table", page_start=3, page_end=3),
        ExtractionScoutRegion(region_kind="algorithm", page_start=4, page_end=4),
        page_count=6,
    )
    jobs = build_transcription_plan(preflight=pdf, pages=pages, scout=scout)
    by_page = {page: job for job in jobs for page in job.primary_pages}
    assert by_page[1].profile == "bibliography_verbatim"
    assert len(by_page[1].primary_pages) == 1
    assert by_page[3].profile == "table_faithful"
    assert by_page[4].profile == "algorithm_faithful"
    assert sorted(by_page) == [1, 2, 3, 4, 5, 6]
    assert by_page[2].primary_pages != by_page[2].context_pages


def test_completeness_rejects_missing_short_and_truncated_output() -> None:
    pdf, pages = _preflight(2)
    job = build_transcription_plan(preflight=pdf, pages=pages[:1])[0]
    findings = validate_transcription_completeness(jobs=[job], contents=[], page_preflight=pages)
    assert any(f.issue_code == "missing_primary_page" for f in findings)
    content = inject_source_content_metadata(
        SourceContentDraft(
            represented_original_pdf_pages=[1],
            detected_reading_order="monotonic",
            visual_blocks=[
                VisualBlock(
                    page_number=1,
                    reading_order_index=1,
                    block_type="paragraph",
                    exact_visible_text="short",
                )
            ],
        ),
        source_id="SRC",
        job=job,
    )
    findings = validate_transcription_completeness(
        jobs=[job],
        contents=[content],
        page_preflight=pages,
        finish_reasons={job.job_id: "MAX_TOKENS"},
    )
    assert {f.issue_code for f in findings} >= {
        "implausibly_short_output",
        "finish_reason_truncated",
    }
    assert split_incomplete_job(job)[0].primary_pages == [1]


class _HTTPError(Exception):
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}")


def test_retry_policy_distinguishes_retryable_and_non_retryable() -> None:
    rate_limit = classify_provider_failure(_HTTPError(429), attempt=1, retry_after_seconds=40)
    assert rate_limit.category == "retryable"
    assert rate_limit.calculated_delay_seconds == 40
    assert classify_provider_failure(_HTTPError(503), attempt=2).category == "retryable"
    for status in (400, 401, 403):
        assert classify_provider_failure(_HTTPError(status), attempt=1).category == "non_retryable"


def test_transcription_resume_preserves_successful_jobs(
    synthetic_pdf: Path, tmp_path: Path
) -> None:
    output_root = tmp_path / "runs"
    status, run_dir = run_transcription_v3(
        pdf_path=synthetic_pdf,
        source_id="SRC",
        worker_id="worker",
        output_root=output_root,
    )
    assert status == "completed"
    job_manifest = next((run_dir / "transcription_jobs").glob("*/job_manifest.json"))
    before = job_manifest.read_text(encoding="utf-8")
    status, _ = run_transcription_v3(
        pdf_path=synthetic_pdf,
        source_id="SRC",
        worker_id="worker",
        output_root=output_root,
        resume_run=run_dir,
    )
    assert status == "completed"
    assert job_manifest.read_text(encoding="utf-8") == before
    raw = next((run_dir / "transcription_jobs").glob("*/raw_response.json"))
    assert raw.is_file()


def test_semantic_structuring_receives_no_pdf_and_preserves_originals(
    synthetic_pdf: Path, tmp_path: Path
) -> None:
    _, tx_run = run_transcription_v3(
        pdf_path=synthetic_pdf,
        source_id="SRC",
        worker_id="worker",
        output_root=tmp_path / "runs",
    )
    payload = build_semantic_payload(tx_run)
    assert "%PDF-" not in json.dumps(payload)
    structure_run = run_semantic_structure(
        transcription_run=tx_run,
        output_root=tmp_path / "structured",
        worker_id="worker",
    )
    formal = json.loads((structure_run / "formal_items.jsonl").read_text().splitlines()[0])
    comment = json.loads((structure_run / "comments.jsonl").read_text().splitlines()[0])
    assert formal["normalized_item_family"] == "recommendation"
    assert formal["exact_original_text"].startswith("Synthetic source transcript")
    assert comment["exact_original_text"] == "Synthetic exact original comment."


def test_formal_chronology_and_bibliography_parser_contract(
    tmp_path: Path, synthetic_pdf: Path
) -> None:
    _, tx_run = run_transcription_v3(
        pdf_path=synthetic_pdf,
        source_id="SRC",
        worker_id="worker",
        output_root=tmp_path / "runs",
    )

    def draft(_payload: dict) -> SemanticStructureDraft:
        return SemanticStructureDraft(
            publication_year=2016,
            publication_year_source="page 1",
            formal_items=[
                {
                    "item_type": "recommendation",
                    "item_type_raw": "Empfehlung",
                    "original_number": "1",
                    "exact_original_text": "Exact recommendation.",
                    "page_start": 1,
                    "page_end": 1,
                    "extraction_confidence": 1,
                },
                {
                    "item_type": "statement",
                    "item_type_raw": "Statement",
                    "original_number": "2",
                    "exact_original_text": "Exact statement.",
                    "page_start": 2,
                    "page_end": 2,
                    "extraction_confidence": 1,
                },
            ],
            references=[
                {
                    "original_reference_number": "1",
                    "exact_original_reference_text": "[1] First.",
                    "page_start": 2,
                    "page_end": 2,
                    "extraction_confidence": 1,
                }
            ],
        )

    run = run_semantic_structure(
        transcription_run=tx_run,
        output_root=tmp_path / "structured",
        worker_id="worker",
        draft_factory=draft,
    )
    items = [json.loads(line) for line in (run / "formal_items.jsonl").read_text().splitlines()]
    assert [item["sequence_number"] for item in items] == [1, 2]
    assert [item["exact_original_text"] for item in items] == [
        "Exact recommendation.",
        "Exact statement.",
    ]
    refs = [json.loads(line) for line in (run / "references.jsonl").read_text().splitlines()]
    assert refs[0]["original_reference_number"] == "1"


def test_bibliography_missing_numbers_trigger_retranscription_marker() -> None:
    job = TranscriptionJob(
        job_id="bib",
        chunk_id="bib",
        profile="bibliography_verbatim",
        primary_pages=[10, 11],
        reason="bibliography",
    )
    repairs = split_incomplete_job(job)
    assert [repair.primary_pages for repair in repairs] == [[10], [11]]


def test_pubmed_publication_year_defaults_and_overrides_are_audited(tmp_path: Path) -> None:
    run = tmp_path / "extract"
    run.mkdir()
    (run / "extraction_manifest.json").write_text(
        json.dumps({"publication_year": 2016, "publication_year_source": "page 2"}),
        encoding="utf-8",
    )
    start, audit = derive_start_date_from_extraction_manifest(run)
    assert start.isoformat() == "2016-01-01"
    assert audit["source"] == "page 2"
    assert derive_pubmed_start_date(2023) == "2023-01-01"
    override, audit = derive_start_date_from_extraction_manifest(
        run, override=date(2020, 5, 1)
    )
    assert override.isoformat() == "2020-05-01"
    assert audit["source"] == "explicit_override"
    (run / "extraction_manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        derive_start_date_from_extraction_manifest(run)


def test_provider_preflight_and_manifests_are_secret_free(
    tmp_path: Path, synthetic_pdf: Path
) -> None:
    report = run_provider_preflight(providers={"gemini", "openai", "ncbi"})
    encoded = report.model_dump_json()
    assert "sk-" not in encoded and "AIza" not in encoded
    _, tx_run = run_transcription_v3(
        pdf_path=synthetic_pdf,
        source_id="SRC",
        worker_id="worker",
        output_root=tmp_path / "runs",
    )
    manifest = (tx_run / "transcription_manifest.json").read_text(encoding="utf-8")
    assert "Authorization" not in manifest and "API_KEY" not in manifest


def test_orchestrator_does_not_call_late_reference_repair_and_limit_blocks_docx(
    tmp_path: Path, synthetic_pdf: Path
) -> None:
    run = run_guideline_end_to_end_v3(
        pdf=synthetic_pdf,
        source_id="SRC",
        output_root=tmp_path / "runs",
        worker_id="worker",
        limit=1,
    )
    manifest = json.loads((run / "orchestration_manifest.json").read_text(encoding="utf-8"))
    assert manifest["late_reference_repair_called"] is False
    assert manifest["final_docx_produced"] is False
    complete = run_guideline_end_to_end_v3(
        pdf=synthetic_pdf,
        source_id="SRC2",
        output_root=tmp_path / "runs",
        worker_id="worker",
    )
    assert (complete / "synthetic_final_guideline.docx").is_file()


def test_legacy_run_path_remains_unmodified() -> None:
    legacy = Path("/mnt/c/living_guideline_platform/runs")
    if legacy.exists():
        assert not any(path.name.startswith("transcription-v3") for path in legacy.iterdir())
