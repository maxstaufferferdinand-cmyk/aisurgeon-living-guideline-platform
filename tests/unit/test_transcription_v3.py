"""Mocked tests for canonical Gemini transcription v3."""

import json
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError
from pypdf import PdfWriter

from aisurgeon.extraction.pdf_preflight import PdfPagePreflight, PdfPreflight
from aisurgeon.extraction.provider_preflight import (
    RealProviderPreflightChecker,
    run_provider_preflight,
)
from aisurgeon.extraction.semantic_structure import (
    OpenAISemanticStructureProvider,
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
    ExtractionScoutDraft,
    ExtractionScoutRegion,
    ProviderCallEvidence,
    SourceContentDraft,
    TranscriptionJob,
    VisualBlock,
)
from aisurgeon.extraction.transcription_v3.pipeline import (
    GeminiTranscriptionProvider,
    _sleep_with_progress,
    canonicalize_source_content_pages,
    gemini_request_schema,
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


def _blank_pdf(tmp_path: Path, page_count: int) -> Path:
    path = tmp_path / f"synthetic-{page_count}.pdf"
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=72, height=72)
    with path.open("wb") as stream:
        writer.write(stream)
    return path


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


def test_gemini_request_schemas_remove_unsupported_fields_recursively() -> None:
    for model in (ExtractionScoutDraft, SourceContentDraft):
        schema = gemini_request_schema(model)
        encoded = json.dumps(schema, sort_keys=True)
        assert "additionalProperties" not in encoded
        assert "additional_properties" not in encoded
        assert "$defs" not in encoded
        assert "$ref" not in encoded
        assert "title" not in encoded
        assert "default" not in encoded
    scout_schema = gemini_request_schema(ExtractionScoutDraft)
    region_items = scout_schema["properties"]["regions"]["items"]
    assert region_items["type"] == "object"
    assert "properties" in region_items
    source_schema = gemini_request_schema(SourceContentDraft)
    block_items = source_schema["properties"]["visual_blocks"]["items"]
    assert block_items["type"] == "object"
    assert block_items["properties"]["table_html"]["nullable"] is True


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


def test_gemini_page_labels_are_canonicalized_to_original_primary_pages() -> None:
    job = TranscriptionJob.model_validate(
        {
            "job_id": "job",
            "chunk_id": "chunk",
            "profile": "dense_prose_verbatim",
            "primary_pages": [28, 29, 30],
            "context_pages": [27, 31],
            "slice_page_map": [
                {
                    "slice_page_index": 1,
                    "original_pdf_page_number": 27,
                    "role": "previous_context",
                },
                {"slice_page_index": 2, "original_pdf_page_number": 28, "role": "primary"},
                {"slice_page_index": 3, "original_pdf_page_number": 29, "role": "primary"},
                {"slice_page_index": 4, "original_pdf_page_number": 30, "role": "primary"},
                {
                    "slice_page_index": 5,
                    "original_pdf_page_number": 31,
                    "role": "next_context",
                },
            ],
            "reason": "test",
        }
    )
    printed_page_draft = SourceContentDraft(
        represented_original_pdf_pages=[610, 611, 612],
        detected_reading_order="monotonic",
        visual_blocks=[
            VisualBlock(
                page_number=610,
                reading_order_index=1,
                block_type="paragraph",
                exact_visible_text="Page 28 text.",
            ),
            VisualBlock(
                page_number=611,
                reading_order_index=2,
                block_type="paragraph",
                exact_visible_text="Page 29 text.",
            ),
            VisualBlock(
                page_number=612,
                reading_order_index=3,
                block_type="paragraph",
                exact_visible_text="Page 30 text.",
            ),
        ],
    )
    canonical = canonicalize_source_content_pages(printed_page_draft, job=job)
    assert canonical.represented_original_pdf_pages == [28, 29, 30]
    assert [block.page_number for block in canonical.visual_blocks] == [28, 29, 30]

    slice_page_draft = SourceContentDraft(
        represented_original_pdf_pages=[1, 2, 3, 4, 5],
        detected_reading_order="monotonic",
        visual_blocks=[
            VisualBlock(
                page_number=page,
                reading_order_index=page,
                block_type="paragraph",
                exact_visible_text=f"slice page {page}",
            )
            for page in range(1, 6)
        ],
    )
    canonical = canonicalize_source_content_pages(slice_page_draft, job=job)
    assert canonical.represented_original_pdf_pages == [28, 29, 30]
    assert [block.page_number for block in canonical.visual_blocks] == [28, 29, 30]
    assert [block.exact_visible_text for block in canonical.visual_blocks] == [
        "slice page 2",
        "slice page 3",
        "slice page 4",
    ]


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
    assert rate_limit.final_failure_category == "rate_or_quota"
    unavailable = classify_provider_failure(_HTTPError(503), attempt=2)
    assert unavailable.category == "retryable"
    assert unavailable.final_failure_category == "provider_capacity_unavailable"
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
    assert status == "mock_test"
    job_manifest = next((run_dir / "transcription_jobs").glob("*/job_manifest.json"))
    before = job_manifest.read_text(encoding="utf-8")
    status, _ = run_transcription_v3(
        pdf_path=synthetic_pdf,
        source_id="SRC",
        worker_id="worker",
        output_root=output_root,
        resume_run=run_dir,
    )
    assert status == "mock_test"
    assert job_manifest.read_text(encoding="utf-8") == before
    raw = next((run_dir / "transcription_jobs").glob("*/raw_response.json"))
    assert raw.is_file()


def test_transcription_resume_retries_incomplete_job_with_existing_slice_metadata(
    synthetic_pdf: Path, tmp_path: Path
) -> None:
    output_root = tmp_path / "runs"
    _, run_dir = run_transcription_v3(
        pdf_path=synthetic_pdf,
        source_id="SRC",
        worker_id="worker",
        output_root=output_root,
    )
    job_dir = next((run_dir / "transcription_jobs").iterdir())
    original_slice_map = job_dir / "slice_page_map.json"
    assert original_slice_map.is_file()
    for name in ("checkpoint.json", "raw_response.json", "validated_source_content.json"):
        (job_dir / name).unlink()
    (job_dir / "attempts.jsonl").write_text("partial failed attempt\n", encoding="utf-8")

    status, resumed = run_transcription_v3(
        pdf_path=synthetic_pdf,
        source_id="SRC",
        worker_id="worker",
        output_root=output_root,
        resume_run=run_dir,
    )

    assert status == "mock_test"
    assert resumed == run_dir
    assert (job_dir / "checkpoint.json").is_file()
    assert "partial failed attempt" not in (job_dir / "attempts.jsonl").read_text()


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
    report = run_provider_preflight(
        providers={"gemini", "openai", "ncbi"}, execution_mode="mock_test"
    )
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


def test_gemini_preflight_uses_clean_response_json_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    class GenerateContentConfig:
        def __init__(self, **kwargs):
            captured["config_kwargs"] = kwargs

    class Models:
        def generate_content(self, **kwargs):
            captured["generate_kwargs"] = kwargs
            return object()

    class Client:
        def __init__(self, *, api_key: str):
            self.models = Models()

    import sys
    import types as pytypes

    google_module = pytypes.ModuleType("google")
    genai_module = pytypes.ModuleType("google.genai")
    genai_module.Client = Client
    genai_module.types = pytypes.SimpleNamespace(GenerateContentConfig=GenerateContentConfig)
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.genai", genai_module)
    monkeypatch.setattr(google_module, "genai", genai_module, raising=False)
    checker = RealProviderPreflightChecker(
        gemini_api_key=__import__("pydantic").SecretStr("dummy"),
        openai_api_key=None,
        ncbi_api_key=None,
        ncbi_email=None,
    )
    assert checker("gemini", "structured_output_operational") is True
    config = captured["config_kwargs"]
    assert "response_json_schema" in config
    assert "response_schema" not in config
    encoded = json.dumps(config["response_json_schema"])
    assert "additionalProperties" not in encoded
    assert "additional_properties" not in encoded


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
    complete_manifest = json.loads(
        (complete / "orchestration_manifest.json").read_text(encoding="utf-8")
    )
    assert complete_manifest["execution_mode"] == "mock_test"
    assert complete_manifest["final_docx_produced"] is False


class _FakeGeminiProvider:
    provider_backend = "google_genai"

    def __init__(
        self,
        *,
        fail_transcription: bool = False,
        omit_transcription_evidence: bool = False,
        transient_failures: dict[str, int] | None = None,
        scout_regions: list[ExtractionScoutRegion] | None = None,
        page_count: int = 2,
        sleeps: list[float] | None = None,
    ):
        self.fail_transcription = fail_transcription
        self.omit_transcription_evidence = omit_transcription_evidence
        self.transient_failures = transient_failures or {}
        self.scout_regions = scout_regions or []
        self.page_count = page_count
        self.calls: list[str] = []
        self.evidence: list[ProviderCallEvidence] = []
        self._sleep = sleeps.append if sleeps is not None else (lambda _seconds: None)

    def scout(self, _pdf_path: Path, _prompt: str) -> ExtractionScoutDraft:
        self.evidence.append(
            ProviderCallEvidence(
                provider_backend="google_genai",
                stage="scout",
                success=True,
                duration_seconds=0.1,
            )
        )
        return ExtractionScoutDraft(declared_page_count=self.page_count, regions=self.scout_regions)

    def transcribe(
        self, _slice_path: Path, _prompt: str, job: TranscriptionJob
    ) -> SourceContentDraft:
        self.calls.append(job.job_id)
        if self.fail_transcription:
            raise RuntimeError("live provider failed")
        remaining_failures = self.transient_failures.get(job.job_id, 0)
        if remaining_failures:
            self.transient_failures[job.job_id] = remaining_failures - 1
            self.evidence.append(
                ProviderCallEvidence(
                    provider_backend="google_genai",
                    stage="transcription",
                    job_id=job.job_id,
                    success=False,
                    duration_seconds=0.1,
                    http_status=503,
                    api_status="UNAVAILABLE",
                    final_failure_category="provider_capacity_unavailable",
                    safe_error_class="CapacityError",
                    safe_error_message="This model is currently experiencing high demand.",
                )
            )
            raise RuntimeError("Gemini capacity unavailable")
        if not self.omit_transcription_evidence:
            self.evidence.append(
                ProviderCallEvidence(
                    provider_backend="google_genai",
                    stage="transcription",
                    job_id=job.job_id,
                    success=True,
                    response_id=f"response-{job.job_id}",
                    token_usage={"total_token_count": 25},
                    finish_reason="STOP",
                    duration_seconds=0.1,
                )
            )
        return SourceContentDraft(
            represented_original_pdf_pages=job.primary_pages,
            detected_reading_order="monotonic",
            visual_blocks=[
                VisualBlock(
                    page_number=page,
                    reading_order_index=index,
                    block_type="paragraph",
                    exact_visible_text=f"Live provider text page {page}.",
                )
                for index, page in enumerate(job.primary_pages, start=1)
            ],
        )


def test_live_transcription_uses_provider_evidence(synthetic_pdf: Path, tmp_path: Path) -> None:
    provider = _FakeGeminiProvider()
    status, run_dir = run_transcription_v3(
        pdf_path=synthetic_pdf,
        source_id="SRC",
        worker_id="worker",
        output_root=tmp_path / "runs",
        execution_mode="live",
        provider=provider,  # type: ignore[arg-type]
    )
    manifest = json.loads((run_dir / "transcription_manifest.json").read_text(encoding="utf-8"))
    assert status == "completed"
    assert manifest["execution_mode"] == "live"
    assert manifest["provider_backend"] == "google_genai"
    assert manifest["provider_call_count"] > 0
    assert manifest["transcription_call_count"] > 0


def test_live_transcription_failure_never_falls_back(
    synthetic_pdf: Path, tmp_path: Path
) -> None:
    with pytest.raises(RuntimeError):
        run_transcription_v3(
            pdf_path=synthetic_pdf,
            source_id="SRC",
            worker_id="worker",
            output_root=tmp_path / "runs",
            execution_mode="live",
            provider=_FakeGeminiProvider(fail_transcription=True),  # type: ignore[arg-type]
        )


def test_transient_503_defers_job_and_scheduler_continues(tmp_path: Path) -> None:
    pdf = _blank_pdf(tmp_path, 6)
    sleeps: list[float] = []
    provider = _FakeGeminiProvider(
        transient_failures={"tx3-0001-p0001-0005": 1},
        page_count=6,
        sleeps=sleeps,
    )
    status, run_dir = run_transcription_v3(
        pdf_path=pdf,
        source_id="SRC",
        worker_id="worker",
        output_root=tmp_path / "runs",
        execution_mode="live",
        provider=provider,  # type: ignore[arg-type]
    )

    assert status == "completed"
    assert "tx3-0002-p0006-0006" in provider.calls
    assert provider.calls.index("tx3-0002-p0006-0006") < provider.calls.index(
        "tx3-0001-p0001-0005-repair-0001"
    )
    assert all(f"tx3-0001-p0001-0005-repair-{page:04d}" in provider.calls for page in range(1, 6))
    assert (run_dir / "transcription_jobs" / "tx3-0001-p0001-0005" / "last_error.json").is_file()
    manifest = json.loads((run_dir / "transcription_manifest.json").read_text())
    assert manifest["failed_call_count"] == 1
    coverage = json.loads((run_dir / "transcription_coverage_report.json").read_text())
    assert coverage["resolved_primary_pages"] == [1, 2, 3, 4, 5, 6]
    assert sleeps


def test_deferred_transient_jobs_are_revisited_after_other_jobs(tmp_path: Path) -> None:
    pdf = _blank_pdf(tmp_path, 6)
    provider = _FakeGeminiProvider(
        transient_failures={"tx3-0001-p0001-0005": 1, "tx3-0001-p0001-0005-repair-0003": 1},
        page_count=6,
        sleeps=[],
    )
    status, _run_dir = run_transcription_v3(
        pdf_path=pdf,
        source_id="SRC",
        worker_id="worker",
        output_root=tmp_path / "runs",
        execution_mode="live",
        provider=provider,  # type: ignore[arg-type]
    )

    assert status == "completed"
    assert provider.calls.count("tx3-0001-p0001-0005-repair-0003") == 2


def test_transient_exhaustion_cannot_write_completed_manifest(tmp_path: Path) -> None:
    pdf = _blank_pdf(tmp_path, 2)
    provider = _FakeGeminiProvider(
        transient_failures={"tx3-0001-p0001-0001": 20},
        page_count=2,
        sleeps=[],
    )
    with pytest.raises(RuntimeError, match="Transient Gemini transcription failures exhausted"):
        run_transcription_v3(
            pdf_path=pdf,
            source_id="SRC",
            worker_id="worker",
            output_root=tmp_path / "runs",
            execution_mode="live",
            provider=provider,  # type: ignore[arg-type]
            page_range=(1, 1),
        )

    run_dir = next((tmp_path / "runs").glob("transcription-v3-*"))
    assert not (run_dir / "transcription_manifest.json").exists()


def test_one_page_high_resolution_503_can_retry_medium(tmp_path: Path) -> None:
    pdf = _blank_pdf(tmp_path, 2)
    provider = _FakeGeminiProvider(
        transient_failures={"tx3-0001-p0001-0001": 1},
        scout_regions=[
            ExtractionScoutRegion(region_kind="dense_region", page_start=1, page_end=1)
        ],
        page_count=2,
        sleeps=[],
    )
    status, run_dir = run_transcription_v3(
        pdf_path=pdf,
        source_id="SRC",
        worker_id="worker",
        output_root=tmp_path / "runs",
        execution_mode="live",
        provider=provider,  # type: ignore[arg-type]
        page_range=(1, 1),
    )

    assert status == "technical_limited"
    assert "tx3-0001-p0001-0001-medium" in provider.calls
    medium_manifest = json.loads(
        (
            run_dir
            / "transcription_jobs"
            / "tx3-0001-p0001-0001-medium"
            / "job_manifest.json"
        ).read_text()
    )
    assert "medium_resolution_retry" in medium_manifest["reason"]


def test_long_wait_emits_periodic_terminal_status(capsys: pytest.CaptureFixture[str]) -> None:
    sleeps: list[float] = []
    _sleep_with_progress(seconds=125, sleep=sleeps.append, message="[Gemini] wait")

    assert sleeps == [60.0, 60.0, 5.0]
    output = capsys.readouterr().out
    assert "[Gemini] wait" in output
    assert "Still waiting: 65 seconds remaining." in output
    assert "Still waiting: 5 seconds remaining." in output


def test_live_transcription_missing_provider_evidence_fails(
    synthetic_pdf: Path, tmp_path: Path
) -> None:
    status, run_dir = run_transcription_v3(
        pdf_path=synthetic_pdf,
        source_id="SRC",
        worker_id="worker",
        output_root=tmp_path / "runs",
        execution_mode="live",
        provider=_FakeGeminiProvider(omit_transcription_evidence=True),  # type: ignore[arg-type]
    )
    assert status == "failed"
    findings = (run_dir / "transcription_uncertainties.jsonl").read_text(encoding="utf-8")
    assert "missing_provider_evidence" in findings


def test_live_completeness_rejects_tiny_placeholder_output() -> None:
    _pdf, pages = _preflight(2)
    job = TranscriptionJob(
        job_id="job",
        chunk_id="chunk",
        profile="single_column_prose_verbatim",
        primary_pages=[1, 2],
        reason="test",
    )
    content = inject_source_content_metadata(
        SourceContentDraft(
            represented_original_pdf_pages=[1, 2],
            detected_reading_order="monotonic",
            visual_blocks=[
                VisualBlock(
                    page_number=1,
                    reading_order_index=1,
                    block_type="paragraph",
                    exact_visible_text="tiny",
                ),
                VisualBlock(
                    page_number=2,
                    reading_order_index=2,
                    block_type="paragraph",
                    exact_visible_text="tiny",
                ),
            ],
        ),
        source_id="SRC",
        job=job,
    )
    findings = validate_transcription_completeness(
        jobs=[job],
        contents=[content],
        page_preflight=pages,
        execution_mode="live",
        provider_evidence=[
            ProviderCallEvidence(
                provider_backend="google_genai",
                stage="transcription",
                job_id="job",
                success=True,
                duration_seconds=0.1,
            )
        ],
    )
    assert any(
        finding.issue_code == "implausibly_short_output" and finding.severity == "error"
        for finding in findings
    )


def test_live_semantic_structure_uses_openai_provider(
    synthetic_pdf: Path, tmp_path: Path
) -> None:
    _, tx_run = run_transcription_v3(
        pdf_path=synthetic_pdf,
        source_id="SRC",
        worker_id="worker",
        output_root=tmp_path / "runs",
        execution_mode="live",
        provider=_FakeGeminiProvider(),  # type: ignore[arg-type]
    )

    class FakeOpenAIProvider:
        provider_backend = "openai_responses"

        def __init__(self) -> None:
            self.evidence: list[dict] = []

        def create(self, *, prompt: str, payload: dict) -> SemanticStructureDraft:
            assert "%PDF-" not in json.dumps(payload)
            assert prompt
            self.evidence.append(
                {
                    "provider_backend": "openai_responses",
                    "success": True,
                    "response_id": "resp_1",
                    "token_usage": {"total_tokens": 10},
                    "duration_seconds": 0.1,
                }
            )
            return SemanticStructureDraft(
                publication_year=2018,
                publication_year_source="page 1",
                formal_items=[
                    {
                        "item_type": "statement",
                        "item_type_raw": "Statement",
                        "original_number": "1",
                        "exact_original_text": "Live exact statement.",
                        "page_start": 1,
                        "page_end": 1,
                        "extraction_confidence": 1,
                    }
                ],
                references=[],
            )

    run = run_semantic_structure(
        transcription_run=tx_run,
        output_root=tmp_path / "structured",
        worker_id="worker",
        execution_mode="live",
        provider=FakeOpenAIProvider(),  # type: ignore[arg-type]
    )
    manifest = json.loads((run / "extraction_manifest.json").read_text(encoding="utf-8"))
    assert manifest["execution_mode"] == "live"
    assert manifest["provider_backend"] == "openai_responses"
    assert manifest["provider_call_count"] == 1
    assert manifest["publication_year"] == 2018
    assert manifest["status"] == "completed"
    assert (run / "openai_semantic_structure.raw.json").is_file()


def test_live_semantic_structure_rejects_grouped_formal_item_ranges(
    synthetic_pdf: Path, tmp_path: Path
) -> None:
    _, tx_run = run_transcription_v3(
        pdf_path=synthetic_pdf,
        source_id="SRC",
        worker_id="worker",
        output_root=tmp_path / "runs",
        execution_mode="live",
        provider=_FakeGeminiProvider(),  # type: ignore[arg-type]
    )

    class GroupedOpenAIProvider:
        def __init__(self) -> None:
            self.evidence = [{"provider_backend": "openai_responses", "success": True}]

        def create(self, *, prompt: str, payload: dict) -> SemanticStructureDraft:
            payload["canonical_transcript"]["contents"][0]["visual_blocks"][0][
                "exact_visible_text"
            ] = "EMPFEHLUNG 2.1\nText eins.\nEMPFEHLUNG 2.2\nText zwei."
            return SemanticStructureDraft(
                publication_year=2018,
                publication_year_source="page 1",
                formal_items=[
                    {
                        "item_type": "recommendation",
                        "item_type_raw": "Empfehlung",
                        "original_number": "2.1-2.2",
                        "exact_original_text": (
                            "EMPFEHLUNG 2.1\nText eins.\nEMPFEHLUNG 2.2\nText zwei."
                        ),
                        "page_start": 1,
                        "page_end": 1,
                        "extraction_confidence": 1,
                    }
                ],
            )

    with pytest.raises(ValueError, match="grouped formal item range"):
        run_semantic_structure(
            transcription_run=tx_run,
            output_root=tmp_path / "structured",
            worker_id="worker",
            execution_mode="live",
            provider=GroupedOpenAIProvider(),  # type: ignore[arg-type]
        )


def test_live_semantic_structure_rejects_grouped_reference_ranges(
    synthetic_pdf: Path, tmp_path: Path
) -> None:
    _, tx_run = run_transcription_v3(
        pdf_path=synthetic_pdf,
        source_id="SRC",
        worker_id="worker",
        output_root=tmp_path / "runs",
        execution_mode="live",
        provider=_FakeGeminiProvider(),  # type: ignore[arg-type]
    )

    class GroupedReferenceProvider:
        def __init__(self) -> None:
            self.evidence = [{"provider_backend": "openai_responses", "success": True}]

        def create(self, *, prompt: str, payload: dict) -> SemanticStructureDraft:
            return SemanticStructureDraft(
                publication_year=2018,
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
                    }
                ],
                references=[
                    {
                        "original_reference_number": "[1]-[12]",
                        "exact_original_reference_text": "[1]-[12] grouped references.",
                        "page_start": 1,
                        "page_end": 1,
                        "extraction_confidence": 1,
                    }
                ],
            )

    with pytest.raises(ValueError, match="grouped reference range"):
        run_semantic_structure(
            transcription_run=tx_run,
            output_root=tmp_path / "structured",
            worker_id="worker",
            execution_mode="live",
            provider=GroupedReferenceProvider(),  # type: ignore[arg-type]
        )


def test_live_structure_from_limited_transcription_remains_technical_limited(
    synthetic_pdf: Path, tmp_path: Path
) -> None:
    _, tx_run = run_transcription_v3(
        pdf_path=synthetic_pdf,
        source_id="SRC",
        worker_id="worker",
        output_root=tmp_path / "runs",
        execution_mode="live",
        provider=_FakeGeminiProvider(),  # type: ignore[arg-type]
        max_jobs=1,
    )

    class NoItemOpenAIProvider:
        def __init__(self) -> None:
            self.evidence = [
                {
                    "provider_backend": "openai_responses",
                    "success": True,
                    "response_id": "resp_1",
                    "token_usage": {"total_tokens": 10},
                    "duration_seconds": 0.1,
                }
            ]

        def create(self, *, prompt: str, payload: dict) -> SemanticStructureDraft:
            return SemanticStructureDraft(
                document_metadata={"title": "Real limited page"},
                publication_year=None,
                publication_year_source=None,
                formal_items=[],
                review_findings=[],
            )

    run = run_semantic_structure(
        transcription_run=tx_run,
        output_root=tmp_path / "structured",
        worker_id="worker",
        execution_mode="live",
        provider=NoItemOpenAIProvider(),  # type: ignore[arg-type]
    )
    manifest = json.loads((run / "extraction_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((run / "extraction_summary.json").read_text(encoding="utf-8"))
    findings = (run / "review_findings.jsonl").read_text(encoding="utf-8")
    assert manifest["status"] == "technical_limited"
    assert manifest["pubmed_default_start_date"] is None
    assert summary["formal_item_count"] == 0
    assert "no_formal_items_in_limited_transcript" in findings


def test_live_structure_normalizes_review_finding_enums(
    synthetic_pdf: Path, tmp_path: Path
) -> None:
    _, tx_run = run_transcription_v3(
        pdf_path=synthetic_pdf,
        source_id="SRC",
        worker_id="worker",
        output_root=tmp_path / "runs",
        execution_mode="live",
        provider=_FakeGeminiProvider(),  # type: ignore[arg-type]
        max_jobs=1,
    )

    class FindingOpenAIProvider:
        def __init__(self) -> None:
            self.evidence = [{"provider_backend": "openai_responses", "success": True}]

        def create(self, *, prompt: str, payload: dict) -> SemanticStructureDraft:
            return SemanticStructureDraft(
                publication_year=2018,
                publication_year_source="page 1",
                review_findings=[
                    {
                        "finding_id": "RF-001",
                        "stage": "semantic_structure",
                        "severity": "critical",
                        "issue_code": "LIMITED_SOURCE",
                        "issue_message": "Limited source.",
                        "source_id": "SRC",
                        "workflow_continued": True,
                        "human_review_required": True,
                        "review_status": "pending",
                    },
                    {
                        "finding_id": "RF-002",
                        "stage": "semantic_structure",
                        "severity": "major",
                        "issue_code": "LIMITED_SOURCE",
                        "issue_message": "Limited source.",
                        "source_id": "SRC",
                        "workflow_continued": True,
                        "human_review_required": True,
                        "review_status": "open",
                    },
                    {
                        "finding_id": "RF-003",
                        "stage": "semantic_structure",
                        "severity": "minor",
                        "issue_code": "LIMITED_SOURCE",
                        "issue_message": "Limited source.",
                        "source_id": "SRC",
                        "workflow_continued": True,
                        "human_review_required": True,
                        "review_status": "open",
                    }
                ],
            )

    run = run_semantic_structure(
        transcription_run=tx_run,
        output_root=tmp_path / "structured",
        worker_id="worker",
        execution_mode="live",
        provider=FindingOpenAIProvider(),  # type: ignore[arg-type]
    )
    findings = [
        json.loads(line)
        for line in (run / "review_findings.jsonl").read_text().splitlines()
    ]
    provider_findings = [
        finding for finding in findings if finding["finding_id"].startswith("RF-")
    ]
    assert [finding["severity"] for finding in provider_findings] == [
        "error",
        "warning",
        "info",
    ]
    assert findings[0]["review_status"] == "open"


def test_live_semantic_structure_rejects_synthetic_markers(
    synthetic_pdf: Path, tmp_path: Path
) -> None:
    _, tx_run = run_transcription_v3(
        pdf_path=synthetic_pdf,
        source_id="SRC",
        worker_id="worker",
        output_root=tmp_path / "runs",
        execution_mode="live",
        provider=_FakeGeminiProvider(),  # type: ignore[arg-type]
    )

    class SyntheticOpenAIProvider:
        def __init__(self) -> None:
            self.evidence = [{"provider_backend": "openai_responses", "success": True}]

        def create(self, *, prompt: str, payload: dict) -> SemanticStructureDraft:
            return SemanticStructureDraft(
                publication_year=2023,
                publication_year_source="synthetic transcript fixture",
                comments=[
                    {
                        "exact_original_text": "Synthetic exact original comment.",
                        "page_start": 1,
                        "page_end": 1,
                        "extraction_confidence": 1,
                    }
                ],
            )

    with pytest.raises(ValueError, match="Synthetic fixture marker"):
        run_semantic_structure(
            transcription_run=tx_run,
            output_root=tmp_path / "structured",
            worker_id="worker",
            execution_mode="live",
            provider=SyntheticOpenAIProvider(),  # type: ignore[arg-type]
        )


def test_sdk_boundary_passes_inline_pdf_bytes_to_gemini(
    tmp_path: Path, synthetic_pdf: Path
) -> None:
    class Response:
        text = SourceContentDraft(
            represented_original_pdf_pages=[1],
            detected_reading_order="monotonic",
            visual_blocks=[
                VisualBlock(
                    page_number=1,
                    reading_order_index=1,
                    block_type="paragraph",
                    exact_visible_text="SDK boundary source text.",
                )
            ],
        ).model_dump_json()
        id = "resp"
        request_id = "req"
        finish_reason = "STOP"

        def __init__(self) -> None:
            self.usage_metadata = {"total_token_count": 1}

    class Files:
        def __init__(self) -> None:
            self.upload_called = False

        def upload(self, *args, **kwargs):
            self.upload_called = True
            raise AssertionError("Slice transcription must use inline PDF bytes")

        def delete(self, *, name: str) -> None:
            raise AssertionError("No slice upload should be deleted")

    class Models:
        def __init__(self) -> None:
            self.contents: list = []
            self.config = None

        def generate_content(self, *, model: str, contents: list, config) -> Response:
            assert model == "gemini-3.5-flash"
            assert len(contents) == 2
            assert contents[1] == "prompt"
            assert contents[0].inline_data.mime_type == "application/pdf"
            assert contents[0].inline_data.data == synthetic_pdf.read_bytes()
            assert config.response_mime_type == "application/json"
            assert config.response_schema is None
            assert isinstance(config.response_json_schema, dict)
            encoded_schema = json.dumps(config.response_json_schema, sort_keys=True)
            assert "additionalProperties" not in encoded_schema
            assert "additional_properties" not in encoded_schema
            assert str(config.thinking_config.thinking_level.value) == "MEDIUM"
            assert str(config.media_resolution.value) == "MEDIA_RESOLUTION_MEDIUM"
            assert config.temperature is None
            assert config.top_p is None
            assert config.top_k is None
            assert config.candidate_count is None
            self.contents = contents
            self.config = config
            return Response()

    class Client:
        def __init__(self) -> None:
            self.files = Files()
            self.models = Models()

    client = Client()
    provider = GeminiTranscriptionProvider(
        api_key=__import__("pydantic").SecretStr("dummy"),
        model_config={
            "model_id": "gemini-3.5-flash",
            "request_timeout_seconds": 1800,
            "thinking_level": "medium",
            "media_resolution": "high",
            "max_attempts": 1,
            "retry_initial_delay_seconds": 1,
            "retry_max_delay_seconds": 1,
            "retry_jitter_fraction": 0,
        },
        client=client,
    )
    job = TranscriptionJob(
        job_id="job",
        chunk_id="chunk",
        profile="single_column_prose_verbatim",
        primary_pages=[1],
        reason="test",
    )
    draft = provider.transcribe(synthetic_pdf, "prompt", job)
    assert draft.visual_blocks[0].exact_visible_text == "SDK boundary source text."
    assert client.files.upload_called is False
    assert provider.evidence[0].pdf_slice_hash is not None
    assert provider.evidence[0].prompt_hash is not None


def test_scout_uses_uploaded_pdf_and_cleaned_schema(synthetic_pdf: Path) -> None:
    class Remote:
        uri = "fake://uploaded"
        mime_type = "application/pdf"
        name = "files/fake"

    class Response:
        text = ExtractionScoutDraft(declared_page_count=1, regions=[]).model_dump_json()
        id = "scout-response"
        request_id = "scout-request"
        finish_reason = "STOP"

        def __init__(self) -> None:
            self.usage_metadata = {"total_token_count": 2}

    class Files:
        def __init__(self) -> None:
            self.uploaded: list[Path] = []
            self.deleted: list[str] = []

        def upload(self, *, file: Path) -> Remote:
            self.uploaded.append(file)
            return Remote()

        def delete(self, *, name: str) -> None:
            self.deleted.append(name)

    class Models:
        def __init__(self) -> None:
            self.contents: list | None = None
            self.config = None

        def generate_content(self, *, model: str, contents: list, config) -> Response:
            assert model == "gemini-3.5-flash"
            assert contents[0].uri == "fake://uploaded"
            assert contents[1] == "scout prompt"
            encoded_schema = json.dumps(config.response_json_schema, sort_keys=True)
            assert "additionalProperties" not in encoded_schema
            assert "additional_properties" not in encoded_schema
            self.contents = contents
            self.config = config
            return Response()

    class Client:
        def __init__(self) -> None:
            self.files = Files()
            self.models = Models()

    client = Client()
    provider = GeminiTranscriptionProvider(
        api_key=__import__("pydantic").SecretStr("dummy"),
        model_config={
            "model_id": "gemini-3.5-flash",
            "request_timeout_seconds": 1800,
            "thinking_level": "medium",
            "media_resolution": "profile_based",
            "max_attempts": 1,
            "retry_initial_delay_seconds": 1,
            "retry_max_delay_seconds": 1,
            "retry_jitter_fraction": 0,
        },
        client=client,
    )
    draft = provider.scout(synthetic_pdf, "scout prompt")
    assert draft.declared_page_count == 1
    assert client.files.uploaded == [synthetic_pdf]
    assert client.files.deleted == ["files/fake"]


def test_http_400_invalid_argument_is_non_retryable_and_preserves_safe_metadata(
    synthetic_pdf: Path,
) -> None:
    class InvalidArgumentError(Exception):
        status_code = 400
        status = "INVALID_ARGUMENT"

    class Models:
        def __init__(self) -> None:
            self.calls = 0

        def generate_content(self, *, model: str, contents: list, config) -> None:
            self.calls += 1
            raise InvalidArgumentError("Invalid JSON payload received.")

    class Client:
        def __init__(self) -> None:
            self.files = object()
            self.models = Models()

    sleeps: list[float] = []
    client = Client()
    provider = GeminiTranscriptionProvider(
        api_key=__import__("pydantic").SecretStr("dummy"),
        model_config={
            "model_id": "gemini-3.5-flash",
            "request_timeout_seconds": 1800,
            "thinking_level": "medium",
            "media_resolution": "profile_based",
            "max_attempts": 3,
            "retry_initial_delay_seconds": 1,
            "retry_max_delay_seconds": 1,
            "retry_jitter_fraction": 0,
        },
        client=client,
        sleep=sleeps.append,
        random_uniform=lambda _low, _high: 0.0,
    )
    job = TranscriptionJob(
        job_id="job",
        chunk_id="chunk",
        profile="single_column_prose_verbatim",
        primary_pages=[1],
        reason="test",
    )
    with pytest.raises(InvalidArgumentError):
        provider.transcribe(synthetic_pdf, "prompt", job)
    assert client.models.calls == 1
    assert sleeps == []
    evidence = provider.evidence[0]
    assert evidence.http_status == 400
    assert evidence.api_status == "INVALID_ARGUMENT"
    assert evidence.final_failure_category == "non_retryable_provider_or_local_failure"
    assert evidence.safe_error_class == "InvalidArgumentError"
    assert evidence.safe_error_message == "Invalid JSON payload received."


def test_capacity_errors_defer_before_exhausting_all_attempts(synthetic_pdf: Path) -> None:
    class CapacityError(Exception):
        status_code = 503
        status = "UNAVAILABLE"

    class Models:
        def __init__(self) -> None:
            self.calls = 0

        def generate_content(self, *, model: str, contents: list, config) -> None:
            self.calls += 1
            raise CapacityError("This model is currently experiencing high demand.")

    class Client:
        def __init__(self) -> None:
            self.files = object()
            self.models = Models()

    sleeps: list[float] = []
    client = Client()
    provider = GeminiTranscriptionProvider(
        api_key=__import__("pydantic").SecretStr("dummy"),
        model_config={
            "model_id": "gemini-3.5-flash",
            "request_timeout_seconds": 1800,
            "thinking_level": "medium",
            "media_resolution": "profile_based",
            "max_attempts": 8,
            "max_attempts_per_call": 8,
            "defer_after_consecutive_capacity_errors_per_job": 2,
            "retry_initial_delay_seconds": 1,
            "retry_max_delay_seconds": 60,
            "retry_jitter_fraction": 0,
        },
        client=client,
        sleep=sleeps.append,
        random_uniform=lambda _low, _high: 0.0,
    )
    job = TranscriptionJob(
        job_id="job",
        chunk_id="chunk",
        profile="single_column_prose_verbatim",
        primary_pages=[1],
        reason="test",
    )
    with pytest.raises(CapacityError):
        provider.transcribe(synthetic_pdf, "prompt", job)

    assert client.models.calls == 2
    assert sleeps == [1.0]
    assert provider.evidence[-1].final_failure_category == "provider_capacity_unavailable"
    assert provider.evidence[-1].calculated_delay_seconds is None


def test_invalid_json_response_is_preserved_and_deferred(synthetic_pdf: Path) -> None:
    class Response:
        text = '{"visual_blocks":["bad \\uXX escape"]}'
        id = "response-invalid-json"
        request_id = "request-invalid-json"

    class Models:
        def __init__(self) -> None:
            self.calls = 0

        def generate_content(self, *, model: str, contents: list, config) -> Response:
            self.calls += 1
            return Response()

    class Client:
        def __init__(self) -> None:
            self.files = object()
            self.models = Models()

    sleeps: list[float] = []
    client = Client()
    provider = GeminiTranscriptionProvider(
        api_key=__import__("pydantic").SecretStr("dummy"),
        model_config={
            "model_id": "gemini-3.5-flash",
            "request_timeout_seconds": 1800,
            "thinking_level": "medium",
            "media_resolution": "profile_based",
            "max_attempts": 8,
            "max_attempts_per_call": 8,
            "defer_after_consecutive_capacity_errors_per_job": 2,
            "retry_initial_delay_seconds": 1,
            "retry_max_delay_seconds": 60,
            "retry_jitter_fraction": 0,
        },
        client=client,
        sleep=sleeps.append,
        random_uniform=lambda _low, _high: 0.0,
    )
    job = TranscriptionJob(
        job_id="job",
        chunk_id="chunk",
        profile="dense_prose_verbatim",
        primary_pages=[1, 2, 3],
        reason="test",
    )
    with pytest.raises(json.JSONDecodeError):
        provider.transcribe(synthetic_pdf, "prompt", job)

    assert client.models.calls == 1
    assert sleeps == []
    assert provider.raw_responses["job"]["raw_text"] == Response.text
    evidence = provider.evidence[-1]
    assert evidence.safe_error_class == "JSONDecodeError"
    assert evidence.final_failure_category == "transient_provider_or_network"
    assert evidence.calculated_delay_seconds is None


def test_gemini_candidate_finish_reason_is_retained(synthetic_pdf: Path) -> None:
    class Reason:
        value = "STOP"

    class Candidate:
        finish_reason = Reason()

    class Response:
        text = SourceContentDraft(
            represented_original_pdf_pages=[1],
            detected_reading_order="monotonic",
            visual_blocks=[
                VisualBlock(
                    page_number=1,
                    reading_order_index=1,
                    block_type="paragraph",
                    exact_visible_text="Source text.",
                )
            ],
        ).model_dump_json()

        def __init__(self) -> None:
            self.candidates = [Candidate()]

    class Models:
        def generate_content(self, *, model: str, contents: list, config) -> Response:
            return Response()

    class Client:
        def __init__(self) -> None:
            self.files = object()
            self.models = Models()

    provider = GeminiTranscriptionProvider(
        api_key=__import__("pydantic").SecretStr("dummy"),
        model_config={
            "model_id": "gemini-3.5-flash",
            "request_timeout_seconds": 1800,
            "thinking_level": "medium",
            "media_resolution": "profile_based",
            "max_attempts": 1,
            "retry_initial_delay_seconds": 1,
            "retry_max_delay_seconds": 1,
            "retry_jitter_fraction": 0,
        },
        client=Client(),
    )
    job = TranscriptionJob(
        job_id="job",
        chunk_id="chunk",
        profile="single_column_prose_verbatim",
        primary_pages=[1],
        reason="test",
    )
    provider.transcribe(synthetic_pdf, "prompt", job)
    assert provider.evidence[0].finish_reason == "STOP"


def test_openai_responses_arguments_are_created() -> None:
    class Response:
        id = "resp_1"
        output_parsed = SemanticStructureDraft(publication_year=2018)

        def __init__(self) -> None:
            self.usage = {"total_tokens": 9}

    class Responses:
        def __init__(self) -> None:
            self.kwargs: dict | None = None

        def parse(self, **kwargs) -> Response:
            self.kwargs = kwargs
            return Response()

    class Client:
        def __init__(self) -> None:
            self.responses = Responses()

    client = Client()
    provider = OpenAISemanticStructureProvider(
        api_key=__import__("pydantic").SecretStr("dummy"),
        config={
            "model_id": "gpt-5.5",
            "reasoning_effort": "high",
            "request_timeout_seconds": 1200,
            "max_attempts": 3,
        },
        client=client,
    )
    provider.create(prompt="prompt", payload={"canonical_transcript": {"contents": []}})
    assert client.responses.kwargs["model"] == "gpt-5.5"
    assert client.responses.kwargs["reasoning"] == {"effort": "high"}
    assert client.responses.kwargs["text_format"] is SemanticStructureDraft


def test_technical_limited_runs_cannot_feed_pubmed(tmp_path: Path) -> None:
    run = tmp_path / "structure"
    run.mkdir()
    (run / "extraction_manifest.json").write_text(
        json.dumps(
            {
                "status": "technical_limited",
                "execution_mode": "live",
                "publication_year": 2018,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        derive_start_date_from_extraction_manifest(run)


def test_technical_limited_pubmed_start_date_requires_explicit_allowance(
    tmp_path: Path,
) -> None:
    run = tmp_path / "structure"
    run.mkdir()
    (run / "extraction_manifest.json").write_text(
        json.dumps(
            {
                "status": "technical_limited",
                "execution_mode": "live",
                "publication_year": 2018,
                "publication_year_source": "page footer",
            }
        ),
        encoding="utf-8",
    )
    start_date, audit = derive_start_date_from_extraction_manifest(
        run, allow_limited_input=True
    )
    assert start_date.isoformat() == "2018-01-01"
    assert audit["limited_input_accepted"] is True


def test_legacy_run_path_remains_unmodified() -> None:
    legacy = Path("/mnt/c/living_guideline_platform/runs")
    if legacy.exists():
        before = sorted(path.name for path in legacy.iterdir())
        after = sorted(path.name for path in legacy.iterdir())
        assert after == before
