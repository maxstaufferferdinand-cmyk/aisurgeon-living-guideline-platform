import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar

import pytest
from pydantic import SecretStr
from pypdf import PdfReader, PdfWriter

from aisurgeon.extraction.gemini.errors import GeminiError
from aisurgeon.search.pubmed.query import sha256_text
from aisurgeon.synthesis.reference_repair import (
    GeminiOriginalReferenceRepairV2Client,
    GeminiReferenceRepairRetryState,
    OriginalReferenceRepairV2PageJobOutput,
    bibliography_page_plan_v2,
    calculate_gemini_retry_delay_seconds,
    classify_gemini_reference_repair_error,
    load_reference_repair_v2_model_config,
    page_job_fingerprint_v2,
    parse_page_job_raw_response_v2,
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

    def request_page_job(self, *, slice_pdf: Path, prompt: str, source_id: str, **kwargs):
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


class NoApiClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def request_page_job(self, *, slice_pdf: Path, prompt: str, source_id: str, **kwargs):
        raise AssertionError("API must not be called when raw_response.json is reusable")


class FirstJobFailsClient:
    calls: ClassVar[list[str]] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def request_page_job(self, *, slice_pdf: Path, prompt: str, source_id: str, **kwargs):
        job_id = kwargs["job_id"]
        self.calls.append(job_id)
        if job_id == "page_0001":
            raise GeminiError("temporary final failure")
        primary = int(prompt.split("primary_original_pdf_page: ", 1)[1].splitlines()[0])
        numbers = [1] if primary == 1 else [2]
        payload = _page_output(primary, numbers)
        return (
            OriginalReferenceRepairV2PageJobOutput.model_validate(payload),
            payload,
            {"attempt_count": 1},
        )


class FakeHttpError(Exception):
    def __init__(self, status_code: int, message: str = "error", headers: dict | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.response = type(
            "Response", (), {"status_code": status_code, "headers": headers or {}}
        )()


class FakeRemote:
    name = "files/1"
    uri = "gemini://files/1"
    mime_type = "application/pdf"
    state = "ACTIVE"


class FakeResponse:
    def __init__(self, payload: dict):
        self.text = json.dumps(payload)
        self.finish_reason = "STOP"
        self.candidates = [object()]
        self.usage_metadata = {"total_tokens": 42}


class FakeFiles:
    def __init__(self, behaviors: list):
        self.behaviors = behaviors
        self.upload_calls = 0

    def upload(self, **kwargs):
        self.upload_calls += 1
        if self.behaviors:
            behavior = self.behaviors.pop(0)
            if isinstance(behavior, Exception):
                raise behavior
            return behavior
        return FakeRemote()

    def get(self, **kwargs):
        return FakeRemote()

    def delete(self, **kwargs):
        return None


class FakeModels:
    def __init__(self, behaviors: list):
        self.behaviors = behaviors
        self.generate_calls = 0

    def generate_content(self, **kwargs):
        self.generate_calls += 1
        if self.behaviors:
            behavior = self.behaviors.pop(0)
            if isinstance(behavior, Exception):
                raise behavior
            return behavior
        return FakeResponse(_page_output(52, [1]))


class FakeGeminiSdkClient:
    def __init__(
        self, *, upload_behaviors: list | None = None, generate_behaviors: list | None = None
    ):
        self.files = FakeFiles(upload_behaviors or [FakeRemote()])
        self.models = FakeModels(generate_behaviors or [FakeResponse(_page_output(52, [1]))])


def _retry_test_client(
    tmp_path: Path,
    sdk_client: FakeGeminiSdkClient,
    *,
    sleeps: list[float] | None = None,
    outputs: list[str] | None = None,
    retry_state: GeminiReferenceRepairRetryState | None = None,
) -> GeminiOriginalReferenceRepairV2Client:
    sleep_sink = sleeps if sleeps is not None else []
    output_sink = outputs if outputs is not None else []
    return GeminiOriginalReferenceRepairV2Client(
        api_key=SecretStr("secret"),
        model_config=load_reference_repair_v2_model_config(),
        client=sdk_client,
        sleep=sleep_sink.append,
        monotonic=lambda: 0.0,
        output=output_sink.append,
        random_fraction=lambda: 0.5,
        retry_state=retry_state,
    )


def _call_retry_client(client: GeminiOriginalReferenceRepairV2Client, tmp_path: Path) -> None:
    pdf = tmp_path / "slice.pdf"
    _pdf(pdf, pages=1)
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    client.request_page_job(
        slice_pdf=pdf,
        prompt="prompt",
        source_id="SRC",
        job_id="page_0052",
        primary_original_pdf_page=52,
        job_dir=job_dir,
    )


def test_bulk_20_page_response_with_25_references_is_rejected() -> None:
    with pytest.raises(ValueError, match="25 of 720"):
        reject_bulk_partial_reference_response({"references": [{} for _ in range(25)]})


def test_schema_version_aliases_are_normalized_and_unknown_versions_fail() -> None:
    canonical_payload = _page_output(52, [1])
    canonical_payload["schema_version"] = "original_reference_repair_v2"
    output, normalization = parse_page_job_raw_response_v2(canonical_payload)
    assert output.schema_version == "original_reference_repair_v2"
    assert normalization["schema_version_normalized"] is False

    alias_payload = _page_output(52, [1])
    alias_payload["schema_version"] = "2.0.0"
    output, normalization = parse_page_job_raw_response_v2(alias_payload)
    assert output.schema_version == "original_reference_repair_v2"
    assert normalization == {
        "raw_schema_version": "2.0.0",
        "normalized_schema_version": "original_reference_repair_v2",
        "schema_version_normalized": True,
    }

    v2_payload = _page_output(52, [1])
    v2_payload["schema_version"] = "v2"
    output, normalization = parse_page_job_raw_response_v2(v2_payload)
    assert output.schema_version == "original_reference_repair_v2"
    assert normalization["raw_schema_version"] == "v2"
    assert normalization["schema_version_normalized"] is True

    unknown_payload = _page_output(52, [1])
    unknown_payload["schema_version"] = "3.0.0"
    with pytest.raises(ValueError, match="Unknown"):
        parse_page_job_raw_response_v2(unknown_payload)


def test_gemini_503_retries_with_backoff_and_attempt_logging(tmp_path: Path) -> None:
    sleeps: list[float] = []
    outputs: list[str] = []
    sdk_client = FakeGeminiSdkClient(
        generate_behaviors=[
            FakeHttpError(503, "Service temporarily unavailable"),
            FakeResponse(_page_output(52, [1])),
        ]
    )
    client = _retry_test_client(tmp_path, sdk_client, sleeps=sleeps, outputs=outputs)
    _call_retry_client(client, tmp_path)
    job_dir = tmp_path / "job"
    attempts = [
        json.loads(line)
        for line in (job_dir / "attempts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert sdk_client.models.generate_calls == 2
    failed_attempt = next(row for row in attempts if row["http_status"] == 503)
    assert failed_attempt["transient"] is True
    assert failed_attempt["retry_planned"] is True
    assert failed_attempt["applied_delay_seconds"] == 15
    assert sleeps == [15.0]
    assert "Neuer Versuch in 15 Sekunden" in outputs[0]
    artifacts = "\n".join(path.read_text(encoding="utf-8") for path in job_dir.glob("*.json*"))
    assert "secret" not in artifacts.lower()
    assert "authorization" not in artifacts.lower()


def test_gemini_429_uses_retry_after(tmp_path: Path) -> None:
    sleeps: list[float] = []
    outputs: list[str] = []
    sdk_client = FakeGeminiSdkClient(
        generate_behaviors=[
            FakeHttpError(429, "RESOURCE_EXHAUSTED", headers={"Retry-After": "300"}),
            FakeResponse(_page_output(52, [1])),
        ]
    )
    client = _retry_test_client(tmp_path, sdk_client, sleeps=sleeps, outputs=outputs)
    _call_retry_client(client, tmp_path)
    attempts = [
        json.loads(line)
        for line in (tmp_path / "job" / "attempts.jsonl").read_text().splitlines()
    ]
    attempt = next(row for row in attempts if row["http_status"] == 429)
    assert attempt["retry_after_seconds"] == 300
    assert attempt["applied_delay_seconds"] == 300
    assert sleeps == [60.0, 60.0, 60.0, 60.0, 60.0]
    assert "HTTP 429" in outputs[0]
    assert "Retry-After: 300 Sekunden" in outputs[0]
    assert any("noch 240 Sekunden" in line for line in outputs)


@pytest.mark.parametrize("status", [500, 502, 504])
def test_retryable_server_statuses_are_transient(status: int) -> None:
    assert classify_gemini_reference_repair_error(FakeHttpError(status))["transient"] is True


def test_read_timeout_is_retried(tmp_path: Path) -> None:
    sleeps: list[float] = []
    sdk_client = FakeGeminiSdkClient(
        generate_behaviors=[TimeoutError("Read timeout"), FakeResponse(_page_output(52, [1]))]
    )
    client = _retry_test_client(tmp_path, sdk_client, sleeps=sleeps)
    _call_retry_client(client, tmp_path)
    assert sdk_client.models.generate_calls == 2
    assert sleeps == [15.0]


@pytest.mark.parametrize("status", [400, 401, 403])
def test_non_retryable_client_statuses_are_not_retried(tmp_path: Path, status: int) -> None:
    sdk_client = FakeGeminiSdkClient(
        generate_behaviors=[FakeHttpError(status, "permission denied")]
    )
    client = _retry_test_client(tmp_path, sdk_client)
    with pytest.raises(Exception, match="Gemini PageJob"):
        _call_retry_client(client, tmp_path)
    assert sdk_client.models.generate_calls == 1
    attempts = [
        json.loads(line)
        for line in (tmp_path / "job" / "attempts.jsonl").read_text().splitlines()
    ]
    attempt = next(row for row in attempts if row["http_status"] == status)
    assert attempt["http_status"] == status
    assert attempt["retry_planned"] is False
    assert (tmp_path / "job" / "last_error.json").is_file()


def test_jitter_and_retry_max_delay_are_bounded() -> None:
    config = load_reference_repair_v2_model_config()
    calculated, low = calculate_gemini_retry_delay_seconds(
        model_config=config, attempt_number=3, random_fraction=0.0
    )
    assert calculated == 60
    assert low == 45
    _, high = calculate_gemini_retry_delay_seconds(
        model_config=config, attempt_number=3, random_fraction=1.0
    )
    assert high == 75
    calculated, capped = calculate_gemini_retry_delay_seconds(
        model_config=config, attempt_number=8, random_fraction=1.0
    )
    assert calculated == 900
    assert capped == 900


def test_success_after_transient_resets_circuit_breaker(tmp_path: Path) -> None:
    state = GeminiReferenceRepairRetryState()
    sdk_client = FakeGeminiSdkClient(
        generate_behaviors=[FakeHttpError(503), FakeResponse(_page_output(52, [1]))]
    )
    client = _retry_test_client(tmp_path, sdk_client, retry_state=state)
    _call_retry_client(client, tmp_path)
    assert state.consecutive_transient_failures == 0


def test_three_transient_failures_trigger_global_cooldown(tmp_path: Path) -> None:
    sleeps: list[float] = []
    outputs: list[str] = []
    state = GeminiReferenceRepairRetryState()
    state.consecutive_transient_failures = 3
    sdk_client = FakeGeminiSdkClient()
    client = _retry_test_client(
        tmp_path, sdk_client, sleeps=sleeps, outputs=outputs, retry_state=state
    )
    _call_retry_client(client, tmp_path)
    assert sum(sleeps) == 900
    assert any("Globaler Cooldown" in line for line in outputs)
    assert any("noch 840 Sekunden" in line for line in outputs)


def test_unknown_non_transient_error_is_hard_fail(tmp_path: Path) -> None:
    sdk_client = FakeGeminiSdkClient(generate_behaviors=[RuntimeError("bad request schema")])
    client = _retry_test_client(tmp_path, sdk_client)
    with pytest.raises(Exception, match="Gemini PageJob"):
        _call_retry_client(client, tmp_path)
    assert sdk_client.models.generate_calls == 1


def test_final_failed_page_job_does_not_block_later_page_jobs(tmp_path: Path) -> None:
    pdf, extraction, synthesis, failed = _runs(tmp_path, bibliography_start=1, bibliography_end=2)
    FirstJobFailsClient.calls = []
    with pytest.raises(RuntimeError, match="Reference repair v2 failed"):
        run_reference_repair_v2_and_rebuild(
            pdf=pdf,
            extraction_run=extraction,
            synthesis_run=synthesis,
            failed_reference_run=failed,
            output_root=tmp_path / "out",
            api_key=SecretStr("secret"),
            client_factory=FirstJobFailsClient,
            now=lambda: datetime(2026, 7, 19, tzinfo=UTC),
        )
    assert FirstJobFailsClient.calls == ["page_0001", "page_0002"]
    run = next((tmp_path / "out").glob("reference-repair-v2-*"))
    assert json.loads((run / "page_jobs" / "page_0001" / "checkpoint.json").read_text())[
        "status"
    ] == "failed"


def test_resume_repairs_saved_alias_raw_response_without_api_call(tmp_path: Path) -> None:
    pdf, extraction, synthesis, failed = _runs(tmp_path, bibliography_start=1, bibliography_end=1)
    FakeV2Client.calls = []
    FakeV2Client.skip_number = None
    with pytest.raises(RuntimeError, match="Reference repair v2 failed"):
        run_reference_repair_v2_and_rebuild(
            pdf=pdf,
            extraction_run=extraction,
            synthesis_run=synthesis,
            failed_reference_run=failed,
            output_root=tmp_path / "out",
            api_key=SecretStr("secret"),
            client_factory=FakeV2Client,
            now=lambda: datetime(2026, 7, 19, tzinfo=UTC),
        )
    run = next((tmp_path / "out").glob("reference-repair-v2-*"))
    job_dir = run / "page_jobs" / "page_0001"
    raw_path = job_dir / "raw_response.json"
    raw = json.loads(raw_path.read_text())
    raw["schema_version"] = "2.0.0"
    _json(raw_path, raw)
    checkpoint_path = job_dir / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint["status"] = "failed"
    _json(checkpoint_path, checkpoint)

    with pytest.raises(RuntimeError, match="Reference repair v2 failed"):
        run_reference_repair_v2_and_rebuild(
            pdf=pdf,
            extraction_run=extraction,
            synthesis_run=synthesis,
            failed_reference_run=failed,
            output_root=tmp_path / "out",
            api_key=SecretStr("secret"),
            resume_run=run,
            client_factory=NoApiClient,
            now=lambda: datetime(2026, 7, 19, tzinfo=UTC),
        )
    manifest = json.loads((job_dir / "job_manifest.json").read_text())
    assert manifest["schema_version_normalization"] == {
        "raw_schema_version": "2.0.0",
        "normalized_schema_version": "original_reference_repair_v2",
        "schema_version_normalized": True,
    }


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
