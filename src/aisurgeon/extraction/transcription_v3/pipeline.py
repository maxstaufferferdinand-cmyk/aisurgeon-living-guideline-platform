"""Canonical Gemini transcription v3 run preparation and mocked execution."""

import hashlib
import json
import random
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, SecretStr, ValidationError
from pypdf import PdfReader, PdfWriter

from aisurgeon.extraction.canonical.outputs import write_json, write_jsonl
from aisurgeon.extraction.gemini.document_map import (
    ensure_output_outside_repository,
    find_project_root,
    git_metadata,
)
from aisurgeon.extraction.gemini.errors import GeminiResponseValidationError
from aisurgeon.extraction.pdf_preflight import PdfPagePreflight, PdfPreflight, run_pdf_preflight
from aisurgeon.extraction.pdf_registration import register_pdf
from aisurgeon.extraction.transcription_v3.completeness import validate_transcription_completeness
from aisurgeon.extraction.transcription_v3.models import (
    CANONICAL_TRANSCRIPTION_SCHEMA_VERSION,
    SCOUT_PROMPT_VERSION,
    SCOUT_SCHEMA_VERSION,
    TRANSCRIPTION_PROMPT_VERSION,
    CompletenessFinding,
    ExecutionMode,
    ExtractionScout,
    ExtractionScoutDraft,
    ProviderCallEvidence,
    SourceContent,
    SourceContentDraft,
    TranscriptionJob,
    VisualBlock,
)
from aisurgeon.extraction.transcription_v3.planner import build_transcription_plan
from aisurgeon.extraction.transcription_v3.retry import classify_provider_failure

GEMINI_MODEL_ID = "gemini-3.5-flash"
MODEL_CONFIG_PATH = Path("config/models/gemini_source_transcription_v3.json")
PROMPT_PATH = Path("config/prompts/gemini_source_transcription_v3.txt")
SCOUT_PROMPT_PATH = Path("config/prompts/gemini_technical_layout_scout_v1.txt")
SYNTHETIC_MARKERS = (
    "Synthetic source transcript",
    "synthetic_monotonic",
    "mocked_scout_in_dry_or_test_run",
)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def inject_scout_metadata(draft: ExtractionScoutDraft, *, source_id: str) -> ExtractionScout:
    return ExtractionScout.model_validate(
        {
            **draft.model_dump(mode="json"),
            "schema_version": SCOUT_SCHEMA_VERSION,
            "source_id": source_id,
            "prompt_version": SCOUT_PROMPT_VERSION,
            "model_id": GEMINI_MODEL_ID,
        }
    )


def inject_source_content_metadata(
    draft: SourceContentDraft, *, source_id: str, job: TranscriptionJob
) -> SourceContent:
    return SourceContent.model_validate(
        {
            **draft.model_dump(mode="json"),
            "schema_version": CANONICAL_TRANSCRIPTION_SCHEMA_VERSION,
            "source_id": source_id,
            "prompt_version": TRANSCRIPTION_PROMPT_VERSION,
            "model_id": GEMINI_MODEL_ID,
            "job_id": job.job_id,
            "chunk_id": job.chunk_id,
        }
    )


def create_pdf_slice(pdf_path: Path, job: TranscriptionJob, output_path: Path) -> None:
    reader = PdfReader(pdf_path, strict=True)
    writer = PdfWriter()
    for entry in job.slice_page_map:
        writer.add_page(reader.pages[entry.original_pdf_page_number - 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as stream:
        writer.write(stream)


def _load_v3_config(project_root: Path) -> dict[str, Any]:
    return json.loads((project_root / MODEL_CONFIG_PATH).read_text(encoding="utf-8"))


def gemini_request_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Return a Gemini request schema while preserving strict local validation."""
    schema = model.model_json_schema()
    definitions = schema.get("$defs", {})
    unsupported = {
        "$defs",
        "$schema",
        "additionalProperties",
        "additional_properties",
        "default",
        "description",
        "examples",
        "title",
    }

    def resolve_ref(ref: str) -> Any:
        prefix = "#/$defs/"
        if not ref.startswith(prefix):
            raise ValueError(f"Unsupported schema reference: {ref}")
        name = ref.removeprefix(prefix)
        if name not in definitions:
            raise ValueError(f"Unresolved schema reference: {ref}")
        return clean(definitions[name])

    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            if "$ref" in value:
                resolved = resolve_ref(str(value["$ref"]))
                merged = {key: item for key, item in value.items() if key != "$ref"}
                if merged:
                    if not isinstance(resolved, dict):
                        raise ValueError("Referenced schema cannot be merged")
                    resolved = {**resolved, **clean(merged)}
                return resolved
            any_of = value.get("anyOf")
            if isinstance(any_of, list):
                non_null = [
                    option
                    for option in any_of
                    if not (isinstance(option, dict) and option.get("type") == "null")
                ]
                if len(non_null) == 1 and len(non_null) != len(any_of):
                    result = clean(non_null[0])
                    if isinstance(result, dict):
                        result["nullable"] = True
                    return result
            result = {
                key: clean(item)
                for key, item in value.items()
                if key not in unsupported and key != "anyOf"
            }
            if "const" in result:
                result["enum"] = [result.pop("const")]
            return result
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    cleaned = clean(schema)
    if not isinstance(cleaned, dict):
        raise ValueError("Gemini request schema must be an object")
    return cleaned


def _safe_usage(usage: Any) -> dict[str, int] | None:
    if usage is None:
        return None
    names = (
        "prompt_token_count",
        "candidates_token_count",
        "thoughts_token_count",
        "cached_content_token_count",
        "total_token_count",
        "total_tokens",
        "input_tokens",
        "output_tokens",
    )
    values = {
        name: value
        for name in names
        if isinstance((value := getattr(usage, name, None)), int)
    }
    if isinstance(usage, dict):
        values.update({key: value for key, value in usage.items() if isinstance(value, int)})
    return values or None


def _safe_finish_reason(response: Any) -> str | None:
    direct = getattr(response, "finish_reason", None)
    if direct:
        value = getattr(direct, "value", direct)
        return str(value)
    candidates = getattr(response, "candidates", None)
    if isinstance(candidates, list) and candidates:
        reason = getattr(candidates[0], "finish_reason", None)
        if reason:
            value = getattr(reason, "value", reason)
            return str(value)
    return None


def _safe_status(exc: Exception) -> tuple[int | None, str | None, float | None]:
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    status = status if isinstance(status, int) else None
    api_status = getattr(exc, "status", None)
    api_status = str(api_status) if api_status is not None else None
    retry_after = None
    headers = getattr(exc, "headers", None)
    if isinstance(headers, dict):
        value = headers.get("retry-after") or headers.get("Retry-After")
        try:
            retry_after = float(value) if value is not None else None
        except ValueError:
            retry_after = None
    return status, api_status, retry_after


def _media_resolution_for_job(job: TranscriptionJob | None) -> str:
    if job is None:
        return "high"
    if job.profile == "single_column_prose_verbatim":
        return "medium"
    return "high"


def _media_resolution_enum(level: str) -> Any:
    from google.genai import types

    values = {
        "medium": types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
        "high": types.MediaResolution.MEDIA_RESOLUTION_HIGH,
    }
    return values[level]


def _thinking_level_enum(level: str) -> Any:
    from google.genai import types

    values = {
        "minimal": types.ThinkingLevel.MINIMAL,
        "low": types.ThinkingLevel.LOW,
        "medium": types.ThinkingLevel.MEDIUM,
        "high": types.ThinkingLevel.HIGH,
    }
    return values[level]


class GeminiTranscriptionProvider:
    """Real Google GenAI boundary for v3 scout and physical-slice transcription."""

    provider_backend = "google_genai"

    def __init__(
        self,
        *,
        api_key: SecretStr,
        model_config: dict[str, Any],
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
        random_uniform: Callable[[float, float], float] = random.uniform,
    ) -> None:
        self.model_config = model_config
        self.evidence: list[ProviderCallEvidence] = []
        self.raw_responses: dict[str, Any] = {}
        self._sleep = sleep
        self._random_uniform = random_uniform
        if client is None:
            from google import genai
            from google.genai import types

            client = genai.Client(
                api_key=api_key.get_secret_value(),
                http_options=types.HttpOptions(
                    timeout=int(model_config["request_timeout_seconds"]) * 1000
                ),
            )
        self._client = client

    def _config(self, schema_model: type[Any], media_resolution: str) -> Any:
        from google.genai import types

        return types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_level=_thinking_level_enum(str(self.model_config["thinking_level"]))
            ),
            media_resolution=_media_resolution_enum(media_resolution),
            response_mime_type="application/json",
            response_json_schema=gemini_request_schema(schema_model),
            http_options=types.HttpOptions(
                timeout=int(self.model_config["request_timeout_seconds"]) * 1000
            ),
        )

    def _evidence(
        self,
        *,
        stage: str,
        job_id: str | None,
        attempt: int,
        success: bool,
        start: float,
        response: Any | None = None,
        exc: Exception | None = None,
        pdf_hash: str | None = None,
        prompt_hash: str | None = None,
        uploaded_file: Any | None = None,
        remote_file_deleted: bool | None = None,
    ) -> ProviderCallEvidence:
        http_status = api_status = retry_after = None
        delay = final_category = None
        if exc is not None:
            http_status, api_status, retry_after = _safe_status(exc)
            decision = classify_provider_failure(
                exc,
                attempt=attempt,
                retry_after_seconds=retry_after,
                base_delay_seconds=float(self.model_config["retry_initial_delay_seconds"]),
                max_delay_seconds=float(self.model_config["retry_max_delay_seconds"]),
                jitter_fraction=0.0,
            )
            delay = decision.calculated_delay_seconds
            final_category = decision.final_failure_category
        return ProviderCallEvidence(
            provider_backend=self.provider_backend,
            stage=cast("Any", stage),
            job_id=job_id,
            attempt=attempt,
            success=success,
            request_id=getattr(response, "request_id", None),
            response_id=getattr(response, "id", None),
            token_usage=_safe_usage(getattr(response, "usage_metadata", None)),
            finish_reason=_safe_finish_reason(response),
            duration_seconds=round(time.monotonic() - start, 3),
            http_status=http_status,
            api_status=api_status,
            calculated_delay_seconds=delay,
            retry_after_seconds=retry_after,
            final_failure_category=final_category,
            pdf_slice_hash=pdf_hash,
            prompt_hash=prompt_hash,
            safe_error_class=type(exc).__name__ if exc is not None else None,
            safe_error_message=str(exc)[:200] if exc is not None else None,
            uploaded_file_name=getattr(uploaded_file, "name", None),
            uploaded_file_uri_present=bool(getattr(uploaded_file, "uri", None)),
            remote_file_deleted=remote_file_deleted,
        )

    def _parse_payload(self, response: Any) -> dict[str, Any]:
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, BaseException):
            parsed = None
        if hasattr(response, "output_parsed"):
            parsed = response.output_parsed
        if parsed is not None:
            if hasattr(parsed, "model_dump"):
                return parsed.model_dump(mode="json")
            if isinstance(parsed, dict):
                return parsed
        raw = getattr(response, "text", None)
        if not isinstance(raw, str):
            raise GeminiResponseValidationError("Gemini response did not contain JSON text.")
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise GeminiResponseValidationError("Gemini response was not a JSON object.")
        return payload

    def _generate_inline_pdf(
        self,
        *,
        pdf_path: Path,
        prompt: str,
        schema_model: type[Any],
        stage: str,
        job_id: str | None = None,
        media_resolution: str,
    ) -> tuple[dict[str, Any], ProviderCallEvidence]:
        from google.genai import types

        pdf_bytes = pdf_path.read_bytes()
        pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        start = time.monotonic()
        try:
            pdf_part = types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")
            response = self._client.models.generate_content(
                model=self.model_config["model_id"],
                contents=[pdf_part, prompt],
                config=self._config(schema_model, media_resolution),
            )
            payload = self._parse_payload(response)
            evidence = self._evidence(
                stage=stage,
                job_id=job_id,
                attempt=1,
                success=True,
                start=start,
                response=response,
                pdf_hash=pdf_hash,
                prompt_hash=prompt_hash,
                remote_file_deleted=None,
            )
            return payload, evidence
        except Exception as exc:
            evidence = self._evidence(
                stage=stage,
                job_id=job_id,
                attempt=1,
                success=False,
                start=start,
                exc=exc,
                pdf_hash=pdf_hash,
                prompt_hash=prompt_hash,
                remote_file_deleted=None,
            )
            self.evidence.append(evidence)
            raise

    def _upload_and_generate(
        self,
        *,
        pdf_path: Path,
        prompt: str,
        schema_model: type[Any],
        stage: str,
        job_id: str | None = None,
        media_resolution: str,
    ) -> tuple[dict[str, Any], ProviderCallEvidence]:
        pdf_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        start = time.monotonic()
        remote = None
        deleted = False
        try:
            remote = self._client.files.upload(file=pdf_path)
            response = self._client.models.generate_content(
                model=self.model_config["model_id"],
                contents=[remote, prompt],
                config=self._config(schema_model, media_resolution),
            )
            payload = self._parse_payload(response)
            evidence = self._evidence(
                stage=stage,
                job_id=job_id,
                attempt=1,
                success=True,
                start=start,
                response=response,
                pdf_hash=pdf_hash,
                prompt_hash=prompt_hash,
                uploaded_file=remote,
                remote_file_deleted=deleted,
            )
            return payload, evidence
        except Exception as exc:
            evidence = self._evidence(
                stage=stage,
                job_id=job_id,
                attempt=1,
                success=False,
                start=start,
                exc=exc,
                pdf_hash=pdf_hash,
                prompt_hash=prompt_hash,
                uploaded_file=remote,
                remote_file_deleted=deleted,
            )
            self.evidence.append(evidence)
            raise
        finally:
            if remote is not None:
                try:
                    self._client.files.delete(name=remote.name)
                    deleted = True
                    if "evidence" in locals():
                        evidence.remote_file_deleted = True
                except Exception:
                    pass

    def scout(self, pdf_path: Path, prompt: str) -> ExtractionScoutDraft:
        payload, evidence = self._upload_and_generate(
            pdf_path=pdf_path,
            prompt=prompt,
            schema_model=ExtractionScoutDraft,
            stage="scout",
            media_resolution="high",
        )
        self.evidence.append(evidence)
        try:
            return ExtractionScoutDraft.model_validate(payload)
        except ValidationError as exc:
            raise GeminiResponseValidationError(
                "Gemini scout response failed schema validation."
            ) from exc

    def transcribe(
        self, slice_path: Path, prompt: str, job: TranscriptionJob
    ) -> SourceContentDraft:
        max_attempts = int(self.model_config["max_attempts"])
        last_exc: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            start_index = len(self.evidence)
            try:
                payload, evidence = self._generate_inline_pdf(
                    pdf_path=slice_path,
                    prompt=prompt,
                    schema_model=SourceContentDraft,
                    stage="transcription",
                    job_id=job.job_id,
                    media_resolution=_media_resolution_for_job(job),
                )
                evidence.attempt = attempt
                self.evidence.append(evidence)
                self.raw_responses[job.job_id] = payload
                return SourceContentDraft.model_validate(payload)
            except ValidationError as exc:
                raise GeminiResponseValidationError(
                    "Gemini transcription response failed schema validation."
                ) from exc
            except Exception as exc:
                last_exc = exc
                if len(self.evidence) > start_index:
                    self.evidence[-1].attempt = attempt
                retry_after = self.evidence[-1].retry_after_seconds if self.evidence else None
                decision = classify_provider_failure(
                    exc,
                    attempt=attempt,
                    retry_after_seconds=retry_after,
                    base_delay_seconds=float(self.model_config["retry_initial_delay_seconds"]),
                    max_delay_seconds=float(self.model_config["retry_max_delay_seconds"]),
                    jitter_fraction=float(self.model_config["retry_jitter_fraction"]),
                )
                if decision.category == "non_retryable" or attempt >= max_attempts:
                    raise
                delay = decision.calculated_delay_seconds or 0
                jitter = self._random_uniform(0, delay * 0.1) if delay else 0
                if self.evidence:
                    self.evidence[-1].calculated_delay_seconds = delay + jitter
                self._sleep(delay + jitter)
        raise RuntimeError("Gemini transcription failed") from last_exc


def _job_prompt(job: TranscriptionJob) -> str:
    return (
        f"prompt_version: {TRANSCRIPTION_PROMPT_VERSION}\n"
        f"profile: {job.profile}\n"
        f"primary_pages: {job.primary_pages}\n"
        f"context_pages: {job.context_pages}\n\n"
        "Transcribe visible source text faithfully. Use generic visual block types only. "
        "Do not classify recommendations, statements, comments, references, clinical conclusions, "
        "IDs, schema versions, or source IDs."
    )


def _mock_source_content(job: TranscriptionJob) -> SourceContentDraft:
    blocks = [
        VisualBlock(
            page_number=page,
            reading_order_index=index,
            block_type="paragraph",
            exact_visible_text=f"Synthetic source transcript for page {page}.",
        )
        for index, page in enumerate(job.primary_pages, start=1)
    ]
    return SourceContentDraft(
        represented_original_pdf_pages=job.primary_pages,
        detected_reading_order="synthetic_monotonic",
        visual_blocks=blocks,
    )


def _write_job_artifacts(
    *,
    run_dir: Path,
    pdf_path: Path,
    source_id: str,
    job: TranscriptionJob,
    execution_mode: ExecutionMode,
    draft_factory: Callable[[TranscriptionJob], SourceContentDraft] | None = None,
    provider: GeminiTranscriptionProvider | None = None,
) -> SourceContent:
    job_dir = run_dir / "transcription_jobs" / job.job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    raw_path = job_dir / "raw_response.json"
    validated_path = job_dir / "validated_source_content.json"
    checkpoint_path = job_dir / "checkpoint.json"
    if checkpoint_path.is_file():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("status") == "completed" and validated_path.is_file():
            return SourceContent.model_validate_json(validated_path.read_text(encoding="utf-8"))
    create_pdf_slice(pdf_path, job, job_dir / "slice.pdf")
    write_json(
        job_dir / "slice_page_map.json",
        [entry.model_dump() for entry in job.slice_page_map],
    )
    (job_dir / "request_prompt.txt").write_text(_job_prompt(job) + "\n", encoding="utf-8")
    write_json(job_dir / "job_manifest.json", job)
    attempts_path = job_dir / "attempts.jsonl"
    attempts_path.write_text("", encoding="utf-8")
    if execution_mode == "live":
        if provider is None:
            raise ValueError("Live transcription requires a Gemini provider")
        before = len(provider.evidence)
        try:
            draft = provider.transcribe(job_dir / "slice.pdf", _job_prompt(job), job)
        finally:
            attempts = [
                evidence
                for evidence in provider.evidence[before:]
                if evidence.job_id == job.job_id
            ]
            attempts_path.write_text(
                "".join(evidence.model_dump_json() + "\n" for evidence in attempts),
                encoding="utf-8",
            )
        raw_responses = getattr(provider, "raw_responses", {})
        raw_payload = raw_responses.get(job.job_id, draft.model_dump(mode="json"))
    elif execution_mode == "mock_test":
        if draft_factory is None:
            raise ValueError("mock_test requires an explicit draft factory")
        draft = draft_factory(job)
        raw_payload = draft.model_dump(mode="json")
    else:
        raise ValueError("dry_run must not write transcription job responses")
    write_json(raw_path, raw_payload)
    content = inject_source_content_metadata(draft, source_id=source_id, job=job)
    write_json(validated_path, content)
    write_json(checkpoint_path, {"status": "completed", "job_id": job.job_id})
    return content


def _page_metrics(
    *,
    page_rows: list[dict[str, Any]],
    page_preflight: list[PdfPagePreflight],
    provider_evidence: list[ProviderCallEvidence],
) -> dict[str, Any]:
    source_chars: dict[int, int] = {}
    provider_job: dict[int, str] = {}
    provider_response: dict[int, str | None] = {}
    provider_success: dict[int, bool] = {}
    for row in page_rows:
        page = int(row["page_number"])
        source_chars[page] = source_chars.get(page, 0) + len(row["exact_visible_text"].strip())
        provider_job[page] = str(row["job_id"])
    evidence_by_job = {item.job_id: item for item in provider_evidence if item.job_id}
    for page, job_id in provider_job.items():
        evidence = evidence_by_job.get(job_id)
        provider_response[page] = evidence.response_id if evidence is not None else None
        provider_success[page] = bool(evidence and evidence.success)
    local_chars = {page.page_number: page.text_layer_character_count for page in page_preflight}
    ratios = {
        page: (round(source_chars.get(page, 0) / chars, 4) if chars else None)
        for page, chars in local_chars.items()
    }
    return {
        "source_characters_by_page": {str(k): v for k, v in sorted(source_chars.items())},
        "local_text_layer_characters_by_page": {str(k): v for k, v in sorted(local_chars.items())},
        "transcription_ratio_by_page": {str(k): v for k, v in sorted(ratios.items())},
        "page_provider_job_id": {str(k): v for k, v in sorted(provider_job.items())},
        "page_provider_response_id": {str(k): v for k, v in sorted(provider_response.items())},
        "page_resolution_method": {
            str(k): "provider_response" if provider_success.get(k) else "local_mock_or_missing"
            for k in sorted(local_chars)
        },
    }


def write_merged_transcript_outputs(
    *,
    run_dir: Path,
    preflight: PdfPreflight,
    page_preflight: list[PdfPagePreflight],
    scout: ExtractionScout,
    jobs: list[TranscriptionJob],
    contents: list[SourceContent],
    execution_mode: ExecutionMode,
    provider_evidence: list[ProviderCallEvidence],
    limit: int | None,
) -> str:
    findings = validate_transcription_completeness(
        jobs=jobs,
        contents=contents,
        page_preflight=page_preflight,
        execution_mode=execution_mode,
        provider_evidence=provider_evidence,
    )
    if execution_mode == "live":
        serialized = json.dumps(
            [content.model_dump(mode="json") for content in contents],
            ensure_ascii=False,
        )
        if any(marker in serialized for marker in SYNTHETIC_MARKERS):
            findings.append(
                CompletenessFinding(
                    finding_id="TX3_FINDING_SYNTHETIC_MARKER",
                    severity="error",
                    issue_code="synthetic_marker_in_live_output",
                    issue_message=(
                        "Known synthetic fixture marker appeared in live transcription output."
                    ),
                    repair_required=True,
                )
            )
    provider_call_count = len(provider_evidence)
    successful_call_count = sum(1 for item in provider_evidence if item.success)
    failed_call_count = provider_call_count - successful_call_count
    critical = any(finding.severity == "error" for finding in findings)
    if execution_mode == "live" and provider_call_count == 0:
        critical = True
    status = (
        "dry_run"
        if execution_mode == "dry_run"
        else "mock_test"
        if execution_mode == "mock_test"
        else "technical_limited"
        if limit is not None
        else "completed"
        if not critical
        else "failed"
    )
    page_rows = []
    for content in contents:
        for block in content.visual_blocks:
            page_rows.append(
                {
                    "schema_version": CANONICAL_TRANSCRIPTION_SCHEMA_VERSION,
                    "source_id": content.source_id,
                    "job_id": content.job_id,
                    "page_number": block.page_number,
                    "reading_order_index": block.reading_order_index,
                    "block_type": block.block_type,
                    "exact_visible_text": block.exact_visible_text,
                    "uncertainty": block.uncertainty,
                }
            )
    write_jsonl(run_dir / "page_transcript.jsonl", page_rows)
    metrics = _page_metrics(
        page_rows=page_rows,
        page_preflight=page_preflight,
        provider_evidence=provider_evidence,
    )
    write_json(
        run_dir / "canonical_transcript.json",
        {
            "schema_version": CANONICAL_TRANSCRIPTION_SCHEMA_VERSION,
            "source_id": preflight.source_id,
            "pdf_sha256": preflight.pdf_sha256,
            "contents": [content.model_dump(mode="json") for content in contents],
        },
    )
    markdown = "\n".join(
        f"## Page {row['page_number']}\n\n{row['exact_visible_text']}" for row in page_rows
    )
    (run_dir / "canonical_transcript.md").write_text(markdown + "\n", encoding="utf-8")
    table_rows = [row for row in page_rows if row["block_type"] == "table"]
    algorithm_rows = [row for row in page_rows if row["block_type"] == "diagram_text"]
    write_jsonl(run_dir / "table_transcripts.jsonl", table_rows)
    write_jsonl(run_dir / "algorithm_transcripts.jsonl", algorithm_rows)
    write_jsonl(run_dir / "transcription_uncertainties.jsonl", findings)
    write_json(
        run_dir / "transcription_coverage_report.json",
        {
            "status": status,
            "planned_primary_pages": sorted({p for job in jobs for p in job.primary_pages}),
            "resolved_primary_pages": sorted({row["page_number"] for row in page_rows}),
            "finding_count": len(findings),
            "limited": limit is not None,
            **metrics,
        },
    )
    write_json(
        run_dir / "transcription_manifest.json",
        {
            "schema_version": CANONICAL_TRANSCRIPTION_SCHEMA_VERSION,
            "status": status,
            "execution_mode": execution_mode,
            "provider_backend": (
                "google_genai"
                if execution_mode == "live"
                else "internal_mock"
                if execution_mode == "mock_test"
                else "none"
            ),
            "provider_call_count": provider_call_count,
            "scout_call_count": sum(1 for item in provider_evidence if item.stage == "scout"),
            "transcription_call_count": sum(
                1 for item in provider_evidence if item.stage == "transcription"
            ),
            "successful_call_count": successful_call_count,
            "failed_call_count": failed_call_count,
            "provider_evidence": [item.model_dump(mode="json") for item in provider_evidence],
            "source_id": preflight.source_id,
            "pdf_sha256": preflight.pdf_sha256,
            "gemini_model_id": GEMINI_MODEL_ID,
            "gemini_concurrency": 1,
            "prompt_version": TRANSCRIPTION_PROMPT_VERSION,
            "scout_prompt_version": SCOUT_PROMPT_VERSION,
            "run_mode": "technical_limited" if limit else "complete",
            "limit": limit,
            "output_hashes": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in run_dir.iterdir()
                if path.is_file()
            },
        },
    )
    return status


def run_transcription_v3(
    *,
    pdf_path: Path,
    source_id: str,
    worker_id: str,
    output_root: Path,
    planner_mode: str = "deterministic",
    gemini_concurrency: int = 1,
    execution_mode: ExecutionMode = "mock_test",
    api_key: SecretStr | None = None,
    limit: int | None = None,
    page_range: tuple[int, int] | None = None,
    max_jobs: int | None = None,
    resume_run: Path | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    draft_factory: Callable[[TranscriptionJob], SourceContentDraft] | None = None,
    provider: GeminiTranscriptionProvider | None = None,
) -> tuple[str, Path]:
    if gemini_concurrency < 1 or gemini_concurrency > 4:
        raise ValueError("gemini_concurrency must be between 1 and 4")
    root = find_project_root()
    model_config = _load_v3_config(root)
    output = ensure_output_outside_repository(output_root, root)
    registration = register_pdf(pdf_path, worker_id=worker_id, source_id=source_id)
    if resume_run is None:
        stamp = now().strftime("%Y%m%dT%H%M%S%fZ")
        run_dir = output / f"transcription-v3-{stamp}-{source_id}-{registration.sha256[:8]}"
        run_dir.mkdir(parents=True, exist_ok=False)
    else:
        run_dir = resume_run.resolve()
        if not run_dir.is_dir():
            raise ValueError("resume_run does not exist")
    preflight_path = run_dir / "pdf_preflight.json"
    pages_path = run_dir / "page_preflight.jsonl"
    if resume_run is not None and preflight_path.is_file() and pages_path.is_file():
        preflight = PdfPreflight.model_validate_json(preflight_path.read_text(encoding="utf-8"))
        pages = [
            PdfPagePreflight.model_validate_json(line)
            for line in pages_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
    else:
        preflight, pages = run_pdf_preflight(
            pdf_path=pdf_path, source_id=source_id, worker_id=worker_id, output_dir=run_dir
        )
    if page_range is not None:
        start_page, end_page = page_range
        if start_page < 1 or end_page < start_page or end_page > preflight.page_count:
            raise ValueError("Invalid page_range")
        pages = [page for page in pages if start_page <= page.page_number <= end_page]
        limit = limit or len(pages)
    provider_evidence: list[ProviderCallEvidence] = []
    if execution_mode == "live":
        if api_key is None and provider is None:
            raise ValueError("Live transcription requires GEMINI_API_KEY")
        if provider is None:
            provider = GeminiTranscriptionProvider(api_key=api_key, model_config=model_config)
        scout_prompt = (root / SCOUT_PROMPT_PATH).read_text(encoding="utf-8")
        scout_draft = provider.scout(registration.resolved_local_path, scout_prompt)
        provider_evidence.extend(provider.evidence)
    elif execution_mode == "mock_test":
        if draft_factory is None:
            draft_factory = _mock_source_content
        scout_draft = ExtractionScoutDraft(
            declared_page_count=preflight.page_count,
            regions=[],
            warnings=["mocked_scout_in_dry_or_test_run"],
        )
    elif execution_mode == "dry_run":
        scout_draft = ExtractionScoutDraft(
            declared_page_count=preflight.page_count,
            regions=[],
            warnings=["dry_run_no_provider_calls"],
        )
    else:
        raise ValueError("Unsupported execution_mode")
    scout = inject_scout_metadata(scout_draft, source_id=source_id)
    scout_path = run_dir / "extraction_scout.json"
    if scout_path.is_file():
        scout = ExtractionScout.model_validate_json(scout_path.read_text(encoding="utf-8"))
    else:
        write_json(scout_path, scout)
    jobs = build_transcription_plan(
        preflight=preflight, pages=pages, scout=scout, planner_mode=planner_mode, limit=limit
    )
    if max_jobs is not None:
        jobs = jobs[:max_jobs]
        limit = limit or sum(len(job.primary_pages) for job in jobs)
    plan_path = run_dir / "extraction_plan.json"
    jobs_path = run_dir / "extraction_jobs.jsonl"
    findings_path = run_dir / "extraction_plan_review_findings.jsonl"
    if not plan_path.is_file():
        write_json(plan_path, {"jobs": [job.model_dump() for job in jobs]})
    if not jobs_path.is_file():
        write_jsonl(jobs_path, jobs)
    if not findings_path.is_file():
        write_jsonl(findings_path, [])
    if execution_mode == "dry_run":
        write_json(
            run_dir / "transcription_manifest.json",
            {
                "status": "dry_run",
                "execution_mode": "dry_run",
                "provider_backend": "none",
                "provider_call_count": 0,
            },
        )
        return "dry_run", run_dir
    contents = [
        _write_job_artifacts(
            run_dir=run_dir,
            pdf_path=registration.resolved_local_path,
            source_id=source_id,
            job=job,
            execution_mode=execution_mode,
            draft_factory=draft_factory,
            provider=provider,
        )
        for job in jobs
    ]
    if provider is not None:
        provider_evidence = provider.evidence
    manifest_path = run_dir / "transcription_manifest.json"
    if resume_run is not None and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        status = str(manifest.get("status") or "completed")
        return status, run_dir
    commit, branch, dirty = git_metadata(root)
    write_json(
        run_dir / "checkpoint.json",
        {
            "status": "jobs_completed",
            "fingerprint": _sha256_json(
                {
                    "source_id": source_id,
                    "pdf_sha256": preflight.pdf_sha256,
                    "planner_mode": planner_mode,
                    "limit": limit,
                }
            ),
            "git_commit": commit,
            "git_branch": branch,
            "dirty_worktree": dirty,
            "secret_free": True,
        },
    )
    status = write_merged_transcript_outputs(
        run_dir=run_dir,
        preflight=preflight,
        page_preflight=pages,
        scout=scout,
        jobs=jobs,
        contents=contents,
        execution_mode=execution_mode,
        provider_evidence=provider_evidence,
        limit=limit,
    )
    return status, run_dir
