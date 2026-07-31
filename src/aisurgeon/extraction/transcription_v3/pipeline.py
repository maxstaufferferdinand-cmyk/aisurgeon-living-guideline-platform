"""Canonical Gemini transcription v3 run preparation and mocked execution."""

import hashlib
import json
import random
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
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
from aisurgeon.extraction.transcription_v3.completeness import (
    split_incomplete_job,
    validate_transcription_completeness,
)
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
TRANSIENT_FAILURE_CATEGORIES = {
    "rate_or_quota",
    "provider_capacity_unavailable",
    "transient_provider_or_network",
}


@dataclass
class DeferredJobFailure:
    job: TranscriptionJob
    status: str
    evidence: ProviderCallEvidence | None
    error: Exception


class TranscriptionJobFailure(RuntimeError):
    def __init__(
        self,
        *,
        job: TranscriptionJob,
        status: str,
        evidence: ProviderCallEvidence | None,
        cause: Exception,
    ) -> None:
        super().__init__(f"Transcription job {job.job_id} failed with status {status}")
        self.job = job
        self.status = status
        self.evidence = evidence
        self.__cause__ = cause


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


def canonicalize_source_content_pages(
    draft: SourceContentDraft, *, job: TranscriptionJob
) -> SourceContentDraft:
    """Map Gemini-returned page labels to original PDF primary pages."""
    original_pages = {entry.original_pdf_page_number for entry in job.slice_page_map}
    slice_to_original = {
        entry.slice_page_index: entry.original_pdf_page_number for entry in job.slice_page_map
    }
    primary_pages = set(job.primary_pages)
    block_page_numbers = [block.page_number for block in draft.visual_blocks]
    unique_block_pages = list(dict.fromkeys(block_page_numbers))

    if set(unique_block_pages).issubset(original_pages):
        page_map = {page: page for page in unique_block_pages}
    elif len(unique_block_pages) == 1 and len(job.primary_pages) == 1:
        page_map = {unique_block_pages[0]: job.primary_pages[0]}
    elif set(unique_block_pages).issubset(slice_to_original):
        page_map = {page: slice_to_original[page] for page in unique_block_pages}
    elif len(unique_block_pages) == len(job.primary_pages):
        page_map = dict(zip(unique_block_pages, job.primary_pages, strict=True))
    elif len(unique_block_pages) == len(job.slice_page_map):
        page_map = {
            page: entry.original_pdf_page_number
            for page, entry in zip(unique_block_pages, job.slice_page_map, strict=True)
        }
    elif len(job.primary_pages) == 1:
        page_map = {page: job.primary_pages[0] for page in unique_block_pages}
    else:
        page_map = {page: page for page in unique_block_pages}

    blocks = []
    for block in draft.visual_blocks:
        mapped_page = page_map.get(block.page_number, block.page_number)
        if mapped_page in primary_pages:
            blocks.append(block.model_copy(update={"page_number": mapped_page}))
    return draft.model_copy(
        update={
            "represented_original_pdf_pages": list(job.primary_pages),
            "visual_blocks": blocks,
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


def _write_json_replace(path: Path, value: Any) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl_replace(path: Path, records: Iterable[Any]) -> None:
    lines = []
    for record in records:
        value = record.model_dump(mode="json") if hasattr(record, "model_dump") else record
        lines.append(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")


def _append_jsonl(path: Path, values: list[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for value in values:
            stream.write(value.model_dump_json() + "\n")


def _prune_invalid_jsonl(path: Path) -> None:
    if not path.is_file():
        return
    valid_lines: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            json.loads(line)
        except json.JSONDecodeError:
            continue
        valid_lines.append(line)
    path.write_text("".join(f"{line}\n" for line in valid_lines), encoding="utf-8")


def _sleep_with_progress(
    *,
    seconds: float,
    sleep: Callable[[float], None],
    message: str,
    progress_interval_seconds: int = 60,
) -> None:
    remaining = max(0.0, seconds)
    if remaining <= 0:
        return
    print(message, flush=True)
    while remaining > 0:
        step = min(remaining, float(progress_interval_seconds))
        sleep(step)
        remaining -= step
        if remaining > 0 and seconds >= progress_interval_seconds:
            print(f"Still waiting: {round(remaining)} seconds remaining.", flush=True)


def _is_transient_evidence(evidence: ProviderCallEvidence | None) -> bool:
    return bool(evidence and evidence.final_failure_category in TRANSIENT_FAILURE_CATEGORIES)


def _write_last_error(
    job_dir: Path,
    *,
    job: TranscriptionJob,
    status: str,
    evidence: ProviderCallEvidence | None,
    exc: Exception,
) -> None:
    _write_json_replace(
        job_dir / "last_error.json",
        {
            "status": status,
            "job_id": job.job_id,
            "primary_pages": job.primary_pages,
            "safe_error_class": type(exc).__name__,
            "safe_error_message": str(exc)[:500],
            "http_status": evidence.http_status if evidence is not None else None,
            "api_status": evidence.api_status if evidence is not None else None,
            "final_failure_category": (
                evidence.final_failure_category if evidence is not None else None
            ),
            "retry_after_seconds": evidence.retry_after_seconds if evidence is not None else None,
        },
    )
    _write_json_replace(job_dir / "checkpoint.json", {"status": status, "job_id": job.job_id})


def _read_job_attempts(run_dir: Path) -> list[ProviderCallEvidence]:
    attempts: list[ProviderCallEvidence] = []
    for path in sorted((run_dir / "transcription_jobs").glob("*/attempts.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                attempts.append(ProviderCallEvidence.model_validate_json(line))
            except ValidationError:
                continue
    return attempts


def _medium_retry_job(job: TranscriptionJob) -> TranscriptionJob:
    return job.model_copy(
        update={
            "job_id": f"{job.job_id}-medium",
            "chunk_id": f"{job.chunk_id}-medium",
            "status": "pending",
            "reason": f"{job.reason}; medium_resolution_retry",
        }
    )


def _can_retry_at_medium(job: TranscriptionJob) -> bool:
    return (
        len(job.primary_pages) == 1
        and "medium_resolution_retry" not in job.reason
        and _media_resolution_for_job(job) == "high"
    )


def _sort_contents_by_page(contents: dict[str, SourceContent]) -> list[SourceContent]:
    return sorted(
        contents.values(),
        key=lambda content: (
            min(content.represented_original_pdf_pages),
            content.job_id,
        ),
    )


def _one_page_repair_job(
    *,
    source_id: str,
    page: int,
    page_count: int,
    profile: str,
    cycle: int,
) -> TranscriptionJob:
    context_pages = [p for p in (page - 1, page + 1) if 1 <= p <= page_count]
    all_pages = sorted({page, *context_pages})
    return TranscriptionJob.model_validate(
        {
            "job_id": f"tx3-repair-c{cycle:02d}-p{page:04d}",
            "chunk_id": f"{source_id}-repair-c{cycle:02d}-p{page:04d}",
            "profile": profile,
            "primary_pages": [page],
            "context_pages": context_pages,
            "slice_page_map": [
                {
                    "slice_page_index": index,
                    "original_pdf_page_number": source_page,
                    "role": (
                        "primary"
                        if source_page == page
                        else "previous_context"
                        if source_page < page
                        else "next_context"
                    ),
                }
                for index, source_page in enumerate(all_pages, start=1)
            ],
            "reason": "targeted completeness repair for unresolved primary page",
        }
    )


def _repair_pages_from_findings(findings: list[CompletenessFinding]) -> set[int]:
    repair_codes = {
        "missing_primary_page",
        "empty_nonblank_primary_page",
    }
    return {
        finding.page_number
        for finding in findings
        if finding.severity == "error"
        and finding.page_number is not None
        and finding.issue_code in repair_codes
    }


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


def _raw_response_snapshot(response: Any) -> dict[str, Any]:
    text = getattr(response, "text", None)
    return {
        "raw_text": text if isinstance(text, str) else None,
        "response_id": getattr(response, "id", None),
        "request_id": getattr(response, "request_id", None),
        "finish_reason": _safe_finish_reason(response),
        "token_usage": _safe_usage(getattr(response, "usage_metadata", None)),
        "parse_status": "unparsed_or_invalid_json",
    }


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
    if "medium_resolution_retry" in job.reason:
        return "medium"
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
            try:
                payload = self._parse_payload(response)
            except Exception:
                if job_id is not None:
                    self.raw_responses[job_id] = _raw_response_snapshot(response)
                raise
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
            try:
                payload = self._parse_payload(response)
            except Exception:
                if job_id is not None:
                    self.raw_responses[job_id] = _raw_response_snapshot(response)
                raise
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
        max_attempts = int(
            self.model_config.get("max_attempts_per_call", self.model_config["max_attempts"])
        )
        defer_after_capacity = int(
            self.model_config.get("defer_after_consecutive_capacity_errors_per_job", max_attempts)
        )
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
                if (
                    decision.final_failure_category
                    in {"provider_capacity_unavailable", "rate_or_quota"}
                    and attempt >= defer_after_capacity
                ):
                    if self.evidence:
                        self.evidence[-1].calculated_delay_seconds = None
                    raise
                if type(exc).__name__ == "JSONDecodeError":
                    if self.evidence:
                        self.evidence[-1].calculated_delay_seconds = None
                    raise
                if decision.category == "non_retryable" or attempt >= max_attempts:
                    if self.evidence:
                        self.evidence[-1].calculated_delay_seconds = None
                    raise
                delay = decision.calculated_delay_seconds or 0
                jitter = self._random_uniform(0, delay * 0.1) if delay else 0
                if self.evidence:
                    self.evidence[-1].calculated_delay_seconds = delay + jitter
                status_label = (
                    f"HTTP {self.evidence[-1].http_status} {self.evidence[-1].api_status}"
                    if self.evidence
                    else "transient Gemini error"
                )
                _sleep_with_progress(
                    seconds=delay + jitter,
                    sleep=self._sleep,
                    message=(
                        f"[Gemini {job.job_id}] {status_label}. "
                        f"Attempt {attempt}/{max_attempts} failed. "
                        f"Next retry in {round(delay + jitter)} seconds."
                    ),
                )
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
    last_error_path = job_dir / "last_error.json"
    if checkpoint_path.is_file():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("status") == "completed" and validated_path.is_file():
            content = SourceContent.model_validate_json(validated_path.read_text(encoding="utf-8"))
            if (
                content.represented_original_pdf_pages == job.primary_pages
                and all(block.page_number in job.primary_pages for block in content.visual_blocks)
            ):
                return content
            if raw_path.is_file():
                draft = canonicalize_source_content_pages(
                    SourceContentDraft.model_validate_json(raw_path.read_text(encoding="utf-8")),
                    job=job,
                )
                content = inject_source_content_metadata(draft, source_id=source_id, job=job)
                _write_json_replace(validated_path, content)
                return content
    for path in (raw_path, validated_path, checkpoint_path):
        if path.exists():
            path.unlink()
    create_pdf_slice(pdf_path, job, job_dir / "slice.pdf")
    _write_json_replace(
        job_dir / "slice_page_map.json",
        [entry.model_dump() for entry in job.slice_page_map],
    )
    (job_dir / "request_prompt.txt").write_text(_job_prompt(job) + "\n", encoding="utf-8")
    _write_json_replace(job_dir / "job_manifest.json", job)
    attempts_path = job_dir / "attempts.jsonl"
    _prune_invalid_jsonl(attempts_path)
    if execution_mode == "live":
        if provider is None:
            raise ValueError("Live transcription requires a Gemini provider")
        before = len(provider.evidence)
        try:
            draft = provider.transcribe(job_dir / "slice.pdf", _job_prompt(job), job)
        except Exception as exc:
            attempts = [
                evidence
                for evidence in provider.evidence[before:]
                if evidence.job_id == job.job_id
            ]
            _append_jsonl(attempts_path, attempts)
            evidence = attempts[-1] if attempts else None
            status = (
                "deferred_transient"
                if _is_transient_evidence(evidence)
                else "failed_nonretryable"
            )
            raw_responses = getattr(provider, "raw_responses", {})
            if job.job_id in raw_responses:
                write_json(raw_path, raw_responses[job.job_id])
            _write_last_error(
                job_dir,
                job=job,
                status=status,
                evidence=evidence,
                exc=exc,
            )
            raise TranscriptionJobFailure(
                job=job,
                status=status,
                evidence=evidence,
                cause=exc,
            ) from exc
        attempts = [
            evidence for evidence in provider.evidence[before:] if evidence.job_id == job.job_id
        ]
        _append_jsonl(attempts_path, attempts)
        draft = canonicalize_source_content_pages(draft, job=job)
        raw_responses = getattr(provider, "raw_responses", {})
        raw_payload = raw_responses.get(job.job_id, draft.model_dump(mode="json"))
    elif execution_mode == "mock_test":
        if draft_factory is None:
            raise ValueError("mock_test requires an explicit draft factory")
        draft = canonicalize_source_content_pages(draft_factory(job), job=job)
        raw_payload = draft.model_dump(mode="json")
    else:
        raise ValueError("dry_run must not write transcription job responses")
    write_json(raw_path, raw_payload)
    content = inject_source_content_metadata(draft, source_id=source_id, job=job)
    write_json(validated_path, content)
    write_json(checkpoint_path, {"status": "completed", "job_id": job.job_id})
    if last_error_path.exists():
        last_error_path.unlink()
    return content


def _run_transcription_job_queue(
    *,
    run_dir: Path,
    pdf_path: Path,
    source_id: str,
    jobs: list[TranscriptionJob],
    execution_mode: ExecutionMode,
    draft_factory: Callable[[TranscriptionJob], SourceContentDraft] | None,
    provider: GeminiTranscriptionProvider | None,
    model_config: dict[str, Any],
) -> tuple[list[SourceContent], list[TranscriptionJob]]:
    contents_by_job: dict[str, SourceContent] = {}
    effective_jobs: dict[str, TranscriptionJob] = {job.job_id: job for job in jobs}
    pending = list(jobs)
    deferred: list[TranscriptionJob] = []
    final_failures: list[DeferredJobFailure] = []
    max_defer_cycles = int(model_config.get("max_defer_cycles", 0))
    cooldown_after = int(
        model_config.get(
            "global_cooldown_after_consecutive_capacity_errors",
            model_config.get("global_cooldown_after_consecutive_transient_failures", 3),
        )
    )
    cooldown_seconds = float(model_config.get("global_cooldown_seconds", 0))
    consecutive_capacity = 0
    cycle = 0
    sleep = getattr(provider, "_sleep", time.sleep) if provider is not None else time.sleep

    while pending:
        next_deferred: list[TranscriptionJob] = []
        for job in pending:
            try:
                content = _write_job_artifacts(
                    run_dir=run_dir,
                    pdf_path=pdf_path,
                    source_id=source_id,
                    job=job,
                    execution_mode=execution_mode,
                    draft_factory=draft_factory,
                    provider=provider,
                )
                contents_by_job[content.job_id] = content
                effective_jobs[job.job_id] = job
                consecutive_capacity = 0
            except TranscriptionJobFailure as exc:
                evidence = exc.evidence
                if not _is_transient_evidence(evidence):
                    final_failures.append(
                        DeferredJobFailure(job=job, status=exc.status, evidence=evidence, error=exc)
                    )
                    continue
                if (
                    evidence is not None
                    and evidence.final_failure_category == "provider_capacity_unavailable"
                ):
                    consecutive_capacity += 1
                    if cooldown_after > 0 and consecutive_capacity >= cooldown_after:
                        _sleep_with_progress(
                            seconds=cooldown_seconds,
                            sleep=sleep,
                            message=(
                                "[Gemini] "
                                f"{consecutive_capacity} consecutive capacity failures. "
                                f"Global cooldown {round(cooldown_seconds)} seconds."
                            ),
                        )
                        consecutive_capacity = 0
                else:
                    consecutive_capacity = 0

                if len(job.primary_pages) > 1 and "-repair-" not in job.job_id:
                    split_jobs = split_incomplete_job(job)
                    effective_jobs.pop(job.job_id, None)
                    effective_jobs.update({split.job_id: split for split in split_jobs})
                    next_deferred.extend(split_jobs)
                elif _can_retry_at_medium(job):
                    medium_job = _medium_retry_job(job)
                    effective_jobs.pop(job.job_id, None)
                    effective_jobs[medium_job.job_id] = medium_job
                    next_deferred.append(medium_job)
                else:
                    next_deferred.append(job)
            except Exception:
                raise
        if final_failures:
            details = ", ".join(
                f"{failure.job.job_id}:{failure.status}" for failure in final_failures
            )
            raise RuntimeError(f"Nonretryable transcription job failure(s): {details}")
        deferred = next_deferred
        if not deferred:
            break
        cycle += 1
        if cycle > max_defer_cycles:
            for job in deferred:
                job_dir = run_dir / "transcription_jobs" / job.job_id
                evidence = None
                attempts = [
                    item for item in _read_job_attempts(run_dir) if item.job_id == job.job_id
                ]
                if attempts:
                    evidence = attempts[-1]
                _write_json_replace(
                    job_dir / "checkpoint.json",
                    {"status": "failed_transient_exhausted", "job_id": job.job_id},
                )
                if not (job_dir / "last_error.json").is_file():
                    write_json(
                        job_dir / "last_error.json",
                        {
                            "status": "failed_transient_exhausted",
                            "job_id": job.job_id,
                            "primary_pages": job.primary_pages,
                            "http_status": evidence.http_status if evidence else None,
                            "api_status": evidence.api_status if evidence else None,
                            "final_failure_category": (
                                evidence.final_failure_category if evidence else None
                            ),
                        },
                    )
            unresolved = ", ".join(job.job_id for job in deferred)
            raise RuntimeError(
                "Transient Gemini transcription failures exhausted after "
                f"{max_defer_cycles} defer cycle(s): {unresolved}"
            )
        _sleep_with_progress(
            seconds=cooldown_seconds,
            sleep=sleep,
            message=(
                f"[Gemini] Revisiting {len(deferred)} deferred transcription job(s) "
                f"after cooldown cycle {cycle}/{max_defer_cycles}."
            ),
        )
        pending = deferred

    return _sort_contents_by_page(contents_by_job), sorted(
        effective_jobs.values(), key=lambda job: (min(job.primary_pages), job.job_id)
    )


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
    _write_jsonl_replace(run_dir / "page_transcript.jsonl", page_rows)
    metrics = _page_metrics(
        page_rows=page_rows,
        page_preflight=page_preflight,
        provider_evidence=provider_evidence,
    )
    _write_json_replace(
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
    _write_jsonl_replace(run_dir / "table_transcripts.jsonl", table_rows)
    _write_jsonl_replace(run_dir / "algorithm_transcripts.jsonl", algorithm_rows)
    _write_jsonl_replace(run_dir / "transcription_uncertainties.jsonl", findings)
    _write_json_replace(
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
    _write_json_replace(
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
    scout_path = run_dir / "extraction_scout.json"
    if execution_mode == "live":
        if api_key is None and provider is None:
            raise ValueError("Live transcription requires GEMINI_API_KEY")
        if provider is None:
            provider = GeminiTranscriptionProvider(api_key=api_key, model_config=model_config)
        if scout_path.is_file():
            scout = ExtractionScout.model_validate_json(scout_path.read_text(encoding="utf-8"))
            scout_draft = ExtractionScoutDraft(
                declared_page_count=scout.declared_page_count,
                regions=scout.regions,
                warnings=scout.warnings,
            )
        else:
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
    contents, effective_jobs = _run_transcription_job_queue(
        run_dir=run_dir,
        pdf_path=registration.resolved_local_path,
        source_id=source_id,
        jobs=jobs,
        execution_mode=execution_mode,
        draft_factory=draft_factory,
        provider=provider,
        model_config=model_config,
    )
    if provider is not None:
        scout_evidence = [item for item in provider.evidence if item.stage == "scout"]
        provider_evidence = [*scout_evidence, *_read_job_attempts(run_dir)]
    if execution_mode == "live":
        profile_by_page = {
            page: job.profile for job in effective_jobs for page in job.primary_pages
        }
        max_repair_cycles = int(model_config.get("max_targeted_repair_cycles", 2))
        for cycle in range(1, max_repair_cycles + 1):
            findings = validate_transcription_completeness(
                jobs=effective_jobs,
                contents=contents,
                page_preflight=pages,
                execution_mode=execution_mode,
                provider_evidence=provider_evidence,
            )
            repair_pages = sorted(_repair_pages_from_findings(findings))
            if not repair_pages:
                break
            repair_jobs = [
                _one_page_repair_job(
                    source_id=source_id,
                    page=page,
                    page_count=preflight.page_count,
                    profile=profile_by_page.get(page, "dense_prose_verbatim"),
                    cycle=cycle,
                )
                for page in repair_pages
            ]
            print(
                f"[Gemini] Targeted completeness repair cycle {cycle}/{max_repair_cycles}: "
                f"pages {repair_pages}",
                flush=True,
            )
            repair_contents, repair_effective_jobs = _run_transcription_job_queue(
                run_dir=run_dir,
                pdf_path=registration.resolved_local_path,
                source_id=source_id,
                jobs=repair_jobs,
                execution_mode=execution_mode,
                draft_factory=draft_factory,
                provider=provider,
                model_config=model_config,
            )
            existing_by_job = {content.job_id: content for content in contents}
            existing_by_job.update({content.job_id: content for content in repair_contents})
            contents = _sort_contents_by_page(existing_by_job)
            effective_by_job = {job.job_id: job for job in effective_jobs}
            effective_by_job.update({job.job_id: job for job in repair_effective_jobs})
            effective_jobs = sorted(
                effective_by_job.values(), key=lambda job: (min(job.primary_pages), job.job_id)
            )
            provider_evidence = [*scout_evidence, *_read_job_attempts(run_dir)]
    manifest_path = run_dir / "transcription_manifest.json"
    if resume_run is not None and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        status = str(manifest.get("status") or "completed")
        if status in {"completed", "completed_with_review", "technical_limited"}:
            return status, run_dir
    commit, branch, dirty = git_metadata(root)
    _write_json_replace(
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
        jobs=effective_jobs,
        contents=contents,
        execution_mode=execution_mode,
        provider_evidence=provider_evidence,
        limit=limit,
    )
    return status, run_dir
