"""Targeted original-bibliography repair followed by dual-namespace DOCX rebuild."""

import json
import random
import shutil
import time
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

from openpyxl import Workbook
from pydantic import BaseModel, Field, SecretStr, ValidationError
from pypdf import PdfReader, PdfWriter

from aisurgeon.extraction.canonical.outputs import write_json
from aisurgeon.extraction.gemini.client import GeminiDocumentMapClient
from aisurgeon.extraction.gemini.errors import GeminiError, GeminiResponseValidationError
from aisurgeon.extraction.gemini.models import GeminiModelConfig
from aisurgeon.search.pubmed.generation import ensure_external_run_root, file_hash, load_jsonl
from aisurgeon.search.pubmed.query import sha256_text
from aisurgeon.synthesis.reference_rebuild import (
    _reference_number,
    find_old_citation_occurrences,
    rebuild_guideline_references,
)
from aisurgeon.synthesis.updated_guideline import _git_commit

REFERENCE_REPAIR_SCHEMA_VERSION = "original_reference_repair_v1"
REFERENCE_REPAIR_PROMPT_VERSION = "gemini_original_reference_repair_v1"
REFERENCE_REPAIR_VERSION = "targeted_original_reference_repair_v1"
REPAIR_PROMPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "config/prompts/gemini_original_reference_repair_v1.txt"
)
REFERENCE_REPAIR_V2_SCHEMA_VERSION = "original_reference_repair_v2"
REFERENCE_REPAIR_V2_PROMPT_VERSION = "gemini_original_reference_repair_v2"
REFERENCE_REPAIR_V2_VERSION = "targeted_original_reference_repair_v2"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPAIR_V2_PROMPT_PATH = PROJECT_ROOT / "config/prompts/gemini_original_reference_repair_v2.txt"
REPAIR_V2_MODEL_CONFIG_PATH = (
    PROJECT_ROOT / "config/models/gemini_original_reference_repair_v2.json"
)


class OriginalReferenceRepairEntry(BaseModel):
    schema_version: Literal["original_reference_repair_v1"] = REFERENCE_REPAIR_SCHEMA_VERSION
    source_id: str
    original_reference_number: str = Field(min_length=1)
    exact_reference_text: str = Field(min_length=1)
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    column_start: Literal["left", "right", "full_width"]
    column_end: Literal["left", "right", "full_width"]
    continuation_detected: bool
    extraction_confidence: float = Field(ge=0, le=1)
    review_required: bool
    review_notes: list[str] = Field(default_factory=list)


class OriginalReferenceRepairBatch(BaseModel):
    schema_version: Literal["original_reference_repair_v1"] = REFERENCE_REPAIR_SCHEMA_VERSION
    source_id: str
    references: list[OriginalReferenceRepairEntry] = Field(default_factory=list)


class GeminiOriginalReferenceRepairClient(GeminiDocumentMapClient):
    """Small Gemini boundary for bibliography-only repair."""

    def upload_pdf(self, pdf_path: Path) -> Any:
        remote = self._with_retry(
            lambda: self._client.files.upload(
                file=pdf_path, config={"mime_type": "application/pdf"}
            )
        )
        remote = self.wait_until_active(remote)
        return remote

    @staticmethod
    def _repair_schema() -> dict[str, Any]:
        schema = GeminiDocumentMapClient.request_schema(OriginalReferenceRepairBatch)

        def remove_defaults(value: Any) -> None:
            if isinstance(value, dict):
                for key in ("schema_version", "source_id"):
                    value.get("properties", {}).pop(key, None)
                    if isinstance(value.get("required"), list) and key in value["required"]:
                        value["required"].remove(key)
                for child in value.values():
                    remove_defaults(child)
            elif isinstance(value, list):
                for child in value:
                    remove_defaults(child)

        remove_defaults(schema)
        return schema

    def request_repair(
        self,
        *,
        remote: Any,
        prompt: str,
        source_id: str,
    ) -> tuple[OriginalReferenceRepairBatch, str, dict[str, int] | None]:
        response = self._with_retry(
            lambda: self._client.models.generate_content(
                model=self._model_config.model_id,
                contents=[
                    {
                        "file_data": {
                            "file_uri": getattr(remote, "uri", None),
                            "mime_type": getattr(remote, "mime_type", "application/pdf"),
                        }
                    },
                    {"text": prompt},
                ],
                config={
                    "thinking_config": {
                        "thinking_level": self._model_config.thinking_level.upper(),
                    },
                    "media_resolution": (
                        f"MEDIA_RESOLUTION_{self._model_config.media_resolution.upper()}"
                    ),
                    "response_mime_type": "application/json",
                    "response_json_schema": self._repair_schema(),
                    "http_options": {
                        "timeout": self._model_config.request_timeout_seconds * 1000,
                    },
                },
            )
        )
        raw = getattr(response, "text", None)
        if not isinstance(raw, str):
            raise GeminiResponseValidationError("Gemini-Antwort enthält kein JSON.")
        try:
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise GeminiResponseValidationError("Gemini-Antwort ist kein JSON-Objekt.")
            payload["schema_version"] = REFERENCE_REPAIR_SCHEMA_VERSION
            payload["source_id"] = source_id
            for entry in payload.get("references", []):
                if isinstance(entry, dict):
                    entry["schema_version"] = REFERENCE_REPAIR_SCHEMA_VERSION
                    entry["source_id"] = source_id
            batch = OriginalReferenceRepairBatch.model_validate(payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise GeminiResponseValidationError(
                "Gemini-Referenzrepair entspricht nicht dem Schema."
            ) from exc
        usage = self.normalize_usage(getattr(response, "usage_metadata", None))
        return batch, raw, usage

    def delete_remote(self, remote: Any) -> bool:
        try:
            self._client.files.delete(name=remote.name)
            return True
        except Exception:
            return False


def _write_xlsx(path: Path, sheet_name: str, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        path.unlink()
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = sheet_name[:31]
    headers = sorted({key for row in rows for key in row}) or ["finding_id"]
    sheet.append(headers)
    for row in rows:
        sheet.append(
            [
                json.dumps(row.get(header), ensure_ascii=False)
                if isinstance(row.get(header), (list, dict))
                else row.get(header)
                for header in headers
            ]
        )
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = f"A1:{sheet.cell(1, len(headers)).coordinate}"
    workbook.save(path)


def _write_json_replace(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _write_jsonl_replace(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    tmp.replace(path)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _repair_reference_id(source_id: str, number: str, text: str) -> str:
    digest = sha256_text(f"{source_id}\x1f{number}\x1f{text}")[:12]
    return f"{source_id}_REF_{digest}"


def _normalize_reference_text(text: str) -> str:
    return " ".join(str(text).split())


def bibliography_pages_from_document_map(document_map: dict[str, Any]) -> dict[str, Any]:
    ranges = document_map.get("bibliography_page_ranges") or []
    if not ranges:
        raise ValueError("Document Map enthält keine Bibliographieseiten.")
    page_start = min(int(row["page_start"]) for row in ranges)
    page_end = max(int(row["page_end"]) for row in ranges)
    return {
        "source": "document_map.validated.json",
        "page_start": page_start,
        "page_end": page_end,
        "page_ranges": ranges,
    }


def original_reference_requirements(
    *, synthesis_run: Path, existing_references: list[dict[str, Any]]
) -> dict[str, Any]:
    blocks = load_jsonl(synthesis_run / "updated_guideline_blocks.jsonl")
    existing_numbers = {str(row["original_reference_number"]) for row in existing_references}
    occurrences, missing = find_old_citation_occurrences(blocks, existing_numbers)
    cited = sorted(
        {number for row in occurrences for number in row["resolved_reference_numbers"]},
        key=int,
    )
    return {
        "cited_original_reference_numbers": cited,
        "existing_original_reference_numbers": sorted(existing_numbers, key=int),
        "missing_original_reference_numbers": missing,
        "old_citation_occurrence_count": len(occurrences),
        "old_citation_occurrences": occurrences,
    }


def _is_obviously_incomplete(old_text: str, repaired_text: str) -> bool:
    old_norm = _normalize_reference_text(old_text)
    repaired_norm = _normalize_reference_text(repaired_text)
    if len(old_norm) < 40 and repaired_norm.startswith(old_norm):
        return True
    if len(old_norm) < len(repaired_norm) * 0.65 and repaired_norm.startswith(old_norm[:25]):
        return True
    return old_norm.endswith((",", ";", ":"))


def merge_repaired_references(
    *,
    source_id: str,
    existing_references: list[dict[str, Any]],
    repaired_entries: list[dict[str, Any]],
    required_numbers: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    findings: list[dict[str, Any]] = []
    merge_report = {
        "existing_reference_count": len(existing_references),
        "required_reference_count": len(required_numbers),
        "added_reference_numbers": [],
        "replaced_reference_numbers": [],
        "kept_existing_reference_numbers": [],
        "conflicting_reference_numbers": [],
    }
    existing_by_number: dict[str, dict[str, Any]] = {}
    for ref in existing_references:
        number = str(ref["original_reference_number"])
        prior = existing_by_number.get(number)
        if prior and prior["exact_original_reference_text"] != ref["exact_original_reference_text"]:
            findings.append(
                {
                    "finding_id": f"DUPLICATE_EXISTING_REF_{number}",
                    "severity": "error",
                    "issue_code": "duplicate_existing_reference_conflict",
                    "reference_number": number,
                }
            )
        existing_by_number[number] = ref
    repaired_by_number: dict[str, dict[str, Any]] = {}
    for entry in repaired_entries:
        number = str(entry["original_reference_number"])
        prior = repaired_by_number.get(number)
        if prior and prior["exact_reference_text"] != entry["exact_reference_text"]:
            findings.append(
                {
                    "finding_id": f"DUPLICATE_REPAIRED_REF_{number}",
                    "severity": "error",
                    "issue_code": "duplicate_repaired_reference_conflict",
                    "reference_number": number,
                }
            )
        repaired_by_number[number] = entry
    merged: dict[str, dict[str, Any]] = dict(existing_by_number)
    for number, entry in repaired_by_number.items():
        repaired_text = entry["exact_reference_text"]
        existing = merged.get(number)
        if existing is None:
            merged[number] = {
                "schema_version": "canonical_extraction_v2",
                "source_id": source_id,
                "page_start": entry["page_start"],
                "page_end": entry["page_end"],
                "extraction_confidence": entry["extraction_confidence"],
                "review_required": entry["review_required"],
                "review_reasons": entry["review_notes"],
                "reference_id": _repair_reference_id(source_id, number, repaired_text),
                "original_reference_number": number,
                "exact_original_reference_text": repaired_text,
            }
            merge_report["added_reference_numbers"].append(number)
            continue
        old_text = existing["exact_original_reference_text"]
        if _normalize_reference_text(old_text) == _normalize_reference_text(repaired_text):
            merge_report["kept_existing_reference_numbers"].append(number)
            continue
        if _is_obviously_incomplete(old_text, repaired_text):
            replacement = dict(existing)
            replacement.update(
                {
                    "page_start": entry["page_start"],
                    "page_end": entry["page_end"],
                    "extraction_confidence": entry["extraction_confidence"],
                    "review_required": True,
                    "review_reasons": [
                        *existing.get("review_reasons", []),
                        "targeted_reference_repair_replaced_incomplete_entry",
                    ],
                    "exact_original_reference_text": repaired_text,
                }
            )
            merged[number] = replacement
            merge_report["replaced_reference_numbers"].append(number)
        else:
            merge_report["conflicting_reference_numbers"].append(number)
            findings.append(
                {
                    "finding_id": f"CONFLICTING_REPAIRED_REF_{number}",
                    "severity": "error",
                    "issue_code": "conflicting_repaired_reference",
                    "reference_number": number,
                    "message": "Vorhandene und reparierte Referenz widersprechen sich.",
                }
            )
    missing_after = [number for number in required_numbers if number not in merged]
    for number in missing_after:
        findings.append(
            {
                "finding_id": f"STILL_MISSING_REF_{number}",
                "severity": "error",
                "issue_code": "missing_original_reference_after_repair",
                "reference_number": number,
            }
        )
    merge_report["missing_after_repair"] = missing_after
    merge_report["final_reference_count"] = len(merged)
    result = sorted(
        merged.values(), key=lambda row: _reference_number(row["original_reference_number"])
    )
    return result, merge_report, findings


def _repair_prompt(
    *,
    base_prompt: str,
    source_id: str,
    page_start: int,
    page_end: int,
    missing_numbers: list[str],
) -> str:
    return (
        f"source_id: {source_id}\n"
        f"Target bibliography pages: {page_start}-{page_end}\n"
        f"Missing cited original reference numbers to prioritize: {', '.join(missing_numbers)}\n\n"
        f"{base_prompt}"
    )


def run_reference_repair_and_rebuild(
    *,
    pdf: Path,
    extraction_run: Path,
    synthesis_run: Path,
    failed_reference_run: Path,
    output_root: Path,
    api_key: SecretStr,
    resume_run: Path | None = None,
    client_factory: Callable[..., GeminiOriginalReferenceRepairClient] = (
        GeminiOriginalReferenceRepairClient
    ),
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> Path:
    if not pdf.is_file():
        raise ValueError("PDF nicht gefunden.")
    failed_manifest = _load_json(failed_reference_run / "reference_rebuild_manifest.json")
    if failed_manifest.get("status") != "failed":
        raise ValueError("Failed Reference-Rebuild-Run ist nicht failed.")
    extraction_manifest = _load_json(extraction_run / "extraction_manifest.json")
    synthesis_manifest = _load_json(synthesis_run / "synthesis_manifest.json")
    source_id = synthesis_manifest["source_id"]
    if extraction_manifest["source_id"] != source_id:
        raise ValueError("source_id zwischen Extraction und Synthesis stimmt nicht überein.")
    document_map = _load_json(extraction_run / "document_map.validated.json")
    target_pages = bibliography_pages_from_document_map(document_map)
    existing_references = load_jsonl(extraction_run / "references.jsonl")
    requirements = original_reference_requirements(
        synthesis_run=synthesis_run, existing_references=existing_references
    )
    missing_numbers = requirements["missing_original_reference_numbers"]
    prompt = REPAIR_PROMPT_PATH.read_text(encoding="utf-8")
    config = GeminiModelConfig(
        provider="google",
        api="generate_content",
        model_id="gemini-3.5-flash",
        thinking_level="medium",
        media_resolution="high",
        prompt_version="gemini_document_map_v1",
        schema_version="document_map_v1",
        request_timeout_seconds=1200,
        max_attempts=3,
    )
    prompt_hash = sha256_text(prompt)
    fingerprint = {
        "source_id": source_id,
        "repair_version": REFERENCE_REPAIR_VERSION,
        "repair_schema_version": REFERENCE_REPAIR_SCHEMA_VERSION,
        "repair_prompt_version": REFERENCE_REPAIR_PROMPT_VERSION,
        "repair_prompt_hash": prompt_hash,
        "gemini_model_id": config.model_id,
        "gemini_config": config.model_dump(),
        "gemini_api_key_present": bool(api_key.get_secret_value()),
        "pdf": str(pdf.resolve()),
        "pdf_hash": file_hash(pdf),
        "extraction_run": str(extraction_run.resolve()),
        "synthesis_run": str(synthesis_run.resolve()),
        "failed_reference_run": str(failed_reference_run.resolve()),
        "input_hashes": {
            str((extraction_run / "references.jsonl").resolve()): file_hash(
                extraction_run / "references.jsonl"
            ),
            str((extraction_run / "document_map.validated.json").resolve()): file_hash(
                extraction_run / "document_map.validated.json"
            ),
            str((synthesis_run / "updated_guideline_blocks.jsonl").resolve()): file_hash(
                synthesis_run / "updated_guideline_blocks.jsonl"
            ),
            str((failed_reference_run / "reference_rebuild_manifest.json").resolve()): file_hash(
                failed_reference_run / "reference_rebuild_manifest.json"
            ),
        },
        "targeted_pages": target_pages,
        "git_commit": _git_commit(),
    }
    if resume_run:
        run_dir = resume_run.resolve()
        if _load_json(run_dir / "checkpoint_fingerprint.json") != fingerprint:
            raise ValueError("Resume fingerprint does not match")
        manifest_path = run_dir / "reference_repair_manifest.json"
        if manifest_path.is_file() and _load_json(manifest_path).get("status") in {
            "completed",
            "completed_with_review",
        }:
            return run_dir
    else:
        root = ensure_external_run_root(output_root, synthesis_run)
        run_dir = root / (
            f"reference-repair-{now():%Y%m%dT%H%M%S%fZ}-{source_id}-"
            f"{sha256_text(json.dumps(fingerprint, sort_keys=True))[:8]}"
        )
        run_dir.mkdir(parents=True, exist_ok=False)
        write_json(run_dir / "checkpoint_fingerprint.json", fingerprint)
    _write_json_replace(run_dir / "original_reference_requirements.json", requirements)
    _write_jsonl_replace(
        run_dir / "missing_original_references.jsonl",
        [{"original_reference_number": number} for number in missing_numbers],
    )
    _write_json_replace(run_dir / "targeted_reference_pages.json", target_pages)
    gateway = client_factory(api_key=api_key, model_config=config)
    remote = None
    remote_deleted = False
    token_usage = None
    findings: list[dict[str, Any]] = []
    try:
        remote = gateway.upload_pdf(pdf)
        batch, raw, token_usage = gateway.request_repair(
            remote=remote,
            prompt=_repair_prompt(
                base_prompt=prompt,
                source_id=source_id,
                page_start=target_pages["page_start"],
                page_end=target_pages["page_end"],
                missing_numbers=missing_numbers,
            ),
            source_id=source_id,
        )
        _write_json_replace(run_dir / "gemini_reference_repair.raw.json", json.loads(raw))
    except GeminiError as exc:
        if remote is not None:
            remote_deleted = gateway.delete_remote(remote)
            remote = None
        findings.append(
            {
                "finding_id": "GEMINI_REFERENCE_REPAIR_FAILED",
                "severity": "error",
                "issue_code": "gemini_reference_repair_failed",
                "message": str(exc),
            }
        )
        _write_jsonl_replace(run_dir / "reference_repair_findings.jsonl", findings)
        _write_xlsx(run_dir / "reference_repair_findings.xlsx", "repair_findings", findings)
        _write_json_replace(
            run_dir / "reference_repair_manifest.json",
            {
                **fingerprint,
                "created_at": now().isoformat(),
                "status": "failed",
                "existing_reference_count": len(existing_references),
                "missing_reference_count": len(missing_numbers),
                "added_reference_count": 0,
                "replaced_reference_count": 0,
                "final_repaired_reference_count": 0,
                "reference_rebuild_run": None,
                "remote_file_deleted": remote_deleted,
                "token_usage": token_usage,
                "output_files": {
                    p.name: file_hash(p)
                    for p in run_dir.iterdir()
                    if p.is_file() and p.name != "reference_repair_manifest.json"
                },
            },
        )
        raise RuntimeError(f"Reference repair failed; run directory: {run_dir}") from exc
    finally:
        if remote is not None:
            remote_deleted = gateway.delete_remote(remote)
    repaired_entries = [entry.model_dump() for entry in batch.references]
    repaired_refs, merge_report, findings = merge_repaired_references(
        source_id=source_id,
        existing_references=existing_references,
        repaired_entries=repaired_entries,
        required_numbers=requirements["cited_original_reference_numbers"],
    )
    _write_jsonl_replace(run_dir / "repaired_original_references.jsonl", repaired_refs)
    _write_json_replace(run_dir / "repaired_references_merge_report.json", merge_report)
    _write_jsonl_replace(run_dir / "reference_repair_findings.jsonl", findings)
    _write_xlsx(run_dir / "reference_repair_findings.xlsx", "repair_findings", findings)
    status = "failed" if any(row.get("severity") == "error" for row in findings) else "completed"
    reference_rebuild_run = None
    if status != "failed":
        try:
            reference_rebuild_run = rebuild_guideline_references(
                synthesis_run=synthesis_run,
                output_root=run_dir,
                output_name=(
                    "AISurgeon_Aktualisierte_Leitlinie_GERD_EoE_2026_"
                    "references_repaired.docx"
                ),
                original_references_path=run_dir / "repaired_original_references.jsonl",
                now=now,
            )
            rebuild_manifest = _load_json(reference_rebuild_run / "reference_rebuild_manifest.json")
            status = rebuild_manifest["status"]
        except RuntimeError as exc:
            status = "failed"
            findings.append(
                {
                    "finding_id": "DUAL_REBUILD_FAILED",
                    "severity": "error",
                    "issue_code": "dual_namespace_rebuild_failed",
                    "message": str(exc),
                }
            )
            _write_jsonl_replace(run_dir / "reference_repair_findings.jsonl", findings)
            _write_xlsx(run_dir / "reference_repair_findings.xlsx", "repair_findings", findings)
    manifest = {
        **fingerprint,
        "created_at": now().isoformat(),
        "status": status,
        "existing_reference_count": len(existing_references),
        "missing_reference_count": len(missing_numbers),
        "added_reference_count": len(merge_report["added_reference_numbers"]),
        "replaced_reference_count": len(merge_report["replaced_reference_numbers"]),
        "final_repaired_reference_count": len(repaired_refs),
        "reference_rebuild_run": str(reference_rebuild_run) if reference_rebuild_run else None,
        "remote_file_deleted": remote_deleted,
        "token_usage": token_usage,
        "output_files": {
            p.name: file_hash(p)
            for p in run_dir.iterdir()
            if p.is_file() and p.name != "reference_repair_manifest.json"
        },
    }
    _write_json_replace(run_dir / "reference_repair_manifest.json", manifest)
    if status == "failed":
        raise RuntimeError(f"Reference repair failed; run directory: {run_dir}")
    return run_dir


class OriginalReferenceRepairV2Reference(BaseModel):
    original_reference_number: int = Field(ge=1)
    exact_reference_text: str = Field(min_length=1)
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    column_start: Literal["left", "right", "full_width"]
    column_end: Literal["left", "right", "full_width"]
    continuation_from_previous_page: bool
    continuation_to_next_page: bool
    extraction_confidence: float = Field(ge=0, le=1)
    review_required: bool
    review_notes: list[str] = Field(default_factory=list)


class OriginalReferenceRepairV2PageJobOutput(BaseModel):
    schema_version: Literal["original_reference_repair_v2"] = REFERENCE_REPAIR_V2_SCHEMA_VERSION
    source_id: str
    job_id: str
    primary_original_pdf_page: int = Field(ge=1)
    context_original_pdf_pages: list[int] = Field(default_factory=list)
    first_reference_number_observed: int | None = Field(default=None, ge=1)
    last_reference_number_observed: int | None = Field(default=None, ge=1)
    observed_reference_numbers: list[int] = Field(default_factory=list)
    references: list[OriginalReferenceRepairV2Reference] = Field(default_factory=list)
    page_complete: bool
    review_required: bool
    review_notes: list[str] = Field(default_factory=list)


SCHEMA_VERSION_ALIASES_V2 = {
    "2.0.0",
    "2.0",
    "v2",
    "2",
    "original-reference-repair-v2",
}


class PageJobValidationFailed(RuntimeError):
    def __init__(
        self,
        *,
        message: str,
        raw_payload: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        validation_errors: Any = None,
    ) -> None:
        super().__init__(message)
        self.raw_payload = raw_payload
        self.metadata = metadata or {}
        self.validation_errors = validation_errors


def normalize_page_job_schema_version_v2(payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("schema_version")
    if raw is None or raw == REFERENCE_REPAIR_V2_SCHEMA_VERSION:
        return {
            "raw_schema_version": raw,
            "normalized_schema_version": raw,
            "schema_version_normalized": False,
        }
    if raw in SCHEMA_VERSION_ALIASES_V2:
        payload["schema_version"] = REFERENCE_REPAIR_V2_SCHEMA_VERSION
        return {
            "raw_schema_version": raw,
            "normalized_schema_version": REFERENCE_REPAIR_V2_SCHEMA_VERSION,
            "schema_version_normalized": True,
        }
    raise ValueError(f"Unknown original_reference_repair_v2 schema_version: {raw}")


def parse_page_job_raw_response_v2(
    raw_payload: dict[str, Any],
) -> tuple[OriginalReferenceRepairV2PageJobOutput, dict[str, Any]]:
    payload = dict(raw_payload)
    normalization = normalize_page_job_schema_version_v2(payload)
    try:
        return OriginalReferenceRepairV2PageJobOutput.model_validate(payload), normalization
    except ValidationError as exc:
        raise PageJobValidationFailed(
            message="OriginalReferenceRepairV2PageJobOutput validation failed.",
            raw_payload=raw_payload,
            metadata={"schema_version_normalization": normalization},
            validation_errors=exc.errors(),
        ) from exc


def load_reference_repair_v2_model_config() -> dict[str, Any]:
    config = _load_json(REPAIR_V2_MODEL_CONFIG_PATH)
    if config.get("repair_prompt_version") != REFERENCE_REPAIR_V2_PROMPT_VERSION:
        raise ValueError("Repair-v2 model config has wrong prompt version.")
    if config.get("repair_schema_version") != REFERENCE_REPAIR_V2_SCHEMA_VERSION:
        raise ValueError("Repair-v2 model config has wrong schema version.")
    required_retry_fields = {
        "request_timeout_seconds": 1800,
        "max_attempts": 8,
        "retry_initial_delay_seconds": 15,
        "retry_backoff_multiplier": 2.0,
        "retry_max_delay_seconds": 900,
        "retry_jitter_fraction": 0.25,
        "global_cooldown_after_consecutive_transient_failures": 3,
        "global_cooldown_seconds": 900,
    }
    for key, expected in required_retry_fields.items():
        if config.get(key) != expected:
            raise ValueError(f"Repair-v2 model config has wrong retry field: {key}.")
    serialized = json.dumps(config, sort_keys=True)
    if "document_map" in serialized:
        raise ValueError("Document-map versions are forbidden in Reference Repair v2 config.")
    return config


def reject_document_map_versions_in_repair_fingerprint(fingerprint: dict[str, Any]) -> None:
    serialized = json.dumps(fingerprint, sort_keys=True)
    if "gemini_document_map_v1" in serialized or "document_map_v1" in serialized:
        raise ValueError("Document-map prompt/schema detected in Reference Repair v2 fingerprint.")


def _compatible_raw_import_fingerprint(
    old_fingerprint: dict[str, Any], new_fingerprint: dict[str, Any]
) -> bool:
    keys = (
        "source_id",
        "repair_version",
        "repair_prompt_version",
        "repair_schema_version",
        "pdf_hash",
        "page_plan",
    )
    return all(old_fingerprint.get(key) == new_fingerprint.get(key) for key in keys)


def bibliography_page_plan_v2(
    *, document_map: dict[str, Any], pdf_page_count: int, pages_per_job: int, context_pages: int
) -> dict[str, Any]:
    if pages_per_job != 1:
        raise ValueError("Reference Repair v2 currently requires pages_per_job=1.")
    pages = bibliography_pages_from_document_map(document_map)
    start = int(pages["page_start"])
    end = int(pages["page_end"])
    if start < 1 or end > pdf_page_count or start > end:
        raise ValueError("Document Map bibliography pages are incompatible with physical PDF.")
    jobs = []
    for page in range(start, end + 1):
        slice_pages = []
        for candidate in range(page - context_pages, page + context_pages + 1):
            if start <= candidate <= end:
                role = (
                    "primary"
                    if candidate == page
                    else ("previous_context" if candidate < page else "next_context")
                )
                slice_pages.append(
                    {
                        "slice_page_index": len(slice_pages) + 1,
                        "original_pdf_page_number": candidate,
                        "role": role,
                    }
                )
        jobs.append(
            {
                "job_id": f"page_{page:04d}",
                "primary_original_pdf_page": page,
                "context_original_pdf_pages": [
                    row["original_pdf_page_number"]
                    for row in slice_pages
                    if row["role"] != "primary"
                ],
                "slice_page_map": slice_pages,
            }
        )
    return {
        "source": "document_map.validated.json",
        "page_conversion_rule": "document_map_pages_are_one_based_physical_pdf_pages",
        "bibliography_page_start": start,
        "bibliography_page_end": end,
        "pages_per_job": pages_per_job,
        "context_pages": context_pages,
        "job_count": len(jobs),
        "jobs": jobs,
    }


def write_pdf_slice(pdf: Path, slice_page_map: list[dict[str, Any]], output_path: Path) -> None:
    reader = PdfReader(str(pdf))
    writer = PdfWriter()
    for row in slice_page_map:
        zero_based = int(row["original_pdf_page_number"]) - 1
        writer.add_page(reader.pages[zero_based])
    with output_path.open("wb") as stream:
        writer.write(stream)


def page_job_fingerprint_v2(
    *,
    source_id: str,
    pdf_hash: str,
    job: dict[str, Any],
    slice_pdf_hash: str,
    model_config: dict[str, Any],
    prompt_hash: str,
) -> dict[str, Any]:
    fingerprint = {
        "source_id": source_id,
        "repair_version": REFERENCE_REPAIR_V2_VERSION,
        "repair_prompt_version": REFERENCE_REPAIR_V2_PROMPT_VERSION,
        "repair_schema_version": REFERENCE_REPAIR_V2_SCHEMA_VERSION,
        "pdf_sha256": pdf_hash,
        "primary_original_pdf_page": job["primary_original_pdf_page"],
        "context_original_pdf_pages": job["context_original_pdf_pages"],
        "slice_page_map": job["slice_page_map"],
        "slice_pdf_sha256": slice_pdf_hash,
        "model_id": model_config["model_id"],
        "model_config": model_config,
        "model_config_hash": sha256_text(json.dumps(model_config, sort_keys=True)),
        "prompt_hash": prompt_hash,
        "page_conversion_rule": "document_map_pages_are_one_based_physical_pdf_pages",
    }
    reject_document_map_versions_in_repair_fingerprint(fingerprint)
    return fingerprint


def validate_page_job_output_v2(
    output: OriginalReferenceRepairV2PageJobOutput,
) -> tuple[str, list[dict[str, Any]]]:
    findings: list[dict[str, Any]] = []
    if not output.page_complete:
        findings.append(
            {
                "severity": "error",
                "issue_code": "page_incomplete",
                "job_id": output.job_id,
            }
        )
    if not output.references:
        findings.append(
            {"severity": "error", "issue_code": "empty_bibliography_page", "job_id": output.job_id}
        )
    observed = output.observed_reference_numbers
    reference_numbers = [ref.original_reference_number for ref in output.references]
    if sorted(set(observed)) != sorted(set(reference_numbers)):
        findings.append(
            {
                "severity": "error",
                "issue_code": "observed_reference_numbers_mismatch",
                "job_id": output.job_id,
                "observed": observed,
                "references": reference_numbers,
            }
        )
    texts_by_number: dict[int, str] = {}
    for ref in output.references:
        prior = texts_by_number.get(ref.original_reference_number)
        if prior is not None and prior != ref.exact_reference_text:
            findings.append(
                {
                    "severity": "error",
                    "issue_code": "duplicate_reference_conflict_in_page_job",
                    "job_id": output.job_id,
                    "reference_number": ref.original_reference_number,
                }
            )
        texts_by_number[ref.original_reference_number] = ref.exact_reference_text
    if reference_numbers != sorted(reference_numbers):
        findings.append(
            {
                "severity": "error",
                "issue_code": "non_monotonic_reference_numbers",
                "job_id": output.job_id,
            }
        )
    if observed:
        expected = set(range(min(observed), max(observed) + 1))
        missing = sorted(expected - set(observed))
        if missing:
            findings.append(
                {
                    "severity": "warning",
                    "issue_code": "visible_sequence_gap",
                    "job_id": output.job_id,
                    "missing_visible_reference_numbers": missing,
                }
            )
    if output.first_reference_number_observed != (min(observed) if observed else None):
        findings.append(
            {
                "severity": "error",
                "issue_code": "first_observed_reference_number_mismatch",
                "job_id": output.job_id,
            }
        )
    if output.last_reference_number_observed != (max(observed) if observed else None):
        findings.append(
            {
                "severity": "error",
                "issue_code": "last_observed_reference_number_mismatch",
                "job_id": output.job_id,
            }
        )
    status = (
        "failed"
        if any(row["severity"] == "error" for row in findings)
        else ("completed_with_review" if findings or output.review_required else "completed")
    )
    return status, findings


def reject_bulk_partial_reference_response(
    raw_payload: dict[str, Any], *, expected_total: int = 720
) -> None:
    count = len(raw_payload.get("references", []))
    if count < expected_total:
        raise ValueError(
            f"Partial bibliography response rejected: {count} of {expected_total} references."
        )


def stitch_page_job_references_v2(
    page_outputs: list[OriginalReferenceRepairV2PageJobOutput],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    findings: list[dict[str, Any]] = []
    parts_by_number: dict[int, list[OriginalReferenceRepairV2Reference]] = {}
    for output in sorted(page_outputs, key=lambda row: row.primary_original_pdf_page):
        for ref in output.references:
            parts_by_number.setdefault(ref.original_reference_number, []).append(ref)
    stitched = []
    for number, parts in sorted(parts_by_number.items()):
        ordered = sorted(parts, key=lambda ref: (ref.page_start, ref.page_end))
        unique_texts = list(
            dict.fromkeys(_normalize_reference_text(p.exact_reference_text) for p in ordered)
        )
        if len(unique_texts) > 1 and not any(
            p.continuation_from_previous_page or p.continuation_to_next_page for p in ordered
        ):
            findings.append(
                {
                    "severity": "error",
                    "issue_code": "duplicate_reference_conflict_across_page_jobs",
                    "reference_number": str(number),
                }
            )
            continue
        text = " ".join(dict.fromkeys(p.exact_reference_text for p in ordered))
        first = ordered[0]
        last = ordered[-1]
        stitched.append(
            {
                "schema_version": "original_reference_repair_v2",
                "source_id": "",
                "original_reference_number": str(number),
                "exact_reference_text": _normalize_reference_text(text),
                "page_start": first.page_start,
                "page_end": last.page_end,
                "column_start": first.column_start,
                "column_end": last.column_end,
                "continuation_from_previous_page": first.continuation_from_previous_page,
                "continuation_to_next_page": last.continuation_to_next_page,
                "extraction_confidence": min(p.extraction_confidence for p in ordered),
                "review_required": any(p.review_required for p in ordered),
                "review_notes": [note for p in ordered for note in p.review_notes],
            }
        )
    return stitched, findings


def validate_complete_reference_sequence_v2(
    references: list[dict[str, Any]], *, expected_total: int = 720
) -> dict[str, Any]:
    numbers = [int(row["original_reference_number"]) for row in references]
    expected = set(range(1, expected_total + 1))
    present = set(numbers)
    duplicate_numbers = sorted({number for number in numbers if numbers.count(number) > 1})
    missing = sorted(expected - present)
    out_of_range = sorted(present - expected)
    empty_text_numbers = sorted(
        int(row["original_reference_number"])
        for row in references
        if not str(
            row.get("exact_reference_text") or row.get("exact_original_reference_text") or ""
        ).strip()
    )
    missing_anchor_numbers = sorted(
        int(row["original_reference_number"])
        for row in references
        if not row.get("page_start") or not row.get("page_end")
    )
    complete = not (
        missing
        or duplicate_numbers
        or out_of_range
        or empty_text_numbers
        or missing_anchor_numbers
    ) and len(present) == expected_total
    return {
        "expected_reference_count": expected_total,
        "actual_reference_count": len(present),
        "complete": complete,
        "missing_reference_numbers": missing,
        "duplicate_reference_numbers": duplicate_numbers,
        "out_of_range_reference_numbers": out_of_range,
        "empty_text_reference_numbers": empty_text_numbers,
        "missing_anchor_reference_numbers": missing_anchor_numbers,
    }


def plan_gap_repair_cycles_v2(
    *, missing_numbers: list[int], references: list[dict[str, Any]], max_cycles: int = 2
) -> dict[str, Any]:
    page_by_number = {
        int(row["original_reference_number"]): int(row["page_start"]) for row in references
    }
    cycles: list[dict[str, Any]] = []
    unresolved = list(missing_numbers)
    for cycle in range(1, max_cycles + 1):
        pages = []
        for number in unresolved:
            lower = max((n for n in page_by_number if n < number), default=None)
            upper = min((n for n in page_by_number if n > number), default=None)
            page = page_by_number.get(lower) or page_by_number.get(upper)
            if page is not None:
                pages.append(page)
        cycles.append(
            {
                "cycle": cycle,
                "target_primary_pages": sorted(set(pages)),
                "remaining_missing_reference_numbers": unresolved,
            }
        )
        if not pages:
            break
    return {"max_cycles": max_cycles, "cycles": cycles}


RETRYABLE_GEMINI_HTTP_STATUSES = {408, 429, 500, 502, 503, 504}
NON_RETRYABLE_GEMINI_HTTP_STATUSES = {400, 401, 403}
SENSITIVE_ERROR_KEYS = {"authorization", "api_key", "x-goog-api-key", "key", "token", "password"}


def _sanitize_error_value(value: Any, *, max_length: int = 600) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower()
            sanitized[str(key)] = (
                "[redacted]"
                if any(secret in lowered for secret in SENSITIVE_ERROR_KEYS)
                else _sanitize_error_value(item, max_length=max_length)
            )
        return sanitized
    if isinstance(value, list):
        return [_sanitize_error_value(item, max_length=max_length) for item in value]
    text = str(value)
    for marker in ("Authorization:", "x-goog-api-key", "api_key", "GEMINI_API_KEY"):
        if marker.lower() in text.lower():
            text = "[redacted]"
            break
    return text[:max_length]


def _exception_http_status(exc: Exception) -> int | None:
    for attribute in ("status_code", "code"):
        value = getattr(exc, attribute, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def _headers_from_exception(exc: Exception) -> dict[str, Any]:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        headers = getattr(exc, "headers", None)
    if headers is None:
        return {}
    return {str(key).lower(): value for key, value in dict(headers).items()}


def _retry_after_seconds(exc: Exception, *, now: Callable[[], datetime]) -> int | None:
    value = _headers_from_exception(exc).get("retry-after")
    if value is None:
        return None
    text = str(value).strip()
    if text.isdigit():
        return max(0, int(text))
    with suppress(Exception):
        parsed = parsedate_to_datetime(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return max(0, int((parsed - now()).total_seconds()))
    return None


def _request_id_from_exception(exc: Exception) -> str | None:
    headers = _headers_from_exception(exc)
    for key in ("x-request-id", "x-goog-request-id", "x-google-request-id"):
        value = headers.get(key)
        if value:
            return str(value)
    for attribute in ("request_id", "requestId"):
        value = getattr(exc, attribute, None)
        if value:
            return str(value)
    return None


def _api_status_from_exception(exc: Exception) -> str | None:
    for attribute in ("status", "reason", "reason_phrase"):
        value = getattr(exc, attribute, None)
        if value:
            return str(value)
    response = getattr(exc, "response", None)
    for attribute in ("reason_phrase", "reason"):
        value = getattr(response, attribute, None)
        if value:
            return str(value)
    return None


def _api_message_from_exception(exc: Exception) -> str:
    for attribute in ("message", "details"):
        value = getattr(exc, attribute, None)
        if value:
            return str(_sanitize_error_value(value))
    response = getattr(exc, "response", None)
    for attribute in ("text", "content"):
        value = getattr(response, attribute, None)
        if value:
            return str(_sanitize_error_value(value))
    return str(_sanitize_error_value(exc))


def classify_gemini_reference_repair_error(exc: Exception) -> dict[str, Any]:
    status = _exception_http_status(exc)
    class_name = exc.__class__.__name__.lower()
    module = exc.__class__.__module__.lower()
    message = _api_message_from_exception(exc).lower()
    non_retryable_message = any(
        phrase in message
        for phrase in (
            "invalid api key",
            "permission denied",
            "invalid model",
            "model not found",
            "invalid request",
            "request schema",
        )
    )
    retryable_class = any(
        token in class_name or token in module
        for token in (
            "readtimeout",
            "connecttimeout",
            "timeout",
            "connectionerror",
            "connectionreset",
            "connectionaborted",
            "servererror",
            "serviceunavailable",
            "temporary",
            "dns",
        )
    )
    if status in NON_RETRYABLE_GEMINI_HTTP_STATUSES or non_retryable_message:
        transient = False
    elif status in RETRYABLE_GEMINI_HTTP_STATUSES or isinstance(
        exc, (ConnectionError, TimeoutError, OSError)
    ):
        transient = True
    else:
        transient = retryable_class
    return {
        "exception_class": exc.__class__.__name__,
        "exception_module": exc.__class__.__module__,
        "http_status": status,
        "api_status": _api_status_from_exception(exc),
        "api_message": _api_message_from_exception(exc),
        "request_id": _request_id_from_exception(exc),
        "transient": transient,
    }


def calculate_gemini_retry_delay_seconds(
    *,
    model_config: dict[str, Any],
    attempt_number: int,
    retry_after_seconds: int | None = None,
    random_fraction: float | None = None,
) -> tuple[int, int]:
    initial = float(model_config["retry_initial_delay_seconds"])
    multiplier = float(model_config["retry_backoff_multiplier"])
    maximum = int(model_config["retry_max_delay_seconds"])
    base_delay = min(maximum, round(initial * (multiplier ** (attempt_number - 1))))
    if retry_after_seconds is not None:
        base_delay = max(base_delay, int(retry_after_seconds))
    jitter_fraction = float(model_config.get("retry_jitter_fraction", 0.0))
    if jitter_fraction and retry_after_seconds is None:
        fraction = random.random() if random_fraction is None else random_fraction
        factor = 1 + ((fraction * 2) - 1) * jitter_fraction
        applied = round(base_delay * factor)
    else:
        applied = base_delay
    return base_delay, min(maximum, max(0, applied))


class GeminiReferenceRepairRetryState:
    def __init__(self) -> None:
        self.consecutive_transient_failures = 0

    def record_success(self) -> None:
        self.consecutive_transient_failures = 0

    def record_failure(self, *, transient: bool) -> None:
        if transient:
            self.consecutive_transient_failures += 1
        else:
            self.consecutive_transient_failures = 0


class GeminiOriginalReferenceRepairV2Client:
    """Gemini boundary for per-page physical PDF slices."""

    def __init__(
        self,
        *,
        api_key: SecretStr,
        model_config: dict[str, Any],
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        output: Callable[[str], None] = print,
        random_fraction: Callable[[], float] = random.random,
        retry_state: GeminiReferenceRepairRetryState | None = None,
    ) -> None:
        config = SimpleNamespace(**model_config)
        self._gateway = GeminiDocumentMapClient(
            api_key=api_key,
            model_config=config,
            client=client,
            sleep=sleep,
            request_timeout_seconds=int(model_config["request_timeout_seconds"]),
        )
        self._model_config = model_config
        self._sleep = sleep
        self._monotonic = monotonic
        self._now = now
        self._output = output
        self._random_fraction = random_fraction
        self._retry_state = retry_state or GeminiReferenceRepairRetryState()

    @staticmethod
    def _repair_schema() -> dict[str, Any]:
        return GeminiDocumentMapClient.request_schema(OriginalReferenceRepairV2PageJobOutput)

    def _write_attempt(self, job_dir: Path, record: dict[str, Any]) -> None:
        sanitized = _sanitize_error_value(record)
        _append_jsonl(job_dir / "attempts.jsonl", sanitized)
        if sanitized.get("exception_class") and (
            sanitized.get("final_failure") or not sanitized.get("retry_planned")
        ):
            _write_json_replace(job_dir / "last_error.json", sanitized)

    def _wait_with_progress(self, *, page: int, seconds: int) -> None:
        remaining = int(seconds)
        while remaining > 0:
            step = min(60, remaining)
            self._sleep(float(step))
            remaining -= step
            if remaining > 0:
                self._output(
                    f"[Seite {page}] Warte auf nächsten Gemini-Versuch: "
                    f"noch {remaining} Sekunden."
                )

    def _apply_global_cooldown_if_needed(self, *, page: int) -> None:
        threshold = int(
            self._model_config["global_cooldown_after_consecutive_transient_failures"]
        )
        if self._retry_state.consecutive_transient_failures < threshold:
            return
        cooldown = int(self._model_config["global_cooldown_seconds"])
        self._output(
            f"[Seite {page}] {self._retry_state.consecutive_transient_failures} "
            f"aufeinanderfolgende transiente Gemini-Fehler. "
            f"Globaler Cooldown: {cooldown} Sekunden."
        )
        self._wait_with_progress(page=page, seconds=cooldown)

    def _attempt_operation(
        self,
        *,
        operation: Callable[[], Any],
        stage: Literal["upload", "file_processing", "generate_content"],
        job_id: str,
        primary_original_pdf_page: int,
        job_dir: Path,
    ) -> Any:
        max_attempts = int(self._model_config["max_attempts"])
        for attempt in range(1, max_attempts + 1):
            self._apply_global_cooldown_if_needed(page=primary_original_pdf_page)
            started_at = self._now()
            started_monotonic = self._monotonic()
            try:
                value = operation()
            except Exception as exc:  # SDK exception types vary by transport.
                finished_at = self._now()
                details = classify_gemini_reference_repair_error(exc)
                retry_after = _retry_after_seconds(exc, now=self._now)
                calculated, applied = calculate_gemini_retry_delay_seconds(
                    model_config=self._model_config,
                    attempt_number=attempt,
                    retry_after_seconds=retry_after if details["http_status"] == 429 else None,
                    random_fraction=self._random_fraction(),
                )
                transient = bool(details["transient"])
                retry_planned = transient and attempt < max_attempts
                self._retry_state.record_failure(transient=transient)
                record = {
                    "job_id": job_id,
                    "primary_original_pdf_page": primary_original_pdf_page,
                    "stage": stage,
                    "attempt_number": attempt,
                    "max_attempts": max_attempts,
                    "started_at": started_at.isoformat(),
                    "finished_at": finished_at.isoformat(),
                    "duration_seconds": round(self._monotonic() - started_monotonic, 3),
                    **details,
                    "retry_after_seconds": retry_after,
                    "calculated_backoff_seconds": calculated,
                    "applied_delay_seconds": applied if retry_planned else None,
                    "model_id": self._model_config["model_id"],
                    "retry_planned": retry_planned,
                    "response_received": False,
                    "raw_response_preserved": False,
                    "final_failure": not retry_planned,
                }
                self._write_attempt(job_dir, record)
                if retry_planned:
                    status = (
                        f"HTTP {details['http_status']} {details.get('api_status') or ''}".strip()
                        if details.get("http_status")
                        else details["exception_class"]
                    )
                    if details["http_status"] == 429 and retry_after is not None:
                        self._output(
                            f"[Seite {primary_original_pdf_page}] HTTP 429 "
                            f"{details.get('api_status') or 'RESOURCE_EXHAUSTED'}.\n"
                            f"Retry-After: {retry_after} Sekunden.\n"
                            f"Neuer Versuch in {applied} Sekunden."
                        )
                    else:
                        self._output(
                            f"[Seite {primary_original_pdf_page}] Gemini-Versuch "
                            f"{attempt}/{max_attempts} fehlgeschlagen:\n"
                            f"{status} - {details['api_message']}\n"
                            f"Neuer Versuch in {applied} Sekunden."
                        )
                    self._wait_with_progress(page=primary_original_pdf_page, seconds=applied)
                    continue
                message = (
                    f"Gemini PageJob {job_id} failed at {stage}: "
                    f"{details.get('http_status') or details['exception_class']} "
                    f"{details['api_message']}"
                )
                raise GeminiError(message) from exc
            self._retry_state.record_success()
            finished_at = self._now()
            self._write_attempt(
                job_dir,
                {
                    "job_id": job_id,
                    "primary_original_pdf_page": primary_original_pdf_page,
                    "stage": stage,
                    "attempt_number": attempt,
                    "max_attempts": max_attempts,
                    "started_at": started_at.isoformat(),
                    "finished_at": finished_at.isoformat(),
                    "duration_seconds": round(self._monotonic() - started_monotonic, 3),
                    "exception_class": None,
                    "exception_module": None,
                    "http_status": None,
                    "api_status": None,
                    "api_message": None,
                    "request_id": None,
                    "retry_after_seconds": None,
                    "calculated_backoff_seconds": None,
                    "applied_delay_seconds": None,
                    "model_id": self._model_config["model_id"],
                    "transient": False,
                    "retry_planned": False,
                    "response_received": stage == "generate_content",
                    "raw_response_preserved": False,
                    "final_failure": False,
                },
            )
            return value
        raise GeminiError(f"Gemini PageJob {job_id} failed after {max_attempts} attempts.")

    def _wait_until_active_v2(
        self, *, remote: Any, job_id: str, primary_original_pdf_page: int, job_dir: Path
    ) -> Any:
        name = getattr(remote, "name", None)
        current = remote
        max_attempts = int(self._model_config["max_attempts"])
        for attempt in range(1, max_attempts + 1):
            state = str(getattr(current, "state", "")).upper().split(".")[-1]
            if state == "ACTIVE":
                return current
            if state == "FAILED":
                raise GeminiError("Gemini-Dateiverarbeitung ist endgültig fehlgeschlagen.")
            if state != "PROCESSING":
                raise GeminiError("Unbekannter Gemini-Dateistatus.")
            if attempt == max_attempts:
                break
            current = self._attempt_operation(
                operation=lambda: self._gateway._client.files.get(name=name),
                stage="file_processing",
                job_id=job_id,
                primary_original_pdf_page=primary_original_pdf_page,
                job_dir=job_dir,
            )
        raise GeminiError("Gemini-Dateiverarbeitung hat das Zeitlimit überschritten.")

    def request_page_job(
        self,
        *,
        slice_pdf: Path,
        prompt: str,
        source_id: str,
        job_id: str,
        primary_original_pdf_page: int,
        job_dir: Path,
    ) -> tuple[OriginalReferenceRepairV2PageJobOutput, dict[str, Any], dict[str, Any]]:
        remote = self._attempt_operation(
            operation=lambda: self._gateway._client.files.upload(
                file=slice_pdf, config={"mime_type": "application/pdf"}
            ),
            stage="upload",
            job_id=job_id,
            primary_original_pdf_page=primary_original_pdf_page,
            job_dir=job_dir,
        )
        try:
            remote = self._wait_until_active_v2(
                remote=remote,
                job_id=job_id,
                primary_original_pdf_page=primary_original_pdf_page,
                job_dir=job_dir,
            )
            response = self._attempt_operation(
                operation=lambda: self._gateway._client.models.generate_content(
                    model=self._model_config["model_id"],
                    contents=[
                        {
                            "file_data": {
                                "file_uri": getattr(remote, "uri", None),
                                "mime_type": getattr(remote, "mime_type", "application/pdf"),
                            }
                        },
                        {"text": prompt},
                    ],
                    config={
                        "thinking_config": {
                            "thinking_level": self._model_config["thinking_level"].upper()
                        },
                        "media_resolution": (
                            f"MEDIA_RESOLUTION_{self._model_config['media_resolution'].upper()}"
                        ),
                        "response_mime_type": "application/json",
                        "response_json_schema": self._repair_schema(),
                        "http_options": {
                            "timeout": self._model_config["request_timeout_seconds"] * 1000
                        },
                    },
                ),
                stage="generate_content",
                job_id=job_id,
                primary_original_pdf_page=primary_original_pdf_page,
                job_dir=job_dir,
            )
            raw_text = getattr(response, "text", None)
            if not isinstance(raw_text, str):
                raw_payload = {"raw_response_text": None}
                metadata = {"response_received": True, "raw_response_preserved": True}
                _write_json_replace(job_dir / "raw_response.json", raw_payload)
                record = {
                    "job_id": job_id,
                    "primary_original_pdf_page": primary_original_pdf_page,
                    "stage": "response_parse",
                    "attempt_number": 1,
                    "max_attempts": int(self._model_config["max_attempts"]),
                    "started_at": self._now().isoformat(),
                    "finished_at": self._now().isoformat(),
                    "duration_seconds": 0,
                    "exception_class": "GeminiResponseValidationError",
                    "exception_module": GeminiResponseValidationError.__module__,
                    "http_status": None,
                    "api_status": None,
                    "api_message": "Gemini-Antwort enthält kein JSON.",
                    "request_id": None,
                    "retry_after_seconds": None,
                    "calculated_backoff_seconds": None,
                    "applied_delay_seconds": None,
                    "model_id": self._model_config["model_id"],
                    "transient": False,
                    "retry_planned": False,
                    "response_received": True,
                    "raw_response_preserved": True,
                    "final_failure": True,
                }
                self._write_attempt(job_dir, record)
                raise PageJobValidationFailed(
                    message="Gemini-Antwort enthält kein JSON.",
                    raw_payload=raw_payload,
                    metadata=metadata,
                )
            try:
                raw_payload = json.loads(raw_text)
            except json.JSONDecodeError as exc:
                raw_payload = {"raw_response_text": raw_text}
                _write_json_replace(job_dir / "raw_response.json", raw_payload)
                record = {
                    "job_id": job_id,
                    "primary_original_pdf_page": primary_original_pdf_page,
                    "stage": "response_parse",
                    "attempt_number": 1,
                    "max_attempts": int(self._model_config["max_attempts"]),
                    "started_at": self._now().isoformat(),
                    "finished_at": self._now().isoformat(),
                    "duration_seconds": 0,
                    "exception_class": exc.__class__.__name__,
                    "exception_module": exc.__class__.__module__,
                    "http_status": None,
                    "api_status": None,
                    "api_message": str(exc),
                    "request_id": None,
                    "retry_after_seconds": None,
                    "calculated_backoff_seconds": None,
                    "applied_delay_seconds": None,
                    "model_id": self._model_config["model_id"],
                    "transient": False,
                    "retry_planned": False,
                    "response_received": True,
                    "raw_response_preserved": True,
                    "final_failure": True,
                }
                self._write_attempt(job_dir, record)
                raise PageJobValidationFailed(
                    message="Gemini-Antwort ist kein valides JSON.",
                    raw_payload=raw_payload,
                    metadata={"response_received": True, "raw_response_preserved": True},
                ) from exc
            metadata = {
                "finish_reason": getattr(response, "finish_reason", None),
                "usage": self._gateway.normalize_usage(getattr(response, "usage_metadata", None)),
                "response_candidate_count": len(getattr(response, "candidates", []) or []),
                "response_size": len(raw_text),
                "attempt_count": int(self._model_config["max_attempts"]),
                "response_received": True,
                "raw_response_preserved": True,
            }
            try:
                validated, normalization = parse_page_job_raw_response_v2(raw_payload)
            except (PageJobValidationFailed, ValueError) as exc:
                _write_json_replace(job_dir / "raw_response.json", raw_payload)
                record = {
                    "job_id": job_id,
                    "primary_original_pdf_page": primary_original_pdf_page,
                    "stage": "schema_validation",
                    "attempt_number": 1,
                    "max_attempts": int(self._model_config["max_attempts"]),
                    "started_at": self._now().isoformat(),
                    "finished_at": self._now().isoformat(),
                    "duration_seconds": 0,
                    "exception_class": exc.__class__.__name__,
                    "exception_module": exc.__class__.__module__,
                    "http_status": None,
                    "api_status": None,
                    "api_message": str(exc),
                    "request_id": None,
                    "retry_after_seconds": None,
                    "calculated_backoff_seconds": None,
                    "applied_delay_seconds": None,
                    "model_id": self._model_config["model_id"],
                    "transient": False,
                    "retry_planned": False,
                    "response_received": True,
                    "raw_response_preserved": True,
                    "final_failure": True,
                }
                self._write_attempt(job_dir, record)
                raise PageJobValidationFailed(
                    message=str(exc),
                    raw_payload=raw_payload,
                    metadata=metadata,
                    validation_errors=getattr(exc, "validation_errors", None),
                ) from exc
            metadata["schema_version_normalization"] = normalization
            return validated, raw_payload, metadata
        finally:
            with suppress(Exception):
                self._gateway._client.files.delete(name=remote.name)


def _page_job_prompt_v2(
    *, base_prompt: str, source_id: str, job: dict[str, Any], slice_page_map: list[dict[str, Any]]
) -> str:
    return (
        f"source_id: {source_id}\n"
        f"job_id: {job['job_id']}\n"
        f"primary_original_pdf_page: {job['primary_original_pdf_page']}\n"
        f"context_original_pdf_pages: {job['context_original_pdf_pages']}\n"
        f"slice_page_map: {json.dumps(slice_page_map, ensure_ascii=False)}\n\n"
        f"{base_prompt}"
    )


def run_reference_repair_v2_and_rebuild(
    *,
    pdf: Path,
    extraction_run: Path,
    synthesis_run: Path,
    failed_reference_run: Path,
    output_root: Path,
    api_key: SecretStr,
    resume_run: Path | None = None,
    pages_per_job: int = 1,
    context_pages: int = 1,
    client_factory: Callable[..., Any] = GeminiOriginalReferenceRepairV2Client,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> Path:
    failed_manifest = _load_json(failed_reference_run / "reference_rebuild_manifest.json")
    if failed_manifest.get("status") != "failed":
        raise ValueError("Failed Reference-Rebuild-Run ist nicht failed.")
    extraction_manifest = _load_json(extraction_run / "extraction_manifest.json")
    synthesis_manifest = _load_json(synthesis_run / "synthesis_manifest.json")
    source_id = synthesis_manifest["source_id"]
    if extraction_manifest["source_id"] != source_id:
        raise ValueError("source_id mismatch.")
    model_config = load_reference_repair_v2_model_config()
    prompt = REPAIR_V2_PROMPT_PATH.read_text(encoding="utf-8")
    prompt_hash = sha256_text(prompt)
    pdf_hash = file_hash(pdf)
    page_count = len(PdfReader(str(pdf)).pages)
    page_plan = bibliography_page_plan_v2(
        document_map=_load_json(extraction_run / "document_map.validated.json"),
        pdf_page_count=page_count,
        pages_per_job=pages_per_job,
        context_pages=context_pages,
    )
    fingerprint = {
        "source_id": source_id,
        "repair_version": REFERENCE_REPAIR_V2_VERSION,
        "repair_prompt_version": REFERENCE_REPAIR_V2_PROMPT_VERSION,
        "repair_schema_version": REFERENCE_REPAIR_V2_SCHEMA_VERSION,
        "prompt_hash": prompt_hash,
        "model_config": model_config,
        "model_config_hash": sha256_text(json.dumps(model_config, sort_keys=True)),
        "pdf": str(pdf.resolve()),
        "pdf_hash": pdf_hash,
        "extraction_run": str(extraction_run.resolve()),
        "synthesis_run": str(synthesis_run.resolve()),
        "failed_reference_run": str(failed_reference_run.resolve()),
        "page_plan": page_plan,
        "git_commit": _git_commit(),
    }
    reject_document_map_versions_in_repair_fingerprint(fingerprint)
    import_raw_from_run: Path | None = None
    if resume_run:
        candidate_resume_run = resume_run.resolve()
        old_fingerprint = _load_json(candidate_resume_run / "checkpoint_fingerprint.json")
        if old_fingerprint == fingerprint:
            run_dir = candidate_resume_run
            manifest_path = run_dir / "reference_repair_v2_manifest.json"
            if manifest_path.is_file() and _load_json(manifest_path).get("status") in {
                "completed",
                "completed_with_review",
            }:
                return run_dir
        elif _compatible_raw_import_fingerprint(old_fingerprint, fingerprint):
            import_raw_from_run = candidate_resume_run
            root = ensure_external_run_root(output_root, synthesis_run)
            run_dir = root / (
                f"reference-repair-v2-{now():%Y%m%dT%H%M%S%fZ}-{source_id}-"
                f"{sha256_text(json.dumps(fingerprint, sort_keys=True))[:8]}"
            )
            run_dir.mkdir(parents=True, exist_ok=False)
            write_json(
                run_dir / "checkpoint_fingerprint.json",
                {**fingerprint, "import_raw_from_run": str(import_raw_from_run)},
            )
        else:
            raise ValueError("Resume fingerprint does not match")
    else:
        root = ensure_external_run_root(output_root, synthesis_run)
        run_dir = root / (
            f"reference-repair-v2-{now():%Y%m%dT%H%M%S%fZ}-{source_id}-"
            f"{sha256_text(json.dumps(fingerprint, sort_keys=True))[:8]}"
        )
        run_dir.mkdir(parents=True, exist_ok=False)
        write_json(run_dir / "checkpoint_fingerprint.json", fingerprint)
    _write_json_replace(run_dir / "bibliography_page_plan.json", page_plan)
    findings: list[dict[str, Any]] = []
    page_outputs: list[OriginalReferenceRepairV2PageJobOutput] = []
    gateway = client_factory(api_key=api_key, model_config=model_config)
    for job in page_plan["jobs"]:
        job_dir = run_dir / "page_jobs" / job["job_id"]
        job_dir.mkdir(parents=True, exist_ok=True)
        slice_pdf = job_dir / "slice.pdf"
        if not slice_pdf.exists():
            write_pdf_slice(pdf, job["slice_page_map"], slice_pdf)
        slice_hash = file_hash(slice_pdf)
        job_fingerprint = page_job_fingerprint_v2(
            source_id=source_id,
            pdf_hash=pdf_hash,
            job=job,
            slice_pdf_hash=slice_hash,
            model_config=model_config,
            prompt_hash=prompt_hash,
        )
        checkpoint_path = job_dir / "checkpoint.json"
        if checkpoint_path.exists():
            checkpoint = _load_json(checkpoint_path)
            if checkpoint.get("fingerprint") == job_fingerprint and checkpoint.get("status") in {
                "completed",
                "completed_with_review",
            }:
                raw = _load_json(job_dir / "raw_response.json")
                output, normalization = parse_page_job_raw_response_v2(raw)
                if normalization["schema_version_normalized"]:
                    checkpoint["metadata"] = {
                        **checkpoint.get("metadata", {}),
                        "schema_version_normalization": normalization,
                    }
                    checkpoint["status"] = "completed"
                    _write_json_replace(checkpoint_path, checkpoint)
                    manifest_path = job_dir / "job_manifest.json"
                    manifest = _load_json(manifest_path) if manifest_path.exists() else {}
                    manifest["schema_version_normalization"] = normalization
                    manifest["status"] = "completed"
                    _write_json_replace(manifest_path, manifest)
                    _write_jsonl_replace(
                        job_dir / "validated_references.jsonl",
                        [ref.model_dump(mode="json") for ref in output.references],
                    )
                page_outputs.append(output)
                continue
        if import_raw_from_run and not (job_dir / "raw_response.json").exists():
            imported_job_dir = import_raw_from_run / "page_jobs" / job["job_id"]
            imported_raw = imported_job_dir / "raw_response.json"
            if imported_raw.exists():
                shutil.copy2(imported_raw, job_dir / "raw_response.json")
                _write_json_replace(
                    job_dir / "imported_compatible_raw_response.json",
                    {
                        "source_run": str(import_raw_from_run),
                        "source_raw_response": str(imported_raw),
                        "imported_compatible_raw_response": True,
                    },
                )
        if (job_dir / "raw_response.json").exists():
            raw = _load_json(job_dir / "raw_response.json")
            try:
                output, normalization = parse_page_job_raw_response_v2(raw)
                status, job_findings = validate_page_job_output_v2(output)
                metadata = {
                    "schema_version_normalization": normalization,
                    "reparsed_from_raw": True,
                    "imported_compatible_raw_response": bool(
                        (job_dir / "imported_compatible_raw_response.json").exists()
                    ),
                }
                _write_jsonl_replace(
                    job_dir / "validated_references.jsonl",
                    [ref.model_dump(mode="json") for ref in output.references],
                )
                _write_json_replace(
                    job_dir / "job_manifest.json",
                    {
                        "job": job,
                        "metadata": metadata,
                        "schema_version_normalization": normalization,
                        "status": status,
                        "findings": job_findings,
                    },
                )
                _write_json_replace(
                    checkpoint_path,
                    {
                        "fingerprint": job_fingerprint,
                        "status": status,
                        "metadata": metadata,
                        "findings": job_findings,
                    },
                )
                findings.extend(job_findings)
                if status != "failed":
                    page_outputs.append(output)
                    continue
            except (PageJobValidationFailed, ValueError) as exc:
                error_finding = {
                    "severity": "error",
                    "issue_code": "page_job_validation_failed",
                    "job_id": job["job_id"],
                    "message": str(exc),
                    "validation_errors": getattr(exc, "validation_errors", None),
                }
                metadata = getattr(exc, "metadata", {})
                _write_json_replace(
                    job_dir / "job_manifest.json",
                    {
                        "job": job,
                        "metadata": metadata,
                        "status": "failed",
                        "findings": [error_finding],
                    },
                )
                _write_json_replace(
                    checkpoint_path,
                    {
                        "fingerprint": job_fingerprint,
                        "status": "failed",
                        "metadata": metadata,
                        "findings": [error_finding],
                    },
                )
                findings.append(error_finding)
                continue
        request_prompt = _page_job_prompt_v2(
            base_prompt=prompt,
            source_id=source_id,
            job=job,
            slice_page_map=job["slice_page_map"],
        )
        (job_dir / "request_prompt.txt").write_text(request_prompt, encoding="utf-8")
        _write_json_replace(job_dir / "slice_page_map.json", job["slice_page_map"])
        try:
            output, raw_payload, metadata = gateway.request_page_job(
                slice_pdf=slice_pdf,
                prompt=request_prompt,
                source_id=source_id,
                job_id=job["job_id"],
                primary_original_pdf_page=job["primary_original_pdf_page"],
                job_dir=job_dir,
            )
        except PageJobValidationFailed as exc:
            raw_payload = exc.raw_payload
            metadata = exc.metadata
            error_finding = {
                "severity": "error",
                "issue_code": "page_job_validation_failed",
                "job_id": job["job_id"],
                "message": str(exc),
                "validation_errors": exc.validation_errors,
            }
            _write_json_replace(job_dir / "raw_response.json", raw_payload)
            _write_json_replace(
                job_dir / "job_manifest.json",
                {
                    "job": job,
                    "metadata": metadata,
                    "status": "failed",
                    "findings": [error_finding],
                },
            )
            _write_json_replace(
                checkpoint_path,
                {
                    "fingerprint": job_fingerprint,
                    "status": "failed",
                    "metadata": metadata,
                    "findings": [error_finding],
                },
            )
            findings.append(error_finding)
            continue
        except GeminiError as exc:
            metadata = {}
            last_error_path = job_dir / "last_error.json"
            if last_error_path.exists():
                metadata["last_error"] = _load_json(last_error_path)
            error_finding = {
                "severity": "error",
                "issue_code": "page_job_gemini_failed",
                "job_id": job["job_id"],
                "message": str(exc),
            }
            _write_json_replace(
                job_dir / "job_manifest.json",
                {
                    "job": job,
                    "metadata": metadata,
                    "status": "failed",
                    "findings": [error_finding],
                },
            )
            _write_json_replace(
                checkpoint_path,
                {
                    "fingerprint": job_fingerprint,
                    "status": "failed",
                    "metadata": metadata,
                    "findings": [error_finding],
                },
            )
            findings.append(error_finding)
            continue
        status, job_findings = validate_page_job_output_v2(output)
        _write_json_replace(job_dir / "raw_response.json", raw_payload)
        _write_jsonl_replace(
            job_dir / "validated_references.jsonl",
            [ref.model_dump(mode="json") for ref in output.references],
        )
        _write_json_replace(
            job_dir / "job_manifest.json",
            {
                "job": job,
                "metadata": metadata,
                "schema_version_normalization": metadata.get("schema_version_normalization"),
                "status": status,
                "findings": job_findings,
            },
        )
        _write_json_replace(
            checkpoint_path,
            {
                "fingerprint": job_fingerprint,
                "status": status,
                "metadata": metadata,
                "findings": job_findings,
            },
        )
        findings.extend(job_findings)
        if status == "failed":
            continue
        page_outputs.append(output)
    coverage_rows = [
        {
            "job_id": output.job_id,
            "primary_original_pdf_page": output.primary_original_pdf_page,
            "first_reference_number_observed": output.first_reference_number_observed,
            "last_reference_number_observed": output.last_reference_number_observed,
            "observed_reference_numbers": output.observed_reference_numbers,
            "page_complete": output.page_complete,
            "review_required": output.review_required,
        }
        for output in page_outputs
    ]
    _write_jsonl_replace(run_dir / "bibliography_page_coverage.jsonl", coverage_rows)
    stitched, stitch_findings = stitch_page_job_references_v2(page_outputs)
    for row in stitched:
        row["source_id"] = source_id
    findings.extend(stitch_findings)
    sequence_report = validate_complete_reference_sequence_v2(stitched)
    _write_json_replace(run_dir / "bibliography_reference_sequence_report.json", sequence_report)
    gap_report = plan_gap_repair_cycles_v2(
        missing_numbers=sequence_report["missing_reference_numbers"], references=stitched
    )
    _write_json_replace(run_dir / "bibliography_gap_repair_report.json", gap_report)
    _write_json_replace(
        run_dir / "gemini_reference_repair_v2_merged.raw.json",
        {"page_jobs": [output.model_dump(mode="json") for output in page_outputs]},
    )
    repaired_refs, merge_report, merge_findings = merge_repaired_references(
        source_id=source_id,
        existing_references=load_jsonl(extraction_run / "references.jsonl"),
        repaired_entries=stitched,
        required_numbers=[str(n) for n in range(1, 721)],
    )
    findings.extend(merge_findings)
    _write_jsonl_replace(run_dir / "repaired_original_references_v2.jsonl", repaired_refs)
    _write_json_replace(run_dir / "repaired_references_merge_report_v2.json", merge_report)
    final_integrity = validate_complete_reference_sequence_v2(repaired_refs)
    _write_json_replace(run_dir / "original_reference_integrity_report.json", final_integrity)
    status = "failed"
    rebuild_run = None
    if final_integrity["complete"] and not any(row["severity"] == "error" for row in findings):
        try:
            rebuild_run = rebuild_guideline_references(
                synthesis_run=synthesis_run,
                output_root=run_dir,
                output_name=(
                    "AISurgeon_Aktualisierte_Leitlinie_GERD_EoE_2026_"
                    "references_repaired_v2.docx"
                ),
                original_references_path=run_dir / "repaired_original_references_v2.jsonl",
                now=now,
            )
            rebuild_manifest = _load_json(rebuild_run / "reference_rebuild_manifest.json")
            status = rebuild_manifest["status"]
            for name in (
                "new_references_numbered.jsonl",
                "new_reference_number_map.json",
                "citation_resolution_report.json",
            ):
                source = rebuild_run / name
                if source.exists():
                    shutil.copy2(source, run_dir / name)
        except RuntimeError as exc:
            findings.append(
                {
                    "severity": "error",
                    "issue_code": "dual_namespace_rebuild_failed",
                    "message": str(exc),
                }
            )
            status = "failed"
    else:
        status = "completed_with_review" if final_integrity["complete"] else "failed"
    _write_jsonl_replace(run_dir / "reference_repair_v2_findings.jsonl", findings)
    _write_xlsx(run_dir / "reference_repair_v2_findings.xlsx", "repair_v2_findings", findings)
    manifest = {
        **fingerprint,
        "created_at": now().isoformat(),
        "status": status,
        "page_job_count": len(page_plan["jobs"]),
        "completed_page_job_count": len(page_outputs),
        "sequence_report": sequence_report,
        "final_integrity": final_integrity,
        "reference_rebuild_run": str(rebuild_run) if rebuild_run else None,
        "output_files": {
            p.name: file_hash(p)
            for p in run_dir.iterdir()
            if p.is_file() and p.name != "reference_repair_v2_manifest.json"
        },
    }
    _write_json_replace(run_dir / "reference_repair_v2_manifest.json", manifest)
    if status == "failed":
        raise RuntimeError(f"Reference repair v2 failed; run directory: {run_dir}")
    return run_dir
