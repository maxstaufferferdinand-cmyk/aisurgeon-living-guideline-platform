"""Targeted original-bibliography repair followed by dual-namespace DOCX rebuild."""

import json
import shutil
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime
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


def load_reference_repair_v2_model_config() -> dict[str, Any]:
    config = _load_json(REPAIR_V2_MODEL_CONFIG_PATH)
    if config.get("repair_prompt_version") != REFERENCE_REPAIR_V2_PROMPT_VERSION:
        raise ValueError("Repair-v2 model config has wrong prompt version.")
    if config.get("repair_schema_version") != REFERENCE_REPAIR_V2_SCHEMA_VERSION:
        raise ValueError("Repair-v2 model config has wrong schema version.")
    serialized = json.dumps(config, sort_keys=True)
    if "document_map" in serialized:
        raise ValueError("Document-map versions are forbidden in Reference Repair v2 config.")
    return config


def reject_document_map_versions_in_repair_fingerprint(fingerprint: dict[str, Any]) -> None:
    serialized = json.dumps(fingerprint, sort_keys=True)
    if "gemini_document_map_v1" in serialized or "document_map_v1" in serialized:
        raise ValueError("Document-map prompt/schema detected in Reference Repair v2 fingerprint.")


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


class GeminiOriginalReferenceRepairV2Client:
    """Gemini boundary for per-page physical PDF slices."""

    def __init__(
        self,
        *,
        api_key: SecretStr,
        model_config: dict[str, Any],
        client: Any | None = None,
    ) -> None:
        config = SimpleNamespace(**model_config)
        self._gateway = GeminiDocumentMapClient(api_key=api_key, model_config=config, client=client)
        self._model_config = model_config

    def request_page_job(
        self, *, slice_pdf: Path, prompt: str, source_id: str
    ) -> tuple[OriginalReferenceRepairV2PageJobOutput, dict[str, Any], dict[str, Any]]:
        remote = self._gateway._with_retry(
            lambda: self._gateway._client.files.upload(
                file=slice_pdf, config={"mime_type": "application/pdf"}
            )
        )
        try:
            remote = self._gateway.wait_until_active(remote)
            response = self._gateway._with_retry(
                lambda: self._gateway._client.models.generate_content(
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
                        "response_json_schema": (
                            OriginalReferenceRepairV2PageJobOutput.model_json_schema()
                        ),
                        "http_options": {
                            "timeout": self._model_config["request_timeout_seconds"] * 1000
                        },
                    },
                )
            )
            raw_text = getattr(response, "text", None)
            if not isinstance(raw_text, str):
                raise GeminiResponseValidationError("Gemini-Antwort enthält kein JSON.")
            raw_payload = json.loads(raw_text)
            validated = OriginalReferenceRepairV2PageJobOutput.model_validate(raw_payload)
            metadata = {
                "finish_reason": getattr(response, "finish_reason", None),
                "usage": self._gateway.normalize_usage(getattr(response, "usage_metadata", None)),
                "response_candidate_count": len(getattr(response, "candidates", []) or []),
                "response_size": len(raw_text),
                "attempt_count": 1,
            }
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
    if resume_run:
        run_dir = resume_run.resolve()
        if _load_json(run_dir / "checkpoint_fingerprint.json") != fingerprint:
            raise ValueError("Resume fingerprint does not match")
        manifest_path = run_dir / "reference_repair_v2_manifest.json"
        if manifest_path.is_file() and _load_json(manifest_path).get("status") in {
            "completed",
            "completed_with_review",
        }:
            return run_dir
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
                page_outputs.append(OriginalReferenceRepairV2PageJobOutput.model_validate(raw))
                continue
        request_prompt = _page_job_prompt_v2(
            base_prompt=prompt,
            source_id=source_id,
            job=job,
            slice_page_map=job["slice_page_map"],
        )
        (job_dir / "request_prompt.txt").write_text(request_prompt, encoding="utf-8")
        _write_json_replace(job_dir / "slice_page_map.json", job["slice_page_map"])
        output, raw_payload, metadata = gateway.request_page_job(
            slice_pdf=slice_pdf, prompt=request_prompt, source_id=source_id
        )
        status, job_findings = validate_page_job_output_v2(output)
        _write_json_replace(job_dir / "raw_response.json", raw_payload)
        _write_jsonl_replace(
            job_dir / "validated_references.jsonl",
            [ref.model_dump(mode="json") for ref in output.references],
        )
        _write_json_replace(
            job_dir / "job_manifest.json",
            {"job": job, "metadata": metadata, "status": status, "findings": job_findings},
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
