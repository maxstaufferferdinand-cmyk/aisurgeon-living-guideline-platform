"""Canonical extraction planning and checkpoint helpers."""

import hashlib
import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import SecretStr

from aisurgeon.extraction.canonical.client import CanonicalGeminiClient
from aisurgeon.extraction.canonical.core import (
    PageWindow,
    assign_object_id,
    link_comments,
    link_references,
    merge_formal_items,
    overall_status,
    plan_windows,
)
from aisurgeon.extraction.canonical.models import (
    CANONICAL_EXTRACTION_SCHEMA_VERSION,
    ExtractionBatch,
    ReferenceBatch,
    VisualObjectBatch,
)
from aisurgeon.extraction.canonical.outputs import (
    CANONICAL_OUTPUTS,
    expert_consensus_view,
    recommendation_view,
    statement_view,
    write_json,
    write_jsonl,
    write_review_workbook,
)
from aisurgeon.extraction.gemini.document_map import (
    ensure_output_outside_repository,
    find_project_root,
    git_metadata,
    load_model_config,
    load_prompt,
    validate_document_map,
)
from aisurgeon.extraction.gemini.errors import GeminiError
from aisurgeon.extraction.gemini.models import (
    DOCUMENT_MAP_SCHEMA_VERSION,
    DocumentMap,
    PageRange,
)
from aisurgeon.extraction.pdf_registration import PdfRegistration, register_pdf

PROMPT_FILES = {
    "clinical": "gemini_formal_items_comments_v2.txt",
    "references": "gemini_references_v1.txt",
    "visuals": "gemini_visual_objects_v1.txt",
}
CLINICAL_PROMPT_VERSION = "gemini_formal_items_comments_v2"
DOCUMENT_MAP_PROMPT_VERSION = "gemini_document_map_v1"


def plan_extraction(
    registration: PdfRegistration,
    document_map: DocumentMap | None,
    *,
    pages_per_job: int,
    overlap_pages: int,
) -> list[PageWindow]:
    if registration.page_count is None:
        raise ValueError("Lokale Seitenzahl ist für die Fensterplanung erforderlich.")
    clinical = (
        document_map.clinical_main_body_page_ranges
        if document_map
        else [
            PageRange(
                page_start=1,
                page_end=registration.page_count,
                description="Dry-run fallback: gesamte synthetische PDF",
            )
        ]
    )
    bibliography = document_map.bibliography_page_ranges if document_map else []
    return plan_windows(
        clinical,
        stage="clinical",
        pages_per_job=pages_per_job,
        overlap_pages=overlap_pages,
        document_page_count=registration.page_count,
    ) + plan_windows(
        bibliography,
        stage="references",
        pages_per_job=pages_per_job,
        overlap_pages=overlap_pages,
        document_page_count=registration.page_count,
    )


def prepare_dry_run(
    *,
    pdf_path: Path,
    worker_id: str,
    source_id: str,
    output_root: Path,
    pages_per_job: int = 8,
    overlap_pages: int = 1,
    project_root: Path | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> tuple[dict[str, Any], Path]:
    root = project_root or find_project_root()
    output = ensure_output_outside_repository(output_root, root)
    registration = register_pdf(pdf_path, worker_id=worker_id, source_id=source_id)
    windows = plan_extraction(
        registration, None, pages_per_job=pages_per_job, overlap_pages=overlap_pages
    )
    _, branch, dirty = git_metadata(root)
    _prompt_text, formal_items_prompt_hash = _load_extraction_prompt(root)
    _document_map_prompt, document_map_prompt_hash = load_prompt(root)
    timestamp = now().astimezone(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = (
        f"extract-dry-{timestamp}-{registration.source_id}-{registration.sha256[:8]}-"
        f"{CLINICAL_PROMPT_VERSION}-{formal_items_prompt_hash[:8]}-"
        f"{CANONICAL_EXTRACTION_SCHEMA_VERSION}"
    )
    run_dir = output / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    plan = {
        "run_id": run_id,
        "status": "dry_run",
        "source_id": registration.source_id,
        "worker_id": worker_id,
        "pdf_sha256": registration.sha256,
        "local_page_count": registration.page_count,
        "git_branch": branch,
        "dirty_worktree": dirty,
        "pages_per_job": pages_per_job,
        "overlap_pages": overlap_pages,
        "document_map_schema_version": DOCUMENT_MAP_SCHEMA_VERSION,
        "canonical_extraction_schema_version": CANONICAL_EXTRACTION_SCHEMA_VERSION,
        "document_map_prompt_version": DOCUMENT_MAP_PROMPT_VERSION,
        "formal_items_prompt_version": CLINICAL_PROMPT_VERSION,
        "document_map_prompt_hash": document_map_prompt_hash,
        "formal_items_prompt_hash": formal_items_prompt_hash,
        "jobs": [window.model_dump() for window in windows],
        "planned_outputs": list(CANONICAL_OUTPUTS),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "network_performed": False,
    }
    (run_dir / "extraction_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return plan, run_dir


def checkpoint_complete(
    checkpoint_dir: Path, job_id: str, compatibility: dict[str, Any] | None = None
) -> bool:
    path = checkpoint_dir / f"{job_id}.json"
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value.get("status") == "completed" and (
            compatibility is None or value.get("compatibility") == compatibility
        )
    except (OSError, json.JSONDecodeError):
        return False


def pending_windows(
    windows: list[PageWindow],
    checkpoint_dir: Path,
    compatibility: dict[str, Any] | None = None,
) -> list[PageWindow]:
    return [
        window for window in windows
        if not checkpoint_complete(checkpoint_dir, window.job_id, compatibility)
    ]


def _read_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_checkpoint(path: Path, value: dict[str, Any]) -> None:
    """Atomically create or replace mutable job state without exposing secrets."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _replace_json(path: Path, value: Any) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _request_with_checkpoint(
    *,
    gateway: CanonicalGeminiClient,
    remote: Any,
    prompt: str,
    model: Any,
    status_path: Path,
    raw_path: Path,
    validated_path: Path,
    compatibility: dict[str, Any],
    job: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, int] | None]:
    _write_checkpoint(
        status_path,
        {"status": "in_progress", "job": job, "compatibility": compatibility},
    )
    try:
        validated, raw, usage = gateway.request_structured(
            remote=remote,
            prompt=prompt,
            model=model,
            source_id=compatibility["source_id"],
        )
    except GeminiError:
        _write_checkpoint(
            status_path,
            {"status": "failed", "job": job, "compatibility": compatibility},
        )
        raise
    _replace_json(raw_path, json.loads(raw))
    _replace_json(validated_path, validated)
    _write_checkpoint(
        status_path,
        {"status": "completed", "job": job, "compatibility": compatibility},
    )
    return validated, usage


def _prompt(root: Path, name: str) -> str:
    return (root / "config" / "prompts" / PROMPT_FILES[name]).read_text(encoding="utf-8")


def _load_extraction_prompt(root: Path) -> tuple[str, str]:
    prompt = _prompt(root, "clinical")
    return prompt, hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _compatibility_context(
    *,
    registration: PdfRegistration,
    model_config: Any,
    document_map_prompt_hash: str,
    formal_items_prompt_hash: str,
    pages_per_job: int,
    overlap_pages: int,
) -> dict[str, Any]:
    model_value = model_config.model_dump(mode="json")
    model_hash = hashlib.sha256(
        json.dumps(model_value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "source_id": registration.source_id,
        "pdf_sha256": registration.sha256,
        "model_id": model_config.model_id,
        "model_config": model_value,
        "model_config_sha256": model_hash,
        "document_map_schema_version": DOCUMENT_MAP_SCHEMA_VERSION,
        "canonical_extraction_schema_version": CANONICAL_EXTRACTION_SCHEMA_VERSION,
        "document_map_prompt_version": DOCUMENT_MAP_PROMPT_VERSION,
        "formal_items_prompt_version": CLINICAL_PROMPT_VERSION,
        "document_map_prompt_hash": document_map_prompt_hash,
        "formal_items_prompt_hash": formal_items_prompt_hash,
        "pages_per_job": pages_per_job,
        "overlap_pages": overlap_pages,
    }


def _job_prompt(base: str, source_id: str, window: PageWindow) -> str:
    return (
        f"source_id: {source_id}\nprimary pages: {window.primary_page_start}-"
        f"{window.primary_page_end}\ncontext pages: {window.context_page_start}-"
        f"{window.context_page_end}\nschema_version: "
        f"{CANONICAL_EXTRACTION_SCHEMA_VERSION}\n"
        f"prompt_version: {CLINICAL_PROMPT_VERSION}\n\n{base}"
    )


def run_live_extraction(
    *,
    pdf_path: Path,
    worker_id: str,
    source_id: str,
    output_root: Path,
    api_key: SecretStr,
    pages_per_job: int = 8,
    overlap_pages: int = 1,
    allow_dirty: bool = False,
    keep_remote_file: bool = False,
    project_root: Path | None = None,
    client_factory: Callable[..., CanonicalGeminiClient] = CanonicalGeminiClient,
    resume_run_dir: Path | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> tuple[str, Path]:
    """Execute one-upload staged extraction with per-window checkpoints."""
    root = project_root or find_project_root()
    output = ensure_output_outside_repository(output_root, root)
    registration = register_pdf(pdf_path, worker_id=worker_id, source_id=source_id)
    commit, branch, dirty = git_metadata(root)
    if dirty and not allow_dirty:
        raise ValueError("Live-Lauf bei Dirty Worktree gesperrt.")
    config, _ = load_model_config(root)
    doc_prompt, document_map_prompt_hash = load_prompt(root)
    clinical_prompt, clinical_prompt_hash = _load_extraction_prompt(root)
    compatibility = _compatibility_context(
        registration=registration,
        model_config=config,
        document_map_prompt_hash=document_map_prompt_hash,
        formal_items_prompt_hash=clinical_prompt_hash,
        pages_per_job=pages_per_job, overlap_pages=overlap_pages,
    )
    if resume_run_dir is None:
        timestamp = now().astimezone(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        run_id = (
            f"extract-{timestamp}-{source_id}-{registration.sha256[:8]}-"
            f"{CLINICAL_PROMPT_VERSION}-{clinical_prompt_hash[:8]}-"
            f"{CANONICAL_EXTRACTION_SCHEMA_VERSION}"
        )
        run_dir = output / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        write_json(run_dir / "run_context.json", compatibility)
    else:
        run_dir = resume_run_dir.resolve()
        if run_dir.parent != output or not run_dir.is_dir():
            raise ValueError("Resume-Run liegt nicht im gewählten Output-Root.")
        context_path = run_dir / "run_context.json"
        if _read_checkpoint(context_path) != compatibility:
            raise ValueError("Resume-Run ist mit der aktuellen Extraktion nicht kompatibel.")
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    gateway = client_factory(api_key=api_key, model_config=config)
    remote = gateway.upload_pdf(pdf_path)
    findings = []
    token_usage: dict[str, int] = {}
    try:
        map_path = run_dir / "document_map.validated.json"
        if map_path.exists():
            document_map = DocumentMap.model_validate_json(map_path.read_text(encoding="utf-8"))
        else:
            document_map, usage = _request_with_checkpoint(
                gateway=gateway, remote=remote,
                prompt=f"source_id: {source_id}\n\n{doc_prompt}", model=DocumentMap,
                status_path=checkpoint_dir / "document-map.json",
                raw_path=run_dir / "document_map.raw.json", validated_path=map_path,
                compatibility=compatibility,
            )
            report = validate_document_map(document_map, registration)
            if not report.valid:
                raise ValueError("Dokumentkarte enthält einen Hard-Fail.")
            token_usage.update(usage or {})
        windows = plan_extraction(
            registration, document_map, pages_per_job=pages_per_job, overlap_pages=overlap_pages
        )
        formal_items, comments, context_blocks, references, visuals = [], [], [], [], []
        for window in windows:
            validated_path = checkpoint_dir / f"{window.job_id}.validated.json"
            status_path = checkpoint_dir / f"{window.job_id}.json"
            model = ExtractionBatch if window.stage == "clinical" else ReferenceBatch
            if (
                checkpoint_complete(checkpoint_dir, window.job_id, compatibility)
                and validated_path.exists()
            ):
                batch = model.model_validate_json(validated_path.read_text(encoding="utf-8"))
            else:
                batch, usage = _request_with_checkpoint(
                    gateway=gateway, remote=remote,
                    prompt=_job_prompt(
                        clinical_prompt if window.stage == "clinical"
                        else _prompt(root, window.stage),
                        source_id, window,
                    ),
                    model=model,
                    status_path=status_path,
                    raw_path=checkpoint_dir / f"{window.job_id}.raw.json",
                    validated_path=validated_path, job=window.model_dump(),
                    compatibility=compatibility,
                )
                for key, value in (usage or {}).items():
                    token_usage[key] = token_usage.get(key, 0) + value
            if isinstance(batch, ExtractionBatch):
                formal_items.extend(batch.formal_items)
                comments.extend(batch.comments)
                context_blocks.extend(batch.clinical_context_blocks)
            else:
                references.extend(batch.references)
        visual_status = checkpoint_dir / "visual-objects.json"
        visual_validated = checkpoint_dir / "visual-objects.validated.json"
        if (
            checkpoint_complete(checkpoint_dir, "visual-objects", compatibility)
            and visual_validated.exists()
        ):
            visual_batch = VisualObjectBatch.model_validate_json(
                visual_validated.read_text(encoding="utf-8")
            )
            usage = None
        else:
            visual_batch, usage = _request_with_checkpoint(
            gateway=gateway, remote=remote,
            prompt=f"source_id: {source_id}\n\n{_prompt(root, 'visuals')}",
            model=VisualObjectBatch,
            status_path=visual_status, raw_path=run_dir / "visual_objects.raw.json",
            validated_path=visual_validated,
            compatibility=compatibility,
            )
        visuals.extend(visual_batch.visual_objects)
        for value in references:
            value.reference_id = assign_object_id(
                source_id,
                "REF",
                value.original_reference_number,
                value.exact_original_reference_text,
            )
        for value in visuals:
            value.object_id = assign_object_id(
                source_id,
                value.object_type,
                value.title_or_caption_raw or "",
                str(value.page_start),
                str(value.page_end),
            )
        formal_items, merge_findings = merge_formal_items(formal_items)
        findings.extend(merge_findings)
        findings.extend(link_comments(comments, formal_items))
        unresolved, reference_findings = link_references(formal_items, comments, references)
        findings.extend(reference_findings)
        write_jsonl(run_dir / "formal_items.jsonl", formal_items)
        write_jsonl(
            run_dir / "recommendations.jsonl",
            recommendation_view(formal_items),
        )
        write_jsonl(
            run_dir / "statements.jsonl",
            statement_view(formal_items),
        )
        write_jsonl(
            run_dir / "expert_consensus_items.jsonl",
            expert_consensus_view(formal_items),
        )
        write_jsonl(run_dir / "comments.jsonl", comments)
        write_jsonl(run_dir / "references.jsonl", references)
        for object_type, filename in (
            ("table", "tables.jsonl"),
            ("algorithm", "algorithms.jsonl"),
            ("decision_tree", "decision_trees.jsonl"),
        ):
            write_jsonl(
                run_dir / filename, (value for value in visuals if value.object_type == object_type)
            )
        write_jsonl(run_dir / "clinical_context_blocks.jsonl", context_blocks)
        write_jsonl(run_dir / "unresolved_links.jsonl", unresolved)
        write_jsonl(run_dir / "review_findings.jsonl", findings)
        write_review_workbook(run_dir / "review_findings.xlsx", findings)
        status = overall_status(findings)
        summary = {
            "status": status,
            "formal_item_count": len(formal_items),
            "comment_count": len(comments),
            "reference_count": len(references),
            "review_finding_count": len(findings),
        }
        write_json(run_dir / "extraction_summary.json", summary)
        output_hashes = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in run_dir.iterdir()
            if path.is_file()
        }
        remote_deleted = False
        if not keep_remote_file:
            remote_deleted = gateway.delete_remote(remote)
            remote = None
        write_json(
            run_dir / "extraction_manifest.json",
            {
                "status": status,
                "source_id": source_id,
                "worker_id": worker_id,
                "pdf_sha256": registration.sha256,
                "git_commit": commit,
                "git_branch": branch,
                "dirty_worktree": dirty,
                "model_id": config.model_id,
                **compatibility,
                "token_usage": token_usage or usage,
                "output_files": output_hashes,
                "remote_file_deleted": remote_deleted,
            },
        )
        return status, run_dir
    finally:
        if remote is not None and not keep_remote_file:
            gateway.delete_remote(remote)
