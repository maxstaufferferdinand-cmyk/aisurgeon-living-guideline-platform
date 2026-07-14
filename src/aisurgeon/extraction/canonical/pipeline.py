"""Canonical extraction planning and checkpoint helpers."""

import hashlib
import json
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
    ExtractionBatch,
    ReferenceBatch,
    VisualObjectBatch,
)
from aisurgeon.extraction.canonical.outputs import (
    CANONICAL_OUTPUTS,
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
from aisurgeon.extraction.gemini.models import DocumentMap, PageRange
from aisurgeon.extraction.pdf_registration import PdfRegistration, register_pdf

PROMPT_FILES = {
    "clinical": "gemini_formal_items_comments_v1.txt",
    "references": "gemini_references_v1.txt",
    "visuals": "gemini_visual_objects_v1.txt",
}


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
) -> tuple[dict[str, Any], Path]:
    root = project_root or find_project_root()
    output = ensure_output_outside_repository(output_root, root)
    registration = register_pdf(pdf_path, worker_id=worker_id, source_id=source_id)
    windows = plan_extraction(
        registration, None, pages_per_job=pages_per_job, overlap_pages=overlap_pages
    )
    _, branch, dirty = git_metadata(root)
    run_id = f"extract-dry-{registration.source_id}-{registration.sha256[:8]}"
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
        "jobs": [window.model_dump() for window in windows],
        "planned_outputs": list(CANONICAL_OUTPUTS),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "network_performed": False,
    }
    (run_dir / "extraction_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return plan, run_dir


def checkpoint_complete(checkpoint_dir: Path, job_id: str) -> bool:
    path = checkpoint_dir / f"{job_id}.json"
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "completed"
    except (OSError, json.JSONDecodeError):
        return False


def pending_windows(windows: list[PageWindow], checkpoint_dir: Path) -> list[PageWindow]:
    return [window for window in windows if not checkpoint_complete(checkpoint_dir, window.job_id)]


def _prompt(root: Path, name: str) -> str:
    return (root / "config" / "prompts" / PROMPT_FILES[name]).read_text(encoding="utf-8")


def _job_prompt(base: str, source_id: str, window: PageWindow) -> str:
    return (
        f"source_id: {source_id}\nprimary pages: {window.primary_page_start}-"
        f"{window.primary_page_end}\ncontext pages: {window.context_page_start}-"
        f"{window.context_page_end}\n\n{base}"
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
) -> tuple[str, Path]:
    """Execute one-upload staged extraction with per-window checkpoints."""
    root = project_root or find_project_root()
    output = ensure_output_outside_repository(output_root, root)
    registration = register_pdf(pdf_path, worker_id=worker_id, source_id=source_id)
    commit, branch, dirty = git_metadata(root)
    if dirty and not allow_dirty:
        raise ValueError("Live-Lauf bei Dirty Worktree gesperrt.")
    run_dir = output / f"extract-{source_id}-{registration.sha256[:8]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    config, _ = load_model_config(root)
    doc_prompt, _ = load_prompt(root)
    gateway = client_factory(api_key=api_key, model_config=config)
    remote = gateway.upload_pdf(pdf_path)
    findings = []
    token_usage: dict[str, int] = {}
    try:
        map_path = run_dir / "document_map.validated.json"
        if map_path.exists():
            document_map = DocumentMap.model_validate_json(map_path.read_text(encoding="utf-8"))
        else:
            document_map, raw, usage = gateway.request_structured(
                remote=remote, prompt=f"source_id: {source_id}\n\n{doc_prompt}", model=DocumentMap
            )
            report = validate_document_map(document_map, registration)
            if not report.valid:
                raise ValueError("Dokumentkarte enthält einen Hard-Fail.")
            write_json(map_path, document_map)
            write_json(run_dir / "document_map.raw.json", json.loads(raw))
            token_usage.update(usage or {})
        windows = plan_extraction(
            registration, document_map, pages_per_job=pages_per_job, overlap_pages=overlap_pages
        )
        formal_items, comments, context_blocks, references, visuals = [], [], [], [], []
        for window in windows:
            validated_path = checkpoint_dir / f"{window.job_id}.validated.json"
            status_path = checkpoint_dir / f"{window.job_id}.json"
            model = ExtractionBatch if window.stage == "clinical" else ReferenceBatch
            if checkpoint_complete(checkpoint_dir, window.job_id) and validated_path.exists():
                batch = model.model_validate_json(validated_path.read_text(encoding="utf-8"))
            else:
                batch, raw, usage = gateway.request_structured(
                    remote=remote,
                    prompt=_job_prompt(_prompt(root, window.stage), source_id, window),
                    model=model,
                )
                write_json(checkpoint_dir / f"{window.job_id}.raw.json", json.loads(raw))
                write_json(validated_path, batch)
                write_json(status_path, {"status": "completed", "job": window.model_dump()})
                for key, value in (usage or {}).items():
                    token_usage[key] = token_usage.get(key, 0) + value
            if isinstance(batch, ExtractionBatch):
                formal_items.extend(batch.formal_items)
                comments.extend(batch.comments)
                context_blocks.extend(batch.clinical_context_blocks)
            else:
                references.extend(batch.references)
        visual_batch, raw, usage = gateway.request_structured(
            remote=remote,
            prompt=f"source_id: {source_id}\n\n{_prompt(root, 'visuals')}",
            model=VisualObjectBatch,
        )
        write_json(run_dir / "visual_objects.raw.json", json.loads(raw))
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
            (item for item in formal_items if item.item_type == "recommendation"),
        )
        write_jsonl(
            run_dir / "statements.jsonl",
            (item for item in formal_items if item.item_type == "statement"),
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
                "token_usage": token_usage or usage,
                "output_files": output_hashes,
                "remote_file_deleted": remote_deleted,
            },
        )
        return status, run_dir
    finally:
        if remote is not None and not keep_remote_file:
            gateway.delete_remote(remote)
