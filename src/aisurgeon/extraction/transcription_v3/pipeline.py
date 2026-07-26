"""Canonical Gemini transcription v3 run preparation and mocked execution."""

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pypdf import PdfReader, PdfWriter

from aisurgeon.extraction.canonical.outputs import write_json, write_jsonl
from aisurgeon.extraction.gemini.document_map import (
    ensure_output_outside_repository,
    find_project_root,
    git_metadata,
)
from aisurgeon.extraction.pdf_preflight import PdfPagePreflight, PdfPreflight, run_pdf_preflight
from aisurgeon.extraction.pdf_registration import register_pdf
from aisurgeon.extraction.transcription_v3.completeness import validate_transcription_completeness
from aisurgeon.extraction.transcription_v3.models import (
    CANONICAL_TRANSCRIPTION_SCHEMA_VERSION,
    SCOUT_PROMPT_VERSION,
    SCOUT_SCHEMA_VERSION,
    TRANSCRIPTION_PROMPT_VERSION,
    ExtractionScout,
    ExtractionScoutDraft,
    SourceContent,
    SourceContentDraft,
    TranscriptionJob,
    VisualBlock,
)
from aisurgeon.extraction.transcription_v3.planner import build_transcription_plan

GEMINI_MODEL_ID = "gemini-3.5-flash"


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
    draft_factory: Callable[[TranscriptionJob], SourceContentDraft],
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
    write_json(job_dir / "attempts.jsonl", [])
    draft = draft_factory(job)
    write_json(raw_path, draft)
    content = inject_source_content_metadata(draft, source_id=source_id, job=job)
    write_json(validated_path, content)
    write_json(checkpoint_path, {"status": "completed", "job_id": job.job_id})
    return content


def write_merged_transcript_outputs(
    *,
    run_dir: Path,
    preflight: PdfPreflight,
    page_preflight: list[PdfPagePreflight],
    scout: ExtractionScout,
    jobs: list[TranscriptionJob],
    contents: list[SourceContent],
    limit: int | None,
) -> str:
    findings = validate_transcription_completeness(
        jobs=jobs, contents=contents, page_preflight=page_preflight
    )
    complete = not any(finding.severity == "error" for finding in findings)
    status = "technical_limited" if limit is not None else "completed" if complete else "failed"
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
        },
    )
    write_json(
        run_dir / "transcription_manifest.json",
        {
            "schema_version": CANONICAL_TRANSCRIPTION_SCHEMA_VERSION,
            "status": status,
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
    dry_run: bool = False,
    limit: int | None = None,
    resume_run: Path | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    draft_factory: Callable[[TranscriptionJob], SourceContentDraft] = _mock_source_content,
) -> tuple[str, Path]:
    if gemini_concurrency < 1 or gemini_concurrency > 4:
        raise ValueError("gemini_concurrency must be between 1 and 4")
    root = find_project_root()
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
    scout_draft = ExtractionScoutDraft(
        declared_page_count=preflight.page_count,
        regions=[],
        warnings=["mocked_scout_in_dry_or_test_run"],
    )
    scout = inject_scout_metadata(scout_draft, source_id=source_id)
    scout_path = run_dir / "extraction_scout.json"
    if scout_path.is_file():
        scout = ExtractionScout.model_validate_json(scout_path.read_text(encoding="utf-8"))
    else:
        write_json(scout_path, scout)
    jobs = build_transcription_plan(
        preflight=preflight, pages=pages, scout=scout, planner_mode=planner_mode, limit=limit
    )
    plan_path = run_dir / "extraction_plan.json"
    jobs_path = run_dir / "extraction_jobs.jsonl"
    findings_path = run_dir / "extraction_plan_review_findings.jsonl"
    if not plan_path.is_file():
        write_json(plan_path, {"jobs": [job.model_dump() for job in jobs]})
    if not jobs_path.is_file():
        write_jsonl(jobs_path, jobs)
    if not findings_path.is_file():
        write_jsonl(findings_path, [])
    if dry_run:
        write_json(run_dir / "transcription_manifest.json", {"status": "dry_run"})
        return "dry_run", run_dir
    contents = [
        _write_job_artifacts(
            run_dir=run_dir,
            pdf_path=registration.resolved_local_path,
            source_id=source_id,
            job=job,
            draft_factory=draft_factory,
        )
        for job in jobs
    ]
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
        limit=limit,
    )
    return status, run_dir
