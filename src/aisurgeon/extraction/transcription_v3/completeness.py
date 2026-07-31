"""Deterministic completeness gates for transcription v3."""

import hashlib
from collections.abc import Iterable

from aisurgeon.extraction.pdf_preflight import PdfPagePreflight
from aisurgeon.extraction.transcription_v3.models import (
    CompletenessFinding,
    ExecutionMode,
    ProviderCallEvidence,
    SlicePageMapEntry,
    SourceContent,
    TranscriptionJob,
)


def _finding(
    code: str,
    message: str,
    *,
    page: int | None = None,
    job: str | None = None,
    critical: bool = False,
) -> CompletenessFinding:
    digest = hashlib.sha256(f"{code}|{page}|{job}|{message}".encode()).hexdigest()[:12]
    severity = (
        "error"
        if critical or code.startswith("missing") or code == "finish_reason_truncated"
        else "warning"
    )
    return CompletenessFinding(
        finding_id=f"TX3_FINDING_{digest}",
        severity=severity,
        issue_code=code,
        issue_message=message,
        page_number=page,
        job_id=job,
        repair_required=True,
    )


def validate_transcription_completeness(
    *,
    jobs: list[TranscriptionJob],
    contents: Iterable[SourceContent],
    page_preflight: list[PdfPagePreflight],
    execution_mode: ExecutionMode = "mock_test",
    provider_evidence: list[ProviderCallEvidence] | None = None,
    finish_reasons: dict[str, str | None] | None = None,
    output_ceiling_jobs: set[str] | None = None,
) -> list[CompletenessFinding]:
    findings: list[CompletenessFinding] = []
    expected = {page for job in jobs for page in job.primary_pages}
    by_page: dict[int, list[SourceContent]] = {}
    by_job = {content.job_id: content for content in contents}
    successful_jobs = {
        evidence.job_id
        for evidence in provider_evidence or []
        if evidence.stage == "transcription" and evidence.success and evidence.job_id
    }
    for content in by_job.values():
        for page in content.represented_original_pdf_pages:
            by_page.setdefault(page, []).append(content)
    for page in sorted(expected):
        if page not in by_page:
            findings.append(
                _finding("missing_primary_page", "Planned primary page missing.", page=page)
            )
    preflight_by_page = {page.page_number: page for page in page_preflight}
    for job in jobs:
        content = by_job.get(job.job_id)
        if content is None:
            continue
        if execution_mode == "live" and job.job_id not in successful_jobs:
            findings.append(
                _finding(
                    "missing_provider_evidence",
                    "Live primary pages require successful Gemini response evidence.",
                    job=job.job_id,
                    critical=True,
                )
            )
        order = [block.reading_order_index for block in content.visual_blocks]
        if order != sorted(order):
            findings.append(
                _finding(
                    "reading_order_non_monotonic",
                    "Reading order is not monotonic.",
                    job=job.job_id,
                )
            )
        text_len = sum(
            len(block.exact_visible_text)
            for page in job.primary_pages
            for page_content in by_page.get(page, [])
            for block in page_content.visual_blocks
            if block.page_number == page
        )
        expected_chars = sum(
            preflight_by_page[p].text_layer_character_count
            for p in job.primary_pages
            if p in preflight_by_page
        )
        if expected_chars >= 200 and text_len < expected_chars * 0.2:
            findings.append(
                _finding(
                    "implausibly_short_output",
                    "Output is short relative to local text layer.",
                    job=job.job_id,
                    critical=execution_mode == "live",
                )
            )
        for primary_page in job.primary_pages:
            page_text_len = sum(
                len(block.exact_visible_text.strip())
                for page_content in by_page.get(primary_page, [])
                for block in page_content.visual_blocks
                if block.page_number == primary_page
            )
            preflight = preflight_by_page.get(primary_page)
            if (
                execution_mode == "live"
                and preflight is not None
                and not preflight.obvious_blank_page
                and not preflight.image_heavy
                and page_text_len == 0
            ):
                findings.append(
                    _finding(
                        "empty_nonblank_primary_page",
                        "Live nonblank primary page has no source text.",
                        page=primary_page,
                        job=job.job_id,
                        critical=True,
                    )
                )
        truncated = {"MAX_TOKENS", "LENGTH", "TRUNCATED"}
        if finish_reasons and finish_reasons.get(job.job_id) in truncated:
            findings.append(
                _finding(
                    "finish_reason_truncated",
                    "Provider finish reason indicates truncation.",
                    job=job.job_id,
                )
            )
        if output_ceiling_jobs and job.job_id in output_ceiling_jobs:
            findings.append(
                _finding(
                    "response_at_output_ceiling",
                    "Response size reached configured ceiling.",
                    job=job.job_id,
                )
            )
    return findings


def split_incomplete_job(job: TranscriptionJob) -> list[TranscriptionJob]:
    """Split a problematic job into one primary-page job per page."""
    split_jobs: list[TranscriptionJob] = []
    source_pages = {entry.original_pdf_page_number for entry in job.slice_page_map}
    for page in job.primary_pages:
        context_pages = [
            p
            for p in (page - 1, page + 1)
            if p in {*job.context_pages, *job.primary_pages, *source_pages}
        ]
        all_pages = sorted({page, *context_pages})
        split_jobs.append(
            job.model_copy(
                update={
                    "job_id": f"{job.job_id}-repair-{page:04d}",
                    "chunk_id": f"{job.chunk_id}-repair-{page:04d}",
                    "primary_pages": [page],
                    "context_pages": context_pages,
                    "slice_page_map": [
                        SlicePageMapEntry(
                            slice_page_index=index,
                            original_pdf_page_number=source_page,
                            role=(
                                "primary"
                                if source_page == page
                                else "previous_context"
                                if source_page < page
                                else "next_context"
                            ),
                        )
                        for index, source_page in enumerate(all_pages, start=1)
                    ],
                    "status": "pending",
                    "reason": f"targeted repair split from {job.job_id}",
                }
            )
        )
    return split_jobs
