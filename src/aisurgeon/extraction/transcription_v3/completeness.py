"""Deterministic completeness gates for transcription v3."""

import hashlib
from collections.abc import Iterable

from aisurgeon.extraction.pdf_preflight import PdfPagePreflight
from aisurgeon.extraction.transcription_v3.models import (
    CompletenessFinding,
    SourceContent,
    TranscriptionJob,
)


def _finding(
    code: str, message: str, *, page: int | None = None, job: str | None = None
) -> CompletenessFinding:
    digest = hashlib.sha256(f"{code}|{page}|{job}|{message}".encode()).hexdigest()[:12]
    severity = (
        "error" if code.startswith("missing") or code == "finish_reason_truncated" else "warning"
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
    finish_reasons: dict[str, str | None] | None = None,
    output_ceiling_jobs: set[str] | None = None,
) -> list[CompletenessFinding]:
    findings: list[CompletenessFinding] = []
    expected = {page for job in jobs for page in job.primary_pages}
    by_page: dict[int, list[SourceContent]] = {}
    by_job = {content.job_id: content for content in contents}
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
            for block in content.visual_blocks
            if block.page_number in job.primary_pages
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
    return [
        job.model_copy(
            update={
                "job_id": f"{job.job_id}-repair-{page:04d}",
                "chunk_id": f"{job.chunk_id}-repair-{page:04d}",
                "primary_pages": [page],
                "context_pages": [
                    p
                    for p in (page - 1, page + 1)
                    if p in {*job.context_pages, *job.primary_pages}
                ],
                "status": "pending",
                "reason": f"targeted repair split from {job.job_id}",
            }
        )
        for page in job.primary_pages
    ]
