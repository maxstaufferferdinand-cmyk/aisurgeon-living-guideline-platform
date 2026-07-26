"""Bounded adaptive extraction planner for transcription v3."""

from aisurgeon.extraction.pdf_preflight import PdfPagePreflight, PdfPreflight
from aisurgeon.extraction.transcription_v3.models import (
    ExtractionScout,
    ExtractionScoutRegion,
    ProfileName,
    SlicePageMapEntry,
    TranscriptionJob,
)

PROFILE_LIMITS: dict[ProfileName, tuple[int, int, int]] = {
    "single_column_prose_verbatim": (4, 10, 5),
    "two_column_prose_verbatim": (3, 5, 4),
    "dense_prose_verbatim": (2, 4, 3),
    "bibliography_verbatim": (1, 1, 1),
    "table_faithful": (1, 2, 1),
    "algorithm_faithful": (1, 2, 1),
    "mixed_layout_verbatim": (2, 5, 3),
    "scanned_page_verbatim": (1, 2, 1),
}


def _regions_for_page(scout: ExtractionScout | None, page: int) -> list[ExtractionScoutRegion]:
    return [
        region
        for region in (scout.regions if scout else [])
        if region.page_start <= page <= region.page_end
    ]


def profile_for_page(
    page: PdfPagePreflight, scout: ExtractionScout | None = None
) -> ProfileName:
    kinds = {region.region_kind for region in _regions_for_page(scout, page.page_number)}
    if "bibliography" in kinds:
        return "bibliography_verbatim"
    if "table" in kinds:
        return "table_faithful"
    if "algorithm" in kinds or "decision_tree" in kinds:
        return "algorithm_faithful"
    if page.image_heavy or "scanned_image_heavy" in kinds:
        return "scanned_page_verbatim"
    if page.approximate_page_density_score >= 2.0 or "dense_region" in kinds:
        return "dense_prose_verbatim"
    if "multi_column" in kinds:
        return "two_column_prose_verbatim"
    if "layout_change" in kinds:
        return "mixed_layout_verbatim"
    return "single_column_prose_verbatim"


def _make_job(
    *,
    source_id: str,
    profile: ProfileName,
    primary_pages: list[int],
    page_count: int,
    ordinal: int,
    reason: str,
) -> TranscriptionJob:
    context = sorted(
        {
            p
            for page in primary_pages
            for p in (page - 1, page + 1)
            if 1 <= p <= page_count and p not in primary_pages
        }
    )
    all_pages = sorted([*context, *primary_pages])
    role_map = {
        page: (
            "primary"
            if page in primary_pages
            else "previous_context"
            if page < min(primary_pages)
            else "next_context"
        )
        for page in all_pages
    }
    job_id = f"tx3-{ordinal:04d}-p{primary_pages[0]:04d}-{primary_pages[-1]:04d}"
    return TranscriptionJob(
        job_id=job_id,
        chunk_id=f"{source_id}-chunk-{ordinal:04d}",
        profile=profile,
        primary_pages=primary_pages,
        context_pages=context,
        slice_page_map=[
            SlicePageMapEntry(
                slice_page_index=index,
                original_pdf_page_number=page,
                role=role_map[page],
            )
            for index, page in enumerate(all_pages, start=1)
        ],
        reason=reason,
    )


def build_transcription_plan(
    *,
    preflight: PdfPreflight,
    pages: list[PdfPagePreflight],
    scout: ExtractionScout | None = None,
    planner_mode: str = "deterministic",
    limit: int | None = None,
) -> list[TranscriptionJob]:
    """Plan bounded jobs; hybrid refinement is intentionally bounded to known profiles."""
    if planner_mode not in {"deterministic", "hybrid"}:
        raise ValueError("planner_mode must be deterministic or hybrid")
    selected_pages = pages[:limit] if limit else pages
    jobs: list[TranscriptionJob] = []
    ordinal = 1
    index = 0
    while index < len(selected_pages):
        page = selected_pages[index]
        profile = profile_for_page(page, scout)
        _minimum, maximum, default = PROFILE_LIMITS[profile]
        if profile in {"bibliography_verbatim", "table_faithful", "algorithm_faithful"}:
            span = 1
        else:
            span = min(default, maximum)
        primary: list[int] = []
        while index < len(selected_pages) and len(primary) < span:
            candidate = selected_pages[index]
            if profile_for_page(candidate, scout) != profile:
                break
            primary.append(candidate.page_number)
            index += 1
        jobs.append(
            _make_job(
                source_id=preflight.source_id,
                profile=profile,
                primary_pages=primary,
                page_count=preflight.page_count,
                ordinal=ordinal,
                reason=f"{planner_mode} bounded profile {profile}",
            )
        )
        ordinal += 1
    return jobs
