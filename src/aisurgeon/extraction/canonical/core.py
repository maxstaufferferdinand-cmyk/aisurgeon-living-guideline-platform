"""Deterministic windowing, identity, merge, links, and review decisions."""

import hashlib
import re
from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, ConfigDict

from aisurgeon.extraction.canonical.models import (
    Comment,
    FormalItem,
    Reference,
    ReviewFinding,
    UnresolvedLink,
)
from aisurgeon.extraction.gemini.models import PageRange


class PageWindow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    job_id: str
    stage: Literal["clinical", "references"]
    primary_page_start: int
    primary_page_end: int
    context_page_start: int
    context_page_end: int


def plan_windows(
    ranges: Iterable[PageRange],
    *,
    stage: Literal["clinical", "references"],
    pages_per_job: int = 8,
    overlap_pages: int = 1,
    document_page_count: int,
) -> list[PageWindow]:
    if pages_per_job < 1 or overlap_pages < 0:
        raise ValueError("Fenstergrößen sind ungültig.")
    windows: list[PageWindow] = []
    for region in ranges:
        start = max(1, region.page_start)
        end = min(document_page_count, region.page_end)
        for primary_start in range(start, end + 1, pages_per_job):
            primary_end = min(end, primary_start + pages_per_job - 1)
            windows.append(
                PageWindow(
                    job_id=f"{stage}-{primary_start:04d}-{primary_end:04d}",
                    stage=stage,
                    primary_page_start=primary_start,
                    primary_page_end=primary_end,
                    context_page_start=max(1, primary_start - overlap_pages),
                    context_page_end=min(document_page_count, primary_end + overlap_pages),
                )
            )
    return windows


def _normalized_number(value: str | None) -> str | None:
    if not value or not value.strip():
        return None
    return re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").upper()


def _short_hash(*values: str) -> str:
    return hashlib.sha256("\x1f".join(values).encode()).hexdigest()[:12]


def normalize_item_family(item_type_raw: str | None, item_type: str) -> str:
    """Normalize source-native labels without changing their raw representation."""
    raw = (item_type_raw or "").strip().casefold()
    compact = re.sub(r"[^a-z0-9äöüß]+", " ", raw).strip()
    if (
        compact == "ek"
        or compact.startswith("ek ")
        or "expertenkonsens" in compact
        or "expert consensus" in compact
    ):
        return "expert_consensus"
    if (
        "konsensusstatement" in compact
        or "konsensbasiertes statement" in compact
        or "konsensbasiert statement" in compact
        or "consensus statement" in compact
        or "consensus based statement" in compact
    ):
        return "consensus_statement"
    if "statement" in compact or "aussage" in compact:
        return "statement"
    if "empfehlung" in compact or "recommendation" in compact:
        return "recommendation"
    if not compact and item_type in {"recommendation", "statement"}:
        return item_type
    return "other_formal_item"


def assign_formal_item_ids(items: list[FormalItem]) -> None:
    counters: dict[str, int] = {}
    labels = {
        "recommendation": "REC",
        "statement": "STMT",
        "consensus_statement": "CSTMT",
        "expert_consensus": "EK",
        "other_formal_item": "FORMAL",
    }
    for item in items:
        family = item.normalized_item_family or normalize_item_family(
            item.item_type_raw, item.item_type
        )
        item.normalized_item_family = family
        label = labels[family]
        number = _normalized_number(item.original_number)
        if number:
            item.formal_item_id = f"{item.source_id}_{label}_{number}"
        else:
            counters[label] = counters.get(label, 0) + 1
            digest = _short_hash(family, item.exact_original_text, str(item.page_start))
            item.formal_item_id = (
                f"{item.source_id}_{label}_{counters[label]:04d}_{digest}"
            )
        item.item_id = item.formal_item_id


def assign_object_id(source_id: str, kind: str, *parts: str) -> str:
    return f"{source_id}_{kind.upper()}_{_short_hash(*parts)}"


def make_finding(
    *,
    source_id: str,
    issue_code: str,
    issue_message: str,
    object_type: str | None = None,
    object_id: str | None = None,
    original_number: str | None = None,
    page_start: int | None = None,
    page_end: int | None = None,
    severity: Literal["info", "warning", "error"] = "warning",
) -> ReviewFinding:
    identity = "|".join(
        str(value or "")
        for value in (issue_code, object_type, object_id, original_number, page_start, page_end)
    )
    return ReviewFinding(
        finding_id=f"{source_id}_FINDING_{_short_hash(identity)}",
        stage="extraction",
        severity=severity,
        issue_code=issue_code,
        issue_message=issue_message,
        source_id=source_id,
        object_type=object_type,
        object_id=object_id,
        original_number=original_number,
        page_start=page_start,
        page_end=page_end,
        workflow_continued=severity != "error",
        human_review_required=severity != "info",
        suggested_action="Originalseiten und Objektzuordnung prüfen.",
    )


def merge_formal_items(items: list[FormalItem]) -> tuple[list[FormalItem], list[ReviewFinding]]:
    """Deduplicate exact overlap copies and retain conflicts with an audit finding."""
    merged: list[FormalItem] = []
    findings: list[ReviewFinding] = []
    exact_seen: set[tuple[str, str, str | None, str]] = set()
    number_text: dict[tuple[str, str, str | None], str] = {}
    # Pipeline batches and objects arrive in primary-window/source order. Preserve that
    # order because printed or model-reported page labels can differ from physical pages.
    ordered = enumerate(items)
    for _, item in ordered:
        item.normalized_item_family = normalize_item_family(item.item_type_raw, item.item_type)
        known_other = (item.item_type_raw or "").casefold()
        if item.normalized_item_family == "other_formal_item" and not any(
            label in known_other
            for label in ("good clinical practice", "gute klinische praxis", "gcp")
        ):
            item.review_required = True
            if "unknown_formal_item_type" not in item.review_reasons:
                item.review_reasons.append("unknown_formal_item_type")
        text_hash = _short_hash(item.exact_original_text)
        key = (
            item.source_id,
            item.normalized_item_family,
            _normalized_number(item.original_number),
            text_hash,
        )
        if key in exact_seen:
            findings.append(
                make_finding(
                    source_id=item.source_id,
                    issue_code="possible_duplicate",
                    issue_message=(
                        "Identische Überlappungsdublette deterministisch zusammengeführt."
                    ),
                    object_type="formal_item",
                    object_id=item.formal_item_id,
                    original_number=item.original_number,
                    page_start=item.page_start,
                    page_end=item.page_end,
                    severity="info",
                )
            )
            continue
        exact_seen.add(key)
        identity = key[:3]
        previous = number_text.get(identity)
        if previous is not None and previous != text_hash:
            item.review_required = True
            item.review_reasons.append("conflicting_duplicate")
            findings.append(
                make_finding(
                    source_id=item.source_id,
                    issue_code="possible_duplicate",
                    issue_message=(
                        "Gleiche formale Nummer mit abweichendem Originaltext; beide erhalten."
                    ),
                    object_type="formal_item",
                    object_id=item.formal_item_id,
                    original_number=item.original_number,
                    page_start=item.page_start,
                    page_end=item.page_end,
                )
            )
        number_text[identity] = text_hash
        merged.append(item)
    assign_formal_item_ids(merged)
    for sequence_number, item in enumerate(merged, start=1):
        item.sequence_number = sequence_number
    return merged, findings


def link_comments(comments: list[Comment], items: list[FormalItem]) -> list[ReviewFinding]:
    findings: list[ReviewFinding] = []
    for comment in comments:
        comment.comment_id = assign_object_id(
            comment.source_id,
            "COMMENT",
            comment.exact_original_text,
            str(comment.page_start),
            str(comment.page_end),
        )
        candidates = [
            item
            for item in items
            if comment.related_original_number
            and _normalized_number(item.original_number)
            == _normalized_number(comment.related_original_number)
        ]
        if not candidates:
            candidates = [item for item in items if item.page_end <= comment.page_start]
            candidates = sorted(candidates, key=lambda item: item.page_end, reverse=True)[:1]
        if len(candidates) == 1 and candidates[0].formal_item_id:
            comment.linked_formal_item_ids = [candidates[0].formal_item_id]
            comment.linked_item_ids = [candidates[0].formal_item_id]
            candidates[0].linked_comment_ids.append(comment.comment_id)
        else:
            comment.review_required = True
            comment.review_reasons.append("comment_link_uncertain")
            findings.append(
                make_finding(
                    source_id=comment.source_id,
                    issue_code="comment_link_uncertain",
                    issue_message=(
                        "Kommentar konnte keinem formalen Item eindeutig zugeordnet werden."
                    ),
                    object_type="comment",
                    object_id=comment.comment_id,
                    original_number=comment.related_original_number,
                    page_start=comment.page_start,
                    page_end=comment.page_end,
                )
            )
    return findings


def expand_reference_numbers(values: Iterable[str]) -> list[str]:
    expanded: list[str] = []
    for raw in values:
        cleaned = raw.strip().strip("[]()")
        for part in re.split(r"[,;]", cleaned):
            token = part.strip()
            match = re.fullmatch(r"(\d+)\s*(?:-|\u2013)\s*(\d+)", token)
            if match and int(match.group(2)) >= int(match.group(1)):
                expanded.extend(
                    str(value) for value in range(int(match.group(1)), int(match.group(2)) + 1)
                )
            elif token:
                expanded.append(token)
    return list(dict.fromkeys(expanded))


def link_references(
    items: list[FormalItem], comments: list[Comment], references: list[Reference]
) -> tuple[list[UnresolvedLink], list[ReviewFinding]]:
    available = {_normalized_number(ref.original_reference_number) for ref in references}
    unresolved: list[UnresolvedLink] = []
    findings: list[ReviewFinding] = []
    for kind, objects in (("formal_item", items), ("comment", comments)):
        for obj in objects:
            numbers = expand_reference_numbers(obj.inline_reference_numbers)
            obj.inline_reference_numbers = numbers
            missing = [number for number in numbers if _normalized_number(number) not in available]
            obj.unresolved_reference_numbers = missing
            if missing:
                obj.review_required = True
                obj.review_reasons.append("reference_unresolved")
                object_id = obj.formal_item_id if isinstance(obj, FormalItem) else obj.comment_id
                for number in missing:
                    unresolved.append(
                        UnresolvedLink(
                            source_id=obj.source_id,
                            object_type=kind,
                            object_id=object_id or "unassigned",
                            reference_number=number,
                        )
                    )
                findings.append(
                    make_finding(
                        source_id=obj.source_id,
                        issue_code="reference_unresolved",
                        issue_message="Mindestens eine Inline-Referenz ist nicht aufgelöst.",
                        object_type=kind,
                        object_id=object_id,
                        page_start=obj.page_start,
                        page_end=obj.page_end,
                    )
                )
    return unresolved, findings


def overall_status(findings: list[ReviewFinding], *, hard_failure: bool = False) -> str:
    if hard_failure:
        return "failed"
    return (
        "completed_with_review" if any(f.human_review_required for f in findings) else "completed"
    )
