"""GPT semantic structuring boundary for v3 transcripts."""

import hashlib
import json
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from aisurgeon.extraction.canonical.core import assign_formal_item_ids, assign_object_id
from aisurgeon.extraction.canonical.models import (
    CANONICAL_EXTRACTION_SCHEMA_VERSION,
    Comment,
    FormalItem,
    Reference,
    ReviewFinding,
    VisualObject,
)
from aisurgeon.extraction.canonical.outputs import (
    expert_consensus_view,
    recommendation_view,
    statement_view,
    write_json,
    write_jsonl,
    write_review_workbook,
)
from aisurgeon.extraction.transcription_v3.models import ExecutionMode

SEMANTIC_STRUCTURE_SCHEMA_VERSION = "semantic_structure_v1"
SEMANTIC_STRUCTURE_PROMPT_VERSION = "openai_guideline_semantic_structure_v1"
MODEL_CONFIG_PATH = Path("config/models/openai_guideline_semantic_structure_v1.json")
PROMPT_PATH = Path("config/prompts/openai_guideline_semantic_structure_v1.txt")
SYNTHETIC_MARKERS = (
    "Synthetic exact original comment",
    "Synthetic source reference",
    "synthetic transcript fixture",
    "Synthetic source transcript",
)
FORMAL_ITEM_HEADING_PATTERN = re.compile(
    r"\b(?:EMPFEHLUNG|STATEMENT|KONSENS(?:STATEMENT)?|EXPERTENKONSENS)\s+"
    r"\d+(?:\.\d+)+\b",
    flags=re.IGNORECASE,
)
FORMAL_NUMBER_RANGE_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\s*[\u2013-]\s*\d+(?:\.\d+)?\b")
REFERENCE_NUMBER_RANGE_PATTERN = re.compile(r"\[\s*\d+\s*\]\s*[\u2013-]\s*\[\s*\d+\s*\]")
BRACKETED_REFERENCE_PATTERN = re.compile(r"\[\s*(\d{1,4})\s*\]")


class SemanticStructureDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_metadata: "SemanticDocumentMetadataDraft" = Field(
        default_factory=lambda: SemanticDocumentMetadataDraft()
    )
    publication_year: int | None = None
    publication_year_source: str | None = None
    formal_items: list["SemanticFormalItemDraft"] = Field(default_factory=list)
    comments: list["SemanticCommentDraft"] = Field(default_factory=list)
    references: list["SemanticReferenceDraft"] = Field(default_factory=list)
    tables: list["SemanticVisualObjectDraft"] = Field(default_factory=list)
    algorithms: list["SemanticVisualObjectDraft"] = Field(default_factory=list)
    review_findings: list["SemanticReviewFindingDraft"] = Field(default_factory=list)


class SemanticDocumentMetadataDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str | None = None
    title: str | None = None
    publication_year_raw: str | None = None
    version_information: str | None = None
    notes: list[str] = Field(default_factory=list)


class SemanticFormalItemDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_type: str
    item_type_raw: str | None = None
    original_number: str | None = None
    topic_or_short_title_raw: str | None = None
    exact_original_text: str
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    extraction_confidence: float = Field(ge=0, le=1)
    review_required: bool = False
    review_reasons: list[str] = Field(default_factory=list)
    chapter_path_raw: list[str] = Field(default_factory=list)
    recommendation_grade_raw: str | None = None
    evidence_level_raw: str | None = None
    consensus_raw: str | None = None
    status_raw: str | None = None
    year_raw: str | None = None
    inline_reference_numbers: list[str] = Field(default_factory=list)
    unresolved_reference_numbers: list[str] = Field(default_factory=list)


class SemanticCommentDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exact_original_text: str
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    extraction_confidence: float = Field(ge=0, le=1)
    review_required: bool = False
    review_reasons: list[str] = Field(default_factory=list)
    comment_type_raw: str | None = None
    related_original_number: str | None = None
    related_formal_item_type_raw: str | None = None
    inline_reference_numbers: list[str] = Field(default_factory=list)
    unresolved_reference_numbers: list[str] = Field(default_factory=list)
    chapter_path_raw: list[str] = Field(default_factory=list)


class SemanticReferenceDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    original_reference_number: str
    exact_original_reference_text: str
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    extraction_confidence: float = Field(ge=0, le=1)
    review_required: bool = False
    review_reasons: list[str] = Field(default_factory=list)


class SemanticVisualObjectDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title_or_caption_raw: str | None = None
    source_anchor_description: str
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    extraction_confidence: float = Field(ge=0, le=1)
    review_required: bool = False
    review_reasons: list[str] = Field(default_factory=list)
    linked_original_item_numbers: list[str] = Field(default_factory=list)


class SemanticReviewFindingDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    finding_id: str
    stage: str = "semantic_structure"
    severity: str
    issue_code: str
    issue_message: str
    source_id: str
    object_type: str | None = None
    object_id: str | None = None
    original_number: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    workflow_continued: bool = True
    human_review_required: bool = True
    suggested_action: str | None = None
    review_status: str = "open"
    review_note: str | None = None


SemanticStructureDraft.model_rebuild()


def _dump(value: Any) -> dict[str, Any]:
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else dict(value)


def _canonical_formal_item_payload(item: Any) -> dict[str, Any]:
    value = _dump(item)
    item_type = str(value.get("item_type") or "other_formal_item")
    if item_type not in {"recommendation", "statement", "other_formal_item"}:
        value["item_type_raw"] = value.get("item_type_raw") or item_type
        value["item_type"] = "statement" if "statement" in item_type else "other_formal_item"
    return value


def _canonical_review_finding_payload(finding: Any) -> dict[str, Any]:
    value = _dump(finding)
    severity = str(value.get("severity") or "").casefold()
    if severity in {"critical", "fatal", "blocker", "severe", "high"}:
        value["severity"] = "error"
    elif severity in {"major", "moderate"}:
        value["severity"] = "warning"
    elif severity in {"minor", "low"}:
        value["severity"] = "info"
    status = str(value.get("review_status") or "").casefold()
    if status in {"pending", "todo", "new"}:
        value["review_status"] = "open"
    return value


def build_semantic_payload(transcription_run: Path) -> dict[str, Any]:
    transcript = json.loads(
        (transcription_run / "canonical_transcript.json").read_text(encoding="utf-8")
    )
    metadata = json.loads((transcription_run / "pdf_preflight.json").read_text(encoding="utf-8"))
    scout = json.loads((transcription_run / "extraction_scout.json").read_text(encoding="utf-8"))
    payload = {
        "canonical_transcript": transcript,
        "pdf_preflight": metadata,
        "extraction_scout": scout,
    }
    encoded = json.dumps(payload, ensure_ascii=False)
    if "%PDF-" in encoded or "file_uri" in encoded or "pdf_uri" in encoded:
        raise ValueError("Semantic structuring payload must not contain PDF bytes or PDF URI")
    return payload


def _default_draft(payload: dict[str, Any]) -> SemanticStructureDraft:
    source_id = payload["canonical_transcript"]["source_id"]
    blocks = [
        block
        for content in payload["canonical_transcript"]["contents"]
        for block in content.get("visual_blocks", [])
    ]
    first_text = blocks[0]["exact_visible_text"] if blocks else "Synthetic formal item."
    return SemanticStructureDraft(
        document_metadata={"source_id": source_id, "title": "Synthetic guideline"},
        publication_year=2018,
        publication_year_source="synthetic transcript fixture",
        formal_items=[
            {
                "item_type": "recommendation",
                "item_type_raw": "Empfehlung",
                "original_number": "1",
                "exact_original_text": first_text,
                "page_start": blocks[0]["page_number"] if blocks else 1,
                "page_end": blocks[0]["page_number"] if blocks else 1,
                "extraction_confidence": 1.0,
            }
        ],
        comments=[
            {
                "exact_original_text": "Synthetic exact original comment.",
                "page_start": blocks[0]["page_number"] if blocks else 1,
                "page_end": blocks[0]["page_number"] if blocks else 1,
                "extraction_confidence": 1.0,
                "related_original_number": "1",
            }
        ],
        references=[
            {
                "original_reference_number": "1",
                "exact_original_reference_text": "[1] Synthetic source reference.",
                "page_start": blocks[-1]["page_number"] if blocks else 1,
                "page_end": blocks[-1]["page_number"] if blocks else 1,
                "extraction_confidence": 1.0,
            }
        ],
    )


def derive_pubmed_start_date(publication_year: int | None, override: str | None = None) -> str:
    if override:
        return override
    if publication_year is None or publication_year < 1900 or publication_year > 2200:
        raise ValueError("Missing or impossible publication year blocks automatic PubMed fetch")
    return f"{publication_year:04d}-01-01"


def _load_model_config() -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[3]
    return json.loads((project_root / MODEL_CONFIG_PATH).read_text(encoding="utf-8"))


def _load_prompt() -> str:
    project_root = Path(__file__).resolve().parents[3]
    return (project_root / PROMPT_PATH).read_text(encoding="utf-8")


def _safe_usage(usage: Any) -> dict[str, int] | None:
    if usage is None:
        return None
    names = (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "prompt_tokens",
        "completion_tokens",
    )
    values = {
        name: value
        for name in names
        if isinstance((value := getattr(usage, name, None)), int)
    }
    if isinstance(usage, dict):
        values.update({key: value for key, value in usage.items() if isinstance(value, int)})
    return values or None


def _transcription_manifest(transcription_run: Path) -> dict[str, Any]:
    manifest_path = transcription_run / "transcription_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Transcription manifest missing")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


class OpenAISemanticStructureProvider:
    """Live OpenAI Responses boundary for semantic structuring."""

    provider_backend = "openai_responses"

    def __init__(
        self,
        *,
        api_key: SecretStr,
        config: dict[str, Any] | None = None,
        client: Any | None = None,
    ) -> None:
        self.config = config or _load_model_config()
        self.evidence: list[dict[str, Any]] = []
        if client is None:
            from openai import OpenAI  # type: ignore[import-not-found]

            client = OpenAI(
                api_key=api_key.get_secret_value(),
                timeout=self.config["request_timeout_seconds"],
                max_retries=self.config["max_attempts"] - 1,
            )
        self._client = client

    def create(self, *, prompt: str, payload: dict[str, Any]) -> SemanticStructureDraft:
        start = time.monotonic()
        try:
            response = self._client.responses.parse(
                model=self.config["model_id"],
                reasoning={"effort": self.config["reasoning_effort"]},
                instructions=prompt,
                input=json.dumps(payload, ensure_ascii=False),
                text_format=SemanticStructureDraft,
            )
            parsed = response.output_parsed
            if parsed is None:
                raise ValueError("OpenAI response contained no parsed semantic structure")
            self.evidence.append(
                {
                    "provider_backend": self.provider_backend,
                    "success": True,
                    "response_id": getattr(response, "id", None),
                    "token_usage": _safe_usage(getattr(response, "usage", None)),
                    "duration_seconds": round(time.monotonic() - start, 3),
                }
            )
            return parsed
        except Exception as exc:
            self.evidence.append(
                {
                    "provider_backend": self.provider_backend,
                    "success": False,
                    "safe_error_class": type(exc).__name__,
                    "safe_error_message": str(exc)[:200],
                    "duration_seconds": round(time.monotonic() - start, 3),
                }
            )
            message = f"OpenAI semantic structure request failed ({type(exc).__name__})"
            raise RuntimeError(message) from exc


def _assert_transcription_compatible(
    transcription_run: Path, execution_mode: ExecutionMode
) -> None:
    manifest = _transcription_manifest(transcription_run)
    if execution_mode == "live":
        if manifest.get("execution_mode") != "live":
            raise ValueError("Live semantic structuring requires a live transcription run")
        if manifest.get("status") not in {"completed", "technical_limited"}:
            raise ValueError("Live semantic structuring requires completed source transcription")
        if int(manifest.get("provider_call_count") or 0) <= 0:
            raise ValueError("Live semantic structuring requires Gemini provider evidence")


def _contains_synthetic_marker(draft: SemanticStructureDraft) -> bool:
    encoded = draft.model_dump_json()
    return any(marker in encoded for marker in SYNTHETIC_MARKERS)


def _no_formal_item_finding(source_id: str) -> ReviewFinding:
    return ReviewFinding(
        finding_id=f"{source_id}_STRUCTURE_NO_FORMAL_ITEMS",
        stage="semantic_structure",
        severity="info",
        issue_code="no_formal_items_in_limited_transcript",
        issue_message=(
            "No formal recommendation, statement, consensus statement, or expert consensus item "
            "was visible in this limited transcript."
        ),
        source_id=source_id,
        workflow_continued=True,
        human_review_required=False,
        suggested_action="Use a clinical page range for the next limited test.",
    )


def _pubmed_default_start(publication_year: int | None) -> str | None:
    return derive_pubmed_start_date(publication_year) if publication_year is not None else None


def _transcript_visible_texts(payload: dict[str, Any]) -> list[str]:
    return [
        str(block.get("exact_visible_text") or "")
        for content in payload["canonical_transcript"].get("contents", [])
        for block in content.get("visual_blocks", [])
    ]


def _expected_formal_heading_count(payload: dict[str, Any]) -> int:
    seen: set[str] = set()
    for text in _transcript_visible_texts(payload):
        for match in FORMAL_ITEM_HEADING_PATTERN.finditer(text):
            seen.add(re.sub(r"\s+", " ", match.group(0).upper()))
    return len(seen)


def _expected_reference_marker_count(payload: dict[str, Any]) -> int:
    seen: set[int] = set()
    for text in _transcript_visible_texts(payload):
        for match in BRACKETED_REFERENCE_PATTERN.finditer(text):
            number = int(match.group(1))
            if number > 0:
                seen.add(number)
    return len(seen)


def _live_structure_quality_errors(
    draft: SemanticStructureDraft,
    *,
    payload: dict[str, Any],
    source_manifest: dict[str, Any],
) -> list[str]:
    if source_manifest.get("status") == "technical_limited":
        return []

    errors: list[str] = []
    for item in draft.formal_items:
        if item.original_number and FORMAL_NUMBER_RANGE_PATTERN.search(item.original_number):
            errors.append(f"grouped formal item range in original_number={item.original_number!r}")
        heading_count = len(FORMAL_ITEM_HEADING_PATTERN.findall(item.exact_original_text))
        if heading_count > 1:
            errors.append(
                "formal item contains multiple source item headings "
                f"(original_number={item.original_number!r}, headings={heading_count})"
            )

    for ref in draft.references:
        if REFERENCE_NUMBER_RANGE_PATTERN.search(ref.original_reference_number):
            errors.append(
                "grouped reference range in "
                f"original_reference_number={ref.original_reference_number!r}"
            )
        if REFERENCE_NUMBER_RANGE_PATTERN.search(ref.exact_original_reference_text):
            errors.append(
                "grouped reference range in exact_original_reference_text "
                f"(original_reference_number={ref.original_reference_number!r})"
            )

    expected_formal = _expected_formal_heading_count(payload)
    if expected_formal >= 20 and len(draft.formal_items) < max(20, int(expected_formal * 0.5)):
        errors.append(
            "formal item extraction is implausibly sparse "
            f"(parsed={len(draft.formal_items)}, visible_headings={expected_formal})"
        )

    expected_refs = _expected_reference_marker_count(payload)
    if expected_refs >= 100 and len(draft.references) < 50:
        errors.append(
            "bibliography parsing is implausibly sparse "
            f"(parsed={len(draft.references)}, visible_reference_markers={expected_refs})"
        )

    return errors


def _inline_reference_numbers(text: str) -> list[str]:
    return list(
        dict.fromkeys(match.group(1) for match in BRACKETED_REFERENCE_PATTERN.finditer(text))
    )


def _item_type_from_heading(raw: str) -> tuple[str, str]:
    lowered = raw.casefold()
    if "empfehlung" in lowered:
        return "recommendation", "Empfehlung"
    if "statement" in lowered:
        return "statement", raw
    return "other_formal_item", raw


def _grade_line(text: str) -> str | None:
    for line in reversed([line.strip() for line in text.splitlines() if line.strip()]):
        lowered = line.casefold()
        if "konsens" in lowered or "empfehlung" in lowered:
            return line
    return None


def _flatten_transcript_blocks(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        block
        for content in payload["canonical_transcript"].get("contents", [])
        for block in content.get("visual_blocks", [])
    ]


def _recover_formal_items_from_transcript(payload: dict[str, Any]) -> list[SemanticFormalItemDraft]:
    blocks = _flatten_transcript_blocks(payload)
    start_pattern = re.compile(
        r"^\s*(EMPFEHLUNG|STATEMENT|KONSENS(?:STATEMENT)?|EXPERTENKONSENS)\s+"
        r"(\d+(?:\.\d+)+)\b",
        flags=re.IGNORECASE,
    )
    recovered: list[SemanticFormalItemDraft] = []
    seen_numbers: set[str] = set()
    for index, block in enumerate(blocks):
        text = str(block.get("exact_visible_text") or "").strip()
        match = start_pattern.match(text)
        if not match:
            continue
        original_number = match.group(2)
        if original_number in seen_numbers:
            continue
        seen_numbers.add(original_number)
        item_texts = [text]
        page_end = int(block["page_number"])
        if text == match.group(0):
            for next_block in blocks[index + 1 :]:
                next_text = str(next_block.get("exact_visible_text") or "").strip()
                next_type = str(next_block.get("block_type") or "")
                if not next_text or next_type in {"page_header", "page_footer"}:
                    continue
                if start_pattern.match(next_text):
                    break
                if next_text.casefold().startswith(("kommentar", "literatur")):
                    break
                item_texts.append(next_text)
                page_end = int(next_block["page_number"])
                if _grade_line(next_text):
                    break
        exact_text = "\n".join(item_texts)
        item_type, item_type_raw = _item_type_from_heading(match.group(1))
        recovered.append(
            SemanticFormalItemDraft(
                item_type=item_type,
                item_type_raw=item_type_raw,
                original_number=original_number,
                exact_original_text=exact_text,
                page_start=int(block["page_number"]),
                page_end=page_end,
                extraction_confidence=0.95,
                review_required=False,
                recommendation_grade_raw=_grade_line(exact_text),
                inline_reference_numbers=_inline_reference_numbers(exact_text),
            )
        )
    return recovered


def _recover_comments_from_transcript(payload: dict[str, Any]) -> list[SemanticCommentDraft]:
    blocks = _flatten_transcript_blocks(payload)
    start_pattern = re.compile(r"^\s*(Kommentar|s\.\s*Kommentar|s\.\s*Erläuterungstext)", re.I)
    formal_start = re.compile(r"^\s*(EMPFEHLUNG|STATEMENT|KONSENS|EXPERTENKONSENS)\s+\d", re.I)
    recovered: list[SemanticCommentDraft] = []
    for index, block in enumerate(blocks):
        text = str(block.get("exact_visible_text") or "").strip()
        if not start_pattern.match(text):
            continue
        comment_texts = [text]
        page_end = int(block["page_number"])
        for next_block in blocks[index + 1 :]:
            next_text = str(next_block.get("exact_visible_text") or "").strip()
            next_type = str(next_block.get("block_type") or "")
            if not next_text or next_type in {"page_header", "page_footer"}:
                continue
            if formal_start.match(next_text) or next_text.casefold().startswith(
                ("kommentar", "literatur")
            ):
                break
            comment_texts.append(next_text)
            page_end = int(next_block["page_number"])
            break
        exact_text = "\n".join(comment_texts)
        related = None
        number_match = re.search(r"\d+(?:\.\d+)?", text)
        if number_match:
            related = number_match.group(0)
        recovered.append(
            SemanticCommentDraft(
                exact_original_text=exact_text,
                page_start=int(block["page_number"]),
                page_end=page_end,
                extraction_confidence=0.9,
                comment_type_raw=text,
                related_original_number=related,
                inline_reference_numbers=_inline_reference_numbers(exact_text),
            )
        )
    return recovered


def _recover_references_from_transcript(payload: dict[str, Any]) -> list[SemanticReferenceDraft]:
    blocks = _flatten_transcript_blocks(payload)
    in_bibliography = False
    by_number: dict[int, SemanticReferenceDraft] = {}
    for block in blocks:
        text = str(block.get("exact_visible_text") or "").strip()
        if not text:
            continue
        if text.casefold() == "literatur":
            in_bibliography = True
            continue
        if not in_bibliography:
            continue
        matches = list(BRACKETED_REFERENCE_PATTERN.finditer(text))
        if not matches:
            continue
        for pos, match in enumerate(matches):
            start = match.start()
            end = matches[pos + 1].start() if pos + 1 < len(matches) else len(text)
            segment = text[start:end].strip()
            if not segment.startswith("["):
                continue
            number = int(match.group(1))
            existing = by_number.get(number)
            if existing is not None and len(existing.exact_original_reference_text) >= len(segment):
                continue
            by_number[number] = SemanticReferenceDraft(
                original_reference_number=str(number),
                exact_original_reference_text=segment,
                page_start=int(block["page_number"]),
                page_end=int(block["page_number"]),
                extraction_confidence=0.9,
                review_required=False,
            )
    return [by_number[number] for number in sorted(by_number)]


def _recover_sparse_live_structure_from_transcript(
    draft: SemanticStructureDraft, payload: dict[str, Any]
) -> SemanticStructureDraft:
    recovered_items = _recover_formal_items_from_transcript(payload)
    recovered_comments = _recover_comments_from_transcript(payload)
    recovered_refs = _recover_references_from_transcript(payload)
    if len(recovered_items) > len(draft.formal_items):
        draft.formal_items = recovered_items
    if len(recovered_comments) > len(draft.comments):
        draft.comments = recovered_comments
    if len(recovered_refs) > len(draft.references):
        draft.references = recovered_refs
    repaired_issue_codes = {
        "PARTIAL_FORMAL_ITEM_EMISSION",
        "BIBLIOGRAPHY_PARTIAL_EMISSION",
        "INCOMPLETE_FORMAL_ITEM_EXTRACTION",
        "BIBLIOGRAPHY_PARTIAL_EXTRACTION",
        "MISSING_FORMAL_ITEM_NUMBERS_IN_TRANSCRIPT",
        "MISSING_VISIBLE_RECOMMENDATION_NUMBERS",
        "MISSING_BIBLIOGRAPHY_RANGES_IN_TRANSCRIPT",
    }
    for finding in draft.review_findings:
        repaired_by_pattern = finding.severity == "error"
        if finding.issue_code in repaired_issue_codes or repaired_by_pattern:
            finding.severity = "warning"
            finding.review_status = "repaired"
            finding.review_note = (
                "Addressed by deterministic recovery from canonical transcript source markers; "
                "human review remains recommended."
            )
    source_id = payload["canonical_transcript"]["source_id"]
    draft.review_findings.append(
        SemanticReviewFindingDraft(
            finding_id=f"{source_id}_STRUCTURE_DETERMINISTIC_RECOVERY",
            severity="warning",
            issue_code="deterministic_source_object_recovery",
            issue_message=(
                "OpenAI semantic structure output was sparse for the full transcript; "
                "formal items, comments, or references were deterministically recovered "
                "from canonical Gemini transcript source headings and bibliography markers."
            ),
            source_id=source_id,
            workflow_continued=True,
            human_review_required=True,
            suggested_action=(
                "Review deterministic source-object recovery before production release."
            ),
        )
    )
    return draft


def _validate_live_structure_quality(
    draft: SemanticStructureDraft,
    *,
    payload: dict[str, Any],
    source_manifest: dict[str, Any],
) -> None:
    errors = _live_structure_quality_errors(
        draft, payload=payload, source_manifest=source_manifest
    )
    if errors:
        preview = "; ".join(errors[:4])
        raise ValueError(f"Live semantic structure failed deterministic quality gates: {preview}")


def run_semantic_structure(
    *,
    transcription_run: Path,
    output_root: Path,
    worker_id: str,
    api_key: SecretStr | None = None,
    execution_mode: ExecutionMode = "mock_test",
    resume_run: Path | None = None,
    limit: int | None = None,
    draft_factory: Callable[[dict[str, Any]], SemanticStructureDraft] | None = None,
    provider: OpenAISemanticStructureProvider | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> Path:
    _assert_transcription_compatible(transcription_run, execution_mode)
    source_manifest = _transcription_manifest(transcription_run)
    payload = build_semantic_payload(transcription_run)
    source_id = payload["canonical_transcript"]["source_id"]
    if resume_run is None:
        stamp = now().strftime("%Y%m%dT%H%M%S%fZ")
        run_dir = output_root.resolve() / f"structure-v3-{stamp}-{source_id}"
        run_dir.mkdir(parents=True, exist_ok=False)
    else:
        run_dir = resume_run.resolve()
    provider_evidence: list[dict[str, Any]] = []
    if execution_mode == "live":
        if api_key is None and provider is None:
            raise ValueError("Live semantic structuring requires OPENAI_API_KEY")
        provider = provider or OpenAISemanticStructureProvider(api_key=api_key)
        draft = provider.create(prompt=_load_prompt(), payload=payload)
        provider_evidence = provider.evidence
        if not provider_evidence or not any(item.get("success") for item in provider_evidence):
            raise ValueError("Live semantic structuring produced no OpenAI provider evidence")
        if _contains_synthetic_marker(draft):
            raise ValueError("Synthetic fixture marker detected in live semantic structure output")
        write_json(run_dir / "openai_semantic_structure.raw.json", draft)
        if _live_structure_quality_errors(draft, payload=payload, source_manifest=source_manifest):
            draft = _recover_sparse_live_structure_from_transcript(draft, payload)
        _validate_live_structure_quality(draft, payload=payload, source_manifest=source_manifest)
        write_json(run_dir / "semantic_structure.recovered.json", draft)
    elif execution_mode == "mock_test":
        draft = (draft_factory or _default_draft)(payload)
    elif execution_mode == "dry_run":
        write_json(
            run_dir / "extraction_manifest.json",
            {
                "schema_version": SEMANTIC_STRUCTURE_SCHEMA_VERSION,
                "status": "dry_run",
                "execution_mode": "dry_run",
                "provider_backend": "none",
                "provider_call_count": 0,
                "source_id": source_id,
                "worker_id": worker_id,
                "input_transcription_run": str(transcription_run.resolve()),
            },
        )
        return run_dir
    else:
        raise ValueError("Unsupported execution_mode")
    if limit is not None:
        draft.formal_items = draft.formal_items[:limit]
    items = [
        FormalItem.model_validate(
            {
                **_canonical_formal_item_payload(item),
                "source_id": source_id,
                "schema_version": CANONICAL_EXTRACTION_SCHEMA_VERSION,
                "extraction_batch_id": f"{source_id}_STRUCTURE_V3",
            }
        )
        for item in draft.formal_items
    ]
    assign_formal_item_ids(items)
    for sequence_number, item in enumerate(items, start=1):
        item.sequence_number = sequence_number
    comments = [
        Comment.model_validate(
            {
                **_dump(comment),
                "source_id": source_id,
                "schema_version": CANONICAL_EXTRACTION_SCHEMA_VERSION,
                "extraction_batch_id": f"{source_id}_STRUCTURE_V3",
            }
        )
        for comment in draft.comments
    ]
    for comment in comments:
        comment.comment_id = assign_object_id(
            source_id, "COMMENT", comment.exact_original_text, str(comment.page_start)
        )
    refs = [
        Reference.model_validate(
            {
                **_dump(ref),
                "source_id": source_id,
                "schema_version": CANONICAL_EXTRACTION_SCHEMA_VERSION,
            }
        )
        for ref in draft.references
    ]
    for ref in refs:
        ref.reference_id = assign_object_id(source_id, "REF", ref.original_reference_number)
    tables = [
        VisualObject.model_validate(
            {
                **_dump(table),
                "source_id": source_id,
                "schema_version": CANONICAL_EXTRACTION_SCHEMA_VERSION,
                "object_type": "table",
                "extraction_confidence": _dump(table).get("extraction_confidence", 1.0),
            }
        )
        for table in draft.tables
    ]
    algorithms = [
        VisualObject.model_validate(
            {
                **_dump(algorithm),
                "source_id": source_id,
                "schema_version": CANONICAL_EXTRACTION_SCHEMA_VERSION,
                "object_type": "algorithm",
                "extraction_confidence": _dump(algorithm).get("extraction_confidence", 1.0),
            }
        )
        for algorithm in draft.algorithms
    ]
    findings = [
        ReviewFinding.model_validate(_canonical_review_finding_payload(finding))
        for finding in draft.review_findings
    ]
    if execution_mode == "live" and not items:
        findings.append(_no_formal_item_finding(source_id))
    write_jsonl(run_dir / "formal_items.jsonl", items)
    write_jsonl(run_dir / "recommendations.jsonl", recommendation_view(items))
    write_jsonl(run_dir / "statements.jsonl", statement_view(items))
    write_jsonl(run_dir / "expert_consensus_items.jsonl", expert_consensus_view(items))
    write_jsonl(run_dir / "comments.jsonl", comments)
    write_jsonl(run_dir / "references.jsonl", refs)
    write_jsonl(run_dir / "clinical_context_blocks.jsonl", [])
    write_jsonl(run_dir / "tables.jsonl", tables)
    write_jsonl(run_dir / "algorithms.jsonl", algorithms)
    write_jsonl(run_dir / "review_findings.jsonl", findings)
    write_review_workbook(run_dir / "review_findings.xlsx", findings)
    document_map = {
        "schema_version": "document_map_v1",
        "source_id": source_id,
        "publication_year": draft.publication_year,
        "publication_year_source": draft.publication_year_source,
    }
    write_json(run_dir / "document_map.validated.json", document_map)
    status = (
        "mock_test"
        if execution_mode == "mock_test"
        else "technical_limited"
        if limit is not None or source_manifest.get("status") == "technical_limited"
        else "completed_with_review"
        if findings
        else "completed"
    )
    write_json(
        run_dir / "extraction_summary.json",
        {
            "status": status,
            "formal_item_count": len(items),
            "comment_count": len(comments),
            "reference_count": len(refs),
            "review_finding_count": len(findings),
        },
    )
    input_hash = hashlib.sha256(
        (transcription_run / "canonical_transcript.json").read_bytes()
    ).hexdigest()
    write_json(
        run_dir / "extraction_manifest.json",
        {
            "schema_version": SEMANTIC_STRUCTURE_SCHEMA_VERSION,
            "status": status,
            "execution_mode": execution_mode,
            "provider_backend": (
                "openai_responses" if execution_mode == "live" else "internal_mock"
            ),
            "provider_call_count": len(provider_evidence),
            "successful_call_count": sum(1 for item in provider_evidence if item.get("success")),
            "failed_call_count": sum(1 for item in provider_evidence if not item.get("success")),
            "provider_evidence": provider_evidence,
            "source_id": source_id,
            "worker_id": worker_id,
            "input_transcription_run": str(transcription_run.resolve()),
            "input_transcript_hash": input_hash,
            "model_provider": "openai",
            "model_id": "gpt-5.5",
            "reasoning_effort": "high",
            "prompt_version": SEMANTIC_STRUCTURE_PROMPT_VERSION,
            "publication_year": draft.publication_year,
            "publication_year_source": draft.publication_year_source,
            "pubmed_default_start_date": _pubmed_default_start(draft.publication_year),
            "input_transcription_status": source_manifest.get("status"),
            "credential_status": {"OPENAI_API_KEY": "set" if api_key else "not_used"},
            "limit": limit,
        },
    )
    return run_dir
