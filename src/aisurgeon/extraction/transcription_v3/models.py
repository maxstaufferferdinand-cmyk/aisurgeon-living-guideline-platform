"""Source-only transcription v3 models."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CANONICAL_TRANSCRIPTION_SCHEMA_VERSION = "canonical_transcription_v3"
TRANSCRIPTION_PROMPT_VERSION = "gemini_source_transcription_v3"
SCOUT_SCHEMA_VERSION = "extraction_scout_v1"
SCOUT_PROMPT_VERSION = "gemini_technical_layout_scout_v1"
ExecutionMode = Literal["live", "dry_run", "mock_test"]

VisualBlockType = Literal[
    "heading",
    "paragraph",
    "list",
    "table",
    "caption",
    "footnote",
    "page_header",
    "page_footer",
    "diagram_text",
    "other_visible_text",
]
RegionKind = Literal[
    "front_matter",
    "table_of_contents",
    "normal_prose",
    "single_column",
    "multi_column",
    "dense_region",
    "bibliography",
    "appendix",
    "table",
    "algorithm",
    "decision_tree",
    "scanned_image_heavy",
    "layout_change",
]
ProfileName = Literal[
    "single_column_prose_verbatim",
    "two_column_prose_verbatim",
    "dense_prose_verbatim",
    "bibliography_verbatim",
    "table_faithful",
    "algorithm_faithful",
    "mixed_layout_verbatim",
    "scanned_page_verbatim",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExtractionScoutRegion(StrictModel):
    region_kind: RegionKind
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    notes: str | None = None

    @model_validator(mode="after")
    def page_order(self) -> "ExtractionScoutRegion":
        if self.page_end < self.page_start:
            raise ValueError("page_end before page_start")
        return self


class ExtractionScoutDraft(StrictModel):
    """Gemini-returned scout content. No Python technical constants are allowed."""

    declared_page_count: int = Field(ge=1)
    regions: list[ExtractionScoutRegion] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ExtractionScout(ExtractionScoutDraft):
    schema_version: str = SCOUT_SCHEMA_VERSION
    source_id: str
    prompt_version: str = SCOUT_PROMPT_VERSION
    model_id: str


class VisualBlock(StrictModel):
    page_number: int = Field(ge=1)
    reading_order_index: int = Field(ge=1)
    block_type: VisualBlockType
    exact_visible_text: str = ""
    table_html: str | None = None
    uncertainty: str | None = None


class SourceContentDraft(StrictModel):
    """Minimal Gemini output for one transcription job."""

    represented_original_pdf_pages: list[int] = Field(min_length=1)
    detected_reading_order: str
    visual_blocks: list[VisualBlock] = Field(default_factory=list)
    continuation_from_previous_page: bool = False
    continuation_to_next_page: bool = False
    unreadable_regions: list[str] = Field(default_factory=list)
    visual_uncertainties: list[str] = Field(default_factory=list)
    page_complete: bool = True


class SourceContent(SourceContentDraft):
    schema_version: str = CANONICAL_TRANSCRIPTION_SCHEMA_VERSION
    source_id: str
    prompt_version: str = TRANSCRIPTION_PROMPT_VERSION
    model_id: str
    job_id: str
    chunk_id: str


class SlicePageMapEntry(StrictModel):
    slice_page_index: int = Field(ge=1)
    original_pdf_page_number: int = Field(ge=1)
    role: Literal["previous_context", "primary", "next_context"]


class TranscriptionJob(StrictModel):
    job_id: str
    chunk_id: str
    profile: ProfileName
    primary_pages: list[int] = Field(min_length=1)
    context_pages: list[int] = Field(default_factory=list)
    slice_page_map: list[SlicePageMapEntry] = Field(default_factory=list)
    reason: str
    status: Literal[
        "pending",
        "completed",
        "incomplete",
        "failed",
        "deferred_transient",
        "failed_transient_exhausted",
        "failed_nonretryable",
        "skipped_compatible_completed",
        "imported_compatible_raw_response",
    ] = "pending"


class CompletenessFinding(StrictModel):
    finding_id: str
    severity: Literal["info", "warning", "error"]
    issue_code: str
    issue_message: str
    page_number: int | None = None
    job_id: str | None = None
    repair_required: bool = False


class ProviderCallEvidence(StrictModel):
    provider_backend: str
    stage: Literal["scout", "transcription"]
    job_id: str | None = None
    attempt: int = 1
    success: bool
    request_id: str | None = None
    response_id: str | None = None
    token_usage: dict[str, int] | None = None
    finish_reason: str | None = None
    duration_seconds: float
    http_status: int | None = None
    api_status: str | None = None
    calculated_delay_seconds: float | None = None
    retry_after_seconds: float | None = None
    final_failure_category: str | None = None
    pdf_slice_hash: str | None = None
    prompt_hash: str | None = None
    safe_error_class: str | None = None
    safe_error_message: str | None = None
    uploaded_file_name: str | None = None
    uploaded_file_uri_present: bool = False
    remote_file_deleted: bool | None = None
