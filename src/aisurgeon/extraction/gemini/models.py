"""Typed models for Gemini document-map requests and audit metadata."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DOCUMENT_MAP_SCHEMA_VERSION = "document_map_v1"


class StrictModel(BaseModel):
    """Forbid unversioned or unexpected fields."""

    model_config = ConfigDict(extra="forbid")


class PageRange(StrictModel):
    """Inclusive, one-based page range."""

    page_start: int
    page_end: int
    description: str | None = None


class InventoryObject(StrictModel):
    """Provisional visual-object inventory entry, never a stable domain identity."""

    provisional_object_id: str = Field(min_length=1)
    object_type: Literal["table", "algorithm", "decision_tree"]
    title_or_caption: str | None = None
    page_start: int
    page_end: int
    source_anchor_description: str = Field(min_length=1)
    extraction_confidence: float = Field(ge=0, le=1)
    review_required: bool

class DocumentMap(StrictModel):
    """PDF-adaptive structural map; this is not recommendation extraction."""

    schema_version: Literal["document_map_v1"]
    source_id: str = Field(min_length=1)
    document_title: str | None = None
    issuing_organization: str | None = None
    guideline_identifier: str | None = None
    guideline_class: str | None = None
    language: str | None = None
    publication_year: int | None = Field(default=None, ge=1900, le=2200)
    version_text: str | None = None
    validity_status: str | None = None
    declared_page_count: int = Field(ge=1)
    detected_document_layout: str | None = None
    column_layout: str | None = None
    recurring_header_footer_description: str | None = None
    front_matter_page_ranges: list[PageRange] = Field(default_factory=list)
    table_of_contents_page_ranges: list[PageRange] = Field(default_factory=list)
    clinical_main_body_page_ranges: list[PageRange] = Field(default_factory=list)
    bibliography_page_ranges: list[PageRange] = Field(default_factory=list)
    appendix_page_ranges: list[PageRange] = Field(default_factory=list)
    recommendation_or_statement_patterns: list[str] = Field(default_factory=list)
    comment_or_rationale_patterns: list[str] = Field(default_factory=list)
    native_grading_systems: list[str] = Field(default_factory=list)
    detected_formal_item_types: list[str] = Field(default_factory=list)
    detected_table_inventory: list[InventoryObject] = Field(default_factory=list)
    detected_algorithm_inventory: list[InventoryObject] = Field(default_factory=list)
    detected_decision_tree_inventory: list[InventoryObject] = Field(default_factory=list)
    uncertain_regions: list[PageRange] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

class GeminiModelConfig(StrictModel):
    """Versioned, shared request configuration."""

    provider: Literal["google"]
    api: Literal["interactions"]
    model_id: Literal["gemini-3.5-flash"]
    thinking_level: Literal["medium"]
    media_resolution: Literal["high"]
    prompt_version: Literal["gemini_document_map_v1"]
    schema_version: Literal["document_map_v1"]
    request_timeout_seconds: int = Field(gt=0, le=600)
    max_attempts: int = Field(ge=1, le=5)


class RemoteFileMetadata(StrictModel):
    """Non-secret metadata for the temporary Gemini file."""

    remote_file_name: str | None = None
    uri: str | None = None
    mime_type: str | None = None
    upload_timestamp_utc: datetime | None = None
    status: str
    remote_file_deleted: bool


class ValidationIssue(StrictModel):
    """One deterministic validation finding."""

    severity: Literal["error", "warning"]
    code: str
    message: str


class DocumentMapValidationReport(StrictModel):
    """Deterministic comparison of remote map and local registration."""

    valid: bool
    review_required: bool
    issues: list[ValidationIssue] = Field(default_factory=list)


class GeminiDocumentMapResult(StrictModel):
    """Validated result returned by the client abstraction."""

    document_map: DocumentMap
    raw_json: str
    remote_file_metadata: RemoteFileMetadata
    token_usage: dict[str, int] | None = None


class RunManifest(StrictModel):
    """Secret-free audit manifest for one document-map attempt."""

    run_id: str
    stage: Literal["gemini_document_map"]
    status: str
    worker_id: str
    source_id: str
    pdf_filename: str
    pdf_sha256: str
    file_size_bytes: int
    local_page_count: int | None
    git_commit: str
    git_branch: str
    dirty_worktree: bool
    python_version: str
    package_version: str
    model_provider: str
    model_id: str
    api_surface: str
    thinking_level: str
    media_resolution: str
    prompt_version: str
    prompt_sha256: str
    schema_version: str
    schema_sha256: str
    start_time_utc: datetime
    end_time_utc: datetime | None = None
    token_usage: dict[str, int] | None = None
    remote_file_deleted: bool = False
    output_files: dict[str, str] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
