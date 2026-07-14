"""Versioned canonical models that preserve native PDF information."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "canonical_extraction_v2"
SUPPORTED_SCHEMA_VERSIONS = Literal["canonical_extraction_v1", "canonical_extraction_v2"]
NormalizedItemFamily = Literal[
    "recommendation",
    "statement",
    "consensus_statement",
    "expert_consensus",
    "other_formal_item",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceObject(StrictModel):
    schema_version: SUPPORTED_SCHEMA_VERSIONS = SCHEMA_VERSION
    source_id: str = Field(min_length=1)
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    extraction_confidence: float = Field(ge=0, le=1)
    review_required: bool = False
    review_reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_page_order(self) -> "SourceObject":
        if self.page_end < self.page_start:
            raise ValueError("page_end muss größer oder gleich page_start sein.")
        return self


class FormalItem(SourceObject):
    extraction_batch_id: str = Field(min_length=1)
    item_id: str | None = None
    formal_item_id: str | None = None
    sequence_number: int | None = Field(default=None, ge=1)
    item_type: Literal["recommendation", "statement", "other_formal_item"]
    item_type_raw: str | None = None
    normalized_item_family: NormalizedItemFamily | None = None
    original_number: str | None = None
    topic_or_short_title_raw: str | None = None
    exact_original_text: str = Field(min_length=1)
    chapter_path_raw: list[str] = Field(default_factory=list)
    recommendation_grade_raw: str | None = None
    evidence_level_raw: str | None = None
    consensus_raw: str | None = None
    status_raw: str | None = None
    year_raw: str | None = None
    inline_reference_numbers: list[str] = Field(default_factory=list)
    linked_comment_ids: list[str] = Field(default_factory=list)
    unresolved_reference_numbers: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def synchronize_legacy_id(self) -> "FormalItem":
        self.formal_item_id = self.formal_item_id or self.item_id
        self.item_id = self.item_id or self.formal_item_id
        return self


class Comment(SourceObject):
    extraction_batch_id: str = Field(min_length=1)
    comment_id: str | None = None
    comment_type_raw: str | None = None
    exact_original_text: str = Field(min_length=1)
    related_original_number: str | None = None
    related_formal_item_type_raw: str | None = None
    linked_item_ids: list[str] = Field(default_factory=list)
    linked_formal_item_ids: list[str] = Field(default_factory=list)
    inline_reference_numbers: list[str] = Field(default_factory=list)
    unresolved_reference_numbers: list[str] = Field(default_factory=list)
    chapter_path_raw: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def synchronize_legacy_links(self) -> "Comment":
        links = list(dict.fromkeys((*self.linked_formal_item_ids, *self.linked_item_ids)))
        self.linked_formal_item_ids = links
        self.linked_item_ids = links
        return self


class Reference(SourceObject):
    reference_id: str | None = None
    original_reference_number: str = Field(min_length=1)
    exact_original_reference_text: str = Field(min_length=1)


class VisualObject(SourceObject):
    object_id: str | None = None
    object_type: Literal["table", "algorithm", "decision_tree"]
    title_or_caption_raw: str | None = None
    source_anchor_description: str = Field(min_length=1)
    linked_original_item_numbers: list[str] = Field(default_factory=list)


class ClinicalContextBlock(SourceObject):
    context_block_id: str | None = None
    exact_original_text: str = Field(min_length=1)
    chapter_path_raw: list[str] = Field(default_factory=list)
    possible_related_original_numbers: list[str] = Field(default_factory=list)
    reason_unassigned: str = Field(min_length=1)


class ReviewFinding(StrictModel):
    finding_id: str
    stage: str
    severity: Literal["info", "warning", "error"]
    issue_code: str
    issue_message: str
    source_id: str
    object_type: str | None = None
    object_id: str | None = None
    original_number: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    workflow_continued: bool
    human_review_required: bool
    suggested_action: str | None = None
    review_status: Literal["open", "accepted", "repaired", "rejected"] = "open"
    review_note: str | None = None


class UnresolvedLink(StrictModel):
    source_id: str
    object_type: Literal["formal_item", "comment"]
    object_id: str
    reference_number: str


class ExtractionBatch(StrictModel):
    formal_items: list[FormalItem] = Field(default_factory=list)
    comments: list[Comment] = Field(default_factory=list)
    clinical_context_blocks: list[ClinicalContextBlock] = Field(default_factory=list)


class ReferenceBatch(StrictModel):
    references: list[Reference] = Field(default_factory=list)


class VisualObjectBatch(StrictModel):
    visual_objects: list[VisualObject] = Field(default_factory=list)
