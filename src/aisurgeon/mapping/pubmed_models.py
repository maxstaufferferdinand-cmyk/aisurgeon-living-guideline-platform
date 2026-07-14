"""Strict records for abstract-based PubMed-to-FormalItem mapping."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAPPING_SCHEMA_VERSION = "pubmed_formal_item_mapping_v1"
CANDIDATE_GENERATION_VERSION = "pubmed_candidate_provenance_v1"
MappingDecision = Literal[
    "include_direct",
    "include_indirect",
    "context_only",
    "exclude_not_relevant",
    "exclude_wrong_population",
    "exclude_wrong_intervention_or_exposure",
    "exclude_wrong_comparator",
    "exclude_wrong_outcome",
    "exclude_wrong_setting",
    "exclude_wrong_study_design",
    "exclude_guideline_or_consensus_document",
    "exclude_duplicate_publication",
    "exclude_no_usable_abstract",
    "uncertain_review_required",
]
StudyDesign = Literal[
    "randomized_controlled_trial",
    "controlled_clinical_trial",
    "meta_analysis_randomized",
    "meta_analysis_nonrandomized_or_mixed",
    "systematic_review",
    "prospective_cohort",
    "retrospective_cohort",
    "case_control",
    "cross_sectional",
    "diagnostic_accuracy",
    "prognostic_study",
    "registry_study",
    "comparative_study",
    "validation_study",
    "evaluation_study",
    "case_series",
    "narrative_review",
    "guideline_or_consensus",
    "other",
    "unclear",
]
Match = Literal["match", "partial", "mismatch", "not_applicable", "unclear"]
Directness = Literal["direct", "indirect", "contextual", "not_relevant", "unclear"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ScreeningDecisionDraft(StrictModel):
    pmid: str
    mapping_decision: MappingDecision
    relevance_score: int = Field(ge=0, le=100)
    directness: Directness
    population_match: Match
    intervention_or_exposure_match: Match
    comparator_match: Match
    outcome_match: Match
    setting_match: Match
    study_design_normalized: StudyDesign
    publication_type_interpretation: str
    concise_mapping_reason: str
    supporting_abstract_passage: str | None = None
    uncertainty_reason: str | None = None
    review_required: bool

    @model_validator(mode="after")
    def uncertainty_is_consistent(self) -> "ScreeningDecisionDraft":
        if self.mapping_decision == "uncertain_review_required" and not self.review_required:
            raise ValueError("uncertain_review_required requires review_required")
        return self


class ScreeningBatchDraft(StrictModel):
    decisions: list[ScreeningDecisionDraft]
