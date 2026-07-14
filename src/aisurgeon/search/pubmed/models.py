"""Typed records for search planning, deterministic queries, and PubMed articles."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SEARCH_SCHEMA_VERSION = "search_units_v1"
QUERY_SCHEMA_VERSION = "pubmed_query_v1"
FETCH_SCHEMA_VERSION = "pubmed_fetch_v1"
SearchRelevance = Literal["search_relevant", "not_search_relevant"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchUnitDraft(StrictModel):
    section_path: list[str] = Field(default_factory=list)
    topic_de: str
    topic_en: str
    linked_formal_item_ids: list[str] = Field(min_length=1)
    relevant_comment_ids: list[str] = Field(default_factory=list)
    search_relevance: SearchRelevance
    exclusion_reason: str | None = None
    clinical_question: str | None = None
    population: list[str] = Field(default_factory=list)
    intervention_or_exposure: list[str] = Field(default_factory=list)
    comparator: list[str] = Field(default_factory=list)
    outcomes: list[str] = Field(default_factory=list)
    disease_terms: list[str] = Field(default_factory=list)
    intervention_terms: list[str] = Field(default_factory=list)
    diagnostic_terms: list[str] = Field(default_factory=list)
    procedure_terms: list[str] = Field(default_factory=list)
    alternative_terms: list[str] = Field(default_factory=list)
    mesh_candidates: list[str] = Field(default_factory=list)
    free_text_candidates: list[str] = Field(default_factory=list)
    query_core: str | None = None
    query_rationale: str | None = None
    uncertainties: list[str] = Field(default_factory=list)
    review_required: bool = False

    @model_validator(mode="after")
    def relevance_consistent(self) -> "SearchUnitDraft":
        if self.search_relevance == "search_relevant" and not self.query_core:
            raise ValueError("search_relevant requires query_core")
        if self.search_relevance == "not_search_relevant" and not self.exclusion_reason:
            raise ValueError("not_search_relevant requires exclusion_reason")
        return self


class SearchPlanDraft(StrictModel):
    search_units: list[SearchUnitDraft]


class SearchUnit(SearchUnitDraft):
    schema_version: Literal["search_units_v1"] = SEARCH_SCHEMA_VERSION
    source_id: str
    search_unit_id: str
    sequence_number: int = Field(ge=1)
    linked_original_item_numbers: list[str]
    linked_formal_item_families: list[str]
    exact_formal_item_texts: list[str]


class FormalItemSearchCoverage(StrictModel):
    schema_version: Literal["search_units_v1"] = SEARCH_SCHEMA_VERSION
    source_id: str
    formal_item_id: str
    search_relevance: SearchRelevance
    linked_search_unit_ids: list[str]
    exclusion_reason: str | None = None
    review_required: bool = False


class PubMedQuery(StrictModel):
    schema_version: Literal["pubmed_query_v1"] = QUERY_SCHEMA_VERSION
    source_id: str
    query_id: str
    search_unit_id: str
    linked_formal_item_ids: list[str]
    query_core: str
    date_filter: str
    humans_filter: str
    evidence_type_filter: str
    exclusion_filter: str
    final_pubmed_query: str
    start_date: str
    end_date: str
    query_version: str
    prompt_version: str
    prompt_hash: str
    model_id: str
    model_configuration: dict[str, Any]
    model_configuration_hash: str
    review_required: bool = False
    review_notes: list[str] = Field(default_factory=list)


class PubMedArticle(StrictModel):
    schema_version: Literal["pubmed_fetch_v1"] = FETCH_SCHEMA_VERSION
    pmid: str
    doi: str | None = None
    title: str
    abstract: str | None = None
    authors: list[str] = Field(default_factory=list)
    journal: str | None = None
    publication_date: str | None = None
    publication_year: int | None = None
    publication_types: list[str] = Field(default_factory=list)
    mesh_terms: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    language: list[str] = Field(default_factory=list)
    electronic_publication_date: str | None = None
    print_publication_date: str | None = None
    query_ids: list[str] = Field(default_factory=list)
    search_unit_ids: list[str] = Field(default_factory=list)
    linked_formal_item_ids: list[str] = Field(default_factory=list)
    fetched_at: str
    raw_source: Literal["NCBI EFetch XML"] = "NCBI EFetch XML"
    has_abstract: bool
