"""DocumentMap schema and deterministic validation tests."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from aisurgeon.extraction.gemini.document_map import (
    load_model_config,
    load_prompt,
    load_schema,
    validate_document_map,
    validate_versioned_schema,
)
from aisurgeon.extraction.gemini.models import DocumentMap, PageRange
from aisurgeon.extraction.pdf_registration import register_pdf


def valid_map(*, source_id: str = "source-test", page_count: int = 3) -> dict:
    return {
        "schema_version": "document_map_v1",
        "source_id": source_id,
        "document_title": "Synthetic guideline",
        "issuing_organization": None,
        "guideline_identifier": None,
        "guideline_class": None,
        "language": "de",
        "publication_year": 2026,
        "version_text": None,
        "validity_status": None,
        "declared_page_count": page_count,
        "detected_document_layout": "single column",
        "column_layout": "single",
        "recurring_header_footer_description": None,
        "front_matter_page_ranges": [{"page_start": 1, "page_end": 1}],
        "table_of_contents_page_ranges": [],
        "clinical_main_body_page_ranges": [{"page_start": 2, "page_end": 2}],
        "bibliography_page_ranges": [{"page_start": 3, "page_end": 3}],
        "appendix_page_ranges": [],
        "recommendation_or_statement_patterns": ["numbered recommendation box"],
        "comment_or_rationale_patterns": [],
        "native_grading_systems": [],
        "detected_formal_item_types": ["Recommendation"],
        "detected_table_inventory": [],
        "detected_algorithm_inventory": [],
        "detected_decision_tree_inventory": [],
        "uncertain_regions": [],
        "warnings": [],
    }


def test_document_map_validates() -> None:
    document_map = DocumentMap.model_validate(valid_map())
    assert document_map.schema_version == "document_map_v1"
    assert document_map.clinical_main_body_page_ranges[0].page_start == 2


def test_invalid_page_range_order_is_parseable_for_review() -> None:
    page_range = PageRange(page_start=3, page_end=2)
    assert page_range.page_start == 3


def test_overlapping_document_regions_are_parseable_for_review() -> None:
    value = valid_map()
    value["table_of_contents_page_ranges"] = [{"page_start": 1, "page_end": 2}]
    assert DocumentMap.model_validate(value).table_of_contents_page_ranges


def test_blank_layout_metadata_is_parseable_for_review() -> None:
    value = valid_map()
    value["detected_document_layout"] = ""
    assert DocumentMap.model_validate(value).detected_document_layout == ""


def test_final_recommendation_id_is_forbidden() -> None:
    value = valid_map()
    value["recommendation_id"] = "LLM-MADE-ID"
    with pytest.raises(ValidationError):
        DocumentMap.model_validate(value)


def test_model_configuration_is_exactly_versioned() -> None:
    root = Path.cwd()
    config, _ = load_model_config(root)
    assert config.model_id == "gemini-3.5-flash"
    assert config.thinking_level == "medium"
    assert config.media_resolution == "high"


def test_prompt_loads_and_hashes() -> None:
    prompt, digest = load_prompt(Path.cwd())
    assert "Do not extract" in prompt
    assert len(digest) == 64


def test_committed_json_schema_matches_pydantic() -> None:
    schema, raw = load_schema(Path.cwd())
    assert json.loads(raw) == DocumentMap.model_json_schema()
    validate_versioned_schema(schema)


def test_page_count_mismatch_is_a_non_blocking_warning(synthetic_pdf: Path) -> None:
    registration = register_pdf(synthetic_pdf, worker_id="worker-test")
    document_map = DocumentMap.model_validate(
        valid_map(source_id=registration.source_id, page_count=3)
    )
    report = validate_document_map(document_map, registration)
    assert report.valid is True
    assert report.review_required is True
    issue = next(issue for issue in report.issues if issue.code == "page_count_mismatch")
    assert issue.severity == "warning"


def test_out_of_bounds_inventory_is_a_non_blocking_warning(synthetic_pdf: Path) -> None:
    registration = register_pdf(synthetic_pdf, worker_id="worker-test")
    value = valid_map(source_id=registration.source_id, page_count=2)
    value["bibliography_page_ranges"] = []
    value["detected_table_inventory"] = [
        {
            "provisional_object_id": "TABLE-PROVISIONAL-1",
            "object_type": "table",
            "title_or_caption": None,
            "page_start": 2,
            "page_end": 3,
            "source_anchor_description": "bottom of page",
            "extraction_confidence": 0.5,
            "review_required": True,
        }
    ]
    report = validate_document_map(DocumentMap.model_validate(value), registration)
    assert report.valid is True
    issue = next(issue for issue in report.issues if issue.code == "page_range_out_of_bounds")
    assert issue.severity == "warning"


def test_empty_formal_item_types_requires_review(synthetic_pdf: Path) -> None:
    registration = register_pdf(synthetic_pdf, worker_id="worker-test")
    value = valid_map(source_id=registration.source_id, page_count=2)
    value["bibliography_page_ranges"] = []
    value["detected_formal_item_types"] = []
    report = validate_document_map(DocumentMap.model_validate(value), registration)
    assert report.valid is True
    assert report.review_required is True


def test_region_overlap_is_a_non_blocking_warning(synthetic_pdf: Path) -> None:
    registration = register_pdf(synthetic_pdf, worker_id="worker-test")
    value = valid_map(source_id=registration.source_id, page_count=2)
    value["front_matter_page_ranges"] = [{"page_start": 1, "page_end": 1}]
    value["table_of_contents_page_ranges"] = [{"page_start": 1, "page_end": 1}]
    value["clinical_main_body_page_ranges"] = [{"page_start": 2, "page_end": 2}]
    value["bibliography_page_ranges"] = []
    report = validate_document_map(DocumentMap.model_validate(value), registration)
    assert report.valid is True
    assert report.review_required is True
    assert any(
        issue.code == "document_region_overlap" and issue.severity == "warning"
        for issue in report.issues
    )


def test_invalid_secondary_range_is_a_non_blocking_warning(synthetic_pdf: Path) -> None:
    registration = register_pdf(synthetic_pdf, worker_id="worker-test")
    value = valid_map(source_id=registration.source_id, page_count=2)
    value["front_matter_page_ranges"] = [{"page_start": 2, "page_end": 1}]
    value["clinical_main_body_page_ranges"] = [{"page_start": 1, "page_end": 2}]
    value["bibliography_page_ranges"] = []
    report = validate_document_map(DocumentMap.model_validate(value), registration)
    assert report.valid is True
    assert any(issue.code == "invalid_page_range" for issue in report.issues)


def test_source_id_mismatch_remains_a_hard_validation_error(synthetic_pdf: Path) -> None:
    registration = register_pdf(synthetic_pdf, worker_id="worker-test")
    value = valid_map(source_id="different-source", page_count=2)
    value["bibliography_page_ranges"] = []
    report = validate_document_map(DocumentMap.model_validate(value), registration)
    assert report.valid is False
    assert any(
        issue.code == "source_id_mismatch" and issue.severity == "error"
        for issue in report.issues
    )


def test_missing_clinical_main_structure_remains_a_hard_error(synthetic_pdf: Path) -> None:
    registration = register_pdf(synthetic_pdf, worker_id="worker-test")
    value = valid_map(source_id=registration.source_id, page_count=2)
    value["clinical_main_body_page_ranges"] = []
    value["bibliography_page_ranges"] = []
    report = validate_document_map(DocumentMap.model_validate(value), registration)
    assert report.valid is False
    assert any(issue.code == "clinical_main_structure_unusable" for issue in report.issues)


def test_minor_layout_metadata_gap_is_a_non_blocking_warning(synthetic_pdf: Path) -> None:
    registration = register_pdf(synthetic_pdf, worker_id="worker-test")
    value = valid_map(source_id=registration.source_id, page_count=2)
    value["detected_document_layout"] = None
    value["column_layout"] = ""
    value["bibliography_page_ranges"] = []
    report = validate_document_map(DocumentMap.model_validate(value), registration)
    assert report.valid is True
    assert report.review_required is True
    assert any(issue.code == "document_layout_metadata_incomplete" for issue in report.issues)
