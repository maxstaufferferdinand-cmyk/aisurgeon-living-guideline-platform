"""Network-free tests for canonical source extraction primitives."""

import json
from pathlib import Path

import pytest
from openpyxl import load_workbook
from pydantic import ValidationError

from aisurgeon.extraction.canonical.core import (
    assign_formal_item_ids,
    expand_reference_numbers,
    link_comments,
    link_references,
    make_finding,
    merge_formal_items,
    overall_status,
    plan_windows,
)
from aisurgeon.extraction.canonical.models import (
    ClinicalContextBlock,
    Comment,
    ExtractionBatch,
    FormalItem,
    Reference,
)
from aisurgeon.extraction.canonical.outputs import (
    expert_consensus_view,
    recommendation_view,
    statement_view,
    write_jsonl,
    write_review_workbook,
)
from aisurgeon.extraction.canonical.pipeline import pending_windows
from aisurgeon.extraction.gemini.models import PageRange


def item(**overrides) -> FormalItem:
    values = {
        "source_id": "SOURCE",
        "extraction_batch_id": "batch-1",
        "item_type": "recommendation",
        "item_type_raw": "Empfehlung",
        "original_number": "2.9",
        "exact_original_text": "Original bleibt original.",
        "page_start": 10,
        "page_end": 10,
        "extraction_confidence": 0.9,
    }
    values.update(overrides)
    return FormalItem.model_validate(values)


def comment(**overrides) -> Comment:
    values = {
        "source_id": "SOURCE",
        "extraction_batch_id": "batch-1",
        "exact_original_text": "Kommentar im Original.",
        "page_start": 10,
        "page_end": 11,
        "extraction_confidence": 0.8,
    }
    values.update(overrides)
    return Comment.model_validate(values)


def reference(number: str) -> Reference:
    return Reference(
        source_id="SOURCE",
        original_reference_number=number,
        exact_original_reference_text=f"[{number}] Original reference.",
        page_start=20,
        page_end=20,
        extraction_confidence=1,
    )


def test_exact_formal_item_validation_and_type_separation() -> None:
    recommendation = item()
    statement = item(
        item_type="statement", original_number="2.16", exact_original_text="Wortgetreues Statement."
    )
    assert recommendation.exact_original_text == "Original bleibt original."
    assert [
        value for value in (recommendation, statement) if value.item_type == "recommendation"
    ] == [recommendation]
    with pytest.raises(ValidationError):
        item(exact_original_text="")
    with pytest.raises(ValidationError):
        item(page_start=11, page_end=10)


def test_two_column_window_has_primary_and_context_boundaries() -> None:
    windows = plan_windows(
        [PageRange(page_start=5, page_end=22)],
        stage="clinical",
        pages_per_job=8,
        overlap_pages=1,
        document_page_count=30,
    )
    assert [(w.primary_page_start, w.primary_page_end) for w in windows] == [
        (5, 12),
        (13, 20),
        (21, 22),
    ]
    assert (windows[1].context_page_start, windows[1].context_page_end) == (12, 21)


def test_deterministic_ids_for_numbered_and_unnumbered_items() -> None:
    first = item()
    second = item(original_number=None, exact_original_text="Unnummerierter Originaltext.")
    assign_formal_item_ids([first, second])
    original_ids = (first.item_id, second.item_id)
    assign_formal_item_ids([first, second])
    assert (first.item_id, second.item_id) == original_ids
    assert first.item_id == "SOURCE_REC_2_9"


def test_overlap_duplicate_is_merged_but_conflict_is_retained() -> None:
    duplicate = item(extraction_batch_id="batch-2", page_start=11, page_end=11)
    conflict = item(exact_original_text="Abweichender Originaltext.", page_start=11, page_end=11)
    merged, findings = merge_formal_items([item(), duplicate, conflict])
    assert len(merged) == 2
    assert any(f.severity == "info" for f in findings)
    assert any(f.human_review_required for f in findings)
    assert conflict in merged


def test_comment_linking_explicit_and_uncertain() -> None:
    formal = item()
    assign_formal_item_ids([formal])
    explicit = comment(related_original_number="2.9")
    findings = link_comments([explicit], [formal])
    assert explicit.linked_item_ids == [formal.item_id]
    assert findings == []
    uncertain = comment(page_start=1, page_end=1)
    findings = link_comments([uncertain], [formal])
    assert uncertain.linked_item_ids == []
    assert findings[0].issue_code == "comment_link_uncertain"


def test_reference_range_expansion_and_unresolved_warning() -> None:
    formal = item(inline_reference_numbers=["[13-15]", "17"])
    assign_formal_item_ids([formal])
    unresolved, findings = link_references(
        [formal], [], [reference("13"), reference("14"), reference("15")]
    )
    assert expand_reference_numbers(["[13-15]"]) == ["13", "14", "15"]
    assert formal.inline_reference_numbers == ["13", "14", "15", "17"]
    assert [link.reference_number for link in unresolved] == ["17"]
    assert findings[0].issue_code == "reference_unresolved"


def test_context_block_remains_verbatim() -> None:
    block = ClinicalContextBlock(
        source_id="SOURCE",
        exact_original_text="Nicht zugeordnet.",
        page_start=3,
        page_end=3,
        extraction_confidence=0.5,
        reason_unassigned="Grenze unklar",
        review_required=True,
    )
    assert block.exact_original_text == "Nicht zugeordnet."


def test_status_only_fails_for_explicit_hard_failure() -> None:
    assert overall_status([]) == "completed"
    warning = make_finding(
        source_id="SOURCE", issue_code="metadata_missing", issue_message="Metadatum fehlt."
    )
    assert overall_status([warning]) == "completed_with_review"
    assert overall_status([], hard_failure=True) == "failed"


def test_jsonl_and_review_excel_outputs(tmp_path: Path) -> None:
    finding = make_finding(
        source_id="SOURCE", issue_code="page_range_warning", issue_message="Seitenanker prüfen."
    )
    jsonl = tmp_path / "review_findings.jsonl"
    workbook_path = tmp_path / "review_findings.xlsx"
    write_jsonl(jsonl, [finding])
    assert json.loads(jsonl.read_text(encoding="utf-8"))["issue_code"] == "page_range_warning"
    write_review_workbook(workbook_path, [finding])
    sheet = load_workbook(workbook_path).active
    assert sheet.freeze_panes == "A2"
    assert sheet.auto_filter.ref is not None
    with pytest.raises(FileExistsError):
        write_review_workbook(workbook_path, [finding])


def test_resume_skips_completed_checkpoint(tmp_path: Path) -> None:
    windows = plan_windows(
        [PageRange(page_start=1, page_end=9)],
        stage="clinical",
        pages_per_job=8,
        overlap_pages=1,
        document_page_count=9,
    )
    tmp_path.joinpath(f"{windows[0].job_id}.json").write_text(
        '{"status":"completed"}', encoding="utf-8"
    )
    assert pending_windows(windows, tmp_path) == [windows[1]]


def test_all_formal_families_share_one_chronological_backbone() -> None:
    labels = [
        ("Evidenzbasierte Empfehlung", "2.1", "recommendation"),
        ("Evidenzbasiertes Statement", "2.2", "statement"),
        ("Konsensbasierte Empfehlung", "2.3", "recommendation"),
        ("Konsensbasiertes Statement", "2.4", "consensus_statement"),
        ("Konsensusstatement", "2.5", "consensus_statement"),
        ("Expertenkonsens", "2.6", "expert_consensus"),
    ]
    items = [
        item(
            item_type=("statement" if "Statement" in raw or "statement" in raw else
                       "recommendation" if "Empfehlung" in raw else "other_formal_item"),
            item_type_raw=raw,
            original_number=number,
            exact_original_text=f"Originaltext {number}",
            page_start=20,
            page_end=20,
        )
        for raw, number, _family in labels
    ]
    merged, _ = merge_formal_items(items)
    assert [value.original_number for value in merged] == [value[1] for value in labels]
    assert [value.normalized_item_family for value in merged] == [
        value[2] for value in labels
    ]
    assert [value.sequence_number for value in merged] == list(range(1, 7))
    assert [value.original_number for value in recommendation_view(merged)] == ["2.1", "2.3"]
    assert [value.original_number for value in statement_view(merged)] == [
        "2.2", "2.4", "2.5"
    ]
    assert [value.original_number for value in expert_consensus_view(merged)] == ["2.6"]

    comments = [comment(related_original_number=number, exact_original_text=f"Kommentar {number}")
                for _raw, number, _family in labels]
    assert link_comments(comments, merged) == []
    assert all(value.linked_formal_item_ids for value in comments)


@pytest.mark.parametrize(
    ("raw", "family"),
    [
        ("Evidenzbasierte Empfehlung", "recommendation"),
        ("Konsensbasierte Empfehlung", "recommendation"),
        ("Evidenzbasiertes Statement", "statement"),
        ("Konsensbasiertes Statement", "consensus_statement"),
        ("Konsensusstatement", "consensus_statement"),
        ("Expertenkonsens", "expert_consensus"),
        ("EK", "expert_consensus"),
        ("EK-Empfehlung", "expert_consensus"),
    ],
)
def test_german_native_item_labels_are_normalized(raw: str, family: str) -> None:
    formal = item(item_type="other_formal_item", item_type_raw=raw)
    merged, _ = merge_formal_items([formal])
    assert merged[0].normalized_item_family == family


def test_unknown_formal_type_is_retained_for_review() -> None:
    unknown = item(item_type="other_formal_item", item_type_raw="Quellenformat X")
    merged, _ = merge_formal_items([unknown])
    assert merged == [unknown]
    assert unknown.normalized_item_family == "other_formal_item"
    assert unknown.review_required is True
    assert "unknown_formal_item_type" in unknown.review_reasons


def test_v1_formal_item_remains_readable() -> None:
    legacy = item(schema_version="canonical_extraction_v1")
    assert legacy.schema_version == "canonical_extraction_v1"


def test_committed_v2_extraction_schema_matches_model() -> None:
    schema = json.loads(
        Path("schemas/extraction/canonical_extraction_v2.schema.json").read_text(
            encoding="utf-8"
        )
    )
    assert schema == ExtractionBatch.model_json_schema()


def test_prompt_v2_is_active_and_v1_remains_historical() -> None:
    v1 = Path("config/prompts/gemini_formal_items_comments_v1.txt").read_text(
        encoding="utf-8"
    )
    v2 = Path("config/prompts/gemini_formal_items_comments_v2.txt").read_text(
        encoding="utf-8"
    )
    assert "prompt_version = gemini_formal_items_comments_v2" in v2
    assert "All source-native formal item types are equally important" in v2
    assert "prompt_version" not in v1
