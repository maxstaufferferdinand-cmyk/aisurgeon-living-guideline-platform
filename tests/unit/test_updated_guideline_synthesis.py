import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import SecretStr

from aisurgeon.synthesis.updated_guideline import (
    build_item_evidence_packets,
    build_updated_guideline,
    consolidate_references,
    validate_synthesis,
)


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _formal(
    n: int,
    family: str = "recommendation",
    raw: str = "Evidenzbasierte Empfehlung",
    refs: list[str] | None = None,
) -> dict:
    return {
        "schema_version": "canonical_extraction_v2",
        "source_id": "SRC",
        "page_start": n,
        "page_end": n,
        "extraction_confidence": 1.0,
        "review_required": False,
        "review_reasons": [],
        "extraction_batch_id": "b",
        "item_id": f"F{n}",
        "formal_item_id": f"F{n}",
        "sequence_number": n,
        "item_type": "statement" if family == "statement" else "recommendation",
        "item_type_raw": raw,
        "normalized_item_family": family,
        "original_number": str(n),
        "topic_or_short_title_raw": f"Topic {n}",
        "exact_original_text": f"Originaltext {n} mit Abb. 1.",
        "chapter_path_raw": ["Kapitel"],
        "recommendation_grade_raw": "B" if family == "recommendation" else None,
        "evidence_level_raw": "2",
        "consensus_raw": "Starker Konsens",
        "status_raw": "NEU",
        "year_raw": "2022",
        "inline_reference_numbers": refs or [],
        "linked_comment_ids": [f"C{n}"],
        "unresolved_reference_numbers": [],
    }


def _comment(n: int) -> dict:
    return {
        "schema_version": "canonical_extraction_v2",
        "source_id": "SRC",
        "page_start": n,
        "page_end": n,
        "extraction_confidence": 1.0,
        "review_required": False,
        "review_reasons": [],
        "extraction_batch_id": "b",
        "comment_id": f"C{n}",
        "comment_type_raw": "Kommentar",
        "exact_original_text": f"Exakter Kommentar {n}",
        "related_original_number": str(n),
        "related_formal_item_type_raw": None,
        "linked_item_ids": [f"F{n}"],
        "linked_formal_item_ids": [f"F{n}"],
        "inline_reference_numbers": [],
        "unresolved_reference_numbers": [],
        "chapter_path_raw": ["Kapitel"],
    }


def _article(pmid: str, *, doi: str | None = None, title: str | None = None) -> dict:
    return {
        "schema_version": "pubmed_fetch_v1",
        "pmid": pmid,
        "doi": doi,
        "title": title or f"Title {pmid}",
        "abstract": f"Abstract {pmid}",
        "authors": ["A Author", "B Author"],
        "journal": "Journal",
        "publication_date": "2025",
        "publication_year": 2025,
        "publication_types": ["Randomized Controlled Trial", "Journal Article"],
        "mesh_terms": ["Humans"],
        "keywords": [],
        "language": ["eng"],
        "electronic_publication_date": None,
        "print_publication_date": "2025",
        "query_ids": ["Q"],
        "search_unit_ids": ["U"],
        "linked_formal_item_ids": ["F1"],
        "fetched_at": "2026-07-16T00:00:00Z",
        "raw_source": "NCBI EFetch XML",
        "has_abstract": True,
    }


def _mapping(pmid: str, item: str, decision: str, score: int = 100) -> dict:
    return {
        "schema_version": "pubmed_formal_item_mapping_v1",
        "source_id": "SRC",
        "candidate_pair_id": f"P{pmid}{item}",
        "pmid": pmid,
        "formal_item_id": item,
        "screening_method": "gpt_abstract_mapping",
        "mapping_decision": decision,
        "relevance_score": score,
        "directness": "direct",
        "population_match": "match",
        "intervention_or_exposure_match": "match",
        "comparator_match": "not_applicable",
        "outcome_match": "match",
        "setting_match": "match",
        "study_design_normalized": "randomized_controlled_trial",
        "publication_type_interpretation": "RCT",
        "concise_mapping_reason": "reason",
        "supporting_abstract_passage": None,
        "uncertainty_reason": None,
        "review_required": False,
    }


def _runs(tmp_path: Path, *, item_count: int = 3) -> tuple[Path, Path, Path, Path]:
    extraction, search, fetch, mapping = [tmp_path / name for name in "esfm"]
    for path in (extraction, search, fetch, mapping):
        path.mkdir()
    formal = [
        _formal(1, "recommendation", "Evidenzbasierte Empfehlung", ["1"]),
        _formal(2, "statement", "STATEMENT", ["2"]),
        _formal(3, "expert_consensus", "Expertenkonsens", []),
    ][:item_count]
    comments = [_comment(i) for i in range(1, item_count + 1)]
    refs = [
        {
            "schema_version": "canonical_extraction_v2",
            "source_id": "SRC",
            "page_start": 10,
            "page_end": 10,
            "extraction_confidence": 1.0,
            "review_required": False,
            "review_reasons": [],
            "reference_id": "R1",
            "original_reference_number": "1",
            "exact_original_reference_text": "Old Ref 1.",
        },
        {
            "schema_version": "canonical_extraction_v2",
            "source_id": "SRC",
            "page_start": 10,
            "page_end": 10,
            "extraction_confidence": 1.0,
            "review_required": False,
            "review_reasons": [],
            "reference_id": "R2",
            "original_reference_number": "2",
            "exact_original_reference_text": "Old Ref 2.",
        },
    ]
    articles = [_article("10", doi="10/x"), _article("11", doi="10/x"), _article("12")]
    mappings = [
        _mapping("10", "F1", "include_direct", score=1),
        _mapping("11", "F1", "include_direct", score=100),
        _mapping("12", "F1", "context_only", score=100),
        _mapping("12", "F2", "uncertain_review_required", score=100),
    ]
    index = [
        {
            "source_id": "SRC",
            "formal_item_id": f"F{i}",
            "review_required": False,
            "direct_article_pmids": ["10", "11"] if i == 1 else [],
            "indirect_article_pmids": [],
            "context_article_pmids": ["12"] if i == 1 else [],
            "uncertain_article_pmids": ["12"] if i == 2 else [],
        }
        for i in range(1, item_count + 1)
    ]
    _jsonl(extraction / "formal_items.jsonl", formal)
    _jsonl(extraction / "comments.jsonl", comments)
    _jsonl(extraction / "references.jsonl", refs)
    _json(extraction / "document_map.validated.json", {})
    _json(extraction / "extraction_manifest.json", {"source_id": "SRC", "status": "completed"})
    _jsonl(search / "search_units.jsonl", [{"search_unit_id": "U"}])
    _jsonl(
        search / "formal_item_search_coverage.jsonl",
        [
            {
                "source_id": "SRC",
                "formal_item_id": f"F{i}",
                "search_relevance": "search_relevant",
                "linked_search_unit_ids": ["U"],
            }
            for i in range(1, item_count + 1)
        ],
    )
    _json(search / "search_manifest.json", {"source_id": "SRC", "status": "completed"})
    _jsonl(fetch / "pubmed_articles.jsonl", articles)
    _json(fetch / "pubmed_fetch_manifest.json", {"source_id": "SRC", "status": "completed"})
    _jsonl(mapping / "article_formal_item_mappings.jsonl", mappings)
    _jsonl(mapping / "formal_item_evidence_index.jsonl", index)
    _jsonl(mapping / "mapping_review_findings.jsonl", [{"finding_id": "M1", "message": "six"}])
    _json(
        mapping / "mapping_manifest.json",
        {"source_id": "SRC", "status": "completed_with_review"},
    )
    return extraction, search, fetch, mapping


def test_packets_support_all_92_items_and_preserve_source_fields() -> None:
    formal = [
        _formal(i, "statement" if i % 2 else "recommendation", "NATIVE")
        for i in range(1, 93)
    ]
    comments = [_comment(i) for i in range(1, 93)]
    packets = build_item_evidence_packets(
        formal_items=formal,
        comments=comments,
        references=[],
        articles=[],
        mappings=[],
        evidence_index=[{"formal_item_id": f"F{i}"} for i in range(1, 93)],
        source_id="SRC",
    )
    assert len(packets) == 92
    assert packets[0]["source_native_item_type"] == "NATIVE"
    assert packets[0]["exact_original_item_text"] == "Originaltext 1 mit Abb. 1."
    assert packets[0]["exact_original_comments"] == ["Exakter Kommentar 1"]


def test_relevance_score_is_ignored_and_only_direct_indirect_drive_decisions(
    tmp_path: Path,
) -> None:
    extraction, search, fetch, mapping = _runs(tmp_path)
    seen_payloads = []

    class Fake:
        def create(self, prompt, payload):
            seen_payloads.append(payload)
            assert '"relevance_score"' not in json.dumps(
                {
                    "formal_item": payload["formal_item"],
                    "direct_articles": payload["direct_articles"],
                    "indirect_articles": payload["indirect_articles"],
                    "context_articles": payload["context_articles"],
                }
            )
            return {
                "new_evidence_de": "Neue direkte Evidenz.",
                "conclusion_de": "Der Text bleibt unverändert.",
                "decision": "rationale_updated",
                "updated_item_text_de": "Originaltext 1 mit Abb. 1.",
                "aisurgeon_evidence_class": "C",
                "used_direct_pmids": ["10"],
                "used_indirect_pmids": [],
                "used_context_pmids": [],
                "uncertainty_de": None,
                "review_required": False,
                "review_notes": [],
            }

    run = build_updated_guideline(
        extraction_run=extraction,
        search_run=search,
        fetch_run=fetch,
        mapping_run=mapping,
        output_root=tmp_path,
        worker_id="w",
        api_key=SecretStr("secret-never-output"),
        client_factory=lambda key, config: Fake(),
        now=lambda: datetime(2026, 7, 16, tzinfo=UTC),
    )
    blocks = [
        json.loads(line)
        for line in (run / "updated_guideline_blocks.jsonl").read_text().splitlines()
    ]
    assert blocks[0]["decision"] == "rationale_updated"
    assert blocks[1]["decision"] == "insufficient_new_evidence"
    assert blocks[2]["decision"] == "insufficient_new_evidence"
    assert seen_payloads[0]["direct_articles"][0]["pmid"] == "10"
    assert "11" not in [a["pmid"] for a in seen_payloads[0]["direct_articles"]]
    assert "secret-never-output" not in "".join(
        path.read_text(errors="ignore") for path in run.rglob("*") if path.is_file()
    )


def test_modified_cannot_be_based_on_context_or_uncertain_only() -> None:
    packet = {
        "direct_article_pmids": [],
        "indirect_article_pmids": [],
        "context_article_pmids": ["12"],
        "exact_original_item_text": "Original",
    }
    raw = {
        "new_evidence_de": "Kontext.",
        "conclusion_de": "Änderung.",
        "decision": "modified",
        "updated_item_text_de": "Neu",
        "aisurgeon_evidence_class": "C",
        "used_direct_pmids": [],
        "used_indirect_pmids": [],
        "used_context_pmids": ["12"],
        "uncertainty_de": None,
        "review_required": False,
        "review_notes": [],
    }
    with pytest.raises(ValueError, match="modified requires"):
        validate_synthesis(raw, packet)


def test_references_are_consolidated_by_first_appearance_and_deduplicated() -> None:
    blocks = [
        {
            "source_id": "SRC",
            "formal_item_id": "F1",
            "old_reference_ids": ["1"],
            "used_direct_pmids": ["10", "11"],
            "used_indirect_pmids": [],
            "used_context_pmids": [],
        },
        {
            "source_id": "SRC",
            "formal_item_id": "F2",
            "old_reference_ids": ["2"],
            "used_direct_pmids": ["12"],
            "used_indirect_pmids": [],
            "used_context_pmids": [],
        },
    ]
    refs, number_map, findings = consolidate_references(
        old_references=[
            {
                "original_reference_number": "1",
                "reference_id": "R1",
                "exact_original_reference_text": "Old 1",
            },
            {
                "original_reference_number": "2",
                "reference_id": "R2",
                "exact_original_reference_text": "Old 2",
            },
        ],
        articles=[_article("10", doi="10/x"), _article("11", doi="10/x"), _article("12")],
        blocks=blocks,
    )
    assert [r["final_reference_number"] for r in refs] == [1, 2, 3, 4]
    assert number_map["old_reference_numbers"] == {"1": 1, "2": 3}
    assert number_map["new_pubmed_pmids"]["10"] == 2
    assert number_map["new_pubmed_pmids"]["11"] == 2
    assert findings == []


def test_docx_structure_unmodified_not_duplicated_modified_separated_and_resume(
    tmp_path: Path,
) -> None:
    extraction, search, fetch, mapping = _runs(tmp_path)

    class Fake:
        def create(self, prompt, payload):
            return {
                "new_evidence_de": "Neue direkte Evidenz.",
                "conclusion_de": "Minimaler Änderungsvorschlag.",
                "decision": "modified",
                "updated_item_text_de": "Aktualisierter Vorschlag.",
                "aisurgeon_evidence_class": "C",
                "used_direct_pmids": ["10"],
                "used_indirect_pmids": [],
                "used_context_pmids": [],
                "uncertainty_de": None,
                "review_required": False,
                "review_notes": [],
            }

    run = build_updated_guideline(
        extraction_run=extraction,
        search_run=search,
        fetch_run=fetch,
        mapping_run=mapping,
        output_root=tmp_path,
        worker_id="w",
        api_key=SecretStr("secret"),
        client_factory=lambda key, config: Fake(),
        now=lambda: datetime(2026, 7, 16, tzinfo=UTC),
    )
    resumed = build_updated_guideline(
        extraction_run=extraction,
        search_run=search,
        fetch_run=fetch,
        mapping_run=mapping,
        output_root=tmp_path,
        worker_id="w",
        api_key=SecretStr("other"),
        resume_run=run,
        client_factory=lambda key, config: Fake(),
    )
    assert resumed == run
    with pytest.raises(ValueError, match="fingerprint"):
        build_updated_guideline(
            extraction_run=extraction,
            search_run=search,
            fetch_run=fetch,
            mapping_run=mapping,
            output_root=tmp_path,
            worker_id="different",
            api_key=SecretStr("secret"),
            resume_run=run,
            client_factory=lambda key, config: Fake(),
        )
    docx = run / "AISurgeon_Aktualisierte_Leitlinie_GERD_EoE_2026.docx"
    with zipfile.ZipFile(docx) as archive:
        xml = archive.read("word/document.xml").decode("utf-8")
        styles = archive.read("word/styles.xml").decode("utf-8")
        assert "TOC" in xml and "Heading1" in xml
        assert "w:hanging" in xml
        assert "Bisheriger Originalwortlaut" in xml
        assert "Aktualisierungsvorschlag" in xml
        assert xml.count("Originaltext 2 mit Abb. 1.") == 1
        assert "GPT-basiert" not in xml
        assert "Prompt" not in xml
        assert "word/header1.xml" in archive.namelist()
        assert "word/footer1.xml" in archive.namelist()
        assert "Arial" in styles
    qa = json.loads((run / "docx_qa_report.json").read_text())
    assert qa["structural_valid"] is True
    assert not qa["critical_layout_errors"]


def test_limited_run_has_no_final_docx(tmp_path: Path) -> None:
    extraction, search, fetch, mapping = _runs(tmp_path)

    class Fake:
        def create(self, prompt, payload):
            return {
                "new_evidence_de": "Neue direkte Evidenz.",
                "conclusion_de": "Der Text bleibt unverändert.",
                "decision": "rationale_updated",
                "updated_item_text_de": "Originaltext 1 mit Abb. 1.",
                "aisurgeon_evidence_class": "C",
                "used_direct_pmids": ["10"],
                "used_indirect_pmids": [],
                "used_context_pmids": [],
                "uncertainty_de": None,
                "review_required": False,
                "review_notes": [],
            }

    run = build_updated_guideline(
        extraction_run=extraction,
        search_run=search,
        fetch_run=fetch,
        mapping_run=mapping,
        output_root=tmp_path,
        worker_id="w",
        api_key=SecretStr("secret"),
        limit=1,
        client_factory=lambda key, config: Fake(),
    )
    manifest = json.loads((run / "synthesis_manifest.json").read_text())
    assert manifest["status"] == "technical_limited"
    assert not (run / "AISurgeon_Aktualisierte_Leitlinie_GERD_EoE_2026.docx").exists()
