import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from aisurgeon.synthesis.reference_rebuild import (
    find_old_citation_occurrences,
    raw_pmids_in_narrative,
    rebuild_guideline_references,
)

EN_DASH = "\u2013"


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _article(pmid: str, *, doi: str | None = None, title: str | None = None) -> dict:
    return {
        "schema_version": "pubmed_fetch_v1",
        "pmid": pmid,
        "doi": doi,
        "title": title or f"Trial Title {pmid}",
        "abstract": f"Abstract {pmid}",
        "authors": ["A Autor", "B Autor"],
        "journal": "Journal",
        "publication_year": 2025,
        "publication_types": ["Randomized Controlled Trial"],
        "mesh_terms": ["Humans"],
        "has_abstract": True,
    }


def _block(
    formal_item_id: str,
    *,
    comments: list[str] | None = None,
    new_evidence: str = "Neue Evidenz.",
    conclusion: str = "Schlussfolgerung.",
    updated_text: str = "Originaltext.",
) -> dict:
    return {
        "source_id": "SRC",
        "formal_item_id": formal_item_id,
        "sequence_number": int(formal_item_id.removeprefix("F")),
        "original_item_number": formal_item_id.removeprefix("F"),
        "source_native_item_type": "Empfehlung",
        "section_path": ["Kapitel"],
        "exact_original_item_text": "Originaltext mit [2].",
        "exact_original_comments": comments or [],
        "new_evidence_de": new_evidence,
        "aisurgeon_evidence_class": "C",
        "conclusion_de": conclusion,
        "decision": "rationale_updated",
        "updated_item_text_de": updated_text,
        "used_direct_pmids": [],
        "used_indirect_pmids": [],
        "used_context_pmids": [],
        "old_reference_ids": ["2"],
        "new_reference_ids": [],
        "review_required": False,
        "review_notes": [],
    }


def _runs(tmp_path: Path) -> tuple[Path, Path]:
    base = tmp_path / "input"
    extraction = base / "extract"
    search = base / "search"
    fetch = base / "fetch"
    mapping = base / "mapping"
    synthesis = base / "synthesis"
    for path in (extraction, search, fetch, mapping, synthesis):
        path.mkdir(parents=True)
    references = [
        {
            "reference_id": "R2",
            "original_reference_number": "2",
            "exact_original_reference_text": "Exakter Originaleintrag 2.",
        },
        {
            "reference_id": "R50",
            "original_reference_number": "50",
            "exact_original_reference_text": "Exakter Originaleintrag 50.",
        },
        {
            "reference_id": "R51",
            "original_reference_number": "51",
            "exact_original_reference_text": "Exakter Originaleintrag 51.",
        },
        {
            "reference_id": "R52",
            "original_reference_number": "52",
            "exact_original_reference_text": "Exakter Originaleintrag 52.",
        },
        {
            "reference_id": "R123",
            "original_reference_number": "123",
            "exact_original_reference_text": (
                "Already Old Trial. J. doi: 10.1000/old. PMID: 99999999."
            ),
        },
    ]
    articles = [
        _article("35324483"),
        _article("37278156"),
        _article("37448170"),
        _article("39223797"),
        _article("99999999", doi="10.1000/OLD", title="Already Old Trial"),
    ]
    blocks = [
        _block(
            "F1",
            comments=[
                f"Kommentar mit [2], Bereich [50{EN_DASH}52], "
                "[Empfehlung, starker Konsens], Tab. 5 und Abb. 1."
            ],
            new_evidence=(
                "Eine Studie (PMID 35324483). Mehrere Studien PMIDs "
                "37278156, 37448170, 39223797."
            ),
            conclusion="Die neue Evidenz bestätigt dies (PMID 35324483).",
        ),
        _block(
            "F2",
            new_evidence="Bereits original zitierte Publikation PMID 99999999.",
            conclusion="Keine neue N-Referenz erforderlich.",
        ),
    ]
    _jsonl(extraction / "references.jsonl", references)
    _json(extraction / "extraction_manifest.json", {"source_id": "SRC", "status": "completed"})
    _jsonl(fetch / "pubmed_articles.jsonl", articles)
    _json(fetch / "pubmed_fetch_manifest.json", {"source_id": "SRC", "status": "completed"})
    _json(synthesis / "synthesis_summary.json", {"processed_formal_items": 2})
    _jsonl(synthesis / "updated_guideline_blocks.jsonl", blocks)
    _json(
        synthesis / "synthesis_manifest.json",
        {
            "source_id": "SRC",
            "status": "completed_with_review",
            "input_runs": [str(extraction), str(search), str(fetch), str(mapping)],
        },
    )
    original_docx = synthesis / "AISurgeon_Aktualisierte_Leitlinie_GERD_EoE_2026.docx"
    original_docx.write_bytes(b"old-docx-content")
    return synthesis, original_docx


def _document_xml(docx: Path) -> str:
    with zipfile.ZipFile(docx) as archive:
        assert "word/document.xml" in archive.namelist()
        return archive.read("word/document.xml").decode("utf-8")


def test_dual_namespace_rebuild_preserves_originals_replaces_pmids_and_writes_docx(
    tmp_path: Path,
) -> None:
    synthesis, original_docx = _runs(tmp_path)
    original_bytes = original_docx.read_bytes()
    run = rebuild_guideline_references(
        synthesis_run=synthesis,
        output_root=tmp_path / "out",
        now=lambda: datetime(2026, 7, 19, tzinfo=UTC),
    )
    fixed_docx = run / "AISurgeon_Aktualisierte_Leitlinie_GERD_EoE_2026_references_fixed.docx"
    xml = _document_xml(fixed_docx)
    original_refs = [
        json.loads(line)
        for line in (run / "original_references_exact.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    new_refs = [
        json.loads(line)
        for line in (run / "new_references_numbered.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    occurrences = [
        json.loads(line)
        for line in (run / "citation_occurrences.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    summary = json.loads((run / "reference_integrity_summary.json").read_text(encoding="utf-8"))

    assert original_docx.read_bytes() == original_bytes
    assert any(
        ref["original_reference_number"] == "2"
        and ref["exact_original_reference_text"] == "Exakter Originaleintrag 2."
        for ref in original_refs
    )
    assert f"Kommentar mit [2], Bereich [50{EN_DASH}52]" in xml
    range_occurrence = next(
        row for row in occurrences if row["raw_citation"] == f"[50{EN_DASH}52]"
    )
    assert range_occurrence["resolved_reference_numbers"] == ["50", "51", "52"]
    assert "[Empfehlung, starker Konsens]" in xml
    assert "[N1]" in xml
    assert f"[N2{EN_DASH}N4]" in xml
    assert "PMID 35324483" not in xml
    assert "(PMID 35324483)" not in xml
    assert "PMIDs 37278156, 37448170, 39223797" not in xml
    assert "Bereits original zitierte Publikation [123]." in xml
    assert [ref["new_reference_number"] for ref in new_refs] == ["N1", "N2", "N3", "N4"]
    assert all(ref["pmid"] != "99999999" for ref in new_refs)
    assert summary["new_articles_deduplicated_to_old_references"] == 1
    assert summary["missing_old_references"] == []
    assert summary["missing_new_references"] == []
    assert summary["uncited_new_references"] == []
    assert summary["remaining_raw_pmid_mentions"] == []
    assert json.loads((run / "reference_rebuild_manifest.json").read_text())["status"] in {
        "completed",
        "completed_with_review",
    }


def test_missing_original_reference_is_hard_fail(tmp_path: Path) -> None:
    synthesis, _ = _runs(tmp_path)
    blocks_path = synthesis / "updated_guideline_blocks.jsonl"
    blocks = [json.loads(line) for line in blocks_path.read_text().splitlines()]
    blocks[0]["exact_original_comments"] = ["Fehlende Originalreferenz [99]."]
    _jsonl(blocks_path, blocks)
    with pytest.raises(RuntimeError, match="Reference rebuild failed"):
        rebuild_guideline_references(
            synthesis_run=synthesis,
            output_root=tmp_path / "out",
            now=lambda: datetime(2026, 7, 19, tzinfo=UTC),
        )
    manifest_path = next(
        (tmp_path / "out").glob("reference-rebuild-*/reference_rebuild_manifest.json")
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "failed"
    assert "99" in manifest["summary"]["missing_old_references"]


def test_missing_new_reference_is_hard_fail(tmp_path: Path) -> None:
    synthesis, _ = _runs(tmp_path)
    blocks_path = synthesis / "updated_guideline_blocks.jsonl"
    blocks = [json.loads(line) for line in blocks_path.read_text().splitlines()]
    blocks[0]["new_evidence_de"] = "Nicht geladene Studie PMID 88888888."
    _jsonl(blocks_path, blocks)
    with pytest.raises(RuntimeError, match="Reference rebuild failed"):
        rebuild_guideline_references(
            synthesis_run=synthesis,
            output_root=tmp_path / "out",
            now=lambda: datetime(2026, 7, 19, tzinfo=UTC),
        )
    summary_path = next(
        (tmp_path / "out").glob("reference-rebuild-*/reference_integrity_summary.json")
    )
    summary = json.loads(summary_path.read_text())
    assert summary["missing_new_references"] == ["88888888"]
    findings_path = summary_path.parent / "citation_resolution_findings.jsonl"
    findings = [json.loads(line) for line in findings_path.read_text().splitlines()]
    assert any(row["issue_code"] == "missing_new_reference" for row in findings)


def test_raw_pmid_detector_and_nonliterature_markers() -> None:
    blocks = [
        _block(
            "F1",
            comments=["[Empfehlung, starker Konsens] und [2] sowie Abb. 1."],
            new_evidence="Rohe Nennung PMID 35324483.",
        )
    ]
    occurrences, missing = find_old_citation_occurrences(blocks, {"2"})
    assert [row["raw_citation"] for row in occurrences] == ["[2]", "[2]"]
    assert missing == []
    assert raw_pmids_in_narrative(blocks) == [
        {"formal_item_id": "F1", "field": "new_evidence_de", "raw": "PMID 35324483"}
    ]
