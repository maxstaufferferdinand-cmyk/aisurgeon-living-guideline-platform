import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

from pydantic import SecretStr

from aisurgeon.synthesis.reference_repair import (
    OriginalReferenceRepairBatch,
    bibliography_pages_from_document_map,
    merge_repaired_references,
    original_reference_requirements,
    run_reference_repair_and_rebuild,
)

EN_DASH = "\u2013"


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _article(pmid: str) -> dict:
    return {
        "schema_version": "pubmed_fetch_v1",
        "pmid": pmid,
        "doi": None,
        "title": f"New Trial {pmid}",
        "abstract": f"Abstract {pmid}",
        "authors": ["A Autor"],
        "journal": "Journal",
        "publication_year": 2025,
        "publication_types": ["Randomized Controlled Trial"],
        "mesh_terms": ["Humans"],
        "has_abstract": True,
    }


def _repair_entry(number: int, text: str | None = None, page: int = 52) -> dict:
    return {
        "schema_version": "original_reference_repair_v1",
        "source_id": "SRC",
        "original_reference_number": str(number),
        "exact_reference_text": text or f"PDF Reference {number}. Journal 2020; 1: 1-2.",
        "page_start": page,
        "page_end": page,
        "column_start": "left",
        "column_end": "left",
        "continuation_detected": False,
        "extraction_confidence": 0.99,
        "review_required": False,
        "review_notes": [],
    }


def _block() -> dict:
    return {
        "source_id": "SRC",
        "formal_item_id": "F1",
        "sequence_number": 1,
        "original_item_number": "1",
        "source_native_item_type": "Empfehlung",
        "section_path": ["Kapitel"],
        "exact_original_item_text": "Originaltext.",
        "exact_original_comments": [
            f"Kommentar mit Bereich [46{EN_DASH}64], "
            "[Empfehlung, starker Konsens], Tab. 5 und Abb. 1."
        ],
        "new_evidence_de": "Neue Evidenz PMID 35324483.",
        "aisurgeon_evidence_class": "C",
        "conclusion_de": "Schlussfolgerung PMID 35324483.",
        "decision": "rationale_updated",
        "updated_item_text_de": "Originaltext.",
        "used_direct_pmids": ["35324483"],
        "used_indirect_pmids": [],
        "used_context_pmids": [],
        "old_reference_ids": [],
        "new_reference_ids": [],
        "review_required": False,
        "review_notes": [],
    }


def _runs(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    base = tmp_path / "input"
    extraction = base / "extract"
    search = base / "search"
    fetch = base / "fetch"
    mapping = base / "mapping"
    synthesis = base / "synthesis"
    failed_reference = base / "failed-reference"
    for path in (extraction, search, fetch, mapping, synthesis, failed_reference):
        path.mkdir(parents=True)
    _json(
        extraction / "document_map.validated.json",
        {
            "source_id": "SRC",
            "bibliography_page_ranges": [
                {"page_start": 52, "page_end": 71, "description": "Literatur"}
            ],
        },
    )
    _json(extraction / "extraction_manifest.json", {"source_id": "SRC", "status": "completed"})
    _jsonl(
        extraction / "references.jsonl",
        [
            {
                "schema_version": "canonical_extraction_v2",
                "source_id": "SRC",
                "page_start": 52,
                "page_end": 52,
                "extraction_confidence": 0.99,
                "review_required": False,
                "review_reasons": [],
                "reference_id": "R1",
                "original_reference_number": "1",
                "exact_original_reference_text": "Existing Reference 1.",
            },
            {
                "schema_version": "canonical_extraction_v2",
                "source_id": "SRC",
                "page_start": 52,
                "page_end": 52,
                "extraction_confidence": 0.99,
                "review_required": False,
                "review_reasons": [],
                "reference_id": "R46",
                "original_reference_number": "46",
                "exact_original_reference_text": "PDF Reference 46. Journal 2020; 1: 1-2.",
            },
        ],
    )
    _jsonl(fetch / "pubmed_articles.jsonl", [_article("35324483")])
    _json(fetch / "pubmed_fetch_manifest.json", {"source_id": "SRC", "status": "completed"})
    _jsonl(synthesis / "updated_guideline_blocks.jsonl", [_block()])
    _json(synthesis / "synthesis_summary.json", {"processed_formal_items": 1})
    _json(
        synthesis / "synthesis_manifest.json",
        {
            "source_id": "SRC",
            "status": "completed_with_review",
            "input_runs": [str(extraction), str(search), str(fetch), str(mapping)],
        },
    )
    _json(
        failed_reference / "reference_rebuild_manifest.json",
        {
            "source_id": "SRC",
            "status": "failed",
            "summary": {"missing_old_references": [str(n) for n in range(47, 65)]},
        },
    )
    pdf = base / "source.pdf"
    pdf.write_bytes(b"%PDF-targeted-test")
    return pdf, extraction, synthesis, failed_reference, fetch


class FakeRepairClient:
    prompts: ClassVar[list[str]] = []
    deleted = False

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def upload_pdf(self, pdf_path: Path):
        return SimpleNamespace(uri=f"file://{pdf_path.name}", mime_type="application/pdf", name="f")

    def request_repair(self, *, remote, prompt: str, source_id: str):
        self.prompts.append(prompt)
        entries = [_repair_entry(n) for n in range(47, 65)]
        entries.append(
            _repair_entry(
                65,
                "A long reference that continues across the right column and the next page.",
                page=53,
            )
        )
        raw = json.dumps(
            {
                "schema_version": "original_reference_repair_v1",
                "source_id": source_id,
                "references": entries,
            },
            ensure_ascii=False,
        )
        return OriginalReferenceRepairBatch.model_validate_json(raw), raw, {"total_tokens": 10}

    def delete_remote(self, remote) -> bool:
        self.deleted = True
        return True


def test_missing_references_46_to_64_are_requirements_and_pages_selected(tmp_path: Path) -> None:
    _, extraction, synthesis, _, _ = _runs(tmp_path)
    refs = [json.loads(line) for line in (extraction / "references.jsonl").read_text().splitlines()]
    requirements = original_reference_requirements(
        synthesis_run=synthesis, existing_references=refs
    )
    assert requirements["missing_original_reference_numbers"] == [str(n) for n in range(47, 65)]
    assert "46" in requirements["cited_original_reference_numbers"]
    assert bibliography_pages_from_document_map(
        json.loads((extraction / "document_map.validated.json").read_text())
    )["page_start"] == 52


def test_merge_rules_keep_existing_repair_incomplete_and_block_conflict() -> None:
    existing = [
        {
            "schema_version": "canonical_extraction_v2",
            "source_id": "SRC",
            "page_start": 52,
            "page_end": 52,
            "extraction_confidence": 0.9,
            "review_required": False,
            "review_reasons": [],
            "reference_id": "R46",
            "original_reference_number": "46",
            "exact_original_reference_text": "Complete Reference 46.",
        },
        {
            "schema_version": "canonical_extraction_v2",
            "source_id": "SRC",
            "page_start": 52,
            "page_end": 52,
            "extraction_confidence": 0.9,
            "review_required": False,
            "review_reasons": [],
            "reference_id": "R47",
            "original_reference_number": "47",
            "exact_original_reference_text": "Truncated",
        },
        {
            "schema_version": "canonical_extraction_v2",
            "source_id": "SRC",
            "page_start": 52,
            "page_end": 52,
            "extraction_confidence": 0.9,
            "review_required": False,
            "review_reasons": [],
            "reference_id": "R48",
            "original_reference_number": "48",
            "exact_original_reference_text": "Different existing reference.",
        },
    ]
    repaired, report, findings = merge_repaired_references(
        source_id="SRC",
        existing_references=existing,
        repaired_entries=[
            _repair_entry(46, "Complete Reference 46."),
            _repair_entry(47, "Truncated but now complete with journal and pages."),
            _repair_entry(48, "Contradictory repaired reference."),
            _repair_entry(49, "New Reference 49."),
        ],
        required_numbers=["46", "47", "48", "49"],
    )
    by_number = {row["original_reference_number"]: row for row in repaired}
    assert by_number["46"]["exact_original_reference_text"] == "Complete Reference 46."
    assert by_number["47"]["exact_original_reference_text"].startswith("Truncated but now")
    assert by_number["48"]["exact_original_reference_text"] == "Different existing reference."
    assert by_number["49"]["exact_original_reference_text"] == "New Reference 49."
    assert report["replaced_reference_numbers"] == ["47"]
    assert any(row["issue_code"] == "conflicting_repaired_reference" for row in findings)


def test_repair_and_rebuild_creates_docx_only_with_complete_integrity(tmp_path: Path) -> None:
    pdf, extraction, synthesis, failed_reference, _ = _runs(tmp_path)
    original_refs_before = (extraction / "references.jsonl").read_bytes()
    run = run_reference_repair_and_rebuild(
        pdf=pdf,
        extraction_run=extraction,
        synthesis_run=synthesis,
        failed_reference_run=failed_reference,
        output_root=tmp_path / "out",
        api_key=SecretStr("secret-never-output"),
        client_factory=FakeRepairClient,
        now=lambda: datetime(2026, 7, 19, tzinfo=UTC),
    )
    assert (extraction / "references.jsonl").read_bytes() == original_refs_before
    manifest = json.loads((run / "reference_repair_manifest.json").read_text())
    rebuild = Path(manifest["reference_rebuild_run"])
    docx = rebuild / "AISurgeon_Aktualisierte_Leitlinie_GERD_EoE_2026_references_repaired.docx"
    assert docx.is_file()
    with zipfile.ZipFile(docx) as archive:
        xml = archive.read("word/document.xml").decode("utf-8")
    assert f"[46{EN_DASH}64]" in xml
    assert "[N1]" in xml
    assert "PMID 35324483" not in xml.split("Literaturverzeichnis", 1)[0]
    summary = json.loads((rebuild / "reference_integrity_summary.json").read_text())
    assert summary["missing_old_references"] == []
    assert summary["remaining_raw_pmid_mentions"] == []
    assert "secret-never-output" not in "".join(
        path.read_text(errors="ignore") for path in run.rglob("*") if path.is_file()
    )


def test_conflicting_repair_blocks_final_docx(tmp_path: Path) -> None:
    existing = [
        {
            "schema_version": "canonical_extraction_v2",
            "source_id": "SRC",
            "page_start": 52,
            "page_end": 52,
            "extraction_confidence": 0.9,
            "review_required": False,
            "review_reasons": [],
            "reference_id": "R46",
            "original_reference_number": "46",
            "exact_original_reference_text": "Existing Reference 46.",
        }
    ]
    _, _, findings = merge_repaired_references(
        source_id="SRC",
        existing_references=existing,
        repaired_entries=[_repair_entry(46, "Conflicting Reference 46.")],
        required_numbers=["46"],
    )
    assert any(row["severity"] == "error" for row in findings)
