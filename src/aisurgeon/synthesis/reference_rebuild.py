"""Postprocess guideline references into old and new citation namespaces."""

import json
import re
import shutil
import subprocess
import zipfile
from collections.abc import Callable
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any

from openpyxl import Workbook

from aisurgeon.extraction.canonical.outputs import write_json, write_jsonl
from aisurgeon.search.pubmed.generation import ensure_external_run_root, file_hash, load_jsonl
from aisurgeon.search.pubmed.query import sha256_text
from aisurgeon.synthesis.updated_guideline import (
    _format_new_reference,
    _git_commit,
    _normalize_title,
    _styles_xml,
    _w_box,
    _w_p,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DUAL_REFERENCE_BUILDER_VERSION = "dual_namespace_reference_builder_v2"
DOCX_REBUILD_VERSION = "aisurgeon_dual_namespace_docx_rebuild_v1"
EN_DASH = "\u2013"
PMID_PATTERN = re.compile(
    r"\(?\bPMIDs?\s*:?\s*((?:\d{7,9})(?:\s*(?:,|;|und|and)\s*\d{7,9})*)\)?",
    re.IGNORECASE,
)
OLD_CITATION_PATTERN = re.compile(r"\[([0-9][0-9,\s\-\u2013]*)\]")


def _json_hash(value: Any) -> str:
    return sha256_text(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _by_id(records: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    return {str(row[key]): row for row in records}


def _write_xlsx(path: Path, sheet_name: str, rows: list[dict[str, Any]]) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = sheet_name[:31]
    headers = sorted({key for row in rows for key in row}) or ["finding_id"]
    sheet.append(headers)
    for row in rows:
        sheet.append(
            [
                json.dumps(row.get(header), ensure_ascii=False)
                if isinstance(row.get(header), (list, dict))
                else row.get(header)
                for header in headers
            ]
        )
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = f"A1:{sheet.cell(1, len(headers)).coordinate}"
    workbook.save(path)


def _reference_number(value: str) -> int:
    return int(str(value).strip())


def _normal_doi(value: str | None) -> str | None:
    if not value:
        return None
    return value.strip().lower().removeprefix("doi:").strip().rstrip(".")


def _extract_doi(text: str) -> str | None:
    match = re.search(r"10\.\d{4,9}/[^\s\]\)]+", text, flags=re.IGNORECASE)
    return _normal_doi(match.group(0)) if match else None


def _extract_pmid(text: str) -> str | None:
    match = re.search(r"\bPMID\s*:?\s*(\d{7,9})\b", text, flags=re.IGNORECASE)
    return match.group(1) if match else None


def _normalize_citation_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def _old_reference_keys(old_refs: list[dict[str, Any]]) -> dict[str, str]:
    keys: dict[str, str] = {}
    for ref in old_refs:
        number = str(ref["original_reference_number"])
        text = ref["exact_original_reference_text"]
        pmid = _extract_pmid(text)
        doi = _extract_doi(text)
        if pmid:
            keys[f"pmid:{pmid}"] = number
        if doi:
            keys[f"doi:{doi}"] = number
        normalized = _normalize_citation_text(text)
        if normalized:
            keys[f"text:{number}"] = normalized
    return keys


def _dedupe_to_old_reference(
    article: dict[str, Any], old_refs: list[dict[str, Any]], old_keys: dict[str, str]
) -> str | None:
    pmid = str(article["pmid"])
    doi = _normal_doi(article.get("doi"))
    if f"pmid:{pmid}" in old_keys:
        return old_keys[f"pmid:{pmid}"]
    if doi and f"doi:{doi}" in old_keys:
        return old_keys[f"doi:{doi}"]
    title = _normalize_title(article.get("title"))
    if len(title) < 24:
        return None
    matches = [
        ref["original_reference_number"]
        for ref in old_refs
        if title and title in _normalize_citation_text(ref["exact_original_reference_text"])
    ]
    return str(matches[0]) if len(matches) == 1 else None


def _expand_old_citation(content: str) -> list[str]:
    values: list[str] = []
    for part in re.split(r"\s*,\s*", content.replace(EN_DASH, "-")):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = [int(x.strip()) for x in part.split("-", 1)]
            values.extend(str(n) for n in range(start, end + 1))
        else:
            values.append(str(int(part)))
    return values


def _format_old_group(numbers: list[str]) -> str:
    return "[" + ", ".join(numbers) + "]"


def _format_new_group(numbers: list[int]) -> str:
    ordered = sorted(set(numbers))
    if not ordered:
        return ""
    ranges: list[str] = []
    start = prev = ordered[0]
    for number in ordered[1:]:
        if number == prev + 1:
            prev = number
            continue
        ranges.append(f"N{start}{EN_DASH}N{prev}" if start != prev else f"N{start}")
        start = prev = number
    ranges.append(f"N{start}{EN_DASH}N{prev}" if start != prev else f"N{start}")
    return "[" + ", ".join(ranges) + "]"


def find_old_citation_occurrences(
    blocks: list[dict[str, Any]], old_reference_numbers: set[str]
) -> tuple[list[dict[str, Any]], list[str]]:
    occurrences: list[dict[str, Any]] = []
    missing: set[str] = set()
    fields = ("exact_original_item_text", "exact_original_comments")
    for block in blocks:
        for field in fields:
            values = block.get(field, [])
            texts = values if isinstance(values, list) else [values]
            for index, text in enumerate(texts):
                for match in OLD_CITATION_PATTERN.finditer(text or ""):
                    numbers = _expand_old_citation(match.group(1))
                    missing.update(
                        number for number in numbers if number not in old_reference_numbers
                    )
                    occurrences.append(
                        {
                            "namespace": "original",
                            "formal_item_id": block["formal_item_id"],
                            "field": field,
                            "field_index": index,
                            "raw_citation": match.group(0),
                            "resolved_reference_numbers": numbers,
                        }
                    )
    return occurrences, sorted(missing, key=int)


def replace_new_pmid_citations(
    blocks: list[dict[str, Any]],
    articles: list[dict[str, Any]],
    old_refs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    articles_by_pmid = _by_id(articles, "pmid")
    old_keys = _old_reference_keys(old_refs)
    new_map: dict[str, int] = {}
    old_dedupe_map: dict[str, str] = {}
    new_refs: list[dict[str, Any]] = []
    occurrences: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []

    def cite_pmids(pmids: list[str], block: dict[str, Any], field: str) -> str:
        old_numbers: list[str] = []
        new_numbers: list[int] = []
        for pmid in pmids:
            article = articles_by_pmid.get(pmid)
            if article is None:
                findings.append(
                    {
                        "finding_id": f"CITE_MISSING_PMID_{block['formal_item_id']}_{pmid}",
                        "severity": "error",
                        "issue_code": "missing_new_reference",
                        "formal_item_id": block["formal_item_id"],
                        "pmid": pmid,
                        "message": f"Zitierte PMID {pmid} fehlt im Fetch-Artikelbestand.",
                    }
                )
                continue
            old_number = _dedupe_to_old_reference(article, old_refs, old_keys)
            if old_number:
                old_dedupe_map[pmid] = old_number
                old_numbers.append(old_number)
                continue
            if pmid not in new_map:
                new_map[pmid] = len(new_map) + 1
                new_refs.append(
                    {
                        "reference_id": f"N{new_map[pmid]}",
                        "new_reference_number": f"N{new_map[pmid]}",
                        "ordinal": new_map[pmid],
                        "pmid": pmid,
                        "doi": article.get("doi"),
                        "normalized_title": _normalize_title(article.get("title")),
                        "full_citation": _format_new_reference(article),
                        "first_seen_in_formal_item_id": block["formal_item_id"],
                    }
                )
            new_numbers.append(new_map[pmid])
        parts = []
        if old_numbers:
            parts.append(_format_old_group(sorted(set(old_numbers), key=int)))
        if new_numbers:
            parts.append(_format_new_group(new_numbers))
        occurrences.append(
            {
                "namespace": "new_pubmed",
                "formal_item_id": block["formal_item_id"],
                "field": field,
                "raw_pmids": pmids,
                "resolved_original_numbers": sorted(set(old_numbers), key=int),
                "resolved_new_numbers": [f"N{n}" for n in sorted(set(new_numbers))],
            }
        )
        return " ".join(parts)

    processed = []
    for block in blocks:
        updated = dict(block)
        for field in ("new_evidence_de", "conclusion_de", "updated_item_text_de"):
            text = str(updated.get(field) or "")

            def repl(
                match: re.Match[str],
                *,
                current_block: dict[str, Any] = block,
                current_field: str = field,
            ) -> str:
                pmids = re.findall(r"\d{7,9}", match.group(1))
                return cite_pmids(pmids, current_block, current_field)

            updated[field] = PMID_PATTERN.sub(repl, text)
        processed.append(updated)
    number_map = {
        "new_pubmed_pmids": {pmid: f"N{number}" for pmid, number in new_map.items()},
        "new_pubmed_ordinals": new_map,
        "deduplicated_to_original": old_dedupe_map,
    }
    return processed, new_refs, number_map, occurrences + findings


def raw_pmids_in_narrative(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hits = []
    for block in blocks:
        for field in ("new_evidence_de", "conclusion_de", "updated_item_text_de"):
            for match in PMID_PATTERN.finditer(str(block.get(field) or "")):
                hits.append(
                    {
                        "formal_item_id": block["formal_item_id"],
                        "field": field,
                        "raw": match.group(0),
                    }
                )
    return hits


def _docx_xml_dual(
    blocks: list[dict[str, Any]],
    original_refs: list[dict[str, Any]],
    new_refs: list[dict[str, Any]],
    summary: dict[str, Any],
) -> str:
    body = [
        _w_p("AISurgeon Aktualisierte Leitlinie GERD/EoE 2026", style="Title"),
        _w_p("Automatisch unterstützter wissenschaftlicher Aktualisierungsentwurf"),
        _w_p("Nicht als konsentierte AWMF-Leitlinie verwenden."),
        '<w:p><w:r><w:br w:type="page"/></w:r></w:p>',
        _w_p("Dokumentstatus und Methodik", style="Heading1"),
        _w_p(
            "Dieser Rebuild übernimmt die vorhandenen Synthesetexte unverändert und korrigiert "
            "ausschließlich die Trennung alter und neuer Literaturverweise.",
            jc="both",
        ),
        _w_p("Inhaltsverzeichnis", style="Heading1"),
        '<w:p><w:r><w:fldChar w:fldCharType="begin"/></w:r><w:r><w:instrText>'
        'TOC \\o "1-3" \\h \\z \\u</w:instrText></w:r><w:r><w:fldChar w:fldCharType="end"/>'
        "</w:r></w:p>",
        '<w:p><w:r><w:br w:type="page"/></w:r></w:p>',
        _w_p("Zusammenfassung des Aktualisierungslaufs", style="Heading1"),
        _w_p(
            f"FormalItems: {summary.get('processed_formal_items', len(blocks))}; "
            f"Originalreferenzen: {summary['original_reference_count']}; "
            f"neue Referenzen: {summary['new_reference_count']}.",
            jc="both",
        ),
        '<w:p><w:r><w:br w:type="page"/></w:r></w:p>',
        _w_p("Aktualisierte Leitlinienblöcke", style="Heading1"),
    ]
    previous_section = None
    for block in blocks:
        section = " / ".join(block.get("section_path") or [])
        if section and section != previous_section:
            body.append(_w_p(section, style="Heading2"))
            previous_section = section
        body.append(
            _w_p(
                f"{block['original_item_number']} - {block['source_native_item_type']}",
                style="Heading3",
            )
        )
        if block["decision"] == "modified":
            body.append(_w_box("Bisheriger Originalwortlaut", block["exact_original_item_text"]))
            body.append(
                _w_box(
                    "Aktualisierungsvorschlag - KI-gestuetzt; nicht freigegeben",
                    block["updated_item_text_de"],
                    changed=True,
                )
            )
        else:
            body.append(
                _w_box(
                    f"Fortbestehendes Item ({block['source_native_item_type']})",
                    block["exact_original_item_text"],
                )
            )
        for comment in block["exact_original_comments"]:
            body.append(_w_p("Bisherige Begründung", style="Heading3"))
            body.append(_w_p(comment, jc="both"))
        body.append(_w_p("Neue Evidenz", style="Heading3"))
        body.append(_w_p(block["new_evidence_de"], jc="both"))
        body.append(_w_p("Schlussfolgerung", style="Heading3"))
        body.append(_w_p(block["conclusion_de"], jc="both"))
        body.append(_w_p(f"Entscheidung: {block['decision']}", bold=True))
    body.extend(
        [
            '<w:p><w:r><w:br w:type="page"/></w:r></w:p>',
            _w_p("Literaturverzeichnis", style="Heading1"),
            _w_p("Originalreferenzen der Leitlinie", style="Heading2"),
        ]
    )
    for ref in original_refs:
        body.append(
            '<w:p><w:pPr><w:ind w:left="720" w:hanging="360"/><w:jc w:val="left"/>'
            f'</w:pPr><w:r><w:t xml:space="preserve">'
            f'[{ref["original_reference_number"]}] '
            f'{escape(ref["exact_original_reference_text"])}</w:t></w:r></w:p>'
        )
    body.append(_w_p("Neue Referenzen der automatisierten Aktualisierung", style="Heading2"))
    for ref in new_refs:
        body.append(
            '<w:p><w:pPr><w:ind w:left="720" w:hanging="360"/><w:jc w:val="left"/>'
            f'</w:pPr><w:r><w:t xml:space="preserve">[{ref["new_reference_number"]}] '
            f'{escape(ref["full_citation"])}</w:t></w:r></w:p>'
        )
    body.extend(
        [
            '<w:p><w:r><w:br w:type="page"/></w:r></w:p>',
            _w_p("Anhang: Review Findings und technische Metadaten", style="Heading1"),
            _w_p("Details stehen in den begleitenden JSONL- und XLSX-Dateien."),
        ]
    )
    sect = (
        '<w:sectPr><w:headerReference w:type="default" r:id="rIdHeader1"/>'
        '<w:footerReference w:type="default" r:id="rIdFooter1"/>'
        '<w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1247" w:right="1247" '
        'w:bottom="1247" w:left="1247" w:header="708" w:footer="708" w:gutter="0"/>'
        "</w:sectPr>"
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"<w:body>{''.join(body)}{sect}</w:body></w:document>"
    )


def write_dual_namespace_docx(
    path: Path,
    blocks: list[dict[str, Any]],
    original_refs: list[dict[str, Any]],
    new_refs: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    if path.exists():
        raise FileExistsError(path)
    pkg_rel = "http://schemas.openxmlformats.org/package/2006/relationships"
    office_rel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    pkg_ct = "http://schemas.openxmlformats.org/package/2006/content-types"
    app_ct = "application/vnd.openxmlformats-officedocument.wordprocessingml"
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<Relationships xmlns="{pkg_rel}">'
        f'<Relationship Id="rId1" Type="{office_rel}/officeDocument" '
        'Target="word/document.xml"/></Relationships>'
    )
    doc_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<Relationships xmlns="{pkg_rel}">'
        f'<Relationship Id="rIdHeader1" Type="{office_rel}/header" Target="header1.xml"/>'
        f'<Relationship Id="rIdFooter1" Type="{office_rel}/footer" Target="footer1.xml"/>'
        "</Relationships>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<Types xmlns="{pkg_ct}">'
        '<Default Extension="rels" '
        'ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        f'<Override PartName="/word/document.xml" ContentType="{app_ct}.document.main+xml"/>'
        f'<Override PartName="/word/styles.xml" ContentType="{app_ct}.styles+xml"/>'
        f'<Override PartName="/word/header1.xml" ContentType="{app_ct}.header+xml"/>'
        f'<Override PartName="/word/footer1.xml" ContentType="{app_ct}.footer+xml"/>'
        '<Override PartName="/docProps/core.xml" '
        'ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
        "</Types>"
    )
    header = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:hdr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"{_w_p('AISurgeon GERD/EoE Aktualisierungsentwurf')}</w:hdr>"
    )
    footer = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:ftr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"{_w_p('Seite X von Y | Automatisch unterstützter Entwurf - ')}"
        f"{_w_p('menschliche Validierung erforderlich')}</w:ftr>"
    )
    core = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<cp:coreProperties '
        'xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/">'
        "<dc:title>AISurgeon Referenz-Rebuild GERD/EoE 2026</dc:title>"
        "<dc:creator>AISurgeon Living Guideline Platform</dc:creator>"
        "</cp:coreProperties>"
    )
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", rels)
        archive.writestr("word/_rels/document.xml.rels", doc_rels)
        archive.writestr(
            "word/document.xml", _docx_xml_dual(blocks, original_refs, new_refs, summary)
        )
        archive.writestr("word/styles.xml", _styles_xml())
        archive.writestr("word/header1.xml", header)
        archive.writestr("word/footer1.xml", footer)
        archive.writestr("docProps/core.xml", core)


def run_reference_docx_qa(docx_path: Path, run_dir: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "docx_path": str(docx_path),
        "structural_valid": False,
        "citation_integrity_valid": False,
        "render_attempted": False,
        "render_successful": False,
        "critical_layout_errors": [],
        "warnings": [],
        "qa_pdf": None,
        "page_images": [],
    }
    with zipfile.ZipFile(docx_path) as archive:
        names = set(archive.namelist())
        required = {"word/document.xml", "word/styles.xml", "word/header1.xml", "word/footer1.xml"}
        missing = sorted(required - names)
        if missing:
            report["critical_layout_errors"].append(f"Missing DOCX parts: {missing}")
        xml = archive.read("word/document.xml").decode("utf-8")
        if "Originalreferenzen der Leitlinie" not in xml or "Neue Referenzen" not in xml:
            report["critical_layout_errors"].append("Dual literature sections missing")
        narrative_xml = xml.split("Literaturverzeichnis", 1)[0]
        if re.search(r"\bPMIDs?\s*:?\s*\d{7,9}", narrative_xml, flags=re.IGNORECASE):
            report["critical_layout_errors"].append("Raw PMID citation remains in document text")
        report["structural_valid"] = not report["critical_layout_errors"]
        report["citation_integrity_valid"] = report["structural_valid"]
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        report["warnings"].append("LibreOffice/soffice not available; PDF render skipped.")
        return report
    qa_dir = run_dir / "docx_render_qa"
    qa_dir.mkdir(exist_ok=True)
    report["render_attempted"] = True
    result = subprocess.run(
        [soffice, "--headless", "--convert-to", "pdf", "--outdir", str(qa_dir), str(docx_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    pdf = qa_dir / f"{docx_path.stem}.pdf"
    if result.returncode != 0 or not pdf.is_file():
        report["critical_layout_errors"].append("LibreOffice PDF conversion failed")
        return report
    report["render_successful"] = True
    report["qa_pdf"] = str(pdf)
    pdftoppm = shutil.which("pdftoppm")
    if pdftoppm:
        prefix = qa_dir / "page"
        subprocess.run([pdftoppm, "-png", str(pdf), str(prefix)], check=False)
        report["page_images"] = [str(p) for p in sorted(qa_dir.glob("page-*.png"))]
    return report


def rebuild_guideline_references(
    *,
    synthesis_run: Path,
    output_root: Path,
    resume_run: Path | None = None,
    output_name: str = "AISurgeon_Aktualisierte_Leitlinie_GERD_EoE_2026_references_fixed.docx",
    original_references_path: Path | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> Path:
    manifest_path = synthesis_run / "synthesis_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("synthesis_manifest.json missing")
    synthesis_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if synthesis_manifest.get("status") not in {"completed", "completed_with_review"}:
        raise ValueError("Synthesis run is not completed")
    extraction_run = Path(synthesis_manifest["input_runs"][0])
    fetch_run = Path(synthesis_manifest["input_runs"][2])
    original_refs_input = original_references_path or extraction_run / "references.jsonl"
    required: dict[Path, list[str]] = {}
    for root, names in (
        (
            synthesis_run,
            [
                "updated_guideline_blocks.jsonl",
                "synthesis_manifest.json",
                "synthesis_summary.json",
            ],
        ),
        (extraction_run, ["extraction_manifest.json"]),
        (original_refs_input.parent, [original_refs_input.name]),
        (fetch_run, ["pubmed_articles.jsonl", "pubmed_fetch_manifest.json"]),
    ):
        required.setdefault(root, []).extend(names)
    input_hashes = {
        str((root / name).resolve()): file_hash(root / name)
        for root, names in required.items()
        for name in names
    }
    source_id = synthesis_manifest["source_id"]
    fingerprint = {
        "source_id": source_id,
        "input_synthesis_run": str(synthesis_run.resolve()),
        "input_extraction_run": str(extraction_run.resolve()),
        "input_fetch_run": str(fetch_run.resolve()),
        "input_original_references_path": str(original_refs_input.resolve()),
        "input_file_hashes": input_hashes,
        "reference_builder_version": DUAL_REFERENCE_BUILDER_VERSION,
        "docx_rebuild_version": DOCX_REBUILD_VERSION,
        "git_commit": _git_commit(),
        "output_name": output_name,
    }
    if resume_run:
        run_dir = resume_run.resolve()
        if json.loads((run_dir / "checkpoint_fingerprint.json").read_text()) != fingerprint:
            raise ValueError("Resume fingerprint does not match")
        if (run_dir / "reference_rebuild_manifest.json").is_file():
            return run_dir
    else:
        root = ensure_external_run_root(output_root, synthesis_run)
        run_dir = root / (
            f"reference-rebuild-{now():%Y%m%dT%H%M%S%fZ}-{source_id}-"
            f"{sha256_text(json.dumps(fingerprint, sort_keys=True))[:8]}"
        )
        run_dir.mkdir(parents=True, exist_ok=False)
        write_json(run_dir / "checkpoint_fingerprint.json", fingerprint)
    blocks = load_jsonl(synthesis_run / "updated_guideline_blocks.jsonl")
    original_refs = sorted(
        load_jsonl(original_refs_input),
        key=lambda row: _reference_number(row["original_reference_number"]),
    )
    articles = load_jsonl(fetch_run / "pubmed_articles.jsonl")
    original_numbers = {str(row["original_reference_number"]) for row in original_refs}
    old_occurrences, missing_old = find_old_citation_occurrences(blocks, original_numbers)
    processed_blocks, new_refs, new_number_map, new_occurrences = replace_new_pmid_citations(
        blocks, articles, original_refs
    )
    raw_pmids = raw_pmids_in_narrative(processed_blocks)
    missing_new = sorted(
        {
            str(row["pmid"])
            for row in new_occurrences
            if row.get("issue_code") == "missing_new_reference"
        }
    )
    cited_new = {ref["new_reference_number"] for ref in new_refs}
    new_numbers = {ref["new_reference_number"] for ref in new_refs}
    missing_new_tokens = sorted(cited_new - new_numbers)
    unquoted_new = sorted(new_numbers - cited_new)
    findings = [
        {
            "finding_id": f"MISSING_OLD_REF_{number}",
            "severity": "error",
            "issue_code": "missing_original_reference",
            "reference_number": number,
            "message": f"Originalreferenz {number} wird zitiert, fehlt aber in references.jsonl.",
        }
        for number in missing_old
    ]
    findings.extend(row for row in new_occurrences if row.get("severity") == "error")
    findings.extend(
        {
            "finding_id": f"RAW_PMID_{index + 1}",
            "severity": "error",
            "issue_code": "raw_pmid_remaining",
            **hit,
        }
        for index, hit in enumerate(raw_pmids)
    )
    occurrences = [*old_occurrences, *[r for r in new_occurrences if "namespace" in r]]
    summary = {
        "source_id": source_id,
        "reference_builder_version": DUAL_REFERENCE_BUILDER_VERSION,
        "original_reference_count": len(original_refs),
        "highest_original_reference_number": max(
            (_reference_number(row["original_reference_number"]) for row in original_refs),
            default=0,
        ),
        "old_citation_occurrences": len(old_occurrences),
        "new_citation_occurrences": sum(
            1 for row in occurrences if row["namespace"] == "new_pubmed"
        ),
        "new_reference_count": len(new_refs),
        "new_articles_deduplicated_to_old_references": len(
            new_number_map["deduplicated_to_original"]
        ),
        "missing_old_references": missing_old,
        "missing_new_references": missing_new + missing_new_tokens,
        "uncited_new_references": unquoted_new,
        "remaining_raw_pmid_mentions": raw_pmids,
        "ambiguous_deduplications": [],
    }
    write_jsonl(run_dir / "original_references_exact.jsonl", original_refs)
    write_jsonl(run_dir / "new_references_numbered.jsonl", new_refs)
    write_json(
        run_dir / "old_reference_number_map.json",
        {
            str(row["original_reference_number"]): str(row["original_reference_number"])
            for row in original_refs
        },
    )
    write_json(run_dir / "new_reference_number_map.json", new_number_map)
    write_jsonl(run_dir / "citation_occurrences.jsonl", occurrences)
    write_json(run_dir / "citation_resolution_report.json", summary)
    write_jsonl(run_dir / "citation_resolution_findings.jsonl", findings)
    _write_xlsx(run_dir / "citation_resolution_findings.xlsx", "citation_findings", findings)
    write_json(run_dir / "reference_integrity_summary.json", summary)
    docx_path = run_dir / output_name
    fatal = bool(
        missing_old or missing_new or missing_new_tokens or unquoted_new or raw_pmids or findings
    )
    qa = None
    if not fatal:
        write_dual_namespace_docx(docx_path, processed_blocks, original_refs, new_refs, summary)
        qa = run_reference_docx_qa(docx_path, run_dir)
        fatal = bool(qa["critical_layout_errors"])
    else:
        qa = {
            "docx_path": None,
            "structural_valid": False,
            "citation_integrity_valid": False,
            "render_attempted": False,
            "render_successful": False,
            "critical_layout_errors": ["Reference integrity hard-fail; DOCX not generated."],
            "warnings": [],
        }
    write_json(run_dir / "docx_qa_report.json", qa)
    status = "failed" if fatal else ("completed_with_review" if qa.get("warnings") else "completed")
    write_json(
        run_dir / "reference_rebuild_manifest.json",
        {
            **fingerprint,
            "created_at": now().isoformat(),
            "status": status,
            "summary": summary,
            "output_files": {
                p.name: file_hash(p)
                for p in run_dir.iterdir()
                if p.is_file() and p.name != "reference_rebuild_manifest.json"
            },
        },
    )
    if fatal:
        raise RuntimeError(f"Reference rebuild failed; run directory: {run_dir}")
    return run_dir
