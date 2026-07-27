"""Synthesis, references, and deterministic DOCX output for updated guidelines."""

import json
import re
import shutil
import subprocess
import zipfile
from collections import Counter, defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any, Literal

from openpyxl import Workbook
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from aisurgeon.extraction.canonical.outputs import write_json, write_jsonl
from aisurgeon.search.pubmed.generation import ensure_external_run_root, file_hash, load_jsonl
from aisurgeon.search.pubmed.query import sha256_text

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_CONFIG_PATH = PROJECT_ROOT / "config/models/openai_guideline_synthesis_v1.json"
PROMPT_PATH = PROJECT_ROOT / "config/prompts/openai_guideline_synthesis_v1.txt"
SYNTHESIS_SCHEMA_VERSION = "updated_guideline_synthesis_v1"
PACKET_VERSION = "item_evidence_packet_v1"
DIGEST_VERSION = "item_evidence_digest_v1"
REFERENCE_BUILDER_VERSION = "consolidated_references_v1"
DOCX_RENDERER_VERSION = "aisurgeon_ooxml_docx_v1"
INFLUENTIAL_MAPPING_DECISIONS = {"include_direct", "include_indirect"}
CONTEXT_DECISIONS = {"context_only"}
UNCERTAIN_DECISIONS = {"uncertain_review_required"}
SYNTHESIS_DECISIONS = {
    "insufficient_new_evidence",
    "unchanged",
    "rationale_updated",
    "modified",
}
PUBLIC_FORBIDDEN_TERMS = (
    "Evidence Packet",
    "Mapping",
    "LLM confidence",
    "Debug",
    "Prompt",
    "full_text_review_required",
    "OLD-",
    "NEW-PMID-",
    "LG-NEW-",
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SynthesisDraft(StrictModel):
    new_evidence_de: str
    conclusion_de: str
    decision: Literal["insufficient_new_evidence", "unchanged", "rationale_updated", "modified"]
    updated_item_text_de: str
    aisurgeon_evidence_class: Literal["A", "B", "C", "none"]
    used_direct_pmids: list[str] = Field(default_factory=list)
    used_indirect_pmids: list[str] = Field(default_factory=list)
    used_context_pmids: list[str] = Field(default_factory=list)
    uncertainty_de: str | None = None
    review_required: bool = False
    review_notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def pmids_are_unique(self) -> "SynthesisDraft":
        for values in (self.used_direct_pmids, self.used_indirect_pmids, self.used_context_pmids):
            if len(values) != len(set(values)):
                raise ValueError("used PMID lists must be unique")
        return self


class OpenAISynthesisClient:
    def __init__(self, api_key: SecretStr, config: dict[str, Any]) -> None:
        from openai import OpenAI  # type: ignore[import-not-found]

        self._client = OpenAI(
            api_key=api_key.get_secret_value(),
            timeout=config["request_timeout_seconds"],
            max_retries=config["max_attempts"] - 1,
        )
        self._config = config

    def create(self, prompt: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self._client.responses.parse(
                model=self._config["model_id"],
                reasoning={"effort": self._config["reasoning_effort"]},
                instructions=prompt,
                input=json.dumps(payload, ensure_ascii=False),
                text_format=SynthesisDraft,
            )
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            suffix = f", HTTP {status}" if isinstance(status, int) else ""
            raise RuntimeError(
                f"OpenAI synthesis request failed ({type(exc).__name__}{suffix})"
            ) from None
        if response.output_parsed is None:
            raise ValueError("OpenAI response contained no parsed synthesis")
        return response.output_parsed.model_dump(mode="json")


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or None


def _json_hash(value: Any) -> str:
    return sha256_text(json.dumps(value, sort_keys=True, ensure_ascii=False))


def _run_paths_hashes(required: dict[Path, list[str]]) -> dict[str, str]:
    return {
        str((root / name).resolve()): file_hash(root / name)
        for root, names in required.items()
        for name in names
    }


def _source_id_from_records(records: list[dict[str, Any]]) -> str:
    ids = {str(row.get("source_id")) for row in records}
    if len(ids) != 1 or "None" in ids:
        raise ValueError("source_id mismatch in synthesis inputs")
    return ids.pop()


def _by_id(records: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    return {str(row[key]): row for row in records}


def _dedupe_articles(pmids: list[str], articles: dict[str, dict[str, Any]]) -> list[str]:
    seen_pmids: set[str] = set()
    seen_dois: set[str] = set()
    result: list[str] = []
    for pmid in pmids:
        article = articles.get(str(pmid))
        if article is None:
            continue
        doi = (article.get("doi") or "").strip().casefold()
        if pmid in seen_pmids or (doi and doi in seen_dois):
            continue
        seen_pmids.add(pmid)
        if doi:
            seen_dois.add(doi)
        result.append(pmid)
    return result


def _article_public(article: dict[str, Any]) -> dict[str, Any]:
    return {
        "pmid": article["pmid"],
        "doi": article.get("doi"),
        "title": article.get("title"),
        "abstract": article.get("abstract"),
        "authors": article.get("authors", []),
        "journal": article.get("journal"),
        "publication_date": article.get("publication_date"),
        "publication_year": article.get("publication_year"),
        "publication_types": article.get("publication_types", []),
        "mesh_terms": article.get("mesh_terms", []),
    }


NO_ORIGINAL_COMMENT_TEXT = "Kein Originalkommentar im extrahierten Quellabschnitt vorhanden."


def _linked_comments(
    item: dict[str, Any], comments: list[dict[str, Any]], comments_by_id: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    linked: list[dict[str, Any]] = []
    seen: set[str] = set()
    for cid in item.get("linked_comment_ids", []):
        key = str(cid)
        if key in comments_by_id and key not in seen:
            linked.append(comments_by_id[key])
            seen.add(key)
    original_number = str(item.get("original_number") or "")
    if original_number:
        for comment in comments:
            cid = str(comment.get("comment_id") or "")
            if cid in seen:
                continue
            if str(comment.get("related_original_number") or "") == original_number:
                linked.append(comment)
                seen.add(cid)
    return linked


def _reference_lookup(references: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["original_reference_number"]): row for row in references}


def _reference_objects(
    numbers: list[str], references_by_number: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[str]]:
    objects, unresolved = [], []
    for number in numbers:
        if number in references_by_number:
            objects.append(references_by_number[number])
        else:
            unresolved.append(number)
    return objects, unresolved


def build_item_evidence_packets(
    *,
    formal_items: list[dict[str, Any]],
    comments: list[dict[str, Any]],
    references: list[dict[str, Any]],
    articles: list[dict[str, Any]],
    mappings: list[dict[str, Any]],
    evidence_index: list[dict[str, Any]],
    source_id: str,
) -> list[dict[str, Any]]:
    comments_by_id = _by_id(comments, "comment_id")
    articles_by_pmid = _by_id(articles, "pmid")
    refs_by_number = _reference_lookup(references)
    index_by_item = _by_id(evidence_index, "formal_item_id")
    mapping_by_item: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for row in mappings:
        decision = row["mapping_decision"]
        if decision in INFLUENTIAL_MAPPING_DECISIONS | CONTEXT_DECISIONS | UNCERTAIN_DECISIONS:
            mapping_by_item[row["formal_item_id"]][decision].append(str(row["pmid"]))
    packets = []
    for item in sorted(formal_items, key=lambda row: row.get("sequence_number") or 0):
        fid = item["formal_item_id"]
        grouped = mapping_by_item[fid]
        direct_pmids = _dedupe_articles(grouped["include_direct"], articles_by_pmid)
        indirect_pmids = _dedupe_articles(grouped["include_indirect"], articles_by_pmid)
        context_pmids = _dedupe_articles(grouped["context_only"], articles_by_pmid)
        uncertain_pmids = _dedupe_articles(grouped["uncertain_review_required"], articles_by_pmid)
        ref_objects, unresolved = _reference_objects(
            [str(x) for x in item.get("inline_reference_numbers", [])], refs_by_number
        )
        linked_comments = _linked_comments(item, comments, comments_by_id)
        packet = {
            "schema_version": PACKET_VERSION,
            "source_id": source_id,
            "formal_item_id": fid,
            "sequence_number": item.get("sequence_number"),
            "source_native_item_type": item.get("item_type_raw")
            or item.get("normalized_item_family"),
            "normalized_item_family": item.get("normalized_item_family"),
            "original_item_number": item.get("original_number"),
            "section_path": item.get("chapter_path_raw", []),
            "exact_original_item_text": item["exact_original_text"],
            "original_grade": item.get("recommendation_grade_raw"),
            "original_level_of_evidence": item.get("evidence_level_raw"),
            "original_consensus": item.get("consensus_raw"),
            "original_status": item.get("status_raw"),
            "linked_comment_ids": [comment["comment_id"] for comment in linked_comments],
            "exact_original_comments": [
                comment["exact_original_text"] for comment in linked_comments
            ],
            "original_comment_count": len(linked_comments),
            "original_inline_reference_ids": item.get("inline_reference_numbers", []),
            "original_reference_objects": ref_objects,
            "direct_article_pmids": direct_pmids,
            "indirect_article_pmids": indirect_pmids,
            "context_article_pmids": context_pmids,
            "uncertain_article_pmids": uncertain_pmids,
            "direct_articles": [_article_public(articles_by_pmid[p]) for p in direct_pmids],
            "indirect_articles": [_article_public(articles_by_pmid[p]) for p in indirect_pmids],
            "context_articles": [_article_public(articles_by_pmid[p]) for p in context_pmids],
            "unresolved_original_links": sorted(
                set(unresolved + [str(x) for x in item.get("unresolved_reference_numbers", [])])
            ),
            "review_required": bool(
                item.get("review_required")
                or unresolved
                or item.get("unresolved_reference_numbers")
                or index_by_item.get(fid, {}).get("review_required")
            ),
        }
        packets.append(packet)
    return packets


def build_item_evidence_digests(
    packets: list[dict[str, Any]], *, batch_size: int = 25
) -> list[dict[str, Any]]:
    digests = []
    for packet in packets:
        articles = [
            *packet["direct_articles"],
            *packet["indirect_articles"],
            *packet["context_articles"],
        ]
        if not articles:
            digests.append(
                {
                    "schema_version": DIGEST_VERSION,
                    "source_id": packet["source_id"],
                    "formal_item_id": packet["formal_item_id"],
                    "batch_index": 0,
                    "article_pmids": [],
                    "digest_de": (
                        "Keine neue direkte, indirekte oder kontextuelle Evidenz "
                        "in der PubMed-Zuordnung."
                    ),
                    "complete_article_coverage": True,
                }
            )
            continue
        for index, offset in enumerate(range(0, len(articles), batch_size)):
            batch = articles[offset : offset + batch_size]
            design_counts = Counter(
                "; ".join(a.get("publication_types", [])) or "unbekannter Publikationstyp"
                for a in batch
            )
            digests.append(
                {
                    "schema_version": DIGEST_VERSION,
                    "source_id": packet["source_id"],
                    "formal_item_id": packet["formal_item_id"],
                    "batch_index": index,
                    "article_pmids": [a["pmid"] for a in batch],
                    "digest_de": (
                        f"Batch {index + 1}: {len(batch)} Artikel; PubMed-Typen: "
                        + "; ".join(f"{k} ({v})" for k, v in sorted(design_counts.items()))
                    ),
                    "complete_article_coverage": True,
                }
            )
    return digests


def _fallback_synthesis(packet: dict[str, Any]) -> dict[str, Any]:
    direct = packet["direct_article_pmids"]
    indirect = packet["indirect_article_pmids"]
    context = packet["context_article_pmids"]
    if direct or indirect:
        evidence = []
        if direct:
            evidence.append(f"Direkte neue Evidenz liegt aus {len(direct)} PubMed-Artikeln vor.")
        if indirect:
            evidence.append(
                f"Indirekte neue Evidenz liegt aus {len(indirect)} PubMed-Artikeln vor."
            )
        decision = "rationale_updated"
        conclusion = (
            "Die neue Evidenz wird als Aktualisierung des wissenschaftlichen Hintergrunds "
            "dokumentiert; der formale Wortlaut bleibt unverändert."
        )
    elif context:
        evidence = (
            "Es liegt ausschließlich Hintergrundliteratur vor; sie wird nicht als Grundlage "
            "einer Textänderung verwendet."
        )
        decision = "insufficient_new_evidence"
        conclusion = "Keine ausreichende neue direkte oder indirekte Evidenz für eine Neubewertung."
    else:
        evidence = "Keine neue relevante direkte oder indirekte Evidenz in der PubMed-Zuordnung."
        decision = "insufficient_new_evidence"
        conclusion = "Der Originalwortlaut bleibt unverändert."
    return {
        "new_evidence_de": " ".join(evidence) if isinstance(evidence, list) else evidence,
        "conclusion_de": conclusion,
        "decision": decision,
        "updated_item_text_de": packet["exact_original_item_text"],
        "aisurgeon_evidence_class": _aisurgeon_evidence_class(packet),
        "used_direct_pmids": direct,
        "used_indirect_pmids": indirect,
        "used_context_pmids": [] if direct or indirect else context[:3],
        "uncertainty_de": None,
        "review_required": packet["review_required"],
        "review_notes": [],
    }


def _aisurgeon_evidence_class(packet: dict[str, Any]) -> str:
    articles = [*packet["direct_articles"], *packet["indirect_articles"]]
    if not articles:
        return "none"
    types = {
        str(value).casefold()
        for article in articles
        for value in article.get("publication_types", [])
    }
    if "meta-analysis" in types and "randomized controlled trial" in types:
        return "A"
    if "meta-analysis" in types or "systematic review" in types:
        return "B"
    return "C"


def validate_synthesis(raw: dict[str, Any], packet: dict[str, Any]) -> dict[str, Any]:
    draft = SynthesisDraft.model_validate(raw)
    allowed_direct = set(packet["direct_article_pmids"])
    allowed_indirect = set(packet["indirect_article_pmids"])
    allowed_context = set(packet["context_article_pmids"])
    if not set(draft.used_direct_pmids).issubset(allowed_direct):
        raise ValueError("Synthesis returned unknown direct PMID")
    if not set(draft.used_indirect_pmids).issubset(allowed_indirect):
        raise ValueError("Synthesis returned unknown indirect PMID")
    if not set(draft.used_context_pmids).issubset(allowed_context):
        raise ValueError("Synthesis returned unknown context PMID")
    if (
        draft.decision != "modified"
        and draft.updated_item_text_de != packet["exact_original_item_text"]
    ):
        raise ValueError("Unmodified decisions must preserve exact original item text")
    if draft.decision == "modified" and not (draft.used_direct_pmids or draft.used_indirect_pmids):
        raise ValueError("modified requires direct or indirect used evidence")
    if (
        draft.decision == "modified"
        and not draft.used_direct_pmids
        and not draft.used_indirect_pmids
        and draft.used_context_pmids
    ):
        raise ValueError("context_only cannot be the sole basis for modified")
    value = draft.model_dump(mode="json")
    if value["aisurgeon_evidence_class"] == "none":
        value["aisurgeon_evidence_class"] = _aisurgeon_evidence_class(packet)
    return value


def _client_payload(packet: dict[str, Any], digests: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "formal_item": {
            "formal_item_id": packet["formal_item_id"],
            "source_native_item_type": packet["source_native_item_type"],
            "original_item_number": packet["original_item_number"],
            "section_path": packet["section_path"],
            "exact_original_item_text": packet["exact_original_item_text"],
            "original_grade": packet["original_grade"],
            "original_level_of_evidence": packet["original_level_of_evidence"],
            "original_consensus": packet["original_consensus"],
            "original_status": packet["original_status"],
            "exact_original_comments": packet["exact_original_comments"],
            "original_references": packet["original_reference_objects"],
        },
        "direct_articles": packet["direct_articles"],
        "indirect_articles": packet["indirect_articles"],
        "context_articles": packet["context_articles"],
        "uncertain_article_pmids": packet["uncertain_article_pmids"],
        "batch_digests": digests,
        "rules": {
            "ignore_relevance_score": True,
            "decisions": sorted(SYNTHESIS_DECISIONS),
            "no_formal_grade": True,
        },
    }


def _source_native_label(packet: dict[str, Any], changed: bool) -> str:
    raw = str(
        packet.get("source_native_item_type") or packet.get("normalized_item_family") or "Item"
    )
    if changed:
        return f"Aktualisierungsvorschlag ({raw})"
    return raw


def build_updated_blocks(
    packets: list[dict[str, Any]], syntheses: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    blocks = []
    for packet in packets:
        synthesis = syntheses[packet["formal_item_id"]]
        blocks.append(
            {
                "schema_version": SYNTHESIS_SCHEMA_VERSION,
                "source_id": packet["source_id"],
                "formal_item_id": packet["formal_item_id"],
                "sequence_number": packet["sequence_number"],
                "original_item_number": packet["original_item_number"],
                "source_native_item_type": packet["source_native_item_type"],
                "normalized_item_family": packet["normalized_item_family"],
                "section_path": packet["section_path"],
                "exact_original_item_text": packet["exact_original_item_text"],
                "linked_comment_ids": packet["linked_comment_ids"],
                "exact_original_comments": packet["exact_original_comments"],
                "original_comment_count": packet["original_comment_count"],
                "new_evidence_de": synthesis["new_evidence_de"],
                "aisurgeon_evidence_class": synthesis["aisurgeon_evidence_class"],
                "conclusion_de": synthesis["conclusion_de"],
                "decision": synthesis["decision"],
                "updated_item_text_de": synthesis["updated_item_text_de"],
                "used_direct_pmids": synthesis["used_direct_pmids"],
                "used_indirect_pmids": synthesis["used_indirect_pmids"],
                "used_context_pmids": synthesis["used_context_pmids"],
                "old_reference_ids": packet["original_inline_reference_ids"],
                "new_reference_ids": [],
                "review_required": bool(packet["review_required"] or synthesis["review_required"]),
                "review_notes": synthesis["review_notes"],
                "original_grade": packet["original_grade"],
                "original_level_of_evidence": packet["original_level_of_evidence"],
                "original_consensus": packet["original_consensus"],
                "original_status": packet["original_status"],
                "display_item_label": _source_native_label(
                    packet, synthesis["decision"] == "modified"
                ),
            }
        )
    return blocks


def _load_reusable_syntheses(reuse_synthesis_run: Path | None) -> dict[str, dict[str, Any]]:
    if reuse_synthesis_run is None:
        return {}
    path = reuse_synthesis_run.resolve() / "updated_guideline_blocks.jsonl"
    if not path.is_file():
        raise ValueError("reuse_synthesis_run does not contain updated_guideline_blocks.jsonl")
    reusable: dict[str, dict[str, Any]] = {}
    for block in load_jsonl(path):
        reusable[str(block["formal_item_id"])] = {
            "new_evidence_de": block["new_evidence_de"],
            "conclusion_de": block["conclusion_de"],
            "decision": block["decision"],
            "updated_item_text_de": block["updated_item_text_de"],
            "aisurgeon_evidence_class": block["aisurgeon_evidence_class"],
            "used_direct_pmids": block.get("used_direct_pmids", []),
            "used_indirect_pmids": block.get("used_indirect_pmids", []),
            "used_context_pmids": block.get("used_context_pmids", []),
            "review_required": block.get("review_required", False),
            "review_notes": block.get("review_notes", []),
        }
    return reusable


def _normalize_title(title: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (title or "").casefold()).strip()


def _format_new_reference(article: dict[str, Any]) -> str:
    authors = article.get("authors", [])
    if authors:
        shown = ", ".join(str(a) for a in authors[:6])
        if len(authors) > 6:
            shown += ", et al"
    else:
        shown = "Autorenschaft nicht angegeben"
    parts = [
        f"{shown}.",
        str(article.get("title") or "[Ohne Titel]").rstrip(".") + ".",
    ]
    journal = article.get("journal")
    year = article.get("publication_year")
    if journal or year:
        parts.append(f"{journal or 'Zeitschrift nicht angegeben'} {year or 'o. J.'}.")
    if article.get("doi"):
        parts.append(f"doi: {article['doi']}.")
    parts.append(f"PMID: {article['pmid']}.")
    return " ".join(parts)


def consolidate_references(
    *,
    old_references: list[dict[str, Any]],
    articles: list[dict[str, Any]],
    blocks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    articles_by_pmid = _by_id(articles, "pmid")
    old_by_number = _reference_lookup(old_references)
    entries: list[dict[str, Any]] = []
    key_to_number: dict[tuple[str, str], int] = {}
    old_number_map: dict[str, int] = {}
    new_number_map: dict[str, int] = {}
    findings: list[dict[str, Any]] = []

    def add_entry(key: tuple[str, str], entry: dict[str, Any]) -> int:
        if key in key_to_number:
            return key_to_number[key]
        number = len(entries) + 1
        key_to_number[key] = number
        entries.append({"final_reference_number": number, **entry})
        return number

    for block in blocks:
        for old_id in block["old_reference_ids"]:
            old_id = str(old_id)
            if old_id not in old_by_number:
                findings.append(
                    {
                        "finding_id": f"REF_UNRESOLVED_{block['formal_item_id']}_{old_id}",
                        "source_id": block["source_id"],
                        "formal_item_id": block["formal_item_id"],
                        "issue_code": "unresolved_original_reference",
                        "message": (
                            f"Originaler Literaturverweis {old_id} konnte nicht aufgelöst werden."
                        ),
                    }
                )
                continue
            ref = old_by_number[old_id]
            number = add_entry(
                ("old", old_id),
                {
                    "source": "original",
                    "internal_reference_id": ref.get("reference_id") or f"OLD-{old_id}",
                    "original_reference_number": old_id,
                    "pmid": None,
                    "doi": None,
                    "normalized_title": None,
                    "full_citation": ref["exact_original_reference_text"],
                    "used_in_formal_item_ids": [block["formal_item_id"]],
                    "first_seen_in_formal_item_id": block["formal_item_id"],
                },
            )
            old_number_map[old_id] = number
        for pmid in [
            *block["used_direct_pmids"],
            *block["used_indirect_pmids"],
            *block["used_context_pmids"],
        ]:
            article = articles_by_pmid.get(str(pmid))
            if article is None:
                findings.append(
                    {
                        "finding_id": f"REF_UNKNOWN_PMID_{block['formal_item_id']}_{pmid}",
                        "source_id": block["source_id"],
                        "formal_item_id": block["formal_item_id"],
                        "issue_code": "used_pmid_missing_from_fetch",
                        "message": f"Verwendete PMID {pmid} fehlt im Fetch-Run.",
                    }
                )
                continue
            doi = (article.get("doi") or "").strip().casefold()
            title = _normalize_title(article.get("title"))
            key = ("pmid", str(pmid))
            if doi:
                key = ("doi", doi)
            elif title:
                key = ("title", title)
            number = add_entry(
                key,
                {
                    "source": "new_pubmed",
                    "internal_reference_id": f"NEW-PMID-{pmid}",
                    "original_reference_number": None,
                    "pmid": str(pmid),
                    "doi": article.get("doi"),
                    "normalized_title": title or None,
                    "full_citation": _format_new_reference(article),
                    "used_in_formal_item_ids": [block["formal_item_id"]],
                    "first_seen_in_formal_item_id": block["formal_item_id"],
                },
            )
            new_number_map[str(pmid)] = number
    for entry in entries:
        seen = []
        for block in blocks:
            old_hit = any(
                old_number_map.get(str(x)) == entry["final_reference_number"]
                for x in block["old_reference_ids"]
            )
            new_hit = any(
                new_number_map.get(str(x)) == entry["final_reference_number"]
                for x in [
                    *block["used_direct_pmids"],
                    *block["used_indirect_pmids"],
                    *block["used_context_pmids"],
                ]
            )
            if (old_hit or new_hit) and block["formal_item_id"] not in seen:
                seen.append(block["formal_item_id"])
        entry["used_in_formal_item_ids"] = seen
    number_map = {
        "old_reference_numbers": old_number_map,
        "new_pubmed_pmids": new_number_map,
    }
    for block in blocks:
        block["new_reference_ids"] = [
            str(new_number_map[str(pmid)])
            for pmid in [
                *block["used_direct_pmids"],
                *block["used_indirect_pmids"],
                *block["used_context_pmids"],
            ]
            if str(pmid) in new_number_map
        ]
    return entries, number_map, findings


def _citation(numbers: list[int]) -> str:
    values = sorted(set(numbers))
    if not values:
        return ""
    ranges: list[str] = []
    start = prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
            continue
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = value
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return "[" + ", ".join(ranges) + "]"


def _old_section_label(block: dict[str, Any]) -> str:
    raw = str(block.get("source_native_item_type") or block.get("normalized_item_family") or "Item")
    lowered = raw.casefold()
    if "statement" in lowered:
        return "Altes Statement"
    if "experten" in lowered or "expert" in lowered:
        return "Alter Expertenkonsens"
    if "konsens" in lowered and "empfehl" not in lowered:
        return "Alter Konsens"
    return "Alte Empfehlung"


def _new_section_label(block: dict[str, Any]) -> str:
    raw = str(block.get("source_native_item_type") or block.get("normalized_item_family") or "Item")
    lowered = raw.casefold()
    prefix = "Neue Empfehlung"
    if "statement" in lowered:
        prefix = "Neues Statement"
    elif "experten" in lowered or "expert" in lowered:
        prefix = "Neuer Expertenkonsens"
    elif "konsens" in lowered and "empfehl" not in lowered:
        prefix = "Neuer Konsens"
    if block["decision"] == "modified":
        return f"{prefix} / automatisierter Aktualisierungsvorschlag"
    return f"{prefix} / fortbestehend unverändert"


def _decision_label(decision: str) -> str:
    return {
        "insufficient_new_evidence": "unzureichende neue Evidenz",
        "unchanged": "unverändert",
        "rationale_updated": "Begründung aktualisiert",
        "modified": "geändert",
    }.get(decision, decision)


def _block_comments(block: dict[str, Any]) -> list[str]:
    comments = list(block.get("exact_original_comments") or [])
    return comments or [NO_ORIGINAL_COMMENT_TEXT]


def render_blocks_markdown(blocks: list[dict[str, Any]], number_map: dict[str, Any]) -> str:
    lines = ["# Aktualisierte Leitlinienblöcke", ""]
    for block in blocks:
        citation_numbers = [
            number_map["old_reference_numbers"].get(str(x))
            for x in block["old_reference_ids"]
            if number_map["old_reference_numbers"].get(str(x)) is not None
        ]
        citation_numbers += [int(x) for x in block["new_reference_ids"]]
        citation = _citation(citation_numbers)
        lines.extend(
            [
                f"## {block['original_item_number']} - {block['source_native_item_type']}",
                "",
                f"**{_old_section_label(block)}**",
                "",
                block["exact_original_item_text"],
                "",
                f"**{_new_section_label(block)}**",
                "",
                block["updated_item_text_de"] + (f" {citation}" if citation else ""),
                "",
                "**Alter Kommentar**",
                "",
                "\n\n".join(_block_comments(block)),
                "",
                f"**Neue Evidenz:** {block['new_evidence_de']}"
                + (f" {citation}" if citation else ""),
                "",
                f"**Schlussfolgerung:** {block['conclusion_de']}",
                "",
                f"**Entscheidung:** {block['decision']} - {_decision_label(block['decision'])}",
                "",
            ]
        )
    return "\n".join(lines)


def _w_p(
    text: str = "", *, style: str | None = None, bold: bool = False, jc: str | None = None
) -> str:
    props = []
    if style:
        props.append(f'<w:pStyle w:val="{style}"/>')
    if jc:
        props.append(f'<w:jc w:val="{jc}"/>')
    ppr = f"<w:pPr>{''.join(props)}</w:pPr>" if props else ""
    run_props = ['<w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:cs="Arial" w:eastAsia="Arial"/>']
    if bold:
        run_props.append("<w:b/>")
    rpr = f"<w:rPr>{''.join(run_props)}</w:rPr>"
    return f'<w:p>{ppr}<w:r>{rpr}<w:t xml:space="preserve">{escape(text)}</w:t></w:r></w:p>'


def _w_box(title: str, text: str, *, changed: bool = False) -> str:
    fill = "E2F0D9" if changed else "DDEBF7"
    border = "548235" if changed else "1F4E79"
    return (
        '<w:tbl><w:tblPr><w:tblW w:w="5000" w:type="pct"/>'
        "<w:tblBorders>"
        f'<w:top w:val="single" w:sz="12" w:color="{border}"/>'
        f'<w:left w:val="single" w:sz="12" w:color="{border}"/>'
        f'<w:bottom w:val="single" w:sz="12" w:color="{border}"/>'
        f'<w:right w:val="single" w:sz="12" w:color="{border}"/>'
        "</w:tblBorders></w:tblPr><w:tr><w:tc><w:tcPr>"
        f'<w:shd w:fill="{fill}"/><w:tcMar><w:top w:w="120" w:type="dxa"/>'
        '<w:left w:w="120" w:type="dxa"/><w:bottom w:w="120" w:type="dxa"/>'
        '<w:right w:w="120" w:type="dxa"/></w:tcMar></w:tcPr>'
        f"{_w_p(title, bold=True)}{_w_p(text, jc='left')}</w:tc></w:tr></w:tbl>"
    )


def _docx_xml(
    blocks: list[dict[str, Any]], refs: list[dict[str, Any]], summary: dict[str, Any]
) -> str:
    document_title = str(
        summary.get("document_title") or "AISurgeon Aktualisierte Leitlinie GERD/EoE 2026"
    )
    body = [
        _w_p(document_title, style="Title"),
        _w_p("Automatisch unterstützter wissenschaftlicher Aktualisierungsentwurf"),
        _w_p("Nicht als konsentierte AWMF-Leitlinie verwenden."),
        '<w:p><w:r><w:br w:type="page"/></w:r></w:p>',
        _w_p("Dokumentstatus und Methodik", style="Heading1"),
        _w_p(
            "Dieser Entwurf basiert auf kanonisch extrahierten FormalItems und einem "
            "PubMed-Zuordnungslauf. Die wissenschaftliche Bewertung ist automatisiert unterstützt "
            "und erfordert menschliche Validierung.",
            jc="both",
        ),
        _w_p("Inhaltsverzeichnis", style="Heading1"),
        '<w:p><w:r><w:fldChar w:fldCharType="begin"/></w:r><w:r><w:instrText>'
        'TOC \\o "1-3" \\h \\z \\u</w:instrText></w:r><w:r><w:fldChar w:fldCharType="end"/>'
        "</w:r></w:p>",
        '<w:p><w:r><w:br w:type="page"/></w:r></w:p>',
        _w_p("Zusammenfassung des Aktualisierungslaufs", style="Heading1"),
        _w_p(
            f"FormalItems: {summary['formal_items']}; geändert: {summary['modified']}; "
            f"Begründung aktualisiert: {summary['rationale_updated']}; unverändert: "
            f"{summary['unchanged']}; unzureichende neue Evidenz: "
            f"{summary['insufficient_new_evidence']}.",
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
        meta = " | ".join(
            value
            for value in [
                f"Empfehlungsgrad: {block['original_grade']}"
                if block.get("original_grade")
                else "",
                (
                    f"Level of Evidence: {block['original_level_of_evidence']}"
                    if block.get("original_level_of_evidence")
                    else ""
                ),
                f"Konsens: {block['original_consensus']}"
                if block.get("original_consensus")
                else "",
                f"Status: {block['original_status']}" if block.get("original_status") else "",
            ]
            if value
        )
        if block["decision"] == "modified":
            body.append(_w_box(_old_section_label(block), block["exact_original_item_text"]))
            body.append(
                _w_box(
                    _new_section_label(block),
                    block["updated_item_text_de"],
                    changed=True,
                )
            )
        else:
            body.append(_w_box(_old_section_label(block), block["exact_original_item_text"]))
            body.append(_w_box(_new_section_label(block), block["updated_item_text_de"]))
        if meta:
            body.append(_w_p(meta))
        body.append(_w_p("Alter Kommentar", style="Heading3"))
        for comment in _block_comments(block):
            body.append(_w_p(comment, jc="both"))
        body.append(_w_p("Neue Evidenz", style="Heading3"))
        body.append(_w_p(block["new_evidence_de"], jc="both"))
        body.append(_w_p("Schlussfolgerung", style="Heading3"))
        body.append(_w_p(block["conclusion_de"], jc="both"))
        body.append(
            _w_p(
                f"Entscheidung: {block['decision']} - {_decision_label(block['decision'])}",
                bold=True,
            )
        )
    body.extend(
        [
            '<w:p><w:r><w:br w:type="page"/></w:r></w:p>',
            _w_p("Konsolidiertes Literaturverzeichnis", style="Heading1"),
        ]
    )
    for ref in refs:
        body.append(
            '<w:p><w:pPr><w:ind w:left="720" w:hanging="360"/><w:jc w:val="left"/></w:pPr>'
            f'<w:r><w:t xml:space="preserve">{ref["final_reference_number"]}. '
            f"{escape(ref['full_citation'])}</w:t></w:r></w:p>"
        )
    body.extend(
        [
            '<w:p><w:r><w:br w:type="page"/></w:r></w:p>',
            _w_p("Anhang: Review Findings und technische Metadaten", style="Heading1"),
            _w_p(
                "Review Findings sind in den begleitenden JSONL- und XLSX-Dateien "
                "vollständig dokumentiert."
            ),
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


def _styles_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        '<w:docDefaults><w:rPrDefault><w:rPr>'
        '<w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:cs="Arial" w:eastAsia="Arial"/>'
        '<w:sz w:val="24"/></w:rPr></w:rPrDefault></w:docDefaults>'
        '<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/>'
        '<w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:cs="Arial" w:eastAsia="Arial"/>'
        '<w:sz w:val="24"/></w:rPr>'
        '<w:pPr><w:spacing w:after="120" w:line="360" w:lineRule="auto"/>'
        '<w:jc w:val="both"/></w:pPr></w:style>'
        '<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/>'
        '<w:rPr><w:b/><w:sz w:val="32"/>'
        '<w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:cs="Arial" w:eastAsia="Arial"/>'
        '</w:rPr></w:style>'
        '<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/>'
        '<w:basedOn w:val="Normal"/><w:uiPriority w:val="9"/><w:qFormat/>'
        '<w:rPr><w:b/><w:color w:val="1F4E79"/><w:sz w:val="32"/>'
        '<w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:cs="Arial" w:eastAsia="Arial"/>'
        '</w:rPr></w:style>'
        '<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/>'
        '<w:basedOn w:val="Normal"/><w:qFormat/><w:rPr><w:b/><w:color w:val="1F4E79"/>'
        '<w:sz w:val="28"/>'
        '<w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:cs="Arial" w:eastAsia="Arial"/>'
        '</w:rPr></w:style>'
        '<w:style w:type="paragraph" w:styleId="Heading3"><w:name w:val="heading 3"/>'
        '<w:basedOn w:val="Normal"/><w:qFormat/><w:rPr><w:b/><w:sz w:val="24"/>'
        '<w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:cs="Arial" w:eastAsia="Arial"/>'
        '</w:rPr></w:style>'
        "</w:styles>"
    )


def _font_table_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:fonts xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        '<w:font w:name="Arial"><w:panose1 w:val="020B0604020202020204"/>'
        '<w:charset w:val="00"/><w:family w:val="swiss"/>'
        '<w:pitch w:val="variable"/></w:font></w:fonts>'
    )


def write_docx(
    path: Path, blocks: list[dict[str, Any]], refs: list[dict[str, Any]], summary: dict[str, Any]
) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pkg_rel = "http://schemas.openxmlformats.org/package/2006/relationships"
    office_rel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    pkg_ct = "http://schemas.openxmlformats.org/package/2006/content-types"
    app_ct = "application/vnd.openxmlformats-officedocument.wordprocessingml"
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<Relationships xmlns="{pkg_rel}">'
        f'<Relationship Id="rId1" Type="{office_rel}/officeDocument" '
        'Target="word/document.xml"/>'
        "</Relationships>"
    )
    doc_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<Relationships xmlns="{pkg_rel}">'
        f'<Relationship Id="rIdHeader1" Type="{office_rel}/header" Target="header1.xml"/>'
        f'<Relationship Id="rIdFooter1" Type="{office_rel}/footer" Target="footer1.xml"/>'
        f'<Relationship Id="rIdFontTable" Type="{office_rel}/fontTable" Target="fontTable.xml"/>'
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
        f'<Override PartName="/word/fontTable.xml" ContentType="{app_ct}.fontTable+xml"/>'
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
    document_title = escape(
        str(summary.get("document_title") or "AISurgeon Aktualisierte Leitlinie GERD/EoE 2026")
    )
    core = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<cp:coreProperties '
        'xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/">'
        f"<dc:title>{document_title}</dc:title>"
        "<dc:creator>AISurgeon Living Guideline Platform</dc:creator>"
        "</cp:coreProperties>"
    )
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", rels)
        archive.writestr("word/_rels/document.xml.rels", doc_rels)
        archive.writestr("word/document.xml", _docx_xml(blocks, refs, summary))
        archive.writestr("word/styles.xml", _styles_xml())
        archive.writestr("word/fontTable.xml", _font_table_xml())
        archive.writestr("word/header1.xml", header)
        archive.writestr("word/footer1.xml", footer)
        archive.writestr("docProps/core.xml", core)


def _write_xlsx(path: Path, sheet_name: str, rows: list[dict[str, Any]]) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = sheet_name[:31]
    headers = sorted({key for row in rows for key in row}) or ["finding_id"]
    sheet.append(headers)
    for row in rows:
        sheet.append(
            [
                json.dumps(row.get(h), ensure_ascii=False)
                if isinstance(row.get(h), (list, dict))
                else row.get(h)
                for h in headers
            ]
        )
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = f"A1:{sheet.cell(1, len(headers)).coordinate}"
    workbook.save(path)


def run_docx_qa(docx_path: Path, run_dir: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "docx_path": str(docx_path),
        "structural_valid": False,
        "render_attempted": False,
        "render_successful": False,
        "critical_layout_errors": [],
        "warnings": [],
        "qa_pdf": None,
        "page_images": [],
    }
    with zipfile.ZipFile(docx_path) as archive:
        names = set(archive.namelist())
        required = {
            "[Content_Types].xml",
            "word/document.xml",
            "word/styles.xml",
            "word/fontTable.xml",
            "word/header1.xml",
            "word/footer1.xml",
        }
        missing = sorted(required - names)
        if missing:
            report["critical_layout_errors"].append(f"Missing DOCX parts: {missing}")
        document_xml = archive.read("word/document.xml").decode("utf-8")
        if "Heading1" not in document_xml or "TOC" not in document_xml:
            report["critical_layout_errors"].append("Heading styles or TOC field missing")
        for label in [
            "Alte Empfehlung",
            "Neue Empfehlung",
            "Alter Kommentar",
            "Neue Evidenz",
            "Schlussfolgerung",
        ]:
            if label not in document_xml:
                report["critical_layout_errors"].append(f"Required visible label missing: {label}")
        for part in [
            "word/document.xml",
            "word/styles.xml",
            "word/fontTable.xml",
            "word/header1.xml",
            "word/footer1.xml",
        ]:
            if part in names and "Calibri" in archive.read(part).decode("utf-8"):
                report["critical_layout_errors"].append(f"Calibri present in {part}")
        if any(term in document_xml for term in PUBLIC_FORBIDDEN_TERMS):
            report["critical_layout_errors"].append("Forbidden public-term marker present")
        report["structural_valid"] = not report["critical_layout_errors"]
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


def build_updated_guideline(
    *,
    extraction_run: Path,
    search_run: Path,
    fetch_run: Path,
    mapping_run: Path,
    output_root: Path,
    worker_id: str,
    api_key: SecretStr,
    resume_run: Path | None = None,
    limit: int | None = None,
    technical_limited_document: bool = False,
    reuse_synthesis_run: Path | None = None,
    client_factory: Callable[[SecretStr, dict[str, Any]], Any] = OpenAISynthesisClient,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> Path:
    required = {
        extraction_run: [
            "formal_items.jsonl",
            "comments.jsonl",
            "references.jsonl",
            "document_map.validated.json",
            "extraction_manifest.json",
        ],
        search_run: [
            "search_units.jsonl",
            "formal_item_search_coverage.jsonl",
            "search_manifest.json",
        ],
        fetch_run: ["pubmed_articles.jsonl", "pubmed_fetch_manifest.json"],
        mapping_run: [
            "article_formal_item_mappings.jsonl",
            "formal_item_evidence_index.jsonl",
            "mapping_review_findings.jsonl",
            "mapping_manifest.json",
        ],
    }
    paths = _run_paths_hashes(required)
    if not all(Path(path).is_file() for path in paths):
        raise ValueError("Synthesis input runs are incomplete")
    formal = load_jsonl(extraction_run / "formal_items.jsonl")
    comments = load_jsonl(extraction_run / "comments.jsonl")
    references = load_jsonl(extraction_run / "references.jsonl")
    articles = load_jsonl(fetch_run / "pubmed_articles.jsonl")
    mappings = load_jsonl(mapping_run / "article_formal_item_mappings.jsonl")
    evidence_index = load_jsonl(mapping_run / "formal_item_evidence_index.jsonl")
    mapping_findings = load_jsonl(mapping_run / "mapping_review_findings.jsonl")
    manifests = [
        json.loads((root / name).read_text(encoding="utf-8"))
        for root, name in [
            (extraction_run, "extraction_manifest.json"),
            (search_run, "search_manifest.json"),
            (fetch_run, "pubmed_fetch_manifest.json"),
            (mapping_run, "mapping_manifest.json"),
        ]
    ]
    allowed_statuses = {"completed", "completed_with_review"}
    if technical_limited_document:
        allowed_statuses.add("technical_limited")
    if any(m.get("status") not in allowed_statuses for m in manifests):
        raise ValueError("Input manifest is not completed")
    limited_input = any(m.get("run_mode") == "technical_limited" for m in manifests)
    if limited_input and not technical_limited_document:
        raise ValueError("Limited upstream runs require technical_limited_document=True")
    source_id = _source_id_from_records(formal)
    if any(str(m.get("source_id")) != source_id for m in manifests):
        raise ValueError("source_id mismatch between synthesis inputs")
    config = json.loads(MODEL_CONFIG_PATH.read_text(encoding="utf-8"))
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    fingerprint = {
        "source_id": source_id,
        "input_runs": [
            str(extraction_run.resolve()),
            str(search_run.resolve()),
            str(fetch_run.resolve()),
            str(mapping_run.resolve()),
        ],
        "input_file_hashes": paths,
        "model_id": config["model_id"],
        "reasoning_effort": config["reasoning_effort"],
        "model_configuration": config,
        "prompt_version": config["prompt_version"],
        "prompt_hash": sha256_text(prompt),
        "synthesis_schema_version": SYNTHESIS_SCHEMA_VERSION,
        "reference_builder_version": REFERENCE_BUILDER_VERSION,
        "docx_renderer_version": DOCX_RENDERER_VERSION,
        "git_commit": _git_commit(),
        "worker_id": worker_id,
        "limit": limit,
        "technical_limited_document": technical_limited_document,
        "reuse_synthesis_run": str(reuse_synthesis_run.resolve()) if reuse_synthesis_run else None,
    }
    if resume_run:
        run_dir = resume_run.resolve()
        if json.loads((run_dir / "checkpoint_fingerprint.json").read_text()) != fingerprint:
            raise ValueError("Resume fingerprint does not match")
        if (run_dir / "synthesis_manifest.json").is_file():
            return run_dir
    else:
        root = ensure_external_run_root(output_root, extraction_run)
        run_dir = root / (
            f"synthesis-{now():%Y%m%dT%H%M%S%fZ}-{source_id}-{fingerprint['prompt_hash'][:8]}"
        )
        run_dir.mkdir(parents=True, exist_ok=False)
        write_json(run_dir / "checkpoint_fingerprint.json", fingerprint)
    packets = build_item_evidence_packets(
        formal_items=formal,
        comments=comments,
        references=references,
        articles=articles,
        mappings=mappings,
        evidence_index=evidence_index,
        source_id=source_id,
    )
    selected_packets = packets[:limit] if limit is not None else packets
    digests = build_item_evidence_digests(selected_packets)
    digests_by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for digest in digests:
        digests_by_item[digest["formal_item_id"]].append(digest)
    raw_dir, checkpoint_dir = run_dir / "raw_model_responses", run_dir / "checkpoints"
    raw_dir.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)
    fingerprint_hash = _json_hash(fingerprint)
    syntheses: dict[str, dict[str, Any]] = {}
    reusable_syntheses = _load_reusable_syntheses(reuse_synthesis_run)
    client: Any | None = None
    for packet in selected_packets:
        fid = packet["formal_item_id"]
        checkpoint = checkpoint_dir / f"{fid.replace('/', '_')}.json"
        if checkpoint.is_file():
            saved = json.loads(checkpoint.read_text(encoding="utf-8"))
            if saved.get("fingerprint_hash") != fingerprint_hash:
                raise ValueError("Checkpoint fingerprint is incompatible")
            syntheses[fid] = saved["synthesis"]
            continue
        if fid in reusable_syntheses:
            synthesis = validate_synthesis(reusable_syntheses[fid], packet)
        elif not (packet["direct_article_pmids"] or packet["indirect_article_pmids"]):
            synthesis = _fallback_synthesis(packet)
        else:
            synthesis = None
            if client is None:
                client = client_factory(api_key, config)
            payload = _client_payload(packet, digests_by_item[fid])
            for attempt in range(1, int(config["max_attempts"]) + 1):
                try:
                    raw = client.create(prompt, payload)
                    write_json(raw_dir / f"{fid}.attempt-{attempt}.json", raw)
                    synthesis = validate_synthesis(raw, packet)
                    break
                except (RuntimeError, ValueError) as exc:
                    write_json(
                        raw_dir / f"{fid}.attempt-{attempt}.error.json",
                        {
                            "formal_item_id": fid,
                            "attempt": attempt,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        },
                    )
            if synthesis is None:
                raise RuntimeError(f"Synthesis item {fid} failed after controlled attempts")
        write_json(
            checkpoint,
            {"status": "completed", "fingerprint_hash": fingerprint_hash, "synthesis": synthesis},
        )
        syntheses[fid] = synthesis
    blocks = build_updated_blocks(selected_packets, syntheses)
    refs, number_map, reference_findings = consolidate_references(
        old_references=references, articles=articles, blocks=blocks
    )
    markdown = render_blocks_markdown(blocks, number_map)
    synthesis_findings = [
        {
            "finding_id": f"SYN_REVIEW_{block['formal_item_id']}",
            "source_id": source_id,
            "formal_item_id": block["formal_item_id"],
            "issue_code": "synthesis_review_required",
            "message": "; ".join(block["review_notes"])
            or "Automatisierte Synthese verlangt Review.",
        }
        for block in blocks
        if block["review_required"]
    ]
    synthesis_findings.extend(mapping_findings)
    decision_counts = Counter(block["decision"] for block in blocks)
    summary = {
        "source_id": source_id,
        "document_title": (
            "LIMITED TEST OUTPUT - AISurgeon NET technical limited pilot subset"
            if technical_limited_document
            else "AISurgeon Aktualisierte Leitlinie GERD/EoE 2026"
        ),
        "formal_items": len(formal),
        "processed_formal_items": len(blocks),
        "insufficient_new_evidence": decision_counts["insufficient_new_evidence"],
        "unchanged": decision_counts["unchanged"],
        "rationale_updated": decision_counts["rationale_updated"],
        "modified": decision_counts["modified"],
        "review_findings": len(synthesis_findings) + len(reference_findings),
        "references": len(refs),
    }
    write_jsonl(run_dir / "item_evidence_packets.jsonl", selected_packets)
    write_jsonl(run_dir / "item_evidence_digests.jsonl", digests)
    write_jsonl(run_dir / "updated_guideline_blocks.jsonl", blocks)
    (run_dir / "updated_guideline_blocks.md").write_text(markdown + "\n", encoding="utf-8")
    write_jsonl(run_dir / "consolidated_references.jsonl", refs)
    write_json(run_dir / "reference_number_map.json", number_map)
    write_jsonl(run_dir / "synthesis_review_findings.jsonl", synthesis_findings)
    _write_xlsx(run_dir / "synthesis_review_findings.xlsx", "synthesis_review", synthesis_findings)
    write_jsonl(run_dir / "reference_review_findings.jsonl", reference_findings)
    _write_xlsx(run_dir / "reference_review_findings.xlsx", "reference_review", reference_findings)
    docx_name = (
        "AISurgeon_LIMITED_TEST_OUTPUT_NET_subset_comments_arial_fixed.docx"
        if technical_limited_document
        else "AISurgeon_Aktualisierte_Leitlinie_GERD_EoE_2026.docx"
    )
    docx_path = run_dir / docx_name
    if limit is None or technical_limited_document:
        write_docx(docx_path, blocks, refs, summary)
        qa = run_docx_qa(docx_path, run_dir)
    else:
        qa = {
            "docx_path": None,
            "structural_valid": False,
            "render_attempted": False,
            "render_successful": False,
            "critical_layout_errors": [],
            "warnings": ["Technical limited run: final DOCX not generated."],
        }
    write_json(run_dir / "docx_qa_report.json", qa)
    write_json(run_dir / "synthesis_summary.json", summary)
    status = (
        "technical_limited"
        if limit is not None or technical_limited_document or limited_input
        else (
            "completed_with_review"
            if synthesis_findings or reference_findings or qa.get("warnings")
            else "completed"
        )
    )
    write_json(
        run_dir / "synthesis_manifest.json",
        {
            **fingerprint,
            "created_at": now().isoformat(),
            "status": status,
            "run_mode": (
                "technical_limited"
                if limit is not None or technical_limited_document or limited_input
                else "complete"
            ),
            "technical_limited_document": technical_limited_document,
            "summary": summary,
            "credential_status": {"OPENAI_API_KEY": "set"},
            "output_files": {
                p.name: file_hash(p)
                for p in run_dir.iterdir()
                if p.is_file() and p.name != "synthesis_manifest.json"
            },
        },
    )
    return run_dir
