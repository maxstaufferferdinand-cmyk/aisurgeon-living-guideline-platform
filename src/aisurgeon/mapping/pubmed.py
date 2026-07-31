"""Reproducible candidate generation and GPT abstract screening."""

import json
import subprocess
import time
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from openpyxl import Workbook
from pydantic import SecretStr

from aisurgeon.extraction.canonical.outputs import write_json, write_jsonl
from aisurgeon.mapping.pubmed_models import (
    CANDIDATE_GENERATION_VERSION,
    MAPPING_SCHEMA_VERSION,
    ScreeningBatchDraft,
)
from aisurgeon.search.pubmed.generation import ensure_external_run_root, file_hash, load_jsonl
from aisurgeon.search.pubmed.query import sha256_text

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_CONFIG_PATH = PROJECT_ROOT / "config/models/openai_pubmed_mapping_v1.json"
PROMPT_PATH = PROJECT_ROOT / "config/prompts/openai_pubmed_mapping_v1.txt"
INCLUDED = {"include_direct", "include_indirect", "context_only", "uncertain_review_required"}
PRIMARY_ELIGIBLE_DESIGNS = {
    "randomized_controlled_trial",
    "meta_analysis_randomized",
    "meta_analysis_nonrandomized_or_mixed",
    "systematic_review",
}
ELIGIBILITY_FILTER_VERSION = "pubmed_primary_design_eligibility_v1"


def classify_study_design(article: dict[str, Any]) -> str:
    """Classify eligibility conservatively from deterministic PubMed metadata."""
    publication_types = {
        str(value).strip().casefold() for value in article.get("publication_types", [])
    }
    if "meta-analysis" in publication_types:
        if "randomized controlled trial" in publication_types:
            return "meta_analysis_randomized"
        return "meta_analysis_nonrandomized_or_mixed"
    if "systematic review" in publication_types:
        return "systematic_review"
    if "randomized controlled trial" in publication_types:
        return "randomized_controlled_trial"
    if "review" in publication_types:
        return "narrative_review"
    known = (
        ("registry study", "registry_study"),
        ("comparative study", "comparative_study"),
        ("observational study", "other"),
        ("clinical trial", "controlled_clinical_trial"),
        ("evaluation study", "evaluation_study"),
        ("validation study", "validation_study"),
    )
    return next((design for label, design in known if label in publication_types), "unclear")


def deterministic_eligibility_decision(
    pair: dict[str, Any], article: dict[str, Any], *, retain_narrative_reviews: bool
) -> dict[str, Any] | None:
    design = classify_study_design(article)
    if design in PRIMARY_ELIGIBLE_DESIGNS:
        return None
    narrative_context = design == "narrative_review" and retain_narrative_reviews
    decision = "context_only" if narrative_context else "exclude_wrong_study_design"
    reason = (
        "Narrative review retained deterministically as context only; it is not primary evidence."
        if narrative_context
        else (
            "Excluded before GPT mapping: PubMed metadata does not identify an eligible "
            "randomized controlled trial, meta-analysis, or systematic review."
        )
    )
    return {
        "schema_version": MAPPING_SCHEMA_VERSION,
        "source_id": pair["source_id"],
        "candidate_pair_id": pair["candidate_pair_id"],
        "pmid": pair["pmid"],
        "formal_item_id": pair["formal_item_id"],
        "mapping_decision": decision,
        "relevance_score": 0,
        "directness": "contextual" if narrative_context else "not_relevant",
        "population_match": "unclear",
        "intervention_or_exposure_match": "unclear",
        "comparator_match": "unclear",
        "outcome_match": "unclear",
        "setting_match": "unclear",
        "study_design_normalized": design,
        "publication_type_interpretation": "; ".join(article.get("publication_types", []))
        or "No eligible PubMed publication type",
        "concise_mapping_reason": reason,
        "supporting_abstract_passage": None,
        "uncertainty_reason": None,
        "review_required": False,
        "screening_method": "deterministic_pre_mapping_eligibility",
    }


def candidate_pair_id(source_id: str, pmid: str, formal_item_id: str) -> str:
    identity = chr(31).join((source_id, pmid, formal_item_id))
    return f"{source_id}_PAIR_{sha256_text(identity)[:16]}"


def build_candidate_pairs(
    formal_items: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    hits: list[dict[str, Any]],
    articles: list[dict[str, Any]],
    source_id: str,
) -> list[dict[str, Any]]:
    formal = {str(x["formal_item_id"]): x for x in formal_items}
    query_by_id = {str(x["query_id"]): x for x in queries}
    article_ids = {str(x["pmid"]) for x in articles}
    paths: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    for hit in hits:
        pmid, query_id = str(hit["pmid"]), str(hit["query_id"])
        if pmid not in article_ids:
            continue
        if query_id not in query_by_id:
            raise ValueError(f"Unknown query_id in query hits: {query_id}")
        query = query_by_id[query_id]
        unit_id = str(query["search_unit_id"])
        for item_id in query["linked_formal_item_ids"]:
            if item_id not in formal:
                raise ValueError(f"Unknown formal_item_id in query provenance: {item_id}")
            paths[(pmid, item_id)].add((unit_id, query_id))
    records = []
    for (pmid, item_id), links in sorted(
        paths.items(), key=lambda x: (formal[x[0][1]].get("sequence_number", 0), int(x[0][0]))
    ):
        item = formal[item_id]
        records.append(
            {
                "schema_version": MAPPING_SCHEMA_VERSION,
                "source_id": source_id,
                "candidate_pair_id": candidate_pair_id(source_id, pmid, item_id),
                "pmid": pmid,
                "formal_item_id": item_id,
                "formal_item_sequence_number": item.get("sequence_number"),
                "formal_item_family": item.get("normalized_item_family") or "other_formal_item",
                "original_item_number": item.get("original_number"),
                "linked_search_unit_ids": sorted({x[0] for x in links}),
                "linked_query_ids": sorted({x[1] for x in links}),
                "provenance_paths": [
                    {"pmid": pmid, "query_id": q, "search_unit_id": u, "formal_item_id": item_id}
                    for u, q in sorted(links)
                ],
                "candidate_generation_version": CANDIDATE_GENERATION_VERSION,
            }
        )
    return records


class OpenAIMappingClient:
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
                text_format=ScreeningBatchDraft,
            )
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            suffix = f", HTTP {status}" if isinstance(status, int) else ""
            raise RuntimeError(
                f"OpenAI mapping request failed ({type(exc).__name__}{suffix})"
            ) from None
        if response.output_parsed is None:
            raise ValueError("OpenAI response contained no parsed mapping batch")
        return response.output_parsed.model_dump(mode="json")


def _validate_batch(
    raw: dict[str, Any], pairs: list[dict[str, Any]], articles: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    draft = ScreeningBatchDraft.model_validate(raw)
    expected = {p["pmid"] for p in pairs}
    returned = [d.pmid for d in draft.decisions]
    unknown = set(returned) - expected
    if unknown:
        raise ValueError(f"Unknown PMID returned by model: {sorted(unknown)}")
    pair_by_pmid = {p["pmid"]: p for p in pairs}
    results = []
    seen_pmids = set()
    for decision in draft.decisions:
        if decision.pmid in seen_pmids:
            continue
        seen_pmids.add(decision.pmid)
        pair, abstract = pair_by_pmid[decision.pmid], articles[decision.pmid].get("abstract") or ""
        quote = decision.supporting_abstract_passage
        if quote and quote not in abstract:
            if decision.mapping_decision in {"include_direct", "include_indirect"}:
                decision.supporting_abstract_passage = None
                decision.review_required = True
                reason = (
                    "Model supplied a non-verbatim supporting abstract passage; "
                    "mapping decision preserved with quote removed for review."
                )
                decision.uncertainty_reason = (
                    f"{decision.uncertainty_reason} {reason}"
                    if decision.uncertainty_reason
                    else reason
                )
            else:
                decision.supporting_abstract_passage = None
        value = decision.model_dump(mode="json")
        results.append(
            {
                "schema_version": MAPPING_SCHEMA_VERSION,
                "source_id": pair["source_id"],
                "candidate_pair_id": pair["candidate_pair_id"],
                "pmid": decision.pmid,
                "formal_item_id": pair["formal_item_id"],
                "screening_method": "gpt_abstract_mapping",
                **{k: v for k, v in value.items() if k != "pmid"},
            }
        )
    for missing_pmid in sorted(expected - seen_pmids):
        pair = pair_by_pmid[missing_pmid]
        article = articles[missing_pmid]
        results.append(
            {
                "schema_version": MAPPING_SCHEMA_VERSION,
                "source_id": pair["source_id"],
                "candidate_pair_id": pair["candidate_pair_id"],
                "pmid": missing_pmid,
                "formal_item_id": pair["formal_item_id"],
                "screening_method": "deterministic_missing_model_decision_fallback",
                "mapping_decision": "uncertain_review_required",
                "relevance_score": 0,
                "directness": "unclear",
                "population_match": "unclear",
                "intervention_or_exposure_match": "unclear",
                "comparator_match": "unclear",
                "outcome_match": "unclear",
                "setting_match": "unclear",
                "study_design_normalized": classify_study_design(article),
                "publication_type_interpretation": "; ".join(article.get("publication_types", []))
                or "No eligible PubMed publication type",
                "concise_mapping_reason": (
                    "OpenAI mapping response omitted this expected PMID after returning a "
                    "schema-valid batch; retained as uncertain review-required evidence."
                ),
                "supporting_abstract_passage": None,
                "uncertainty_reason": "Model omitted a required PMID-level mapping decision.",
                "review_required": True,
            }
        )
    return results


def _write_review_xlsx(path: Path, findings: list[dict[str, Any]]) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "mapping_review_findings"
    headers = sorted({key for row in findings for key in row}) or ["finding_id"]
    sheet.append(headers)
    for row in findings:
        sheet.append([row.get(key) for key in headers])
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = f"A1:{sheet.cell(1, len(headers)).coordinate}"
    workbook.save(path)


def map_pubmed_evidence(
    *,
    extraction_run: Path,
    search_run: Path,
    fetch_run: Path,
    output_root: Path,
    worker_id: str,
    api_key: SecretStr,
    resume_run: Path | None = None,
    batch_size: int = 10,
    mapping_concurrency: int = 1,
    limit: int | None = None,
    retain_narrative_reviews: bool = False,
    client_factory: Callable[[SecretStr, dict[str, Any]], Any] = OpenAIMappingClient,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    sleep: Callable[[float], None] = time.sleep,
) -> Path:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if mapping_concurrency < 1:
        raise ValueError("mapping_concurrency must be positive")
    required = {
        extraction_run: [
            "formal_items.jsonl",
            "comments.jsonl",
            "document_map.validated.json",
            "extraction_manifest.json",
        ],
        search_run: [
            "search_units.jsonl",
            "formal_item_search_coverage.jsonl",
            "pubmed_queries.jsonl",
            "search_manifest.json",
        ],
        fetch_run: [
            "pubmed_articles.jsonl",
            "pubmed_query_hits.jsonl",
            "pubmed_pmids.jsonl",
            "pubmed_fetch_manifest.json",
        ],
    }
    paths = {
        str(root.resolve() / name): root.resolve() / name
        for root, names in required.items()
        for name in names
    }
    if not all(p.is_file() for p in paths.values()):
        raise ValueError("Mapping input runs are incomplete")
    formal = load_jsonl(extraction_run / "formal_items.jsonl")
    comments = load_jsonl(extraction_run / "comments.jsonl")
    queries = load_jsonl(search_run / "pubmed_queries.jsonl")
    hits = load_jsonl(fetch_run / "pubmed_query_hits.jsonl")
    articles_list = load_jsonl(fetch_run / "pubmed_articles.jsonl")
    articles = {str(x["pmid"]): x for x in articles_list}
    manifests = [
        json.loads((root / name).read_text())
        for root, name in [
            (extraction_run, "extraction_manifest.json"),
            (search_run, "search_manifest.json"),
            (fetch_run, "pubmed_fetch_manifest.json"),
        ]
    ]
    allowed = {"completed", "completed_with_review", "technical_limited"}
    if any(m.get("status") not in allowed for m in manifests):
        raise ValueError("Input manifest is not completed")
    limited_input = any(m.get("run_mode") == "technical_limited" for m in manifests)
    source_ids = {str(x.get("source_id")) for x in formal} | {
        str(x.get("source_id")) for x in queries
    }
    source_ids.update(str(m.get("source_id")) for m in manifests)
    if len(source_ids) != 1 or "None" in source_ids:
        raise ValueError("source_id mismatch between mapping inputs")
    source_id = source_ids.pop()
    config = json.loads(MODEL_CONFIG_PATH.read_text())
    prompt = PROMPT_PATH.read_text()
    fingerprint = {
        "source_id": source_id,
        "input_runs": [
            str(extraction_run.resolve()),
            str(search_run.resolve()),
            str(fetch_run.resolve()),
        ],
        "input_file_hashes": {str(p): file_hash(p) for p in paths.values()},
        "model_id": config["model_id"],
        "reasoning_effort": config["reasoning_effort"],
        "model_configuration": config,
        "prompt_version": config["prompt_version"],
        "prompt_hash": sha256_text(prompt),
        "mapping_schema_version": MAPPING_SCHEMA_VERSION,
        "candidate_generation_version": CANDIDATE_GENERATION_VERSION,
        "batch_size": batch_size,
        "limit": limit,
        "eligibility_filter_version": ELIGIBILITY_FILTER_VERSION,
        "primary_eligible_designs": sorted(PRIMARY_ELIGIBLE_DESIGNS),
        "retain_narrative_reviews": retain_narrative_reviews,
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        or None,
    }
    if resume_run:
        run_dir = resume_run.resolve()
        if json.loads((run_dir / "checkpoint_fingerprint.json").read_text()) != fingerprint:
            raise ValueError("Resume fingerprint does not match")
        if (run_dir / "mapping_manifest.json").is_file():
            return run_dir
    else:
        root = ensure_external_run_root(output_root, extraction_run)
        run_dir = (
            root / f"mapping-{now():%Y%m%dT%H%M%S%fZ}-{source_id}-{fingerprint['prompt_hash'][:8]}"
        )
        run_dir.mkdir(parents=True, exist_ok=False)
        write_json(run_dir / "checkpoint_fingerprint.json", fingerprint)
    raw_dir, checkpoint_dir = run_dir / "raw_model_responses", run_dir / "checkpoints"
    raw_dir.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)
    pairs = build_candidate_pairs(formal, queries, hits, articles_list, source_id)
    selected = pairs[:limit] if limit is not None else pairs
    by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
    screening = []
    for pair in selected:
        deterministic = deterministic_eligibility_decision(
            pair,
            articles[pair["pmid"]],
            retain_narrative_reviews=retain_narrative_reviews,
        )
        if deterministic is None:
            by_item[pair["formal_item_id"]].append(pair)
        else:
            screening.append(deterministic)
    comments_by_id = {x.get("comment_id"): x for x in comments}
    formal_by_id = {x["formal_item_id"]: x for x in formal}
    fingerprint_hash = sha256_text(json.dumps(fingerprint, sort_keys=True))
    unresolved_batches = []

    def screen_batch(task: dict[str, Any]) -> list[dict[str, Any]]:
        key = task["key"]
        batch = task["batch"]
        payload = task["payload"]
        decisions = None
        client = client_factory(api_key, config)
        for attempt in range(1, int(config["max_attempts"]) + 1):
            raw_path = raw_dir / f"{key}.attempt-{attempt}.json"
            if raw_path.exists():
                try:
                    decisions = _validate_batch(json.loads(raw_path.read_text()), batch, articles)
                    break
                except (RuntimeError, ValueError) as exc:
                    error_path = raw_dir / f"{key}.attempt-{attempt}.error.json"
                    if not error_path.exists():
                        write_json(
                            error_path,
                            {
                                "batch_key": key,
                                "attempt": attempt,
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            },
                        )
                    continue
            try:
                raw = client.create(prompt, payload)
                if raw_path.exists():
                    raw = json.loads(raw_path.read_text())
                else:
                    try:
                        write_json(raw_path, raw)
                    except FileExistsError:
                        raw = json.loads(raw_path.read_text())
                decisions = _validate_batch(raw, batch, articles)
                break
            except (RuntimeError, ValueError) as exc:
                error_path = raw_dir / f"{key}.attempt-{attempt}.error.json"
                if not error_path.exists():
                    write_json(
                        error_path,
                        {
                            "batch_key": key,
                            "attempt": attempt,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        },
                    )
                if "HTTP 429" in str(exc) or "RateLimitError" in str(exc):
                    sleep(min(60.0 * attempt, 300.0))
        if decisions is None:
            raise RuntimeError(f"Mapping batch {key} failed after controlled attempts")
        checkpoint = checkpoint_dir / f"{key}.json"
        if not checkpoint.exists():
            write_json(
                checkpoint,
                {
                    "status": "completed",
                    "fingerprint_hash": fingerprint_hash,
                    "decisions": decisions,
                },
            )
        return decisions

    for item_id, item_pairs in by_item.items():
        item = formal_by_id[item_id]
        for offset in range(0, len(item_pairs), batch_size):
            batch = item_pairs[offset : offset + batch_size]
            key = f"{item_id.replace('/', '_')}-{offset // batch_size:05d}"
            checkpoint = checkpoint_dir / f"{key}.json"
            if checkpoint.is_file():
                saved = json.loads(checkpoint.read_text())
                if saved.get("fingerprint_hash") != fingerprint_hash:
                    raise ValueError("Checkpoint fingerprint is incompatible")
                screening.extend(saved["decisions"])
                continue
            payload = {
                "formal_item": {
                    "exact_original_text": item["exact_original_text"],
                    "item_type": item.get("item_type_raw") or item.get("normalized_item_family"),
                    "original_number": item.get("original_number"),
                    "chapter_context": item.get("chapter_path_raw", []),
                },
                "comments": [
                    comments_by_id[x]
                    for x in item.get("linked_comment_ids", [])
                    if x in comments_by_id
                ],
                "articles": [
                    {
                        k: articles[p["pmid"]].get(k)
                        for k in (
                            "pmid",
                            "title",
                            "abstract",
                            "publication_types",
                            "mesh_terms",
                            "publication_year",
                            "journal",
                            "doi",
                        )
                    }
                    for p in batch
                ],
            }
            unresolved_batches.append({"key": key, "batch": batch, "payload": payload})
    if mapping_concurrency == 1:
        for task in unresolved_batches:
            screening.extend(screen_batch(task))
    else:
        with ThreadPoolExecutor(max_workers=mapping_concurrency) as executor:
            futures = {
                executor.submit(screen_batch, task): task["key"] for task in unresolved_batches
            }
            for future in as_completed(futures):
                screening.extend(future.result())
    if len(screening) != len(selected) or len({x["candidate_pair_id"] for x in screening}) != len(
        selected
    ):
        raise ValueError("Every candidate pair must have exactly one final decision")
    decision_by_pair = {x["candidate_pair_id"]: x for x in screening}
    coverage_in = {
        x["formal_item_id"]: x for x in load_jsonl(search_run / "formal_item_search_coverage.jsonl")
    }
    evidence_index = []
    coverage = []
    findings = []
    for item in formal:
        fid = item["formal_item_id"]
        all_p = [p for p in pairs if p["formal_item_id"] == fid]
        done = [
            decision_by_pair[p["candidate_pair_id"]]
            for p in all_p
            if p["candidate_pair_id"] in decision_by_pair
        ]
        groups = {d: [x["pmid"] for x in done if x["mapping_decision"] == d] for d in INCLUDED}
        complete = len(done) == len(all_p) and limit is None
        evidence_index.append(
            {
                "schema_version": MAPPING_SCHEMA_VERSION,
                "source_id": source_id,
                "formal_item_id": fid,
                "sequence_number": item.get("sequence_number"),
                "original_item_number": item.get("original_number"),
                "normalized_item_family": item.get("normalized_item_family"),
                "candidate_article_count": len(all_p),
                "screened_article_count": len(done),
                "direct_article_pmids": groups["include_direct"],
                "indirect_article_pmids": groups["include_indirect"],
                "context_article_pmids": groups["context_only"],
                "uncertain_article_pmids": groups["uncertain_review_required"],
                "excluded_article_count": sum(
                    x["mapping_decision"].startswith("exclude_") for x in done
                ),
                "primary_eligible_article_count": sum(
                    x["screening_method"] == "gpt_abstract_mapping" for x in done
                ),
                "pre_mapping_excluded_article_count": sum(
                    x["screening_method"] == "deterministic_pre_mapping_eligibility"
                    and x["mapping_decision"] == "exclude_wrong_study_design"
                    for x in done
                ),
                "mapping_complete": complete,
                "review_required": any(x["review_required"] for x in done),
            }
        )
        search_cov = coverage_in.get(fid)
        if search_cov is None:
            raise ValueError(f"FormalItem missing from search coverage: {fid}")
        relevant = any(x["mapping_decision"] in INCLUDED for x in done)
        coverage.append(
            {
                "schema_version": MAPPING_SCHEMA_VERSION,
                "source_id": source_id,
                "formal_item_id": fid,
                "search_relevance": search_cov["search_relevance"],
                "linked_search_unit_ids": search_cov["linked_search_unit_ids"],
                "candidate_count": len(all_p),
                "screened_count": len(done),
                "relevant_evidence_found": relevant,
                "no_relevant_evidence_found": complete and not relevant,
                "mapping_complete": complete,
                "review_required": any(x["review_required"] for x in done) or not complete,
            }
        )
        for x in done:
            if x["review_required"]:
                findings.append(
                    {
                        "finding_id": f"MAP_REVIEW_{x['candidate_pair_id']}",
                        "source_id": source_id,
                        "formal_item_id": fid,
                        "pmid": x["pmid"],
                        "issue_code": "mapping_review_required",
                        "message": x.get("uncertainty_reason") or x["concise_mapping_reason"],
                    }
                )
    status = (
        "technical_limited"
        if limit is not None or limited_input
        else (
            "completed_with_review"
            if findings or any(not x["mapping_complete"] for x in coverage)
            else "completed"
        )
    )
    write_jsonl(run_dir / "candidate_pairs.jsonl", pairs)
    write_jsonl(run_dir / "article_screening.jsonl", screening)
    write_jsonl(
        run_dir / "article_formal_item_mappings.jsonl",
        [x for x in screening if x["mapping_decision"] in INCLUDED],
    )
    write_jsonl(run_dir / "formal_item_evidence_index.jsonl", evidence_index)
    write_jsonl(run_dir / "mapping_coverage.jsonl", coverage)
    write_jsonl(run_dir / "mapping_review_findings.jsonl", findings)
    _write_review_xlsx(run_dir / "mapping_review_findings.xlsx", findings)
    summary = {
        "source_id": source_id,
        "status": status,
        "formal_items": len(formal),
        "candidate_pairs": len(pairs),
        "screened_pairs": len(screening),
        "included_mappings": sum(x["mapping_decision"] in INCLUDED for x in screening),
        "review_findings": len(findings),
    }
    write_json(run_dir / "mapping_summary.json", summary)
    write_json(
        run_dir / "mapping_manifest.json",
        {
            **fingerprint,
            "worker_id": worker_id,
            "created_at": now().isoformat(),
            "status": status,
            "run_mode": "technical_limited" if limit is not None or limited_input else "complete",
            "limited_input_accepted": limited_input,
            "summary": summary,
            "credential_status": {"OPENAI_API_KEY": "set"},
            "output_files": {
                p.name: file_hash(p)
                for p in run_dir.iterdir()
                if p.is_file() and p.name != "mapping_manifest.json"
            },
        },
    )
    return run_dir
