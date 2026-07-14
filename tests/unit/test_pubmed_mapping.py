import json
from pathlib import Path

import pytest
from pydantic import SecretStr

from aisurgeon.mapping.pubmed import (
    _validate_batch,
    build_candidate_pairs,
    candidate_pair_id,
    classify_study_design,
    deterministic_eligibility_decision,
    map_pubmed_evidence,
)
from aisurgeon.orchestration.pubmed_mapping import run_to_mapping


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _jsonl(path: Path, values: list[dict]) -> None:
    path.write_text("".join(json.dumps(x) + "\n" for x in values), encoding="utf-8")


def _decision(pmid: str, *, quote: str | None = "Useful") -> dict:
    return {
        "pmid": pmid,
        "mapping_decision": "include_direct",
        "relevance_score": 90,
        "directness": "direct",
        "population_match": "match",
        "intervention_or_exposure_match": "match",
        "comparator_match": "not_applicable",
        "outcome_match": "match",
        "setting_match": "match",
        "study_design_normalized": "randomized_controlled_trial",
        "publication_type_interpretation": "RCT",
        "concise_mapping_reason": "Directly relevant",
        "supporting_abstract_passage": quote,
        "uncertainty_reason": None,
        "review_required": False,
    }


def test_candidate_provenance_deduplicates_and_preserves_many_to_many() -> None:
    formal = [
        {"formal_item_id": "F1", "sequence_number": 1, "normalized_item_family": "recommendation"},
        {"formal_item_id": "F2", "sequence_number": 2, "normalized_item_family": "statement"},
    ]
    queries = [
        {"query_id": "Q1", "search_unit_id": "U1", "linked_formal_item_ids": ["F1", "F2"]},
        {"query_id": "Q2", "search_unit_id": "U2", "linked_formal_item_ids": ["F1"]},
    ]
    hits = [
        {"pmid": "1", "query_id": "Q1"},
        {"pmid": "1", "query_id": "Q1"},
        {"pmid": "1", "query_id": "Q2"},
    ]
    pairs = build_candidate_pairs(formal, queries, hits, [{"pmid": "1"}], "SRC")
    assert len(pairs) == 2
    assert pairs[0]["linked_query_ids"] == ["Q1", "Q2"]
    assert {x["formal_item_id"] for x in pairs} == {"F1", "F2"}
    assert candidate_pair_id("SRC", "1", "F1") == candidate_pair_id("SRC", "1", "F1")


def test_model_ids_and_abstract_quotes_are_strictly_validated() -> None:
    pair = {"source_id": "SRC", "candidate_pair_id": "P1", "pmid": "1", "formal_item_id": "F1"}
    articles = {"1": {"abstract": "Useful exact passage."}}
    assert (
        _validate_batch({"decisions": [_decision("1")]}, [pair], articles)[0]["formal_item_id"]
        == "F1"
    )
    with pytest.raises(ValueError, match="Unknown PMID"):
        _validate_batch({"decisions": [_decision("2")]}, [pair], articles)
    with pytest.raises(ValueError, match="not verbatim"):
        _validate_batch({"decisions": [_decision("1", quote="invented")]}, [pair], articles)


@pytest.mark.parametrize(
    ("publication_type", "expected"),
    [
        ("Randomized Controlled Trial", "randomized_controlled_trial"),
        ("Meta-Analysis", "meta_analysis_nonrandomized_or_mixed"),
        ("Systematic Review", "systematic_review"),
        ("Registry Study", "registry_study"),
        ("Comparative Study", "comparative_study"),
        ("Review", "narrative_review"),
    ],
)
def test_deterministic_study_design_classification(publication_type: str, expected: str) -> None:
    assert classify_study_design({"publication_types": [publication_type]}) == expected


def test_narrative_review_is_optional_context_not_primary() -> None:
    pair = {
        "source_id": "SRC",
        "candidate_pair_id": "P1",
        "pmid": "1",
        "formal_item_id": "F1",
    }
    article = {"publication_types": ["Review"]}
    excluded = deterministic_eligibility_decision(pair, article, retain_narrative_reviews=False)
    context = deterministic_eligibility_decision(pair, article, retain_narrative_reviews=True)
    assert excluded and excluded["mapping_decision"] == "exclude_wrong_study_design"
    assert context and context["mapping_decision"] == "context_only"


def _runs(tmp_path: Path) -> tuple[Path, Path, Path]:
    extraction, search, fetch = (tmp_path / x for x in ("extraction", "search", "fetch"))
    for path in (extraction, search, fetch):
        path.mkdir()
    formal = []
    for n, family in enumerate(
        ("recommendation", "statement", "consensus_statement", "expert_consensus"), 1
    ):
        formal.append(
            {
                "source_id": "SRC",
                "formal_item_id": f"F{n}",
                "sequence_number": n,
                "original_number": str(n),
                "normalized_item_family": family,
                "exact_original_text": f"Text {n}",
                "chapter_path_raw": ["Chapter"],
                "linked_comment_ids": [],
            }
        )
    _jsonl(extraction / "formal_items.jsonl", formal)
    _jsonl(extraction / "comments.jsonl", [])
    _json(extraction / "document_map.validated.json", {})
    _json(extraction / "extraction_manifest.json", {"source_id": "SRC", "status": "completed"})
    units = []
    queries = []
    coverage = []
    hits = []
    for n in range(1, 5):
        units.append({"search_unit_id": f"U{n}"})
        queries.append(
            {
                "source_id": "SRC",
                "query_id": f"Q{n}",
                "search_unit_id": f"U{n}",
                "linked_formal_item_ids": [f"F{n}"],
            }
        )
        coverage.append(
            {
                "formal_item_id": f"F{n}",
                "search_relevance": "search_relevant",
                "linked_search_unit_ids": [f"U{n}"],
            }
        )
        hits.append({"pmid": str(n), "query_id": f"Q{n}", "search_unit_id": f"U{n}"})
    _jsonl(search / "search_units.jsonl", units)
    _jsonl(search / "formal_item_search_coverage.jsonl", coverage)
    _jsonl(search / "pubmed_queries.jsonl", queries)
    _json(search / "search_manifest.json", {"source_id": "SRC", "status": "completed"})
    articles = [
        {
            "pmid": str(n),
            "title": f"T{n}",
            "abstract": "Useful exact passage.",
            "publication_types": ["Randomized Controlled Trial"],
            "mesh_terms": [],
        }
        for n in range(1, 5)
    ]
    _jsonl(fetch / "pubmed_articles.jsonl", articles)
    _jsonl(fetch / "pubmed_query_hits.jsonl", hits)
    _jsonl(fetch / "pubmed_pmids.jsonl", [{"pmid": str(n)} for n in range(1, 5)])
    _json(fetch / "pubmed_fetch_manifest.json", {"source_id": "SRC", "status": "completed"})
    return extraction, search, fetch


def test_complete_mapping_outputs_every_item_and_pair_and_limit_is_technical(
    tmp_path: Path,
) -> None:
    extraction, search, fetch = _runs(tmp_path)

    class Fake:
        def create(self, prompt, payload):
            return {"decisions": [_decision(x["pmid"]) for x in payload["articles"]]}

    kwargs = dict(
        extraction_run=extraction,
        search_run=search,
        fetch_run=fetch,
        output_root=tmp_path,
        worker_id="w",
        api_key=SecretStr("secret"),
        client_factory=lambda key, config: Fake(),
    )
    run = map_pubmed_evidence(**kwargs)
    assert len((run / "article_screening.jsonl").read_text().splitlines()) == 4
    assert len((run / "formal_item_evidence_index.jsonl").read_text().splitlines()) == 4
    assert len((run / "mapping_coverage.jsonl").read_text().splitlines()) == 4
    assert "secret" not in "".join(
        p.read_text(errors="ignore") for p in run.rglob("*") if p.is_file()
    )
    assert map_pubmed_evidence(**kwargs, resume_run=run) == run
    with pytest.raises(ValueError, match="fingerprint"):
        map_pubmed_evidence(**kwargs, resume_run=run, batch_size=2)
    limited = map_pubmed_evidence(**kwargs, limit=1)
    assert (
        json.loads((limited / "mapping_manifest.json").read_text())["status"] == "technical_limited"
    )
    assert not all(
        json.loads(x)["mapping_complete"]
        for x in (limited / "mapping_coverage.jsonl").read_text().splitlines()
    )


def test_invalid_structured_response_is_retried_at_most_controlled_attempts(
    tmp_path: Path,
) -> None:
    extraction, search, fetch = _runs(tmp_path)
    calls = 0

    class Fake:
        def create(self, prompt, payload):
            nonlocal calls
            calls += 1
            pmids = [x["pmid"] for x in payload["articles"]]
            if calls == 1:
                return {"decisions": [_decision("999")]}
            return {"decisions": [_decision(pmid) for pmid in pmids]}

    run = map_pubmed_evidence(
        extraction_run=extraction,
        search_run=search,
        fetch_run=fetch,
        output_root=tmp_path,
        worker_id="w",
        api_key=SecretStr("secret"),
        client_factory=lambda key, config: Fake(),
    )
    assert calls == 5
    assert list((run / "raw_model_responses").glob("*.error.json"))


def test_pre_mapping_filter_excludes_ineligible_designs_before_gpt(tmp_path: Path) -> None:
    extraction, search, fetch = _runs(tmp_path)
    publication_types = {
        "1": "Randomized Controlled Trial",
        "2": "Meta-Analysis",
        "3": "Systematic Review",
        "4": "Observational Study",
        "5": "Registry Study",
        "6": "Comparative Study",
        "7": "Review",
    }
    articles = [
        {
            "pmid": pmid,
            "title": f"T{pmid}",
            "abstract": "Useful exact passage.",
            "publication_types": [publication_type],
            "mesh_terms": [],
        }
        for pmid, publication_type in publication_types.items()
    ]
    hits = [{"pmid": pmid, "query_id": "Q1", "search_unit_id": "U1"} for pmid in publication_types]
    _jsonl(fetch / "pubmed_articles.jsonl", articles)
    _jsonl(fetch / "pubmed_query_hits.jsonl", hits)
    _jsonl(fetch / "pubmed_pmids.jsonl", [{"pmid": pmid} for pmid in publication_types])
    received: list[str] = []

    class Fake:
        def create(self, prompt, payload):
            received.extend(x["pmid"] for x in payload["articles"])
            return {"decisions": [_decision(x["pmid"]) for x in payload["articles"]]}

    run = map_pubmed_evidence(
        extraction_run=extraction,
        search_run=search,
        fetch_run=fetch,
        output_root=tmp_path,
        worker_id="w",
        api_key=SecretStr("secret"),
        client_factory=lambda key, config: Fake(),
    )
    assert received == ["1", "2", "3"]
    screening = [
        json.loads(line) for line in (run / "article_screening.jsonl").read_text().splitlines()
    ]
    assert len(screening) == 7
    excluded = [x for x in screening if x["mapping_decision"] == "exclude_wrong_study_design"]
    assert {x["pmid"] for x in excluded} == {"4", "5", "6", "7"}
    assert all(x["screening_method"] == "deterministic_pre_mapping_eligibility" for x in excluded)
    index = [
        json.loads(line)
        for line in (run / "formal_item_evidence_index.jsonl").read_text().splitlines()
    ]
    first = next(x for x in index if x["formal_item_id"] == "F1")
    assert first["primary_eligible_article_count"] == 3
    assert first["pre_mapping_excluded_article_count"] == 4


def test_orchestrator_passes_paths_and_resume_reuses_completed_children(tmp_path: Path) -> None:
    extraction = tmp_path / "extraction"
    extraction.mkdir()
    _json(extraction / "extraction_manifest.json", {"status": "completed"})
    calls = []

    def search_runner(**kwargs):
        calls.append(("search", kwargs["input_run"]))
        p = tmp_path / "s"
        p.mkdir(exist_ok=True)
        return p

    def fetch_runner(**kwargs):
        calls.append(("fetch", kwargs["input_run"]))
        p = tmp_path / "f"
        p.mkdir(exist_ok=True)
        return p

    def mapping_runner(**kwargs):
        calls.append(("mapping", kwargs["search_run"], kwargs["fetch_run"]))
        p = tmp_path / "m"
        p.mkdir(exist_ok=True)
        _json(p / "mapping_manifest.json", {"status": "completed_with_review"})
        return p

    kwargs = dict(
        extraction_run=extraction,
        output_root=tmp_path,
        worker_id="w",
        openai_api_key=SecretStr("x"),
        ncbi_email=SecretStr("e"),
        ncbi_api_key=None,
        ncbi_tool="t",
        start_date=__import__("datetime").date(2023, 1, 1),
        end_date=__import__("datetime").date(2026, 7, 14),
        search_runner=search_runner,
        fetch_runner=fetch_runner,
        mapping_runner=mapping_runner,
    )
    run = run_to_mapping(**kwargs)
    assert calls == [
        ("search", extraction),
        ("fetch", tmp_path / "s"),
        ("mapping", tmp_path / "s", tmp_path / "f"),
    ]
    assert run_to_mapping(**kwargs, resume_run=run) == run and len(calls) == 3
    assert (
        json.loads((run / "orchestration_manifest.json").read_text())["status"]
        == "completed_with_review"
    )
