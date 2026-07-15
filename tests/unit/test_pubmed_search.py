import json
from datetime import date
from pathlib import Path

import pytest
from pydantic import SecretStr

from aisurgeon.search.pubmed import ncbi
from aisurgeon.search.pubmed.generation import generate_searches, normalize_search_plan
from aisurgeon.search.pubmed.models import SearchPlanDraft, SearchUnitDraft
from aisurgeon.search.pubmed.ncbi import (
    ESearchQueryError,
    HttpResponse,
    NcbiClient,
    fetch_pubmed,
    parse_pubmed_xml,
)
from aisurgeon.search.pubmed.query import (
    EVIDENCE_TYPE_FILTER,
    build_query,
    validate_final_pubmed_query,
    validate_query_core,
)


def query_record(query_id: str, unit_id: str, formal_id: str, core: str) -> dict:
    date_filter = '("2023/01/01"[Date - Publication] : "2026/07/14"[Date - Publication])'
    humans = "NOT (animals[mh] NOT humans[mh])"
    evidence = (
        '("Randomized Controlled Trial"[pt] OR "Meta-Analysis"[pt] OR "Systematic Review"[pt])'
    )
    exclusion = 'NOT ("Practice Guideline"[pt] OR "Guideline"[pt])'
    return {
        "schema_version": "pubmed_query_v1",
        "source_id": "SRC",
        "query_id": query_id,
        "search_unit_id": unit_id,
        "linked_formal_item_ids": [formal_id],
        "query_core": core,
        "date_filter": date_filter,
        "humans_filter": humans,
        "evidence_type_filter": evidence,
        "exclusion_filter": exclusion,
        "final_pubmed_query": f"({core} AND {date_filter} AND {evidence}) {humans} {exclusion}",
        "start_date": "2023-01-01",
        "end_date": "2026-07-14",
        "query_version": "pubmed_query_builder_v4",
        "prompt_version": "v1",
        "prompt_hash": "h",
        "model_id": "gpt-5.5",
        "model_configuration": {},
        "model_configuration_hash": "m",
        "review_required": False,
        "review_notes": [],
    }


def item(number: str, family: str) -> dict:
    return {
        "formal_item_id": f"SRC_{number}",
        "original_number": number,
        "normalized_item_family": family,
        "exact_original_text": f"Original {number}",
    }


def draft(ids: list[str], *, relevance="search_relevant") -> SearchUnitDraft:
    return SearchUnitDraft(
        section_path=["Therapie"],
        topic_de="Thema",
        topic_en="Topic",
        linked_formal_item_ids=ids,
        search_relevance=relevance,
        exclusion_reason="Administrative item" if relevance != "search_relevant" else None,
        clinical_question="Question",
        query_core='("GERD"[Title/Abstract] OR "Gastroesophageal Reflux"[Mesh])',
    )


def test_all_formal_families_are_equal_and_units_can_group_items() -> None:
    records = [
        item("1", "recommendation"),
        item("2", "statement"),
        item("3", "consensus_statement"),
        item("4", "expert_consensus"),
    ]
    units, coverage = normalize_search_plan(
        SearchPlanDraft(search_units=[draft([r["formal_item_id"] for r in records])]),
        records,
        "SRC",
    )
    assert units[0].linked_formal_item_families == [r["normalized_item_family"] for r in records]
    assert len(coverage) == 4 and all(row.linked_search_unit_ids for row in coverage)


def test_coverage_hard_fails_and_explicit_exclusion_is_retained() -> None:
    records = [item("1", "statement"), item("2", "other_formal_item")]
    with pytest.raises(ValueError, match="coverage incomplete"):
        normalize_search_plan(SearchPlanDraft(search_units=[draft(["SRC_1"])]), records, "SRC")
    units, coverage = normalize_search_plan(
        SearchPlanDraft(
            search_units=[draft(["SRC_1"]), draft(["SRC_2"], relevance="not_search_relevant")]
        ),
        records,
        "SRC",
    )
    assert units[1].exclusion_reason and coverage[1].search_relevance == "not_search_relevant"


def test_ids_are_deterministic_and_exact_text_is_from_canonical_input() -> None:
    records = [item("1", "statement")]
    plan = SearchPlanDraft(search_units=[draft(["SRC_1"])])
    first = normalize_search_plan(plan, records, "SRC")[0][0]
    second = normalize_search_plan(plan, records, "SRC")[0][0]
    assert first.search_unit_id == second.search_unit_id
    assert first.exact_formal_item_texts == ["Original 1"]


def test_python_adds_technical_filters() -> None:
    unit = normalize_search_plan(
        SearchPlanDraft(search_units=[draft(["SRC_1"])]), [item("1", "recommendation")], "SRC"
    )[0][0]
    query = build_query(
        unit,
        start_date=date(2023, 1, 1),
        end_date=date(2026, 7, 14),
        prompt_version="v1",
        prompt_hash="h",
        model_config={"model_id": "gpt-5.5"},
        model_config_hash="m",
    )
    assert "2023/01/01" in query.date_filter
    assert "animals[mh] NOT humans[mh]" in query.humans_filter
    assert "Systematic Review" in query.evidence_type_filter
    assert "Observational Study" not in query.evidence_type_filter
    assert "Comparative Study" not in query.evidence_type_filter
    assert "Practice Guideline" in query.exclusion_filter
    expected = (
        f"({unit.query_core} AND {query.date_filter} AND {EVIDENCE_TYPE_FILTER}) "
        "NOT (animals[mh] NOT humans[mh]) "
        'NOT ("Practice Guideline"[pt] OR "Guideline"[pt])'
    )
    assert query.final_pubmed_query == expected
    assert "AND NOT (" not in query.final_pubmed_query


@pytest.mark.parametrize("value", ["(GERD OR reflux", "()", "AND GERD", "GERD OR AND reflux"])
def test_boolean_validation(value: str) -> None:
    assert validate_query_core(value)


def test_esearch_paginates_retries_and_never_places_key_in_cache(tmp_path: Path) -> None:
    calls = []

    def transport(url, params, timeout):
        calls.append(dict(params))
        if len(calls) == 1:
            return HttpResponse(429, b"", {"Retry-After": "0"})
        start = int(params["retstart"])
        ids = [str(n) for n in range(start + 1, min(3, start + 2) + 1)] if start < 3 else []
        return HttpResponse(
            200,
            json.dumps(
                {"esearchresult": {"count": "3", "idlist": ids, "querytranslation": "GERD"}}
            ).encode(),
        )

    client = NcbiClient(
        email=SecretStr("owner@example.test"),
        api_key=SecretStr("top-secret-key"),
        tool="aisurgeon-tests",
        cache_dir=tmp_path,
        transport=transport,
        sleep=lambda _: None,
    )
    result = client.esearch("GERD", page_size=2)
    assert result["pmids"] == ["1", "2", "3"]
    assert result["count"] == 3
    assert "top-secret-key" not in "".join(path.name for path in tmp_path.iterdir())
    assert len(calls) == 3


def test_permanent_error_is_not_retried(tmp_path: Path) -> None:
    calls = 0

    def transport(*args):
        nonlocal calls
        calls += 1
        return HttpResponse(400, b"bad")

    client = NcbiClient(
        email=SecretStr("x@y.test"),
        api_key=None,
        tool="test",
        cache_dir=tmp_path,
        transport=transport,
        sleep=lambda _: None,
    )
    with pytest.raises(RuntimeError, match="permanent"):
        client.esearch("GERD")
    assert calls == 1


def test_invalid_live_ncbi_animals_translation_and_not_warning_are_rejected(
    tmp_path: Path,
) -> None:
    translation = 'GERD AND ("animals"[MeSH Terms] NOT "humans"[MeSH Terms])'
    assert "animals_exclusion_used_as_positive_filter" in validate_final_pubmed_query(translation)

    def transport(*args):
        return HttpResponse(
            200,
            json.dumps(
                {
                    "esearchresult": {
                        "count": "0",
                        "idlist": [],
                        "querytranslation": translation,
                        "warninglist": {"outputmessage": ["NOT"]},
                        "errorlist": {},
                    }
                }
            ).encode(),
        )

    client = NcbiClient(
        email=SecretStr("x@y.test"),
        api_key=None,
        tool="test",
        cache_dir=tmp_path,
        transport=transport,
        sleep=lambda _: None,
    )
    with pytest.raises(ESearchQueryError, match="query syntax problem") as exc_info:
        client.esearch("GERD")
    assert exc_info.value.metadata["warninglist"] == {"outputmessage": ["NOT"]}
    assert exc_info.value.metadata["querytranslation"] == translation


def test_esearch_phrase_not_found_warning_is_completed_with_review(tmp_path: Path) -> None:
    def transport(*args):
        return HttpResponse(
            200,
            json.dumps(
                {
                    "esearchresult": {
                        "count": "2",
                        "idlist": ["10", "11"],
                        "querytranslation": "GERD",
                        "warninglist": {"quotedphrasesnotfound": ['"GERD chest pain"[tiab]']},
                        "errorlist": {},
                    }
                }
            ).encode(),
        )

    client = NcbiClient(
        email=SecretStr("x@y.test"),
        api_key=None,
        tool="test",
        cache_dir=tmp_path,
        transport=transport,
        sleep=lambda _: None,
    )
    result = client.esearch("GERD")
    assert result["status"] == "completed_with_review"
    assert result["pmids"] == ["10", "11"]
    assert result["soft_warnings"] == [
        {"category": "quotedphrasesnotfound", "message": '"GERD chest pain"[tiab]'}
    ]


def test_esearch_phrase_not_found_zero_hits_is_review_not_failed(tmp_path: Path) -> None:
    def transport(*args):
        return HttpResponse(
            200,
            json.dumps(
                {
                    "esearchresult": {
                        "count": "0",
                        "idlist": [],
                        "querytranslation": "GERD",
                        "warninglist": {"quotedphrasesnotfound": ['"without dysplasia"[tiab]']},
                        "errorlist": {},
                    }
                }
            ).encode(),
        )

    client = NcbiClient(
        email=SecretStr("x@y.test"),
        api_key=None,
        tool="test",
        cache_dir=tmp_path,
        transport=transport,
        sleep=lambda _: None,
    )
    result = client.esearch("GERD")
    assert result["status"] == "completed_with_review"
    assert result["count"] == 0
    assert result["pmids"] == []


def test_esearch_non_empty_errorlist_is_failed(tmp_path: Path) -> None:
    def transport(*args):
        return HttpResponse(
            200,
            json.dumps(
                {
                    "esearchresult": {
                        "count": "0",
                        "idlist": [],
                        "querytranslation": "GERD",
                        "warninglist": {},
                        "errorlist": {"phrasesnotfound": ["broken"]},
                    }
                }
            ).encode(),
        )

    client = NcbiClient(
        email=SecretStr("x@y.test"),
        api_key=None,
        tool="test",
        cache_dir=tmp_path,
        transport=transport,
        sleep=lambda _: None,
    )
    with pytest.raises(ESearchQueryError, match="query syntax problem") as exc_info:
        client.esearch("GERD")
    assert exc_info.value.metadata["hard_errors"] == ["broken"]


def test_xml_parser_combines_abstract_parts_collective_authors_and_missing_abstract() -> None:
    xml = b"""<PubmedArticleSet>
    <PubmedArticle><MedlineCitation><PMID>1</PMID><Article>
    <ArticleTitle>Title</ArticleTitle><Abstract>
    <AbstractText Label="BACKGROUND">First</AbstractText>
    <AbstractText>Second</AbstractText></Abstract>
    <AuthorList><Author><CollectiveName>Study Group</CollectiveName></Author></AuthorList>
    <Journal><Title>Journal</Title><JournalIssue><PubDate><Year>2025</Year>
    <Month>Jan</Month></PubDate></JournalIssue></Journal>
    </Article></MedlineCitation></PubmedArticle>
    <PubmedArticle><MedlineCitation><PMID>2</PMID><Article>
    <ArticleTitle>No abstract</ArticleTitle></Article></MedlineCitation></PubmedArticle>
    </PubmedArticleSet>"""
    articles = parse_pubmed_xml(xml, fetched_at="2026-07-14T00:00:00Z")
    assert articles[0].abstract == "BACKGROUND: First\nSecond"
    assert articles[0].authors == ["Study Group"]
    assert articles[1].has_abstract is False


def test_fetch_deduplicates_pmids_and_preserves_many_to_many_provenance(tmp_path: Path) -> None:
    search_run = tmp_path / "search"
    search_run.mkdir()
    queries = [query_record("Q1", "U1", "F1", "one"), query_record("Q2", "U2", "F2", "two")]
    (search_run / "pubmed_queries.jsonl").write_text(
        "\n".join(json.dumps(value) for value in queries) + "\n", encoding="utf-8"
    )
    (search_run / "search_manifest.json").write_text(
        json.dumps({"run_mode": "complete"}), encoding="utf-8"
    )

    class FakeClient:
        def esearch(self, query, limit=None):
            return ["1"] if "(one)" in query else ["1", "2"]

        def efetch(self, pmids):
            records = "".join(
                f"<PubmedArticle><MedlineCitation><PMID>{pmid}</PMID><Article>"
                f"<ArticleTitle>T{pmid}</ArticleTitle></Article></MedlineCitation></PubmedArticle>"
                for pmid in pmids
            )
            return f"<PubmedArticleSet>{records}</PubmedArticleSet>".encode()

    def factory(**kwargs):
        return FakeClient()

    run = fetch_pubmed(
        input_run=search_run,
        output_root=tmp_path,
        worker_id="worker",
        email=SecretStr("owner@example.test"),
        api_key=SecretStr("secret-never-output"),
        client_factory=factory,
    )
    articles = [
        json.loads(line) for line in (run / "pubmed_articles.jsonl").read_text().splitlines()
    ]
    assert [article["pmid"] for article in articles] == ["1", "2"]
    assert articles[0]["query_ids"] == ["Q1", "Q2"]
    assert articles[0]["linked_formal_item_ids"] == ["F1", "F2"]
    assert "secret-never-output" not in "".join(
        path.read_text(errors="ignore") for path in run.rglob("*") if path.is_file()
    )


def test_all_zero_gate_fails_multiple_queries_but_single_zero_is_valid(tmp_path: Path) -> None:
    class ZeroClient:
        def esearch(self, query, limit=None):
            return {
                "pmids": [],
                "count": 0,
                "querytranslation": query,
                "warninglist": ["No items found."],
                "errorlist": [],
            }

        def efetch(self, pmids):
            raise AssertionError("No EFetch expected for zero hits")

    def factory(**kwargs):
        return ZeroClient()

    multi = tmp_path / "multi-search"
    multi.mkdir()
    _queries = [query_record("Q1", "U1", "F1", "one"), query_record("Q2", "U2", "F2", "two")]
    (multi / "pubmed_queries.jsonl").write_text(
        "".join(json.dumps(query) + "\n" for query in _queries), encoding="utf-8"
    )
    (multi / "search_manifest.json").write_text(
        json.dumps({"run_mode": "complete"}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="all_queries_returned_zero_hits"):
        fetch_pubmed(
            input_run=multi,
            output_root=tmp_path,
            worker_id="worker",
            email=SecretStr("owner@example.test"),
            api_key=None,
            client_factory=factory,
        )
    failed_run = next(tmp_path.glob("pubmed-fetch-*"))
    manifest = json.loads((failed_run / "pubmed_fetch_manifest.json").read_text())
    assert manifest["status"] == "failed"
    errors = [
        json.loads(line)
        for line in (failed_run / "pubmed_fetch_errors.jsonl").read_text().splitlines()
    ]
    assert errors[-1]["error_code"] == "all_queries_returned_zero_hits"
    esearch = [
        json.loads(line)
        for line in (failed_run / "pubmed_esearch_results.jsonl").read_text().splitlines()
    ]
    assert esearch[0]["count"] == 0
    assert esearch[0]["warninglist"] == {"messages": ["No items found."]}

    single = tmp_path / "single-search"
    single.mkdir()
    (single / "pubmed_queries.jsonl").write_text(
        json.dumps(query_record("Q3", "U3", "F3", "three")) + "\n", encoding="utf-8"
    )
    (single / "search_manifest.json").write_text(
        json.dumps({"run_mode": "complete"}), encoding="utf-8"
    )
    single_run = fetch_pubmed(
        input_run=single,
        output_root=tmp_path,
        worker_id="worker",
        email=SecretStr("owner@example.test"),
        api_key=None,
        client_factory=factory,
    )
    assert (
        json.loads((single_run / "pubmed_fetch_manifest.json").read_text())["status"]
        == "completed_with_review"
    )
    warnings = [
        json.loads(line)
        for line in (single_run / "pubmed_fetch_warnings.jsonl").read_text().splitlines()
    ]
    assert warnings[0]["message"] == "No items found."


def test_fetch_resume_requires_identical_fingerprint(tmp_path: Path) -> None:
    search_run = tmp_path / "search"
    search_run.mkdir()
    query_path = search_run / "pubmed_queries.jsonl"
    query_path.write_text(
        json.dumps(query_record("Q1", "U1", "F1", "one")) + "\n", encoding="utf-8"
    )
    (search_run / "search_manifest.json").write_text(
        json.dumps({"run_mode": "complete"}), encoding="utf-8"
    )
    resume = tmp_path / "resume"
    resume.mkdir()
    (resume / "checkpoint_fingerprint.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        fetch_pubmed(
            input_run=search_run,
            output_root=tmp_path,
            worker_id="worker",
            email=SecretStr("owner@example.test"),
            api_key=None,
            resume_run=resume,
        )


def test_failed_fetch_resume_reclassifies_soft_warning_checkpoints_and_reuses_articles(
    tmp_path: Path,
) -> None:
    search_run = tmp_path / "search"
    search_run.mkdir()
    queries = [
        query_record("Q1", "U1", "F1", "one"),
        query_record("Q2", "U2", "F2", "two"),
        query_record("Q3", "U3", "F3", "three"),
        query_record("Q4", "U4", "F4", "four"),
    ]
    (search_run / "pubmed_queries.jsonl").write_text(
        "".join(json.dumps(query) + "\n" for query in queries), encoding="utf-8"
    )
    (search_run / "search_manifest.json").write_text(
        json.dumps({"run_mode": "complete"}), encoding="utf-8"
    )

    class InitiallyHardClient:
        def esearch(self, query, limit=None):
            if "(one AND" in query:
                return ["1"]
            raise ESearchQueryError(
                "NCBI ESearch query syntax problem: NOT",
                {
                    "pmids": [],
                    "count": 0,
                    "querytranslation": 'GERD AND ("animals"[MeSH Terms] NOT "humans"[MeSH Terms])',
                    "warninglist": {"outputmessage": ["NOT"]},
                    "errorlist": {},
                },
            )

        def efetch(self, pmids):
            return (
                b"<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>1</PMID>"
                b"<Article><ArticleTitle>T1</ArticleTitle></Article></MedlineCitation>"
                b"</PubmedArticle></PubmedArticleSet>"
            )

    with pytest.raises(RuntimeError, match="esearch_query_failed"):
        fetch_pubmed(
            input_run=search_run,
            output_root=tmp_path,
            worker_id="worker",
            email=SecretStr("owner@example.test"),
            api_key=None,
            client_factory=lambda **kwargs: InitiallyHardClient(),
        )
    failed_run = next(tmp_path.glob("pubmed-fetch-*"))
    soft_specs = {
        "Q2": (["2", "3"], {"quotedphrasesnotfound": ['"GERD chest pain"[tiab]']}),
        "Q3": (["4"], {"phrasesignored": ['"Hiatal Hernia"[Mesh]']}),
        "Q4": (["5", "6"], {"messages": ['"without dysplasia"[tiab]']}),
    }
    for query_id, (pmids, warninglist) in soft_specs.items():
        query = next(item for item in queries if item["query_id"] == query_id)
        (failed_run / "checkpoints" / f"{query_id}.json").write_text(
            json.dumps(
                {
                    "status": "failed",
                    "pmids": pmids,
                    "count": len(pmids),
                    "querytranslation": query["final_pubmed_query"],
                    "warninglist": warninglist,
                    "errorlist": {},
                }
            ),
            encoding="utf-8",
        )

    efetch_batches: list[list[str]] = []

    class ResumeClient:
        def esearch(self, query, limit=None):
            raise AssertionError("ESearch should be reused from checkpoints")

        def efetch(self, pmids):
            efetch_batches.append(list(pmids))
            records = "".join(
                f"<PubmedArticle><MedlineCitation><PMID>{pmid}</PMID><Article>"
                f"<ArticleTitle>T{pmid}</ArticleTitle></Article></MedlineCitation></PubmedArticle>"
                for pmid in pmids
            )
            return f"<PubmedArticleSet>{records}</PubmedArticleSet>".encode()

    resumed = fetch_pubmed(
        input_run=search_run,
        output_root=tmp_path,
        worker_id="worker",
        email=SecretStr("owner@example.test"),
        api_key=None,
        resume_run=failed_run,
        client_factory=lambda **kwargs: ResumeClient(),
    )
    manifest = json.loads((resumed / "pubmed_fetch_manifest.json").read_text())
    assert manifest["status"] == "completed_with_review"
    assert manifest["summary"]["warning_queries"] == 3
    assert all(
        json.loads((resumed / "checkpoints" / f"{query_id}.json").read_text())["status"]
        == "completed_with_review"
        for query_id in soft_specs
    )
    assert json.loads((resumed / "checkpoints" / "Q1.json").read_text())["status"] == "completed"
    assert efetch_batches and "1" not in {pmid for batch in efetch_batches for pmid in batch}
    articles = [
        json.loads(line) for line in (resumed / "pubmed_articles.jsonl").read_text().splitlines()
    ]
    assert [article["pmid"] for article in articles] == ["1", "2", "3", "4", "5", "6"]
    warnings = [
        json.loads(line)
        for line in (resumed / "pubmed_fetch_warnings.jsonl").read_text().splitlines()
    ]
    assert {warning["message"] for warning in warnings} == {
        '"GERD chest pain"[tiab]',
        '"Hiatal Hernia"[Mesh]',
        '"without dysplasia"[tiab]',
    }


def test_search_generation_is_mocked_resumable_and_covers_every_item(tmp_path: Path) -> None:
    extraction = tmp_path / "extraction"
    extraction.mkdir()
    formal = [
        {**item("1", "recommendation"), "source_id": "SRC"},
        {**item("2", "statement"), "source_id": "SRC"},
    ]
    (extraction / "formal_items.jsonl").write_text(
        "\n".join(json.dumps(value) for value in formal) + "\n", encoding="utf-8"
    )
    (extraction / "comments.jsonl").write_text("", encoding="utf-8")
    (extraction / "document_map.validated.json").write_text("{}", encoding="utf-8")
    (extraction / "extraction_manifest.json").write_text(
        json.dumps({"status": "completed_with_review", "source_id": "SRC"}), encoding="utf-8"
    )
    calls = 0

    class FakeOpenAI:
        def create(self, prompt, payload):
            nonlocal calls
            calls += 1
            return SearchPlanDraft(search_units=[draft(["SRC_1", "SRC_2"])]).model_dump(mode="json")

    def factory(api_key, config):
        return FakeOpenAI()

    run = generate_searches(
        input_run=extraction,
        output_root=tmp_path,
        worker_id="worker",
        api_key=SecretStr("dummy-secret-never-output"),
        start_date=date(2023, 1, 1),
        end_date=date(2026, 7, 14),
        client_factory=factory,
    )
    resumed = generate_searches(
        input_run=extraction,
        output_root=tmp_path,
        worker_id="worker",
        api_key=SecretStr("different-dummy-secret"),
        start_date=date(2023, 1, 1),
        end_date=date(2026, 7, 14),
        resume_run=run,
        client_factory=factory,
    )
    assert resumed == run and calls == 1
    assert len((run / "formal_item_search_coverage.jsonl").read_text().splitlines()) == 2
    manifest = (run / "search_manifest.json").read_text()
    assert "dummy-secret-never-output" not in manifest
    with pytest.raises(ValueError, match="fingerprint"):
        generate_searches(
            input_run=extraction,
            output_root=tmp_path,
            worker_id="worker",
            api_key=SecretStr("dummy"),
            start_date=date(2024, 1, 1),
            end_date=date(2026, 7, 14),
            resume_run=run,
            client_factory=factory,
        )


def test_limited_search_is_marked_incomplete_and_cannot_be_fetched(tmp_path: Path) -> None:
    extraction = tmp_path / "extraction"
    extraction.mkdir()
    formal = [
        {**item("1", "recommendation"), "source_id": "SRC"},
        {**item("2", "statement"), "source_id": "SRC"},
    ]
    (extraction / "formal_items.jsonl").write_text(
        "\n".join(json.dumps(value) for value in formal) + "\n", encoding="utf-8"
    )
    (extraction / "comments.jsonl").write_text("", encoding="utf-8")
    (extraction / "document_map.validated.json").write_text("{}", encoding="utf-8")
    (extraction / "extraction_manifest.json").write_text(
        json.dumps({"status": "completed_with_review", "source_id": "SRC"}), encoding="utf-8"
    )

    class FakeOpenAI:
        def create(self, prompt, payload):
            return SearchPlanDraft(search_units=[draft(["SRC_1"])]).model_dump(mode="json")

    run = generate_searches(
        input_run=extraction,
        output_root=tmp_path,
        worker_id="worker",
        api_key=SecretStr("dummy"),
        start_date=date(2023, 1, 1),
        end_date=date(2026, 7, 14),
        limit=1,
        client_factory=lambda api_key, config: FakeOpenAI(),
    )
    manifest = json.loads((run / "search_manifest.json").read_text())
    assert manifest["status"] == "technical_limited"
    assert manifest["coverage_complete"] is False
    with pytest.raises(ValueError, match="limited Search run"):
        fetch_pubmed(
            input_run=run,
            output_root=tmp_path,
            worker_id="worker",
            email=SecretStr("owner@example.test"),
            api_key=None,
        )


def test_transport_uses_post_for_long_requests(monkeypatch) -> None:
    captured = {}

    class Response:
        def __init__(self):
            self.status = 200
            self.headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b"ok"

    def urlopen(request, timeout):
        captured["method"] = request.get_method()
        captured["url"] = request.full_url
        return Response()

    monkeypatch.setattr(ncbi.urllib.request, "urlopen", urlopen)
    ncbi._transport("https://example.test", {"term": "x" * 2000}, 1)
    assert captured == {"method": "POST", "url": "https://example.test"}
    ncbi._transport("https://example.test", {"term": "short", "email": "owner@example.test"}, 1)
    assert captured == {"method": "POST", "url": "https://example.test"}
