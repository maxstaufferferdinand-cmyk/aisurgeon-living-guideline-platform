"""Official NCBI E-Utilities retrieval with retry, caching, pagination, and provenance."""

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

from openpyxl import Workbook
from pydantic import SecretStr

from aisurgeon.extraction.canonical.outputs import write_json, write_jsonl
from aisurgeon.search.pubmed.generation import ensure_external_run_root, file_hash, load_jsonl
from aisurgeon.search.pubmed.models import PubMedArticle, PubMedQuery
from aisurgeon.search.pubmed.query import (
    EVIDENCE_TYPE_FILTER,
    EXCLUSION_FILTER,
    HUMANS_FILTER,
    validate_query_core,
)

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _validated_queries(path: Path) -> list[dict[str, Any]]:
    records = [PubMedQuery.model_validate(value) for value in load_jsonl(path)]
    if not records:
        raise ValueError("pubmed_queries.jsonl is empty")
    validated = []
    for record in records:
        if record.humans_filter != HUMANS_FILTER:
            raise ValueError("Unexpected Humans filter in PubMed query")
        if record.evidence_type_filter != EVIDENCE_TYPE_FILTER:
            raise ValueError("Unexpected evidence-type filter in PubMed query")
        if record.exclusion_filter != EXCLUSION_FILTER:
            raise ValueError("Unexpected exclusion filter in PubMed query")
        errors = validate_query_core(record.query_core)
        if errors:
            raise ValueError(f"Invalid stored query_core: {', '.join(errors)}")
        expected = (
            f"({record.query_core}) AND {record.date_filter} AND {HUMANS_FILTER} "
            f"AND {EVIDENCE_TYPE_FILTER} {EXCLUSION_FILTER}"
        )
        if record.final_pubmed_query != expected:
            raise ValueError("Stored final_pubmed_query is not deterministic")
        validated.append(record.model_dump(mode="json"))
    return validated


class HttpResponse:
    def __init__(self, status: int, body: bytes, headers: dict[str, str] | None = None) -> None:
        self.status, self.body, self.headers = status, body, headers or {}


def _transport(url: str, params: dict[str, str], timeout: float) -> HttpResponse:
    encoded = urllib.parse.urlencode(params).encode()
    if len(encoded) > 1500 or "api_key" in params or "email" in params:
        request = urllib.request.Request(
            url,
            data=encoded,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
    else:
        request = urllib.request.Request(f"{url}?{encoded.decode()}", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return HttpResponse(response.status, response.read(), dict(response.headers))
    except urllib.error.HTTPError as exc:
        return HttpResponse(exc.code, exc.read(), dict(exc.headers))


class NcbiClient:
    def __init__(
        self,
        *,
        email: SecretStr,
        api_key: SecretStr | None,
        tool: str,
        cache_dir: Path,
        transport: Callable[..., HttpResponse] = _transport,
        sleep: Callable[[float], None] = time.sleep,
        timeout: float = 60,
        throttle_seconds: float | None = None,
        max_attempts: int = 3,
    ) -> None:
        self._email, self._key, self._tool = email, api_key, tool
        self._cache, self._transport, self._sleep = cache_dir, transport, sleep
        self._timeout = timeout
        self._throttle = (
            throttle_seconds if throttle_seconds is not None else (0.11 if api_key else 0.34)
        )
        self._attempts = max_attempts
        cache_dir.mkdir(parents=True, exist_ok=True)

    def _request(self, endpoint: str, params: dict[str, str]) -> bytes:
        safe = {
            **params,
            "email": "configured",
            "api_key": "configured" if self._key else "missing",
        }
        key = hashlib.sha256(json.dumps([endpoint, safe], sort_keys=True).encode()).hexdigest()
        cached = self._cache / f"{key}.response"
        if cached.exists():
            return cached.read_bytes()
        request_params = {**params, "email": self._email.get_secret_value(), "tool": self._tool}
        if self._key:
            request_params["api_key"] = self._key.get_secret_value()
        for attempt in range(1, self._attempts + 1):
            try:
                response = self._transport(f"{BASE_URL}/{endpoint}", request_params, self._timeout)
            except (TimeoutError, OSError):
                if attempt == self._attempts:
                    raise RuntimeError("NCBI transient request failure after retries") from None
                self._sleep(2 ** (attempt - 1))
                continue
            if response.status == 200:
                cached.write_bytes(response.body)
                self._sleep(self._throttle)
                return response.body
            if response.status == 429 or response.status >= 500:
                if attempt == self._attempts:
                    raise RuntimeError(
                        f"NCBI transient HTTP status {response.status} after retries"
                    )
                retry = response.headers.get("Retry-After")
                if retry and retry.isdigit():
                    delay = float(retry)
                elif retry:
                    try:
                        retry_at = parsedate_to_datetime(retry)
                        delay = max(
                            0.0,
                            (retry_at - datetime.now(retry_at.tzinfo)).total_seconds(),
                        )
                    except (TypeError, ValueError, OverflowError):
                        delay = 2 ** (attempt - 1)
                else:
                    delay = 2 ** (attempt - 1)
                self._sleep(delay)
                continue
            raise RuntimeError(f"NCBI permanent HTTP status {response.status}")
        raise AssertionError("unreachable")

    def esearch(self, query: str, *, page_size: int = 500, limit: int | None = None) -> list[str]:
        pmids: list[str] = []
        seen: set[str] = set()
        total: int | None = None
        retstart = 0
        while total is None or retstart < total:
            wanted = page_size if limit is None else min(page_size, limit - len(pmids))
            if wanted <= 0:
                break
            body = self._request(
                "esearch.fcgi",
                {
                    "db": "pubmed",
                    "term": query,
                    "retmode": "json",
                    "retstart": str(retstart),
                    "retmax": str(wanted),
                },
            )
            try:
                result = json.loads(body)["esearchresult"]
                total = int(result["count"])
                page = [str(value) for value in result.get("idlist", [])]
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                raise RuntimeError("NCBI ESearch returned an invalid JSON response") from None
            retstart += len(page)
            for pmid in page:
                if not pmid.isdigit():
                    raise RuntimeError("NCBI ESearch returned a non-numeric PMID")
                if pmid not in seen:
                    seen.add(pmid)
                    pmids.append(pmid)
            if not page:
                if retstart < total:
                    raise RuntimeError(
                        "NCBI ESearch pagination ended before the reported result count"
                    )
                break
        return pmids[:limit] if limit is not None else pmids

    def efetch(self, pmids: list[str]) -> bytes:
        return self._request(
            "efetch.fcgi", {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}
        )


def _text(element: ET.Element | None) -> str | None:
    if element is None:
        return None
    value = "".join(element.itertext()).strip()
    return value or None


def _date(element: ET.Element | None) -> tuple[str | None, int | None]:
    if element is None:
        return None, None
    year = _text(element.find("Year"))
    if not year:
        medline = _text(element.find("MedlineDate"))
        year = medline[:4] if medline and medline[:4].isdigit() else None
        return medline, int(year) if year else None
    parts = [year, _text(element.find("Month")), _text(element.find("Day"))]
    return "-".join(part for part in parts if part), int(year)


def parse_pubmed_xml(body: bytes, *, fetched_at: str) -> list[PubMedArticle]:
    root = ET.fromstring(body)
    articles = []
    for record in root.findall(".//PubmedArticle"):
        citation, article = record.find("MedlineCitation"), record.find("MedlineCitation/Article")
        if citation is None or article is None:
            continue
        pmid = _text(citation.find("PMID"))
        if not pmid:
            continue
        abstract_parts = []
        for part in article.findall("Abstract/AbstractText"):
            value = _text(part)
            if value:
                label = part.attrib.get("Label")
                abstract_parts.append(f"{label}: {value}" if label else value)
        authors = []
        for author in article.findall("AuthorList/Author"):
            collective = _text(author.find("CollectiveName"))
            personal = " ".join(
                filter(None, [_text(author.find("ForeName")), _text(author.find("LastName"))])
            )
            if collective or personal:
                authors.append(collective or personal)
        pub_date, year = _date(article.find("Journal/JournalIssue/PubDate"))
        doi = next(
            (
                _text(node)
                for node in record.findall("PubmedData/ArticleIdList/ArticleId")
                if node.attrib.get("IdType") == "doi"
            ),
            None,
        )
        if doi is None:
            doi = next(
                (
                    _text(node)
                    for node in article.findall("ELocationID")
                    if node.attrib.get("EIdType") == "doi"
                ),
                None,
            )
        electronic_date, _ = _date(article.find("ArticleDate"))
        print_date = pub_date
        articles.append(
            PubMedArticle(
                pmid=pmid,
                doi=doi,
                title=_text(article.find("ArticleTitle")) or "[No title]",
                abstract="\n".join(abstract_parts) or None,
                authors=authors,
                journal=_text(article.find("Journal/Title")),
                publication_date=pub_date,
                publication_year=year,
                publication_types=[
                    value
                    for node in article.findall("PublicationTypeList/PublicationType")
                    if (value := _text(node))
                ],
                mesh_terms=[
                    value
                    for node in citation.findall("MeshHeadingList/MeshHeading/DescriptorName")
                    if (value := _text(node))
                ],
                keywords=[
                    value
                    for node in citation.findall("KeywordList/Keyword")
                    if (value := _text(node))
                ],
                language=[value for node in article.findall("Language") if (value := _text(node))],
                electronic_publication_date=electronic_date,
                print_publication_date=print_date,
                fetched_at=fetched_at,
                has_abstract=bool(abstract_parts),
            )
        )
    return articles


def write_articles_xlsx(path: Path, articles: list[PubMedArticle]) -> None:
    if path.exists():
        raise FileExistsError(path)
    workbook, sheet = Workbook(), None
    sheet = workbook.active
    sheet.title = "pubmed_articles"
    headers = list(PubMedArticle.model_fields)
    sheet.append(headers)
    for article in articles:
        data = article.model_dump(mode="json")
        sheet.append(
            [
                json.dumps(data[h], ensure_ascii=False) if isinstance(data[h], list) else data[h]
                for h in headers
            ]
        )
    sheet.freeze_panes, sheet.auto_filter.ref = "A2", f"A1:{sheet.cell(1, len(headers)).coordinate}"
    workbook.save(path)


def fetch_pubmed(
    *,
    input_run: Path,
    output_root: Path,
    worker_id: str,
    email: SecretStr,
    api_key: SecretStr | None,
    tool: str = "aisurgeon",
    resume_run: Path | None = None,
    limit: int | None = None,
    expected_start_date: str | None = None,
    expected_end_date: str | None = None,
    client_factory: Callable[..., Any] = NcbiClient,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> Path:
    query_path = input_run.resolve() / "pubmed_queries.jsonl"
    if not query_path.is_file():
        raise ValueError("pubmed_queries.jsonl missing")
    search_manifest_path = input_run.resolve() / "search_manifest.json"
    if not search_manifest_path.is_file():
        raise ValueError("Search run has no search_manifest.json")
    search_manifest = json.loads(search_manifest_path.read_text(encoding="utf-8"))
    if search_manifest.get("run_mode") != "complete":
        raise ValueError("A technical limited Search run cannot be used for PubMed fetch")
    queries = _validated_queries(query_path)
    starts = {query.get("start_date") for query in queries}
    ends = {query.get("end_date") for query in queries}
    if len(starts) > 1 or len(ends) > 1:
        raise ValueError("PubMed queries do not share one immutable date interval")
    query_start = next(iter(starts), None)
    query_end = next(iter(ends), None)
    if expected_start_date is not None and expected_start_date != query_start:
        raise ValueError("--start-date does not match the immutable PubMed queries")
    if expected_end_date is not None and expected_end_date != query_end:
        raise ValueError("--end-date does not match the immutable PubMed queries")
    fingerprint = {
        "input_search_run": str(input_run.resolve()),
        "queries_sha256": file_hash(query_path),
        "search_manifest_sha256": file_hash(search_manifest_path),
        "limit_per_query": limit,
        "start_date": query_start,
        "end_date": query_end,
        "ncbi_tool": tool,
        "api_key_configured": api_key is not None,
        "email_configured": True,
    }
    if resume_run:
        run_dir = resume_run.resolve()
        if json.loads((run_dir / "checkpoint_fingerprint.json").read_text()) != fingerprint:
            raise ValueError("Resume fingerprint does not match")
        if (run_dir / "pubmed_fetch_manifest.json").is_file():
            return run_dir
    else:
        external_output_root = ensure_external_run_root(output_root, input_run)
        run_dir = (
            external_output_root
            / f"pubmed-fetch-{now():%Y%m%dT%H%M%S%fZ}-{fingerprint['queries_sha256'][:8]}"
        )
        run_dir.mkdir(parents=True, exist_ok=False)
        write_json(run_dir / "checkpoint_fingerprint.json", fingerprint)
    checkpoints, cache = run_dir / "checkpoints", run_dir / "cache"
    checkpoints.mkdir(exist_ok=True)
    client = client_factory(email=email, api_key=api_key, tool=tool, cache_dir=cache)
    provenance: dict[str, dict[str, set[str]]] = {}
    hits, errors = [], []
    for query in queries:
        checkpoint = checkpoints / f"{query['query_id']}.json"
        try:
            pmids = (
                json.loads(checkpoint.read_text())["pmids"]
                if checkpoint.exists()
                else client.esearch(query["final_pubmed_query"], limit=limit)
            )
            if not checkpoint.exists():
                write_json(checkpoint, {"status": "completed", "pmids": pmids})
            for pmid in pmids:
                hits.append(
                    {
                        "pmid": pmid,
                        "query_id": query["query_id"],
                        "search_unit_id": query["search_unit_id"],
                    }
                )
                item = provenance.setdefault(
                    pmid, {"query_ids": set(), "search_unit_ids": set(), "formal_item_ids": set()}
                )
                item["query_ids"].add(query["query_id"])
                item["search_unit_ids"].add(query["search_unit_id"])
                item["formal_item_ids"].update(query["linked_formal_item_ids"])
        except RuntimeError as exc:
            errors.append({"query_id": query["query_id"], "error": str(exc)})
    articles_by_id: dict[str, PubMedArticle] = {}
    all_pmids = list(provenance)
    for offset in range(0, len(all_pmids), 200):
        batch = all_pmids[offset : offset + 200]
        try:
            parsed = parse_pubmed_xml(client.efetch(batch), fetched_at=now().isoformat())
        except (RuntimeError, ET.ParseError) as exc:
            errors.append({"pmids": batch, "stage": "efetch", "error": str(exc)})
            continue
        for article in parsed:
            if article.pmid not in provenance:
                errors.append(
                    {
                        "pmid": article.pmid,
                        "stage": "efetch",
                        "error": "Unexpected PMID in EFetch response",
                    }
                )
                continue
            links = provenance[article.pmid]
            article.query_ids = sorted(links["query_ids"])
            article.search_unit_ids = sorted(links["search_unit_ids"])
            article.linked_formal_item_ids = sorted(links["formal_item_ids"])
            articles_by_id[article.pmid] = article
        returned = {article.pmid for article in parsed}
        for missing_pmid in sorted(set(batch) - returned, key=int):
            errors.append(
                {
                    "pmid": missing_pmid,
                    "stage": "efetch",
                    "error": "PMID missing from successful EFetch response",
                }
            )
    articles = [articles_by_id[key] for key in sorted(articles_by_id, key=int)]
    pmid_records = [
        {
            "pmid": pmid,
            "query_ids": sorted(provenance[pmid]["query_ids"]),
            "search_unit_ids": sorted(provenance[pmid]["search_unit_ids"]),
            "linked_formal_item_ids": sorted(provenance[pmid]["formal_item_ids"]),
        }
        for pmid in sorted(provenance, key=int)
    ]
    write_jsonl(run_dir / "pubmed_pmids.jsonl", pmid_records)
    write_jsonl(run_dir / "pubmed_query_hits.jsonl", hits)
    write_jsonl(run_dir / "pubmed_articles.jsonl", articles)
    write_jsonl(run_dir / "pubmed_fetch_errors.jsonl", errors)
    write_articles_xlsx(run_dir / "pubmed_articles.xlsx", articles)
    summary = {"unique_pmids": len(provenance), "articles": len(articles), "errors": len(errors)}
    write_json(run_dir / "pubmed_fetch_summary.json", summary)
    output_files = {
        path.name: file_hash(path)
        for path in run_dir.iterdir()
        if path.is_file() and path.name != "pubmed_fetch_manifest.json"
    }
    write_json(
        run_dir / "pubmed_fetch_manifest.json",
        {
            **fingerprint,
            "worker_id": worker_id,
            "created_at": now().isoformat(),
            "status": "technical_limited" if limit is not None else "completed",
            "run_mode": "technical_limited" if limit is not None else "complete",
            "limit_per_query": limit,
            "ncbi_configuration": {
                "api_key": "set" if api_key else "missing",
                "email": "set",
                "tool": tool,
            },
            "start_date": queries[0].get("start_date") if queries else None,
            "end_date": queries[0].get("end_date") if queries else None,
            "summary": summary,
            "output_files": output_files,
        },
    )
    return run_dir
