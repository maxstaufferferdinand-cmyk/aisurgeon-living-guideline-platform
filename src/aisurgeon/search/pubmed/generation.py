"""Search-unit normalization, complete coverage, manifests, and GPT boundary."""

import hashlib
import json
import subprocess
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from pydantic import SecretStr

from aisurgeon.extraction.canonical.outputs import write_json, write_jsonl
from aisurgeon.search.pubmed.models import (
    FormalItemSearchCoverage,
    SearchPlanDraft,
    SearchUnit,
)
from aisurgeon.search.pubmed.query import (
    EVIDENCE_TYPE_FILTER,
    EXCLUSION_FILTER,
    HUMANS_FILTER,
    QUERY_BUILDER_VERSION,
    build_query,
    sha256_text,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
MODEL_CONFIG_PATH = PROJECT_ROOT / "config/models/openai_pubmed_search_v1.json"
PROMPT_PATH = PROJECT_ROOT / "config/prompts/openai_pubmed_search_units_v1.txt"


def derive_start_date_from_extraction_manifest(
    input_run: Path, *, override: date | None = None
) -> tuple[date, dict[str, Any]]:
    """Derive Jan 1 of publication year unless an audited override is supplied."""
    if override is not None:
        return override, {"source": "explicit_override", "override": override.isoformat()}
    manifest_path = input_run.resolve() / "extraction_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("extraction_manifest.json is required to derive PubMed start date")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") is not None and manifest.get("status") not in {
        "completed",
        "completed_with_review",
    }:
        raise ValueError("PubMed start date requires a complete live structure run")
    year = manifest.get("publication_year")
    if not isinstance(year, int) or year < 1900 or year > 2200:
        document_map_path = input_run.resolve() / "document_map.validated.json"
        if document_map_path.is_file():
            year = json.loads(document_map_path.read_text(encoding="utf-8")).get(
                "publication_year"
            )
    if not isinstance(year, int) or year < 1900 or year > 2200:
        raise ValueError("Missing or impossible publication year blocks automatic PubMed fetch")
    return date(year, 1, 1), {
        "source": manifest.get("publication_year_source") or "extraction_manifest",
        "publication_year": year,
        "override": None,
    }


def ensure_external_run_root(output_root: Path, immutable_input_run: Path) -> Path:
    """Reject repository and immutable-input destinations before creating outputs."""
    resolved = output_root.resolve()
    if resolved == PROJECT_ROOT or resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError("Run output root must be outside the repository")
    input_resolved = immutable_input_run.resolve()
    if resolved == input_resolved or resolved.is_relative_to(input_resolved):
        raise ValueError("Run output root must not be inside the immutable input run")
    return resolved


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def normalize_search_plan(
    draft: SearchPlanDraft, formal_items: list[dict[str, Any]], source_id: str
) -> tuple[list[SearchUnit], list[FormalItemSearchCoverage]]:
    by_id = {str(item["formal_item_id"]): item for item in formal_items}
    linked: dict[str, list[tuple[str, str, str | None, bool]]] = {key: [] for key in by_id}
    units: list[SearchUnit] = []
    for sequence, candidate in enumerate(draft.search_units, start=1):
        unknown = set(candidate.linked_formal_item_ids) - set(by_id)
        if unknown:
            raise ValueError(f"Unknown formal_item_ids: {sorted(unknown)}")
        identity = (
            "\x1f".join(candidate.linked_formal_item_ids) + "\x1f" + (candidate.query_core or "")
        )
        unit_id = f"{source_id}_SEARCH_{sha256_text(identity)[:12]}"
        items = [by_id[item_id] for item_id in candidate.linked_formal_item_ids]
        unit = SearchUnit(
            **candidate.model_dump(),
            source_id=source_id,
            search_unit_id=unit_id,
            sequence_number=sequence,
            linked_original_item_numbers=[str(item.get("original_number") or "") for item in items],
            linked_formal_item_families=[
                str(item.get("normalized_item_family") or "other_formal_item") for item in items
            ],
            exact_formal_item_texts=[str(item["exact_original_text"]) for item in items],
        )
        units.append(unit)
        for item_id in candidate.linked_formal_item_ids:
            linked[item_id].append(
                (
                    unit_id,
                    candidate.search_relevance,
                    candidate.exclusion_reason,
                    candidate.review_required,
                )
            )
    for item_id, values in linked.items():
        relevances = {value[1] for value in values}
        if len(relevances) > 1:
            raise ValueError(f"Conflicting search relevance for formal item: {item_id}")
    missing = [item_id for item_id, values in linked.items() if not values]
    if missing:
        raise ValueError(f"Formal-item search coverage incomplete: {missing}")
    coverage = []
    for item_id, values in linked.items():
        relevance = (
            "search_relevant"
            if any(value[1] == "search_relevant" for value in values)
            else "not_search_relevant"
        )
        reasons = [value[2] for value in values if value[2]]
        coverage.append(
            FormalItemSearchCoverage(
                source_id=source_id,
                formal_item_id=item_id,
                search_relevance=relevance,
                linked_search_unit_ids=[value[0] for value in values],
                exclusion_reason=None if relevance == "search_relevant" else "; ".join(reasons),
                review_required=any(value[3] for value in values),
            )
        )
    return units, coverage


class OpenAISearchClient:
    """Small injectable Responses-API adapter; imports the SDK only for a live command."""

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
                text_format=SearchPlanDraft,
            )
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            detail = f", HTTP {status}" if isinstance(status, int) else ""
            raise RuntimeError(
                f"OpenAI search request failed ({type(exc).__name__}{detail})"
            ) from None
        parsed = response.output_parsed
        if parsed is None:
            raise ValueError("OpenAI response contained no parsed search plan")
        return parsed.model_dump(mode="json")


def _git_value(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or None


def generate_searches(
    *,
    input_run: Path,
    output_root: Path,
    worker_id: str,
    api_key: SecretStr,
    start_date: date,
    end_date: date,
    resume_run: Path | None = None,
    limit: int | None = None,
    start_date_audit: dict[str, Any] | None = None,
    client_factory: Callable[[SecretStr, dict[str, Any]], Any] = OpenAISearchClient,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> Path:
    required = [
        "formal_items.jsonl",
        "comments.jsonl",
        "document_map.validated.json",
        "extraction_manifest.json",
    ]
    paths = {name: input_run.resolve() / name for name in required}
    optional_context = input_run.resolve() / "clinical_context_blocks.jsonl"
    if optional_context.is_file():
        paths[optional_context.name] = optional_context
    if not all(path.is_file() for path in paths.values()):
        raise ValueError("Input extraction run is incomplete")
    config = json.loads(MODEL_CONFIG_PATH.read_text(encoding="utf-8"))
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    input_hashes = {name: file_hash(path) for name, path in paths.items()}
    all_formal = load_jsonl(paths["formal_items.jsonl"])
    if not all_formal:
        raise ValueError("formal_items.jsonl is empty")
    formal_ids = [item.get("formal_item_id") for item in all_formal]
    if any(not value for value in formal_ids) or len(formal_ids) != len(set(formal_ids)):
        raise ValueError("Formal items require unique non-empty formal_item_id values")
    source_ids = {item.get("source_id") for item in all_formal}
    if len(source_ids) != 1 or None in source_ids:
        raise ValueError("Formal items require one consistent non-empty source_id")
    extraction_manifest = json.loads(paths["extraction_manifest.json"].read_text(encoding="utf-8"))
    if extraction_manifest.get("status") not in {"completed", "completed_with_review"}:
        raise ValueError("Input extraction run is not completed")
    if extraction_manifest.get("source_id") != next(iter(source_ids)):
        raise ValueError("Extraction manifest source_id does not match formal_items.jsonl")
    if start_date > end_date:
        raise ValueError("start_date must not be after end_date")
    if start_date_audit is None:
        derived_start_date, start_date_audit = derive_start_date_from_extraction_manifest(
            input_run, override=start_date
        )
        if derived_start_date != start_date:
            raise ValueError("Derived PubMed start date mismatch")
    fingerprint = {
        "source_id": next(iter(source_ids)),
        "input_extraction_run": str(input_run.resolve()),
        "input_file_hashes": input_hashes,
        "model_id": config["model_id"],
        "model_configuration_hash": sha256_text(json.dumps(config, sort_keys=True)),
        "prompt_version": config["prompt_version"],
        "prompt_hash": sha256_text(prompt),
        "query_builder_version": QUERY_BUILDER_VERSION,
        "start_date": start_date.isoformat(),
        "start_date_audit": start_date_audit,
        "end_date": end_date.isoformat(),
        "humans_filter": HUMANS_FILTER,
        "evidence_type_filter": EVIDENCE_TYPE_FILTER,
        "exclusion_filter": EXCLUSION_FILTER,
        "limit": limit,
    }
    if resume_run:
        run_dir = resume_run.resolve()
        prior = json.loads((run_dir / "checkpoint_fingerprint.json").read_text(encoding="utf-8"))
        if prior != fingerprint:
            raise ValueError("Resume fingerprint does not match")
        if (run_dir / "search_manifest.json").exists():
            return run_dir
    else:
        stamp = now().strftime("%Y%m%dT%H%M%S%fZ")
        external_output_root = ensure_external_run_root(output_root, input_run)
        run_dir = (
            external_output_root
            / f"search-{stamp}-{fingerprint['source_id']}-{fingerprint['prompt_hash'][:8]}"
        )
        run_dir.mkdir(parents=True, exist_ok=False)
        write_json(run_dir / "checkpoint_fingerprint.json", fingerprint)
    formal = all_formal
    if limit is not None:
        formal = formal[:limit]
    comments = load_jsonl(paths["comments.jsonl"])
    comment_ids = {comment.get("comment_id") for comment in comments if comment.get("comment_id")}
    document_map = json.loads(paths["document_map.validated.json"].read_text(encoding="utf-8"))
    payload = {"formal_items": formal, "comments": comments, "document_map": document_map}
    if optional_context.is_file():
        payload["clinical_context_blocks"] = load_jsonl(optional_context)
    raw_plan_path = run_dir / "gpt_search_plan.raw.json"
    if raw_plan_path.exists():
        raw_plan = json.loads(raw_plan_path.read_text(encoding="utf-8"))
    else:
        raw_plan = client_factory(api_key, config).create(prompt, payload)
        write_json(raw_plan_path, raw_plan)
    draft = SearchPlanDraft.model_validate(raw_plan)
    unknown_comment_ids = {
        comment_id
        for unit in draft.search_units
        for comment_id in unit.relevant_comment_ids
        if comment_id not in comment_ids
    }
    if unknown_comment_ids:
        raise ValueError(f"Unknown relevant_comment_ids: {sorted(unknown_comment_ids)}")
    source_id = str(fingerprint["source_id"])
    units, coverage = normalize_search_plan(draft, formal, source_id)
    queries = [
        build_query(
            unit,
            start_date=start_date,
            end_date=end_date,
            prompt_version=config["prompt_version"],
            prompt_hash=fingerprint["prompt_hash"],
            model_config=config,
            model_config_hash=fingerprint["model_configuration_hash"],
        )
        for unit in units
        if unit.search_relevance == "search_relevant"
    ]
    final_paths = [
        run_dir / "search_units.jsonl",
        run_dir / "formal_item_search_coverage.jsonl",
        run_dir / "pubmed_queries.jsonl",
        run_dir / "search_manifest.json",
    ]
    if any(path.exists() for path in final_paths):
        raise FileExistsError("Partial Search outputs exist; refusing dangerous overwrite")
    write_jsonl(run_dir / "search_units.jsonl", units)
    write_jsonl(run_dir / "formal_item_search_coverage.jsonl", coverage)
    write_jsonl(run_dir / "pubmed_queries.jsonl", queries)
    outputs = {path.name: file_hash(path) for path in run_dir.iterdir() if path.is_file()}
    write_json(
        run_dir / "search_manifest.json",
        {
            **fingerprint,
            "worker_id": worker_id,
            "created_at": now().isoformat(),
            "git_commit": _git_value("rev-parse", "HEAD"),
            "model_configuration": config,
            "status": "technical_limited" if limit is not None else "completed",
            "run_mode": "technical_limited" if limit is not None else "complete",
            "limit": limit,
            "coverage_complete": limit is None,
            "credential_status": {"OPENAI_API_KEY": "set"},
            "counts": {
                "input_formal_items": len(all_formal),
                "processed_formal_items": len(formal),
                "search_units": len(units),
                "queries": len(queries),
            },
            "output_files": outputs,
        },
    )
    return run_dir
