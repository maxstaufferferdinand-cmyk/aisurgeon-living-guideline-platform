"""GPT semantic structuring boundary for v3 transcripts."""

import hashlib
import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from aisurgeon.extraction.canonical.core import assign_formal_item_ids, assign_object_id
from aisurgeon.extraction.canonical.models import (
    CANONICAL_EXTRACTION_SCHEMA_VERSION,
    Comment,
    FormalItem,
    Reference,
    ReviewFinding,
    VisualObject,
)
from aisurgeon.extraction.canonical.outputs import (
    expert_consensus_view,
    recommendation_view,
    statement_view,
    write_json,
    write_jsonl,
    write_review_workbook,
)
from aisurgeon.extraction.transcription_v3.models import ExecutionMode

SEMANTIC_STRUCTURE_SCHEMA_VERSION = "semantic_structure_v1"
SEMANTIC_STRUCTURE_PROMPT_VERSION = "openai_guideline_semantic_structure_v1"
MODEL_CONFIG_PATH = Path("config/models/openai_guideline_semantic_structure_v1.json")
PROMPT_PATH = Path("config/prompts/openai_guideline_semantic_structure_v1.txt")
SYNTHETIC_MARKERS = (
    "Synthetic exact original comment",
    "Synthetic source reference",
    "synthetic transcript fixture",
    "Synthetic source transcript",
)


class SemanticStructureDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_metadata: dict[str, Any] = Field(default_factory=dict)
    publication_year: int | None = None
    publication_year_source: str | None = None
    formal_items: list[dict[str, Any]] = Field(default_factory=list)
    comments: list[dict[str, Any]] = Field(default_factory=list)
    references: list[dict[str, Any]] = Field(default_factory=list)
    tables: list[dict[str, Any]] = Field(default_factory=list)
    algorithms: list[dict[str, Any]] = Field(default_factory=list)
    review_findings: list[dict[str, Any]] = Field(default_factory=list)


def build_semantic_payload(transcription_run: Path) -> dict[str, Any]:
    transcript = json.loads(
        (transcription_run / "canonical_transcript.json").read_text(encoding="utf-8")
    )
    metadata = json.loads((transcription_run / "pdf_preflight.json").read_text(encoding="utf-8"))
    scout = json.loads((transcription_run / "extraction_scout.json").read_text(encoding="utf-8"))
    payload = {
        "canonical_transcript": transcript,
        "pdf_preflight": metadata,
        "extraction_scout": scout,
    }
    encoded = json.dumps(payload, ensure_ascii=False)
    if "%PDF-" in encoded or "file_uri" in encoded or "pdf_uri" in encoded:
        raise ValueError("Semantic structuring payload must not contain PDF bytes or PDF URI")
    return payload


def _default_draft(payload: dict[str, Any]) -> SemanticStructureDraft:
    source_id = payload["canonical_transcript"]["source_id"]
    blocks = [
        block
        for content in payload["canonical_transcript"]["contents"]
        for block in content.get("visual_blocks", [])
    ]
    first_text = blocks[0]["exact_visible_text"] if blocks else "Synthetic formal item."
    return SemanticStructureDraft(
        document_metadata={"source_id": source_id, "title": "Synthetic guideline"},
        publication_year=2018,
        publication_year_source="synthetic transcript fixture",
        formal_items=[
            {
                "item_type": "recommendation",
                "item_type_raw": "Empfehlung",
                "original_number": "1",
                "exact_original_text": first_text,
                "page_start": blocks[0]["page_number"] if blocks else 1,
                "page_end": blocks[0]["page_number"] if blocks else 1,
                "extraction_confidence": 1.0,
            }
        ],
        comments=[
            {
                "exact_original_text": "Synthetic exact original comment.",
                "page_start": blocks[0]["page_number"] if blocks else 1,
                "page_end": blocks[0]["page_number"] if blocks else 1,
                "extraction_confidence": 1.0,
                "related_original_number": "1",
            }
        ],
        references=[
            {
                "original_reference_number": "1",
                "exact_original_reference_text": "[1] Synthetic source reference.",
                "page_start": blocks[-1]["page_number"] if blocks else 1,
                "page_end": blocks[-1]["page_number"] if blocks else 1,
                "extraction_confidence": 1.0,
            }
        ],
    )


def derive_pubmed_start_date(publication_year: int | None, override: str | None = None) -> str:
    if override:
        return override
    if publication_year is None or publication_year < 1900 or publication_year > 2200:
        raise ValueError("Missing or impossible publication year blocks automatic PubMed fetch")
    return f"{publication_year:04d}-01-01"


def _load_model_config() -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[3]
    return json.loads((project_root / MODEL_CONFIG_PATH).read_text(encoding="utf-8"))


def _load_prompt() -> str:
    project_root = Path(__file__).resolve().parents[3]
    return (project_root / PROMPT_PATH).read_text(encoding="utf-8")


def _safe_usage(usage: Any) -> dict[str, int] | None:
    if usage is None:
        return None
    names = (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "prompt_tokens",
        "completion_tokens",
    )
    values = {
        name: value
        for name in names
        if isinstance((value := getattr(usage, name, None)), int)
    }
    if isinstance(usage, dict):
        values.update({key: value for key, value in usage.items() if isinstance(value, int)})
    return values or None


class OpenAISemanticStructureProvider:
    """Live OpenAI Responses boundary for semantic structuring."""

    provider_backend = "openai_responses"

    def __init__(
        self,
        *,
        api_key: SecretStr,
        config: dict[str, Any] | None = None,
        client: Any | None = None,
    ) -> None:
        self.config = config or _load_model_config()
        self.evidence: list[dict[str, Any]] = []
        if client is None:
            from openai import OpenAI  # type: ignore[import-not-found]

            client = OpenAI(
                api_key=api_key.get_secret_value(),
                timeout=self.config["request_timeout_seconds"],
                max_retries=self.config["max_attempts"] - 1,
            )
        self._client = client

    def create(self, *, prompt: str, payload: dict[str, Any]) -> SemanticStructureDraft:
        start = time.monotonic()
        try:
            response = self._client.responses.parse(
                model=self.config["model_id"],
                reasoning={"effort": self.config["reasoning_effort"]},
                instructions=prompt,
                input=json.dumps(payload, ensure_ascii=False),
                text_format=SemanticStructureDraft,
            )
            parsed = response.output_parsed
            if parsed is None:
                raise ValueError("OpenAI response contained no parsed semantic structure")
            self.evidence.append(
                {
                    "provider_backend": self.provider_backend,
                    "success": True,
                    "response_id": getattr(response, "id", None),
                    "token_usage": _safe_usage(getattr(response, "usage", None)),
                    "duration_seconds": round(time.monotonic() - start, 3),
                }
            )
            return parsed
        except Exception as exc:
            self.evidence.append(
                {
                    "provider_backend": self.provider_backend,
                    "success": False,
                    "safe_error_class": type(exc).__name__,
                    "safe_error_message": str(exc)[:200],
                    "duration_seconds": round(time.monotonic() - start, 3),
                }
            )
            message = f"OpenAI semantic structure request failed ({type(exc).__name__})"
            raise RuntimeError(message) from exc


def _assert_transcription_compatible(
    transcription_run: Path, execution_mode: ExecutionMode
) -> None:
    manifest_path = transcription_run / "transcription_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Transcription manifest missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if execution_mode == "live":
        if manifest.get("execution_mode") != "live":
            raise ValueError("Live semantic structuring requires a live transcription run")
        if manifest.get("status") not in {"completed", "technical_limited"}:
            raise ValueError("Live semantic structuring requires completed source transcription")
        if int(manifest.get("provider_call_count") or 0) <= 0:
            raise ValueError("Live semantic structuring requires Gemini provider evidence")


def _contains_synthetic_marker(draft: SemanticStructureDraft) -> bool:
    encoded = draft.model_dump_json()
    return any(marker in encoded for marker in SYNTHETIC_MARKERS)


def run_semantic_structure(
    *,
    transcription_run: Path,
    output_root: Path,
    worker_id: str,
    api_key: SecretStr | None = None,
    execution_mode: ExecutionMode = "mock_test",
    resume_run: Path | None = None,
    limit: int | None = None,
    draft_factory: Callable[[dict[str, Any]], SemanticStructureDraft] | None = None,
    provider: OpenAISemanticStructureProvider | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> Path:
    _assert_transcription_compatible(transcription_run, execution_mode)
    payload = build_semantic_payload(transcription_run)
    source_id = payload["canonical_transcript"]["source_id"]
    if resume_run is None:
        stamp = now().strftime("%Y%m%dT%H%M%S%fZ")
        run_dir = output_root.resolve() / f"structure-v3-{stamp}-{source_id}"
        run_dir.mkdir(parents=True, exist_ok=False)
    else:
        run_dir = resume_run.resolve()
    provider_evidence: list[dict[str, Any]] = []
    if execution_mode == "live":
        if api_key is None and provider is None:
            raise ValueError("Live semantic structuring requires OPENAI_API_KEY")
        provider = provider or OpenAISemanticStructureProvider(api_key=api_key)
        draft = provider.create(prompt=_load_prompt(), payload=payload)
        provider_evidence = provider.evidence
        if not provider_evidence or not any(item.get("success") for item in provider_evidence):
            raise ValueError("Live semantic structuring produced no OpenAI provider evidence")
        if _contains_synthetic_marker(draft):
            raise ValueError("Synthetic fixture marker detected in live semantic structure output")
    elif execution_mode == "mock_test":
        draft = (draft_factory or _default_draft)(payload)
    elif execution_mode == "dry_run":
        write_json(
            run_dir / "extraction_manifest.json",
            {
                "schema_version": SEMANTIC_STRUCTURE_SCHEMA_VERSION,
                "status": "dry_run",
                "execution_mode": "dry_run",
                "provider_backend": "none",
                "provider_call_count": 0,
                "source_id": source_id,
                "worker_id": worker_id,
                "input_transcription_run": str(transcription_run.resolve()),
            },
        )
        return run_dir
    else:
        raise ValueError("Unsupported execution_mode")
    if limit is not None:
        draft.formal_items = draft.formal_items[:limit]
    items = [
        FormalItem.model_validate(
            {
                **item,
                "source_id": source_id,
                "schema_version": CANONICAL_EXTRACTION_SCHEMA_VERSION,
                "extraction_batch_id": f"{source_id}_STRUCTURE_V3",
            }
        )
        for item in draft.formal_items
    ]
    assign_formal_item_ids(items)
    for sequence_number, item in enumerate(items, start=1):
        item.sequence_number = sequence_number
    comments = [
        Comment.model_validate(
            {
                **comment,
                "source_id": source_id,
                "schema_version": CANONICAL_EXTRACTION_SCHEMA_VERSION,
                "extraction_batch_id": f"{source_id}_STRUCTURE_V3",
            }
        )
        for comment in draft.comments
    ]
    for comment in comments:
        comment.comment_id = assign_object_id(
            source_id, "COMMENT", comment.exact_original_text, str(comment.page_start)
        )
    refs = [
        Reference.model_validate(
            {
                **ref,
                "source_id": source_id,
                "schema_version": CANONICAL_EXTRACTION_SCHEMA_VERSION,
            }
        )
        for ref in draft.references
    ]
    for ref in refs:
        ref.reference_id = assign_object_id(source_id, "REF", ref.original_reference_number)
    tables = [
        VisualObject.model_validate(
            {
                **table,
                "source_id": source_id,
                "schema_version": CANONICAL_EXTRACTION_SCHEMA_VERSION,
                "object_type": "table",
                "extraction_confidence": table.get("extraction_confidence", 1.0),
            }
        )
        for table in draft.tables
    ]
    algorithms = [
        VisualObject.model_validate(
            {
                **algorithm,
                "source_id": source_id,
                "schema_version": CANONICAL_EXTRACTION_SCHEMA_VERSION,
                "object_type": "algorithm",
                "extraction_confidence": algorithm.get("extraction_confidence", 1.0),
            }
        )
        for algorithm in draft.algorithms
    ]
    findings = [
        ReviewFinding.model_validate(finding)
        for finding in draft.review_findings
    ]
    write_jsonl(run_dir / "formal_items.jsonl", items)
    write_jsonl(run_dir / "recommendations.jsonl", recommendation_view(items))
    write_jsonl(run_dir / "statements.jsonl", statement_view(items))
    write_jsonl(run_dir / "expert_consensus_items.jsonl", expert_consensus_view(items))
    write_jsonl(run_dir / "comments.jsonl", comments)
    write_jsonl(run_dir / "references.jsonl", refs)
    write_jsonl(run_dir / "clinical_context_blocks.jsonl", [])
    write_jsonl(run_dir / "tables.jsonl", tables)
    write_jsonl(run_dir / "algorithms.jsonl", algorithms)
    write_jsonl(run_dir / "review_findings.jsonl", findings)
    write_review_workbook(run_dir / "review_findings.xlsx", findings)
    document_map = {
        "schema_version": "document_map_v1",
        "source_id": source_id,
        "publication_year": draft.publication_year,
        "publication_year_source": draft.publication_year_source,
    }
    write_json(run_dir / "document_map.validated.json", document_map)
    status = (
        "mock_test"
        if execution_mode == "mock_test"
        else "technical_limited"
        if limit is not None
        else "completed"
    )
    write_json(
        run_dir / "extraction_summary.json",
        {
            "status": status,
            "formal_item_count": len(items),
            "comment_count": len(comments),
            "reference_count": len(refs),
            "review_finding_count": len(findings),
        },
    )
    input_hash = hashlib.sha256(
        (transcription_run / "canonical_transcript.json").read_bytes()
    ).hexdigest()
    write_json(
        run_dir / "extraction_manifest.json",
        {
            "schema_version": SEMANTIC_STRUCTURE_SCHEMA_VERSION,
            "status": status,
            "execution_mode": execution_mode,
            "provider_backend": (
                "openai_responses" if execution_mode == "live" else "internal_mock"
            ),
            "provider_call_count": len(provider_evidence),
            "successful_call_count": sum(1 for item in provider_evidence if item.get("success")),
            "failed_call_count": sum(1 for item in provider_evidence if not item.get("success")),
            "provider_evidence": provider_evidence,
            "source_id": source_id,
            "worker_id": worker_id,
            "input_transcription_run": str(transcription_run.resolve()),
            "input_transcript_hash": input_hash,
            "model_provider": "openai",
            "model_id": "gpt-5.5",
            "reasoning_effort": "high",
            "prompt_version": SEMANTIC_STRUCTURE_PROMPT_VERSION,
            "publication_year": draft.publication_year,
            "publication_year_source": draft.publication_year_source,
            "pubmed_default_start_date": derive_pubmed_start_date(draft.publication_year),
            "credential_status": {"OPENAI_API_KEY": "set" if api_key else "not_used"},
            "limit": limit,
        },
    )
    return run_dir
