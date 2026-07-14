"""Deterministic orchestration and audit output for the Gemini smoke test."""

import hashlib
import json
import os
import platform
import re
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import SecretStr, ValidationError

from aisurgeon import __version__
from aisurgeon.extraction.gemini.client import GeminiDocumentMapClient
from aisurgeon.extraction.gemini.errors import GeminiConfigurationError, GeminiError
from aisurgeon.extraction.gemini.models import (
    DocumentMap,
    DocumentMapValidationReport,
    GeminiModelConfig,
    RemoteFileMetadata,
    RunManifest,
    ValidationIssue,
)
from aisurgeon.extraction.pdf_registration import PdfRegistration, register_pdf, sha256_file

MODEL_CONFIG_RELATIVE_PATH = Path("config/models/gemini_document_map_v1.json")
PROMPT_RELATIVE_PATH = Path("config/prompts/gemini_document_map_v1.txt")
SCHEMA_RELATIVE_PATH = Path("schemas/extraction/document_map_v1.schema.json")
_SAFE_RUN_COMPONENT = re.compile(r"[^A-Za-z0-9._-]+")


def find_project_root(start: Path | None = None) -> Path:
    """Find versioned configuration without embedding a machine-specific path."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "AGENTS.md").is_file():
            return candidate
    raise GeminiConfigurationError("Projektwurzel mit versionierter Konfiguration nicht gefunden.")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_model_config(project_root: Path) -> tuple[GeminiModelConfig, bytes]:
    path = project_root / MODEL_CONFIG_RELATIVE_PATH
    try:
        raw = path.read_bytes()
        return GeminiModelConfig.model_validate_json(raw), raw
    except (OSError, ValidationError) as exc:
        raise GeminiConfigurationError("Gemini-Modellkonfiguration ist ungültig.") from exc


def load_prompt(project_root: Path) -> tuple[str, str]:
    path = project_root / PROMPT_RELATIVE_PATH
    try:
        raw = path.read_bytes()
        prompt = raw.decode("utf-8").strip()
    except (OSError, UnicodeError) as exc:
        raise GeminiConfigurationError("Gemini-Prompt ist nicht lesbar.") from exc
    if not prompt:
        raise GeminiConfigurationError("Gemini-Prompt ist leer.")
    return prompt, sha256_bytes(raw)


def load_schema(project_root: Path) -> tuple[dict[str, Any], bytes]:
    path = project_root / SCHEMA_RELATIVE_PATH
    try:
        raw = path.read_bytes()
        schema = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise GeminiConfigurationError("DocumentMap-JSON-Schema ist ungültig.") from exc
    return schema, raw


def validate_versioned_schema(schema: dict[str, Any]) -> None:
    """Ensure the committed schema is exactly the Pydantic schema used at runtime."""
    if schema != DocumentMap.model_json_schema():
        raise GeminiConfigurationError(
            "Versioniertes DocumentMap-Schema weicht vom Pydantic-Modell ab."
        )


def _safe_component(value: str, *, limit: int = 40) -> str:
    sanitized = _SAFE_RUN_COMPONENT.sub("-", value).strip("-._")
    return (sanitized or "unknown")[:limit]


def make_run_id(
    *,
    timestamp: datetime,
    worker_id: str,
    source_id: str,
    pdf_sha256: str,
) -> str:
    """Create a sortable unique run ID from explicit audit components."""
    utc_timestamp = timestamp.astimezone(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return "_".join(
        (
            utc_timestamp,
            _safe_component(worker_id),
            _safe_component(source_id),
            pdf_sha256[:8],
        )
    )


def git_metadata(project_root: Path) -> tuple[str, str, bool]:
    """Return commit, branch, and dirty state without changing Git."""
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    try:
        commit = run("rev-parse", "HEAD")
        branch = run("branch", "--show-current") or "detached"
        dirty = bool(run("status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError) as exc:
        raise GeminiConfigurationError("Git-Metadaten konnten nicht gelesen werden.") from exc
    return commit, branch, dirty


def ensure_output_outside_repository(output_root: Path, project_root: Path) -> Path:
    resolved = output_root.expanduser().resolve()
    repository = project_root.resolve()
    if resolved == repository or repository in resolved.parents:
        raise GeminiConfigurationError("Run-Ausgaben müssen außerhalb des Repositorys liegen.")
    return resolved


def _all_page_spans(document_map: DocumentMap) -> list[tuple[str, int, int]]:
    spans: list[tuple[str, int, int]] = []
    for field_name in (
        "front_matter_page_ranges",
        "table_of_contents_page_ranges",
        "clinical_main_body_page_ranges",
        "bibliography_page_ranges",
        "appendix_page_ranges",
        "uncertain_regions",
    ):
        spans.extend(
            (field_name, item.page_start, item.page_end)
            for item in getattr(document_map, field_name)
        )
    for field_name in (
        "detected_table_inventory",
        "detected_algorithm_inventory",
        "detected_decision_tree_inventory",
    ):
        spans.extend(
            (field_name, item.page_start, item.page_end)
            for item in getattr(document_map, field_name)
        )
    return spans


def _document_region_spans(document_map: DocumentMap) -> list[tuple[str, int, int]]:
    spans: list[tuple[str, int, int]] = []
    for field_name in (
        "front_matter_page_ranges",
        "table_of_contents_page_ranges",
        "clinical_main_body_page_ranges",
        "bibliography_page_ranges",
        "appendix_page_ranges",
    ):
        spans.extend(
            (field_name, item.page_start, item.page_end)
            for item in getattr(document_map, field_name)
        )
    return spans


def validate_document_map(
    document_map: DocumentMap,
    registration: PdfRegistration,
) -> DocumentMapValidationReport:
    """Compare the structured map against immutable local technical facts."""
    issues: list[ValidationIssue] = []
    if document_map.source_id != registration.source_id:
        issues.append(
            ValidationIssue(
                severity="error",
                code="source_id_mismatch",
                message="DocumentMap source_id stimmt nicht mit der Registrierung überein.",
            )
        )
    usable_clinical_ranges = [
        item
        for item in document_map.clinical_main_body_page_ranges
        if item.page_start >= 1 and item.page_start <= item.page_end
    ]
    if not usable_clinical_ranges:
        issues.append(
            ValidationIssue(
                severity="error",
                code="clinical_main_structure_unusable",
                message="Gemini hat keine verwertbare klinische Hauptstruktur geliefert.",
            )
        )
    if registration.page_count is None:
        issues.append(
            ValidationIssue(
                severity="warning",
                code="local_page_count_unavailable",
                message="Lokale Seitenzahl ist nicht verfügbar; menschliche Prüfung erforderlich.",
            )
        )
    elif document_map.declared_page_count != registration.page_count:
        issues.append(
            ValidationIssue(
                severity="warning",
                code="page_count_mismatch",
                message=(
                    "Deklarierte und lokale PDF-Seitenzahl weichen ab; "
                    "menschliche Prüfung erforderlich."
                ),
            )
        )
    invalid_range_fields = {
        field_name
        for field_name, page_start, page_end in _all_page_spans(document_map)
        if page_start < 1 or page_start > page_end
    }
    for field_name in sorted(invalid_range_fields):
        issues.append(
            ValidationIssue(
                severity="warning",
                code="invalid_page_range",
                message=f"Mindestens ein Seitenbereich in {field_name} ist unklar oder ungültig.",
            )
        )
    if registration.page_count is not None:
        out_of_bounds_fields: set[str] = set()
        for field_name, _page_start, page_end in _all_page_spans(document_map):
            if page_end > registration.page_count:
                out_of_bounds_fields.add(field_name)
        for field_name in sorted(out_of_bounds_fields):
            issues.append(
                ValidationIssue(
                    severity="warning",
                    code="page_range_out_of_bounds",
                    message=(
                        f"Seitenbereich in {field_name} liegt außerhalb der lokal ermittelten "
                        "Seitenzahl; menschliche Prüfung erforderlich."
                    ),
                )
            )
    ordered_regions = sorted(_document_region_spans(document_map), key=lambda item: item[1:])
    overlap_pairs: set[tuple[str, str]] = set()
    for index, current in enumerate(ordered_regions):
        for following in ordered_regions[index + 1 :]:
            if following[1] > current[2]:
                break
            overlap_pairs.add(tuple(sorted((current[0], following[0]))))
    for first, second in sorted(overlap_pairs):
        issues.append(
            ValidationIssue(
                severity="warning",
                code="document_region_overlap",
                message=f"Dokumentregionen {first} und {second} überlappen oder sind unklar.",
            )
        )
    if not document_map.detected_formal_item_types:
        issues.append(
            ValidationIssue(
                severity="warning",
                code="formal_item_types_empty",
                message="Keine formalen Itemtypen erkannt; menschliche Prüfung erforderlich.",
            )
        )
    missing_orientation_fields = [
        field_name
        for field_name in ("detected_document_layout", "column_layout")
        if not (getattr(document_map, field_name) or "").strip()
    ]
    if missing_orientation_fields:
        issues.append(
            ValidationIssue(
                severity="warning",
                code="document_layout_metadata_incomplete",
                message=(
                    "Dokumentlayout-Metadaten sind unvollständig; "
                    "menschliche Prüfung erforderlich."
                ),
            )
        )
    return DocumentMapValidationReport(
        valid=not any(issue.severity == "error" for issue in issues),
        review_required=bool(issues or document_map.uncertain_regions or document_map.warnings),
        issues=issues,
    )


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json(path: Path, value: Any) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    _atomic_write(path, (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode())


def _hash_outputs(run_dir: Path, filenames: list[str]) -> dict[str, str]:
    return {filename: sha256_file(run_dir / filename) for filename in filenames}


def run_document_map(
    *,
    pdf_path: Path,
    worker_id: str,
    output_root: Path,
    api_key: SecretStr | None,
    source_id: str | None = None,
    dry_run: bool = False,
    allow_dirty: bool = False,
    keep_remote_file: bool = False,
    project_root: Path | None = None,
    client_factory: Callable[..., GeminiDocumentMapClient] = GeminiDocumentMapClient,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> tuple[RunManifest, Path]:
    """Run or plan one document-map attempt with secret-free audit files."""
    root = project_root or find_project_root()
    output = ensure_output_outside_repository(output_root, root)
    model_config, model_config_raw = load_model_config(root)
    prompt, prompt_hash = load_prompt(root)
    schema, schema_raw = load_schema(root)
    validate_versioned_schema(schema)
    registration = register_pdf(pdf_path, worker_id=worker_id, source_id=source_id)
    commit, branch, dirty = git_metadata(root)
    if not dry_run and dirty and not allow_dirty:
        raise GeminiConfigurationError(
            "Live-Lauf bei Dirty Worktree gesperrt; --allow-dirty ausdrücklich verwenden."
        )
    if not dry_run and api_key is None:
        raise GeminiConfigurationError("GEMINI_API_KEY fehlt für den Live-Lauf.")
    if registration.encrypted:
        raise GeminiConfigurationError("Verschlüsselte PDFs werden nicht an Gemini übertragen.")

    started = now()
    run_id = make_run_id(
        timestamp=started,
        worker_id=worker_id,
        source_id=registration.source_id,
        pdf_sha256=registration.sha256,
    )
    run_dir = output / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "logs").mkdir(mode=0o700)
    manifest = RunManifest(
        run_id=run_id,
        stage="gemini_document_map",
        status="dry_run" if dry_run else "running",
        worker_id=worker_id,
        source_id=registration.source_id,
        pdf_filename=registration.original_filename,
        pdf_sha256=registration.sha256,
        file_size_bytes=registration.file_size_bytes,
        local_page_count=registration.page_count,
        git_commit=commit,
        git_branch=branch,
        dirty_worktree=dirty,
        python_version=platform.python_version(),
        package_version=__version__,
        model_provider=model_config.provider,
        model_id=model_config.model_id,
        api_surface=model_config.api,
        thinking_level=model_config.thinking_level,
        media_resolution=model_config.media_resolution,
        prompt_version=model_config.prompt_version,
        prompt_sha256=prompt_hash,
        schema_version=model_config.schema_version,
        schema_sha256=sha256_bytes(schema_raw),
        start_time_utc=started,
    )
    _write_json(run_dir / "pdf_registration.json", registration)
    _atomic_write(run_dir / "model_config_snapshot.json", model_config_raw)
    _atomic_write(run_dir / "prompt_snapshot.txt", prompt.encode("utf-8") + b"\n")
    _atomic_write(run_dir / "logs/run.log", b"Gemini document-map run initialized.\n")

    output_names = [
        "pdf_registration.json",
        "model_config_snapshot.json",
        "prompt_snapshot.txt",
        "logs/run.log",
    ]
    if dry_run:
        manifest.end_time_utc = now()
        manifest.output_files = _hash_outputs(run_dir, output_names)
        _write_json(run_dir / "run_manifest.json", manifest)
        return manifest, run_dir

    gateway = client_factory(api_key=api_key, model_config=model_config)
    try:
        result = gateway.create_document_map(
            pdf_path=pdf_path,
            prompt=prompt,
            source_id=registration.source_id,
            keep_remote_file=keep_remote_file,
        )
        report = validate_document_map(result.document_map, registration)
        _atomic_write(run_dir / "document_map.raw.json", result.raw_json.encode("utf-8"))
        _write_json(run_dir / "document_map.validated.json", result.document_map)
        _write_json(run_dir / "validation_report.json", report)
        _write_json(run_dir / "remote_file_metadata.json", result.remote_file_metadata)
        output_names.extend(
            [
                "document_map.raw.json",
                "document_map.validated.json",
                "validation_report.json",
                "remote_file_metadata.json",
            ]
        )
        manifest.status = "succeeded" if report.valid else "validation_failed"
        manifest.token_usage = result.token_usage
        manifest.remote_file_deleted = result.remote_file_metadata.remote_file_deleted
    except GeminiError as exc:
        metadata: RemoteFileMetadata = gateway.last_remote_metadata
        _write_json(run_dir / "remote_file_metadata.json", metadata)
        output_names.append("remote_file_metadata.json")
        manifest.status = "failed"
        manifest.remote_file_deleted = metadata.remote_file_deleted
        manifest.errors.append(str(exc))
    manifest.end_time_utc = now()
    manifest.output_files = _hash_outputs(run_dir, output_names)
    _write_json(run_dir / "run_manifest.json", manifest)
    return manifest, run_dir
