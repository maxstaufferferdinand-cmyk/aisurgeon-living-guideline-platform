"""Minimal local setup and configuration CLI."""

import os
import platform
from datetime import date
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from aisurgeon import __version__
from aisurgeon.config.settings import SERVICE_FIELDS, Settings
from aisurgeon.extraction.canonical.pipeline import prepare_dry_run, run_live_extraction
from aisurgeon.extraction.gemini.document_map import (
    ensure_output_outside_repository,
    find_project_root,
    run_document_map,
)
from aisurgeon.extraction.gemini.errors import GeminiConfigurationError
from aisurgeon.extraction.pdf_registration import PdfRegistrationError, register_pdf
from aisurgeon.mapping.pubmed import map_pubmed_evidence
from aisurgeon.orchestration.pubmed_mapping import run_to_mapping as orchestrate_to_mapping
from aisurgeon.search.pubmed.generation import generate_searches
from aisurgeon.search.pubmed.ncbi import fetch_pubmed

app = typer.Typer(
    name="aisurgeon",
    help="Local scaffold for the source-bound AISurgeon Living Guideline Platform.",
    no_args_is_help=True,
)

EnvFileOption = Annotated[
    Path | None,
    typer.Option(
        "--env-file",
        help="Explicit local env file; the project .env is never automatic.",
    ),
]


def _present(value: object | None) -> str:
    return "gesetzt" if value is not None else "fehlt"


def _path_state(path: Path | None, *, writable: bool = False) -> tuple[str, bool]:
    if path is None:
        return "fehlt", False
    if not path.exists():
        return "existiert nicht", False
    if not path.is_dir():
        return "ist kein Verzeichnis", False
    mode = os.W_OK if writable else os.R_OK
    if not os.access(path, mode):
        return "nicht beschreibbar" if writable else "nicht lesbar", False
    return "vorhanden, beschreibbar" if writable else "vorhanden, lesbar", True


def _load_settings(env_file: Path | None, **overrides: object) -> Settings:
    if env_file is not None and (not env_file.exists() or not env_file.is_file()):
        typer.echo(f"Konfigurationsfehler: env-Datei nicht gefunden: {env_file}", err=True)
        raise typer.Exit(2)
    try:
        return Settings.from_env_file(env_file, **overrides)
    except ValidationError as exc:
        typer.echo("Konfigurationsfehler: ungültige lokale Einstellungen.", err=True)
        raise typer.Exit(2) from exc


def _show_path(label: str, path: Path | None, *, writable: bool = False) -> bool:
    state, valid = _path_state(path, writable=writable)
    typer.echo(f"{label}: {path if path is not None else 'fehlt'} ({state})")
    return valid


@app.command("pdf-register")
def pdf_register(
    pdf: Annotated[Path, typer.Option("--pdf", help="Local PDF to register without modifying it.")],
    env_file: EnvFileOption = None,
    output_dir: Annotated[Path | None, typer.Option("--output-dir")] = None,
    source_id: Annotated[str | None, typer.Option("--source-id")] = None,
) -> None:
    """Register deterministic local PDF metadata without semantic extraction."""
    settings = _load_settings(env_file)
    if settings.worker_id is None:
        typer.echo("Worker-ID fehlt.", err=True)
        raise typer.Exit(2)
    try:
        registration = register_pdf(pdf, worker_id=settings.worker_id, source_id=source_id)
        if output_dir is not None:
            root = find_project_root()
            target_dir = ensure_output_outside_repository(output_dir, root)
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / "pdf_registration.json"
            if target.exists():
                typer.echo("Ausgabedatei existiert bereits; nichts überschrieben.", err=True)
                raise typer.Exit(3)
            target.write_text(registration.model_dump_json(indent=2) + "\n", encoding="utf-8")
            typer.echo(f"Registrierung geschrieben: {target}")
        typer.echo(f"source_id: {registration.source_id}")
        typer.echo(f"SHA-256: {registration.sha256}")
        typer.echo(f"Seitenzahl: {registration.page_count or 'nicht verfügbar'}")
        typer.echo(f"Verschlüsselt: {'ja' if registration.encrypted else 'nein'}")
    except PdfRegistrationError as exc:
        typer.echo(f"PDF-Registrierung fehlgeschlagen: {exc}", err=True)
        raise typer.Exit(3) from exc
    except GeminiConfigurationError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from exc


@app.command("config-check")
def config_check(
    env_file: EnvFileOption = None,
    require_service_credentials: Annotated[
        bool,
        typer.Option(help="Treat missing external-service credentials as an error."),
    ] = False,
) -> None:
    """Validate local configuration without revealing credentials."""
    settings = _load_settings(env_file)
    typer.echo(f"AISurgeon-Version: {__version__}")
    typer.echo(f"Python-Version: {platform.python_version()}")
    typer.echo(f"Betriebssystem: {platform.system()}")
    typer.echo(f"Plattform: {platform.platform()}")
    typer.echo(f"Worker-ID: {settings.worker_id or 'fehlt'}")

    missing_local = any(
        value is None for value in (settings.worker_id, settings.data_root, settings.pdf_source_dir)
    )
    path_results = [
        _show_path("Datenwurzel", settings.data_root, writable=True),
        _show_path("PDF-Quellordner", settings.pdf_source_dir),
        _show_path("Runs-Verzeichnis", settings.runs_dir, writable=True),
        _show_path("Cache-Verzeichnis", settings.cache_dir, writable=True),
        _show_path("Export-Verzeichnis", settings.exports_dir, writable=True),
        _show_path("Log-Verzeichnis", settings.logs_dir, writable=True),
    ]

    missing_services = False
    for display_name, attribute in SERVICE_FIELDS:
        value = getattr(settings, attribute)
        typer.echo(f"{display_name}: {_present(value)}")
        missing_services = missing_services or value is None

    if missing_local:
        typer.echo("Erforderliche lokale Konfiguration fehlt.", err=True)
        raise typer.Exit(2)
    if not all(path_results):
        typer.echo("Mindestens ein lokaler Pfad ist ungültig.", err=True)
        raise typer.Exit(3)
    if require_service_credentials and missing_services:
        typer.echo("Erforderliche Service-Zugangsdaten fehlen.", err=True)
        raise typer.Exit(2)
    if missing_services:
        typer.echo("Warnung: Externe Service-Zugangsdaten sind teilweise nicht gesetzt.")
    typer.echo("Lokale Grundkonfiguration ist gültig.")


def _create_env_file(target: Path, settings: Settings) -> bool:
    if target.exists():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    content = (
        "# Local AISurgeon configuration. Never commit this file.\n"
        f"AISURGEON_WORKER_ID={settings.worker_id or ''}\n"
        f"AISURGEON_DATA_ROOT={settings.data_root or ''}\n"
        f"AISURGEON_PDF_SOURCE_DIR={settings.pdf_source_dir or ''}\n"
        "AISURGEON_RUNS_DIR=\nAISURGEON_CACHE_DIR=\n"
        "AISURGEON_EXPORTS_DIR=\nAISURGEON_LOGS_DIR=\n\n"
        "GEMINI_API_KEY=\nOPENAI_API_KEY=\nNCBI_API_KEY=\nNCBI_EMAIL=\nNCBI_TOOL=\n"
    )
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(content)
    return True


@app.command("setup-local")
def setup_local(
    worker_id: Annotated[str | None, typer.Option("--worker-id")] = None,
    data_root: Annotated[Path | None, typer.Option("--data-root")] = None,
    pdf_source_dir: Annotated[Path | None, typer.Option("--pdf-source-dir")] = None,
    env_file: EnvFileOption = None,
    create_env_if_missing: Annotated[
        bool,
        typer.Option(help="Create the explicitly named env file if absent; never overwrite it."),
    ] = False,
) -> None:
    """Create idempotent local output directories without touching PDFs."""
    if env_file is not None and not env_file.exists() and not create_env_if_missing:
        typer.echo(f"Konfigurationsfehler: env-Datei nicht gefunden: {env_file}", err=True)
        raise typer.Exit(2)

    settings = _load_settings(
        env_file if env_file is not None and env_file.exists() else None,
        worker_id=worker_id,
        data_root=data_root,
        pdf_source_dir=pdf_source_dir,
    )
    if settings.worker_id is None or settings.data_root is None or settings.pdf_source_dir is None:
        typer.echo("Worker-ID, Datenwurzel und PDF-Quellordner sind erforderlich.", err=True)
        raise typer.Exit(2)
    _, pdf_valid = _path_state(settings.pdf_source_dir)
    if not pdf_valid:
        typer.echo("PDF-Quellordner fehlt oder ist nicht lesbar.", err=True)
        raise typer.Exit(3)

    settings.data_root.mkdir(parents=True, exist_ok=True)
    if not os.access(settings.data_root, os.W_OK):
        typer.echo("Datenwurzel ist nicht beschreibbar.", err=True)
        raise typer.Exit(3)
    for output_dir in (
        settings.runs_dir,
        settings.cache_dir,
        settings.exports_dir,
        settings.logs_dir,
    ):
        if output_dir is None:
            raise typer.Exit(2)
        output_dir.mkdir(parents=True, exist_ok=True)

    created_env = False
    if create_env_if_missing:
        if env_file is None:
            typer.echo("--create-env-if-missing erfordert --env-file.", err=True)
            raise typer.Exit(2)
        created_env = _create_env_file(env_file, settings)

    typer.echo(f"Worker-ID: {settings.worker_id}")
    typer.echo(f"Datenwurzel: {settings.data_root}")
    typer.echo(f"PDF-Quellordner: {settings.pdf_source_dir}")
    typer.echo("Lokale Verzeichnisse sind bereit: runs, cache, exports, logs.")
    if env_file is not None:
        message = (
            "Lokale env-Datei wurde erstellt."
            if created_env
            else "Vorhandene env-Datei blieb unverändert."
        )
        typer.echo(message)


@app.command("gemini-document-map")
def gemini_document_map(
    pdf: Annotated[Path, typer.Option("--pdf", help="Local PDF for the document-map run.")],
    env_file: EnvFileOption = None,
    source_id: Annotated[str | None, typer.Option("--source-id")] = None,
    output_root: Annotated[Path | None, typer.Option("--output-root")] = None,
    allow_dirty: Annotated[bool, typer.Option("--allow-dirty")] = False,
    keep_remote_file: Annotated[bool, typer.Option("--keep-remote-file")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Plan or execute the isolated Gemini PDF document-map smoke test."""
    settings = _load_settings(env_file)
    if settings.worker_id is None:
        typer.echo("Worker-ID fehlt.", err=True)
        raise typer.Exit(2)
    selected_output = output_root or settings.runs_dir
    if selected_output is None:
        typer.echo("Output-Root beziehungsweise AISURGEON_RUNS_DIR fehlt.", err=True)
        raise typer.Exit(2)
    try:
        manifest, run_dir = run_document_map(
            pdf_path=pdf,
            worker_id=settings.worker_id,
            output_root=selected_output,
            api_key=settings.gemini_api_key,
            source_id=source_id,
            dry_run=dry_run,
            allow_dirty=allow_dirty,
            keep_remote_file=keep_remote_file,
        )
    except (GeminiConfigurationError, PdfRegistrationError) as exc:
        typer.echo(f"Gemini-Dokumentkarte nicht gestartet: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(f"Run-ID: {manifest.run_id}")
    typer.echo(f"Status: {manifest.status}")
    typer.echo(f"Run-Verzeichnis: {run_dir}")
    typer.echo(f"Modell: {manifest.model_id}")
    typer.echo(f"Thinking Level: {manifest.thinking_level}")
    typer.echo(f"Media Resolution: {manifest.media_resolution}")
    if dry_run:
        typer.echo("Dry Run: kein Upload und kein Gemini-API-Aufruf ausgeführt.")
    elif manifest.status != "succeeded":
        raise typer.Exit(4)


@app.command("extract-guideline")
def extract_guideline(
    pdf: Annotated[Path, typer.Option("--pdf")],
    source_id: Annotated[str, typer.Option("--source-id")],
    output_root: Annotated[Path, typer.Option("--output-root")],
    env_file: EnvFileOption = None,
    pages_per_job: Annotated[int, typer.Option("--pages-per-job", min=1)] = 8,
    overlap_pages: Annotated[int, typer.Option("--overlap-pages", min=0)] = 1,
    allow_dirty: Annotated[bool, typer.Option("--allow-dirty")] = False,
    keep_remote_file: Annotated[bool, typer.Option("--keep-remote-file")] = False,
    resume_run: Annotated[Path | None, typer.Option("--resume-run")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Plan or run canonical native extraction against one remote PDF."""
    settings = _load_settings(env_file)
    if settings.worker_id is None:
        typer.echo("Worker-ID fehlt.", err=True)
        raise typer.Exit(2)
    try:
        if dry_run:
            plan, run_dir = prepare_dry_run(
                pdf_path=pdf,
                worker_id=settings.worker_id,
                source_id=source_id,
                output_root=output_root,
                pages_per_job=pages_per_job,
                overlap_pages=overlap_pages,
            )
            status, job_count = plan["status"], len(plan["jobs"])
        else:
            if settings.gemini_api_key is None:
                typer.echo("GEMINI_API_KEY fehlt für den Live-Lauf.", err=True)
                raise typer.Exit(2)
            status, run_dir = run_live_extraction(
                pdf_path=pdf,
                worker_id=settings.worker_id,
                source_id=source_id,
                output_root=output_root,
                api_key=settings.gemini_api_key,
                pages_per_job=pages_per_job,
                overlap_pages=overlap_pages,
                allow_dirty=allow_dirty,
                keep_remote_file=keep_remote_file,
                resume_run_dir=resume_run,
            )
            job_count = -1
    except (GeminiConfigurationError, PdfRegistrationError, ValueError, FileExistsError) as exc:
        typer.echo(f"Extraktionsplanung fehlgeschlagen: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(f"Status: {status}")
    if job_count >= 0:
        typer.echo(f"Geplante Jobs: {job_count}")
    typer.echo(f"Run-Verzeichnis: {run_dir}")
    if dry_run:
        typer.echo("Dry Run: kein Upload und kein Gemini-API-Aufruf ausgeführt.")


@app.command("generate-pubmed-searches")
def generate_pubmed_searches(
    input_run: Annotated[Path, typer.Option("--input-run")],
    output_root: Annotated[Path, typer.Option("--output-root")],
    env_file: EnvFileOption = None,
    start_date: Annotated[
        str, typer.Option("--start-date", help="Inclusive publication start date (YYYY-MM-DD).")
    ] = "2023-01-01",
    end_date: Annotated[
        str | None,
        typer.Option("--end-date", help="Inclusive publication end date; defaults to today."),
    ] = None,
    resume_run: Annotated[Path | None, typer.Option("--resume-run")] = None,
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit",
            min=1,
            help=(
                "Process only the first N chronological FormalItems. Produces an incomplete "
                "technical run that fetch-pubmed refuses."
            ),
        ),
    ] = None,
) -> None:
    """Generate GPT semantic search blocks and deterministic PubMed queries."""
    settings = _load_settings(env_file)
    if not settings.worker_id or settings.openai_api_key is None:
        typer.echo("Worker-ID oder OPENAI_API_KEY fehlt.", err=True)
        raise typer.Exit(2)
    try:
        parsed_start = date.fromisoformat(start_date)
        parsed_end = date.fromisoformat(end_date) if end_date else date.today()
        run_dir = generate_searches(
            input_run=input_run,
            output_root=output_root,
            worker_id=settings.worker_id,
            api_key=settings.openai_api_key,
            start_date=parsed_start,
            end_date=parsed_end,
            resume_run=resume_run,
            limit=limit,
        )
    except (ValueError, FileExistsError, RuntimeError) as exc:
        typer.echo(f"Search-Generierung fehlgeschlagen: {exc}", err=True)
        raise typer.Exit(4) from exc
    typer.echo(f"Search-Run-Verzeichnis: {run_dir}")


@app.command("fetch-pubmed")
def fetch_pubmed_command(
    input_run: Annotated[Path, typer.Option("--input-run")],
    output_root: Annotated[Path, typer.Option("--output-root")],
    env_file: EnvFileOption = None,
    start_date: Annotated[
        str | None,
        typer.Option("--start-date", help="Assert the immutable query start date (YYYY-MM-DD)."),
    ] = None,
    end_date: Annotated[
        str | None,
        typer.Option("--end-date", help="Assert the immutable query end date (YYYY-MM-DD)."),
    ] = None,
    resume_run: Annotated[Path | None, typer.Option("--resume-run")] = None,
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit",
            min=1,
            help=(
                "Fetch at most N PMIDs per query. Produces a technical_limited run; its "
                "fingerprint cannot resume as a complete run."
            ),
        ),
    ] = None,
) -> None:
    """Fetch existing PubMed queries through official NCBI E-Utilities."""
    settings = _load_settings(env_file)
    if not settings.worker_id or settings.ncbi_email is None:
        typer.echo("Worker-ID oder NCBI_EMAIL fehlt.", err=True)
        raise typer.Exit(2)
    try:
        run_dir = fetch_pubmed(
            input_run=input_run,
            output_root=output_root,
            worker_id=settings.worker_id,
            email=settings.ncbi_email,
            api_key=settings.ncbi_api_key,
            tool=settings.ncbi_tool or "aisurgeon",
            resume_run=resume_run,
            limit=limit,
            expected_start_date=start_date,
            expected_end_date=end_date,
        )
    except (ValueError, FileExistsError, RuntimeError) as exc:
        typer.echo(f"PubMed-Abruf fehlgeschlagen: {exc}", err=True)
        raise typer.Exit(4) from exc
    typer.echo(f"PubMed-Fetch-Run-Verzeichnis: {run_dir}")


@app.command("map-pubmed-evidence")
def map_pubmed_evidence_command(
    extraction_run: Annotated[Path, typer.Option("--extraction-run")],
    search_run: Annotated[Path, typer.Option("--search-run")],
    fetch_run: Annotated[Path, typer.Option("--fetch-run")],
    output_root: Annotated[Path, typer.Option("--output-root")],
    env_file: EnvFileOption = None,
    resume_run: Annotated[Path | None, typer.Option("--resume-run")] = None,
    batch_size: Annotated[int, typer.Option("--batch-size", min=1)] = 10,
    limit: Annotated[int | None, typer.Option("--limit", min=1)] = None,
    retain_narrative_reviews_as_context: Annotated[
        bool, typer.Option("--retain-narrative-reviews-as-context")
    ] = False,
) -> None:
    """Map fetched PubMed abstracts to every canonical FormalItem."""
    settings = _load_settings(env_file)
    if not settings.worker_id or settings.openai_api_key is None:
        typer.echo("Worker-ID oder OPENAI_API_KEY fehlt.", err=True)
        raise typer.Exit(2)
    try:
        run_dir = map_pubmed_evidence(
            extraction_run=extraction_run,
            search_run=search_run,
            fetch_run=fetch_run,
            output_root=output_root,
            worker_id=settings.worker_id,
            api_key=settings.openai_api_key,
            resume_run=resume_run,
            batch_size=batch_size,
            limit=limit,
            retain_narrative_reviews=retain_narrative_reviews_as_context,
        )
    except (ValueError, FileExistsError, RuntimeError) as exc:
        typer.echo(f"PubMed-Mapping fehlgeschlagen: {exc}", err=True)
        raise typer.Exit(4) from exc
    typer.echo(f"Mapping-Run-Verzeichnis: {run_dir}")


@app.command("run-to-mapping")
def run_to_mapping_command(
    extraction_run: Annotated[Path, typer.Option("--extraction-run")],
    output_root: Annotated[Path, typer.Option("--output-root")],
    env_file: EnvFileOption = None,
    start_date: Annotated[str, typer.Option("--start-date")] = "2023-01-01",
    end_date: Annotated[str | None, typer.Option("--end-date")] = None,
    mapping_batch_size: Annotated[int, typer.Option("--mapping-batch-size", min=1)] = 10,
    resume_run: Annotated[Path | None, typer.Option("--resume-run")] = None,
    limit: Annotated[int | None, typer.Option("--limit", min=1)] = None,
    retain_narrative_reviews_as_context: Annotated[
        bool, typer.Option("--retain-narrative-reviews-as-context")
    ] = False,
) -> None:
    """Run GPT search planning, NCBI fetch, and final abstract mapping."""
    settings = _load_settings(env_file)
    if not settings.worker_id or settings.openai_api_key is None or settings.ncbi_email is None:
        typer.echo("Worker-ID, OPENAI_API_KEY oder NCBI_EMAIL fehlt.", err=True)
        raise typer.Exit(2)
    try:
        run_dir = orchestrate_to_mapping(
            extraction_run=extraction_run,
            output_root=output_root,
            worker_id=settings.worker_id,
            openai_api_key=settings.openai_api_key,
            ncbi_email=settings.ncbi_email,
            ncbi_api_key=settings.ncbi_api_key,
            ncbi_tool=settings.ncbi_tool or "aisurgeon",
            start_date=date.fromisoformat(start_date),
            end_date=date.fromisoformat(end_date) if end_date else date.today(),
            mapping_batch_size=mapping_batch_size,
            resume_run=resume_run,
            limit=limit,
            retain_narrative_reviews=retain_narrative_reviews_as_context,
        )
    except (ValueError, FileExistsError, RuntimeError) as exc:
        typer.echo(f"Orchestrierung fehlgeschlagen: {exc}", err=True)
        raise typer.Exit(4) from exc
    manifest = __import__("json").loads((run_dir / "orchestration_manifest.json").read_text())
    typer.echo(f"Orchestrierungs-Run: {run_dir}")
    for name, path in manifest["run_paths"].items():
        typer.echo(f"{name.capitalize()}-Run: {path}")
    typer.echo(f"Status: {manifest['status']}")


if __name__ == "__main__":
    app()
