"""Minimal local setup and configuration CLI."""

import os
import platform
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from aisurgeon import __version__
from aisurgeon.config.settings import SERVICE_FIELDS, Settings

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
        value is None
        for value in (settings.worker_id, settings.data_root, settings.pdf_source_dir)
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
        "GEMINI_API_KEY=\nOPENAI_API_KEY=\nNCBI_API_KEY=\nNCBI_EMAIL=\n"
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


if __name__ == "__main__":
    app()
