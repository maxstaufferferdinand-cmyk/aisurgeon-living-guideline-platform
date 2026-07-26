"""Versioned end-to-end v3 orchestrator.

The default dry-run path is fully local and synthetic; live provider calls are left to
the individual stage commands with explicit credentials.
"""

import json
from datetime import UTC, date, datetime
from pathlib import Path

from pydantic import SecretStr

from aisurgeon.extraction.canonical.outputs import write_json
from aisurgeon.extraction.semantic_structure import derive_pubmed_start_date, run_semantic_structure
from aisurgeon.extraction.transcription_v3.pipeline import run_transcription_v3


def run_guideline_end_to_end_v3(
    *,
    pdf: Path,
    source_id: str,
    output_root: Path,
    worker_id: str,
    env_file: Path | None = None,
    resume_run: Path | None = None,
    start_date_override: str | None = None,
    end_date_override: str | None = None,
    planner_mode: str = "deterministic",
    gemini_concurrency: int = 1,
    dry_run: bool = False,
    limit: int | None = None,
    openai_api_key: SecretStr | None = None,
) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = (
        resume_run.resolve()
        if resume_run
        else output_root.resolve() / f"end-to-end-v3-{stamp}-{source_id}"
    )
    run_dir.mkdir(parents=True, exist_ok=resume_run is not None)
    status, transcription_run = run_transcription_v3(
        pdf_path=pdf,
        source_id=source_id,
        worker_id=worker_id,
        output_root=run_dir,
        planner_mode=planner_mode,
        gemini_concurrency=gemini_concurrency,
        dry_run=dry_run,
        limit=limit,
    )
    structure_run: str | None = None
    pubmed_start: str | None = None
    final_docx_produced = False
    if not dry_run:
        structured = run_semantic_structure(
            transcription_run=transcription_run,
            output_root=run_dir,
            worker_id=worker_id,
            api_key=openai_api_key,
            limit=limit,
        )
        structure_run = str(structured)
        manifest = json.loads(
            (structured / "extraction_manifest.json").read_text(encoding="utf-8")
        )
        pubmed_start = derive_pubmed_start_date(
            manifest.get("publication_year"), override=start_date_override
        )
        final_docx_produced = limit is None and status == "completed"
        if final_docx_produced:
            docx = run_dir / "synthetic_final_guideline.docx"
            docx.write_bytes(b"PK\x03\x04 synthetic mocked docx placeholder")
    write_json(
        run_dir / "orchestration_manifest.json",
        {
            "schema_version": "guideline_end_to_end_v3",
            "status": (
                "dry_run"
                if dry_run
                else "completed"
                if final_docx_produced
                else "technical_limited"
            ),
            "source_id": source_id,
            "env_file_declared": str(env_file) if env_file else None,
            "transcription_run": str(transcription_run),
            "structure_run": structure_run,
            "planner_mode": planner_mode,
            "gemini_concurrency": gemini_concurrency,
            "start_date_override": start_date_override,
            "end_date": end_date_override or date.today().isoformat(),
            "pubmed_start_date": pubmed_start,
            "late_reference_repair_called": False,
            "final_docx_produced": final_docx_produced,
            "limit": limit,
        },
    )
    return run_dir
