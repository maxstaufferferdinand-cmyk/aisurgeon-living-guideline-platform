"""One-command, resumable Search -> Fetch -> Mapping orchestration."""

import json
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path

from pydantic import SecretStr

from aisurgeon.extraction.canonical.outputs import write_json
from aisurgeon.mapping.pubmed import map_pubmed_evidence
from aisurgeon.search.pubmed.generation import (
    ensure_external_run_root,
    file_hash,
    generate_searches,
)
from aisurgeon.search.pubmed.ncbi import fetch_pubmed


def run_to_mapping(
    *,
    extraction_run: Path,
    output_root: Path,
    worker_id: str,
    openai_api_key: SecretStr,
    ncbi_email: SecretStr,
    ncbi_api_key: SecretStr | None,
    ncbi_tool: str,
    start_date: date,
    end_date: date,
    mapping_batch_size: int = 10,
    resume_run: Path | None = None,
    limit: int | None = None,
    retain_narrative_reviews: bool = False,
    search_runner: Callable[..., Path] = generate_searches,
    fetch_runner: Callable[..., Path] = fetch_pubmed,
    mapping_runner: Callable[..., Path] = map_pubmed_evidence,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> Path:
    fingerprint = {
        "extraction_run": str(extraction_run.resolve()),
        "extraction_manifest_sha256": file_hash(
            extraction_run.resolve() / "extraction_manifest.json"
        ),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "mapping_batch_size": mapping_batch_size,
        "limit": limit,
        "retain_narrative_reviews": retain_narrative_reviews,
    }
    if resume_run:
        run_dir = resume_run.resolve()
        if json.loads((run_dir / "orchestration_fingerprint.json").read_text()) != fingerprint:
            raise ValueError("Orchestration resume fingerprint does not match")
        state = json.loads((run_dir / "orchestration_state.json").read_text())
        if (run_dir / "orchestration_manifest.json").is_file():
            return run_dir
    else:
        root = ensure_external_run_root(output_root, extraction_run)
        run_dir = root / f"run-to-mapping-{now():%Y%m%dT%H%M%S%fZ}"
        run_dir.mkdir(parents=True, exist_ok=False)
        write_json(run_dir / "orchestration_fingerprint.json", fingerprint)
        state = {
            "phases": {
                "search": {"status": "pending"},
                "fetch": {"status": "pending"},
                "mapping": {"status": "pending"},
            }
        }
        write_json(run_dir / "orchestration_state.json", state)

    def save() -> None:
        path = run_dir / "orchestration_state.json"
        path.unlink(missing_ok=True)
        write_json(path, state)

    try:
        phase = state["phases"]["search"]
        if phase["status"] != "completed":
            phase["status"] = "running"
            save()
            phase["run_path"] = str(
                search_runner(
                    input_run=extraction_run,
                    output_root=output_root,
                    worker_id=worker_id,
                    api_key=openai_api_key,
                    start_date=start_date,
                    end_date=end_date,
                    limit=None,
                )
            )
            phase["status"] = "completed"
            save()
        search = Path(phase["run_path"])
        phase = state["phases"]["fetch"]
        if phase["status"] != "completed":
            phase["status"] = "running"
            save()
            phase["run_path"] = str(
                fetch_runner(
                    input_run=search,
                    output_root=output_root,
                    worker_id=worker_id,
                    email=ncbi_email,
                    api_key=ncbi_api_key,
                    tool=ncbi_tool,
                    limit=limit,
                )
            )
            phase["status"] = "completed"
            save()
        fetch = Path(phase["run_path"])
        phase = state["phases"]["mapping"]
        if phase["status"] != "completed":
            phase["status"] = "running"
            save()
            phase["run_path"] = str(
                mapping_runner(
                    extraction_run=extraction_run,
                    search_run=search,
                    fetch_run=fetch,
                    output_root=output_root,
                    worker_id=worker_id,
                    api_key=openai_api_key,
                    batch_size=mapping_batch_size,
                    limit=limit,
                    retain_narrative_reviews=retain_narrative_reviews,
                )
            )
            phase["status"] = "completed"
            save()
        mapping = Path(phase["run_path"])
        mapping_status = json.loads((mapping / "mapping_manifest.json").read_text())["status"]
        write_json(
            run_dir / "orchestration_manifest.json",
            {
                **fingerprint,
                "worker_id": worker_id,
                "created_at": now().isoformat(),
                "status": mapping_status,
                "phases": state["phases"],
                "run_paths": {
                    "extraction": str(extraction_run.resolve()),
                    "search": str(search.resolve()),
                    "fetch": str(fetch.resolve()),
                    "mapping": str(mapping.resolve()),
                },
            },
        )
    except Exception as exc:
        for phase in state["phases"].values():
            if phase["status"] == "running":
                phase["status"] = "failed"
                phase["error_type"] = type(exc).__name__
        save()
        raise
    return run_dir
