# AISurgeon Living Guideline Platform

AISurgeon is a private, source-bound platform for reproducible, auditable, and clinically
reviewable living-guideline workflows.

## Current status

This repository now contains the v3 foundations for canonical Gemini transcription and downstream
living-guideline processing. Gemini is the source transcription layer only. GPT semantic
structuring runs later from the canonical transcript and emits the familiar extraction files for
Search, PubMed Fetch, Mapping, Synthesis, references, and DOCX. Existing GERD/EoE runs are immutable
legacy artifacts; v3 does not rewrite or reinterpret them.

## Binding architecture and roles

- Gemini is the sole canonical native PDF extractor in later phases.
- GPT/OpenAI operates only after a validated Source Lock for search planning, evidence mapping,
  and recommendation-level synthesis.
- Codex develops and reviews software; it is not an autonomous clinical author.
- Deterministic Python handles validation, IDs, versioning, references, audit data, and eventual
  document generation.
- The clinical owner is the methodological owner and final approval authority.

Read [AGENTS.md](AGENTS.md) and the binding
[Master Project Brief](docs/project/AISurgeon_Codex_Master_Project_Brief_v2.txt) before substantial
work.

## Prerequisites

- WSL 2 with Ubuntu
- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- Git
- Docker Desktop only for planned later phases

Do not install project packages globally and do not use `sudo` for project setup.

## Central development laptop

`LAPTOP-Q5JVQTL4` is the central development laptop. From a checked-out development branch:

```bash
uv sync
uv run aisurgeon --help
```

Create a local configuration from `.env.example` manually, or use the explicit setup command.
The real `.env` is local, ignored by Git, and must never contain shared or committed credentials.

Example local values for the central laptop:

```dotenv
AISURGEON_WORKER_ID=laptop-02
AISURGEON_DATA_ROOT=/mnt/c/living_guideline_platform
AISURGEON_PDF_SOURCE_DIR=/mnt/c/living_guideline_platform/source_pdfs
```

## Second laptop

`LAPTOP-JGER957R` receives reviewed code through Git. Its exact local filesystem paths are not yet
assumed. Use laptop-specific placeholders, for example:

```dotenv
AISURGEON_WORKER_ID=<second-laptop-worker-id>
AISURGEON_DATA_ROOT=<second-laptop-data-root>
AISURGEON_PDF_SOURCE_DIR=<second-laptop-pdf-source-dir>
```

Exact transfer routine:

```bash
git checkout main
git pull --ff-only
uv sync --frozen
```

Both laptops use identical code, commit, Python version, lockfile, schemas, prompts, and tests.
Only worker ID, local paths, credentials, cache, and assigned jobs differ.

## Local configuration

Copy `.env.example` to a local ignored `.env`, or explicitly create a new target:

```bash
uv run aisurgeon setup-local \
  --worker-id laptop-02 \
  --data-root /path/to/local/data \
  --pdf-source-dir /path/to/source-pdfs \
  --env-file /path/to/local/.env \
  --create-env-if-missing
```

The creation flag works only with an explicitly named target, creates it only if absent, leaves
existing env files untouched, and writes empty credential fields. `setup-local` validates the PDF
source and idempotently creates `runs`, `cache`, `exports`, and `logs`. It never moves or edits PDFs.

Never alter Python code to accommodate local paths.

Check an explicitly selected configuration:

```bash
uv run aisurgeon config-check --env-file /path/to/local/.env
uv run aisurgeon config-check --env-file /path/to/local/.env \
  --require-service-credentials
```

Without the optional strict flag, absent external-service credentials are warnings during this
scaffold phase. Credential values are never printed.

## Security and operations

- Never commit `.env`, keys, source PDFs, runs, logs, caches, or exports.
- Never print credentials in commands, logs, exceptions, reviews, or chat.
- Each laptop has its own local `.env` and worker ID.
- Assign production jobs explicitly; two workers must never process the same production job.
- Do not independently modify central framework files on the second laptop.
- External API calls are outside this scaffold phase.

## PDF registration and Gemini document-map smoke test

Register a local PDF without changing it or performing semantic extraction:

```bash
uv run aisurgeon pdf-register \
  --pdf /path/to/guideline.pdf \
  --env-file /path/to/local/.env \
  --output-dir /path/outside/the/repository/registration
```

Plan a document-map run without uploading the PDF or calling Gemini:

```bash
uv run aisurgeon gemini-document-map \
  --pdf /path/to/guideline.pdf \
  --env-file /path/to/local/.env \
  --output-root /path/outside/the/repository/runs \
  --dry-run
```

The later live command uses the same arguments without `--dry-run`. Live execution requires a
clean Git worktree unless `--allow-dirty` is explicitly supplied. Uploaded Gemini files are deleted
best effort by default; `--keep-remote-file` is an explicit exception for controlled debugging.
Never place PDFs, run outputs, or credentials in Git. See
[Gemini PDF smoke-test architecture](docs/architecture/gemini_pdf_smoke_test.md).

## PubMed search planning and retrieval

The canonical `formal_items.jsonl` is the search basis; statements and recommendations are treated
equally. GPT creates only semantic English SearchUnit cores. Python deterministically adds dates,
the Humans filter, evidence-type and guideline-exclusion filters. Search generation and NCBI fetch
are independent, fingerprinted, resumable runs:

```bash
uv run aisurgeon generate-pubmed-searches \
  --input-run "/path/to/immutable-extraction-run" \
  --output-root "/path/outside/repository/runs" \
  --env-file ".env" \
  --end-date 2026-07-14

uv run aisurgeon fetch-pubmed \
  --input-run "/path/to/generated-search-run" \
  --output-root "/path/outside/repository/runs" \
  --env-file ".env"
```

Both commands accept `--resume-run`; only an identical fingerprint is accepted. On generation,
`--limit N` processes only the first N chronological FormalItems and marks coverage incomplete;
that Search run cannot be fetched. On fetch, `--limit N` caps PMIDs per query and creates a
`technical_limited` run whose fingerprint cannot resume as a complete run. An explicitly selected
repository-root `.env` works as `--env-file ".env"`; it is never loaded automatically. The default
Search start date is January 1 of the extracted guideline publication year; `--start-date` is an
audited override and is included in the fingerprint. See
[PubMed search and fetch architecture](docs/architecture/pubmed_search_and_fetch.md).

## Complete run through final abstract mapping

From a completed canonical extraction, one command creates Search, Fetch, and final mapping runs:

```bash
uv run aisurgeon run-to-mapping \
  --extraction-run "/path/to/completed-GERD-extraction-run" \
  --output-root "/path/outside/repository/runs" \
  --env-file "/path/to/local/.env" \
  --start-date 2023-01-01 --end-date 2026-07-14 \
  --mapping-batch-size 10
```

Recommendations, statements, consensus statements, expert consensus, and other canonical
FormalItems are mapped without priority. Every candidate decision is retained, including
exclusions. See [mapping architecture](docs/architecture/pubmed_evidence_mapping.md).

## Updated guideline synthesis and DOCX

From completed Extraction, Search, Fetch, and Mapping runs, one command builds item evidence
packets, German syntheses, update decisions, consolidated references, QA artifacts, and the final
DOCX draft:

```bash
uv run aisurgeon build-updated-guideline \
  --extraction-run "/path/to/completed-GERD-extraction-run" \
  --search-run "/path/to/completed-search-run" \
  --fetch-run "/path/to/completed-fetch-run" \
  --mapping-run "/path/to/completed-mapping-run" \
  --output-root "/path/outside/repository/runs" \
  --env-file "/path/to/local/.env"
```

The command accepts `--resume-run` for identical fingerprints. `--limit N` is only for technical
tests and does not produce a complete final DOCX. The resulting DOCX is an automatically supported
scientific update draft, not a consented AWMF guideline. See
[synthesis and DOCX architecture](docs/architecture/updated_guideline_synthesis_and_docx.md).

If an existing synthesis run must be rebuilt without changing the German syntheses, use the
deterministic dual-namespace reference rebuild:

```bash
uv run aisurgeon rebuild-guideline-references \
  --synthesis-run "/path/to/completed-synthesis-run" \
  --output-root "/path/outside/repository/runs"
```

Original guideline references keep their original numeric markers (`[1]`, `[2]`, ...). New
PubMed references are cited as `[N1]`, `[N2]`, ... and raw PMID mentions are not retained as final
in-text citations.

## Planned, not implemented

Live-pilot validation, Source Lock, targeted repair, PostgreSQL, Redis, FastAPI, MCP, and
deployment infrastructure remain planned work after the pilot gates.

## Canonical transcription v3

Provider checks are explicit and secret-free. In Codex/test runs they are mocked and do not call
Gemini, OpenAI, or NCBI:

```bash
uv run aisurgeon provider-preflight --env-file /path/to/local/.env
```

Run source-only canonical transcription:

```bash
uv run aisurgeon transcribe-guideline \
  --pdf /path/to/guideline.pdf \
  --source-id <confirmed-source-id> \
  --output-root /path/outside/the/repository/runs \
  --env-file /path/to/local/.env \
  --planner-mode hybrid \
  --gemini-concurrency 1
```

Run semantic structuring from the transcript only:

```bash
uv run aisurgeon structure-guideline \
  --transcription-run /path/to/transcription-v3-run \
  --output-root /path/outside/the/repository/runs \
  --env-file /path/to/local/.env
```

Run the versioned v3 pipeline:

```bash
uv run aisurgeon run-guideline-end-to-end-v3 \
  --pdf /path/to/guideline.pdf \
  --source-id <confirmed-source-id> \
  --output-root /path/outside/the/repository/runs \
  --env-file /path/to/local/.env \
  --planner-mode hybrid \
  --gemini-concurrency 1
```

Gemini v3 output is limited to faithful visual source transcription with generic layout labels.
Python injects `source_id`, schema/prompt versions, job IDs, page maps, and stable IDs. GPT later
classifies recommendations, statements, comments, bibliography entries, tables, and algorithms from
the transcript without receiving the original PDF.

The normal v3 workflow does not use late Gemini reference repair. The existing repair commands are
legacy/manual recovery tools for historic runs.

For the later S2k neuroendocrine-tumour pilot, after real provider preflight outside Codex:

```bash
uv run aisurgeon run-guideline-end-to-end-v3 \
  --pdf /path/to/S2k-neuroendocrine-tumour-guideline.pdf \
  --source-id <confirmed-source-id> \
  --output-root /mnt/c/living_guideline_platform/runs \
  --env-file /path/to/local/.env \
  --planner-mode hybrid \
  --gemini-concurrency 1
```

## Canonical extraction dry run

Plan the canonical jobs without uploading a PDF or calling Gemini:

```bash
uv run aisurgeon extract-guideline \
  --pdf /path/to/synthetic-or-guideline.pdf \
  --source-id <confirmed-source-id> \
  --output-root /path/outside/the/repository/runs \
  --env-file /path/to/local/.env \
  --pages-per-job 8 --overlap-pages 1 --dry-run
```

See [canonical PDF extraction](docs/architecture/canonical_pdf_extraction.md). Canonical source
text is never summarized, paraphrased, or spelling-corrected. The GERD/EoE document is only the
planned first two-column live pilot; this README does not claim that live extraction succeeded.
