# End-to-End Guideline Pipeline V3

V3 separates source transcription from semantic interpretation.

1. `aisurgeon provider-preflight` checks Gemini, OpenAI, and NCBI readiness without logging secrets.
2. `aisurgeon transcribe-guideline` performs local PDF preflight, non-semantic layout scout, bounded
   planning, physical PDF slicing, canonical Gemini transcription, and deterministic completeness
   gates.
3. `aisurgeon structure-guideline` sends only the canonical transcript and deterministic metadata to
   GPT-5.5. GPT classifies formal items, comments, bibliography, tables, algorithms, document
   metadata, publication year, and review findings. It never receives raw PDF bytes or a PDF URI.
4. Search derives the default PubMed start date as January 1 of the extracted publication year.
   An explicit `--start-date` override is fingerprinted and audited.
5. Existing Search, NCBI Fetch, Mapping, German Synthesis, dual-namespace references, DOCX, and QA
   phases consume the structured v3 extraction files.

Legacy `extract-guideline`, `gemini-document-map`, and late reference repair commands remain
available for historical/manual recovery. The v3 default orchestrator does not call late reference
repair and does not treat old fingerprints as v3-compatible.

## Commands

```bash
uv run aisurgeon provider-preflight --env-file /path/to/local/.env

uv run aisurgeon transcribe-guideline \
  --pdf /path/to/guideline.pdf \
  --source-id <source-id> \
  --output-root /path/outside/repository/runs \
  --env-file /path/to/local/.env \
  --planner-mode hybrid \
  --gemini-concurrency 1

uv run aisurgeon structure-guideline \
  --transcription-run /path/to/transcription-v3-run \
  --output-root /path/outside/repository/runs \
  --env-file /path/to/local/.env

uv run aisurgeon run-guideline-end-to-end-v3 \
  --pdf /path/to/guideline.pdf \
  --source-id <source-id> \
  --output-root /path/outside/repository/runs \
  --env-file /path/to/local/.env \
  --planner-mode hybrid \
  --gemini-concurrency 1
```

Limited runs are technical only and must not emit or claim a complete final guideline DOCX.

## Next Pilot Template

For the later S2k neuroendocrine-tumour guideline pilot, use the generic live form only after
provider preflight has been run outside Codex:

```bash
uv run aisurgeon provider-preflight --env-file /path/to/local/.env

uv run aisurgeon run-guideline-end-to-end-v3 \
  --pdf /path/to/S2k-neuroendocrine-tumour-guideline.pdf \
  --source-id <confirmed-source-id> \
  --output-root /mnt/c/living_guideline_platform/runs \
  --env-file /path/to/local/.env \
  --planner-mode hybrid \
  --gemini-concurrency 1
```

Do not supply `--start-date` unless the publication year is missing/ambiguous and the override has
been audited. The normal PubMed window starts on January 1 of the publication year extracted from
that PDF and ends on the run date unless `--end-date` is explicitly supplied.
