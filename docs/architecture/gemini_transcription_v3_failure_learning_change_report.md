# Gemini Transcription V3 Failure-Learning Change Report

Date: 2026-07-26
Branch: `feature/gemini-transcription-v3`
Checkpoint before branch: `a0b9b08 Checkpoint Gemini retry diagnostics before transcription v3`

## A. Existing End-to-End Architecture

The current platform registers an immutable local PDF, uses Gemini as the canonical native PDF extractor, writes a Gemini document map, then uses Gemini again to extract semantically structured formal guideline items, comments, references, and visual-object inventories. Python validates, assigns stable IDs, merges overlaps, links comments/references, emits canonical extraction files, and then downstream GPT/NCBI stages operate on those extraction outputs.

Downstream phases are already separated and resumable:

- Search: GPT creates English SearchUnit cores from `formal_items.jsonl`; Python adds date, Humans, study-design, and guideline-exclusion filters.
- Fetch: NCBI ESearch/EFetch retrieves and deduplicates PMIDs with retry and caching.
- Mapping: Python screens study design and GPT maps eligible abstracts to formal items.
- Synthesis/DOCX: GPT synthesizes item-level German update blocks; Python rebuilds references and deterministic DOCX output.

Existing GERD/EoE run evidence:

- Extraction run `extract-20260714T163814613127Z-AWMF_021-013_GERD_EOE_2023-...` completed with review: 92 formal items, 89 comments, 109 references, 77 findings.
- Search run `search-20260714T203354710189Z-AWMF_021-013_GERD_EOE_2023-...` completed: 92 input formal items, 31 SearchUnits, 31 queries, fixed start date `2023-01-01`.
- Fetch run `pubmed-fetch-20260714T204034147024Z-9cac1846` completed: 31 queries, 448 unique PMIDs, 4 warning queries, 1 zero-hit query.
- Mapping run `mapping-20260715T154306908191Z-AWMF_021-013_GERD_EOE_2023-...` completed with review: 3685 candidate pairs, 1369 included mappings.
- Synthesis run `synthesis-20260716T160908717094Z-AWMF_021-013_GERD_EOE_2023-...` produced 92 blocks, 189 references, and the final DOCX path family.

## B. Exact Historical Gemini Request Paths

Historical commits show these Gemini paths:

1. `9cdfd91`: document-map smoke test used Gemini Files API upload plus Interactions API structured request (`client.interactions.create`) with `document_map_v1`.
2. `d99fea4`: canonical extraction added staged Gemini extraction from one uploaded PDF.
3. `7efc07b`: canonical extraction used resumable background Interactions, polling background operations.
4. `66d1bd5`: canonical extraction switched to stateless `models.generate_content` with file URI input.
5. `58154f1`: semantic extraction prompt/schema v2 treated all formal items as one chronological backbone.
6. `524ba55`: v2 audit metadata/fingerprints were finalized, separating document-map and extraction schema versions.
7. `98a00c5`: late Gemini reference repair used a targeted whole-bibliography GenerateContent request.
8. `6de6299` and `a0b9b08`: late reference repair v2 used physical page-slice PDFs, GenerateContent, one-page jobs, 30-minute timeout, checkpointed attempts, Retry-After, cooldown, and safer error logging.

## C. Background Interactions Versus GenerateContent History

The document-map smoke test started with Interactions. Canonical extraction briefly used background Interactions to support long-running requests, but this added orchestration complexity and fingerprint sensitivity. Commit `66d1bd5` moved canonical extraction to direct GenerateContent for native PDF extraction and kept the Files API for upload. Reference repair v2 confirmed GenerateContent plus physical PDF slices was the more inspectable and resumable operational pattern.

V3 therefore keeps GenerateContent for isolated transcription jobs and avoids reintroducing background Interactions into the default canonical transcription path.

## D. Complete Gemini Failure Timeline

- Early smoke/extraction runs included failed or incomplete extraction directories with only checkpoints/context, indicating provider or orchestration interruption before final outputs.
- The successful v2 extraction completed but under-extracted bibliography: 109 references against a document-map bibliography description of references 1-720.
- The first late repair run identified 589 missing references and requested repair over pages 52-71, but added only 25 references and failed with 134 final references.
- Reference repair v2 introduced page-level planning over bibliography pages 52-71, one primary page per job with context, physical slices, per-job prompts, attempts, checkpoints, and long retry configuration.
- Later retry diagnostics expanded provider failure capture but remained a downstream manual recovery workflow.

## E. Separation of Provider Errors From Local Orchestration Errors

Provider-side issues include HTTP 408/429/5xx, transient upload/file-processing errors, timeouts, connection resets, and provider capacity/quota events. Local orchestration issues include schema-version mismatches, incompatible fingerprints, unsafe output roots, dirty worktree gating, missing checkpoint compatibility, validation failures after a syntactically valid response, and incomplete coverage accepted too late.

The old `GeminiDocumentMapClient._classify` compressed many failures into generic German messages such as `Temporärer Gemini-Dienstfehler`, which was insufficient for postmortem diagnosis. Reference repair v2 improved this with attempt logs and safe status capture, but the normal extraction path still lacks equivalent granularity.

## F. Cause of the Incomplete Initial `references.jsonl`

The initial v2 canonical extraction asked Gemini to perform clinical semantic structuring and bibliography extraction in wide page windows. Completeness was judged mainly through schema-valid JSON and downstream link checks, not by deterministic bibliography continuity and source-page coverage. The extraction produced valid outputs but only 109 references while later requirements inferred 696 required references and the document map described references 1-720.

Root cause: semantic extraction and transcription were coupled, bibliography pages were chunked too coarsely for faithful full-bibliography transcription, and there was no hard gate requiring every bibliography number/page to be transcribed before Search/Fetch/Synthesis.

## G. Cause of the 25-Reference Partial Repair Response

The first late reference repair targeted the full bibliography page range after synthesis/rebuild failure. It asked Gemini to repair many missing references in one request and received a valid but partial response: added numbers 46-61, 82, 83, 86, 89, 90, and 121-124 only. The merge report still listed hundreds of missing references.

Root cause: the repair job was too broad and too late. A single structured response over many bibliography pages was treated as a recovery extraction, and valid JSON did not imply complete coverage. The repair also lacked one-primary-page job boundaries until v2.

## H. Cause of Schema-Version Validation Failures

Reference repair v2 explicitly warned Gemini that `schema_version` must be exactly `original_reference_repair_v2`, not `2.0.0`, `2.0`, `v2`, or `2`. This indicates earlier model responses could emit plausible but noncanonical version constants. The broader v2 extraction also asked Gemini to emit `source_id` and schema metadata, creating opportunities for model-generated constants to diverge from Python expectations.

Root cause: technical constants were exposed to the model-returned schema. V3 removes them from Gemini output and injects them only in Python.

## I. Cause of Fingerprint Mismatches

Existing checkpoints bind source/PDF hash, model config, prompt version/hash, schema versions, page-window settings, and run context. Commits `524ba55` and later changed audit metadata, schema separation, retry configuration, and prompt text. Old runs therefore cannot be safely resumed under changed config even when their raw content is useful.

Root cause: fingerprint compatibility correctly changed as the architecture changed. V3 must not reinterpret legacy fingerprints as v3-compatible.

## J. What Cannot Be Diagnosed Because Earlier Logging Was Insufficient

For early failures, the logs do not reliably show exact safe exception class, HTTP status, API status, request ID, finish reason, Retry-After, provider quota/capacity signal, or output-ceiling/truncation state. The successful v2 extraction manifest has aggregate token usage but not per-job finish reasons or response-ceiling diagnostics. This prevents definitive distinction between model omission, truncation, provider-side interruption, and prompt/schema overload for each incomplete page range.

## K. Lessons Learned

- Valid JSON is not extraction completeness.
- Bibliography extraction must happen during initial source acquisition, not after final DOCX failure.
- Gemini should transcribe source content, not classify clinical semantics.
- Python must own source IDs, schema versions, prompt versions, stable IDs, fingerprints, page maps, and coverage gates.
- Physical PDF slices produce clearer retry/resume boundaries than one remote whole-PDF semantic extraction.
- Page-level attempt logs and safe provider diagnostics are mandatory for long-running Gemini work.
- Search start dates must derive from the guideline publication year, not from a GERD-specific or current-year constant.

## L. Exact Planned V3 Architecture

V3 separates the pipeline into:

1. Provider preflight with mocked tests and secret-free status.
2. Deterministic local PDF preflight: hash, size, page count, version, encryption, dimensions, rotation, text-layer counts, density, blank pages, warnings.
3. Gemini technical layout scout: non-semantic regions only.
4. Bounded adaptive planner: deterministic baseline plus optional GPT bounded refinement over preflight/scout metadata only.
5. Gemini canonical transcription v3: physical slices, GenerateContent per job, minimal visual source-content schema, per-job checkpoints, raw response preservation, completeness gates, targeted in-stage repair.
6. GPT semantic structuring: consumes only canonical transcripts and deterministic metadata; emits existing downstream canonical files.
7. Publication-year-derived PubMed Search/Fetch/Mapping/Synthesis/DOCX using existing downstream contracts.

## M. Files and Modules Expected To Change

Expected additions:

- `src/aisurgeon/extraction/transcription_v3/*`
- `src/aisurgeon/extraction/pdf_preflight.py`
- `src/aisurgeon/extraction/provider_preflight.py`
- `src/aisurgeon/extraction/semantic_structure.py`
- `src/aisurgeon/orchestration/guideline_v3.py`
- v3 model configs, prompts, schemas, docs, tests.

Expected changes:

- `src/aisurgeon/cli/app.py`
- `src/aisurgeon/search/pubmed/generation.py`
- `README.md`
- `.env.example`
- architecture docs and tests.

Legacy reference-repair modules remain available but deprecated.

## N. Backward-Compatibility Plan

Existing GERD/EoE run artifacts remain immutable and are not rewritten. Legacy commands stay functional where practical. Legacy extraction fingerprints are not v3-compatible. V3 semantic structuring emits the same downstream filenames where practical so existing Search, Fetch, Mapping, Synthesis, references, and DOCX code can be reused with minimal changes.

## O. Migration Risks

- Synthetic tests can prove orchestration, schemas, IDs, retries, and gating but not real Gemini visual fidelity.
- GPT semantic structuring from transcripts may need prompt tuning after the next real pilot.
- Local text-layer density is only an approximation for visual complexity.
- Bibliography continuity repair is bounded; a badly scanned bibliography should hard-fail before PubMed.
- Existing downstream code assumes extraction manifests and files; v3 must preserve those names carefully.

## P. Tests and Acceptance Criteria

Acceptance criteria:

- Gemini source-content schema contains no clinical semantic labels and no model-generated technical constants.
- Python injects all technical metadata deterministically.
- Planner chunk profiles are bounded and bibliography uses one primary page per job.
- Every physical page is either processed or explicitly unresolved; no complete run with missing primary pages.
- Truncation, implausibly short output, and missing pages trigger split/retry or hard failure.
- Retry behavior distinguishes retryable and non-retryable provider failures and logs safe diagnostics.
- GPT structuring receives no raw PDF bytes or URI.
- Bibliography is parsed from the initial transcript, with bounded in-stage retranscription if continuity fails.
- Search defaults to January 1 of the extracted publication year and fingerprints explicit overrides.
- V3 orchestrator never calls late reference repair.
- Fully mocked synthetic end-to-end dry run reaches a valid DOCX; limited runs never claim final completeness.
