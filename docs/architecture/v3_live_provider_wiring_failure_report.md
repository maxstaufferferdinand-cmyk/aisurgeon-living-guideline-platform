# V3 Live Provider Wiring Failure Report

Date: 2026-07-26

## Invalid Runs

The following NET v3 runs are invalid and must not be used as source truth:

- `/mnt/c/living_guideline_platform/runs/transcription-v3-20260726T171801339944Z-AWMF_021-026_NEUROENDOKRINE_TUMORE_2018-c9d1c6da`
- `/mnt/c/living_guideline_platform/runs/structure-v3-20260726T172658325464Z-AWMF_021-026_NEUROENDOKRINE_TUMORE_2018`

The transcription manifest reports `status=completed` with no `execution_mode`, no
`provider_backend`, no `provider_call_count`, no token usage, no request IDs, no finish reasons,
and no uploaded-file metadata. The transcript contains `"Synthetic source transcript for page N."`
for pages 1-99. `canonical_transcript.md` is 5,229 bytes, `page_transcript.jsonl` is 27,900 bytes,
and table/algorithm transcript files are empty.

The structure run reports `status=completed`, `publication_year=2023`, and
`publication_year_source="synthetic transcript fixture"`. It emits exactly one FormalItem, one
Comment, and one Reference, including `"Synthetic exact original comment."` and
`"[1] Synthetic source reference."`.

## Exact Call Graph and Root Cause

### CLI to Transcription

`src/aisurgeon/cli/app.py:384-417` defines `aisurgeon transcribe-guideline`. It has
`dry_run=False` by default, but it has no `--live` flag and no required execution-mode selection.
At `src/aisurgeon/cli/app.py:402-412`, it calls `run_transcription_v3(...)` without passing an API
key, provider factory, Gemini client, or live-mode selector.

`src/aisurgeon/extraction/transcription_v3/pipeline.py:226-239` defines `run_transcription_v3`.
The default argument at line 238 is:

`draft_factory: Callable[[TranscriptionJob], SourceContentDraft] = _mock_source_content`

That default is the production-path defect. In a non-dry-run CLI invocation, the runner proceeds to
`src/aisurgeon/extraction/transcription_v3/pipeline.py:292-301`, where `_write_job_artifacts(...)`
is called for every job using the default `_mock_source_content`.

`src/aisurgeon/extraction/transcription_v3/pipeline.py:94-108` defines `_mock_source_content` and
creates `"Synthetic source transcript for page {page}."` plus `"synthetic_monotonic"`. This exactly
matches the invalid NET transcription artifacts.

No code in `run_transcription_v3` calls `google.genai.Client`, `files.upload`, or
`models.generate_content`. Therefore no Gemini SDK request occurred for the invalid run.

### Scout Path

`src/aisurgeon/extraction/transcription_v3/pipeline.py:266-270` creates a local
`ExtractionScoutDraft` with `warnings=["mocked_scout_in_dry_or_test_run"]`. It does not call a
Gemini layout-scout client. This marker appears in the invalid transcription run.

### Transcription Status and Completeness

`src/aisurgeon/extraction/transcription_v3/completeness.py:18-20` classifies
`implausibly_short_output` as `warning`, not `error`. In the invalid run, all 34 findings are
warnings with `repair_required=true`.

`src/aisurgeon/extraction/transcription_v3/pipeline.py:157-158` sets `complete` by checking only
whether any finding has severity `error`. Because the 34 short-output findings are warnings, the
run becomes `status="completed"`.

`src/aisurgeon/extraction/transcription_v3/pipeline.py:193-201` records only planned and resolved
page numbers, not provider evidence or character-ratio evidence. A page is treated as resolved
because the synthetic block names that page, not because a real Gemini response exists.

### CLI to Structure

`src/aisurgeon/cli/app.py:420-445` defines `aisurgeon structure-guideline`. It has no `--live`, no
`--dry-run`, and no execution-mode selector. At `src/aisurgeon/cli/app.py:434-441`, it calls
`run_semantic_structure(...)` with an optional API key but no OpenAI client factory and no live
enforcement.

`src/aisurgeon/extraction/semantic_structure.py:121-130` defines `run_semantic_structure`; line
129 defaults `draft_factory` to `_default_draft`.

`src/aisurgeon/extraction/semantic_structure.py:69-110` defines `_default_draft`, which emits:

- `publication_year=2023` at line 79
- `publication_year_source="synthetic transcript fixture"` at line 80
- `"Synthetic exact original comment."` at line 94
- `"[1] Synthetic source reference."` at line 104

This exactly matches the invalid structure run. No code in `run_semantic_structure` calls
`openai.OpenAI`, `responses.create`, or `responses.parse`. Therefore no OpenAI SDK request occurred.

### End-to-End Orchestrator

`src/aisurgeon/orchestration/guideline_v3.py:1-5` explicitly states that the default path is local
and synthetic. `src/aisurgeon/orchestration/guideline_v3.py:41-50` calls `run_transcription_v3`
without a live provider. `src/aisurgeon/orchestration/guideline_v3.py:55-61` calls
`run_semantic_structure` without a live OpenAI client. Lines 70-72 can write
`synthetic_final_guideline.docx`. The v3 orchestrator is therefore explicitly mock-only in the
current code.

## Determinations

1. No Gemini SDK request occurred in the invalid transcription run. The v3 transcription runner has
   no Gemini client path.
2. No OpenAI SDK request occurred in the invalid structure run. The v3 structure runner has no
   OpenAI client path.
3. The live CLI selected synthetic output because no explicit execution mode existed and the
   production function defaults were mock factories.
4. Provider errors did not fall back to synthetic output because providers were never called.
5. `--env-file` was used only through Settings to make credentials available/present; the v3
   commands did not use those credentials to construct live provider clients.
6. Manifests reported `completed` because status semantics were based on local synthetic object
   presence, not provider evidence.
7. The 34 findings did not block completion because `implausibly_short_output` was classified as a
   warning and completion checked only for severity `error`.
8. Provider request evidence was absent because no provider boundary existed in v3 transcription or
   semantic structuring.
9. `provider-preflight` is mocked in normal CLI mode. `src/aisurgeon/cli/app.py:358-381` exposes no
   `--live` flag and `src/aisurgeon/extraction/provider_preflight.py:63-72` defaults to mocked mode
   when no injected `live_checker` exists.
10. The end-to-end v3 orchestrator is explicitly mock/synthetic and can write a synthetic DOCX.

## Required Fix

Introduce explicit execution modes `live`, `dry_run`, and `mock_test`. Normal CLI v3 commands must
fail unless the user supplies `--live` or `--dry-run`. Mock mode must be internal/test-only.

For live transcription, instantiate a real Google GenAI client from the explicitly supplied env
file, send each physical slice as `application/pdf` content or an uploaded PDF URI with the v3
prompt and minimal source schema, and persist provider evidence. A live run with zero provider
calls must fail.

For live semantic structuring, instantiate the real OpenAI client from the explicitly supplied env
file, call the Responses API with `gpt-5.5`, high reasoning effort, structured output, and transcript
input only. Synthetic markers and fixed publication-year fixtures must hard-fail in live mode.

Completeness must require provider evidence, non-empty source text for every nonblank primary page,
per-page character ratios, raw responses, and no critical findings before `completed` or
`completed_with_review`.
