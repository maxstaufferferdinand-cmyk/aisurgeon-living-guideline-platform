# Gemini V3 503 Capacity Resilience Report

Date: 2026-07-28

## Current Failure

The full NET transcription run at
`/mnt/c/living_guideline_platform/runs/transcription-v3-20260727T181947817456Z-AWMF_021-026_NEUROENDOKRINE_TUMORE_2018-c9d1c6da`
registered the 99-page PDF and planned 50 transcription jobs. Jobs
`tx3-0001-p0001-0003` through `tx3-0004-p0010-0012` completed with raw Gemini
responses and checkpoints. Job `tx3-0005-p0013-0015` exhausted six live Gemini
attempts with HTTP 503 / API status `UNAVAILABLE` and no `Retry-After` header.

The diagnostic one-page run at
`/mnt/c/living_guideline_platform/runs/transcription-v3-20260727T190601272507Z-AWMF_021-026_NEUROENDOKRINE_TUMORE_2018-c9d1c6da`
failed the one-page job `tx3-0001-p0013-0013` the same way: six attempts, all
HTTP 503 / `UNAVAILABLE`, no `Retry-After`.

Safe provider message:

```text
This model is currently experiencing high demand. Spikes in demand are usually temporary. Please try again later.
```

Provider preflight had passed immediately before these failures, including
Gemini authentication, model access, text generation, structured output, PDF
input, and upload/inline checks. The current blocker is therefore provider
capacity on one source region, not key validity, model access, schema
construction, or local PDF request construction.

## Official Guidance Consulted

- Gemini troubleshooting: retry `429 RESOURCE_EXHAUSTED` and `503 UNAVAILABLE`
  with exponential backoff and jitter; do not retry request/client defects such
  as `400` or `403`.
- Gemini rate limits: rate limits are project-scoped and can return
  `429 RESOURCE_EXHAUSTED`; mitigation is waiting, reducing expensive
  requests, or increasing quota when persistent.
- Gemini document processing: PDF inputs are supported for GenerateContent and
  remain the canonical path for this extractor.
- Gemini Batch API: viable future asynchronous option for large workloads, but
  it is a separate run model with delayed result retrieval and is not required
  to fix the current GenerateContent queue behavior.
- Python GenAI SDK docs: SDK exposes API exceptions with safe status/code
  fields and Batch API helpers; the SDK itself also has transient retry logic.
- Priority/Flex inference docs: priority may improve scheduling where enabled
  by the project/tier; flex is lower-priority and not a reliability fallback.
  Neither should replace the current GenerateContent live path in this fix.

## Exact Current Retry Configuration

`config/models/gemini_source_transcription_v3.json` currently configures:

- `request_timeout_seconds`: 1800
- `max_attempts`: 6
- `retry_initial_delay_seconds`: 15
- `retry_backoff_multiplier`: 2.0
- `retry_max_delay_seconds`: 900
- `retry_jitter_fraction`: 0.25
- `global_cooldown_after_consecutive_transient_failures`: 3
- `global_cooldown_seconds`: 900
- `default_concurrency`: 1
- `max_concurrency`: 4

The configured global cooldown keys are not used by the v3 transcription
scheduler today.

## Exact Stage That Failed

The failure occurs inside `GeminiTranscriptionProvider.transcribe()` in
`src/aisurgeon/extraction/transcription_v3/pipeline.py`, after
`_write_job_artifacts()` creates a physical slice and sends it through
`_generate_inline_pdf()` with `types.Part.from_bytes(..., mime_type="application/pdf")`.

The outer run aborts in `run_transcription_v3()` because jobs are executed by a
single list comprehension. A raised exception from one exhausted PageJob stops
the full run immediately.

## Queue Behavior

The run stopped immediately after job `tx3-0005-p0013-0015` exhausted retries.
The remaining planned jobs for pages 16-99 were not attempted.

The current implementation can skip already completed checkpoints on resume,
but it cannot mark transient jobs as deferred, continue later independent jobs,
or revisit deferred jobs after a cooldown. It also overwrites failed
`attempts.jsonl` for incomplete jobs on retry and does not write a
`last_error.json` artifact.

## Retry-After and SDK Retry Interaction

No observed 503 attempt had a `Retry-After` value. The local classifier can
read `Retry-After` from exception headers for 429, but the current job loop
does not emit long-wait progress.

The Google GenAI SDK has its own default transient retry behavior for timeouts,
network errors, 429, and 5xx. AISurgeon also wraps calls with its own retry
loop. The observed six AISurgeon attempts therefore represent outer attempts;
each may already include SDK-level transient retries.

## Chunk Size and Media Resolution

The failing full-run job spans pages 13-15. A separate one-page page-13
diagnostic also failed with 503, so splitting is necessary but not sufficient
for this specific source page. The code can safely split multi-page transient
failures into one-page jobs because physical slices are independent. For
one-page jobs that use high media resolution, a bounded retry with medium
media resolution is acceptable when the output passes completeness checks and
is review-marked.

## Batch, Priority, and Flex

Batch API is viable as a future execution mode for large asynchronous
transcription, but implementing it now would add a second provider execution
model, polling, result materialization, and compatibility rules. It is not
required to fix the current GenerateContent scheduler defect.

Priority inference may be viable only if the configured project and installed
SDK expose the required `service_tier` setting for the current GenerateContent
path. Flex inference is lower-priority, can be interrupted by capacity, and is
not suitable as a resilience upgrade for this production extraction path.

## Required Changes Now

1. Extend retry classification so 429, 503, other 5xx/timeouts, and
   nonretryable 400/401/403 are recorded distinctly.
2. Increase the full-run retry/cooldown defaults for long-latency guideline
   extraction.
3. Add long-wait progress output for per-call retries and global cooldowns.
4. Preserve previous attempt logs when retrying or resuming an incomplete job.
5. Write `last_error.json` and non-completed checkpoints for transient and
   nonretryable job failures.
6. Replace the fail-fast list comprehension with a queue that defers transient
   jobs, continues later independent jobs, revisits deferred work, and fails
   only after the run-wide defer budget is exhausted.
7. Split transient-failed multi-page jobs into one-page jobs and permit a
   bounded high-to-medium media-resolution retry for one-page high-resolution
   jobs.
8. Keep completed checkpoints immutable and fingerprint-compatible for resume.
9. Ensure no run can write a completed manifest while any primary page remains
   unresolved.
