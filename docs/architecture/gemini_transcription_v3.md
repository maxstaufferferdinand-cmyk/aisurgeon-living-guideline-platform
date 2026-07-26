# Canonical Gemini Transcription V3

Gemini remains the only native PDF reader, but v3 narrows its canonical role to source
transcription. It visually reads physical PDF slices and returns minimal source-content JSON:
represented pages, reading order, generic visual blocks, exact visible text, tables, footnotes,
headers/footers, diagram text, unreadable regions, and uncertainty.

Gemini does not emit clinical semantic labels or technical constants. The model-returned schema
does not include `source_id`, `schema_version`, `prompt_version`, `job_id`, `chunk_id`, stable IDs,
publication years, recommendations, statements, comments, evidence grades, PubMed queries, or
clinical conclusions. Python injects all technical metadata after validation.

## Stages

1. Local PDF preflight writes `pdf_preflight.json` and `page_preflight.jsonl`.
2. Gemini layout scout writes non-semantic `extraction_scout.json`.
3. The bounded planner writes `extraction_plan.json`, `extraction_jobs.jsonl`, and review findings.
4. Each transcription job writes `slice.pdf`, `slice_page_map.json`, `request_prompt.txt`,
   `raw_response.json`, `validated_source_content.json`, `attempts.jsonl`, `checkpoint.json`, and
   `job_manifest.json`.
5. Merged outputs include `page_transcript.jsonl`, `canonical_transcript.json`,
   `canonical_transcript.md`, table/algorithm transcript JSONL files, uncertainty findings,
   coverage report, and manifest.

## Chunk Profiles

- `single_column_prose_verbatim`: default 5 primary pages, maximum 10.
- `two_column_prose_verbatim`: 3-5 primary pages.
- `dense_prose_verbatim`: 2-4 primary pages.
- `bibliography_verbatim`: exactly 1 primary page with adjacent context.
- `table_faithful`: one table or page-sized bounded job.
- `algorithm_faithful`: one algorithm/decision tree or page-sized bounded job.
- `mixed_layout_verbatim`: bounded mixed layout jobs.
- `scanned_page_verbatim`: 1-2 primary pages.

The planner never divides every PDF into ten equal chunks. Bibliography transcription is part of
the initial canonical transcription stage, not a late repair after DOCX generation.

## Reliability

The v3 retry policy distinguishes retryable provider failures such as 408, 429, 500, 502, 503, 504,
timeouts, connection resets, transient DNS, and transient upload/file processing from non-retryable
400, 401, 403, invalid key, permission, unavailable model, invalid schema, validation failure, and
fingerprint incompatibility. Attempt logs are secret-free.

Completeness gates reject a complete run when planned primary pages are missing, output is
implausibly short, reading order is non-monotonic, finish reason indicates truncation, output hits
the configured ceiling, bibliography pages are unresolved, or critical tables/algorithms lack
transcripts or findings.
