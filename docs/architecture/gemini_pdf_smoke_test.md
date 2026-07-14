# Gemini PDF registration and document-map smoke test

## Scope

Gemini is the sole canonical native PDF extractor. This phase implements only deterministic local
PDF registration, temporary upload through the Gemini Files API, and one structured document-map
request through the Interactions API. It does not extract complete recommendations, comments, or
references and does not create a Source Lock.

`pypdf` is restricted to technical metadata: PDF validity, version, encryption state, and page
count. It is not a canonical semantic source.

## Versioned request configuration

The shared configuration is `config/models/gemini_document_map_v1.json`:

- provider: Google
- API surface: Interactions API
- model: `gemini-3.5-flash`
- thinking level: `medium`
- PDF media resolution: `high`
- no application-specified temperature, top-p, or top-k
- structured output using the versioned DocumentMap JSON Schema
- no automatic model fallback

The prompt and JSON Schema are independently versioned and hashed into every run manifest.

## Technical registration

Registration validates the extension and PDF signature, streams SHA-256, records file size and
technical PDF metadata, and generates a versioned deterministic `source_id` from the PDF hash when
the owner has not supplied one. The absolute local path is audit metadata only and never a
cross-device identity. The input PDF is never modified, moved, or renamed.

## Dry run

Dry run performs registration, loads and hashes model configuration, prompt, and schema, reads Git
metadata, and writes a local plan under an output root outside the repository. It never constructs
the Gemini client, uploads a file, or performs an API request.

```bash
uv run aisurgeon gemini-document-map \
  --pdf /path/to/guideline.pdf \
  --env-file /path/to/local/.env \
  --output-root /path/to/local/runs \
  --dry-run
```

## Later live run

A future human-authorized pilot uses the same command without `--dry-run`. Live runs require a
clean worktree by default. The PDF is uploaded through the Files API, passed to the Interactions API
as a document with high media resolution, and the returned JSON is strictly validated. The local
page count is compared with the declared count. The temporary remote file is deleted best effort
after success or failure unless `--keep-remote-file` is explicitly requested.

Placeholder pilot command:

```bash
uv run aisurgeon gemini-document-map \
  --pdf /confirmed/local/path/to/pilot-guideline.pdf \
  --env-file /confirmed/local/path/to/.env \
  --output-root /confirmed/local/path/to/runs
```

This command has not been executed or clinically validated in this phase.

## Audit outputs

Each run directory contains registration, manifest, configuration and prompt snapshots, structured
response and validation artifacts when applicable, remote-file metadata, and a minimal log. Run
directories must remain outside Git. Manifests contain no API keys, authentication headers, or
credential fragments.

## Validation philosophy

The smoke test distinguishes technical unusability from reviewable source uncertainty. A missing
or technically unreadable PDF, an unparseable structured response, a mismatched `source_id`, a
missing live API key, unsafe output handling, secret leakage, or the complete absence of a usable
clinical main-body structure is blocking. Page-count differences, overlapping or uncertain
document regions, incomplete formal item types, grading or consensus gaps, unclear comment
boundaries, uncertain visual-object classification, unresolved references, and smaller metadata
gaps are warnings with `review_required=true`. They do not by themselves stop extraction.

The document map remains a narrow orientation layer: it locates the clinical main body and
separates it from front matter, contents, summaries, “what is new”, bibliography, and appendices;
it identifies formal patterns and inventories original visual objects. It is not an extraction
coverage gate.

## Next canonical extraction step

The next development phase—not implemented here—will produce these source-bound outputs:

- `recommendations.jsonl`
- `statements.jsonl`
- `comments.jsonl`
- `references.jsonl`
- `tables.jsonl`
- `algorithms.jsonl`

Its highest priorities are word-for-word `exact_original_text`, correct separation of
recommendations, statements, and comments, correct comment-to-item links, preservation of native
grading and consensus metadata, reference links, and original-page anchors. Recommendations and
statements must never be summarized or paraphrased. Local uncertainty is retained as review
metadata for later targeted checking or repair rather than silently changing source content.

## Security boundaries

- Never commit or print keys, PDFs, run outputs, or local env files.
- Env files are loaded only when explicitly named.
- Remote files are temporary transport objects, never AISurgeon source records.
- No automatic model substitution is allowed.
- Tests fully mock Gemini and need no network.
- This phase performs no OpenAI, PubMed, database, DOCX, or clinical processing.
