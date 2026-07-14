# Canonical native PDF extraction

Gemini is the sole canonical native PDF extractor. AISurgeon registers one immutable local PDF,
uploads it once, obtains or loads its document map, and processes controlled page windows against
that same remote file. This is staged extraction, not an independent duplicate extraction. The
default clinical window has eight primary pages and one context page on either side. An object is
new only in the job containing its formal beginning; overlap pages provide context only.

The first planned live pilot is `AWMF_021-013_GERD_EOE_2023`, a two-column guideline selected to
test reading-order fidelity. No live success is asserted here. Source recommendations, statements,
comments, references, and unassigned clinical blocks must remain verbatim: no summarization,
paraphrase, spelling correction, or invented native metadata. Python alone assigns stable IDs,
deduplicates exact overlap copies, resolves references, and records conflicts.

Comments are linked first by explicit original number, then by formal block, chapter/page context,
and an unambiguous preceding item. Ambiguity creates a review finding while preserving the comment.
Simple reference ranges are expanded deterministically; unresolved numbers are warnings and do not
stop the workflow. Tables, algorithms, and decision trees are inventories of original objects, not
recreations.

Runs write JSONL source objects, unresolved links, a secret-free manifest and summary, plus
`review_findings.jsonl` and a filterable `review_findings.xlsx`. Completed checkpoints are reused
on resume; failed or absent windows alone are retried. Small uncertainty yields
`completed_with_review`, while hard failure is reserved for unreadable PDFs, unparseable Gemini
responses, source mismatch, absent usable clinical structure, secret leakage, or dangerous output
overwrite. Uploaded files are deleted best effort by default and there is no model fallback.

Dry run registers the PDF and plans windows and outputs without upload or network access. A later
live pilot will use a placeholder command such as:

```bash
uv run aisurgeon extract-guideline --pdf /path/to/guideline.pdf \
  --source-id AWMF_021-013_GERD_EOE_2023 --output-root /path/outside/repository/runs \
  --env-file /path/to/local/.env
```

PubMed retrieval, evidence mapping, update decisions, GPT synthesis, Source Lock, targeted repair,
and DOCX generation are not part of this phase.

## Formal-item backbone version 2

`formal_items.jsonl` is the sole chronological canonical master. Recommendations, statements,
consensus statements, expert-consensus/EK items, Good Clinical Practice items, and other native
formal types enter the same merge without priority. Python preserves `item_type_raw`, assigns the
normalized family and the cross-family `sequence_number` after merge, and derives recommendation,
statement, and expert-consensus views from the master.

New runs use `gemini_formal_items_comments_v2` and `canonical_extraction_v2`. Their directory name
contains UTC time, source ID, PDF hash prefix, prompt version/hash, and schema version. A checkpoint
is reusable only when source/PDF, model and model configuration, prompt version/hash, schema, and
window settings exactly match the run context. Resume requires an explicit compatible run path;
older runs without that context cannot be resumed.
