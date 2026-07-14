# PubMed evidence mapping

`aisurgeon map-pubmed-evidence` consumes immutable completed extraction, Search, and Fetch runs.
Python follows each PubMed hit through `query_id`, `search_unit_id`, and linked FormalItem IDs and
creates one deterministic `(pmid, formal_item_id)` candidate. All formal families are equal and
`formal_items.jsonl` remains unchanged.

GPT-5.5 screens one FormalItem at a time in configurable article batches with high reasoning and a
strict response schema. Python owns all technical IDs, requires exactly one decision per candidate,
rejects unknown PMIDs and non-verbatim abstract passages, and writes included and excluded
decisions. This phase performs neither GRADE nor synthesis.

Before batching, Python classifies PubMed publication types. Only randomized controlled trials,
meta-analyses, and systematic reviews reach GPT. Other designs receive a deterministic
`exclude_wrong_study_design` screening record and incur no OpenAI cost. Narrative reviews are
excluded by default; `--retain-narrative-reviews-as-context` retains them deterministically as
`context_only`, never as primary evidence.

Runs contain candidates, complete screening, included mappings, an evidence index, per-item
coverage, review JSONL/XLSX, summary, manifest, raw structured responses, and completed batch
checkpoints. Resume requires an exact fingerprint match. `--limit` is only a technical run and
cannot assert complete mapping coverage.

`aisurgeon run-to-mapping` executes Search planning, PubMed fetch, and mapping, recording each child
path and phase status. Resume skips recorded completed phases and rejects a changed fingerprint.
