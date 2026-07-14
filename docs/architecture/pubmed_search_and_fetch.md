# GPT search planning and NCBI fetch

This workflow begins from an immutable canonical extraction run. Its authoritative search basis is
`formal_items.jsonl`; recommendations, statements, consensus statements, expert-consensus items,
and literature-relevant other formal items have equal standing. Comments and the document map
provide context but never replace exact formal-item text.

## Search planning

`aisurgeon generate-pubmed-searches` uses the centrally configured OpenAI Responses API model
`gpt-5.5` with high reasoning effort. GPT creates only semantic English search cores: coherent
multi-item SearchUnits, concepts, synonyms, abbreviations, MeSH candidates, Title/Abstract
candidates, and Boolean grouping. It cannot set stable IDs, dates, Humans filters, publication
types, exclusions, pagination, API parameters, or rate limits. Python creates stable IDs and adds
these versioned filters deterministically.

The evidence filter always includes randomized controlled trials, meta-analyses, and systematic
reviews. It also retains clinical trials and observational, comparative, evaluation, and validation
studies so diagnostic, prognostic, safety, and rare-disease questions are not inadvertently reduced
to RCT-only evidence. Guidelines and Practice Guidelines remain excluded as new primary evidence.

`formal_item_search_coverage.jsonl` contains one row per FormalItem. Missing coverage is a hard
failure. A non-literature-relevant item remains visible with an explicit exclusion reason.
`pubmed_queries.jsonl` keeps `query_core` separate from every technical filter.

## NCBI retrieval

`aisurgeon fetch-pubmed` consumes an existing Search run, so retrieval can be resumed without
another GPT call. It uses official ESearch and EFetch E-Utilities, result pagination, bounded retry
for network/429/5xx failures, `Retry-After`, conservative throttling, local response caching, and
query checkpoints. XML parsing supports multipart abstracts, collective authors, missing abstracts,
and available date variants.

PMIDs are globally deduplicated. `pubmed_query_hits.jsonl` retains every query hit, while each
unique article carries all contributing query IDs, SearchUnit IDs, and FormalItem IDs. Search and
fetch use separate run directories and exact fingerprints; incompatible resume is rejected.
Manifests contain only credential presence, never credential values.

`generate-pubmed-searches --limit N` processes the first N canonical FormalItems in chronological
order. Its manifest is `technical_limited`, `coverage_complete=false`, and `fetch-pubmed` refuses
it; it is never a completed Gate-2 result. `fetch-pubmed --limit N` means at most N PMIDs per query.
The Fetch manifest is `technical_limited`, and the limit is part of its fingerprint so it cannot
later resume as a complete fetch. Fetch `--start-date` and `--end-date` are assertions against the
already immutable query interval and never rewrite a query.

Live E-Utility requests use POST so the configured contact email and optional API key are not placed
in URLs; the transport also switches generic long parameter sets to POST. Default throttling is
conservative for the official limits: approximately three requests per second without an API key
and fewer than ten per second with one.

This phase performs no evidence grading, article-to-item relevance mapping, synthesis, update
decision, or guideline rewriting.
