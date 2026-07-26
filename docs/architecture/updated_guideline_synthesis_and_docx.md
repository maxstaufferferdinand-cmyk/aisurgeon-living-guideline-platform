# Updated guideline synthesis and DOCX

`aisurgeon build-updated-guideline` consumes immutable Extraction, Search, Fetch, and Mapping runs.
The phase creates one synthesis/DOCX run and never mutates its inputs. Resume requires an exact
fingerprint over all relevant input files, model configuration, prompt hash, schema versions,
reference-builder version, DOCX renderer version, Git commit, worker ID, and `--limit`.

The run first builds deterministic `item_evidence_packets.jsonl` records for every canonical
FormalItem in chronological order. Recommendations, statements, consensus statements,
expert-consensus items, and other formal item families are treated equally. Exact original item
texts, source-native item labels, comments, native grading, consensus, status fields, and old
inline reference numbers are copied from the Extraction run without paraphrase.

Only `include_direct` and `include_indirect` mappings can influence update decisions.
`context_only` may be passed as background but cannot alone justify `modified`.
`uncertain_review_required` is retained for review and cannot drive an automatic text change.
`relevance_score` is intentionally not used in packets, prompts, sorting, weighting, or decisions.
Articles are deduplicated per FormalItem by PMID and then DOI.

For each item, GPT-5.5 receives a structured packet and returns German structured synthesis:
new evidence, conclusion, one of `insufficient_new_evidence`, `unchanged`,
`rationale_updated`, or `modified`, a possibly updated item text, used PMIDs, uncertainty, and
review metadata. For `unchanged`, `rationale_updated`, and `insufficient_new_evidence`, Python
requires the updated item text to equal the exact original text. `modified` requires direct or
indirect used evidence. The internal AISurgeon evidence class is stored separately from all
source-native grades and never replaces them.

References use two deterministic, non-colliding namespaces. Original references from
`references.jsonl` keep their source guideline numbers and exact extracted wording; old inline
markers in original item text, comments, and rationales are not renumbered. Newly cited PubMed
articles use the separate update namespace `[N1]`, `[N2]`, ... in first occurrence order within
new evidence and conclusion text. Raw PMID mentions are not final in-text citations and are
rewritten to `[N...]` tokens during the deterministic reference rebuild. New articles are
deduplicated against original references first by PMID, then normalized DOI, then a conservative
normalized title match; confirmed duplicates cite the original number and do not receive an
`N` reference. Old and new citations remain in separate bracket groups when both are needed.
Unresolved original links become hard-fail findings for the final DOCX rebuild.

The DOCX is generated deterministically from structured blocks using a small OOXML writer. It
contains a title page, status/method text, a real Word TOC field based on heading styles, run
summary, all updated blocks, consolidated references, and an appendix. Unchanged items are shown
once. Modified items show old and proposed updated wording separately. Header, footer, page setup,
Arial styles, hanging reference indents, and visual item boxes are encoded directly in the DOCX.

DOCX QA validates required OOXML parts, heading/TOC presence, selected forbidden public markers,
header/footer parts, and render capability. If LibreOffice/soffice is available, the DOCX is
rendered to PDF and page images are produced with `pdftoppm` when present. If no renderer exists,
the run records a QA warning and still preserves the structurally validated DOCX and report.

`aisurgeon rebuild-guideline-references` performs a deterministic postprocessing rebuild from an
existing completed synthesis run. It does not call an LLM, does not mutate the synthesis run, and
writes a new Reference-/DOCX-Rebuild run containing `original_references_exact.jsonl`,
`new_references_numbered.jsonl`, namespace maps, citation occurrence logs, integrity reports, and
a new DOCX with the corrected dual literature structure.

`aisurgeon repair-and-rebuild-guideline-references-v2` uses Gemini only for targeted original
bibliography repair from physical page-slice PDFs. Each PageJob has a 30-minute request timeout
and up to eight attempts. Retryable failures are limited to transient transport/API conditions
(HTTP 408, 429, 500, 502, 503, 504, connection resets/aborts, connect/read timeouts, temporary DNS
or SDK service-unavailable/server errors, and temporary Google file-upload failures). HTTP 400,
401, 403, invalid credentials, invalid model/request/schema failures, and validated-response
schema errors are hard failures.

Backoff starts at 15 seconds and doubles to a 900-second cap, with ±25% jitter. HTTP 429 preserves
Retry-After and uses at least that delay. After three consecutive transient API failures across
PageJobs, the worker enters a 900-second global cooldown before the next API attempt; any
successful API call resets the counter. Every failed or successful API attempt is written
immediately to `page_jobs/page_XXXX/attempts.jsonl`; terminal failures also update
`last_error.json`, `job_manifest.json`, and `checkpoint.json` without storing secrets or
Authorization headers. Long waits emit progress at least every 60 seconds. Completed PageJobs keep
their checkpoints; failed or incomplete PageJobs can be retried later, and compatible saved
`raw_response.json` files may be imported and revalidated when only the retry configuration
changed. This prevents temporary Gemini service latency from discarding already completed
bibliography-page work.
