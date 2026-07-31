# Full NET Reference System Defect Report

Date: 2026-07-31

## Latest Full NET Output Inspected

- Synthesis run: `/mnt/c/living_guideline_platform/runs/synthesis-20260730T072238458729Z-AWMF_021-026_NEUROENDOKRINE_TUMORE_2018-71bf85a0`
- DOCX: `AISurgeon_Aktualisierte_Leitlinie_AWMF_021-026_NEUROENDOKRINE_TUMORE_2018_2026.docx`
- Blocks: `updated_guideline_blocks.jsonl`
- Manifest: `synthesis_manifest.json`
- Summary: `synthesis_summary.json`
- Reference outputs: `consolidated_references.jsonl`, `reference_number_map.json`, `reference_review_findings.jsonl`
- Structure run: `/mnt/c/living_guideline_platform/runs/structure-v3-20260729T200442327100Z-AWMF_021-026_NEUROENDOKRINE_TUMORE_2018`
- Fetch run: `/mnt/c/living_guideline_platform/runs/pubmed-fetch-20260729T231130643063Z-24329183`
- Mapping run: `/mnt/c/living_guideline_platform/runs/mapping-20260729T232034490024Z-AWMF_021-026_NEUROENDOKRINE_TUMORE_2018-8b077ac4`

## Raw PMID Defect

The DOCX still contains raw PubMed IDs in narrative text before the bibliography.
The extracted DOCX narrative has 71 raw 7- to 9-digit PMID-like tokens before
`Konsolidiertes Literaturverzeichnis`.

Examples:

- `([N13], 36440195, 40490785)`
- `([N31], 41167510)`
- `([N6], 34662810, 34773595; indirekt [N80])`
- `([N76], 31585816, 34352556, 35528006)`
- `(37270677)`, `(34472210)`, `(32050284)`

The user-observed PMIDs `30733049` and `40096872` are present in:

- `pubmed_articles.jsonl`
- `article_formal_item_mappings.jsonl`
- structured synthesis used-PMID fields
- generated text fields in blocks 6.4, 7.7, 7.8, 7.9, and 7.16

Root cause: `src/aisurgeon/synthesis/updated_guideline.py`,
`replace_raw_pmids_in_update_text()`, only replaces explicit `PMID:` patterns.
It does not scan citation contexts containing bare PMIDs, parenthesized PMIDs,
or mixed `[N...]` plus raw PMID groups.

## Original Reference Gap

The final DOCX construction did not drop original references that were present
in `structure/references.jsonl`: the final `consolidated_references.jsonl`
contains all 592 original references parsed by structure.

However, `structure/references.jsonl` itself has numeric gaps:

- missing 136-170
- missing 554-590

The canonical transcript contains bibliography text around both gaps. For
example, the transcript jumps from `[135]` on page 81 to `[171]` on page 82,
and from `[553]` on page 91 to `[591]` on page 92. This indicates that the
verified canonical transcript also contains a visible page-transition gap in
the bibliography text, not only a final DOCX writer omission.

The final reference builder must therefore preserve all parsed original
references and emit explicit final reference-resolution findings for missing
original bibliography numbers, instead of silently presenting the bibliography
as complete.

## Responsible Files And Functions

- `src/aisurgeon/synthesis/updated_guideline.py`
  - `consolidate_references()`: builds original and new reference registry.
  - `replace_raw_pmids_in_update_text()`: currently too narrow; misses bare PMIDs.
  - `render_blocks_markdown()`: appends citation strings after replacement.
  - `_docx_xml()` / `write_docx()`: render final narrative and bibliography.
  - `run_docx_qa()`: does not currently fail raw narrative PMIDs or unresolved citations.
- `tests/unit/test_updated_guideline_synthesis.py`
  - Existing tests cover explicit `PMID:` replacement but not mixed `[N], PMID`
    groups, bare parenthesized PMIDs, unresolved N citations, or missing original
    bibliography spans.

## Planned Fix

Add a deterministic final reference-resolution pass after synthesis and before
DOCX writing. It will:

1. Build a complete reference registry from structure references and fetched
   PubMed metadata.
2. Preserve all parsed original references as `[1]`, `[2]`, etc.
3. Assign `N` references to new PubMed articles used by synthesis/mapping.
4. Scan generated narrative fields for raw PubMed IDs in citation contexts.
5. Replace mixed citation groups such as `([N247], 30733049, 40096872)` with
   grouped N citations.
6. Preserve PMID metadata only inside bibliography entries.
7. Emit final reference registry, occurrence, findings, and resolution reports.
8. Extend DOCX QA so raw narrative PMIDs, unresolved N citations, uncited N
   references, mixed old/new groups, and missing cited original references are
   detected.

No Gemini, OpenAI, or NCBI rerun is required for this fix.
