# Limited NET DOCX Output Defect Report

Date: 2026-07-27

## Inspected Run

Synthesis/DOCX run:

`/mnt/c/living_guideline_platform/runs/synthesis-20260726T200641923455Z-AWMF_021-026_NEUROENDOKRINE_TUMORE_2018-71bf85a0`

Input structure run:

`/mnt/c/living_guideline_platform/runs/structure-v3-20260726T195213837727Z-AWMF_021-026_NEUROENDOKRINE_TUMORE_2018`

## Current DOCX Structure

The current `word/document.xml` renders each item as:

1. heading
2. grade/consensus metadata
3. a single box labelled `Fortbestehendes Item (...)` or an update proposal
4. `Neue Evidenz`
5. `Schlussfolgerung`
6. `Entscheidung`

It does not render the required visible order:

1. `Alte Empfehlung`
2. `Neue Empfehlung`
3. `Alter Kommentar`
4. `Neue Evidenz`
5. `Schlussfolgerung`

Responsible function:

`src/aisurgeon/synthesis/updated_guideline.py::_docx_xml`

The Markdown output has the same structural problem in:

`src/aisurgeon/synthesis/updated_guideline.py::render_blocks_markdown`

## Exact Reason Original Comments Are Missing

The structure run contains 7 comments in `comments.jsonl`. Four are item-related comments with `related_original_number` values:

- `2.2`
- `2.2`
- `2.4`
- `2.5`

The five formal items in `formal_items.jsonl` have empty `linked_comment_ids`.

`build_item_evidence_packets()` calls `_comment_texts(item, comments_by_id)`, and `_comment_texts()` only reads `item["linked_comment_ids"]`. It does not fall back to `comments[*].related_original_number`, so every packet receives:

`exact_original_comments: []`

Therefore comments are already absent in:

- `item_evidence_packets.jsonl`
- `updated_guideline_blocks.jsonl`
- `updated_guideline_blocks.md`
- the final DOCX

Responsible functions:

- `src/aisurgeon/synthesis/updated_guideline.py::_comment_texts`
- `src/aisurgeon/synthesis/updated_guideline.py::build_item_evidence_packets`
- `src/aisurgeon/synthesis/updated_guideline.py::build_updated_blocks`

## Exact Reason Arial Is Not Robustly Enforced

`word/styles.xml` sets Arial only for `Normal` and `Title` `w:ascii`/`w:hAnsi`.

It does not set:

- document defaults via `w:docDefaults`
- `w:cs`
- `w:eastAsia`
- explicit run fonts in `word/document.xml`
- explicit fonts in `Heading1`, `Heading2`, `Heading3`
- explicit fonts in header/footer runs
- a `word/fontTable.xml`

The package contains no `fontTable.xml` or theme XML. `Calibri` was not present in the inspected XML parts, but the DOCX relies partly on inherited/default Word behavior and can display as Calibri in clients that do not honor the sparse styles consistently.

Responsible functions:

- `src/aisurgeon/synthesis/updated_guideline.py::_styles_xml`
- `src/aisurgeon/synthesis/updated_guideline.py::_w_p`
- `src/aisurgeon/synthesis/updated_guideline.py::write_docx`

## Whether OpenAI Rerun Is Required

OpenAI rerun is not required for this fix if previous item synthesis checkpoints or previous `updated_guideline_blocks.jsonl` are reused. The synthesis conclusions and decisions already exist for the five limited real NET formal items. The defect is deterministic data flow and deterministic DOCX rendering.

The rebuild should:

1. recompute item evidence packets from existing structure/search/fetch/mapping artifacts,
2. link comments by explicit `linked_comment_ids` and fallback `related_original_number`,
3. reuse prior synthesis outputs for each formal item,
4. rebuild blocks, Markdown, references, QA, and DOCX in a new run directory.

No Gemini or NCBI rerun is required. OpenAI is only required if prior synthesis outputs are unavailable or incompatible.
