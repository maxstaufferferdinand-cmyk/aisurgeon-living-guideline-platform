# V3 Live Structure Wiring Failure Report

Date: 2026-07-26

## Invalid Structure Run

Invalid run:

`/mnt/c/living_guideline_platform/runs/structure-v3-20260726T172658325464Z-AWMF_021-026_NEUROENDOKRINE_TUMORE_2018`

Observed artifacts:

- `extraction_manifest.json` has `status=completed`.
- `publication_year=2023`.
- `publication_year_source="synthetic transcript fixture"`.
- `formal_items.jsonl` contains one fake recommendation derived from synthetic transcript text.
- `comments.jsonl` contains `"Synthetic exact original comment."`.
- `references.jsonl` contains `"[1] Synthetic source reference."`.
- No provider evidence, execution mode, response ID, usage metadata, or raw OpenAI response exists.

## Historical Root Cause

The invalid run was produced by the old unisolated production path:

- `src/aisurgeon/cli/app.py` exposed `structure-guideline` without explicit `--live` or
  `--dry-run`.
- `structure-guideline` called `run_semantic_structure(...)` without a live provider boundary.
- `src/aisurgeon/extraction/semantic_structure.py` defaulted
  `draft_factory: Callable[[dict[str, Any]], SemanticStructureDraft] = _default_draft`.
- `_default_draft()` emitted the synthetic values visible in the invalid run:
  `publication_year_source="synthetic transcript fixture"`,
  `"Synthetic exact original comment."`, and `"[1] Synthetic source reference."`.

No OpenAI SDK request occurred in that invalid run. The synthetic fixture entered the normal CLI
path because there was no explicit execution mode separating live, dry-run, and internal test
execution.

## Current Call Graph

Current `structure-guideline` call graph:

1. `src/aisurgeon/cli/app.py::structure_guideline`
2. `_execution_mode_from_flags(live=..., dry_run=..., require_env_file=...)`
3. `_load_settings(env_file)`
4. `run_semantic_structure(...)`
5. `_assert_transcription_compatible(...)`
6. `build_semantic_payload(...)`
7. `OpenAISemanticStructureProvider(api_key=...)`
8. `OpenAISemanticStructureProvider.create(...)`
9. `client.responses.parse(...)`
10. `SemanticStructureDraft` parsing
11. canonical object validation and output persistence

The installed OpenAI SDK is `openai 2.45.0`; `responses.parse` supports `text_format=...`.

## Remaining Live-Mode Risks Before This Fix

The previous mock-isolation patch added explicit modes and a live provider, but the live structure
boundary still needed hardening before feeding downstream stages:

- `SemanticStructureDraft` allows free-form `dict[str, Any]` records, so live malformed model
  output can fail only later during canonical object validation.
- no raw provider response artifact is written;
- provider evidence lacks request duration and response ID/usage checks strong enough for a live
  completion gate;
- a live limited transcription with no `--limit` can be marked `completed` even though the source
  run is `technical_limited`;
- `derive_pubmed_start_date(draft.publication_year)` is called unconditionally while writing the
  manifest, so a valid limited page-1 structure without visible publication year can fail during
  output persistence rather than recording a bounded review finding;
- no explicit no-formal-item finding is injected when a live limited page has no formal items;
- downstream search currently refuses `technical_limited` structure runs, so a bounded real mini
  pipeline needs an explicit audited limited-test compatibility path rather than pretending the
  subset is complete.

## Planned Fix

Keep mock fixtures available only in internal `mock_test`. For live structure:

- keep `responses.parse` with `model_id=gpt-5.5`, `reasoning_effort=high`, and structured output;
- persist provider evidence and raw parsed response;
- mark live outputs from `technical_limited` transcription as `technical_limited`;
- never emit `completed` for a limited source;
- add a real review finding when no formal item is present in a live limited page range;
- preserve strict synthetic-marker rejection;
- allow downstream limited test stages to run only when explicitly marked/audited, without claiming
  a full guideline completion.
