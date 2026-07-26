# Gemini Structured Output Live Failure Report

Date: 2026-07-26

## Original Observed Error

```text
ClientError: 400 INVALID_ARGUMENT. {'error': {'code': 400, 'message':
'Invalid JSON payload received. Unknown name "additional_properties" at
\'generation_config.response_schema\': Cannot find field.\nInvalid JSON
payload received. Unknown name "additional_properties" at
\'generation_config.response_schema.properties[1].value.items\': Cannot find
field.', 'status': 'INVALID_ARGUMENT', 'details': [{'@type':
'type.googleapis.com/google.rpc.BadRequest', 'fieldViolations': [{'field':
'generation_config.response_schema', 'description': 'Invalid JSON payload
received. Unknown name "additional_properties" at
\'generation_config.response_schema\': Cannot find field.'}, {'field':
'generation_config.response_schema.properties[1].value.items',
'description': 'Invalid JSON payload received. Unknown name
"additional_properties" at
\'generation_config.response_schema.properties[1].value.items\': Cannot find
field.'}]}]}}
```

## Root Cause

`GeminiTranscriptionProvider._config` passed the strict local Pydantic model class
directly as `GenerateContentConfig(response_schema=schema_model)`.

Both `ExtractionScoutDraft` and `SourceContentDraft` inherit from `StrictModel`, which uses
`ConfigDict(extra="forbid")`. Their local Pydantic JSON Schema therefore contains
`additionalProperties: false` at the object root and in nested child models:

- `ExtractionScoutDraft.additionalProperties`
- `ExtractionScoutDraft.$defs.ExtractionScoutRegion.additionalProperties`
- `SourceContentDraft.additionalProperties`
- `SourceContentDraft.$defs.VisualBlock.additionalProperties`

The installed `google-genai 2.11.0` SDK accepts `responseSchema` and `responseJsonSchema` in
`GenerateContentConfig`. Passing the Pydantic model through `response_schema` caused the SDK/API
request path to send an unsupported `additional_properties` field in
`generation_config.response_schema`, including in nested array item schemas. The API rejected the
request with HTTP 400 `INVALID_ARGUMENT` before the technical layout scout completed.

## Fix

Strict local Pydantic models remain unchanged. The Gemini request boundary now:

1. Generates `model.model_json_schema()`.
2. Resolves local `$defs` / `$ref` references recursively.
3. Converts optional `anyOf[..., {"type": "null"}]` values into the non-null schema plus
   `nullable: true`.
4. Removes unsupported request-only schema keys recursively:
   `$defs`, `$schema`, `additionalProperties`, `additional_properties`, `default`,
   `description`, `examples`, and `title`.
5. Sends the cleaned schema via `GenerateContentConfig(response_json_schema=...)`.
6. Parses Gemini JSON and validates the response with the original strict Pydantic model.

This applies to both the whole-PDF technical scout (`ExtractionScoutDraft`) and the physical-slice
source transcription (`SourceContentDraft`).

## Live Verification

Final bounded one-page live run:

`/mnt/c/living_guideline_platform/runs/transcription-v3-20260726T184317687706Z-AWMF_021-026_NEUROENDOKRINE_TUMORE_2018-c9d1c6da`

Observed final state:

- `status`: `technical_limited`
- `execution_mode`: `live`
- `provider_backend`: `google_genai`
- `provider_call_count`: 2
- `scout_call_count`: 1
- `transcription_call_count`: 1
- `successful_call_count`: 2
- `failed_call_count`: 0
- `finish_reason`: `STOP` for scout and transcription evidence
- page 1 source characters: 4164
- local text-layer characters: 4102
- transcription ratio: 1.0151

No synthetic marker or obvious secret marker was found in the final run artifacts.
