"""GenerateContent adapter for staged canonical extraction from one uploaded PDF."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, SecretStr, ValidationError

from aisurgeon.extraction.canonical.models import (
    CANONICAL_EXTRACTION_SCHEMA_VERSION,
    ExtractionBatch,
    ReferenceBatch,
    VisualObjectBatch,
)
from aisurgeon.extraction.gemini.client import GeminiDocumentMapClient
from aisurgeon.extraction.gemini.errors import GeminiResponseValidationError
from aisurgeon.extraction.gemini.models import (
    DOCUMENT_MAP_SCHEMA_VERSION,
    DocumentMap,
    GeminiModelConfig,
    RemoteFileMetadata,
)

ModelT = TypeVar("ModelT", bound=BaseModel)
GENERATE_CONTENT_TIMEOUT_SECONDS = 1200


class CanonicalGeminiClient(GeminiDocumentMapClient):
    """Reuse one Files-API URI for synchronous GenerateContent extraction jobs."""

    def __init__(
        self,
        *,
        api_key: SecretStr,
        model_config: GeminiModelConfig,
        client: Any | None = None,
        sleep: Any = None,
    ) -> None:
        kwargs = {
            "api_key": api_key,
            "model_config": model_config,
            "client": client,
            "request_timeout_seconds": GENERATE_CONTENT_TIMEOUT_SECONDS,
        }
        if sleep is not None:
            kwargs["sleep"] = sleep
        super().__init__(**kwargs)

    def upload_pdf(self, pdf_path: Path) -> Any:
        remote = self._with_retry(
            lambda: self._client.files.upload(
                file=pdf_path, config={"mime_type": "application/pdf"}
            )
        )
        remote = self.wait_until_active(remote)
        self.last_remote_metadata = RemoteFileMetadata(
            remote_file_name=getattr(remote, "name", None),
            uri=getattr(remote, "uri", None),
            mime_type=getattr(remote, "mime_type", "application/pdf"),
            upload_timestamp_utc=datetime.now(UTC),
            status="ACTIVE",
            remote_file_deleted=False,
        )
        return remote

    @staticmethod
    def _canonical_request_schema(model: type[BaseModel]) -> dict[str, Any]:
        schema = GeminiDocumentMapClient.request_schema(model)
        python_only_fields = {
            "item_id",
            "formal_item_id",
            "sequence_number",
            "normalized_item_family",
            "schema_version",
            "source_id",
            "comment_id",
            "reference_id",
            "object_id",
            "context_block_id",
            "linked_comment_ids",
            "linked_item_ids",
            "unresolved_reference_numbers",
        }

        def remove_python_fields(value: Any) -> None:
            if isinstance(value, dict):
                properties = value.get("properties")
                if isinstance(properties, dict):
                    for name in python_only_fields:
                        properties.pop(name, None)
                required = value.get("required")
                if isinstance(required, list):
                    value["required"] = [
                        name for name in required if name not in python_only_fields
                    ]
                for child in value.values():
                    remove_python_fields(child)
            elif isinstance(value, list):
                for child in value:
                    remove_python_fields(child)

        remove_python_fields(schema)
        return schema

    @staticmethod
    def _inject_technical_fields(
        payload: dict[str, Any], *, model: type[BaseModel], source_id: str
    ) -> None:
        """Inject model-specific trusted fields without weakening content validation."""
        if model is DocumentMap:
            payload["schema_version"] = DOCUMENT_MAP_SCHEMA_VERSION
            payload["source_id"] = source_id
            return
        if model not in {ExtractionBatch, ReferenceBatch, VisualObjectBatch}:
            raise GeminiResponseValidationError("Unbekannter strukturierter Extraktionstyp.")
        payload["schema_version"] = CANONICAL_EXTRACTION_SCHEMA_VERSION
        payload["source_id"] = source_id

        def inject_source_objects(value: Any) -> None:
            if isinstance(value, dict):
                if "page_start" in value and "page_end" in value:
                    value["schema_version"] = CANONICAL_EXTRACTION_SCHEMA_VERSION
                    value["source_id"] = source_id
                for child in value.values():
                    inject_source_objects(child)
            elif isinstance(value, list):
                for child in value:
                    inject_source_objects(child)

        inject_source_objects(payload)

    @staticmethod
    def normalize_generate_content_usage(usage: Any) -> dict[str, int] | None:
        """Map SDK usage_metadata fields into secret-free stable audit names."""
        aliases = {
            "total_input_tokens": ("prompt_token_count", "input_token_count"),
            "total_output_tokens": ("candidates_token_count", "output_token_count"),
            "total_thought_tokens": ("thoughts_token_count", "thought_token_count"),
            "total_cached_tokens": ("cached_content_token_count", "cached_token_count"),
            "total_tokens": ("total_token_count", "total_tokens"),
        }
        if usage is None:
            return None
        source = usage if isinstance(usage, dict) else {}
        normalized: dict[str, int] = {}
        for output_name, candidates in aliases.items():
            for candidate in candidates:
                value = source.get(candidate) if isinstance(usage, dict) else getattr(
                    usage, candidate, None
                )
                if isinstance(value, int):
                    normalized[output_name] = value
                    break
        return normalized or None

    def request_structured(
        self, *, remote: Any, prompt: str, model: type[ModelT], source_id: str
    ) -> tuple[ModelT, str, dict[str, int] | None]:
        """Call gemini-3.5-flash directly; no agent routing or background interaction."""
        schema = self._canonical_request_schema(model)
        response = self._with_retry(
            lambda: self._client.models.generate_content(
                model=self._model_config.model_id,
                contents=[
                    {
                        "file_data": {
                            "file_uri": getattr(remote, "uri", None),
                            "mime_type": getattr(remote, "mime_type", "application/pdf"),
                        }
                    },
                    {"text": prompt},
                ],
                config={
                    "thinking_config": {
                        "thinking_level": self._model_config.thinking_level.upper(),
                    },
                    "media_resolution": (
                        f"MEDIA_RESOLUTION_{self._model_config.media_resolution.upper()}"
                    ),
                    "response_mime_type": "application/json",
                    "response_json_schema": schema,
                    "http_options": {
                        "timeout": GENERATE_CONTENT_TIMEOUT_SECONDS * 1000,
                    },
                },
            )
        )
        raw = getattr(response, "text", None)
        if not isinstance(raw, str):
            raise GeminiResponseValidationError("Gemini-Antwort enthält kein JSON.")
        try:
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise GeminiResponseValidationError("Gemini-Antwort ist kein JSON-Objekt.")
            self._inject_technical_fields(payload, model=model, source_id=source_id)
            validated = model.model_validate(payload)
        except (ValidationError, json.JSONDecodeError) as exc:
            raise GeminiResponseValidationError(
                "Gemini-Extraktionsantwort entspricht nicht dem vollständigen Schema."
            ) from exc
        usage = self.normalize_generate_content_usage(
            getattr(response, "usage_metadata", None)
        )
        return validated, raw, usage

    def delete_remote(self, remote: Any) -> bool:
        try:
            self._client.files.delete(name=remote.name)
            self.last_remote_metadata.remote_file_deleted = True
            return True
        except Exception:
            self.last_remote_metadata.status = "delete_failed"
            return False
