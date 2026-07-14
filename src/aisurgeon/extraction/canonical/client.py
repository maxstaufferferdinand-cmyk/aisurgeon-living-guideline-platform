"""One-upload Gemini session for staged canonical extraction."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, SecretStr, ValidationError

from aisurgeon.extraction.gemini.client import GeminiDocumentMapClient
from aisurgeon.extraction.gemini.errors import GeminiResponseValidationError
from aisurgeon.extraction.gemini.models import GeminiModelConfig, RemoteFileMetadata

ModelT = TypeVar("ModelT", bound=BaseModel)


class CanonicalGeminiClient(GeminiDocumentMapClient):
    """Reuse one temporary PDF for every structured extraction job."""

    def __init__(
        self,
        *,
        api_key: SecretStr,
        model_config: GeminiModelConfig,
        client: Any | None = None,
        sleep: Any = None,
    ) -> None:
        kwargs = {"api_key": api_key, "model_config": model_config, "client": client}
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

    def request_structured(
        self, *, remote: Any, prompt: str, model: type[ModelT]
    ) -> tuple[ModelT, str, dict[str, int] | None]:
        schema = self.request_schema(model)
        python_only_fields = {
            "item_id", "comment_id", "reference_id", "object_id", "context_block_id",
            "linked_comment_ids", "linked_item_ids", "unresolved_reference_numbers",
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
        response = self._with_retry(
            lambda: self._client.interactions.create(
                model=self._model_config.model_id,
                input=[
                    {
                        "type": "document",
                        "uri": getattr(remote, "uri", None),
                        "mime_type": "application/pdf",
                        "resolution": self._model_config.media_resolution,
                    },
                    {"type": "text", "text": prompt},
                ],
                generation_config={"thinking_level": self._model_config.thinking_level},
                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": schema,
                },
                store=False,
            )
        )
        raw = getattr(response, "output_text", None)
        if not isinstance(raw, str):
            raise GeminiResponseValidationError("Gemini-Antwort enthält kein JSON.")
        try:
            validated = model.model_validate_json(raw)
        except (ValidationError, json.JSONDecodeError) as exc:
            raise GeminiResponseValidationError(
                "Gemini-Extraktionsantwort entspricht nicht dem vollständigen Schema."
            ) from exc
        return validated, raw, self.normalize_usage(getattr(response, "usage", None))

    def delete_remote(self, remote: Any) -> bool:
        try:
            self._client.files.delete(name=remote.name)
            self.last_remote_metadata.remote_file_deleted = True
            return True
        except Exception:
            self.last_remote_metadata.status = "delete_failed"
            return False
