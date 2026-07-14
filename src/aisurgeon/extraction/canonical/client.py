"""One-upload Gemini session for staged canonical extraction."""

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, SecretStr, ValidationError

from aisurgeon.extraction.gemini.client import GeminiDocumentMapClient
from aisurgeon.extraction.gemini.errors import (
    GeminiInteractionFailedError,
    GeminiResponseValidationError,
    GeminiTransientError,
)
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
        poll_interval_seconds: float = 5.0,
        max_poll_attempts: int = 120,
    ) -> None:
        kwargs = {"api_key": api_key, "model_config": model_config, "client": client}
        if sleep is not None:
            kwargs["sleep"] = sleep
        super().__init__(**kwargs)
        self._poll_interval_seconds = poll_interval_seconds
        self._max_poll_attempts = max_poll_attempts

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
        self,
        *,
        remote: Any,
        prompt: str,
        model: type[ModelT],
        interaction_id: str | None = None,
        on_started: Callable[[str], None] | None = None,
    ) -> tuple[ModelT, str, dict[str, int] | None, str]:
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
        if interaction_id is None:
            interaction = self._with_retry(
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
                background=True,
                store=True,
            )
            )
            interaction_id = getattr(interaction, "id", None)
            if not isinstance(interaction_id, str) or not interaction_id:
                raise GeminiResponseValidationError(
                    "Gemini-Hintergrundinteraktion enthält keine ID."
                )
            if on_started is not None:
                on_started(interaction_id)

        completed = None
        for attempt in range(1, self._max_poll_attempts + 1):
            try:
                current = self._with_retry(
                    lambda: self._client.interactions.get(id=interaction_id)
                )
            except GeminiTransientError:
                if attempt == self._max_poll_attempts:
                    raise
                self._sleep(self._poll_interval_seconds)
                continue
            status = str(getattr(current, "status", "")).lower().split(".")[-1]
            if status == "completed":
                completed = current
                break
            if status in {"failed", "cancelled"}:
                raise GeminiInteractionFailedError(status)
            if status != "in_progress":
                raise GeminiResponseValidationError(
                    "Gemini-Hintergrundinteraktion hat einen unbekannten Status."
                )
            if attempt < self._max_poll_attempts:
                self._sleep(self._poll_interval_seconds)
        if completed is None:
            raise GeminiTransientError(
                "Gemini-Hintergrundinteraktion läuft nach Poll-Limit weiter."
            )
        raw = getattr(completed, "output_text", None)
        if not isinstance(raw, str):
            raise GeminiResponseValidationError("Gemini-Antwort enthält kein JSON.")
        try:
            validated = model.model_validate_json(raw)
        except (ValidationError, json.JSONDecodeError) as exc:
            raise GeminiResponseValidationError(
                "Gemini-Extraktionsantwort entspricht nicht dem vollständigen Schema."
            ) from exc
        return (
            validated,
            raw,
            self.normalize_usage(getattr(completed, "usage", None)),
            interaction_id,
        )

    def delete_interaction(self, interaction_id: str) -> bool:
        """Delete a persisted interaction after local outputs are durable."""
        try:
            self._client.interactions.delete(id=interaction_id)
            return True
        except Exception:
            return False

    def delete_remote(self, remote: Any) -> bool:
        try:
            self._client.files.delete(name=remote.name)
            self.last_remote_metadata.remote_file_deleted = True
            return True
        except Exception:
            self.last_remote_metadata.status = "delete_failed"
            return False
