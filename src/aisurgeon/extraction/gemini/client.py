"""Injectable, bounded-retry wrapper around google-genai."""

import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import SecretStr, ValidationError

from aisurgeon.extraction.gemini.errors import (
    GeminiAuthenticationError,
    GeminiError,
    GeminiRateLimitError,
    GeminiResponseValidationError,
    GeminiTransientError,
)
from aisurgeon.extraction.gemini.models import (
    DocumentMap,
    GeminiDocumentMapResult,
    GeminiModelConfig,
    RemoteFileMetadata,
)


class GeminiDocumentMapClient:
    """Small Gemini boundary with no model fallback and no secret logging."""

    def __init__(
        self,
        *,
        api_key: SecretStr,
        model_config: GeminiModelConfig,
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._model_config = model_config
        self._sleep = sleep
        self.last_remote_metadata = RemoteFileMetadata(
            status="not_uploaded", remote_file_deleted=False
        )
        if client is None:
            from google import genai

            client = genai.Client(
                api_key=api_key.get_secret_value(),
                http_options={"timeout": model_config.request_timeout_seconds * 1000},
            )
        self._client = client

    @staticmethod
    def _status_code(exc: Exception) -> int | None:
        for attribute in ("status_code", "code"):
            value = getattr(exc, attribute, None)
            if isinstance(value, int):
                return value
        response = getattr(exc, "response", None)
        value = getattr(response, "status_code", None)
        return value if isinstance(value, int) else None

    @classmethod
    def _classify(cls, exc: Exception) -> GeminiError:
        status = cls._status_code(exc)
        if status in {401, 403}:
            return GeminiAuthenticationError("Gemini-Authentifizierung fehlgeschlagen.")
        if status == 429:
            return GeminiRateLimitError("Gemini-Rate-Limit erreicht.")
        if status in {408, 409, 425} or (status is not None and 500 <= status <= 599):
            return GeminiTransientError("Temporärer Gemini-Dienstfehler.")
        if isinstance(exc, (ConnectionError, TimeoutError)):
            return GeminiTransientError("Temporärer Gemini-Verbindungsfehler.")
        return GeminiError("Gemini-Anfrage fehlgeschlagen.")

    def _with_retry(self, operation: Callable[[], Any]) -> Any:
        last_error: GeminiError | None = None
        for attempt in range(1, self._model_config.max_attempts + 1):
            try:
                return operation()
            except Exception as exc:  # SDK exception types vary by transport.
                classified = self._classify(exc)
                retryable = isinstance(classified, (GeminiTransientError, GeminiRateLimitError))
                if not retryable or attempt == self._model_config.max_attempts:
                    raise classified from exc
                last_error = classified
                self._sleep(float(2 ** (attempt - 1)))
        raise last_error or GeminiError("Gemini-Anfrage fehlgeschlagen.")

    def create_document_map(
        self,
        *,
        pdf_path: Path,
        prompt: str,
        source_id: str,
        keep_remote_file: bool = False,
    ) -> GeminiDocumentMapResult:
        """Upload one PDF, request one map, validate it, and delete the remote file."""
        remote: Any | None = None
        metadata = RemoteFileMetadata(status="upload_pending", remote_file_deleted=False)
        try:
            remote = self._with_retry(
                lambda: self._client.files.upload(
                    file=pdf_path,
                    config={"mime_type": "application/pdf"},
                )
            )
            metadata = RemoteFileMetadata(
                remote_file_name=getattr(remote, "name", None),
                uri=getattr(remote, "uri", None),
                mime_type=getattr(remote, "mime_type", "application/pdf"),
                upload_timestamp_utc=datetime.now(UTC),
                status=str(getattr(remote, "state", "uploaded")),
                remote_file_deleted=False,
            )
            response = self._with_retry(
                lambda: self._client.interactions.create(
                    model=self._model_config.model_id,
                    input=[
                        {
                            "type": "document",
                            "uri": metadata.uri,
                            "mime_type": metadata.mime_type,
                            "resolution": self._model_config.media_resolution,
                        },
                        {"type": "text", "text": f"source_id: {source_id}\n\n{prompt}"},
                    ],
                    generation_config={
                        "thinking_level": self._model_config.thinking_level,
                    },
                    response_format={
                        "type": "text",
                        "mime_type": "application/json",
                        "schema": DocumentMap.model_json_schema(),
                    },
                    store=False,
                )
            )
            raw_json = getattr(response, "output_text", None)
            if not isinstance(raw_json, str):
                raise GeminiResponseValidationError("Gemini-Antwort enthält kein JSON.")
            try:
                document_map = DocumentMap.model_validate_json(raw_json)
            except (ValidationError, json.JSONDecodeError) as exc:
                raise GeminiResponseValidationError(
                    "Gemini-Dokumentkarte entspricht nicht dem Schema."
                ) from exc
            usage = getattr(response, "usage", None)
            token_usage = usage if isinstance(usage, dict) else None
            return GeminiDocumentMapResult(
                document_map=document_map,
                raw_json=raw_json,
                remote_file_metadata=metadata,
                token_usage=token_usage,
            )
        finally:
            if remote is not None and not keep_remote_file:
                try:
                    self._client.files.delete(name=remote.name)
                    metadata.remote_file_deleted = True
                except Exception:
                    metadata.status = "delete_failed"
            self.last_remote_metadata = metadata
