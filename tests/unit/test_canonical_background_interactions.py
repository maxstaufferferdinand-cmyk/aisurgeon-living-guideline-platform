"""Mocked background Interaction lifecycle and resume tests."""

from types import SimpleNamespace

import httpx
import pytest
from pydantic import SecretStr

from aisurgeon.extraction.canonical.client import CanonicalGeminiClient
from aisurgeon.extraction.canonical.models import VisualObjectBatch
from aisurgeon.extraction.gemini.errors import (
    GeminiInteractionFailedError,
    GeminiTransientError,
)
from aisurgeon.extraction.gemini.models import GeminiModelConfig


def model_config() -> GeminiModelConfig:
    return GeminiModelConfig(
        provider="google", api="interactions", model_id="gemini-3.5-flash",
        thinking_level="medium", media_resolution="high",
        prompt_version="gemini_document_map_v1", schema_version="document_map_v1",
        request_timeout_seconds=10, max_attempts=3,
    )


class FakeInteractions:
    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.create_calls = []
        self.get_ids = []
        self.deleted = []

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return SimpleNamespace(id="interactions/123", status="in_progress")

    def get(self, *, id: str):
        self.get_ids.append(id)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def delete(self, *, id: str) -> None:
        self.deleted.append(id)


def completed():
    value = VisualObjectBatch()
    return SimpleNamespace(status="completed", output_text=value.model_dump_json(),
                           usage={"total_tokens": 4})


def client(interactions: FakeInteractions, sleeps: list[float]) -> CanonicalGeminiClient:
    sdk = SimpleNamespace(interactions=interactions, files=SimpleNamespace())
    return CanonicalGeminiClient(api_key=SecretStr("dummy"), model_config=model_config(),
                                 client=sdk, sleep=sleeps.append,
                                 poll_interval_seconds=0.25, max_poll_attempts=5)


def test_background_start_is_stored_and_id_is_immediately_reported() -> None:
    interactions = FakeInteractions([
        SimpleNamespace(status="in_progress"), completed()
    ])
    sleeps = []
    saved_ids = []
    result = client(interactions, sleeps).request_structured(
        remote=SimpleNamespace(uri="mock://pdf"), prompt="prompt",
        model=VisualObjectBatch, on_started=saved_ids.append,
    )
    request = interactions.create_calls[0]
    assert request["background"] is True
    assert request["store"] is True
    assert saved_ids == ["interactions/123"]
    assert result[3] == "interactions/123"
    assert interactions.get_ids == ["interactions/123", "interactions/123"]


def test_resume_polls_saved_interaction_without_duplicate_create() -> None:
    interactions = FakeInteractions([completed()])
    result = client(interactions, []).request_structured(
        remote=SimpleNamespace(uri="mock://pdf"), prompt="unused",
        model=VisualObjectBatch, interaction_id="interactions/saved",
    )
    assert result[3] == "interactions/saved"
    assert interactions.create_calls == []
    assert interactions.get_ids == ["interactions/saved"]


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_terminal_failure_statuses_are_distinguished(status: str) -> None:
    interactions = FakeInteractions([SimpleNamespace(status=status)])
    with pytest.raises(GeminiInteractionFailedError, match=status):
        client(interactions, []).request_structured(
            remote=SimpleNamespace(uri="mock://pdf"), prompt="prompt",
            model=VisualObjectBatch,
        )


def test_timeout_of_individual_poll_request_is_retried() -> None:
    request = httpx.Request("GET", "https://invalid.test")
    interactions = FakeInteractions([httpx.ReadTimeout("synthetic", request=request), completed()])
    sleeps = []
    result = client(interactions, sleeps).request_structured(
        remote=SimpleNamespace(uri="mock://pdf"), prompt="prompt", model=VisualObjectBatch,
    )
    assert result[0] == VisualObjectBatch()
    assert interactions.get_ids == ["interactions/123", "interactions/123"]
    assert sleeps == [1.0]


def test_interaction_delete_is_best_effort() -> None:
    interactions = FakeInteractions([])
    gateway = client(interactions, [])
    assert gateway.delete_interaction("interactions/123") is True
    assert interactions.deleted == ["interactions/123"]


def test_google_api_timeout_class_name_is_transient() -> None:
    api_timeout_type = type("APITimeoutError", (Exception,), {})
    classified = CanonicalGeminiClient._classify(api_timeout_type("synthetic"))
    assert isinstance(classified, GeminiTransientError)
