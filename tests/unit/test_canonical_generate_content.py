"""Mocked GenerateContent transport tests for canonical PDF extraction."""

from types import SimpleNamespace

import httpx
import pytest
from pydantic import SecretStr

from aisurgeon.extraction.canonical.client import (
    GENERATE_CONTENT_TIMEOUT_SECONDS,
    CanonicalGeminiClient,
)
from aisurgeon.extraction.canonical.models import VisualObjectBatch
from aisurgeon.extraction.gemini.errors import GeminiError
from aisurgeon.extraction.gemini.models import GeminiModelConfig


def model_config() -> GeminiModelConfig:
    return GeminiModelConfig(
        provider="google", api="interactions", model_id="gemini-3.5-flash",
        thinking_level="medium", media_resolution="high",
        prompt_version="gemini_document_map_v1", schema_version="document_map_v1",
        request_timeout_seconds=180, max_attempts=3,
    )


class ApiFailure(Exception):
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        super().__init__("remote details excluded")


class FakeModels:
    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def response():
    return SimpleNamespace(
        text=VisualObjectBatch().model_dump_json(),
        usage_metadata=SimpleNamespace(
            prompt_token_count=11, candidates_token_count=7, thoughts_token_count=3,
            cached_content_token_count=2, total_token_count=21,
        ),
    )


def gateway(models: FakeModels, sleeps=None) -> CanonicalGeminiClient:
    sdk = SimpleNamespace(models=models, files=SimpleNamespace())
    selected_sleeps = sleeps if sleeps is not None else []
    return CanonicalGeminiClient(api_key=SecretStr("dummy"), model_config=model_config(),
                                 client=sdk, sleep=selected_sleeps.append)


def test_generate_content_uses_pdf_uri_model_and_structured_schema() -> None:
    models = FakeModels([response()])
    result = gateway(models).request_structured(
        remote=SimpleNamespace(uri="mock://files/source", mime_type="application/pdf"),
        prompt="exact prompt", model=VisualObjectBatch,
    )
    request = models.calls[0]
    assert request["model"] == "gemini-3.5-flash"
    assert "-agent" not in request["model"]
    assert request["contents"][0]["file_data"] == {
        "file_uri": "mock://files/source", "mime_type": "application/pdf"
    }
    assert request["contents"][1] == {"text": "exact prompt"}
    assert request["config"]["thinking_config"] == {"thinking_level": "MEDIUM"}
    assert request["config"]["media_resolution"] == "MEDIA_RESOLUTION_HIGH"
    assert request["config"]["response_mime_type"] == "application/json"
    assert isinstance(request["config"]["response_json_schema"], dict)
    assert "temperature" not in request["config"]
    assert request["config"]["http_options"] == {"timeout": 1_200_000}
    assert result[2] == {
        "total_input_tokens": 11, "total_output_tokens": 7,
        "total_thought_tokens": 3, "total_cached_tokens": 2, "total_tokens": 21,
    }


def test_client_timeout_is_1200_seconds(monkeypatch) -> None:
    captured = {}

    def fake_client(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr("google.genai.Client", fake_client)
    CanonicalGeminiClient(api_key=SecretStr("dummy"), model_config=model_config())
    assert GENERATE_CONTENT_TIMEOUT_SECONDS == 1200
    assert captured["http_options"] == {"timeout": 1_200_000}


def test_read_timeout_retries_with_exponential_backoff() -> None:
    request = httpx.Request("POST", "https://invalid.test")
    models = FakeModels([
        httpx.ReadTimeout("synthetic", request=request),
        httpx.ReadTimeout("synthetic", request=request), response(),
    ])
    sleeps = []
    gateway(models, sleeps).request_structured(
        remote=SimpleNamespace(uri="mock://pdf"), prompt="prompt", model=VisualObjectBatch,
    )
    assert len(models.calls) == 3
    assert sleeps == [1.0, 2.0]


def test_http_400_is_not_retried() -> None:
    models = FakeModels([ApiFailure(400)])
    with pytest.raises(GeminiError):
        gateway(models).request_structured(
            remote=SimpleNamespace(uri="mock://pdf"), prompt="prompt",
            model=VisualObjectBatch,
        )
    assert len(models.calls) == 1


def test_secret_is_absent_from_transport_error() -> None:
    secret = "PREFIX_dummy_secret_SUFFIX"
    models = FakeModels([ApiFailure(400)])
    sdk = SimpleNamespace(models=models, files=SimpleNamespace())
    client = CanonicalGeminiClient(
        api_key=SecretStr(secret), model_config=model_config(), client=sdk
    )
    with pytest.raises(GeminiError) as captured:
        client.request_structured(remote=SimpleNamespace(uri="mock://pdf"), prompt="prompt",
                                  model=VisualObjectBatch)
    assert secret not in str(captured.value)
    assert "PREFIX_dummy" not in str(captured.value)
    assert "secret_SUFFIX" not in str(captured.value)
