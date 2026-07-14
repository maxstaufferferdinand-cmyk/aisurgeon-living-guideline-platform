"""Fully mocked Gemini client boundary tests."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from aisurgeon.extraction.gemini.client import GeminiDocumentMapClient
from aisurgeon.extraction.gemini.errors import (
    GeminiAuthenticationError,
    GeminiResponseValidationError,
)
from aisurgeon.extraction.gemini.models import GeminiModelConfig


def model_config() -> GeminiModelConfig:
    return GeminiModelConfig(
        provider="google",
        api="interactions",
        model_id="gemini-3.5-flash",
        thinking_level="medium",
        media_resolution="high",
        prompt_version="gemini_document_map_v1",
        schema_version="document_map_v1",
        request_timeout_seconds=10,
        max_attempts=3,
    )


def map_json(source_id: str) -> str:
    return json.dumps(
        {
            "schema_version": "document_map_v1",
            "source_id": source_id,
            "document_title": None,
            "issuing_organization": None,
            "guideline_identifier": None,
            "guideline_class": None,
            "language": None,
            "publication_year": None,
            "version_text": None,
            "validity_status": None,
            "declared_page_count": 2,
            "detected_document_layout": "single-column synthetic fixture",
            "column_layout": "single",
            "recurring_header_footer_description": None,
            "front_matter_page_ranges": [],
            "table_of_contents_page_ranges": [],
            "clinical_main_body_page_ranges": [{"page_start": 1, "page_end": 2}],
            "bibliography_page_ranges": [],
            "appendix_page_ranges": [],
            "recommendation_or_statement_patterns": [],
            "comment_or_rationale_patterns": [],
            "native_grading_systems": [],
            "detected_formal_item_types": [],
            "detected_table_inventory": [],
            "detected_algorithm_inventory": [],
            "detected_decision_tree_inventory": [],
            "uncertain_regions": [],
            "warnings": ["synthetic fixture"],
        }
    )


class ApiFailure(Exception):
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        super().__init__("remote detail deliberately excluded")


class FakeFiles:
    def __init__(self, upload_failures: list[Exception] | None = None) -> None:
        self.upload_failures = list(upload_failures or [])
        self.upload_calls = 0
        self.deleted: list[str] = []
        self.get_states: list[str] = []

    def upload(self, **kwargs):
        self.upload_calls += 1
        if self.upload_failures:
            raise self.upload_failures.pop(0)
        assert kwargs["config"] == {"mime_type": "application/pdf"}
        return SimpleNamespace(
            name="files/mock-pdf",
            uri="mock://files/mock-pdf",
            mime_type="application/pdf",
            state="ACTIVE",
        )

    def delete(self, *, name: str) -> None:
        self.deleted.append(name)

    def get(self, *, name: str):
        state = self.get_states.pop(0)
        return SimpleNamespace(
            name=name, uri="mock://files/mock-pdf", mime_type="application/pdf", state=state
        )


class FakeInteractions:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=self.output_text, usage={"total_tokens": 12})


def fake_sdk(source_id: str, *, upload_failures=None, output_text: str | None = None):
    return SimpleNamespace(
        files=FakeFiles(upload_failures),
        interactions=FakeInteractions(output_text or map_json(source_id)),
    )


def test_mocked_upload_and_structured_interaction(synthetic_pdf: Path) -> None:
    sdk = fake_sdk("source-test")
    client = GeminiDocumentMapClient(
        api_key=SecretStr("dummy-key"), model_config=model_config(), client=sdk
    )
    result = client.create_document_map(
        pdf_path=synthetic_pdf,
        prompt="synthetic prompt",
        source_id="source-test",
    )
    request = sdk.interactions.calls[0]
    assert result.document_map.source_id == "source-test"
    assert request["model"] == "gemini-3.5-flash"
    assert request["generation_config"] == {"thinking_level": "medium"}
    assert request["input"][0]["resolution"] == "high"
    assert request["store"] is False


def test_invalid_structured_response_is_rejected_and_deleted(synthetic_pdf: Path) -> None:
    sdk = fake_sdk("source-test", output_text='{"not": "a document map"}')
    client = GeminiDocumentMapClient(
        api_key=SecretStr("dummy-key"), model_config=model_config(), client=sdk
    )
    with pytest.raises(GeminiResponseValidationError):
        client.create_document_map(
            pdf_path=synthetic_pdf,
            prompt="prompt",
            source_id="source-test",
        )
    assert sdk.files.deleted == ["files/mock-pdf"]


def test_transient_upload_error_retries_with_backoff(synthetic_pdf: Path) -> None:
    sdk = fake_sdk("source-test", upload_failures=[ApiFailure(500), ApiFailure(503)])
    sleeps: list[float] = []
    client = GeminiDocumentMapClient(
        api_key=SecretStr("dummy-key"),
        model_config=model_config(),
        client=sdk,
        sleep=sleeps.append,
    )
    client.create_document_map(
        pdf_path=synthetic_pdf,
        prompt="prompt",
        source_id="source-test",
    )
    assert sdk.files.upload_calls == 3
    assert sleeps == [1.0, 2.0]


def test_authentication_error_is_not_retried(synthetic_pdf: Path) -> None:
    sdk = fake_sdk("source-test", upload_failures=[ApiFailure(401)])
    client = GeminiDocumentMapClient(
        api_key=SecretStr("dummy-key"), model_config=model_config(), client=sdk
    )
    with pytest.raises(GeminiAuthenticationError, match="Authentifizierung"):
        client.create_document_map(
            pdf_path=synthetic_pdf,
            prompt="prompt",
            source_id="source-test",
        )
    assert sdk.files.upload_calls == 1


def test_remote_file_is_deleted_by_default(synthetic_pdf: Path) -> None:
    sdk = fake_sdk("source-test")
    client = GeminiDocumentMapClient(
        api_key=SecretStr("dummy-key"), model_config=model_config(), client=sdk
    )
    result = client.create_document_map(
        pdf_path=synthetic_pdf,
        prompt="prompt",
        source_id="source-test",
    )
    assert sdk.files.deleted == ["files/mock-pdf"]
    assert result.remote_file_metadata.remote_file_deleted is True


def test_keep_remote_file_prevents_deletion(synthetic_pdf: Path) -> None:
    sdk = fake_sdk("source-test")
    client = GeminiDocumentMapClient(
        api_key=SecretStr("dummy-key"), model_config=model_config(), client=sdk
    )
    result = client.create_document_map(
        pdf_path=synthetic_pdf,
        prompt="prompt",
        source_id="source-test",
        keep_remote_file=True,
    )
    assert sdk.files.deleted == []
    assert result.remote_file_metadata.remote_file_deleted is False


def test_no_automatic_model_fallback(synthetic_pdf: Path) -> None:
    sdk = fake_sdk("source-test")
    client = GeminiDocumentMapClient(
        api_key=SecretStr("dummy-key"), model_config=model_config(), client=sdk
    )
    client.create_document_map(
        pdf_path=synthetic_pdf,
        prompt="prompt",
        source_id="source-test",
    )
    assert {call["model"] for call in sdk.interactions.calls} == {"gemini-3.5-flash"}


def test_api_key_not_present_in_safe_exception(synthetic_pdf: Path) -> None:
    secret = "PREFIX_dummy_secret_SUFFIX"
    sdk = fake_sdk("source-test", upload_failures=[ApiFailure(401)])
    client = GeminiDocumentMapClient(
        api_key=SecretStr(secret), model_config=model_config(), client=sdk
    )
    with pytest.raises(GeminiAuthenticationError) as captured:
        client.create_document_map(
            pdf_path=synthetic_pdf,
            prompt="prompt",
            source_id="source-test",
        )
    rendered = str(captured.value)
    assert secret not in rendered
    assert "PREFIX_dummy" not in rendered
    assert "secret_SUFFIX" not in rendered


def test_processing_file_is_polled_until_active(synthetic_pdf: Path) -> None:
    sdk = fake_sdk("source-test")
    sdk.files.get_states = ["ACTIVE"]
    original_upload = sdk.files.upload
    sdk.files.upload = lambda **kwargs: SimpleNamespace(
        name="files/mock-pdf",
        uri="mock://files/mock-pdf",
        mime_type="application/pdf",
        state="PROCESSING",
    )
    sleeps: list[float] = []
    client = GeminiDocumentMapClient(
        api_key=SecretStr("dummy-key"), model_config=model_config(), client=sdk, sleep=sleeps.append
    )
    client.create_document_map(pdf_path=synthetic_pdf, prompt="prompt", source_id="source-test")
    assert sleeps == [1.0]
    sdk.files.upload = original_upload


def test_request_schema_removes_defaults_and_usage_is_normalized() -> None:
    schema = GeminiDocumentMapClient.request_schema(
        __import__("aisurgeon.extraction.gemini.models", fromlist=["DocumentMap"]).DocumentMap
    )
    assert '"default"' not in json.dumps(schema)
    usage = SimpleNamespace(
        total_tokens=10, total_input_tokens=7, total_output_tokens=3, secret="must-not-copy"
    )
    assert GeminiDocumentMapClient.normalize_usage(usage) == {
        "total_tokens": 10,
        "total_input_tokens": 7,
        "total_output_tokens": 3,
    }
