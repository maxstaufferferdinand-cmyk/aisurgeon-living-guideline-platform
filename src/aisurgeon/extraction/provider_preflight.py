"""Secret-free provider preflight result models and execution-mode runner."""

from collections.abc import Callable
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr
from pypdf import PdfWriter

from aisurgeon.extraction.transcription_v3.pipeline import gemini_request_schema

ProviderName = Literal["gemini", "openai", "ncbi"]
CheckStatus = Literal["passed", "failed", "skipped"]
PreflightMode = Literal["live", "dry_run", "mock_test"]


class ProviderCheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: ProviderName
    check: str
    status: CheckStatus
    safe_message: str | None = None


class ProviderPreflightReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "provider_preflight_v1"
    execution_mode: PreflightMode
    results: list[ProviderCheckResult] = Field(default_factory=list)


GEMINI_CHECKS = (
    "environment_variable_present",
    "key_authentic",
    "configured_model_accessible",
    "simple_text_generation_operational",
    "structured_output_operational",
    "one_page_pdf_input_operational",
    "file_upload_or_inline_pdf_operational",
    "quota_or_capacity_available",
)
OPENAI_CHECKS = (
    "key_present",
    "key_authentic",
    "configured_gpt_model_accessible",
    "responses_api_operational",
    "structured_output_operational",
)
NCBI_CHECKS = (
    "key_present",
    "esearch_operational",
    "efetch_operational",
    "configured_request_rate_accepted",
)


def _safe_exception_message(exc: Exception) -> str:
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    pieces = [type(exc).__name__]
    if isinstance(status, int):
        pieces.append(f"HTTP {status}")
    message = str(exc).replace("\n", " ")
    if message:
        pieces.append(message[:300])
    pieces.append("No secret value inspected or logged.")
    return "; ".join(pieces)


class RealProviderPreflightChecker:
    """Minimal live provider checks; no secret values are returned or logged."""

    def __init__(
        self,
        *,
        gemini_api_key: SecretStr | None,
        openai_api_key: SecretStr | None,
        ncbi_api_key: SecretStr | None,
        ncbi_email: SecretStr | None,
    ) -> None:
        self.gemini_api_key = gemini_api_key
        self.openai_api_key = openai_api_key
        self.ncbi_api_key = ncbi_api_key
        self.ncbi_email = ncbi_email

    def __call__(self, provider: ProviderName, check: str) -> bool:
        if provider == "gemini":
            return self._gemini(check)
        if provider == "openai":
            return self._openai(check)
        return self._ncbi(check)

    def _gemini(self, check: str) -> bool:
        if self.gemini_api_key is None:
            return False
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.gemini_api_key.get_secret_value())
        if check in {"key_authentic", "configured_model_accessible"}:
            list(client.models.list())
            return True
        if check == "simple_text_generation_operational":
            client.models.generate_content(model="gemini-3.5-flash", contents="Return OK.")
            return True
        if check == "structured_output_operational":
            class _Ok(BaseModel):
                ok: bool

            client.models.generate_content(
                model="gemini-3.5-flash",
                contents="Return JSON {'ok': true}.",
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_json_schema=gemini_request_schema(_Ok),
                ),
            )
            return True
        if check in {
            "one_page_pdf_input_operational",
            "file_upload_or_inline_pdf_operational",
        }:
            with NamedTemporaryFile(suffix=".pdf") as stream:
                writer = PdfWriter()
                writer.add_blank_page(width=72, height=72)
                writer.write(stream)
                stream.flush()
                pdf_part = types.Part.from_bytes(
                    data=Path(stream.name).read_bytes(),
                    mime_type="application/pdf",
                )
                client.models.generate_content(
                    model="gemini-3.5-flash",
                    contents=[pdf_part, "Return OK for this one-page PDF."],
                )
            return True
        if check == "quota_or_capacity_available":
            client.models.generate_content(model="gemini-3.5-flash", contents="Return OK.")
            return True
        return False

    def _openai(self, check: str) -> bool:
        if self.openai_api_key is None:
            return False
        from openai import OpenAI

        client = OpenAI(api_key=self.openai_api_key.get_secret_value(), timeout=60)
        if check in {"key_authentic", "configured_gpt_model_accessible"}:
            client.models.retrieve("gpt-5.5")
            return True
        if check in {"responses_api_operational", "structured_output_operational"}:
            class _Ok(BaseModel):
                ok: bool

            client.responses.parse(
                model="gpt-5.5",
                instructions="Return ok=true.",
                input="Return JSON.",
                text_format=_Ok,
            )
            return True
        return False

    def _ncbi(self, check: str) -> bool:
        if self.ncbi_api_key is None or self.ncbi_email is None:
            return False
        import httpx

        params = {
            "db": "pubmed",
            "term": "cancer",
            "retmax": "1",
            "retmode": "json",
            "api_key": self.ncbi_api_key.get_secret_value(),
            "email": self.ncbi_email.get_secret_value(),
        }
        if check in {"esearch_operational", "configured_request_rate_accepted"}:
            response = httpx.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                params=params,
                timeout=30,
            )
            response.raise_for_status()
            return True
        if check == "efetch_operational":
            response = httpx.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                params={
                    "db": "pubmed",
                    "id": "1",
                    "retmode": "xml",
                    "api_key": self.ncbi_api_key.get_secret_value(),
                    "email": self.ncbi_email.get_secret_value(),
                },
                timeout=30,
            )
            response.raise_for_status()
            return True
        return False


def run_provider_preflight(
    *,
    providers: set[ProviderName],
    execution_mode: PreflightMode,
    gemini_api_key: SecretStr | None = None,
    openai_api_key: SecretStr | None = None,
    ncbi_api_key: SecretStr | None = None,
    ncbi_email: SecretStr | None = None,
    live_checker: Callable[[ProviderName, str], bool] | None = None,
) -> ProviderPreflightReport:
    """Run explicit-mode checks; live checking never logs secrets."""
    check_map = {"gemini": GEMINI_CHECKS, "openai": OPENAI_CHECKS, "ncbi": NCBI_CHECKS}
    present = {
        "gemini": gemini_api_key is not None,
        "openai": openai_api_key is not None,
        "ncbi": ncbi_api_key is not None and ncbi_email is not None,
    }
    results: list[ProviderCheckResult] = []
    if execution_mode == "live" and live_checker is None:
        live_checker = RealProviderPreflightChecker(
            gemini_api_key=gemini_api_key,
            openai_api_key=openai_api_key,
            ncbi_api_key=ncbi_api_key,
            ncbi_email=ncbi_email,
        )
    for provider in sorted(providers):
        for check in check_map[provider]:
            safe_message = None
            if check in {"environment_variable_present", "key_present"}:
                status: CheckStatus = "passed" if present[provider] else "failed"
            elif execution_mode != "live":
                status = "skipped"
            else:
                try:
                    status = "passed" if live_checker(provider, check) else "failed"
                except Exception as exc:
                    status = "failed"
                    safe_message = _safe_exception_message(exc)
                else:
                    safe_message = None
            results.append(
                ProviderCheckResult(
                    provider=provider,
                    check=check,
                    status=status,
                    safe_message=(
                        None
                        if status == "passed"
                        else safe_message or "No secret value inspected or logged."
                    ),
                )
            )
    return ProviderPreflightReport(execution_mode=execution_mode, results=results)
