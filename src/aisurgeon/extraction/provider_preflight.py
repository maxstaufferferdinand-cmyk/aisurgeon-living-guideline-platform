"""Secret-free provider preflight result models and mocked check runner."""

from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr

ProviderName = Literal["gemini", "openai", "ncbi"]
CheckStatus = Literal["passed", "failed", "skipped"]


class ProviderCheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: ProviderName
    check: str
    status: CheckStatus
    safe_message: str | None = None


class ProviderPreflightReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "provider_preflight_v1"
    mode: Literal["mocked", "live"]
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


def run_provider_preflight(
    *,
    providers: set[ProviderName],
    gemini_api_key: SecretStr | None = None,
    openai_api_key: SecretStr | None = None,
    ncbi_api_key: SecretStr | None = None,
    ncbi_email: SecretStr | None = None,
    live_checker: Callable[[ProviderName, str], bool] | None = None,
) -> ProviderPreflightReport:
    """Run mocked checks by default; live checking is injectable and never logs secrets."""
    check_map = {"gemini": GEMINI_CHECKS, "openai": OPENAI_CHECKS, "ncbi": NCBI_CHECKS}
    present = {
        "gemini": gemini_api_key is not None,
        "openai": openai_api_key is not None,
        "ncbi": ncbi_api_key is not None and ncbi_email is not None,
    }
    results: list[ProviderCheckResult] = []
    mode = "live" if live_checker else "mocked"
    for provider in sorted(providers):
        for check in check_map[provider]:
            if check in {"environment_variable_present", "key_present"}:
                status: CheckStatus = "passed" if present[provider] else "failed"
            elif live_checker is None:
                status = "skipped"
            else:
                status = "passed" if live_checker(provider, check) else "failed"
            results.append(
                ProviderCheckResult(
                    provider=provider,
                    check=check,
                    status=status,
                    safe_message=(
                        None if status == "passed" else "No secret value inspected or logged."
                    ),
                )
            )
    return ProviderPreflightReport(mode=mode, results=results)
