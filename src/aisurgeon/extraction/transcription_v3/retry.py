"""Versioned provider retry classification for transcription v3."""

from dataclasses import dataclass
from typing import Literal

RetryCategory = Literal["retryable", "non_retryable"]
FailureCategory = Literal[
    "rate_or_quota",
    "provider_capacity_unavailable",
    "transient_provider_or_network",
    "non_retryable_provider_or_local_failure",
]


@dataclass(frozen=True)
class RetryDecision:
    category: RetryCategory
    safe_exception_class: str
    http_status: int | None
    calculated_delay_seconds: float | None
    final_failure_category: FailureCategory


RETRYABLE_HTTP = {408, 429, 500, 502, 503, 504}
NON_RETRYABLE_HTTP = {400, 401, 403}


def classify_provider_failure(
    exc: Exception,
    *,
    attempt: int,
    retry_after_seconds: float | None = None,
    base_delay_seconds: float = 15,
    max_delay_seconds: float = 900,
    jitter_fraction: float = 0.0,
) -> RetryDecision:
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    status = status if isinstance(status, int) else None
    name = type(exc).__name__
    message = str(exc).casefold()
    transient_name = any(token in name.lower() for token in ("timeout", "connection", "dns"))
    malformed_model_json = name == "JSONDecodeError"
    retryable = status in RETRYABLE_HTTP or transient_name or "connection reset" in message
    retryable = retryable or malformed_model_json
    if (
        status in NON_RETRYABLE_HTTP
        or "invalid api key" in message
        or "permission denied" in message
    ):
        retryable = False
    if retryable:
        delay = min(max_delay_seconds, base_delay_seconds * (2 ** max(attempt - 1, 0)))
        if status == 429 and retry_after_seconds is not None:
            delay = max(delay, retry_after_seconds)
        if jitter_fraction:
            delay *= 1 + jitter_fraction
        if status == 429 or "resource_exhausted" in message:
            final_category: FailureCategory = "rate_or_quota"
        elif status == 503 or "unavailable" in message or "high demand" in message:
            final_category = "provider_capacity_unavailable"
        elif malformed_model_json:
            final_category = "transient_provider_or_network"
        else:
            final_category = "transient_provider_or_network"
        return RetryDecision("retryable", name, status, delay, final_category)
    return RetryDecision(
        "non_retryable",
        name,
        status,
        None,
        "non_retryable_provider_or_local_failure",
    )
