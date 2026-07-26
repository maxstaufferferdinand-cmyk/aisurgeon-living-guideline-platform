"""Versioned provider retry classification for transcription v3."""

from dataclasses import dataclass
from typing import Literal

RetryCategory = Literal["retryable", "non_retryable"]


@dataclass(frozen=True)
class RetryDecision:
    category: RetryCategory
    safe_exception_class: str
    http_status: int | None
    calculated_delay_seconds: float | None
    final_failure_category: str


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
    retryable = status in RETRYABLE_HTTP or transient_name or "connection reset" in message
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
        return RetryDecision("retryable", name, status, delay, "transient_provider_failure")
    return RetryDecision(
        "non_retryable",
        name,
        status,
        None,
        "non_retryable_provider_or_local_failure",
    )
