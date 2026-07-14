"""Secret-safe Gemini integration errors."""


class GeminiError(RuntimeError):
    """Base error whose messages must never contain credentials."""


class GeminiAuthenticationError(GeminiError):
    """Authentication or authorization failed."""


class GeminiRateLimitError(GeminiError):
    """The remote service rejected the request due to rate limiting."""


class GeminiTransientError(GeminiError):
    """A bounded retry may resolve the remote failure."""


class GeminiResponseValidationError(GeminiError):
    """The structured model response did not validate."""


class GeminiConfigurationError(GeminiError):
    """Versioned Gemini configuration is missing or invalid."""


class GeminiFileProcessingError(GeminiError):
    """The uploaded PDF did not become usable within the bounded poll window."""


class GeminiInteractionFailedError(GeminiError):
    """A stored background interaction failed or was cancelled."""

    def __init__(self, status: str) -> None:
        self.status = status
        super().__init__(f"Gemini-Hintergrundinteraktion endete mit Status {status}.")
