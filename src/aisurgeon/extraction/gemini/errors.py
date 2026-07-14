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

