"""Typed, explicitly loaded local configuration."""

from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Local settings; an env file is loaded only when explicitly supplied."""

    model_config = SettingsConfigDict(
        env_file=None,
        env_ignore_empty=True,
        extra="ignore",
        case_sensitive=True,
        validate_default=True,
    )

    worker_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("AISURGEON_WORKER_ID", "worker_id"),
    )
    data_root: Path | None = Field(
        default=None,
        validation_alias=AliasChoices("AISURGEON_DATA_ROOT", "data_root"),
    )
    pdf_source_dir: Path | None = Field(
        default=None,
        validation_alias=AliasChoices("AISURGEON_PDF_SOURCE_DIR", "pdf_source_dir"),
    )
    runs_dir: Path | None = Field(
        default=None,
        validation_alias=AliasChoices("AISURGEON_RUNS_DIR", "runs_dir"),
    )
    cache_dir: Path | None = Field(
        default=None,
        validation_alias=AliasChoices("AISURGEON_CACHE_DIR", "cache_dir"),
    )
    exports_dir: Path | None = Field(
        default=None,
        validation_alias=AliasChoices("AISURGEON_EXPORTS_DIR", "exports_dir"),
    )
    logs_dir: Path | None = Field(
        default=None,
        validation_alias=AliasChoices("AISURGEON_LOGS_DIR", "logs_dir"),
    )
    gemini_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("GEMINI_API_KEY", "gemini_api_key"),
    )
    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_API_KEY", "openai_api_key"),
    )
    ncbi_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("NCBI_API_KEY", "ncbi_api_key"),
    )
    ncbi_email: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("NCBI_EMAIL", "ncbi_email"),
    )
    ncbi_tool: str | None = Field(
        default=None,
        validation_alias=AliasChoices("NCBI_TOOL", "ncbi_tool"),
    )

    @model_validator(mode="after")
    def derive_output_paths(self) -> "Settings":
        """Derive output directories from data_root unless individually configured."""
        if self.data_root is not None:
            self.runs_dir = self.runs_dir or self.data_root / "runs"
            self.cache_dir = self.cache_dir or self.data_root / "cache"
            self.exports_dir = self.exports_dir or self.data_root / "exports"
            self.logs_dir = self.logs_dir or self.data_root / "logs"
        return self

    @classmethod
    def from_env_file(cls, env_file: Path | None = None, **overrides: object) -> "Settings":
        """Load process environment plus one explicitly selected env file."""
        source = {key: value for key, value in overrides.items() if value is not None}
        if env_file is None:
            return cls(**source)
        return cls(_env_file=env_file, _env_file_encoding="utf-8", **source)


SERVICE_FIELDS = (
    ("GEMINI_API_KEY", "gemini_api_key"),
    ("OPENAI_API_KEY", "openai_api_key"),
    ("NCBI_API_KEY", "ncbi_api_key"),
    ("NCBI_EMAIL", "ncbi_email"),
)
