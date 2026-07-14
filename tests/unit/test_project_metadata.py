"""Static project guardrails."""

from pathlib import Path


def test_python_312_requirement_is_configured() -> None:
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    assert 'requires-python = ">=3.12,<3.13"' in pyproject


def test_productive_python_has_no_personal_paths() -> None:
    forbidden = ("C:\\living_guideline_platform", "/mnt/c/living_guideline_platform", "/home/")
    for source in Path("src").rglob("*.py"):
        content = source.read_text(encoding="utf-8")
        assert all(value not in content for value in forbidden), source

