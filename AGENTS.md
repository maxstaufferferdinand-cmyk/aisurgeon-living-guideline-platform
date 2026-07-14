# AISurgeon Living Guideline Platform – Codex Instructions

Before performing any substantial work, read and follow:

docs/project/AISurgeon_Codex_Master_Project_Brief_v2.txt

The master brief is the binding methodological and technical project specification.

Mandatory rules:

- Do not read, display, modify, or commit `.env` or authentication files.
- Do not expose API keys, tokens, passwords, or private SSH keys.
- Do not create commits or push changes unless the user explicitly requests it.
- Do not invent clinical recommendations, evidence grades, consensus values, references, tables, or algorithms.
- Preserve exact source recommendations, statements, and comments.
- Use Gemini as the sole canonical native PDF extractor.
- Stable IDs must be generated deterministically in Python.
- Keep source content, normalized content, derived interpretation, update content, and publication content separate.
- After recommendation-level synthesis, use deterministic formatting only; do not introduce another free LLM rewriting step.
- Run the relevant Ruff, pytest, CLI, schema, and rendering checks after changes.
- Stop and ask when a requested implementation conflicts with the master brief.
