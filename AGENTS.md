# AGENTS.md — watson-lite

## Overview

Extractive QA pipeline that answers factual questions using Wikipedia as its
knowledge base. No LLM, no trained weights, no paid APIs. All components are
off-the-shelf pretrained models running CPU-only inference.

## Commands

| Command | Description |
|---------|------------|
| `pytest` | Run test suite |
| `ruff format src/ tests/` | Format code |
| `prospector --with-tool ruff --with-tool mypy src/` | Lint + type check |
| `vulture --min-confidence 90 src/` | Dead/unused code detection |

## Development

```bash
pip install -e ".[test]"
pytest
ruff format src/ tests/
prospector --with-tool ruff --with-tool mypy src/
```

## Testing

Tests use pytest with coverage. Run `pytest -v` to execute all tests.

## Code Style

- Format: ruff format
- Lint + Type check: prospector (ruff + mypy)
- Docstrings: Google style

## Release

```bash
bumpversion patch
git tag v<version>
git push && git push --tags
```
