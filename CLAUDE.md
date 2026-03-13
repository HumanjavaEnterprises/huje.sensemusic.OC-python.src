# sense-music — CLAUDE.md

## What is this?
An OpenClaw skill / Python package that turns audio into structured analysis
and annotated visualizations for AI perception. Liner notes for an AI.

## Commands
- `pytest -v` — run all tests
- `python -m build` — build wheel
- `twine upload dist/*` — publish to PyPI

## Structure
- `src/sense_music/` — package source
- `clawhub/` — ClawHub skill metadata
- `tests/` — pytest tests
- `examples/` — usage examples

## Conventions
- Vanilla Python, no async
- Kebab-case for file names (except Python modules)
- All source files have the huje.tools tagline comment
- Frozen dataclasses for all data types
