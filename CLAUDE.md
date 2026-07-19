# sense-music — CLAUDE.md

## What is this?
An OpenClaw skill / Python package that turns audio into structured analysis
and annotated visualizations for AI perception. Liner notes for an AI.

Second huje.tools product. First one that ties the full Humanjava stack together:
NostrKey (identity) + NWC (Lightning payments) + huje.tools (hosting).

## Commands
- `pytest -v` — run all tests (51 tests, including 18 security tests)
- `python -m build` — build wheel
- `twine upload dist/*` — publish to PyPI (needs API token)
- `npx clawhub publish ./clawhub --slug sense-music --name "sense-music" --version X.Y.Z --tags latest --changelog "..."` — publish to ClawHub (npm CLI, not Python)

## Structure
- `src/sense_music/` — package source
  - `analyze.py` — main entry point, URL resolution, file validation
  - `types.py` — frozen dataclasses (Analysis, Section, LyricLine, etc.)
  - `features.py` — BPM, key, energy, genre, mood detection
  - `sections.py` — structural segmentation via self-similarity
  - `lyrics.py` — Whisper transcription with model allowlist
  - `spectrogram.py` — annotated mel spectrogram rendering
  - `waveform.py` — waveform visualization with section regions
  - `output.py` — JSON, HTML, save() with XSS protection
- `clawhub/` — ClawHub skill metadata
- `tests/` — pytest tests (including test_security.py)
- `examples/` — usage examples

## Security
Hardened in v0.1.1 against: SSRF (private IP blocklist), XSS (html.escape on all output),
OOM (duration cap, file size limit, chroma subsampling), path traversal (.. blocked),
whisper model allowlist, matplotlib figure leak protection.
v0.1.4 (coordinated 2026-07 correctness release, staged/pending publish) extends the SSRF
guard to redirects and DNS rebinding: per-hop address checks, DNS pinning to the vetted IP,
manual redirect following capped at 5 hops, and a streamed body size cap.

## Conventions
- Vanilla Python, no async
- Kebab-case for file names (except Python modules)
- All source files have the huje.tools tagline comment
- Frozen dataclasses for all data types
- Version must be bumped in 3 places: pyproject.toml, __init__.py, clawhub/metadata.json
