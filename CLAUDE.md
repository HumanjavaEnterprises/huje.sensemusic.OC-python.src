# sense-music — CLAUDE.md

## What is this?
An OpenClaw skill / Python package that turns audio into structured analysis
and annotated visualizations for AI perception. Liner notes for an AI.

Second huje.tools product. First one that ties the full Humanjava stack together:
NostrKey (identity) + NWC (Lightning payments) + huje.tools (hosting).

## Deep-perception layers (v0.3) — install + Blackwell/py3.12 notes
`analyze()` flags gate each: `rhythm=` `chords=` `embedding=` `clap_tags=` `stems=` `caption=`
(rhythm/embedding/clap_tags default ON; chords/stems/caption opt-in/heavy). Each degrades
gracefully if its dep or model is missing. Install groups: `pip install -e ".[full]"`.
- **CLAP** (embedding + tags): uses HF `transformers` `ClapModel` (no laion_clap dep). ⚠️
  transformers ≥5 renamed the processor kwarg `audios`→`audio` (the v0.3 fix). Weights
  `laion/clap-htsat-unfused` (~1.2G) auto-download. The embedding is the qualifier metric.
- **madmom** (rhythm + chords): py3.12 needs the **git build** (`pip install --no-build-isolation
  git+https://github.com/CPJKU/madmom.git` → 0.17.dev0); the PyPI 0.16.1 won't build on 3.12.
  Bundles its own beat/downbeat/chord models. CPU.
- **Demucs** (stems): `pip install demucs` (4.0.1). `htdemucs` weights auto-download. GPU, ~10-30s/track.
- **Qwen2-Audio** (caption): HF classes ship with transformers; the 7B weights download on first use.
- **GPU doctrine:** pin the bench 4000 by UUID (`CUDA_VISIBLE_DEVICES=$BENCH`), never a voice card.

## Commands
- `pytest -v` — run all tests (65 tests, including 18 security tests)
- `python -m build` — build wheel
- `twine upload dist/*` — publish to PyPI (needs API token)
- `npx clawhub publish ./clawhub --slug sense-music --name "sense-music" --version X.Y.Z --tags latest --changelog "..."` — publish to ClawHub (npm CLI, not Python)

## Structure
- `src/sense_music/` — package source
  - `analyze.py` — main entry point, URL resolution, file validation
  - `types.py` — frozen dataclasses (Analysis, Section, LyricLine, etc.)
  - `features.py` — BPM, key, energy, genre, mood detection
  - `sections.py` — structural segmentation via self-similarity + relative-energy narrative labels (intro/build/groove/peak/breakdown/bridge/outro)
  - `loops.py` — loop/motif detection (which sections are reprises → A-B-A-C structure) + per-section key (modulation) timeline. The "narrative" layer for loop-driven composition.
  - `rhythm.py` — beat/downbeat/tempo via **madmom** (SOTA, gives the BAR grid), librosa fallback.
  - `chords.py` — chord-progression recognition via **madmom** (CNN+CRF), chroma-template fallback.
  - `embedding.py` — **CLAP** (HF transformers) audio embedding + zero-shot semantic tags. The embedding IS the music-qualifier's similarity metric ("does this sound like me?").
  - `stems.py` — **Demucs** source separation → arrangement timeline (element in/out over the loop). The layer that reads NARRATIVE. Opt-in (heavy).
  - `caption.py` — **Qwen2-Audio** free-text "liner notes". Opt-in (loads a 7B model).
  - `cutgrid.py` — fuse downbeats+sections+arrangement+chords+key into a ranked, **bar-aligned, repeatable** edit-point list; `match_reference()` reads a reference video's cut times against the song grid. The bridge to video editing.
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
