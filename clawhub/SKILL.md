---
name: sense-music
description: Audio analysis for AI perception — BPM, key, structure, genre, mood, lyrics, and annotated visualizations from any audio file.
version: 0.1.1
metadata:
  openclaw:
    requires:
      bins:
        - pip
    install:
      - kind: uv
        package: sense-music
        bins: []
    homepage: https://github.com/HumanjavaEnterprises/huje.sensemusic.OC-python.src
---

# sense-music — Audio Analysis for AI Perception

Turn any audio file into structured analysis and annotated visualizations. Detects BPM, musical key, song structure, genre, mood, and transcribes lyrics with timestamps. Generates annotated spectrograms and waveforms. Think of it as liner notes written by an AI, for an AI.

> **Import:** `pip install sense-music` → `from sense_music import analyze`

## Install

```bash
pip install sense-music
```

Dependencies: `librosa`, `matplotlib`, `Pillow`, `numpy`, `openai-whisper`.

## Quickstart

```python
from sense_music import analyze

result = analyze("song.mp3")
print(result.bpm.tempo)        # 120.0
print(result.key.key)          # "A"
print(result.key.mode)         # "minor"
print(result.genre)            # "electronic"
print(result.mood)             # ["energetic", "bright"]
print(result.summary)          # Natural language description
result.save("output/")         # Writes JSON, HTML, spectrogram.png, waveform.png
```

## Core Capabilities

### Analyze an Audio File

```python
from sense_music import analyze

result = analyze("song.mp3")
```

Works with local files (`.mp3`, `.wav`, `.flac`, `.ogg`, `.m4a`, `.aac`, `.wma`, `.opus`) and HTTP/HTTPS URLs.

### Structural Sections

```python
for section in result.sections:
    print(f"{section.label}: {section.start}s - {section.end}s")
# intro: 0.0s - 15.2s
# verse: 15.2s - 45.8s
# chorus: 45.8s - 76.3s
```

Section labels: `intro`, `verse`, `chorus`, `bridge`, `outro`, `instrumental`.

### Lyrics Transcription

```python
result = analyze("song.mp3", lyrics=True, whisper_model="base")

for line in result.lyrics:
    print(f"[{line.start:.1f}s] {line.text}")
```

Powered by Whisper. Disable with `lyrics=False` to skip transcription.

### Visualizations

```python
# Annotated mel spectrogram with section markers and energy curve
result.spectrogram  # PIL.Image.Image

# Waveform with colored section regions
result.waveform     # PIL.Image.Image

# Save everything
result.save("output/")  # spectrogram.png, waveform.png, analysis.json, analysis.html
```

### Export Formats

```python
# Structured dictionary (no images)
data = result.to_json()

# Self-contained HTML page with embedded images
html = result.to_html()

# Write HTML to file
result.render_page("analysis.html")
```

## Use Cases

- AI companions analyzing music shared by humans
- Automated liner notes for AI-generated tracks (Suno, Udio)
- Music production feedback — visualize structure and energy
- Accessibility — structured descriptions of audio content

## Response Format

### Analysis (returned by `analyze()`)

| Field | Type | Description |
|-------|------|-------------|
| `file_info` | `FileInfo` | Source audio metadata |
| `duration` | `float` | Length in seconds |
| `bpm` | `BPMInfo` | Tempo detection result |
| `key` | `KeyInfo` | Key detection result |
| `sections` | `list[Section]` | Structural segments |
| `lyrics` | `list[LyricLine]` | Transcribed lyrics with timestamps |
| `energy_curve` | `list[float]` | Per-second RMS energy (0.0–1.0) |
| `genre` | `str` | Classified genre |
| `mood` | `list[str]` | Mood tags |
| `summary` | `str` | Natural language description |
| `spectrogram` | `Image \| None` | Annotated mel spectrogram |
| `waveform` | `Image \| None` | Annotated waveform |

### BPMInfo

| Field | Type | Description |
|-------|------|-------------|
| `tempo` | `float` | Beats per minute |
| `confidence` | `float` | 0.0–1.0 detection confidence |

### KeyInfo

| Field | Type | Description |
|-------|------|-------------|
| `key` | `str` | Musical key (C, C#, D, ... B) |
| `mode` | `str` | "major" or "minor" |
| `confidence` | `float` | 0.0–1.0 detection confidence |

### Section

| Field | Type | Description |
|-------|------|-------------|
| `label` | `str` | intro, verse, chorus, bridge, outro, or instrumental |
| `start` | `float` | Start time in seconds |
| `end` | `float` | End time in seconds |

### LyricLine

| Field | Type | Description |
|-------|------|-------------|
| `start` | `float` | Start time in seconds |
| `end` | `float` | End time in seconds |
| `text` | `str` | Transcribed lyric text |

### FileInfo

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Filename from source |
| `duration` | `float` | Length in seconds |
| `sample_rate` | `int` | Hz (normalized to 22050) |
| `channels` | `int` | Always 1 (forced mono) |
| `format` | `str` | File extension |

## Common Patterns

### Analyze from URL

```python
result = analyze("https://example.com/track.mp3")
```

URLs are downloaded to a temp file. Private/loopback IPs are blocked (SSRF protection).

### Skip Lyrics for Speed

```python
result = analyze("song.mp3", lyrics=False)
# Skips Whisper — much faster
```

### Choose Whisper Model

```python
result = analyze("song.mp3", whisper_model="large-v3")
# Allowed: tiny, base, small, medium, large, large-v2, large-v3
```

### Cap Duration

```python
result = analyze("long-mix.mp3", max_duration=300)  # Cap at 5 minutes
```

## Security

- **SSRF protection.** URLs with private, loopback, or link-local IPs are blocked.
- **XSS protection.** All values in HTML output are escaped with `html.escape()`.
- **OOM prevention.** Audio capped at 600 seconds and 500 MB. Chroma subsampled to max 2000 frames.
- **Path traversal blocked.** `..` components rejected in save/render paths.
- **Whisper model allowlist.** Only approved model names accepted: `tiny`, `base`, `small`, `medium`, `large`, `large-v2`, `large-v3`.
- **No network access beyond URL downloads.** Analysis is entirely local.

## Configuration

### analyze() Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source` | `str` | required | File path or HTTP/HTTPS URL |
| `lyrics` | `bool` | `True` | Transcribe lyrics with Whisper |
| `whisper_model` | `str` | `"base"` | Whisper model size |
| `max_duration` | `float` | `600` | Max audio length in seconds |

### Supported Formats

`.mp3`, `.wav`, `.flac`, `.ogg`, `.m4a`, `.aac`, `.wma`, `.opus`

### Genre Classification

Rule-based heuristics returning one of: `rock`, `electronic`, `ambient`, `dance`, `acoustic`, `r&b`, `pop`.

### Mood Tags

Possible values: `energetic`, `calm`, `bright`, `warm`, `uplifting`, `contemplative`, `neutral`.

## Links

- [PyPI](https://pypi.org/project/sense-music/)
- [GitHub](https://github.com/HumanjavaEnterprises/huje.sensemusic.OC-python.src)
- [huje.tools](https://huje.tools)
- [ClawHub](https://clawhub.ai/u/vveerrgg)

License: MIT
