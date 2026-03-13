# sense-music

Turn audio into structured analysis and annotated visualizations for AI perception. Liner notes for an AI.

Built by [humanjava.com](https://humanjava.com) — find this and other tools for the agentic age at [huje.tools](https://huje.tools).

## Install

```bash
pip install sense-music
```

## Quick Start

```python
from sense_music import analyze

result = analyze("song.mp3")
print(result.summary)
result.save("output/")
```

## Full Example

```python
from sense_music import analyze

result = analyze("song.mp3")

# Structured data
print(f"{result.bpm.tempo} BPM, {result.key.key} {result.key.mode}")
print(f"Genre: {result.genre}, Mood: {', '.join(result.mood)}")

# Sections
for s in result.sections:
    print(f"  {s.label}: {s.start:.1f}s — {s.end:.1f}s")

# Lyrics (requires whisper)
for line in result.lyrics:
    print(f"  [{line.start:.1f}s] {line.text}")

# Save everything
result.save("output/")           # spectrogram.png, waveform.png, analysis.json, analysis.html
result.render_page("song.html")  # self-contained HTML report
```

## Skip Lyrics

If you don't have Whisper installed or want faster analysis:

```python
result = analyze("song.mp3", lyrics=False)
```

## What You Get

| Output | Description |
|--------|-------------|
| `result.spectrogram` | PIL Image — annotated mel spectrogram with section markers and energy curve |
| `result.waveform` | PIL Image — waveform with colored section regions |
| `result.bpm` | BPMInfo(tempo, confidence) |
| `result.key` | KeyInfo(key, mode, confidence) |
| `result.sections` | List of Section(label, start, end) |
| `result.lyrics` | List of LyricLine(start, end, text) |
| `result.energy_curve` | Per-second normalized energy values |
| `result.genre` | Simple genre classification |
| `result.mood` | List of mood descriptors |
| `result.summary` | Natural language track description |

## Dependencies

- [librosa](https://librosa.org/) — audio analysis
- [matplotlib](https://matplotlib.org/) — visualization
- [Pillow](https://pillow.readthedocs.io/) — image handling
- [openai-whisper](https://github.com/openai/whisper) — lyrics transcription (optional via `lyrics=False`)

## Usage & Copyright

You are responsible for ensuring you have the legal right to analyze any audio you submit to this tool, whether running locally or via the hosted service at [huje.tools](https://huje.tools). sense-music provides compute and analysis only — it does not store, host, or redistribute audio content. By using this tool, you accept full responsibility for the content you process and how you use the results.

For details, see [huje.tools/support](https://huje.tools/support).

## License

MIT — Humanjava Enterprises Inc.
