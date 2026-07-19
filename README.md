# sense-music

Turn audio into structured analysis and annotated visualizations for AI perception. Liner notes for an AI.

Built by [humanjava.com](https://humanjava.com) — find this and other tools for the agentic age at [huje.tools](https://huje.tools).

## Install

```bash
pip install sense-music              # core: sections, loops, key, energy, spectrogram, lyrics
pip install "sense-music[full]"      # + deep perception (CLAP, madmom, Demucs, Qwen2-Audio)
```

Deep-perception layers are optional extras (`embedding`, `rhythm`, `stems`, `caption`, `loudness`) —
each degrades gracefully if its dependency is absent. ⚠️ On Python 3.12, `madmom` needs the git build
(`pip install git+https://github.com/CPJKU/madmom.git`); the PyPI 0.16.1 won't build on 3.12.

> **Security:** `analyze()` URL fetching is hardened against SSRF via redirects and DNS
> rebinding — per-hop address checks, DNS pinning to the vetted IP, a 5-hop redirect cap, and a
> streamed body size cap. See [`CHANGELOG.md`](./CHANGELOG.md).

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
| `result.motifs` | Recurring LOOPS (Motif label, count, occurrences) — which sections reprise |
| `result.structure` | Motif sequence, e.g. `"A-B-A-A-C-A"` |
| `result.key_changes` | Modulation timeline (per-section key changes) |
| `result.rhythm` | madmom beats/**downbeats**/tempo + bar grid (`rhythm=True`) |
| `result.chords` | Chord progression + timeline (`chords=True`) |
| `result.loudness` | LUFS + crest factor |
| `result.clap_tags` | CLAP zero-shot semantic tags (`clap_tags=True`) |
| `result.embedding` | CLAP 512-d audio embedding — a similarity metric ("does this sound like X") (`embedding=True`) |
| `result.arrangement` | Demucs stem activity + element in/out timeline (`stems=True`) |
| `result.caption` | Qwen2-Audio free-text liner notes (`caption=True`) |

## Deep perception (v0.3)

Each layer is an `analyze()` flag, fail-soft if its dep is missing:

```python
result = analyze("song.mp3", rhythm=True, embedding=True, clap_tags=True,
                 chords=True, stems=True, caption=False)
```

- **rhythm** (madmom) — SOTA beat/**downbeat** tracking → the BAR grid (the thing video editors cut on).
- **embedding + clap_tags** (CLAP) — a 512-d audio embedding (the similarity metric) + zero-shot tags.
- **stems** (Demucs) — source separation → an arrangement timeline (which element enters/exits when).
- **chords** (madmom) — chord-progression recognition.
- **caption** (Qwen2-Audio) — natural-language "liner notes" (heavy; loads a 7B model).

### Cut grid (for video editing)

```python
from sense_music.cutgrid import edit_points, match_reference
pts = edit_points(result, snap=True)             # bar-aligned, ranked edit points
hits = match_reference([4.1, 8.0, 12.2], result) # what song event each reference cut lands on
```

## Dependencies

- [librosa](https://librosa.org/) — audio analysis · [matplotlib](https://matplotlib.org/) — visualization · [Pillow](https://pillow.readthedocs.io/) — image handling
- [openai-whisper](https://github.com/openai/whisper) — lyrics (optional via `lyrics=False`)
- **Deep-perception extras:** [transformers](https://github.com/huggingface/transformers) (CLAP + Qwen2-Audio), [madmom](https://github.com/CPJKU/madmom) (beat/downbeat/chords), [demucs](https://github.com/adefossez/demucs) (stems), [pyloudnorm](https://github.com/csteinmetz1/pyloudnorm) (LUFS)

## Usage & Copyright

You are responsible for ensuring you have the legal right to analyze any audio you submit to this tool, whether running locally or via the hosted service at [huje.tools](https://huje.tools). sense-music provides compute and analysis only — it does not store, host, or redistribute audio content. By using this tool, you accept full responsibility for the content you process and how you use the results.

For details, see [huje.tools/support](https://huje.tools/support).

## License

MIT — Humanjava Enterprises Inc.
