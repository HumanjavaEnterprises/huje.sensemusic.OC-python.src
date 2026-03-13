# sense-music — audio analysis for AI perception
# Built by humanjava.com — find this and other tools for the agentic age at huje.tools

from __future__ import annotations

import base64
import html
import io
import json
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sense_music.types import Analysis


def _esc(value: str) -> str:
    """HTML-escape a string to prevent XSS."""
    return html.escape(str(value), quote=True)


def to_json(analysis: Analysis) -> dict:
    """Convert analysis to a structured dict."""
    return {
        "file": {
            "name": analysis.file_info.name,
            "duration": analysis.file_info.duration,
            "sample_rate": analysis.file_info.sample_rate,
            "channels": analysis.file_info.channels,
            "format": analysis.file_info.format,
        },
        "duration": analysis.duration,
        "bpm": {
            "tempo": analysis.bpm.tempo,
            "confidence": analysis.bpm.confidence,
        },
        "key": {
            "key": analysis.key.key,
            "mode": analysis.key.mode,
            "confidence": analysis.key.confidence,
        },
        "sections": [
            {"label": s.label, "start": s.start, "end": s.end}
            for s in analysis.sections
        ],
        "lyrics": [
            {"start": l.start, "end": l.end, "text": l.text}
            for l in analysis.lyrics
        ],
        "energy_curve": analysis.energy_curve,
        "genre": analysis.genre,
        "mood": analysis.mood,
        "summary": analysis.summary,
    }


def _image_to_base64(img) -> str:
    """Convert a PIL Image to a base64-encoded PNG string."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def to_html(analysis: Analysis) -> str:
    """Generate a self-contained HTML analysis page."""
    data = to_json(analysis)

    spec_b64 = _image_to_base64(analysis.spectrogram) if analysis.spectrogram else ""
    wave_b64 = _image_to_base64(analysis.waveform) if analysis.waveform else ""

    sections_html = ""
    for s in analysis.sections:
        m1, s1 = divmod(int(s.start), 60)
        m2, s2 = divmod(int(s.end), 60)
        sections_html += (
            f'<div class="section-tag">'
            f'<span class="label">{_esc(s.label)}</span> '
            f'{m1}:{s1:02d} — {m2}:{s2:02d}'
            f'</div>\n'
        )

    lyrics_html = ""
    for line in analysis.lyrics:
        m, s = divmod(int(line.start), 60)
        lyrics_html += f'<div class="lyric"><span class="ts">{m}:{s:02d}</span> {_esc(line.text)}</div>\n'

    mood_tags = " ".join(f'<span class="tag">{_esc(m)}</span>' for m in analysis.mood)

    dm, ds = divmod(int(analysis.duration), 60)

    file_name = _esc(analysis.file_info.name)
    file_format = _esc(analysis.file_info.format)
    genre = _esc(analysis.genre)
    key_str = _esc(f"{analysis.key.key} {analysis.key.mode}")
    summary = _esc(analysis.summary)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>sense-music — {file_name}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0f172a;color:#e2e8f0;font-family:system-ui,-apple-system,sans-serif;padding:2rem;max-width:1200px;margin:0 auto}}
h1{{color:#f97316;font-size:1.5rem;margin-bottom:0.5rem}}
h2{{color:#f97316;font-size:1.1rem;margin:1.5rem 0 0.5rem;border-bottom:1px solid #334155;padding-bottom:0.25rem}}
.meta{{color:#94a3b8;font-size:0.85rem;margin-bottom:1rem}}
.card{{background:#1e293b;border-radius:8px;padding:1rem;margin-bottom:1rem}}
img{{width:100%;border-radius:4px;margin:0.5rem 0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:0.75rem}}
.stat{{background:#334155;border-radius:6px;padding:0.75rem;text-align:center}}
.stat .value{{font-size:1.5rem;font-weight:700;color:#f97316}}
.stat .label{{font-size:0.75rem;color:#94a3b8;text-transform:uppercase}}
.section-tag{{display:inline-block;background:#334155;border-radius:4px;padding:0.25rem 0.5rem;margin:0.25rem;font-size:0.8rem}}
.section-tag .label{{color:#f97316;font-weight:700;text-transform:uppercase}}
.tag{{display:inline-block;background:#334155;border-radius:4px;padding:0.2rem 0.5rem;margin:0.15rem;font-size:0.8rem}}
.lyric{{font-size:0.85rem;padding:0.2rem 0;border-bottom:1px solid #1e293b}}
.lyric .ts{{color:#f97316;font-size:0.75rem;margin-right:0.5rem}}
.summary{{font-style:italic;color:#cbd5e1;line-height:1.5}}
.footer{{text-align:center;color:#64748b;font-size:0.75rem;margin-top:2rem}}
a{{color:#f97316}}
</style>
</head>
<body>
<h1>sense-music</h1>
<div class="meta">{file_name} &middot; {dm}:{ds:02d} &middot; {file_format}</div>

<div class="card grid">
  <div class="stat"><div class="value">{analysis.bpm.tempo}</div><div class="label">BPM</div></div>
  <div class="stat"><div class="value">{key_str}</div><div class="label">Key</div></div>
  <div class="stat"><div class="value">{genre}</div><div class="label">Genre</div></div>
  <div class="stat"><div class="value">{dm}:{ds:02d}</div><div class="label">Duration</div></div>
</div>

<h2>Spectrogram</h2>
<div class="card">{"<img src='data:image/png;base64," + spec_b64 + "' alt='Spectrogram'>" if spec_b64 else "<p>No spectrogram</p>"}</div>

<h2>Waveform</h2>
<div class="card">{"<img src='data:image/png;base64," + wave_b64 + "' alt='Waveform'>" if wave_b64 else "<p>No waveform</p>"}</div>

<h2>Sections</h2>
<div class="card">{sections_html}</div>

<h2>Mood</h2>
<div class="card">{mood_tags}</div>

{"<h2>Lyrics</h2><div class='card'>" + lyrics_html + "</div>" if lyrics_html else ""}

<h2>What this sounds like</h2>
<div class="card"><p class="summary">{summary}</p></div>

<div class="footer">Generated by <a href="https://huje.tools">sense-music</a> &middot; Built by <a href="https://humanjava.com">humanjava.com</a></div>
</body>
</html>"""


def save(analysis: Analysis, directory: str) -> None:
    """Save all analysis outputs to a directory."""
    _validate_output_path(directory)
    os.makedirs(directory, exist_ok=True)

    # images
    if analysis.spectrogram:
        analysis.spectrogram.save(os.path.join(directory, "spectrogram.png"))
    if analysis.waveform:
        analysis.waveform.save(os.path.join(directory, "waveform.png"))

    # json
    with open(os.path.join(directory, "analysis.json"), "w") as f:
        json.dump(to_json(analysis), f, indent=2)

    # html
    with open(os.path.join(directory, "analysis.html"), "w") as f:
        f.write(to_html(analysis))


def render_page(analysis: Analysis, path: str) -> None:
    """Save just the HTML page to a file."""
    _validate_output_path(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        f.write(to_html(analysis))


def _validate_output_path(path: str) -> None:
    """Reject paths that attempt traversal via '..' components."""
    resolved = os.path.realpath(path)
    # block if the path contains .. traversal that escapes the intended directory
    if ".." in os.path.normpath(path).split(os.sep):
        raise ValueError(f"Path traversal not allowed: {path}")
