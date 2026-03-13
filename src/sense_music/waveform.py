# sense-music — audio analysis for AI perception
# Built by humanjava.com — find this and other tools for the agentic age at huje.tools

from __future__ import annotations

import io

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from sense_music.types import Section


# huje.tools palette
_BG_COLOR = "#0f172a"
_ACCENT = "#f97316"
_TEXT_COLOR = "#e2e8f0"

_SECTION_COLORS = {
    "intro": "#38bdf8",
    "verse": "#818cf8",
    "chorus": "#f97316",
    "bridge": "#a78bfa",
    "outro": "#38bdf8",
    "instrumental": "#2dd4bf",
}


def render_waveform(
    y: np.ndarray,
    sr: int,
    sections: list[Section] | None = None,
    width: int = 1200,
    height: int = 300,
) -> Image.Image:
    """Render a waveform visualization with section boundaries as a PIL Image."""
    dpi = 100
    fig_w = width / dpi
    fig_h = height / dpi

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor(_BG_COLOR)
    ax.set_facecolor(_BG_COLOR)

    duration = len(y) / sr
    times = np.linspace(0, duration, len(y))

    # draw colored section regions
    if sections:
        for section in sections:
            color = _SECTION_COLORS.get(section.label, _ACCENT)
            ax.axvspan(section.start, section.end, alpha=0.15, color=color)
            mid = (section.start + section.end) / 2
            ax.text(
                mid, ax.get_ylim()[1] if ax.get_ylim()[1] != 0 else 0.9,
                section.label, color=color, fontsize=8, fontweight="bold",
                ha="center", va="top",
            )

    # waveform
    ax.plot(times, y, color=_ACCENT, linewidth=0.3, alpha=0.8)
    ax.fill_between(times, y, alpha=0.3, color=_ACCENT)

    # section boundary lines
    if sections:
        for section in sections:
            ax.axvline(x=section.start, color=_TEXT_COLOR, linewidth=0.8, alpha=0.5, linestyle=":")

    # style
    ax.set_xlabel("Time", color=_TEXT_COLOR, fontsize=9)
    ax.set_ylabel("Amplitude", color=_TEXT_COLOR, fontsize=9)
    ax.tick_params(colors=_TEXT_COLOR, labelsize=7)
    ax.set_xlim(0, duration)
    for spine in ax.spines.values():
        spine.set_color(_TEXT_COLOR)

    # format time axis as mm:ss
    def _fmt_time(x, _pos):
        m, s = divmod(int(x), 60)
        return f"{m}:{s:02d}"
    ax.xaxis.set_major_formatter(plt.FuncFormatter(_fmt_time))

    fig.tight_layout(pad=0.5)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()
