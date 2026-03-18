# sense-music — audio analysis for AI perception
# Built by humanjava.com — find this and other tools for the agentic age at huje.tools

from __future__ import annotations

import io

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image

from sense_music.types import Section
from sense_music._palette import BG_COLOR as _BG_COLOR, ACCENT as _ACCENT, TEXT_COLOR as _TEXT_COLOR, SECTION_COLORS as _SECTION_COLORS


def render_spectrogram(
    y: np.ndarray,
    sr: int,
    sections: list[Section] | None = None,
    energy_curve: list[float] | None = None,
    width: int = 1200,
    height: int = 500,
) -> Image.Image:
    """Render an annotated mel spectrogram as a PIL Image."""
    dpi = 100
    fig_w = width / dpi
    fig_h = height / dpi

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    try:
        fig.patch.set_facecolor(_BG_COLOR)
        ax.set_facecolor(_BG_COLOR)

        # mel spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=512)
        S_db = librosa.power_to_db(S, ref=np.max)
        duration = len(y) / sr

        img = librosa.display.specshow(
            S_db, sr=sr, hop_length=512, x_axis="time", y_axis="mel",
            ax=ax, cmap="magma",
        )

        # section markers
        if sections:
            for section in sections:
                color = _SECTION_COLORS.get(section.label, _ACCENT)
                ax.axvline(x=section.start, color=color, linewidth=1.5, alpha=0.8, linestyle="--")
                ax.text(
                    section.start + 0.5, ax.get_ylim()[1] * 0.92,
                    section.label, color=color, fontsize=8, fontweight="bold",
                    va="top",
                )

        # energy curve overlay
        if energy_curve:
            ax2 = ax.twinx()
            times = np.linspace(0, duration, len(energy_curve))
            ax2.plot(times, energy_curve, color=_ACCENT, linewidth=1.2, alpha=0.7)
            ax2.set_ylim(0, 1.5)
            ax2.set_ylabel("Energy", color=_ACCENT, fontsize=8)
            ax2.tick_params(axis="y", colors=_ACCENT, labelsize=7)
            ax2.spines["right"].set_color(_ACCENT)

        # style axes
        ax.set_xlabel("Time", color=_TEXT_COLOR, fontsize=9)
        ax.set_ylabel("Frequency (Hz)", color=_TEXT_COLOR, fontsize=9)
        ax.tick_params(colors=_TEXT_COLOR, labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(_TEXT_COLOR)

        # format time axis as mm:ss
        def _fmt_time(x, _pos):
            m, s = divmod(int(x), 60)
            return f"{m}:{s:02d}"
        ax.xaxis.set_major_formatter(plt.FuncFormatter(_fmt_time))

        fig.tight_layout(pad=0.5)

        # convert to PIL
        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight")
        buf.seek(0)
        return Image.open(buf).copy()
    finally:
        plt.close(fig)
