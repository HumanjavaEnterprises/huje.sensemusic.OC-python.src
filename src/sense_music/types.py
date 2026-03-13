# sense-music — audio analysis for AI perception
# Built by humanjava.com — find this and other tools for the agentic age at huje.tools

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from PIL import Image


@dataclass(frozen=True)
class FileInfo:
    """Metadata about the source audio file."""
    name: str
    duration: float
    sample_rate: int
    channels: int
    format: str


@dataclass(frozen=True)
class BPMInfo:
    """Beat tempo detection result."""
    tempo: float
    confidence: float


@dataclass(frozen=True)
class KeyInfo:
    """Musical key detection result."""
    key: str
    mode: str
    confidence: float


@dataclass(frozen=True)
class Section:
    """A structural segment of the track."""
    label: str
    start: float
    end: float


@dataclass(frozen=True)
class LyricLine:
    """A single line of transcribed lyrics with timestamps."""
    start: float
    end: float
    text: str


@dataclass
class Analysis:
    """Complete analysis result for an audio track."""
    file_info: FileInfo
    duration: float
    bpm: BPMInfo
    key: KeyInfo
    sections: list[Section]
    lyrics: list[LyricLine]
    energy_curve: list[float]
    genre: str
    mood: list[str]
    summary: str
    spectrogram: Optional[Image.Image] = field(default=None, repr=False)
    waveform: Optional[Image.Image] = field(default=None, repr=False)

    def save(self, directory: str) -> None:
        from sense_music.output import save
        save(self, directory)

    def to_json(self) -> dict:
        from sense_music.output import to_json
        return to_json(self)

    def to_html(self) -> str:
        from sense_music.output import to_html
        return to_html(self)

    def render_page(self, path: str) -> None:
        from sense_music.output import render_page
        render_page(self, path)
