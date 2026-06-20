# sense-music — audio analysis for AI perception
# Built by humanjava.com — find this and other tools for the agentic age at huje.tools

from sense_music.analyze import analyze
from sense_music.types import Analysis, Section, Motif, LyricLine, FileInfo, BPMInfo, KeyInfo

__version__ = "0.2.0"
__all__ = [
    "analyze",
    "Analysis",
    "Section",
    "Motif",
    "LyricLine",
    "FileInfo",
    "BPMInfo",
    "KeyInfo",
]
