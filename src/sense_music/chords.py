# sense-music — audio analysis for AI perception
# Built by humanjava.com — find this and other tools for the agentic age at huje.tools
"""Chord recognition via madmom, with a chroma-template fallback.

The founder's drama is HARMONIC — but the per-section key detector is noisy on a
steady loop (Krumhansl flips between relative/neighbor keys). A chord model turns
that jitter into a truthful progression (Cm -> Ab -> Eb ...), which is far more
honest about what's actually happening over the loop. Returns a chord timeline +
a compact unique-progression summary.
"""
from __future__ import annotations

import numpy as np

_NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def analyze_chords(path: str, y: np.ndarray, sr: int) -> dict:
    """Return {timeline:[{start,end,chord}], progression:[...], source}."""
    try:
        return _madmom_chords(path)
    except Exception as e:
        return _template_chords(y, sr, note=f"madmom unavailable: {e}")


def _collapse(timeline: list[dict]) -> list[str]:
    """Unique consecutive chord labels — the progression as a readable sequence."""
    prog, prev = [], None
    for c in timeline:
        if c["chord"] != prev and c["chord"] != "N":   # N = no-chord
            prog.append(c["chord"]); prev = c["chord"]
    return prog


def _madmom_chords(path: str) -> dict:
    from madmom.features.chords import CNNChordFeatureProcessor, CRFChordRecognitionProcessor
    feats = CNNChordFeatureProcessor()(path)
    chords = CRFChordRecognitionProcessor()(feats)   # [(start, end, label)]
    timeline = [{"start": round(float(s), 2), "end": round(float(e), 2), "chord": str(l)}
                for s, e, l in chords]
    return {"timeline": timeline, "progression": _collapse(timeline), "source": "madmom"}


def _template_chords(y: np.ndarray, sr: int, note: str = "") -> dict:
    """Fallback: per-beat-ish chroma matched to 24 major/minor triad templates."""
    import librosa
    hop = 4096
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop)
    maj = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0.0])
    minr = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0.0])
    templates, labels = [], []
    for i in range(12):
        templates.append(np.roll(maj, i)); labels.append(f"{_NOTES[i]}:maj")
        templates.append(np.roll(minr, i)); labels.append(f"{_NOTES[i]}:min")
    T = np.array(templates)
    timeline = []
    for j in range(chroma.shape[1]):
        v = chroma[:, j]
        if v.sum() < 1e-6:
            lab = "N"
        else:
            lab = labels[int(np.argmax(T @ (v / (np.linalg.norm(v) + 1e-9))))]
        timeline.append({"start": round(float(times[j]), 2), "chord": lab})
    # merge consecutive equal labels into spans
    merged = []
    for c in timeline:
        if merged and merged[-1]["chord"] == c["chord"]:
            merged[-1]["end"] = c["start"]
        else:
            if merged:
                merged[-1]["end"] = c["start"]
            merged.append({"start": c["start"], "end": c["start"], "chord": c["chord"]})
    return {"timeline": merged, "progression": _collapse(merged),
            "source": "chroma-template", "note": note}
