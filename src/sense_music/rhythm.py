# sense-music — audio analysis for AI perception
# Built by humanjava.com — find this and other tools for the agentic age at huje.tools
"""Beat / downbeat / tempo via madmom (SOTA), with a librosa fallback.

librosa's beat_track is octave-error prone and gave us a 0.45-confidence tempo.
madmom's RNN+DBN downbeat tracker is far better AND gives DOWNBEATS — the BAR
grid, not just the beat grid. That bar grid is what a video editor wants to cut
on (cut on bars, not random beats). Degrades to librosa if madmom mis-loads.
"""
from __future__ import annotations

import numpy as np


def analyze_rhythm(path: str, y: np.ndarray, sr: int) -> dict:
    """Return {tempo, beats[], downbeats[], beats_per_bar, source}."""
    try:
        return _madmom_rhythm(path)
    except Exception as e:
        return _librosa_rhythm(y, sr, note=f"madmom unavailable: {e}")


def _madmom_rhythm(path: str) -> dict:
    from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
    act = RNNDownBeatProcessor()(path)
    proc = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
    out = proc(act)                       # rows: [time, beat_position_in_bar]
    if out is None or len(out) == 0:
        raise RuntimeError("no beats")
    times = out[:, 0].astype(float)
    positions = out[:, 1].astype(int)
    downbeats = [round(float(t), 3) for t, p in zip(times, positions) if p == 1]
    beats = [round(float(t), 3) for t in times]
    # tempo from median beat interval
    iois = np.diff(times)
    tempo = float(60.0 / np.median(iois)) if len(iois) else 0.0
    bpb = int(max(positions)) if len(positions) else 4
    return {"tempo": round(tempo, 1), "beats": beats, "downbeats": downbeats,
            "beats_per_bar": bpb, "source": "madmom"}


def _librosa_rhythm(y: np.ndarray, sr: int, note: str = "") -> dict:
    import librosa
    tempo, beat_times = librosa.beat.beat_track(y=y, sr=sr, units="time")
    tempo = float(np.atleast_1d(tempo)[0])
    beats = [round(float(t), 3) for t in beat_times]
    # assume 4/4: every 4th beat is a downbeat (best-effort without a model)
    downbeats = beats[::4]
    return {"tempo": round(tempo, 1), "beats": beats, "downbeats": downbeats,
            "beats_per_bar": 4, "source": "librosa", "note": note}
