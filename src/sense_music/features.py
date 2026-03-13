# sense-music — audio analysis for AI perception
# Built by humanjava.com — find this and other tools for the agentic age at huje.tools

from __future__ import annotations

import numpy as np
import librosa

from sense_music.types import BPMInfo, KeyInfo


# Krumhansl-Schmuckler key profiles
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                           2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                           2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

_KEY_NAMES = ["C", "C#", "D", "D#", "E", "F",
              "F#", "G", "G#", "A", "A#", "B"]


def detect_bpm(y: np.ndarray, sr: int) -> BPMInfo:
    """Detect BPM using librosa beat tracking."""
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    # librosa may return an array; take scalar
    t = float(np.atleast_1d(tempo)[0])
    # confidence from beat strength
    if len(beats) > 1:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        beat_strengths = onset_env[beats]
        confidence = float(np.clip(np.mean(beat_strengths) / (np.max(onset_env) + 1e-6), 0, 1))
    else:
        confidence = 0.0
    return BPMInfo(tempo=round(t, 1), confidence=round(confidence, 2))


def detect_key(y: np.ndarray, sr: int) -> KeyInfo:
    """Detect musical key via chroma features + Krumhansl-Schmuckler."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_vals = np.mean(chroma, axis=1)

    best_corr = -2.0
    best_key = 0
    best_mode = "major"

    for i in range(12):
        rolled = np.roll(chroma_vals, -i)
        major_corr = float(np.corrcoef(rolled, _MAJOR_PROFILE)[0, 1])
        minor_corr = float(np.corrcoef(rolled, _MINOR_PROFILE)[0, 1])
        if major_corr > best_corr:
            best_corr = major_corr
            best_key = i
            best_mode = "major"
        if minor_corr > best_corr:
            best_corr = minor_corr
            best_key = i
            best_mode = "minor"

    confidence = max(0.0, min(1.0, (best_corr + 1) / 2))
    return KeyInfo(
        key=_KEY_NAMES[best_key],
        mode=best_mode,
        confidence=round(confidence, 2),
    )


def compute_energy(y: np.ndarray, sr: int) -> list[float]:
    """Compute per-second RMS energy curve."""
    hop = sr  # one value per second
    rms_frames = []
    for i in range(0, len(y), hop):
        frame = y[i:i + hop]
        if len(frame) == 0:
            break
        rms_frames.append(float(np.sqrt(np.mean(frame ** 2))))
    # normalize to 0-1
    peak = max(rms_frames) if rms_frames else 1.0
    if peak > 0:
        rms_frames = [round(v / peak, 3) for v in rms_frames]
    return rms_frames


def classify_genre(y: np.ndarray, sr: int) -> str:
    """Simple rule-based genre classification from spectral features."""
    spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    spectral_rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
    tempo = float(np.atleast_1d(librosa.beat.beat_track(y=y, sr=sr)[0])[0])

    # simple heuristic boundaries
    if zcr > 0.15:
        return "rock"
    if spectral_centroid > 3000 and tempo > 120:
        return "electronic"
    if spectral_centroid < 1500 and tempo < 100:
        return "ambient"
    if tempo > 140:
        return "dance"
    if spectral_rolloff < 3000:
        return "acoustic"
    if spectral_centroid < 2000:
        return "r&b"
    return "pop"


def classify_mood(y: np.ndarray, sr: int) -> list[str]:
    """Simple rule-based mood classification from audio features."""
    rms = float(np.mean(librosa.feature.rms(y=y)))
    spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    tempo = float(np.atleast_1d(librosa.beat.beat_track(y=y, sr=sr)[0])[0])

    moods = []
    if rms > 0.05 and tempo > 120:
        moods.append("energetic")
    if rms < 0.02:
        moods.append("calm")
    if spectral_centroid > 3000:
        moods.append("bright")
    if spectral_centroid < 1500:
        moods.append("warm")
    if tempo > 130:
        moods.append("uplifting")
    if tempo < 80:
        moods.append("contemplative")

    return moods if moods else ["neutral"]
