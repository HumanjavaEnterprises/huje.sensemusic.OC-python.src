# sense-music — groove (drum-voice + feel) tests
# Built by humanjava.com — find this and other tools for the agentic age at huje.tools
"""Unit tests for sense_music.groove — the drum-voice + feel hearing.

Exercise the pure-DSP paths on synthetic signals (no Demucs, no GPU), the feel/swing
math, the fail-soft guards, and the JSON round-trip of the new `groove` field.
"""
import numpy as np

from sense_music.groove import analyze_groove, _swing_feel
from sense_music.types import Analysis, FileInfo, BPMInfo, KeyInfo


def _click_track(sr=44100, dur=8.0, bpm=120.0, low_hz=60.0):
    """A synthetic 'drums stem': a low thud on every beat (kick-ish)."""
    n = int(sr * dur)
    y = np.zeros(n, np.float32)
    spb = 60.0 / bpm
    for t in np.arange(0.0, dur, spb):
        i = int(t * sr)
        seg = min(sr // 20, n - i)                      # ~50ms
        env = np.exp(-np.arange(seg) / (sr * 0.01))
        y[i:i + seg] += (env * np.sin(2 * np.pi * low_hz * np.arange(seg) / sr)).astype(np.float32)
    return y, sr


def test_empty_and_short_input_is_failsoft():
    assert analyze_groove(None, 44100) == {}
    assert analyze_groove(np.zeros(10, np.float32), 44100) == {}


def test_returns_expected_shape_on_a_click_track():
    y, sr = _click_track()
    out = analyze_groove(y, sr, beats=None, duration=8.0)
    assert out.get("source") == "drums_stem"
    assert "voices" in out and "onsets" in out
    assert "kick2" in out and set(out["kick2"]) >= {"present", "count", "confidence"}
    assert "toms" in out and "present" in out["toms"]
    assert isinstance(out["onsets"], int) and out["onsets"] > 0


def test_no_main_kick_means_kick2_not_present():
    # a hat-only high-band signal: no low kick classified -> kick2 must be present=False,
    # confidence 0.0 (never a spurious denominator)
    sr = 44100
    y, _ = _click_track(sr=sr, low_hz=12000.0)          # high-band 'hats'
    out = analyze_groove(y, sr, duration=8.0)
    assert out["kick2"]["present"] is False
    assert out["kick2"]["confidence"] == 0.0


def test_swing_feel_straight_vs_shuffle_and_push_pull():
    beats = [i * 0.5 for i in range(9)]                  # 120 BPM
    straight = sorted(beats + [b + 0.25 for b in beats[:-1]])
    swung = sorted(beats + [b + 0.33 for b in beats[:-1]])
    fs, fw = _swing_feel(straight, beats), _swing_feel(swung, beats)
    assert abs(fs["swing_percent"] - 50.0) < 2 and fs["feel"] == "straight"
    assert fw["swing_percent"] > 60 and "shuffle" in fw["feel"]
    pushed = _swing_feel(sorted(b - 0.012 for b in beats), beats)
    assert pushed["push_pull_ms"] < 0 and pushed["timing"] == "pushed"


def test_swing_feel_guards():
    assert _swing_feel([], [0.0, 0.5, 1.0]) == {}
    assert _swing_feel([0.1, 0.2], []) == {}
    assert _swing_feel([0.1], [0.0, 0.5]) == {}          # <3 beats


def test_groove_round_trips_to_json():
    a = Analysis(
        file_info=FileInfo(name="x.wav", duration=8.0, sample_rate=44100, channels=1, format="wav"),
        duration=8.0, bpm=BPMInfo(tempo=120.0, confidence=0.9),
        key=KeyInfo(key="A", mode="minor", confidence=0.8),
        sections=[], lyrics=[], energy_curve=[0.5], genre="techno", mood=["dark"], summary="",
        groove={"kick2": {"present": True, "count": 9, "confidence": 0.4},
                "toms": {"present": True, "count": 4, "median_hz": 172.3},
                "feel": {"swing_percent": 47.1, "feel": "straight"}},
    )
    j = a.to_json()
    assert "groove" in j
    assert j["groove"]["kick2"]["present"] is True
    assert j["groove"]["feel"]["feel"] == "straight"
