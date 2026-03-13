# sense-music — tests for features module
# Built by humanjava.com — find this and other tools for the agentic age at huje.tools

from sense_music.features import detect_bpm, detect_key, compute_energy, classify_genre, classify_mood
from sense_music.types import BPMInfo, KeyInfo


def test_detect_bpm(test_audio_array):
    y, sr = test_audio_array
    result = detect_bpm(y, sr)
    assert isinstance(result, BPMInfo)
    assert result.tempo >= 0
    assert 0 <= result.confidence <= 1


def test_detect_key(test_audio_array):
    y, sr = test_audio_array
    result = detect_key(y, sr)
    assert isinstance(result, KeyInfo)
    assert result.key in ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    assert result.mode in ["major", "minor"]
    assert 0 <= result.confidence <= 1


def test_compute_energy(test_audio_array):
    y, sr = test_audio_array
    energy = compute_energy(y, sr)
    assert isinstance(energy, list)
    assert len(energy) >= 1
    assert all(0 <= v <= 1.0 for v in energy)


def test_energy_length_matches_duration(test_audio_array):
    y, sr = test_audio_array
    duration = len(y) / sr
    energy = compute_energy(y, sr)
    assert abs(len(energy) - duration) <= 1


def test_classify_genre(test_audio_array):
    y, sr = test_audio_array
    genre = classify_genre(y, sr)
    assert isinstance(genre, str)
    assert len(genre) > 0


def test_classify_mood(test_audio_array):
    y, sr = test_audio_array
    mood = classify_mood(y, sr)
    assert isinstance(mood, list)
    assert len(mood) >= 1
    assert all(isinstance(m, str) for m in mood)
