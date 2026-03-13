# sense-music — tests for sections module
# Built by humanjava.com — find this and other tools for the agentic age at huje.tools

from sense_music.sections import detect_sections
from sense_music.types import Section


def test_returns_list(test_audio_array):
    y, sr = test_audio_array
    duration = len(y) / sr
    sections = detect_sections(y, sr, duration)
    assert isinstance(sections, list)
    assert len(sections) >= 1


def test_all_sections_typed(test_audio_array):
    y, sr = test_audio_array
    duration = len(y) / sr
    sections = detect_sections(y, sr, duration)
    assert all(isinstance(s, Section) for s in sections)


def test_sections_cover_duration(test_audio_array):
    y, sr = test_audio_array
    duration = len(y) / sr
    sections = detect_sections(y, sr, duration)
    assert sections[0].start == 0.0
    assert abs(sections[-1].end - round(duration, 1)) <= 0.2


def test_sections_no_gaps(test_audio_array):
    y, sr = test_audio_array
    duration = len(y) / sr
    sections = detect_sections(y, sr, duration)
    for i in range(1, len(sections)):
        assert sections[i].start == sections[i - 1].end


def test_valid_labels(test_audio_array):
    y, sr = test_audio_array
    duration = len(y) / sr
    sections = detect_sections(y, sr, duration)
    valid = {"intro", "verse", "chorus", "bridge", "outro", "instrumental"}
    for s in sections:
        assert s.label in valid
