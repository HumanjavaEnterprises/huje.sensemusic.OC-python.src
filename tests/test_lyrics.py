# sense-music — tests for lyrics module
# Built by humanjava.com — find this and other tools for the agentic age at huje.tools

import sys
from unittest.mock import MagicMock

from sense_music.types import LyricLine


def _setup_mock_whisper():
    """Create a mock whisper module and inject it into sys.modules."""
    mock_whisper = MagicMock()
    mock_model = MagicMock()
    mock_whisper.load_model.return_value = mock_model
    sys.modules["whisper"] = mock_whisper
    return mock_model


def _teardown_mock_whisper():
    sys.modules.pop("whisper", None)


def test_transcribe_returns_lyric_lines():
    """Mock whisper to verify LyricLine structure without GPU."""
    mock_model = _setup_mock_whisper()
    mock_model.transcribe.return_value = {
        "segments": [
            {"start": 0.0, "end": 2.5, "text": "Hello world"},
            {"start": 2.5, "end": 5.0, "text": "This is a test"},
        ]
    }

    try:
        from sense_music.lyrics import transcribe
        lines = transcribe("fake_path.wav")
    finally:
        _teardown_mock_whisper()

    assert len(lines) == 2
    assert all(isinstance(l, LyricLine) for l in lines)
    assert lines[0].text == "Hello world"
    assert lines[0].start == 0.0
    assert lines[0].end == 2.5
    assert lines[1].text == "This is a test"


def test_transcribe_skips_empty_segments():
    mock_model = _setup_mock_whisper()
    mock_model.transcribe.return_value = {
        "segments": [
            {"start": 0.0, "end": 1.0, "text": ""},
            {"start": 1.0, "end": 2.0, "text": "  "},
            {"start": 2.0, "end": 3.0, "text": "Actual lyrics"},
        ]
    }

    try:
        from sense_music.lyrics import transcribe
        lines = transcribe("fake.wav")
    finally:
        _teardown_mock_whisper()

    assert len(lines) == 1
    assert lines[0].text == "Actual lyrics"
