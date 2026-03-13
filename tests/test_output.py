# sense-music — tests for output module
# Built by humanjava.com — find this and other tools for the agentic age at huje.tools

import json
import os
import tempfile

from PIL import Image

from sense_music.output import to_json, to_html, save
from sense_music.types import Analysis, FileInfo, BPMInfo, KeyInfo, Section, LyricLine


def _make_analysis():
    """Create a minimal Analysis for testing output."""
    spec = Image.new("RGB", (100, 50), color="black")
    wave = Image.new("RGB", (100, 30), color="black")
    return Analysis(
        file_info=FileInfo(name="test.wav", duration=5.0, sample_rate=22050, channels=1, format="wav"),
        duration=5.0,
        bpm=BPMInfo(tempo=120.0, confidence=0.9),
        key=KeyInfo(key="C", mode="minor", confidence=0.8),
        sections=[Section(label="intro", start=0.0, end=2.5), Section(label="verse", start=2.5, end=5.0)],
        lyrics=[LyricLine(start=0.5, end=2.0, text="Hello world")],
        energy_curve=[0.3, 0.5, 0.8, 0.6, 0.2],
        genre="electronic",
        mood=["energetic", "bright"],
        summary="A test track.",
        spectrogram=spec,
        waveform=wave,
    )


def test_to_json_returns_dict():
    data = to_json(_make_analysis())
    assert isinstance(data, dict)
    assert data["duration"] == 5.0
    assert data["bpm"]["tempo"] == 120.0
    assert data["key"]["key"] == "C"
    assert len(data["sections"]) == 2
    assert len(data["lyrics"]) == 1


def test_to_json_serializable():
    data = to_json(_make_analysis())
    serialized = json.dumps(data)
    assert isinstance(serialized, str)


def test_to_html_returns_string():
    html = to_html(_make_analysis())
    assert isinstance(html, str)
    assert "sense-music" in html
    assert "test.wav" in html
    assert "120.0" in html
    assert "humanjava.com" in html


def test_to_html_contains_images():
    html = to_html(_make_analysis())
    assert "data:image/png;base64," in html


def test_save_creates_files():
    analysis = _make_analysis()
    with tempfile.TemporaryDirectory() as tmpdir:
        save(analysis, tmpdir)
        assert os.path.exists(os.path.join(tmpdir, "spectrogram.png"))
        assert os.path.exists(os.path.join(tmpdir, "waveform.png"))
        assert os.path.exists(os.path.join(tmpdir, "analysis.json"))
        assert os.path.exists(os.path.join(tmpdir, "analysis.html"))

        with open(os.path.join(tmpdir, "analysis.json")) as f:
            data = json.load(f)
        assert data["duration"] == 5.0


def test_save_creates_directory():
    analysis = _make_analysis()
    with tempfile.TemporaryDirectory() as tmpdir:
        nested = os.path.join(tmpdir, "sub", "dir")
        save(analysis, nested)
        assert os.path.exists(os.path.join(nested, "analysis.json"))
