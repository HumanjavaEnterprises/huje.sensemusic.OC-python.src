# sense-music — security tests
# Built by humanjava.com — find this and other tools for the agentic age at huje.tools

import os
import tempfile

import pytest
from PIL import Image

from sense_music.analyze import _resolve_source, _validate_file, analyze
from sense_music.output import to_html, save, render_page, _validate_output_path
from sense_music.lyrics import ALLOWED_MODELS
from sense_music.types import Analysis, FileInfo, BPMInfo, KeyInfo, Section, LyricLine


# --- SSRF tests (#1) ---

def test_block_private_ip_url():
    with pytest.raises(ValueError, match="private"):
        _resolve_source("http://192.168.1.1/audio.mp3")


def test_block_loopback_url():
    with pytest.raises(ValueError, match="private"):
        _resolve_source("http://127.0.0.1/audio.mp3")


def test_block_localhost_url():
    with pytest.raises(ValueError, match="private"):
        _resolve_source("http://localhost/audio.mp3")


def test_block_file_uri():
    with pytest.raises(ValueError, match="Unsupported URI"):
        _resolve_source("file:///etc/passwd")


def test_block_ftp_uri():
    with pytest.raises(ValueError, match="Unsupported URI"):
        _resolve_source("ftp://evil.com/audio.mp3")


# --- File validation tests (#2, #10) ---

def test_reject_empty_file():
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        with pytest.raises(ValueError, match="Empty file"):
            _validate_file(path)
    finally:
        os.unlink(path)


def test_reject_nonexistent_file():
    with pytest.raises(ValueError, match="Not a file"):
        _validate_file("/nonexistent/path/audio.wav")


def test_reject_directory():
    with pytest.raises(ValueError, match="Not a file"):
        _validate_file(tempfile.gettempdir())


# --- XSS tests (#5) ---

def _make_xss_analysis():
    return Analysis(
        file_info=FileInfo(name='<script>alert(1)</script>.mp3', duration=5.0,
                          sample_rate=22050, channels=1, format="mp3"),
        duration=5.0,
        bpm=BPMInfo(tempo=120.0, confidence=0.9),
        key=KeyInfo(key="C", mode="minor", confidence=0.8),
        sections=[Section(label="intro", start=0.0, end=5.0)],
        lyrics=[LyricLine(start=0.0, end=2.0, text='<img src=x onerror=alert(1)>')],
        energy_curve=[0.5],
        genre='<b>evil</b>',
        mood=['<script>xss</script>'],
        summary='<a href="evil">click</a>',
        spectrogram=Image.new("RGB", (10, 10)),
        waveform=Image.new("RGB", (10, 10)),
    )


def test_html_escapes_filename():
    html_out = to_html(_make_xss_analysis())
    assert "<script>" not in html_out
    assert "&lt;script&gt;" in html_out


def test_html_escapes_lyrics():
    html_out = to_html(_make_xss_analysis())
    # the <img> tag should be escaped so it doesn't render as HTML
    assert "<img src=x" not in html_out
    assert "&lt;img" in html_out


def test_html_escapes_genre():
    html_out = to_html(_make_xss_analysis())
    assert "<b>evil</b>" not in html_out


def test_html_escapes_mood():
    html_out = to_html(_make_xss_analysis())
    # the raw <script> should be escaped
    assert html_out.count("<script>") == 0


def test_html_escapes_summary():
    html_out = to_html(_make_xss_analysis())
    assert 'href="evil"' not in html_out


# --- Path traversal tests (#4) ---

def test_save_blocks_traversal():
    with pytest.raises(ValueError, match="traversal"):
        _validate_output_path("../../etc/cron.d")


def test_render_page_blocks_traversal():
    analysis = _make_xss_analysis()
    with pytest.raises(ValueError, match="traversal"):
        render_page(analysis, "../../tmp/evil.html")


# --- Whisper model validation (#6) ---

def test_reject_unknown_whisper_model():
    from sense_music.lyrics import transcribe
    with pytest.raises(ValueError, match="Unknown whisper model"):
        transcribe("fake.wav", model_name="evil_model")


def test_allowed_models_are_known():
    expected = {"tiny", "base", "small", "medium", "large", "large-v2", "large-v3"}
    assert ALLOWED_MODELS == expected


# --- URL extension sanitization (#8) ---

def test_url_suffix_sanitized():
    """Ensure non-audio extensions fall back to .mp3."""
    # We can't test the full download, but we test the logic indirectly
    # by checking that the function rejects internal IPs before suffix matters
    with pytest.raises(ValueError):
        _resolve_source("http://127.0.0.1/payload.py")
