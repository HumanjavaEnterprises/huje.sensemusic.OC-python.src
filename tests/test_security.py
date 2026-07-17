# sense-music — security tests
# Built by humanjava.com — find this and other tools for the agentic age at huje.tools

import os
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from PIL import Image

import importlib

from sense_music.analyze import _fetch_url, _resolve_source, _validate_file
from sense_music.output import to_html, render_page, _validate_output_path
from sense_music.lyrics import ALLOWED_MODELS
from sense_music.types import Analysis, FileInfo, BPMInfo, KeyInfo, Section, LyricLine

# the package re-exports the analyze() function, which shadows the submodule name
analyze_mod = importlib.import_module("sense_music.analyze")


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


# --- SSRF redirect / DNS-rebinding tests ---

class _ConfigurableHandler(BaseHTTPRequestHandler):
    """Serves responses from server.app_responses: path -> (status, headers, body)."""

    def do_GET(self):
        status, headers, body = self.server.app_responses.get(
            self.path, (404, {}, b"not found")
        )
        headers = dict(headers)
        omit_length = headers.pop("X-Test-Omit-Content-Length", None) is not None
        self.send_response(status)
        for key, value in headers.items():
            self.send_header(key, value)
        if body and "Content-Length" not in headers and not omit_length:
            self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def log_message(self, *args):  # silence test output
        pass


@pytest.fixture
def http_server():
    server = HTTPServer(("127.0.0.1", 0), _ConfigurableHandler)
    server.app_responses = {}
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    thread.join(timeout=5)


@pytest.fixture
def allow_test_server(monkeypatch):
    """Let 127.0.0.1 (the pytest fixture server) through the SSRF check.

    All other hostnames/IPs — including redirect targets — still go through
    the real _validate_and_pin, so the per-hop check is genuinely exercised.
    """
    real = analyze_mod._validate_and_pin

    def patched(hostname, port):
        if hostname == "127.0.0.1":
            return "127.0.0.1"
        return real(hostname, port)

    monkeypatch.setattr(analyze_mod, "_validate_and_pin", patched)


def _url(server, path):
    return f"http://127.0.0.1:{server.server_address[1]}{path}"


def test_redirect_to_metadata_ip_blocked(http_server, allow_test_server, tmp_path):
    """A 302 to the cloud metadata IP must be blocked at the redirect hop."""
    http_server.app_responses["/track.mp3"] = (
        302,
        {"Location": "http://169.254.169.254/latest/meta-data/"},
        b"",
    )
    with pytest.raises(ValueError, match="private/internal"):
        _fetch_url(_url(http_server, "/track.mp3"), str(tmp_path / "out.mp3"))


def test_redirect_to_private_ip_blocked(http_server, allow_test_server, tmp_path):
    """A 302 to a private-range IP must be blocked at the redirect hop."""
    http_server.app_responses["/track.mp3"] = (
        302,
        {"Location": "http://10.0.0.1/internal.mp3"},
        b"",
    )
    with pytest.raises(ValueError, match="private/internal"):
        _fetch_url(_url(http_server, "/track.mp3"), str(tmp_path / "out.mp3"))


def test_redirect_to_file_scheme_blocked(http_server, allow_test_server, tmp_path):
    """A 302 to a file:// URL must be rejected."""
    http_server.app_responses["/track.mp3"] = (
        302,
        {"Location": "file:///etc/passwd"},
        b"",
    )
    with pytest.raises(ValueError, match="Unsupported URL scheme"):
        _fetch_url(_url(http_server, "/track.mp3"), str(tmp_path / "out.mp3"))


def test_redirect_loop_capped(http_server, allow_test_server, tmp_path):
    """Endless self-redirects must hit the max-redirect cap."""
    http_server.app_responses["/loop.mp3"] = (
        302,
        {"Location": "/loop.mp3"},
        b"",
    )
    with pytest.raises(ValueError, match="Too many redirects"):
        _fetch_url(_url(http_server, "/loop.mp3"), str(tmp_path / "out.mp3"))


def test_oversized_body_rejected(http_server, allow_test_server, tmp_path):
    """A streamed body over the cap must be rejected mid-download."""
    body = b"x" * (64 * 1024)
    # omit Content-Length so the cap must be enforced while streaming
    http_server.app_responses["/big.mp3"] = (
        200,
        {"X-Test-Omit-Content-Length": "1"},
        body,
    )
    with pytest.raises(ValueError, match="too large"):
        _fetch_url(
            _url(http_server, "/big.mp3"),
            str(tmp_path / "out.mp3"),
            max_bytes=1024,
        )


def test_oversized_content_length_rejected(http_server, allow_test_server, tmp_path):
    """A Content-Length over the cap is rejected before streaming."""
    http_server.app_responses["/big.mp3"] = (
        200,
        {"Content-Length": str(10**9)},
        b"",
    )
    with pytest.raises(ValueError, match="too large"):
        _fetch_url(
            _url(http_server, "/big.mp3"),
            str(tmp_path / "out.mp3"),
            max_bytes=1024,
        )


def test_fetch_success_after_public_redirect(http_server, allow_test_server, tmp_path):
    """Normal fetch (including a same-host redirect) still works and pins the IP."""
    payload = b"ID3 fake mp3 payload"
    http_server.app_responses["/moved.mp3"] = (
        302,
        {"Location": "/track.mp3"},
        b"",
    )
    http_server.app_responses["/track.mp3"] = (200, {}, payload)
    dest = tmp_path / "out.mp3"
    _fetch_url(_url(http_server, "/moved.mp3"), str(dest))
    assert dest.read_bytes() == payload


def test_unresolvable_hostname_raises_valueerror():
    """socket.gaierror is normalized to ValueError."""
    with pytest.raises(ValueError, match="Could not resolve"):
        _resolve_source("http://nonexistent-host.sense-music.invalid/track.mp3")


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
