# sense-music — audio analysis for AI perception
# Built by humanjava.com — find this and other tools for the agentic age at huje.tools

from __future__ import annotations

import ipaddress
import logging
import os
import socket
import tempfile
import urllib.parse
import urllib.request

import librosa
import numpy as np

from sense_music.types import Analysis, FileInfo
from sense_music.features import detect_bpm, detect_key, compute_energy, classify_genre, classify_mood
from sense_music.sections import detect_sections
from sense_music.spectrogram import render_spectrogram
from sense_music.waveform import render_waveform

logger = logging.getLogger("sense_music")

# safety limits
MAX_DURATION = 600  # seconds (10 minutes)
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MB
ALLOWED_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".opus"}


def analyze(
    source: str,
    *,
    lyrics: bool = True,
    whisper_model: str = "base",
    max_duration: float = MAX_DURATION,
) -> Analysis:
    """Analyze an audio file and return a complete Analysis.

    Args:
        source: File path or URL to an audio file.
        lyrics: Whether to transcribe lyrics with Whisper (default True).
        whisper_model: Whisper model size (default "base").
        max_duration: Maximum audio duration in seconds (default 600).

    Returns:
        An Analysis object with all structured data and visualizations.
    """
    # resolve source — download if URL
    audio_path = _resolve_source(source)

    try:
        # validate file before loading
        _validate_file(audio_path)

        # load audio with duration cap and fixed sample rate
        y, sr = librosa.load(audio_path, sr=22050, mono=True, duration=max_duration)
        duration = float(len(y) / sr)

        # file info
        name = os.path.basename(source if not source.startswith(("http://", "https://")) else audio_path)
        ext = os.path.splitext(name)[1].lstrip(".").lower() or "wav"
        file_info = FileInfo(
            name=name,
            duration=round(duration, 2),
            sample_rate=sr,
            channels=1,  # forced mono
            format=ext,
        )

        # features
        bpm = detect_bpm(y, sr)
        key = detect_key(y, sr)
        energy_curve = compute_energy(y, sr)
        genre = classify_genre(y, sr)
        mood = classify_mood(y, sr)

        # sections
        sections = detect_sections(y, sr, duration)

        # lyrics
        lyric_lines = []
        if lyrics:
            try:
                from sense_music.lyrics import transcribe
                lyric_lines = transcribe(audio_path, model_name=whisper_model)
            except ImportError:
                pass  # whisper not installed
            except Exception as exc:
                logger.warning("Lyrics transcription failed: %s", exc)

        # visualizations
        spectrogram_img = render_spectrogram(y, sr, sections=sections, energy_curve=energy_curve)
        waveform_img = render_waveform(y, sr, sections=sections)

        # summary
        summary = _generate_summary(file_info, bpm, key, sections, genre, mood, energy_curve)

        return Analysis(
            file_info=file_info,
            duration=round(duration, 2),
            bpm=bpm,
            key=key,
            sections=sections,
            lyrics=lyric_lines,
            energy_curve=energy_curve,
            genre=genre,
            mood=mood,
            summary=summary,
            spectrogram=spectrogram_img,
            waveform=waveform_img,
        )
    finally:
        # clean up temp file if we downloaded
        if audio_path != source and os.path.exists(audio_path):
            os.unlink(audio_path)


def _validate_file(path: str) -> None:
    """Validate file exists, is a regular file, and is within size limits."""
    if not os.path.isfile(path):
        raise ValueError(f"Not a file: {path}")
    size = os.path.getsize(path)
    if size == 0:
        raise ValueError("Empty file")
    if size > MAX_FILE_SIZE:
        raise ValueError(f"File too large: {size} bytes (max {MAX_FILE_SIZE})")


def _resolve_source(source: str) -> str:
    """Download URL to temp file or return path as-is."""
    if source.startswith(("http://", "https://")):
        parsed = urllib.parse.urlparse(source)

        # block non-http(s) schemes
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")

        # resolve hostname and block private/loopback/link-local IPs
        hostname = parsed.hostname
        if not hostname:
            raise ValueError("URL has no hostname")
        for info in socket.getaddrinfo(hostname, parsed.port or 443):
            addr = ipaddress.ip_address(info[4][0])
            if addr.is_private or addr.is_loopback or addr.is_link_local:
                raise ValueError(f"URL resolves to private/internal address")

        # sanitize file extension
        suffix = os.path.splitext(parsed.path)[1].lower()
        if suffix not in ALLOWED_EXTENSIONS:
            suffix = ".mp3"

        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        try:
            os.close(fd)
            urllib.request.urlretrieve(source, tmp_path)
        except Exception:
            # clean up on download failure
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
        return tmp_path

    # block non-http URI schemes (file://, ftp://, etc.)
    if "://" in source:
        raise ValueError(f"Unsupported URI scheme: {source}")

    return source


def _generate_summary(file_info, bpm, key, sections, genre, mood, energy_curve) -> str:
    """Generate a natural language summary of the track."""
    dm, ds = divmod(int(file_info.duration), 60)
    duration_str = f"{dm}:{ds:02d}"

    section_labels = [s.label for s in sections]
    unique_sections = list(dict.fromkeys(section_labels))

    mood_str = ", ".join(mood) if mood else "neutral"

    # energy arc
    if len(energy_curve) >= 3:
        first_third = np.mean(energy_curve[:len(energy_curve) // 3])
        last_third = np.mean(energy_curve[-(len(energy_curve) // 3):])
        if last_third > first_third * 1.3:
            arc = "builds in energy over its duration"
        elif first_third > last_third * 1.3:
            arc = "gradually winds down"
        else:
            arc = "maintains a consistent energy level"
    else:
        arc = "is brief"

    return (
        f"A {duration_str} {genre} track in {key.key} {key.mode} at {bpm.tempo} BPM. "
        f"The mood is {mood_str}. The track {arc} and features "
        f"{len(sections)} section{'s' if len(sections) != 1 else ''} "
        f"({', '.join(unique_sections)}). "
    )
