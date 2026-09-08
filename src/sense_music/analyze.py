# sense-music — audio analysis for AI perception
# Built by humanjava.com — find this and other tools for the agentic age at huje.tools

from __future__ import annotations

import http.client
import ipaddress
import logging
import os
import socket
import ssl
import tempfile
import urllib.parse

import librosa
import numpy as np

from sense_music.types import Analysis, FileInfo
from sense_music.features import (detect_bpm, detect_key, compute_energy, classify_genre,
                                  classify_mood, compute_loudness)
from sense_music.sections import detect_sections
from sense_music.spectrogram import render_spectrogram
from sense_music.waveform import render_waveform

logger = logging.getLogger("sense_music")

# safety limits
MAX_DURATION = 600  # seconds (10 minutes)
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MB
MAX_REDIRECTS = 5
FETCH_TIMEOUT = 30  # seconds (connect + per-read)
ALLOWED_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".opus"}


def analyze(
    source: str,
    *,
    lyrics: bool = True,
    whisper_model: str = "base",
    max_duration: float = MAX_DURATION,
    rhythm: bool = True,
    embedding: bool = True,
    clap_tags: bool = True,
    chords: bool = False,
    stems: bool = False,
    groove: bool = False,
    caption: bool = False,
    device: str = "cuda",
) -> Analysis:
    """Analyze an audio file and return a complete Analysis.

    Args:
        source: File path or URL to an audio file.
        lyrics: Transcribe lyrics with Whisper (default True).
        whisper_model: Whisper model size (default "base").
        max_duration: Maximum audio duration in seconds (default 600).
        rhythm: madmom beat/downbeat/tempo + bar grid (default True; librosa fallback).
        embedding: CLAP audio embedding — the qualifier's similarity metric (default True).
        clap_tags: CLAP zero-shot semantic tags (default True).
        chords: chord-progression recognition (default False; madmom, heavier).
        stems: Demucs stem separation -> arrangement timeline (default False; ~10-30s/track).
        groove: drum-voice hearing (kick/kick2/toms/hats) + feel (swing, push/pull) from the
            drums stem (default False; needs Demucs — reuses the stems separation if stems=True).
        caption: Qwen2-Audio free-text liner notes (default False; loads a 7B model).
        device: torch device for the ML models (default "cuda").

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
        loudness = compute_loudness(y, sr)

        # sections
        sections = detect_sections(y, sr, duration)

        # loops/motifs + per-section key (modulation) timeline — the "narrative" layer
        from sense_music.loops import detect_motifs, key_changes, structure_string
        sections, motifs = detect_motifs(y, sr, sections)
        structure = structure_string(sections)
        key_change_list = key_changes(sections)

        # ── v0.3 deeper perception layers (each gated + fail-soft) ──
        rhythm_info = {}
        if rhythm:
            try:
                from sense_music.rhythm import analyze_rhythm
                rhythm_info = analyze_rhythm(audio_path, y, sr)
            except Exception as exc:
                logger.warning("Rhythm analysis failed: %s", exc)

        chord_info = {}
        if chords:
            try:
                from sense_music.chords import analyze_chords
                chord_info = analyze_chords(audio_path, y, sr)
            except Exception as exc:
                logger.warning("Chord analysis failed: %s", exc)

        clap_tag_list, embed = [], []
        if clap_tags or embedding:
            try:
                from sense_music import embedding as _emb
                if embedding:
                    embed = _emb.embed_audio(y, sr, device=device)
                if clap_tags:
                    clap_tag_list = _emb.zero_shot_tags(y, sr, device=device)
            except Exception as exc:
                logger.warning("CLAP analysis failed: %s", exc)

        arrangement = {}
        _drums_stem, _stem_sr = None, None
        if stems or groove:
            try:
                from sense_music.stems import separate, stem_activity, arrangement_events
                stem_audio, _stem_sr = separate(audio_path, device=device, duration=max_duration)
                _drums_stem = stem_audio.get("drums")
                if stems:
                    activity = stem_activity(stem_audio, _stem_sr)
                    arrangement = {"activity": activity,
                                   "events": arrangement_events(activity)}
            except Exception as exc:
                logger.warning("Stem separation failed: %s", exc)

        groove_info = {}
        if groove:
            try:
                from sense_music.groove import analyze_groove
                beats = rhythm_info.get("beats") if isinstance(rhythm_info, dict) else None
                groove_info = analyze_groove(_drums_stem, _stem_sr, beats=beats)
            except Exception as exc:
                logger.warning("Groove analysis failed: %s", exc)

        caption_text = ""
        if caption:
            try:
                from sense_music.caption import caption_audio
                caption_text = caption_audio(y, sr, device=device)
            except Exception as exc:
                logger.warning("Captioning failed: %s", exc)

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
        summary = _generate_summary(file_info, bpm, key, sections, genre, mood,
                                    energy_curve, motifs, structure, key_change_list)

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
            motifs=motifs,
            structure=structure,
            key_changes=key_change_list,
            rhythm=rhythm_info,
            chords=chord_info,
            loudness=loudness,
            clap_tags=clap_tag_list,
            embedding=embed,
            arrangement=arrangement,
            groove=groove_info,
            caption=caption_text,
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
        raise ValueError(f"Not a file: {os.path.basename(path)}")
    size = os.path.getsize(path)
    if size == 0:
        raise ValueError("Empty file")
    if size > MAX_FILE_SIZE:
        raise ValueError(f"File too large: {size} bytes (max {MAX_FILE_SIZE})")


def _resolve_source(source: str) -> str:
    """Download URL to temp file or return path as-is."""
    if source.startswith(("http://", "https://")):
        parsed = urllib.parse.urlparse(source)

        # sanitize file extension
        suffix = os.path.splitext(parsed.path)[1].lower()
        if suffix not in ALLOWED_EXTENSIONS:
            suffix = ".mp3"

        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        try:
            os.close(fd)
            _fetch_url(source, tmp_path)
        except Exception:
            # clean up on download failure
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
        return tmp_path

    # block non-http URI schemes (file://, ftp://, etc.) — leak only the scheme
    if "://" in source:
        scheme = source.split("://", 1)[0]
        raise ValueError(f"Unsupported URI scheme: {scheme}")

    return source


def _validate_and_pin(hostname: str, port: int) -> str:
    """Resolve *hostname*, reject private/internal addresses, return a pinned IP.

    Every resolved address is checked; the first one is returned so the caller
    can connect to exactly the address that was validated (closing the
    DNS-rebinding TOCTOU between check and fetch).
    """
    # IP literals resolve to themselves; getaddrinfo handles both cases
    try:
        infos = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve hostname: {hostname}") from exc
    if not infos:
        raise ValueError(f"Could not resolve hostname: {hostname}")

    for info in infos:
        addr = ipaddress.ip_address(info[4][0])
        # unwrap IPv4-mapped IPv6 (::ffff:127.0.0.1) before checking
        mapped = getattr(addr, "ipv4_mapped", None)
        if mapped is not None:
            addr = mapped
        if (
            addr.is_private
            or addr.is_loopback
            or addr.is_link_local
            or addr.is_reserved
            or addr.is_multicast
            or addr.is_unspecified
        ):
            raise ValueError("URL resolves to private/internal address")

    return infos[0][4][0]


class _PinnedHTTPConnection(http.client.HTTPConnection):
    """HTTP connection that connects to a pre-validated IP.

    The Host header is still derived from the original hostname (passed to the
    constructor), but the TCP connection goes to the pinned IP — so the address
    that was security-checked is exactly the address we fetch from.
    """

    def __init__(self, host, port, pinned_ip, timeout):
        super().__init__(host, port, timeout=timeout)
        self._pinned_ip = pinned_ip

    def connect(self):
        self.sock = socket.create_connection(
            (self._pinned_ip, self.port), timeout=self.timeout
        )


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS connection pinned to a pre-validated IP with proper SNI/cert checks."""

    def __init__(self, host, port, pinned_ip, timeout):
        super().__init__(host, port, timeout=timeout, context=ssl.create_default_context())
        self._pinned_ip = pinned_ip

    def connect(self):
        sock = socket.create_connection(
            (self._pinned_ip, self.port), timeout=self.timeout
        )
        # SNI + certificate verification against the original hostname
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)


def _fetch_url(
    url: str,
    dest_path: str,
    *,
    max_redirects: int = MAX_REDIRECTS,
    max_bytes: int = MAX_FILE_SIZE,
    timeout: float = FETCH_TIMEOUT,
) -> None:
    """Download *url* to *dest_path* with SSRF protection on every redirect hop.

    - The private/loopback/link-local/reserved check runs on every hop, and the
      connection is made to the exact IP that passed the check (DNS pinning).
    - Redirects are followed manually (never automatically) and capped.
    - The body is streamed with a hard size cap.
    """
    for _hop in range(max_redirects + 1):
        parsed = urllib.parse.urlparse(url)

        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")

        hostname = parsed.hostname
        if not hostname:
            raise ValueError("URL has no hostname")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)

        pinned_ip = _validate_and_pin(hostname, port)

        conn_cls = (
            _PinnedHTTPSConnection if parsed.scheme == "https" else _PinnedHTTPConnection
        )
        conn = conn_cls(hostname, port, pinned_ip, timeout)
        try:
            request_path = parsed.path or "/"
            if parsed.query:
                request_path += "?" + parsed.query
            conn.request("GET", request_path, headers={"User-Agent": "sense-music"})
            response = conn.getresponse()

            if response.status in (301, 302, 303, 307, 308):
                location = response.getheader("Location")
                if not location:
                    raise ValueError("Redirect response without Location header")
                url = urllib.parse.urljoin(url, location)
                continue  # next hop is re-validated at the top of the loop

            if response.status != 200:
                raise ValueError(f"Download failed: HTTP {response.status}")

            content_length = response.getheader("Content-Length")
            if content_length is not None and content_length.isdigit():
                if int(content_length) > max_bytes:
                    raise ValueError(
                        f"Download too large: {content_length} bytes (max {max_bytes})"
                    )

            written = 0
            with open(dest_path, "wb") as fh:
                while True:
                    chunk = response.read(64 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > max_bytes:
                        raise ValueError(
                            f"Download too large: exceeded {max_bytes} bytes"
                        )
                    fh.write(chunk)
            return
        finally:
            conn.close()

    raise ValueError(f"Too many redirects (max {max_redirects})")


def _generate_summary(file_info, bpm, key, sections, genre, mood, energy_curve,
                      motifs=None, structure="", key_change_list=None) -> str:
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

    out = (
        f"A {duration_str} {genre} track in {key.key} {key.mode} at {bpm.tempo} BPM. "
        f"The mood is {mood_str}. The track {arc} and features "
        f"{len(sections)} section{'s' if len(sections) != 1 else ''} "
        f"({', '.join(unique_sections)}). "
    )

    # loops / narrative
    if motifs:
        recurring = [m for m in motifs if m.count > 1]
        out += f"Built from {len(motifs)} distinct loop{'s' if len(motifs) != 1 else ''}"
        if recurring:
            top = max(recurring, key=lambda m: m.count)
            out += f"; loop {top.label} recurs {top.count}× (the spine)"
        out += f". Structure: {structure}. "
    if key_change_list:
        out += f"{len(key_change_list)} key change{'s' if len(key_change_list) != 1 else ''} "
        out += "(" + "; ".join(f"{kc['from']}→{kc['to']} @ {int(kc['time'])}s" for kc in key_change_list[:4])
        out += ("; …" if len(key_change_list) > 4 else "") + "). "

    return out
