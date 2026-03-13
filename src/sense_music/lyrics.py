# sense-music — audio analysis for AI perception
# Built by humanjava.com — find this and other tools for the agentic age at huje.tools

from __future__ import annotations

from sense_music.types import LyricLine

ALLOWED_MODELS = {"tiny", "base", "small", "medium", "large", "large-v2", "large-v3"}


def transcribe(audio_path: str, model_name: str = "base") -> list[LyricLine]:
    """Transcribe lyrics using OpenAI Whisper with timestamps."""
    if model_name not in ALLOWED_MODELS:
        raise ValueError(f"Unknown whisper model: {model_name}. Allowed: {', '.join(sorted(ALLOWED_MODELS))}")

    import whisper

    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path, word_timestamps=True)

    lines: list[LyricLine] = []
    for segment in result.get("segments", []):
        text = segment.get("text", "").strip()
        if not text:
            continue
        lines.append(LyricLine(
            start=round(float(segment["start"]), 2),
            end=round(float(segment["end"]), 2),
            text=text,
        ))

    return lines


def detect_language(audio_path: str, model_name: str = "base") -> str:
    """Detect the language of the audio."""
    if model_name not in ALLOWED_MODELS:
        raise ValueError(f"Unknown whisper model: {model_name}. Allowed: {', '.join(sorted(ALLOWED_MODELS))}")

    import whisper

    model = whisper.load_model(model_name)
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, probs = model.detect_language(mel)
    return max(probs, key=probs.get)
