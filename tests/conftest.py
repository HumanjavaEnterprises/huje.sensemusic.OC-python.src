# sense-music — test fixtures
# Built by humanjava.com — find this and other tools for the agentic age at huje.tools

import os
import tempfile

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture
def test_audio_path():
    """Generate a 5-second stereo test audio file (sine wave + harmonics)."""
    sr = 22050
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Mix of frequencies to give features something to detect
    y = (
        0.5 * np.sin(2 * np.pi * 440 * t) +       # A4
        0.3 * np.sin(2 * np.pi * 554.37 * t) +     # C#5
        0.2 * np.sin(2 * np.pi * 659.25 * t)        # E5
    ).astype(np.float32)

    # Add an amplitude envelope so energy varies
    envelope = np.concatenate([
        np.linspace(0, 1, int(sr * 1)),      # 1s fade in
        np.ones(int(sr * 3)),                 # 3s sustain
        np.linspace(1, 0, int(sr * 1)),       # 1s fade out
    ])
    # pad or trim to match
    envelope = envelope[:len(y)]
    if len(envelope) < len(y):
        envelope = np.pad(envelope, (0, len(y) - len(envelope)))
    y = y * envelope

    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(path, y, sr)
    yield path
    os.unlink(path)


@pytest.fixture
def test_audio_array():
    """Return a (y, sr) tuple for direct array tests."""
    sr = 22050
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    return y, sr
