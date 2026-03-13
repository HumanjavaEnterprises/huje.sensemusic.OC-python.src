# sense-music — tests for spectrogram module
# Built by humanjava.com — find this and other tools for the agentic age at huje.tools

from PIL import Image

from sense_music.spectrogram import render_spectrogram
from sense_music.types import Section


def test_render_returns_image(test_audio_array):
    y, sr = test_audio_array
    img = render_spectrogram(y, sr)
    assert isinstance(img, Image.Image)


def test_render_dimensions(test_audio_array):
    y, sr = test_audio_array
    img = render_spectrogram(y, sr, width=800, height=400)
    # matplotlib may add padding, but should be roughly right
    assert img.width >= 700
    assert img.height >= 300


def test_render_with_sections(test_audio_array):
    y, sr = test_audio_array
    sections = [Section(label="intro", start=0.0, end=2.5), Section(label="verse", start=2.5, end=5.0)]
    img = render_spectrogram(y, sr, sections=sections)
    assert isinstance(img, Image.Image)


def test_render_with_energy(test_audio_array):
    y, sr = test_audio_array
    energy = [0.5, 0.7, 0.9, 0.8, 0.3]
    img = render_spectrogram(y, sr, energy_curve=energy)
    assert isinstance(img, Image.Image)
