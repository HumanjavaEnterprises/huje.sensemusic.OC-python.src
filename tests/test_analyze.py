# sense-music — tests for analyze module
# Built by humanjava.com — find this and other tools for the agentic age at huje.tools

from sense_music import analyze, Analysis, Section, BPMInfo, KeyInfo, FileInfo


def test_analyze_returns_analysis(test_audio_path):
    result = analyze(test_audio_path, lyrics=False)
    assert isinstance(result, Analysis)


def test_analyze_file_info(test_audio_path):
    result = analyze(test_audio_path, lyrics=False)
    assert isinstance(result.file_info, FileInfo)
    assert result.file_info.name.endswith(".wav")
    assert result.file_info.duration > 0
    assert result.file_info.sample_rate > 0
    assert result.file_info.format == "wav"


def test_analyze_duration(test_audio_path):
    result = analyze(test_audio_path, lyrics=False)
    assert 4.5 <= result.duration <= 5.5


def test_analyze_bpm(test_audio_path):
    result = analyze(test_audio_path, lyrics=False)
    assert isinstance(result.bpm, BPMInfo)
    assert result.bpm.tempo >= 0


def test_analyze_key(test_audio_path):
    result = analyze(test_audio_path, lyrics=False)
    assert isinstance(result.key, KeyInfo)
    assert result.key.key in ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    assert result.key.mode in ["major", "minor"]


def test_analyze_sections(test_audio_path):
    result = analyze(test_audio_path, lyrics=False)
    assert isinstance(result.sections, list)
    assert len(result.sections) >= 1
    assert all(isinstance(s, Section) for s in result.sections)


def test_analyze_energy_curve(test_audio_path):
    result = analyze(test_audio_path, lyrics=False)
    assert isinstance(result.energy_curve, list)
    assert len(result.energy_curve) >= 1
    assert all(isinstance(v, float) for v in result.energy_curve)


def test_analyze_genre_mood(test_audio_path):
    result = analyze(test_audio_path, lyrics=False)
    assert isinstance(result.genre, str)
    assert isinstance(result.mood, list)
    assert len(result.mood) >= 1


def test_analyze_summary(test_audio_path):
    result = analyze(test_audio_path, lyrics=False)
    assert isinstance(result.summary, str)
    assert len(result.summary) > 10


def test_analyze_images(test_audio_path):
    from PIL import Image
    result = analyze(test_audio_path, lyrics=False)
    assert isinstance(result.spectrogram, Image.Image)
    assert isinstance(result.waveform, Image.Image)
