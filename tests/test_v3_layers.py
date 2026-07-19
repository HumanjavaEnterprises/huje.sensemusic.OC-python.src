# sense-music — v0.3 deep-perception layer tests (no heavy models)
# Built by humanjava.com — find this and other tools for the agentic age at huje.tools

from sense_music import analyze
from sense_music.features import compute_loudness


def test_loudness_is_json_safe(test_audio_array):
    y, sr = test_audio_array
    L = compute_loudness(y, sr)
    assert "lufs" in L and "crest_factor_db" in L
    # must be plain python floats (not numpy) so to_json works
    assert L["crest_factor_db"] is None or isinstance(L["crest_factor_db"], float)
    assert L["lufs"] is None or isinstance(L["lufs"], float)


def test_v3_fields_default_empty_with_flags_off(test_audio_path):
    """With every heavy layer off, analyze() still succeeds and the new fields
    are present as their empty defaults (graceful degradation)."""
    r = analyze(test_audio_path, lyrics=False, rhythm=False, embedding=False,
                clap_tags=False, chords=False, stems=False, caption=False)
    assert r.rhythm == {}
    assert r.chords == {}
    assert r.clap_tags == []
    assert r.embedding == []
    assert r.arrangement == {}
    assert r.caption == ""
    # loudness always computed (cheap)
    assert "crest_factor_db" in r.loudness


def test_v3_fields_serialize(test_audio_path):
    import json
    r = analyze(test_audio_path, lyrics=False, rhythm=False, embedding=False,
                clap_tags=False, chords=False)
    d = r.to_json()
    for k in ("rhythm", "chords", "loudness", "clap_tags", "embedding",
              "arrangement", "caption"):
        assert k in d
    json.dumps(d)  # must be fully JSON-serializable


def test_cosine_metric():
    from sense_music.embedding import cosine
    assert round(cosine([1, 0, 0], [1, 0, 0]), 5) == 1.0
    assert round(cosine([1, 0], [0, 1]), 5) == 0.0
