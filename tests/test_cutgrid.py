# sense-music — cut-grid / edit-points tests
# Built by humanjava.com — find this and other tools for the agentic age at huje.tools

from sense_music.types import Analysis, FileInfo, BPMInfo, KeyInfo, Section
from sense_music.cutgrid import edit_points, snap_to_grid, phrase_grid, match_reference


def _fake_analysis():
    """A hand-built Analysis with a clean 120 BPM / 4-4 grid for deterministic tests."""
    fi = FileInfo(name="x.wav", duration=16.0, sample_rate=22050, channels=1, format="wav")
    # 120 BPM -> 0.5s/beat, downbeat every 2.0s
    downbeats = [round(2.0 * i, 3) for i in range(8)]
    return Analysis(
        file_info=fi, duration=16.0,
        bpm=BPMInfo(tempo=120.0, confidence=0.9),
        key=KeyInfo(key="A", mode="minor", confidence=0.8),
        sections=[Section("intro", 0.0, 4.0, motif="A"),
                  Section("groove", 4.0, 12.0, motif="A"),
                  Section("outro", 12.0, 16.0, motif="B")],
        lyrics=[], energy_curve=[0.2]*4 + [0.8]*8 + [0.3]*4,
        genre="techno", mood=["dark"], summary="",
        key_changes=[{"time": 4.3, "from": "A minor", "to": "C major"}],
        rhythm={"tempo": 120.0, "beats": [round(0.5*i,3) for i in range(32)],
                "downbeats": downbeats, "beats_per_bar": 4, "source": "test"},
        chords={"timeline": [{"start": 0.0, "end": 4.1, "chord": "A:min"},
                             {"start": 4.1, "end": 12.0, "chord": "C:maj"}],
                "progression": ["A:min", "C:maj"], "source": "test"},
        arrangement={"events": [{"time": 4.2, "stem": "drums", "event": "in"},
                                {"time": 11.9, "stem": "drums", "event": "out"}]},
    )


def test_snap_to_downbeat():
    a = _fake_analysis()
    assert snap_to_grid(4.3, a) == 4.0      # nearest downbeat
    assert snap_to_grid(5.2, a) == 6.0


def test_phrase_grid():
    a = _fake_analysis()
    pg = phrase_grid(a, bars_per_phrase=4)
    assert pg == [0.0, 8.0]                  # every 4th downbeat


def test_edit_points_snap_and_kinds():
    a = _fake_analysis()
    pts = edit_points(a, snap=True)
    kinds = {p["kind"] for p in pts}
    # all major layers contribute candidates
    assert {"section", "arrangement", "key", "chord", "downbeat", "phrase"} <= kinds
    # snapped events land exactly on the bar grid
    db = set(a.rhythm["downbeats"])
    for p in pts:
        if p["kind"] in ("section", "arrangement", "key", "chord"):
            assert p["time"] in db


def test_edit_points_min_spacing_keeps_strongest():
    a = _fake_analysis()
    pts = edit_points(a, snap=True, min_spacing=2.0)
    times = [p["time"] for p in pts]
    assert times == sorted(times)
    # no two cuts closer than the spacing
    assert all(times[i+1] - times[i] >= 2.0 for i in range(len(times)-1))


def test_match_reference():
    a = _fake_analysis()
    res = match_reference([4.15, 8.05], a)
    assert res[0]["nearest_downbeat"] == 4.0
    assert res[1]["nearest_downbeat"] == 8.0
    assert all("lands_on" in r for r in res)
