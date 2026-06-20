# sense-music — loop/motif + key-timeline tests
# Built by humanjava.com — find this and other tools for the agentic age at huje.tools

import numpy as np

from sense_music.loops import detect_motifs, key_changes, structure_string
from sense_music.types import Section


def _two_loop_audio():
    """Build audio with TWO distinct harmonic loops (A, B) alternating: A B A B.

    Loop A = an A-major-ish triad; loop B = a clearly different C-minor-ish triad.
    detect_motifs should recognize the reprises and group them.
    """
    sr = 22050
    seg = 4.0
    t = np.linspace(0, seg, int(sr * seg), endpoint=False)
    A = (0.5 * np.sin(2 * np.pi * 440 * t) +     # A4
         0.3 * np.sin(2 * np.pi * 554.37 * t) +  # C#5
         0.2 * np.sin(2 * np.pi * 659.25 * t)).astype(np.float32)
    B = (0.5 * np.sin(2 * np.pi * 261.63 * t) +  # C4
         0.3 * np.sin(2 * np.pi * 311.13 * t) +  # Eb4
         0.2 * np.sin(2 * np.pi * 392.0 * t)).astype(np.float32)
    y = np.concatenate([A, B, A, B])
    sections = [
        Section("intro", 0.0, 4.0),
        Section("groove", 4.0, 8.0),
        Section("groove", 8.0, 12.0),
        Section("outro", 12.0, 16.0),
    ]
    return y, sr, sections


def test_detect_motifs_groups_reprises():
    y, sr, sections = _two_loop_audio()
    enriched, motifs = detect_motifs(y, sr, sections, similarity_threshold=0.7)
    # sections 0 & 2 should share a motif; 1 & 3 should share the other
    assert enriched[0].motif == enriched[2].motif
    assert enriched[1].motif == enriched[3].motif
    assert enriched[0].motif != enriched[1].motif
    # exactly two distinct loops, each recurring twice
    assert len(motifs) == 2
    assert all(m.count == 2 for m in motifs)


def test_each_section_gets_a_key():
    y, sr, sections = _two_loop_audio()
    enriched, _ = detect_motifs(y, sr, sections)
    assert all(s.key for s in enriched)


def test_structure_string():
    y, sr, sections = _two_loop_audio()
    enriched, _ = detect_motifs(y, sr, sections, similarity_threshold=0.7)
    s = structure_string(enriched)
    assert s == "A-B-A-B"


def test_key_changes_timeline():
    y, sr, sections = _two_loop_audio()
    enriched, _ = detect_motifs(y, sr, sections, similarity_threshold=0.7)
    changes = key_changes(enriched)
    # different harmonic content between A and B -> at least one modulation
    assert len(changes) >= 1
    assert all("from" in c and "to" in c and "time" in c for c in changes)


def test_first_loop_is_A():
    y, sr, sections = _two_loop_audio()
    enriched, _ = detect_motifs(y, sr, sections, similarity_threshold=0.7)
    assert enriched[0].motif == "A"
