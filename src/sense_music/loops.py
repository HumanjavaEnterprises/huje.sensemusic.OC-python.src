# sense-music — audio analysis for AI perception
# Built by humanjava.com — find this and other tools for the agentic age at huje.tools
"""Loop / motif detection + per-section key (modulation) timeline.

Reads the *narrative* of a loop-driven composition: which sections are reprises
of one another (the recurring loops), the order they appear in (the A-B-A-C
structure), and where the key changes. This is the layer that turns a flat list
of boundaries into "the loops and narrative" of a piece.
"""
from __future__ import annotations

import string

import numpy as np
import librosa

from sense_music.types import Section, Motif
from sense_music.features import detect_key

# cap analysis frames per section to bound cost on long tracks
_MAX_SEG_SAMPLES_FACTOR = 1


def _section_fingerprint(y: np.ndarray, sr: int, start: float, end: float):
    """A timbre+harmony fingerprint for a section: mean chroma (12) + mean MFCC (13).

    Chroma captures the harmonic/loop content; MFCC captures the timbre (which
    instruments/texture). Together they identify a recurring musical phrase even
    when energy drifts. Returned L2-normalized for cosine comparison.
    """
    a = int(start * sr)
    b = min(int(end * sr), len(y))
    seg = y[a:b]
    if len(seg) < sr // 2:  # < 0.5s — too short to fingerprint
        return None
    chroma = librosa.feature.chroma_cqt(y=seg, sr=sr).mean(axis=1)
    mfcc = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=13).mean(axis=1)
    # normalize each block independently so neither dominates by scale
    def _n(v):
        n = np.linalg.norm(v)
        return v / n if n > 0 else v
    return np.concatenate([_n(chroma), _n(mfcc)])


def _segment_key(y: np.ndarray, sr: int, start: float, end: float):
    """Estimate the musical key of a single section."""
    a = int(start * sr)
    b = min(int(end * sr), len(y))
    seg = y[a:b]
    if len(seg) < sr // 2:
        return None
    return detect_key(seg, sr)


def detect_motifs(y: np.ndarray, sr: int, sections: list[Section],
                  similarity_threshold: float = 0.88) -> tuple[list[Section], list[Motif]]:
    """Group sections into recurring motifs (loops) and tag each with its key.

    Greedy single-pass clustering on section fingerprints: each section either
    joins the most-similar existing motif (cosine >= threshold) or starts a new
    one (next letter A, B, C...). Returns enriched Sections (with .motif and
    .key filled) plus a Motif summary per group (where each loop recurs).
    """
    fps = [_section_fingerprint(y, sr, s.start, s.end) for s in sections]
    keys = [_segment_key(y, sr, s.start, s.end) for s in sections]

    motif_centroids: list[np.ndarray] = []   # representative fingerprint per motif
    motif_members: list[list[int]] = []      # section indices per motif
    assigned: list[int] = [-1] * len(sections)

    for i, fp in enumerate(fps):
        if fp is None:
            continue
        best_m, best_sim = -1, -1.0
        for m, cen in enumerate(motif_centroids):
            sim = float(np.dot(fp, cen) / ((np.linalg.norm(fp) * np.linalg.norm(cen)) or 1.0))
            if sim > best_sim:
                best_sim, best_m = sim, m
        if best_m >= 0 and best_sim >= similarity_threshold:
            assigned[i] = best_m
            motif_members[best_m].append(i)
            # update centroid as running mean (keeps it representative)
            n = len(motif_members[best_m])
            motif_centroids[best_m] = motif_centroids[best_m] + (fp - motif_centroids[best_m]) / n
        else:
            assigned[i] = len(motif_centroids)
            motif_centroids.append(fp.copy())
            motif_members.append([i])

    # motif letters by first appearance (A is the first-heard loop)
    letters = list(string.ascii_uppercase)
    enriched: list[Section] = []
    for i, s in enumerate(sections):
        m = assigned[i]
        letter = letters[m] if 0 <= m < len(letters) else None
        k = keys[i]
        key_str = f"{k.key} {k.mode}" if k else None
        enriched.append(Section(label=s.label, start=s.start, end=s.end,
                                motif=letter, key=key_str))

    motifs: list[Motif] = []
    for m, members in enumerate(motif_members):
        if m >= len(letters):
            break
        occ = [round(sections[idx].start, 1) for idx in members]
        total = round(sum(sections[idx].end - sections[idx].start for idx in members), 1)
        motifs.append(Motif(label=letters[m], count=len(members),
                            occurrences=occ, total_seconds=total))
    return enriched, motifs


def key_changes(sections: list[Section]) -> list[dict]:
    """Timeline of key changes (modulations) across the enriched sections."""
    changes = []
    prev = None
    for s in sections:
        if s.key and s.key != prev:
            if prev is not None:
                changes.append({"time": s.start, "from": prev, "to": s.key})
            prev = s.key
    return changes


def structure_string(sections: list[Section]) -> str:
    """The narrative map as a compact motif sequence, e.g. 'A-B-A-A-C-A'."""
    return "-".join(s.motif or "?" for s in sections)
