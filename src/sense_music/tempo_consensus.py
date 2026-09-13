# sense-music — audio analysis for AI perception
# Built by humanjava.com — find this and other tools for the agentic age at huje.tools
"""Quorum / consensus tempo resolver — the octave-robust BPM tool.

The problem: any single detector (madmom, librosa, even the drums-MIDI) octave-errors on ~74% of
a slow-soul-to-fast-techno catalog, because a track's machine-measurable periodicity often sits at
2x (busy hats) or 1/2 (sparse backbone) of the perceived tempo. No single estimator fixes it.

The fix (synthesised from Mixxx/SoundTouch + Zapata/Essentia TempoTapMaxAgreement + Ellis/librosa):
 1. BAND-CONSTRAINED onset-autocorrelation — evaluate ONLY lags inside a plausible [min,max] BPM
    band, so an out-of-band octave literally cannot be reported (the strongest DJ-grade anti-octave
    trick). The band comes from the genre qualifier or a caller hint.
 2. Multi-estimator VOTE in log2 space — fold every candidate's {1/3,1/2,1,2,3} set into a
    reference band, cluster within a 4% (MIREX ACC1) tolerance, take the weighted MEDIAN (robust to
    one doubled outlier). The drums-derived candidate is weighted highest (percussion-first tempo is
    the octave-stable one).
 3. OCTAVE resolution — pick the winning cluster's actual octave via the band (hard-ish gate) + a
    Moelants ~120 log-normal perceptual prior (soft), or anchor to the drums candidate when present.
 4. CONFIDENCE from agreement + tightness + prior + octave-ambiguity gap; flag `uncertain` on an
    octave-tie so the UI can say "we're not sure" instead of showing a confident-wrong number.

Commercial-clean: librosa (ISC) only. madmom / drums-MIDI values are passed IN as extra candidates
(the caller decides licensing); this module imports no GPL/AGPL/CC-BY-NC code.
"""
from __future__ import annotations

import math
import numpy as np

_PERC_CENTER = 120.0     # Moelants/Ellis perceptual prior center (BPM)
_PERC_STD_OCT = 1.0      # log2 std of the prior (librosa default)
_TOL = 0.04              # 4% cluster/agreement tolerance (MIREX ACC1)
_REF_BAND = (70.0, 140.0)  # reference fold band (one octave, ~log-centered 99)


def _l2(b: float) -> float:
    return math.log2(b) if b > 0 else 0.0


def _fold_into(bpm: float, lo: float, hi: float) -> float:
    if bpm <= 0:
        return bpm
    while bpm < lo:
        bpm *= 2.0
    while bpm >= hi:
        bpm /= 2.0
    return bpm


def _prior_weight(bpm: float, band: tuple | None) -> float:
    w = math.exp(-0.5 * ((_l2(bpm) - _l2(_PERC_CENTER)) / _PERC_STD_OCT) ** 2)
    if band and not (band[0] <= bpm <= band[1]):
        w *= 0.05  # hard-ish genre/band gate
    return w


def band_constrained_acf_bpm(y: np.ndarray, sr: int, band: tuple, hop: int = 512) -> float:
    """The DJ-grade primary estimator: onset-strength autocorrelation, peak picked ONLY within the
    lag window for [band.lo, band.hi] BPM. An out-of-band octave can't be returned."""
    import librosa
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop, aggregate=np.median)
    if env.size < 4:
        return 0.0
    fps = sr / hop
    lag_hi = int(round(fps * 60.0 / band[0]))   # slowest BPM -> longest lag
    lag_lo = int(round(fps * 60.0 / band[1]))   # fastest BPM -> shortest lag
    ac = librosa.autocorrelate(env, max_size=lag_hi + 2)
    lag_lo = max(1, lag_lo)
    lag_hi = min(lag_hi, len(ac) - 1)
    if lag_hi <= lag_lo:
        return 0.0
    best = lag_lo + int(np.argmax(ac[lag_lo:lag_hi + 1]))
    return float(fps * 60.0 / best) if best > 0 else 0.0


def consensus_tempo(y: np.ndarray, sr: int, band: tuple | None = None,
                    extra_candidates: list | None = None) -> dict:
    """Resolve tempo by quorum. `band` = (min,max) plausible BPM (from genre qualifier / founder
    hint); when given it drives the band-constrained ACF + gates the octave. `extra_candidates` =
    [(name, bpm, weight), ...] e.g. ("drums_midi", 66.3, 2.0), ("madmom", 122.4, 1.0)."""
    import librosa
    cands = []  # (name, bpm, weight)
    # librosa perceptual-prior tempo (ISC)
    try:
        lt = float(np.atleast_1d(librosa.feature.tempo(y=y, sr=sr))[0])
        cands.append(("librosa_tempo", lt, 1.0))
    except Exception:
        pass
    try:
        bt = float(np.atleast_1d(librosa.beat.beat_track(y=y, sr=sr)[0])[0])
        cands.append(("librosa_beat", bt, 0.8))
    except Exception:
        pass
    # band-constrained ACF — the octave-anchor when a band is known (highest audio weight)
    if band:
        bc = band_constrained_acf_bpm(y, sr, band)
        if bc > 0:
            cands.append(("band_acf", bc, 2.0))
    for c in (extra_candidates or []):
        if c and c[1] and c[1] > 0:
            cands.append((c[0], float(c[1]), float(c[2]) if len(c) > 2 else 1.0))
    if not cands:
        return {"bpm": 0.0, "confidence": 0.0, "uncertain": True, "candidates": []}

    # fold every candidate's octave set into the reference band, then cluster in log2
    votes = []  # (folded_bpm, weight, name)
    for name, bpm, w in cands:
        for mult in (1 / 3, 1 / 2, 1.0, 2.0, 3.0):
            votes.append((_fold_into(bpm * mult, *_REF_BAND), w, name))
    votes.sort(key=lambda v: _l2(v[0]))
    eps = math.log2(1 + _TOL)
    clusters, cur = [], [votes[0]]
    for v in votes[1:]:
        if _l2(v[0]) - _l2(cur[-1][0]) <= eps:
            cur.append(v)
        else:
            clusters.append(cur); cur = [v]
    clusters.append(cur)

    def summarize(c):
        names = {v[2] for v in c}
        weight = sum(v[1] for v in c)  # per-vote; distinct-name below is the real support
        wl2 = np.average([_l2(v[0]) for v in c], weights=[v[1] for v in c])
        spread = math.sqrt(np.average([(_l2(v[0]) - wl2) ** 2 for v in c], weights=[v[1] for v in c]))
        return {"names": names, "n": len(names), "weight": weight,
                "center": 2 ** wl2, "spread": spread}
    summ = sorted((summarize(c) for c in clusters), key=lambda s: (s["n"], s["weight"]), reverse=True)
    top = summ[0]
    second = summ[1] if len(summ) > 1 else None

    # resolve the winning cluster's actual octave: score its octave set by band + 120 prior,
    # anchored to a drums candidate if we have one.
    drum = next((b for (n, b, w) in cands if "drum" in n.lower()), None)
    octs = [top["center"] * m for m in (1 / 3, 1 / 2, 1.0, 2.0, 3.0)]
    if drum:
        final = min(octs, key=lambda o: abs(_l2(o) - _l2(drum)))  # nearest to drum octave
        # but if a band is set and the drum-nearest octave is out of band, prefer in-band
        if band and not (band[0] <= final <= band[1]):
            inb = [o for o in octs if band[0] <= o <= band[1]]
            if inb:
                final = max(inb, key=lambda o: _prior_weight(o, band))
    else:
        final = max(octs, key=lambda o: _prior_weight(o, band))

    # Drums-anchor override: the percussion-derived tempo is the octave-stable signal. If a drums
    # candidate sits in the (10%-relaxed) band but the vote landed somewhere non-octave-related to
    # it, trust the drums value and flag the audio/drums conflict for the founder ear. This catches
    # f-eel (true 53.8, just under the band floor, with a spurious 72-BPM audio peak).
    drum_conflict = False
    if drum and band:
        rlo, rhi = band[0] * 0.9, band[1] * 1.1
        if rlo <= drum <= rhi and abs(_l2(final) - _l2(drum)) > math.log2(1.08):
            final = drum
            drum_conflict = True

    n_est = len({n for (n, _, _) in cands})
    agree = top["n"] / max(1, n_est)
    tight = math.exp(-(top["spread"] / eps)) if eps else 0.0
    prior_ok = _prior_weight(final, band)
    gap = top["weight"] / (second["weight"] + 1e-9) if second else 9.9
    amb = max(0.0, min(1.0, gap - 1.0))
    confidence = (agree * tight * prior_ok * (0.4 + 0.6 * amb)) ** 0.25
    octave_tie = bool(second and abs(round(_l2(top["center"]) - _l2(second["center"]), 3)) in
                      (1.0, 1.585) and gap < 1.3)
    uncertain = bool(confidence < 0.5 or agree < 0.5 or octave_tie or drum_conflict)

    return {
        "bpm": round(final, 1),
        "confidence": round(confidence, 3),
        "uncertain": uncertain,
        "bpm_alt": round(second["center"], 1) if (uncertain and second) else None,
        "agreement": round(agree, 2),
        "cluster_support": top["n"],
        "candidates": [(n, round(b, 1), w) for (n, b, w) in cands],
    }
