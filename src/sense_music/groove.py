# sense-music — audio analysis for AI perception
# Built by humanjava.com — find this and other tools for the agentic age at huje.tools
"""Groove — drum-voice hearing + feel, from the DRUMS stem.

A mixed signal folds the low end together: a track's SECOND KICK (deep/thin
dynamics switch) and its TOMS (pitched, ringing, often tuned near kick level)
vanish into the main kick when you onset-detect the mix. On a bass-free DRUMS
stem the low band DECAYS between hits, so toms (a pitched tail) separate from
kicks (a short thud), and the sub-kick / 2nd-kick splits from the main kick by
fundamental band. This module hears those voices and the FEEL (swing + push/pull
against the bar grid) that a plain tempo number can't carry.

Grounded in the 56-track Nigil Caenaan corpus rebuild (toms 55/56, real 2nd-kick
32/56; tom tunings clustered at i/iv/v). See the 7777 data-corpus runbooks.

Heavy (wants a Demucs drums stem), so opt-in: analyze(groove=True).
"""
from __future__ import annotations

import numpy as np

_NM = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def _stft(x: np.ndarray, sr: int, N: int = 2048, H: int = 256):
    win = np.hanning(N).astype(np.float32)
    f = 1 + (len(x) - N) // H
    if f < 1:
        return np.zeros((0, N // 2 + 1), np.float32), np.fft.rfftfreq(N, 1 / sr), H
    S = np.empty((f, N // 2 + 1), np.float32)
    for i in range(f):
        S[i] = np.abs(np.fft.rfft(x[i * H:i * H + N] * win))
    return S, np.fft.rfftfreq(N, 1 / sr), H


def _band(S, fr, lo, hi):
    m = (fr >= lo) & (fr < hi)
    return S[:, m].sum(1)


def _onsets(env, H, sr, k: float = 1.0, gap: float = 0.05):
    fx = np.concatenate([[0], np.maximum(0, np.diff(env))])
    thr = fx.mean() + k * fx.std()
    raw = [i for i in range(1, len(fx) - 1)
           if fx[i] > thr and fx[i] >= fx[i - 1] and fx[i] > fx[i + 1]]
    out, last, sep = [], -(10 ** 9), int(gap * sr / H)
    for i in raw:
        if i - last >= sep:
            out.append(i)
            last = i
    return out


def _swing_feel(onset_times: list[float], beats: list[float]) -> dict:
    """Feel from onset timing vs the beat grid.

    swing_percent: 50 = straight 8ths; >50 = the offbeat 'and' pushed late
      (a swung/shuffled feel). Measured as the median position, within each
      beat, of onsets landing near the halfway point.
    push_pull_ms: mean signed deviation of on-beat onsets from the grid —
      negative = ahead of the beat (pushed/driving), positive = behind
      (laid-back). The near-straight Nigil home value lives around 0.
    """
    if not beats or len(beats) < 3 or not onset_times:
        return {}
    beats = sorted(float(b) for b in beats)
    ot = np.array(sorted(onset_times))
    swing_ratios, onbeat_dev = [], []
    for b0, b1 in zip(beats, beats[1:]):
        ibi = b1 - b0
        if ibi <= 0:
            continue
        rel = (ot[(ot >= b0) & (ot < b1)] - b0) / ibi   # 0..1 within the beat
        for r in rel:
            if 0.30 <= r <= 0.70:          # the offbeat 'and'
                swing_ratios.append(r)
            elif r <= 0.15:                 # sits on the beat
                onbeat_dev.append(r * ibi * 1000.0)
            elif r >= 0.85:                 # just before the next beat
                onbeat_dev.append((r - 1.0) * ibi * 1000.0)
    out = {}
    if swing_ratios:
        sp = float(np.median(swing_ratios)) * 100.0
        out["swing_percent"] = round(sp, 1)
        out["feel"] = ("straight" if sp < 54 else "light_shuffle" if sp < 60
                       else "shuffle" if sp < 67 else "hard_shuffle")
    if onbeat_dev:
        pp = float(np.median(onbeat_dev))
        out["push_pull_ms"] = round(pp, 1)
        out["timing"] = ("pushed" if pp < -6 else "laid_back" if pp > 6 else "on_the_grid")
    return out


def analyze_groove(drums: np.ndarray, sr: int, beats: list[float] | None = None,
                   duration: float | None = None) -> dict:
    """Hear the drum voices + feel from a (bass-free) DRUMS stem.

    Returns {voices{count,per_sec}, kick2{present,confidence}, toms{present,count,
    pitches,median_hz}, hat{open_fraction}, feel{swing_percent,feel,push_pull_ms,
    timing}, source}. Fail-soft: any sub-analysis that can't resolve is omitted.
    """
    if drums is None or len(drums) < sr // 2:
        return {}
    S, fr, H = _stft(drums, sr)
    frames = len(S)
    if frames < 4:
        return {}
    dur = float(duration if duration else len(drums) / sr) or 1.0
    b = {
        "sub": _band(S, fr, 25, 55),      "kick": _band(S, fr, 55, 120),
        "lowmid": _band(S, fr, 120, 320), "snare": _band(S, fr, 200, 600),
        "noise": _band(S, fr, 2000, 8000), "hat": _band(S, fr, 8000, 16000),
    }
    allon = _onsets(S.sum(1), H, sr)
    import collections
    cls = collections.Counter()
    toms, hat_open = [], 0
    tom_mask = (fr >= 120) & (fr < 320)
    tom_freqs = fr[tom_mask]
    for o in allon:
        e = {k: v[o] for k, v in b.items()}
        tot = sum(e.values()) + 1e-9
        r = {k: v / tot for k, v in e.items()}
        pk = b["kick"][o] + b["sub"][o] + b["lowmid"][o]
        dec = 0
        for j in range(o, min(o + 60, frames)):
            if (b["kick"][j] + b["sub"][j] + b["lowmid"][j]) < 0.3 * pk:
                break
            dec += 1
        dec_ms = dec * H / sr * 1000
        if r["hat"] > 0.35:
            c = "hat"
            # open hat = a hi-band tail that rings past ~80ms
            hd = 0
            hpk = b["hat"][o]
            for j in range(o, min(o + 40, frames)):
                if b["hat"][j] < 0.3 * hpk:
                    break
                hd += 1
            if hd * H / sr * 1000 > 80:
                hat_open += 1
        elif r["noise"] > 0.20 and r["snare"] > 0.12:
            c = "snare"
        elif r["lowmid"] > r["kick"] and dec_ms > 70:
            c = "tom"
            seg = S[o][tom_mask]
            if len(seg):
                toms.append(float(tom_freqs[int(np.argmax(seg))]))
        elif r["sub"] > r["kick"] and r["sub"] > 0.22:
            c = "sub-kick"
        else:
            c = "kick"
        cls[c] += 1

    voices = {k: {"count": int(v), "per_sec": round(v / dur, 2)} for k, v in cls.items()}
    nkick = cls.get("kick", 0)
    sub = cls.get("sub-kick", 0)
    out = {
        "voices": voices,
        "onsets": len(allon),
        "kick2": {
            # a 2nd kick only means something relative to a main kick; with no main
            # kick classified, we can't call it present (avoid a spurious denominator)
            "present": bool(nkick) and sub >= 0.15 * nkick,
            "count": int(sub),
            "confidence": round(min(1.0, sub / nkick), 2) if nkick else 0.0,
        },
        "source": "drums_stem",
    }
    if cls.get("hat", 0):
        out["hat"] = {"open_fraction": round(hat_open / cls["hat"], 2)}
    if toms:
        tp = collections.Counter(_NM[int(round(69 + 12 * np.log2(t / 440))) % 12]
                                 for t in toms if t > 0)
        out["toms"] = {
            "present": len(toms) >= 3,
            "count": len(toms),
            "pitches": dict(tp.most_common()),
            "median_hz": round(float(np.median(toms)), 1),
        }
    else:
        out["toms"] = {"present": False, "count": 0}
    feel = _swing_feel([o * H / sr for o in allon], beats or [])
    if feel:
        out["feel"] = feel
    return out
