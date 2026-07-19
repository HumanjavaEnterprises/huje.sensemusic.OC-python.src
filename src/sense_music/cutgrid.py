# sense-music — audio analysis for AI perception
# Built by humanjava.com — find this and other tools for the agentic age at huje.tools
"""Edit/cut grid — fuse the analysis layers into specific, repeatable cut points.

The point of all the perception (downbeats, sections, arrangement events, chord &
key changes) for VIDEO is this: turn "cut on vibe" into a **bar-aligned, ranked
list of structurally-meaningful cut points** — so edits are specific and
repeatable, and so you can reverse-engineer a reference video (read where THEIR
cuts land against the song's grid, then hit the same depth in yours).

Pure fusion over an Analysis — no audio, deterministic, cheap.
"""
from __future__ import annotations

import numpy as np

# relative weights — how "strong" a cut candidate each event type is
_WEIGHTS = {
    "section": 1.0,        # a structural boundary — the strongest cut
    "arrangement": 0.9,    # an element drops in/out — a natural visual hit
    "key": 0.8,            # modulation
    "phrase": 0.7,         # a 4/8-bar phrase start (downbeat on a phrase boundary)
    "chord": 0.5,          # a chord change
    "downbeat": 0.4,       # a bar start
    "peak": 0.85,          # an energy peak
}


def _downbeats(analysis):
    db = (analysis.rhythm or {}).get("downbeats") or []
    return [float(t) for t in db]


def phrase_grid(analysis, bars_per_phrase: int = 4) -> list[float]:
    """Downbeats that begin a phrase (every Nth bar) — the usual edit unit."""
    db = _downbeats(analysis)
    return [db[i] for i in range(0, len(db), bars_per_phrase)]


def snap_to_grid(time: float, analysis, grid: str = "downbeat") -> float:
    """Snap an arbitrary time to the nearest downbeat (or phrase) — makes any
    chosen/borrowed cut land on the bar so it reads as intentional + repeatable."""
    pts = phrase_grid(analysis) if grid == "phrase" else _downbeats(analysis)
    if not pts:
        return round(time, 3)
    return round(min(pts, key=lambda t: abs(t - time)), 3)


def edit_points(analysis, snap: bool = True, min_spacing: float = 0.0) -> list[dict]:
    """Ranked, time-sorted candidate cut points fused from every layer.

    Each point: {time, kind, weight, detail}. With snap=True, section/arrangement/
    key/chord/peak events are pulled to the nearest downbeat so every cut is
    bar-aligned. min_spacing drops cuts closer than N seconds (keeps the strongest).
    """
    pts = []

    for s in analysis.sections:
        pts.append({"time": float(s.start), "kind": "section",
                    "detail": f"{s.label}" + (f" [{s.motif}]" if s.motif else "")})

    for ev in (analysis.arrangement or {}).get("events", []):
        pts.append({"time": float(ev["time"]), "kind": "arrangement",
                    "detail": f"{ev['stem']} {ev['event']}"})

    for kc in analysis.key_changes or []:
        pts.append({"time": float(kc["time"]), "kind": "key",
                    "detail": f"{kc['from']}→{kc['to']}"})

    for ch in (analysis.chords or {}).get("timeline", []):
        if "start" in ch:
            pts.append({"time": float(ch["start"]), "kind": "chord",
                        "detail": ch.get("chord", "")})

    for t in phrase_grid(analysis):
        pts.append({"time": float(t), "kind": "phrase", "detail": "phrase start"})
    for t in _downbeats(analysis):
        pts.append({"time": float(t), "kind": "downbeat", "detail": "bar"})

    # energy peaks (local maxima of the per-second curve)
    ec = analysis.energy_curve or []
    for i in range(1, len(ec) - 1):
        if ec[i] >= 0.75 and ec[i] >= ec[i - 1] and ec[i] > ec[i + 1]:
            pts.append({"time": float(i), "kind": "peak", "detail": f"energy {ec[i]:.2f}"})

    for p in pts:
        p["weight"] = _WEIGHTS.get(p["kind"], 0.3)
        if snap and p["kind"] not in ("downbeat", "phrase"):
            p["time"] = snap_to_grid(p["time"], analysis)

    pts.sort(key=lambda p: (p["time"], -p["weight"]))

    if min_spacing > 0:
        kept = []
        for p in pts:
            if kept and p["time"] - kept[-1]["time"] < min_spacing:
                if p["weight"] > kept[-1]["weight"]:
                    kept[-1] = p           # keep the stronger of two close cuts
            else:
                kept.append(p)
        pts = kept

    for p in pts:
        p["time"] = round(p["time"], 3)
        p["weight"] = round(p["weight"], 2)
    return pts


def match_reference(cut_times: list[float], analysis) -> list[dict]:
    """Reverse-engineer a reference video: for each of THEIR cut times, report
    what song event it lands on (snapped to our grid) — so you can see their edit
    logic ('they cut every 2 bars / on each arrangement change') and reuse it."""
    pts = edit_points(analysis, snap=False)
    out = []
    for t in cut_times:
        near = min(pts, key=lambda p: abs(p["time"] - t)) if pts else None
        out.append({
            "cut": round(float(t), 3),
            "nearest_downbeat": snap_to_grid(t, analysis),
            "lands_on": (near["kind"] if near else None),
            "detail": (near["detail"] if near else None),
            "offset": (round(t - near["time"], 3) if near else None),
        })
    return out
