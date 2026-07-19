# sense-music — audio analysis for AI perception
# Built by humanjava.com — find this and other tools for the agentic age at huje.tools
"""Stem separation (Demucs) -> arrangement reading.

A mixed signal hides the thing a loop-driven track's NARRATIVE is made of:
elements dropping in and out over a steady loop (drums-only breakdown, the bass
entering, vocals arriving). Demucs splits drums/bass/other/vocals; we then read
each stem's activity over time and emit an ARRANGEMENT TIMELINE — when each
element enters/exits. That's the layer a mixed RMS curve smears together.

Demucs is heavy (~10-30s/track on GPU), so this is opt-in (analyze(stems=True)).
"""
from __future__ import annotations

import numpy as np

_model = None
STEMS = ("drums", "bass", "other", "vocals")


def _load(device: str = "cuda"):
    global _model
    if _model is None:
        import torch
        from demucs.pretrained import get_model
        _model = get_model("htdemucs")
        dev = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        _model.to(dev).eval()
        _model._sm_device = dev
    return _model


def separate(path: str, device: str = "cuda", save_dir: str | None = None) -> tuple[dict, int]:
    """Separate a file into stems. Returns ({stem_name: mono float array}, sr)."""
    import torch, os, numpy as np, librosa
    model = _load(device)
    sr = model.samplerate
    # load via librosa (torchaudio 2.11 load now needs torchcodec; we avoid that dep)
    arr, _ = librosa.load(path, sr=sr, mono=False)   # [ch, time] stereo, or [time] mono
    arr = np.atleast_2d(arr)
    wav = torch.from_numpy(arr.astype("float32"))
    if wav.shape[0] == 1:                          # mono -> stereo (demucs wants 2ch)
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > 2:
        wav = wav[:2]
    ref = wav.mean(0)
    w = (wav - ref.mean()) / (ref.std() + 1e-8)
    from demucs.apply import apply_model
    with torch.no_grad():
        sources = apply_model(model, w[None].to(model._sm_device),
                              device=model._sm_device, progress=False)[0]
    sources = sources * ref.std() + ref.mean()    # [n_src, ch, time]
    out = {}
    for name, src in zip(model.sources, sources.cpu()):
        mono = src.mean(0).numpy().astype(np.float32)
        out[name] = mono
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            import soundfile as sf
            sf.write(os.path.join(save_dir, f"{name}.wav"), src.mean(0).numpy(), sr)
    return out, sr


def _rms_per_sec(y: np.ndarray, sr: int) -> np.ndarray:
    n = max(1, int(len(y) / sr))
    return np.array([float(np.sqrt(np.mean(y[i*sr:(i+1)*sr] ** 2) + 1e-12)) for i in range(n)])


def stem_activity(stems: dict, sr: int) -> dict:
    """Per-stem activity: active fraction + the (start,end) spans it's present.

    A stem counts as 'active' in a second if its RMS clears max(8% of its own
    peak, a small absolute floor) — robust to demucs bleed on silent stems.
    """
    activity = {}
    for name, y in stems.items():
        rps = _rms_per_sec(y, sr)
        peak = float(rps.max()) if len(rps) else 0.0
        thr = max(0.08 * peak, 1e-3)
        on = rps > thr
        spans, start = [], None
        for i, a in enumerate(on):
            if a and start is None:
                start = i
            elif not a and start is not None:
                spans.append((float(start), float(i))); start = None
        if start is not None:
            spans.append((float(start), float(len(on))))
        activity[name] = {
            "active_fraction": round(float(on.mean()) if len(on) else 0.0, 3),
            "spans": spans,
        }
    return activity


def arrangement_events(activity: dict, min_gap: float = 3.0) -> list[dict]:
    """Flatten activity into an in/out timeline (the arrangement narrative).

    Spans shorter than min_gap apart are merged so we report real arrangement
    moves, not flicker.
    """
    events = []
    for name, info in activity.items():
        merged = []
        for s, e in info["spans"]:
            if merged and s - merged[-1][1] < min_gap:
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))
        for s, e in merged:
            events.append({"time": round(s, 1), "stem": name, "event": "in"})
            events.append({"time": round(e, 1), "stem": name, "event": "out"})
    events.sort(key=lambda x: x["time"])
    return events
