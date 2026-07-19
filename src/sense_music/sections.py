# sense-music — audio analysis for AI perception
# Built by humanjava.com — find this and other tools for the agentic age at huje.tools

from __future__ import annotations

import numpy as np
import librosa

from sense_music.types import Section

# cap chroma frames to prevent quadratic memory in recurrence matrix
# 2000 frames ~= 46 seconds at hop_length=512, sr=22050
# for longer files we subsample the chroma
MAX_CHROMA_FRAMES = 2000


def detect_sections(y: np.ndarray, sr: int, duration: float) -> list[Section]:
    """Detect structural sections using spectral self-similarity and novelty."""
    hop_length = 512

    # compute chroma features
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)

    # subsample chroma if too many frames to avoid quadratic memory
    n_orig_frames = chroma.shape[1]
    subsample_factor = 1
    if n_orig_frames > MAX_CHROMA_FRAMES:
        subsample_factor = int(np.ceil(n_orig_frames / MAX_CHROMA_FRAMES))
        chroma = chroma[:, ::subsample_factor]

    # recurrence matrix (now bounded to MAX_CHROMA_FRAMES^2)
    rec = librosa.segment.recurrence_matrix(chroma, mode="affinity", sym=True)

    # checkerboard kernel novelty detection
    kernel_size = 16
    n_frames_rec = rec.shape[0]
    novelty = np.zeros(n_frames_rec)
    half = kernel_size // 2
    for i in range(half, n_frames_rec - half):
        tl = rec[i - half:i, i - half:i].mean()
        br = rec[i:i + half, i:i + half].mean()
        tr = rec[i - half:i, i:i + half].mean()
        bl = rec[i:i + half, i - half:i].mean()
        novelty[i] = (tl + br) - (tr + bl)

    # map subsampled frames back to original time
    frame_indices = np.arange(n_frames_rec) * subsample_factor
    times = librosa.frames_to_time(frame_indices, sr=sr, hop_length=hop_length)

    # adaptive threshold: use median + std
    threshold = np.median(novelty) + np.std(novelty)
    peaks = []
    for i in range(1, n_frames_rec - 1):
        if novelty[i] > threshold and novelty[i] >= novelty[i - 1] and novelty[i] >= novelty[i + 1]:
            peaks.append(i)

    # convert to times and add boundaries
    boundary_times = [0.0]
    min_section_duration = 5.0  # seconds
    for p in peaks:
        t = float(times[p])
        if t - boundary_times[-1] >= min_section_duration and duration - t >= min_section_duration:
            boundary_times.append(round(t, 1))
    boundary_times.append(round(duration, 1))

    # assign labels using RELATIVE energy across this track's own sections, so a
    # steady loop-driven piece still gets a real narrative arc (not all "verse").
    boundaries = [(boundary_times[i], boundary_times[i + 1]) for i in range(len(boundary_times) - 1)]
    labels = _label_sections(y, sr, boundaries)
    sections = [Section(label=lab, start=s, end=e) for (s, e), lab in zip(boundaries, labels)]

    return sections if sections else [Section(label="intro", start=0.0, end=round(duration, 1))]


def _label_sections(y: np.ndarray, sr: int, boundaries: list[tuple[float, float]]) -> list[str]:
    """Label sections by their energy RANK within this track + position + rise/fall.

    Absolute RMS thresholds collapse loop tracks (consistent energy) to all-"verse".
    Ranking each section against the track's own distribution recovers the arc:
    intro / build / groove / peak / breakdown / bridge / outro.
    """
    n = len(boundaries)
    if n == 0:
        return []
    rms = []
    for s, e in boundaries:
        seg = y[int(s * sr):min(int(e * sr), len(y))]
        rms.append(float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0)
    arr = np.array(rms)
    # percentile rank of each section's energy within the track
    order = arr.argsort()
    pct = np.empty(n)
    pct[order] = np.linspace(0, 1, n) if n > 1 else np.array([0.5])

    labels = []
    for i in range(n):
        s, e = boundaries[i]
        dur = e - s
        if i == 0:
            labels.append("intro")
            continue
        if i == n - 1:
            labels.append("outro")
            continue
        rose = arr[i] > arr[i - 1] * 1.15
        fell = arr[i] < arr[i - 1] * 0.85
        if pct[i] >= 0.80:
            labels.append("peak")
        elif pct[i] <= 0.20:
            labels.append("breakdown" if dur < 12 else "bridge")
        elif rose:
            labels.append("build")
        elif fell:
            labels.append("breakdown")
        else:
            labels.append("groove")
    return labels
