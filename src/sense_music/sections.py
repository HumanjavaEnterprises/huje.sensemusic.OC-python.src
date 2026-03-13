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

    # assign labels based on position and spectral characteristics
    sections = []
    n = len(boundary_times) - 1
    for i in range(n):
        start = boundary_times[i]
        end = boundary_times[i + 1]
        label = _assign_label(i, n, y, sr, start, end)
        sections.append(Section(label=label, start=start, end=end))

    return sections if sections else [Section(label="intro", start=0.0, end=round(duration, 1))]


def _assign_label(index: int, total: int, y: np.ndarray, sr: int,
                  start: float, end: float) -> str:
    """Assign a section label based on position and energy."""
    # extract segment audio
    start_sample = int(start * sr)
    end_sample = min(int(end * sr), len(y))
    segment = y[start_sample:end_sample]

    if len(segment) == 0:
        return "instrumental"

    rms = float(np.sqrt(np.mean(segment ** 2)))
    overall_rms = float(np.sqrt(np.mean(y ** 2)))

    # position-based heuristics
    if index == 0 and (end - start) < 20:
        return "intro"
    if index == total - 1 and (end - start) < 20:
        return "outro"

    # energy-based
    if rms > overall_rms * 1.2:
        return "chorus"
    if rms < overall_rms * 0.5:
        return "bridge"
    return "verse"
