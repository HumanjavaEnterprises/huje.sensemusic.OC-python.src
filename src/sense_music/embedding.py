# sense-music — audio analysis for AI perception
# Built by humanjava.com — find this and other tools for the agentic age at huje.tools
"""CLAP audio embedding + zero-shot semantic tagging.

Two things, both load-bearing:
  1. A real semantic READ — CLAP embeds audio and text in one space, so "is this
     dub techno? smoky? Rhodes-driven?" becomes a measured score (replaces the
     rule-based genre/mood guesswork).
  2. An EMBEDDING vector — the similarity metric the music qualifier needs:
     score a generated track against a reference corpus centroid ("does this
     sound like me?"). This is the missing piece of over-generate-verify-cull
     for music.

Uses HuggingFace transformers ClapModel (no extra laion_clap dep). Model loads
lazily and is cached process-wide. CLAP wants 48kHz mono.
"""
from __future__ import annotations

import numpy as np

_MODEL_ID = "laion/clap-htsat-unfused"
_CLAP_SR = 48000
_model = None
_processor = None

# default zero-shot vocabulary — genre / texture / mood / instrument descriptors.
# Tuned toward the electronic/Detroit palette this stack works in; override per-call.
DEFAULT_TAGS = [
    # genre / lineage
    "Detroit techno", "dub techno", "deep house", "acid house", "electro",
    "trip hop", "downtempo", "ambient", "hip hop", "trap", "mumble rap",
    "soul", "R&B", "funk", "disco", "drum and bass", "breakbeat", "rock", "pop",
    # mood
    "melancholy", "euphoric", "dark", "warm", "smoky", "hypnotic", "meditative",
    "aggressive", "dreamy", "tense", "uplifting", "nostalgic", "spacious",
    # texture / production
    "lo-fi", "hi-fi polished", "gritty", "wet reverb", "dub delay", "distorted",
    "minimal", "dense layered", "analog warm", "digital cold",
    # instrumentation
    "Rhodes electric piano", "acid synth bassline", "deep sub bass",
    "TR-909 drum machine", "vinyl crackle", "strings", "piano", "guitar",
    "female vocal", "male vocal", "spoken word", "instrumental no vocals",
]


def _load(device: str = "cuda"):
    global _model, _processor
    if _model is None:
        import torch
        from transformers import ClapModel, ClapProcessor
        _processor = ClapProcessor.from_pretrained(_MODEL_ID)
        dev = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        _model = ClapModel.from_pretrained(_MODEL_ID).to(dev).eval()
        _model._sm_device = dev
    return _model, _processor


def _to_clap_audio(y: np.ndarray, sr: int) -> np.ndarray:
    import librosa
    if sr != _CLAP_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=_CLAP_SR)
    return y.astype(np.float32)


def _projected_embeds(model, proc, wav, tags, device):
    """Run the full CLAP forward → projected, comparable audio_embeds + text_embeds.

    transformers >=5 makes get_audio_features return the raw encoder output; the
    512-d projected space (where audio↔text cosine is meaningful) lives on the
    full-forward output as .audio_embeds / .text_embeds.
    """
    import torch
    a = proc(audio=wav, sampling_rate=_CLAP_SR, return_tensors="pt")
    t = proc(text=tags, return_tensors="pt", padding=True)
    inputs = {"input_features": a["input_features"].to(device),
              "input_ids": t["input_ids"].to(device),
              "attention_mask": t["attention_mask"].to(device)}
    if "is_longer" in a:
        inputs["is_longer"] = a["is_longer"].to(device)
    with torch.no_grad():
        out = model(**inputs)
    ae = out.audio_embeds / out.audio_embeds.norm(dim=-1, keepdim=True)
    te = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
    return ae, te


def embed_audio(y: np.ndarray, sr: int, device: str = "cuda") -> list[float]:
    """Return the L2-normalized CLAP audio embedding (512-d) as a plain list."""
    model, proc = _load(device)
    wav = _to_clap_audio(y, sr)
    ae, _ = _projected_embeds(model, proc, wav, ["music"], model._sm_device)
    return ae[0].cpu().float().tolist()


def zero_shot_tags(y: np.ndarray, sr: int, tags: list[str] | None = None,
                   top_k: int = 12, device: str = "cuda") -> list[dict]:
    """Score the audio against a tag vocabulary via CLAP cosine similarity.

    Returns the top_k tags as [{"tag": str, "score": float}], softmaxed so the
    scores read as relative confidences.
    """
    import torch
    tags = tags or DEFAULT_TAGS
    model, proc = _load(device)
    wav = _to_clap_audio(y, sr)
    ae, te = _projected_embeds(model, proc, wav, tags, model._sm_device)
    sims = (ae @ te.T)[0]                       # cosine, [n_tags]
    probs = torch.softmax(sims / 0.05, dim=-1)  # temperature-sharpened
    order = torch.argsort(probs, descending=True)[:top_k]
    return [{"tag": tags[i], "score": round(float(probs[i]), 4)} for i in order]


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two embeddings (the qualifier's distance metric)."""
    va, vb = np.array(a), np.array(b)
    n = (np.linalg.norm(va) * np.linalg.norm(vb)) or 1.0
    return float(np.dot(va, vb) / n)
