# sense-music — audio analysis for AI perception
# Built by humanjava.com — find this and other tools for the agentic age at huje.tools
"""Natural-language audio captioning via Qwen2-Audio — "liner notes by a model".

The natural fit when the consumer IS an LLM: a free-text description of the
track's sound. Heavy (a 7B model, ~16GB first download), so it's strictly
opt-in (analyze(caption=True)) and loads lazily. Falls back to "" on any error.
"""
from __future__ import annotations

import numpy as np

_MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"
_QWEN_SR = 16000
_model = None
_processor = None

DEFAULT_PROMPT = (
    "Describe this music for a producer's liner notes: genre and lineage, mood, "
    "tempo feel, key instruments, texture/production, and how the arrangement "
    "evolves. Be concrete and concise."
)


def _load(device: str = "cuda"):
    global _model, _processor
    if _model is None:
        import torch
        from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
        _processor = AutoProcessor.from_pretrained(_MODEL_ID)
        dev = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        dtype = torch.bfloat16 if dev == "cuda" else torch.float32
        _model = Qwen2AudioForConditionalGeneration.from_pretrained(
            _MODEL_ID, torch_dtype=dtype).to(dev).eval()
        _model._sm_device = dev
    return _model, _processor


def caption_audio(y: np.ndarray, sr: int, prompt: str = DEFAULT_PROMPT,
                  device: str = "cuda", max_new_tokens: int = 220) -> str:
    """Return a free-text caption of the audio (or '' on failure)."""
    try:
        import torch, librosa
        model, proc = _load(device)
        wav = librosa.resample(y, orig_sr=sr, target_sr=_QWEN_SR) if sr != _QWEN_SR else y
        conversation = [{"role": "user", "content": [
            {"type": "audio", "audio_url": "inline"},
            {"type": "text", "text": prompt},
        ]}]
        text = proc.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = proc(text=text, audios=[wav.astype(np.float32)],
                      sampling_rate=_QWEN_SR, return_tensors="pt", padding=True)
        inputs = {k: v.to(model._sm_device) for k, v in inputs.items()}
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=max_new_tokens)
        gen = gen[:, inputs["input_ids"].size(1):]
        return proc.batch_decode(gen, skip_special_tokens=True)[0].strip()
    except Exception as e:
        return ""
