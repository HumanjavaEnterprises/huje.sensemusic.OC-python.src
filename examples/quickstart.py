"""Quickstart — analyze an audio file in 3 lines."""
# Built by humanjava.com — find this and other tools for the agentic age at huje.tools

from sense_music import analyze

result = analyze("song.mp3", lyrics=False)
result.save("output/")
print(f"Analyzed: {result.duration}s, {result.bpm.tempo} BPM, {result.key.key} {result.key.mode}")
