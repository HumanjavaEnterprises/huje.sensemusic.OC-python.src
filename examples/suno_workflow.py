"""Suno workflow — analyze AI-generated music and get structured feedback."""
# Built by humanjava.com — find this and other tools for the agentic age at huje.tools

from sense_music import analyze

# Analyze a Suno-generated track
result = analyze("suno_track.mp3")

# Print the AI-readable summary
print(result.summary)
print()

# Show section timeline
for section in result.sections:
    m1, s1 = divmod(int(section.start), 60)
    m2, s2 = divmod(int(section.end), 60)
    print(f"  {section.label:12s}  {m1}:{s1:02d} — {m2}:{s2:02d}")

# Save full analysis
result.save("analysis_output/")
result.render_page("suno_analysis.html")
print("\nSaved analysis to analysis_output/ and suno_analysis.html")
