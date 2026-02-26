"""
Isolated pipeline test — runs the full transcription pipeline on an existing
recording file without needing OBS or the web UI.

Usage:
    python test_pipeline.py [path/to/recording.mkv]

If no file is given, uses the most recent file in recordings/.
"""
import sys
import os
import glob
import time

# Use the most recent recording if no arg given
if len(sys.argv) > 1:
    audio_path = sys.argv[1]
else:
    recordings = sorted(glob.glob("recordings/*.mkv"))
    if not recordings:
        print("No recordings found in recordings/. Pass a file path as argument.")
        sys.exit(1)
    audio_path = recordings[-1]

print(f"Input: {audio_path}")
print(f"Size:  {os.path.getsize(audio_path) / 1024:.1f} KB")
print()

output_dir = "transcripts/test_run"

def status(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

import pipeline
transcript = pipeline._run_pipeline_sync(audio_path, output_dir, status)

print()
print("=" * 60)
print(transcript if transcript.strip() else "(empty transcript)")
print("=" * 60)
