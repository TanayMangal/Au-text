"""
STEP 2: Parse TORGO Dataset into Audio/Transcript Pairs
========================================================
Run this after downloading. Walks through the TORGO folder structure,
pairs each .wav file with its transcript from the prompts/ folder,
and saves everything to a clean CSV ready for training.

TORGO structure:
  torgo_data/
    F01/
      Session1/
        wav_headMic/      <- audio files (0001.wav, 0002.wav, ...)
        prompts/          <- transcript files (0001, 0002, ...)
    M01/
      ...
"""

import os
import csv
import re

TORGO_DIR = "torgo_data"
OUTPUT_CSV = "torgo_pairs.csv"

def get_transcript(prompts_dir, utterance_id):
    """Read transcript for a given utterance ID."""
    # Try both with and without .txt extension
    prompt_file = os.path.join(prompts_dir, utterance_id + ".txt")
    if not os.path.exists(prompt_file):
        prompt_file = os.path.join(prompts_dir, utterance_id)
    if not os.path.exists(prompt_file):
        return None

    with open(prompt_file, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().strip()

    # Skip spurious/unusable prompts
    if text == "xxx" or text.endswith(".jpg") or not text:
        return None

    return text

def parse_torgo(torgo_dir):
    """Walk TORGO directory and collect all valid audio/transcript pairs."""
    pairs = []

    for speaker in sorted(os.listdir(torgo_dir)):
        speaker_dir = os.path.join(torgo_dir, speaker)
        if not os.path.isdir(speaker_dir):
            continue

        # Determine if dysarthric or control
        # Speaker codes: F01, F03, M01... = dysarthric | FC01, MC01... = control
        is_dysarthric = not ("C" in speaker[1:3])  # FC or MC = control
        gender = "F" if speaker.startswith("F") else "M"

        for session in sorted(os.listdir(speaker_dir)):
            session_dir = os.path.join(speaker_dir, session)
            if not os.path.isdir(session_dir) or not session.startswith("Session"):
                continue

            # Prefer head microphone (cleaner for ASR)
            wav_dir = os.path.join(session_dir, "wav_headMic")
            if not os.path.exists(wav_dir):
                wav_dir = os.path.join(session_dir, "wav_arrayMic")
            if not os.path.exists(wav_dir):
                continue

            prompts_dir = os.path.join(session_dir, "prompts")
            if not os.path.exists(prompts_dir):
                continue

            for wav_file in sorted(os.listdir(wav_dir)):
                if not wav_file.endswith(".wav"):
                    continue

                utterance_id = os.path.splitext(wav_file)[0]  # e.g. "0001"
                wav_path = os.path.abspath(os.path.join(wav_dir, wav_file))
                transcript = get_transcript(prompts_dir, utterance_id)

                if transcript is None:
                    continue

                pairs.append({
                    "audio_path": wav_path,
                    "transcript": transcript,
                    "speaker": speaker,
                    "session": session,
                    "gender": gender,
                    "is_dysarthric": is_dysarthric,
                })

    return pairs

print("Parsing TORGO dataset...")
pairs = parse_torgo(TORGO_DIR)

# Save to CSV
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["audio_path", "transcript", "speaker", "session", "gender", "is_dysarthric"])
    writer.writeheader()
    writer.writerows(pairs)

# Summary
dysarthric = [p for p in pairs if p["is_dysarthric"]]
control = [p for p in pairs if not p["is_dysarthric"]]

print(f"\nDone! Saved to {OUTPUT_CSV}")
print(f"  Total pairs:        {len(pairs)}")
print(f"  Dysarthric speech:  {len(dysarthric)}")
print(f"  Control speech:     {len(control)}")
print(f"\nSpeakers found: {sorted(set(p['speaker'] for p in pairs))}")
