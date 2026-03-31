"""
STEP 4: Transcribe Speech with Your Fine-tuned Model
=====================================================
Run this after training. Transcribes audio files using your fine-tuned
Whisper model. Can also record live from your microphone.

Requirements:
    pip install transformers librosa sounddevice scipy
"""

import torch
import librosa
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration

MODEL_DIR = "./whisper-torgo"   # Path to your fine-tuned model

# ── Load model ────────────────────────────────────────────────────────────────

print(f"Loading model from {MODEL_DIR}...")
processor = WhisperProcessor.from_pretrained(MODEL_DIR)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_DIR)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()
print(f"Model loaded on {device}\n")

# ── Transcribe an audio file ──────────────────────────────────────────────────

def transcribe_file(audio_path):
    """Transcribe a .wav or .mp3 file."""
    audio, sr = librosa.load(audio_path, sr=16000)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            language="en",
            task="transcribe",
        )

    transcript = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcript.strip()

# ── Record from microphone ────────────────────────────────────────────────────

def transcribe_microphone(duration=5):
    """Record from microphone and transcribe."""
    import sounddevice as sd
    from scipy.io.wavfile import write

    print(f"Recording for {duration} seconds... Speak now!")
    audio = sd.rec(int(duration * 16000), samplerate=16000, channels=1, dtype=np.int16)
    sd.wait()
    print("Recording done. Transcribing...")

    # Save temp file and transcribe
    temp_path = "temp_recording.wav"
    write(temp_path, 16000, audio)
    return transcribe_file(temp_path)

# ── Evaluate WER on test files ────────────────────────────────────────────────

def evaluate_on_csv(csv_path, num_samples=50):
    """Run WER evaluation on a sample of your dataset."""
    import pandas as pd
    import evaluate

    wer_metric = evaluate.load("wer")
    df = pd.read_csv(csv_path).sample(min(num_samples, len(pd.read_csv(csv_path))))

    predictions = []
    references  = []

    for _, row in df.iterrows():
        try:
            pred = transcribe_file(row["audio_path"])
            predictions.append(pred)
            references.append(row["transcript"])
            print(f"  REF:  {row['transcript']}")
            print(f"  PRED: {pred}\n")
        except Exception as e:
            print(f"  Error on {row['audio_path']}: {e}")

    wer = wer_metric.compute(predictions=predictions, references=references)
    print(f"Word Error Rate (WER): {wer:.2%}")
    print(f"(Lower is better — baseline Whisper on dysarthric speech is typically 40-80%)")

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Transcribe a file passed as argument: python 4_transcribe.py audio.wav
        audio_file = sys.argv[1]
        print(f"Transcribing: {audio_file}")
        result = transcribe_file(audio_file)
        print(f"Transcript: {result}")

    else:
        # Interactive menu
        print("What would you like to do?")
        print("  1. Transcribe an audio file")
        print("  2. Record from microphone")
        print("  3. Evaluate WER on dataset")
        choice = input("\nChoice (1/2/3): ").strip()

        if choice == "1":
            path = input("Path to audio file: ").strip()
            print(f"\nTranscript: {transcribe_file(path)}")

        elif choice == "2":
            secs = int(input("Recording duration in seconds (default 5): ").strip() or "5")
            print(f"\nTranscript: {transcribe_microphone(secs)}")

        elif choice == "3":
            csv = input("Path to CSV (default: torgo_pairs.csv): ").strip() or "torgo_pairs.csv"
            n   = int(input("Number of samples to evaluate (default 50): ").strip() or "50")
            evaluate_on_csv(csv, n)
