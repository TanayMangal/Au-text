"""
STEP 5: Grammar Correction Layer
=================================
Takes the raw transcript from Whisper and corrects it into proper
American English while preserving the speaker's intended meaning.

Uses happy-transformer GEC model — free, local, best quality for this task.
Runs on your GPU automatically.

Requirements:
    pip install transformers torch sentencepiece
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ── Load model ────────────────────────────────────────────────────────────────

# This model is specifically trained for grammar error correction
# and handles incomplete/broken speech much better
MODEL_NAME = "prithivida/grammar_error_correcter_v1"

print("Loading grammar correction model...")
print("(First run downloads ~500MB — subsequent runs are instant)")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
model.eval()

print("Model loaded!\n")

# ── Core correction function ──────────────────────────────────────────────────

def correct_grammar(raw_text, max_length=128):
    """
    Takes a raw/broken transcript and returns a grammatically
    correct version that preserves the original meaning.

    Example:
        Input:  "i wan go park today with dog"
        Output: "I want to go to the park today with the dog."
    """
    if not raw_text or not raw_text.strip():
        return ""

    # This model uses "gec:" prefix
    prompt = f"gec: {raw_text.strip()}"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=256,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=5,
            early_stopping=True,
        )

    corrected = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Strip any leftover prefix the model might include
    for prefix in ["gec:", "grammar:", "correct:"]:
        if corrected.lower().startswith(prefix):
            corrected = corrected[len(prefix):].strip()

    # Capitalise first letter
    if corrected:
        corrected = corrected[0].upper() + corrected[1:]

    return corrected

# ── Meaning check ─────────────────────────────────────────────────────────────

def meaning_check(original, corrected):
    """
    Check the corrected sentence hasn't drifted too far from original meaning.
    Returns True if they seem to be about the same thing.
    """
    original_words  = set(original.lower().split())
    corrected_words = set(corrected.lower().split())

    # Remove filler words from comparison
    stopwords = {"i", "a", "the", "to", "and", "is", "it", "of",
                 "in", "my", "me", "gec", "grammar"}
    original_words  -= stopwords
    corrected_words -= stopwords

    if not original_words:
        return True

    overlap    = original_words & corrected_words
    similarity = len(overlap) / len(original_words)
    return similarity >= 0.4

# ── Full pipeline function ────────────────────────────────────────────────────

def process_transcript(raw_text, confidence=1.0, confidence_threshold=0.7):
    """
    Full processing pipeline:
    1. Correct grammar
    2. Check meaning is preserved
    3. Flag for user review if low confidence
    """
    corrected    = correct_grammar(raw_text)
    meaning_ok   = meaning_check(raw_text, corrected)
    needs_review = (confidence < confidence_threshold) or not meaning_ok

    return {
        "original":          raw_text,
        "corrected":         corrected,
        "meaning_preserved": meaning_ok,
        "needs_review":      needs_review,
    }

# ── Test it ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("Grammar Correction Layer — Test Mode")
    print("=" * 55)
    print("Type a sentence as it might come out of Whisper.")
    print("Type 'quit' to exit.\n")

    # Built in tests
    test_phrases = [
        "i wan go park today",
        "i ned use bathroom",
        "she like eat apple every day",
        "my name john i want drink water",
        "can help me find phone please",
        "i feeling happy today go outside",
        "i wan go parc",
    ]

    print("--- Built-in tests ---")
    for phrase in test_phrases:
        result = process_transcript(phrase)
        print(f"  Input:      {result['original']}")
        print(f"  Corrected:  {result['corrected']}")
        print(f"  Meaning OK: {result['meaning_preserved']}")
        print()

    # Interactive mode
    print("--- Try your own ---")
    while True:
        user_input = input("Enter raw transcript (or 'quit'): ").strip()
        if user_input.lower() == "quit":
            break
        if not user_input:
            continue

        result = process_transcript(user_input)
        print(f"\n  Raw:       {result['original']}")
        print(f"  Corrected: {result['corrected']}")
        if result['needs_review']:
            print("  Warning: Low confidence - user should review this")
        print()
