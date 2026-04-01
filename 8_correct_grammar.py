"""
STEP 8: Grammar Correction — Production Module
===============================================
This is the drop-in replacement for 5_grammar_correction.py.
It uses YOUR fine-tuned model (from Step 7) instead of the generic
prithivida GEC model, making it domain-adapted to atypical speech.

Plug it into your pipeline after Step 4 (transcribe.py):
    raw_text  = transcribe_file("audio.wav")           # Step 4
    result    = correct(raw_text)                       # This file
    speak(result['corrected'])                          # TTS

Requirements:
    pip install transformers torch sentencepiece
"""

import torch
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ── Config ─────────────────────────────────────────────────────────────────────

GEC_MODEL_DIR = "./gec-model"        # Your fine-tuned model (Step 7)
TASK_PREFIX   = "Fix grammar: "
MAX_LENGTH    = 128
NUM_BEAMS     = 5

# Meaning preservation: minimum fraction of content words that must survive
MEANING_THRESHOLD = 0.40

# Stopwords to exclude from meaning check (function words only)
STOPWORDS = {
    'i', 'a', 'an', 'the', 'to', 'and', 'is', 'it', 'of', 'in',
    'my', 'me', 'gec', 'grammar', 'fix', 'you', 'he', 'she', 'we',
    'they', 'was', 'are', 'be', 'been', 'have', 'has', 'had', 'do',
    'did', 'will', 'would', 'can', 'could', 'that', 'this', 'at',
    'on', 'for', 'with', 'as', 'by', 'from', 'or', 'but', 'not',
}

# ── Load model ─────────────────────────────────────────────────────────────────

_tokenizer = None
_model     = None
_device    = None

def _load():
    global _tokenizer, _model, _device
    if _model is not None:
        return

    print(f"Loading GEC model from {GEC_MODEL_DIR}...")
    _tokenizer = AutoTokenizer.from_pretrained(GEC_MODEL_DIR)
    _model     = AutoModelForSeq2SeqLM.from_pretrained(GEC_MODEL_DIR)
    _device    = "cuda" if torch.cuda.is_available() else "cpu"
    _model     = _model.to(_device)
    _model.eval()
    print(f"  Loaded on {_device}\n")

# ── Core correction ────────────────────────────────────────────────────────────

def _correct_raw(text: str) -> str:
    """Run the model. Returns corrected string."""
    prompt = TASK_PREFIX + text.strip()
    inputs = _tokenizer(
        prompt,
        return_tensors="pt",
        max_length=MAX_LENGTH,
        truncation=True,
    ).to(_device)

    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_length=MAX_LENGTH,
            num_beams=NUM_BEAMS,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

    corrected = _tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    # Strip task prefix if the model echoed it
    for prefix in [TASK_PREFIX, "Fix grammar:", "grammar:", "correct:"]:
        if corrected.lower().startswith(prefix.lower()):
            corrected = corrected[len(prefix):].strip()

    # Capitalise
    if corrected:
        corrected = corrected[0].upper() + corrected[1:]

    return corrected

# ── Meaning preservation check ────────────────────────────────────────────────

def _content_words(text: str):
    """Return set of content (non-stopword) words, lowercased."""
    return {w.lower() for w in re.findall(r'\b\w+\b', text) if w.lower() not in STOPWORDS}

def meaning_preserved(original: str, corrected: str) -> bool:
    """
    True if the core content words of `original` are substantially
    represented in `corrected`. Handles additions (OK) but flags deletions.
    """
    orig_content = _content_words(original)
    if not orig_content:
        return True   # Nothing to preserve
    corr_content = _content_words(corrected)
    overlap = orig_content & corr_content
    return (len(overlap) / len(orig_content)) >= MEANING_THRESHOLD

# ── Alternatives (beam diversity) ─────────────────────────────────────────────

def _correct_with_alternatives(text: str, n: int = 3):
    """Return top-n beam candidates for low-confidence situations."""
    prompt = TASK_PREFIX + text.strip()
    inputs = _tokenizer(
        prompt,
        return_tensors="pt",
        max_length=MAX_LENGTH,
        truncation=True,
    ).to(_device)

    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_length=MAX_LENGTH,
            num_beams=max(n * 2, 8),
            num_return_sequences=n,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

    results = []
    for ids in output_ids:
        text_out = _tokenizer.decode(ids, skip_special_tokens=True).strip()
        for prefix in [TASK_PREFIX, "Fix grammar:", "grammar:", "correct:"]:
            if text_out.lower().startswith(prefix.lower()):
                text_out = text_out[len(prefix):].strip()
        if text_out:
            text_out = text_out[0].upper() + text_out[1:]
        results.append(text_out)

    # Deduplicate while preserving order
    seen, unique = set(), []
    for r in results:
        if r.lower() not in seen:
            seen.add(r.lower())
            unique.append(r)
    return unique

# ── Public API ────────────────────────────────────────────────────────────────

def correct(raw_text: str, confidence: float = 1.0, confidence_threshold: float = 0.7) -> dict:
    """
    Main entry point. Takes raw Whisper transcript, returns correction dict.

    Args:
        raw_text:             Raw transcript from Whisper (Step 4)
        confidence:           Whisper confidence score (0–1). Pass from Step 4
                              if available; defaults to 1.0 (assume confident).
        confidence_threshold: Below this, surface alternatives for user review.

    Returns dict with keys:
        original          — the raw input
        corrected         — best correction
        alternatives      — list of other candidates (only when low confidence)
        meaning_preserved — bool: did we keep the original meaning?
        needs_review      — bool: should the user confirm before speaking?
    """
    _load()

    raw = raw_text.strip()
    if not raw:
        return {
            'original': raw,
            'corrected': '',
            'alternatives': [],
            'meaning_preserved': True,
            'needs_review': False,
        }

    low_confidence = confidence < confidence_threshold

    if low_confidence:
        # Get multiple candidates
        candidates   = _correct_with_alternatives(raw, n=3)
        corrected    = candidates[0] if candidates else raw
        alternatives = candidates[1:] if len(candidates) > 1 else []
    else:
        corrected    = _correct_raw(raw)
        alternatives = []

    meaning_ok   = meaning_preserved(raw, corrected)
    needs_review = low_confidence or not meaning_ok

    # If meaning check failed, add corrected as an alternative and fall back to raw
    if not meaning_ok and not alternatives:
        alternatives = [corrected]
        corrected    = raw[0].upper() + raw[1:] if raw else raw

    return {
        'original':          raw,
        'corrected':         corrected,
        'alternatives':      alternatives,
        'meaning_preserved': meaning_ok,
        'needs_review':      needs_review,
    }

# ── Batch correction ──────────────────────────────────────────────────────────

def correct_batch(texts: list) -> list:
    """
    Correct multiple transcripts at once (more efficient for evaluation).
    Returns list of corrected strings (no dict overhead).
    """
    _load()
    if not texts:
        return []

    prompts = [TASK_PREFIX + t.strip() for t in texts]
    inputs  = _tokenizer(
        prompts,
        return_tensors="pt",
        max_length=MAX_LENGTH,
        truncation=True,
        padding=True,
    ).to(_device)

    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_length=MAX_LENGTH,
            num_beams=NUM_BEAMS,
            early_stopping=True,
        )

    results = []
    for ids in output_ids:
        text_out = _tokenizer.decode(ids, skip_special_tokens=True).strip()
        for prefix in [TASK_PREFIX, "Fix grammar:", "grammar:", "correct:"]:
            if text_out.lower().startswith(prefix.lower()):
                text_out = text_out[len(prefix):].strip()
        if text_out:
            text_out = text_out[0].upper() + text_out[1:]
        results.append(text_out)

    return results

# ── Evaluate BLEU on CSV ──────────────────────────────────────────────────────

def evaluate_on_csv(csv_path: str = "gec_pairs.csv", n: int = 100):
    """Quick BLEU/exact-match evaluation on the parsed TalkBank pairs."""
    import pandas as pd
    import evaluate as ev

    _load()
    bleu_metric = ev.load("sacrebleu")

    df = pd.read_csv(csv_path)
    # Prefer echo pairs for evaluation (real data, not synthetic)
    echo_df = df[df['pair_type'] == 'echo_recast']
    if len(echo_df) >= n:
        df = echo_df.sample(n, random_state=42)
    else:
        df = df.sample(min(n, len(df)), random_state=42)

    sources      = df['source'].tolist()
    references   = df['target'].tolist()
    predictions  = correct_batch(sources)

    bleu   = bleu_metric.compute(predictions=predictions,
                                  references=[[r] for r in references])
    exact  = sum(p.lower().strip() == r.lower().strip()
                 for p, r in zip(predictions, references)) / len(predictions)

    print(f"Evaluated on {len(df)} pairs")
    print(f"  BLEU:        {bleu['score']:.2f}")
    print(f"  Exact match: {exact:.2%}")
    print("\nSample outputs:")
    for src, ref, pred in list(zip(sources, references, predictions))[:8]:
        print(f"  INPUT:  {src}")
        print(f"  TARGET: {ref}")
        print(f"  PRED:   {pred}")
        print()

# ── CLI / test mode ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    _load()

    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        csv  = sys.argv[2] if len(sys.argv) > 2 else "gec_pairs.csv"
        n    = int(sys.argv[3]) if len(sys.argv) > 3 else 100
        evaluate_on_csv(csv, n)
        sys.exit(0)

    print("=" * 55)
    print("Grammar Correction Module — Fine-tuned on TalkBank")
    print("=" * 55)

    test_phrases = [
        "want go park",
        "need bathroom",
        "she like apple every day",
        "my name john want drink water",
        "can help me find phone",
        "feeling happy today outside",
        "go store mama",
        "no want that",
        "more cookie please",
        "did work Peggy at school",
    ]

    print("\n--- Built-in tests ---")
    for phrase in test_phrases:
        result = correct(phrase)
        status = "✓" if result['meaning_preserved'] else "⚠"
        print(f"  {status} IN:  {result['original']}")
        print(f"     OUT: {result['corrected']}")
        if result['needs_review']:
            print(f"     [Review needed]")
        print()

    print("\n--- Try your own (type 'quit' to exit) ---")
    while True:
        user_input = input("Raw transcript: ").strip()
        if user_input.lower() in ('quit', 'q', 'exit'):
            break
        if not user_input:
            continue

        result = correct(user_input)
        print(f"\n  Raw:       {result['original']}")
        print(f"  Corrected: {result['corrected']}")
        if result['alternatives']:
            print(f"  Other options: {result['alternatives']}")
        if result['needs_review']:
            print("  ⚠ Low confidence — please review before speaking")
        print()
