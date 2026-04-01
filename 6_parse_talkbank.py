"""
STEP 6: Parse TalkBank CHAT Files into GEC Training Pairs
==========================================================
Extracts (atypical_utterance, corrected_utterance) pairs from .cha files
for training the grammar correction model.

Strategy:
  *CHI: lines  → the atypical/broken utterance (model input)
  *MOT:/*INV: lines that immediately ECHO or RECAST the CHI line → target

  We use two extraction methods:
    A) Direct echo detection — adult says nearly the same words back correctly
       (e.g. CHI: "want cookie" → MOT: "you want a cookie?")
    B) All CHI utterances → fed into rule-based augmenter to create synthetic
       pairs for pre-training (missing articles, dropped subjects, etc.)

Output: gec_pairs.csv with columns: source, target, pair_type, speaker, corpus

Run from root of project:
    python 6_parse_talkbank.py
"""

import os
import re
import csv
import zipfile
import tempfile
import shutil
from pathlib import Path
from difflib import SequenceMatcher

# ── Config ─────────────────────────────────────────────────────────────────────

# Zip files containing the .cha corpora
# Place all your zip files in the same directory as this script
ZIP_FILES = [
    "Flusberg.zip",
    "Rollins.zip",
    "Eigsti.zip",
    "AAC.zip",
    "QuigleyMcNally.zip",
    "NYU-Emerson.zip",
    "Nadig.zip",
    "AFG_datap1.zip",
]

OUTPUT_CSV = "gec_pairs.csv"
MIN_WORDS  = 2    # Skip single-word utterances
MAX_WORDS  = 40   # Skip very long utterances (likely transcription artifacts)

# ── CHAT parser ────────────────────────────────────────────────────────────────

UNINTELLIGIBLE = re.compile(r'\b(xxx|yyy|www|0)\b')
CHAT_CLEANUP   = re.compile(r'[&+\[\]<>@%]|\(\.+\)|_|\+[/!?]')
# Remove CHAT special codes like &-uh, [: word], etc.
CHAT_CODES     = re.compile(r'&-\w+|\[:\s*[^\]]+\]|\[!\]|\[//\]|\[/\]|\[=!\s*[^\]]+\]|\+\.\.\.')

def clean_chat_line(text: str) -> str:
    """Strip CHAT-format markup from a transcript line."""
    text = CHAT_CODES.sub('', text)
    text = CHAT_CLEANUP.sub(' ', text)
    # Remove trailing punctuation markers (. ! ?)
    text = re.sub(r'\s*[.!?]+\s*$', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def is_usable(text: str) -> bool:
    """Return True if the utterance is usable for training."""
    if not text:
        return False
    if UNINTELLIGIBLE.search(text):
        return False
    words = text.split()
    if len(words) < MIN_WORDS or len(words) > MAX_WORDS:
        return False
    # Skip pure filler
    if all(w.lower() in {'uh', 'um', 'mm', 'hmm', 'oh', 'ah', 'yeah', 'no', 'yes', 'ok'} for w in words):
        return False
    return True

def parse_cha_file(filepath: str):
    """
    Parse a .cha file and return list of (speaker_code, cleaned_text) tuples.
    speaker_code is one of: CHI, ADU (adult — MOT/INV/FAT/etc.)
    """
    utterances = []
    try:
        with open(filepath, encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception:
        return utterances

    i = 0
    while i < len(lines):
        line = lines[i].rstrip('\n')

        # Handle continuation lines (start with \t, no speaker prefix)
        if line.startswith('*'):
            match = re.match(r'^\*([A-Z]+):\t(.+)', line)
            if match:
                code = match.group(1)
                text = match.group(2)

                # Collect continuation lines
                j = i + 1
                while j < len(lines) and lines[j].startswith('\t') and not lines[j].startswith('\t%'):
                    text += ' ' + lines[j].strip()
                    j += 1

                text = clean_chat_line(text)
                speaker = 'CHI' if code == 'CHI' else 'ADU'
                utterances.append((speaker, text, code))
        i += 1

    return utterances

# ── Echo/recast detection ──────────────────────────────────────────────────────

def word_overlap(a: str, b: str) -> float:
    """Fraction of words from `a` that appear in `b`."""
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa:
        return 0.0
    return len(wa & wb) / len(wa)

def is_echo_or_recast(chi_text: str, adu_text: str) -> bool:
    """
    Return True if the adult utterance is an echo or recast of the child's.
    A recast keeps the core vocabulary but adds/fixes grammar.
    We want:
      - High word overlap (adult uses child's content words)
      - Adult version is longer or same length (they add grammar)
      - Adult version is NOT identical (that's just a pure echo, less useful)
    """
    chi_words = chi_text.lower().split()
    adu_words = adu_text.lower().split()

    overlap = word_overlap(chi_text, adu_text)
    seq_ratio = SequenceMatcher(None, chi_text.lower(), adu_text.lower()).ratio()

    # Good recast: high overlap, adult ≥ child length, not identical
    if overlap >= 0.55 and len(adu_words) >= len(chi_words) and chi_text.lower() != adu_text.lower():
        return True
    # Also catch near-identical but slightly expanded (seq ratio high)
    if seq_ratio >= 0.7 and len(adu_words) > len(chi_words):
        return True
    return False

def extract_echo_pairs(utterances):
    """
    Slide a window: for each CHI utterance, look at the next 1-3 ADU utterances
    for echo/recasts. This is the highest-quality pair type.
    """
    pairs = []
    for i, (speaker, text, code) in enumerate(utterances):
        if speaker != 'CHI':
            continue
        if not is_usable(text):
            continue

        # Look at next 3 utterances for a recast
        for j in range(i + 1, min(i + 4, len(utterances))):
            nspeaker, ntext, ncode = utterances[j]
            if nspeaker == 'CHI':
                break  # CHI spoke again, window closed
            if nspeaker == 'ADU' and is_usable(ntext):
                if is_echo_or_recast(text, ntext):
                    pairs.append({
                        'source': text,
                        'target': capitalize(ntext),
                        'pair_type': 'echo_recast',
                    })
                    break  # Take first good recast only

    return pairs

def capitalize(s: str) -> str:
    return s[0].upper() + s[1:] if s else s

# ── Synthetic augmentation ─────────────────────────────────────────────────────
# For CHI utterances without a detected recast, we apply rule-based augmentation
# to generate (broken → corrected) pairs. This extends the dataset significantly.

import random
random.seed(42)

ARTICLES = {'a', 'an', 'the'}
AUX_VERBS = {'is', 'are', 'was', 'were', 'am', 'be', 'been', 'being',
              'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
              'can', 'could', 'shall', 'should', 'may', 'might', 'must'}
PRONOUNS = {'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'me', 'him', 'her', 'us', 'them', 'my', 'your',
            'his', 'its', 'our', 'their'}

def drop_articles(text: str) -> str:
    """Simulate dropped articles — common in Down syndrome speech."""
    words = text.split()
    result = []
    for w in words:
        if w.lower() in ARTICLES and random.random() < 0.6:
            continue
        result.append(w)
    return ' '.join(result) if result else text

def drop_subject(text: str) -> str:
    """Drop leading pronoun — 'I want water' → 'want water'."""
    words = text.split()
    if words and words[0].lower() in PRONOUNS and len(words) > 2:
        return ' '.join(words[1:])
    return text

def drop_aux(text: str) -> str:
    """Drop auxiliary verb — 'I am going' → 'I going'."""
    words = text.split()
    result = []
    for i, w in enumerate(words):
        if w.lower() in AUX_VERBS and i > 0 and random.random() < 0.5:
            continue
        result.append(w)
    return ' '.join(result) if result else text

def make_telegraphic(text: str) -> str:
    """Combine multiple simplifications to produce telegraphic speech."""
    funcs = [drop_articles, drop_subject, drop_aux]
    random.shuffle(funcs)
    result = text
    for fn in funcs[:random.randint(1, 2)]:
        result = fn(result)
    return result

def augment_chi_utterance(text: str):
    """
    Given a (relatively grammatical) CHI utterance, produce synthetic
    (broken, corrected) pairs by degrading the input.

    Also handles the reverse: CHI utterances that ARE already broken
    become the source directly, and we just use them as-is.
    """
    corrected = capitalize(text)
    broken = make_telegraphic(text)
    if broken == corrected.lower() or broken == text:
        return None  # No change, not useful
    return {
        'source': broken,
        'target': corrected,
        'pair_type': 'synthetic_augmented',
    }

# ── File extraction helpers ────────────────────────────────────────────────────

def find_cha_files(directory: str):
    """Recursively find all .cha and .cha.txt files."""
    found = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.endswith('.cha') or f.endswith('.cha.txt'):
                found.append(os.path.join(root, f))
    return found

def extract_zip_to_tmp(zip_path: str) -> str:
    """Extract a zip file to a temp directory and return the path."""
    tmp = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(tmp)
    return tmp

# ── Main ───────────────────────────────────────────────────────────────────────

def process_corpus(zip_path: str, corpus_name: str):
    """Process one zip file, return list of pair dicts."""
    print(f"  Processing {corpus_name}...")
    tmp_dir = extract_zip_to_tmp(zip_path)
    cha_files = find_cha_files(tmp_dir)
    print(f"    Found {len(cha_files)} .cha files")

    all_pairs  = []
    chi_count  = 0

    for cha_file in cha_files:
        utterances = parse_cha_file(cha_file)

        # Method A: echo/recast pairs (highest quality)
        echo_pairs = extract_echo_pairs(utterances)
        for p in echo_pairs:
            p['corpus'] = corpus_name
        all_pairs.extend(echo_pairs)

        # Method B: collect all CHI utterances for augmentation
        for speaker, text, code in utterances:
            if speaker == 'CHI' and is_usable(text):
                chi_count += 1
                aug = augment_chi_utterance(text)
                if aug:
                    aug['corpus'] = corpus_name
                    all_pairs.append(aug)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    echo_count = sum(1 for p in all_pairs if p['pair_type'] == 'echo_recast')
    synth_count = sum(1 for p in all_pairs if p['pair_type'] == 'synthetic_augmented')
    print(f"    CHI utterances: {chi_count} | Echo pairs: {echo_count} | Synthetic pairs: {synth_count}")
    return all_pairs


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    all_pairs  = []

    print("=" * 60)
    print("TalkBank CHAT → GEC Pairs Parser")
    print("=" * 60)

    for zip_name in ZIP_FILES:
        zip_path = os.path.join(script_dir, zip_name)
        if not os.path.exists(zip_path):
            print(f"  WARNING: {zip_name} not found, skipping.")
            continue
        corpus_name = zip_name.replace('.zip', '')
        pairs = process_corpus(zip_path, corpus_name)
        all_pairs.extend(pairs)

    # Deduplicate on (source, target)
    seen   = set()
    unique = []
    for p in all_pairs:
        key = (p['source'].lower().strip(), p['target'].lower().strip())
        if key not in seen and p['source'].lower() != p['target'].lower():
            seen.add(key)
            unique.append(p)

    print(f"\nTotal unique pairs: {len(unique)}")

    # Save
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['source', 'target', 'pair_type', 'corpus'])
        writer.writeheader()
        writer.writerows(unique)

    # Summary
    echo_total   = sum(1 for p in unique if p['pair_type'] == 'echo_recast')
    synth_total  = sum(1 for p in unique if p['pair_type'] == 'synthetic_augmented')
    corpora_seen = sorted(set(p['corpus'] for p in unique))

    print(f"\nSaved to {OUTPUT_CSV}")
    print(f"  Echo/recast pairs:   {echo_total}")
    print(f"  Synthetic pairs:     {synth_total}")
    print(f"  Corpora used:        {corpora_seen}")
    print("\nSample pairs:")
    for p in unique[:5]:
        print(f"  [{p['pair_type']}]")
        print(f"    SRC: {p['source']}")
        print(f"    TGT: {p['target']}")
