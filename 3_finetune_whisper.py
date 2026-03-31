"""
STEP 3: Fine-tune Whisper on TORGO Dataset
==========================================
Run this after parsing. Fine-tunes openai/whisper-small on the TORGO
dysarthric speech data. Optimised for an NVIDIA RTX 4060 (8GB VRAM).

Requirements:
    pip install transformers datasets accelerate librosa soundfile evaluate jiwer torch
"""

import os
import torch
import librosa
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from datasets import Dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import evaluate

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME      = "openai/whisper-small"
CSV_PATH        = "torgo_pairs.csv"
OUTPUT_DIR      = "./whisper-torgo"
DYSARTHRIC_ONLY = True
TRAIN_SPLIT     = 0.85
SEED            = 42

# ── Load data ─────────────────────────────────────────────────────────────────

print("Loading dataset...")
df = pd.read_csv(CSV_PATH)

if DYSARTHRIC_ONLY:
    df = df[df["is_dysarthric"] == True].reset_index(drop=True)
    print(f"Using dysarthric speakers only: {df['speaker'].unique().tolist()}")

print(f"Total samples: {len(df)}")

# Train/eval split by speaker to avoid data leakage
speakers = df["speaker"].unique().tolist()
np.random.seed(SEED)
np.random.shuffle(speakers)

split_idx      = int(len(speakers) * TRAIN_SPLIT)
train_speakers = speakers[:split_idx]
eval_speakers  = speakers[split_idx:]

train_df = df[df["speaker"].isin(train_speakers)].reset_index(drop=True)
eval_df  = df[df["speaker"].isin(eval_speakers)].reset_index(drop=True)

print(f"Train samples: {len(train_df)} | Eval samples: {len(eval_df)}")
print(f"Train speakers: {train_speakers}")
print(f"Eval speakers:  {eval_speakers}")

# ── Load model & processor ────────────────────────────────────────────────────

print(f"\nLoading {MODEL_NAME}...")
processor = WhisperProcessor.from_pretrained(MODEL_NAME)
model     = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

model.config.forced_decoder_ids = None
model.generation_config.suppress_tokens = []
model.generation_config.forced_decoder_ids = None

# ── Preprocessing — load audio directly with librosa ─────────────────────────

def make_dataset(dataframe, label):
    """Load audio files and extract Whisper features using librosa."""
    input_features_list = []
    labels_list         = []
    skipped             = 0
    total               = len(dataframe)

    for i, row in dataframe.iterrows():
        try:
            audio, _ = librosa.load(row["audio_path"], sr=16000)
            inputs    = processor(audio, sampling_rate=16000, return_tensors="pt")
            input_features_list.append(inputs.input_features[0].numpy())
            label_ids = processor.tokenizer(row["transcript"]).input_ids
            labels_list.append(label_ids)
        except Exception:
            skipped += 1
            continue

        count = len(input_features_list)
        if count % 200 == 0:
            print(f"  Processed {count}/{total} files...")

    print(f"  Done. Skipped {skipped} files due to errors.")
    return Dataset.from_dict({
        "input_features": input_features_list,
        "labels":         labels_list,
    })

print("\nPreprocessing training data (this takes a few minutes)...")
train_dataset = make_dataset(train_df, "train")

print("Preprocessing eval data...")
eval_dataset = make_dataset(eval_df, "eval")

# ── Data collator ─────────────────────────────────────────────────────────────

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Pad input features
        input_features = [{"input_features": torch.tensor(f["input_features"])} for f in features]
        batch          = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Pad labels
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch   = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 so loss ignores it
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Remove bos token if present
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# ── Metrics ───────────────────────────────────────────────────────────────────

wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids  = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str  = processor.batch_decode(pred_ids,  skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": round(wer, 4)}

# ── Training arguments ────────────────────────────────────────────────────────
# Compatible with both old and new versions of transformers

train_kwargs = dict(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    fp16=True,
    predict_with_generate=True,
    learning_rate=1e-5,
    warmup_steps=100,
    max_steps=2000,
    eval_steps=200,
    save_steps=200,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    logging_steps=25,
    report_to="none",
    push_to_hub=False,
)

# Handle renamed argument between transformers versions
import transformers
version = tuple(int(x) for x in transformers.__version__.split(".")[:2])
if version >= (4, 41):
    train_kwargs["eval_strategy"] = "steps"
else:
    train_kwargs["evaluation_strategy"] = "steps"

training_args = Seq2SeqTrainingArguments(**train_kwargs)

# ── Trainer ───────────────────────────────────────────────────────────────────

# Handle renamed argument between transformers versions
trainer_kwargs = dict(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# tokenizer was renamed to processing_class in newer versions
try:
    trainer = Seq2SeqTrainer(**trainer_kwargs, processing_class=processor.feature_extractor)
except TypeError:
    trainer = Seq2SeqTrainer(**trainer_kwargs, tokenizer=processor.feature_extractor)

# ── Train ─────────────────────────────────────────────────────────────────────

print("\nStarting training...")
print(f"Model:  {MODEL_NAME}")
print(f"Device: {'CUDA (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU (slow!)'}")
if torch.cuda.is_available():
    print(f"VRAM:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print("")

trainer.train()

# ── Save ──────────────────────────────────────────────────────────────────────

print("\nSaving model...")
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}/")
print("\nTraining complete!")
