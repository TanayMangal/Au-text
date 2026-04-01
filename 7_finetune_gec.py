import os
os.environ["HF_HOME"] = r"C:\Users\mysti\OneDrive\Desktop\angfnweigow\hf_cache"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
"""
STEP 7: Fine-tune T5 for Grammar Error Correction (GEC)
=========================================================
Trains a seq2seq model to convert atypical/telegraphic speech
(Down syndrome, ASD, dysarthria) into grammatically correct English
while preserving the speaker's intended meaning.

Architecture choice:
  We use google/flan-t5-base — it's already instruction-tuned, handles
  GEC with a task prefix naturally, and fits in 8GB VRAM comfortably.
  It outperforms the prithivida GEC model (Step 5) on domain-specific
  atypical speech because we're fine-tuning it on YOUR data.

Two-stage training:
  Stage 1 — Warm up on JFLEG (standard GEC benchmark data, downloaded
             automatically from HuggingFace). This gives the model general
             GEC ability before seeing atypical speech.
  Stage 2 — Fine-tune on your TalkBank pairs (gec_pairs.csv).
             Echo/recast pairs are weighted 3x (higher quality).

Requirements:
    pip install transformers datasets accelerate sentencepiece evaluate jiwer torch
    pip install huggingface_hub  # for JFLEG download

Usage:
    python 7_finetune_gec.py
"""

import os
import torch
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
import evaluate

# ── Config ─────────────────────────────────────────────────────────────────────

MODEL_NAME     = "google/flan-t5-base"    # ~250MB, fits in 8GB VRAM fine
CSV_PATH       = "gec_pairs.csv"
OUTPUT_DIR     = "./gec-model"
TRAIN_SPLIT    = 0.90
SEED           = 42

# Task prefix — T5/Flan-T5 uses these to route task
TASK_PREFIX    = "Fix grammar: "

# Training hyperparameters (tuned for RTX 4060 8GB)
BATCH_SIZE          = 8
GRAD_ACCUM          = 4          # effective batch = 32
LEARNING_RATE       = 3e-4
WARMUP_STEPS        = 200
MAX_TRAIN_STEPS     = 1000       # increase to 5000 if you have more pairs
EVAL_STEPS          = 200
MAX_SOURCE_LENGTH   = 128
MAX_TARGET_LENGTH   = 128

# ── Load tokenizer & model ─────────────────────────────────────────────────────

print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ── Load data ──────────────────────────────────────────────────────────────────

print(f"\nLoading {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=['source', 'target'])
df = df[df['source'].str.strip() != '']
df = df[df['target'].str.strip() != '']
# Drop pairs where source == target (no correction needed)
df = df[df['source'].str.lower().str.strip() != df['target'].str.lower().str.strip()]

print(f"Total pairs loaded: {len(df)}")
print(df['pair_type'].value_counts().to_string())

# Upsample echo/recast pairs 3x (they are higher quality)
echo_df  = df[df['pair_type'] == 'echo_recast']
synth_df = df[df['pair_type'] != 'echo_recast']
df_weighted = pd.concat([echo_df, echo_df, echo_df, synth_df], ignore_index=True)
df_weighted = df_weighted.sample(frac=1, random_state=SEED).reset_index(drop=True)
print(f"After upsampling echo pairs (3x): {len(df_weighted)} total rows")

# ── Optionally add JFLEG for Stage 1 warmup ───────────────────────────────────

def load_jfleg():
    """Load JFLEG GEC benchmark from HuggingFace. Returns a DataFrame."""
    try:
        print("  Downloading JFLEG dataset from HuggingFace...")
        jfleg = load_dataset("jhu-clsp/jfleg", trust_remote_code=True)
        rows = []
        for split in ['validation', 'test']:
            if split not in jfleg:
                continue
            for item in jfleg[split]:
                src = item['sentence'].strip()
                for correction in item['corrections']:
                    tgt = correction.strip()
                    if tgt and tgt.lower() != src.lower():
                        rows.append({'source': src, 'target': tgt,
                                     'pair_type': 'jfleg', 'corpus': 'JFLEG'})
        jfleg_df = pd.DataFrame(rows).drop_duplicates(subset=['source', 'target'])
        print(f"  JFLEG pairs loaded: {len(jfleg_df)}")
        return jfleg_df
    except Exception as e:
        print(f"  Could not load JFLEG ({e}). Continuing with TalkBank data only.")
        return pd.DataFrame()

jfleg_df = load_jfleg()
if not jfleg_df.empty:
    # Add JFLEG to the training pool — it gives general GEC ability
    df_weighted = pd.concat([df_weighted, jfleg_df], ignore_index=True)
    df_weighted = df_weighted.sample(frac=1, random_state=SEED).reset_index(drop=True)
    print(f"Combined dataset size: {len(df_weighted)}")

# ── Train / eval split ─────────────────────────────────────────────────────────

n          = len(df_weighted)
split_idx  = int(n * TRAIN_SPLIT)
train_df   = df_weighted.iloc[:split_idx].reset_index(drop=True)
eval_df    = df_weighted.iloc[split_idx:].reset_index(drop=True)

# Make sure eval has echo pairs for meaningful evaluation
echo_eval = df_weighted[df_weighted['pair_type'] == 'echo_recast'].tail(min(100, len(echo_df) // 5))
eval_df   = pd.concat([eval_df, echo_eval]).drop_duplicates().reset_index(drop=True)

print(f"\nTrain samples: {len(train_df)} | Eval samples: {len(eval_df)}")

# ── Tokenisation ───────────────────────────────────────────────────────────────

def preprocess(examples):
    """Tokenise source and target sequences."""
    inputs = [TASK_PREFIX + s for s in examples['source']]
    targets = examples['target']

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_SOURCE_LENGTH,
        truncation=True,
        padding=False,
    )
    labels = tokenizer(
        text_target=targets,
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding=False,
    )
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

print("\nTokenising datasets...")
train_dataset = Dataset.from_pandas(train_df[['source', 'target']]).map(
    preprocess, batched=True, remove_columns=['source', 'target']
)
eval_dataset = Dataset.from_pandas(eval_df[['source', 'target']]).map(
    preprocess, batched=True, remove_columns=['source', 'target']
)
print(f"Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")

# ── Data collator ──────────────────────────────────────────────────────────────

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=-100,
    pad_to_multiple_of=8,
)

# ── Metrics ────────────────────────────────────────────────────────────────────

# We track BLEU (standard for GEC generation) and a simple exact-match rate
bleu_metric = evaluate.load("sacrebleu")

def compute_metrics(pred):
    pred_ids  = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 padding
    label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)

    # Clip pred logits if needed (Seq2Seq sometimes returns logit arrays)
    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]
    pred_ids = np.clip(pred_ids, 0, tokenizer.vocab_size - 1)

    pred_str  = tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Strip task prefix if model repeats it
    pred_str = [p.replace(TASK_PREFIX, '').strip() for p in pred_str]

    # BLEU
    bleu = bleu_metric.compute(
        predictions=pred_str,
        references=[[l] for l in label_str]
    )

    # Exact match
    exact = sum(p.lower().strip() == l.lower().strip()
                for p, l in zip(pred_str, label_str)) / max(len(pred_str), 1)

    # Print a few examples for inspection
    for i in range(min(3, len(pred_str))):
        print(f"  SRC: (see training data) | TGT: {label_str[i]} | PRED: {pred_str[i]}")

    return {
        "bleu":        round(bleu['score'], 2),
        "exact_match": round(exact, 4),
    }

# ── Training arguments ─────────────────────────────────────────────────────────

import transformers
version = tuple(int(x) for x in transformers.__version__.split(".")[:2])
eval_strategy_key = "eval_strategy" if version >= (4, 41) else "evaluation_strategy"

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    gradient_checkpointing=True,
    fp16=False,
    predict_with_generate=True,
    generation_max_length=MAX_TARGET_LENGTH,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    max_steps=MAX_TRAIN_STEPS,
    **{eval_strategy_key: "steps"},
    eval_steps=EVAL_STEPS,
    save_steps=EVAL_STEPS,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    greater_is_better=True,
    logging_steps=50,
    report_to="none",
    push_to_hub=False,
    dataloader_num_workers=0,
)

# ── Trainer ────────────────────────────────────────────────────────────────────

try:
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )
except TypeError:
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

# ── Train ──────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("Starting GEC fine-tuning")
print(f"  Base model:  {MODEL_NAME}")
print(f"  Train pairs: {len(train_dataset)}")
print(f"  Max steps:   {MAX_TRAIN_STEPS}")
device_str = f"CUDA ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "CPU"
print(f"  Device:      {device_str}")
if torch.cuda.is_available():
    print(f"  VRAM:        {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print("="*60 + "\n")

trainer.train()

# ── Save ───────────────────────────────────────────────────────────────────────

print("\nSaving model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"GEC model saved to {OUTPUT_DIR}/")
print("\nDone! Run 8_correct_grammar.py to use the model.")
