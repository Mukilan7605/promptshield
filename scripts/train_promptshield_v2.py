"""
PromptShield — Step 4: Fine-Tune PromptShield v2 with Contextual Hard Negatives
================================================================================

Fine-tunes DistilBERT on original training data + contextual hard negatives.
Saves model output to `models/fine_tuned/promptshield_v2/best_model`.
Ensures zero data leakage from held-out evaluation set.
"""

from __future__ import annotations

import json
import logging
import random
import sys
import time
from pathlib import Path

root_dir = Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("promptshield.train_v2")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


class PromptDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer: AutoTokenizer, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


def main() -> None:
    output_dir = Path("models/fine_tuned/promptshield_v2/best_model")
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading base dataset (train.parquet & validation.parquet)...")
    train_df = pd.read_parquet("datasets/processed/train.parquet")
    val_df = pd.read_parquet("datasets/processed/validation.parquet")
    hard_neg_df = pd.read_parquet("datasets/processed/contextual_hard_negatives.parquet")

    # Split hard_neg_df into 70% train and 30% held-out eval
    eval_mask = np.random.rand(len(hard_neg_df)) < 0.3
    hard_eval = hard_neg_df[eval_mask].copy()
    hard_train = hard_neg_df[~eval_mask].copy()

    # Save held-out evaluation set
    eval_path = Path("datasets/processed/contextual_eval_heldout.parquet")
    hard_eval.to_parquet(eval_path)
    log.info(f"Saved {len(hard_eval)} held-out evaluation samples to {eval_path}")

    # Combine training data
    train_combined = pd.concat([
        train_df[["text", "label"]],
        hard_train[["text", "label"]],
    ], ignore_index=True).drop_duplicates(subset=["text"])

    log.info(f"Combined Training Set Size: {len(train_combined)} samples.")

    model_name = "distilbert-base-uncased"
    v1_path = Path("models/fine_tuned/promptshield_v1/best_model")
    if v1_path.exists():
        model_name = str(v1_path)
        log.info(f"Fine-tuning starting from existing checkpoint: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_dataset = PromptDataset(train_combined["text"].tolist(), train_combined["label"].tolist(), tokenizer)
    val_dataset = PromptDataset(val_df["text"].tolist(), val_df["label"].tolist(), tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    epochs = 3
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    criterion = nn.CrossEntropyLoss()

    log.info(f"Starting training for {epochs} epochs on {device}...")
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        t0 = time.perf_counter()

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                val_loss += loss.item()

                preds = torch.argmax(outputs.logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        elapsed = time.perf_counter() - t0

        log.info(f"Epoch {epoch+1}/{epochs} ({elapsed:.1f}s) | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            tokenizer.save_pretrained(output_dir)
            model.save_pretrained(output_dir)
            log.info(f"Saved best v2 model checkpoint to {output_dir}")

    log.info("PromptShield v2 training complete!")


if __name__ == "__main__":
    main()
