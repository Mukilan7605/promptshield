"""
PromptShield — Model Training Pipeline
=======================================

Fine-tunes distilbert-base-uncased for binary prompt-injection classification.

Design principles
-----------------
- Fully reproducible: fixed seeds on Python, NumPy, PyTorch, and Transformers.
- Never touches the test set during training or model selection.
- Validation F1-macro drives early stopping and checkpoint selection.
- Class weights handle the ~60/40 label imbalance.
- CPU-safe: no CUDA assumptions; fp16 auto-disabled on CPU.
- All hyperparameters live in ml/config/training.yaml.

Usage
-----
    python -m ml.train                         # default config
    python -m ml.train --config path/to.yaml   # custom config
    python -m ml.train --epochs 3              # CLI override
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from ml.evaluate import compute_binary_metrics, run_evaluation, print_metrics_table

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("promptshield.train")

_DEFAULT_CONFIG = Path(__file__).parent / "config" / "training.yaml"
DEFAULT_TRAINING_CONFIG_PATH = _DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    base_model: str = "distilbert-base-uncased"
    num_labels: int = 2
    max_seq_length: int = 256
    cache_dir: str = "./models/cache"


@dataclass
class PathsConfig:
    processed_dir: str = "datasets/processed"
    train_file: str = "train.parquet"
    validation_file: str = "validation.parquet"
    test_file: str = "test.parquet"
    output_dir: str = "models/fine_tuned/promptshield_v1"
    training_log: str = "models/fine_tuned/promptshield_v1/training_log.json"
    eval_report: str = "models/fine_tuned/promptshield_v1/eval_report.json"


@dataclass
class TrainingConfig:
    seed: int = 42
    num_epochs: int = 5
    train_batch_size: int = 32
    eval_batch_size: int = 64
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    use_class_weights: bool = True
    early_stopping_patience: int = 3
    model_selection_metric: str = "f1_macro"
    save_steps: int = 0
    logging_steps: int = 50
    fp16: bool = False


@dataclass
class TrainingFullConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    id2label: dict[int, str] = field(default_factory=lambda: {0: "benign", 1: "injection"})
    label2id: dict[str, int] = field(default_factory=lambda: {"benign": 0, "injection": 1})


def load_training_config(config_path: Path) -> TrainingFullConfig:
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = TrainingFullConfig()
    if "model" in raw:
        for k, v in raw["model"].items():
            if hasattr(cfg.model, k):
                setattr(cfg.model, k, v)
    if "paths" in raw:
        for k, v in raw["paths"].items():
            if hasattr(cfg.paths, k):
                setattr(cfg.paths, k, v)
    if "training" in raw:
        for k, v in raw["training"].items():
            if hasattr(cfg.training, k):
                setattr(cfg.training, k, v)
    if "labels" in raw:
        cfg.id2label = {int(k): v for k, v in raw["labels"]["id2label"].items()}
        cfg.label2id = raw["labels"]["label2id"]
    return cfg


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Makes CUDA ops deterministic (may slow training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    log.info(f"Random seed set to {seed}")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PromptDataset(Dataset):
    """PyTorch Dataset wrapping a tokenised DataFrame."""

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: Any,
        max_length: int,
    ) -> None:
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


# ---------------------------------------------------------------------------
# Class weight computation
# ---------------------------------------------------------------------------

def compute_class_weights(labels: list[int], num_classes: int = 2) -> torch.Tensor:
    """
    Compute inverse-frequency class weights.
    weight[c] = total_samples / (num_classes * count[c])
    """
    labels_arr = np.array(labels)
    weights = []
    for c in range(num_classes):
        count = (labels_arr == c).sum()
        w = len(labels_arr) / (num_classes * max(count, 1))
        weights.append(w)
    log.info(f"Class weights: benign={weights[0]:.4f}  injection={weights[1]:.4f}")
    return torch.tensor(weights, dtype=torch.float)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    criterion: nn.Module,
    device: torch.device,
    cfg: TrainingConfig,
    epoch: int,
    total_steps: int,
) -> tuple[float, int]:
    """Run one training epoch. Returns (avg_loss, steps_done)."""
    model.train()
    total_loss = 0.0
    steps = 0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = criterion(logits, labels)

        if cfg.gradient_accumulation_steps > 1:
            loss = loss / cfg.gradient_accumulation_steps

        loss.backward()
        total_loss += loss.item() * cfg.gradient_accumulation_steps

        if (step + 1) % cfg.gradient_accumulation_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            steps += 1

        global_step = (epoch - 1) * len(loader) + step + 1
        if cfg.logging_steps > 0 and global_step % cfg.logging_steps == 0:
            log.info(
                f"  Epoch {epoch}  step {global_step}/{total_steps}"
                f"  loss={loss.item() * cfg.gradient_accumulation_steps:.4f}"
            )

    avg_loss = total_loss / len(loader)
    return avg_loss, steps


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run evaluation on a DataLoader.
    Returns (avg_loss, all_labels, all_preds, all_probs).
    """
    model.eval()
    total_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = criterion(logits, labels)
        total_loss += loss.item()

        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())  # prob of injection

    avg_loss = total_loss / len(loader)
    return (
        avg_loss,
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
    )


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def run_training(config_path: Path = _DEFAULT_CONFIG, **overrides: Any) -> dict[str, Any]:
    """
    Execute the full training pipeline.

    Parameters
    ----------
    config_path : Path to the YAML training config.
    **overrides : Key-value pairs that override config fields.
                  Supported: num_epochs, train_batch_size, learning_rate, seed.

    Returns
    -------
    A dict containing best validation metrics, test metrics, and timing info.
    """
    cfg = load_training_config(config_path)

    # Apply CLI / programmatic overrides
    for k, v in overrides.items():
        if hasattr(cfg.training, k):
            setattr(cfg.training, k, v)
            log.info(f"Config override: training.{k} = {v}")

    # ── Paths ──────────────────────────────────────────────────────────────
    processed_dir = Path(cfg.paths.processed_dir)
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Reproducibility ────────────────────────────────────────────────────
    set_seed(cfg.training.seed)

    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = cfg.training.fp16 and device.type == "cuda"
    log.info(f"Device: {device}  |  fp16: {use_fp16}")

    # ── Load data ─────────────────────────────────────────────────────────
    log.info("Loading processed datasets …")
    train_df = pd.read_parquet(processed_dir / cfg.paths.train_file, engine="fastparquet")
    val_df   = pd.read_parquet(processed_dir / cfg.paths.validation_file, engine="fastparquet")
    test_df  = pd.read_parquet(processed_dir / cfg.paths.test_file, engine="fastparquet")

    log.info(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")
    log.info(f"Train injection rate: {train_df['label'].mean():.1%}")

    train_texts  = train_df["text"].tolist()
    train_labels = train_df["label"].tolist()
    val_texts    = val_df["text"].tolist()
    val_labels   = val_df["label"].tolist()
    test_texts   = test_df["text"].tolist()
    test_labels  = test_df["label"].tolist()

    # ── Tokenizer ─────────────────────────────────────────────────────────
    log.info(f"Loading tokenizer: {cfg.model.base_model} …")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.base_model,
        cache_dir=cfg.model.cache_dir,
    )

    # ── Datasets & loaders ────────────────────────────────────────────────
    log.info("Tokenising splits …")
    train_dataset = PromptDataset(train_texts, train_labels, tokenizer, cfg.model.max_seq_length)
    val_dataset   = PromptDataset(val_texts,   val_labels,   tokenizer, cfg.model.max_seq_length)
    test_dataset  = PromptDataset(test_texts,  test_labels,  tokenizer, cfg.model.max_seq_length)

    train_loader = DataLoader(train_dataset, batch_size=cfg.training.train_batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=cfg.training.eval_batch_size,  shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=cfg.training.eval_batch_size,  shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────
    log.info(f"Loading model: {cfg.model.base_model} …")
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.base_model,
        num_labels=cfg.model.num_labels,
        id2label=cfg.id2label,
        label2id=cfg.label2id,
        cache_dir=cfg.model.cache_dir,
    )
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model parameters: {n_params:,}  ({n_params/1e6:.1f}M)")

    # ── Loss with class weights ────────────────────────────────────────────
    if cfg.training.use_class_weights:
        class_weights = compute_class_weights(train_labels).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # ── Optimiser & scheduler ─────────────────────────────────────────────
    no_decay = ["bias", "LayerNorm.weight"]
    param_groups = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": cfg.training.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=cfg.training.learning_rate)

    total_steps = len(train_loader) * cfg.training.num_epochs // cfg.training.gradient_accumulation_steps
    warmup_steps = int(total_steps * cfg.training.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    log.info(
        f"Training config: epochs={cfg.training.num_epochs}  "
        f"batch={cfg.training.train_batch_size}  lr={cfg.training.learning_rate}  "
        f"total_steps={total_steps}  warmup={warmup_steps}"
    )

    # ── Training loop ──────────────────────────────────────────────────────
    training_log: list[dict[str, Any]] = []
    best_val_metric = -1.0
    best_epoch = 0
    no_improve_count = 0
    best_model_path = output_dir / "best_model"
    best_model_path.mkdir(exist_ok=True)

    wall_start = time.time()

    for epoch in range(1, cfg.training.num_epochs + 1):
        epoch_start = time.time()
        log.info(f"\n{'='*60}")
        log.info(f"Epoch {epoch}/{cfg.training.num_epochs}")

        # Train
        train_loss, _ = train_epoch(
            model, train_loader, optimizer, scheduler, criterion,
            device, cfg.training, epoch, total_steps,
        )

        # Validate
        val_loss, val_labels_arr, val_preds, val_probs = evaluate_epoch(
            model, val_loader, criterion, device
        )
        val_metrics = compute_binary_metrics(val_labels_arr, val_preds, val_probs, prefix="val_")
        val_metric = val_metrics[f"val_{cfg.training.model_selection_metric}"]

        epoch_time = time.time() - epoch_start
        log.info(
            f"Epoch {epoch} done in {epoch_time:.0f}s  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_f1_macro={val_metrics['val_f1_macro']:.4f}  "
            f"val_acc={val_metrics['val_accuracy']:.4f}"
        )

        # Log entry
        epoch_log = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "epoch_time_s": round(epoch_time, 1),
            **{k: round(v, 6) if isinstance(v, float) else v
               for k, v in val_metrics.items()
               if "confusion" not in k and k not in ("val_tn","val_fp","val_fn","val_tp")},
        }
        training_log.append(epoch_log)

        # Model selection
        if val_metric > best_val_metric:
            best_val_metric = val_metric
            best_epoch = epoch
            no_improve_count = 0
            log.info(f"  >> New best {cfg.training.model_selection_metric}={val_metric:.4f} — saving checkpoint")
            model.save_pretrained(str(best_model_path))
            tokenizer.save_pretrained(str(best_model_path))
            # Save training config for reproducibility
            with open(best_model_path / "training_config.json", "w") as f:
                json.dump({
                    "base_model": cfg.model.base_model,
                    "max_seq_length": cfg.model.max_seq_length,
                    "seed": cfg.training.seed,
                    "num_epochs": cfg.training.num_epochs,
                    "train_batch_size": cfg.training.train_batch_size,
                    "learning_rate": cfg.training.learning_rate,
                    "weight_decay": cfg.training.weight_decay,
                    "warmup_ratio": cfg.training.warmup_ratio,
                    "use_class_weights": cfg.training.use_class_weights,
                    "best_epoch": epoch,
                    "best_val_f1_macro": round(val_metric, 6),
                }, f, indent=2)
        else:
            no_improve_count += 1
            log.info(f"  No improvement for {no_improve_count} epoch(s)")

        # Early stopping
        if no_improve_count >= cfg.training.early_stopping_patience:
            log.info(f"Early stopping triggered at epoch {epoch} (patience={cfg.training.early_stopping_patience})")
            break

    total_train_time = time.time() - wall_start
    log.info(f"\nTraining complete in {total_train_time:.0f}s  Best epoch: {best_epoch}  Best val_{cfg.training.model_selection_metric}: {best_val_metric:.4f}")

    # Save training log
    log_path = Path(cfg.paths.training_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)
    log.info(f"Training log saved → {log_path}")

    # ── Final evaluation on TEST SET (ONLY here, never during training) ────
    log.info("\n" + "="*60)
    log.info("Loading best checkpoint for final test evaluation …")
    best_model = AutoModelForSequenceClassification.from_pretrained(str(best_model_path))
    best_model = best_model.to(device)
    best_tokenizer = AutoTokenizer.from_pretrained(str(best_model_path))

    # Re-tokenise with best tokenizer to be safe
    test_dataset_final = PromptDataset(test_texts, test_labels, best_tokenizer, cfg.model.max_seq_length)
    test_loader_final = DataLoader(test_dataset_final, batch_size=cfg.training.eval_batch_size, shuffle=False)

    _, test_labels_arr, test_preds, test_probs = evaluate_epoch(
        best_model, test_loader_final, criterion, device
    )

    # Attach predictions to test_df for per-group breakdown
    test_df = test_df.copy()
    test_df["pred_label"] = test_preds
    test_df["pred_prob"]  = test_probs

    # Also attach best validation predictions for the report
    val_dataset_final = PromptDataset(val_texts, val_labels, best_tokenizer, cfg.model.max_seq_length)
    val_loader_final = DataLoader(val_dataset_final, batch_size=cfg.training.eval_batch_size, shuffle=False)
    _, _, val_preds_best, val_probs_best = evaluate_epoch(
        best_model, val_loader_final, criterion, device
    )
    val_df = val_df.copy()
    val_df["pred_label"] = val_preds_best
    val_df["pred_prob"]  = val_probs_best

    log.info("Best validation metrics:")
    val_report = run_evaluation(val_df, split_name="validation")

    log.info("Final test metrics:")
    test_report = run_evaluation(test_df, split_name="test")

    # ── Model size on disk ────────────────────────────────────────────────
    model_size_mb = sum(
        p.stat().st_size for p in best_model_path.rglob("*") if p.is_file()
    ) / 1e6

    # ── Save full eval report ─────────────────────────────────────────────
    eval_report = {
        "pipeline_version": "1.0.0",
        "base_model": cfg.model.base_model,
        "max_seq_length": cfg.model.max_seq_length,
        "seed": cfg.training.seed,
        "device": str(device),
        "total_train_time_s": round(total_train_time, 1),
        "best_epoch": best_epoch,
        "epochs_trained": len(training_log),
        "model_size_mb": round(model_size_mb, 2),
        "model_path": str(best_model_path),
        "training_log": training_log,
        "best_val_metrics": val_report,
        "test_metrics": test_report,
        "config": {
            "num_epochs": cfg.training.num_epochs,
            "train_batch_size": cfg.training.train_batch_size,
            "eval_batch_size": cfg.training.eval_batch_size,
            "learning_rate": cfg.training.learning_rate,
            "weight_decay": cfg.training.weight_decay,
            "warmup_ratio": cfg.training.warmup_ratio,
            "use_class_weights": cfg.training.use_class_weights,
            "early_stopping_patience": cfg.training.early_stopping_patience,
            "model_selection_metric": cfg.training.model_selection_metric,
        },
    }

    report_path = Path(cfg.paths.eval_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(eval_report, f, indent=2, default=str)
    log.info(f"Eval report saved → {report_path}")

    # ── Console summary ───────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  TRAINING COMPLETE — SUMMARY")
    print("="*60)
    print(f"  Device:            {device}")
    print(f"  Total train time:  {total_train_time:.0f}s  ({total_train_time/60:.1f} min)")
    print(f"  Best epoch:        {best_epoch} / {len(training_log)}")
    print(f"  Model size:        {model_size_mb:.1f} MB")
    print(f"  Model saved to:    {best_model_path}")
    print()
    print("  BEST VALIDATION:")
    vm = val_report["overall"]
    print(f"    Accuracy:        {vm['accuracy']:.4f}")
    print(f"    F1 (binary):     {vm['f1']:.4f}")
    print(f"    F1 (macro):      {vm['f1_macro']:.4f}")
    print(f"    ROC-AUC:         {vm.get('roc_auc', 'N/A')}")
    print()
    print("  FINAL TEST:")
    tm = test_report["overall"]
    print(f"    Accuracy:        {tm['accuracy']:.4f}")
    print(f"    Precision:       {tm['precision']:.4f}")
    print(f"    Recall:          {tm['recall']:.4f}")
    print(f"    F1 (binary):     {tm['f1']:.4f}")
    print(f"    F1 (macro):      {tm['f1_macro']:.4f}")
    print(f"    ROC-AUC:         {tm.get('roc_auc', 'N/A')}")
    cm = tm.get("confusion_matrix", [])
    if cm:
        print(f"\n  Confusion Matrix:")
        print(f"                   Pred Benign   Pred Injection")
        print(f"    True Benign       {cm[0][0]:>6}          {cm[0][1]:>6}")
        print(f"    True Injection    {cm[1][0]:>6}          {cm[1][1]:>6}")
    print("="*60 + "\n")

    return eval_report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PromptShield Model Training")
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument("--epochs", type=int, default=None, help="Override num_epochs")
    parser.add_argument("--batch-size", type=int, default=None, dest="train_batch_size")
    parser.add_argument("--lr", type=float, default=None, dest="learning_rate")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    overrides = {k: v for k, v in vars(args).items()
                 if k not in ("config",) and v is not None}
    run_training(config_path=args.config, **overrides)
