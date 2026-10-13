"""
PromptShield — Evaluation Module (pure-NumPy implementation)
=============================================================

Computes all classification metrics using only NumPy — no scikit-learn,
no scipy. This is intentional: Application Control policies on restricted
machines block scipy's compiled extensions (.pyd DLLs).

Metrics computed
----------------
- Accuracy
- Precision (binary, macro)
- Recall    (binary, macro)
- F1        (binary, macro, weighted)
- ROC-AUC   (trapezoidal rule on manually sorted thresholds)
- Confusion matrix (TP, FP, FN, TN)
- Per-category and per-category-group breakdowns
- Hard-negative subset evaluation

Usage (standalone):
    python -m ml.evaluate  \\
        --predictions models/fine_tuned/promptshield_v1/test_predictions.json \\
        --report      models/fine_tuned/promptshield_v1/eval_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger("promptshield.evaluate")


# ---------------------------------------------------------------------------
# Pure-NumPy metric primitives
# ---------------------------------------------------------------------------

def _confusion_elements(
    y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1
) -> tuple[int, int, int, int]:
    """Return (TP, FP, FN, TN) for a binary classification."""
    tp = int(((y_pred == pos_label) & (y_true == pos_label)).sum())
    fp = int(((y_pred == pos_label) & (y_true != pos_label)).sum())
    fn = int(((y_pred != pos_label) & (y_true == pos_label)).sum())
    tn = int(((y_pred != pos_label) & (y_true != pos_label)).sum())
    return tp, fp, fn, tn


def _precision(tp: int, fp: int) -> float:
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def _recall(tp: int, fn: int) -> float:
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def _roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute ROC-AUC using the trapezoidal rule.

    Equivalent to sklearn.metrics.roc_auc_score for binary classification.
    Returns 0.5 if only one class is present (degenerate case).
    """
    classes = np.unique(y_true)
    if len(classes) < 2:
        return 0.5

    # Sort by descending score
    desc_idx = np.argsort(-y_prob)
    y_true_sorted = y_true[desc_idx]
    y_prob_sorted = y_prob[desc_idx]

    n_pos = (y_true == 1).sum()
    n_neg = (y_true == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Accumulate TP and FP at each threshold
    tprs, fprs = [0.0], [0.0]
    tp = fp = 0

    # Group tied scores together
    prev_prob = None
    for i, (label, prob) in enumerate(zip(y_true_sorted, y_prob_sorted)):
        if prev_prob is not None and prob != prev_prob:
            tprs.append(tp / n_pos)
            fprs.append(fp / n_neg)
        if label == 1:
            tp += 1
        else:
            fp += 1
        prev_prob = prob

    tprs.append(tp / n_pos)
    fprs.append(fp / n_neg)

    # Trapezoidal integration (np.trapezoid is the NumPy 2.0+ API; np.trapz is removed)
    tprs_arr = np.array(tprs)
    fprs_arr = np.array(fprs)
    _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    auc = float(_trapz(tprs_arr, fprs_arr))
    # Ensure positive (can be negative if fprs not sorted ascending)
    return abs(auc)


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def compute_binary_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    y_prob: list[float] | np.ndarray | None = None,
    prefix: str = "",
) -> dict[str, Any]:
    """
    Compute a full suite of binary classification metrics using pure NumPy.

    Parameters
    ----------
    y_true : Ground-truth labels (0 or 1).
    y_pred : Predicted labels (0 or 1).
    y_prob : Predicted probability for the positive class (label=1).
             Required for ROC-AUC. Pass None to skip.
    prefix : Optional string prefix for all metric keys (e.g. "val_").

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, f1_macro, f1_weighted,
                    roc_auc (if y_prob provided), confusion_matrix, tn, fp, fn, tp.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)
    p = prefix

    # Per-class stats for macro/weighted averages
    classes = [0, 1]
    per_class: dict[int, dict[str, float]] = {}
    for c in classes:
        tp_c, fp_c, fn_c, tn_c = _confusion_elements(y_true, y_pred, pos_label=c)
        prec_c = _precision(tp_c, fp_c)
        rec_c  = _recall(tp_c, fn_c)
        f1_c   = _f1(prec_c, rec_c)
        per_class[c] = {"precision": prec_c, "recall": rec_c, "f1": f1_c,
                        "support": int((y_true == c).sum())}

    # Binary metrics (positive class = 1 = injection)
    tp, fp, fn, tn = _confusion_elements(y_true, y_pred, pos_label=1)
    prec = _precision(tp, fp)
    rec  = _recall(tp, fn)
    f1b  = _f1(prec, rec)

    # Macro averages (unweighted)
    f1_macro     = float(np.mean([per_class[c]["f1"]        for c in classes]))
    prec_macro   = float(np.mean([per_class[c]["precision"] for c in classes]))
    rec_macro    = float(np.mean([per_class[c]["recall"]    for c in classes]))

    # Weighted averages
    total = len(y_true)
    f1_weighted = float(
        sum(per_class[c]["f1"] * per_class[c]["support"] for c in classes) / max(total, 1)
    )

    metrics: dict[str, Any] = {
        f"{p}accuracy":    _accuracy(y_true, y_pred),
        f"{p}precision":   prec,
        f"{p}recall":      rec,
        f"{p}f1":          f1b,
        f"{p}f1_macro":    f1_macro,
        f"{p}f1_weighted": f1_weighted,
        f"{p}tp": tp, f"{p}fp": fp,
        f"{p}fn": fn, f"{p}tn": tn,
        f"{p}confusion_matrix": [[tn, fp], [fn, tp]],
    }

    if y_prob is not None:
        y_prob_arr = np.asarray(y_prob, dtype=np.float64)
        try:
            metrics[f"{p}roc_auc"] = _roc_auc(y_true, y_prob_arr)
        except Exception as e:
            log.warning(f"ROC-AUC computation failed: {e}")
            metrics[f"{p}roc_auc"] = None

    return metrics


# ---------------------------------------------------------------------------
# Per-group evaluation
# ---------------------------------------------------------------------------

def evaluate_by_group(
    df: pd.DataFrame,
    group_col: str,
    label_col: str = "label",
    pred_col: str = "pred_label",
    prob_col: str | None = "pred_prob",
) -> dict[str, dict[str, Any]]:
    """
    Compute binary metrics broken down by a grouping column.
    Groups with fewer than 5 samples are skipped.
    """
    results: dict[str, dict[str, Any]] = {}
    for group_val, group_df in df.groupby(group_col):
        n = len(group_df)
        if n < 5:
            log.debug(f"Skipping group '{group_val}' — only {n} samples")
            continue
        y_true = group_df[label_col].values.astype(np.int32)
        y_pred = group_df[pred_col].values.astype(np.int32)
        y_prob = group_df[prob_col].values if (prob_col and prob_col in group_df.columns) else None

        # Skip ROC-AUC if only one class
        if y_prob is not None and len(np.unique(y_true)) < 2:
            y_prob = None

        m = compute_binary_metrics(y_true, y_pred, y_prob)
        m["n_samples"] = n
        m["injection_rate"] = float(y_true.mean())
        results[str(group_val)] = m
    return results


# ---------------------------------------------------------------------------
# Hard-negative evaluation
# ---------------------------------------------------------------------------

def evaluate_hard_negatives(
    df: pd.DataFrame,
    label_col: str = "label",
    pred_col: str = "pred_label",
    prob_col: str | None = "pred_prob",
    tags_col: str = "tags",
) -> dict[str, Any] | None:
    """
    Evaluate on samples tagged 'hard_negative'. Returns None if none found.
    """
    if tags_col not in df.columns:
        return None

    def _has_hard_neg(tags: Any) -> bool:
        if isinstance(tags, list):
            return "hard_negative" in tags
        if isinstance(tags, str):
            return "hard_negative" in tags
        return False

    mask = df[tags_col].apply(_has_hard_neg)
    hn_df = df[mask]

    if len(hn_df) == 0:
        log.info("No hard-negative samples found in this split.")
        return None

    log.info(f"Evaluating {len(hn_df)} hard-negative samples …")
    y_true = hn_df[label_col].values.astype(np.int32)
    y_pred = hn_df[pred_col].values.astype(np.int32)
    y_prob = hn_df[prob_col].values if (prob_col and prob_col in hn_df.columns) else None

    if y_prob is not None and len(np.unique(y_true)) < 2:
        y_prob = None

    m = compute_binary_metrics(y_true, y_pred, y_prob)
    m["n_samples"] = len(hn_df)
    m["injection_rate"] = float(y_true.mean())
    return m


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def print_metrics_table(metrics: dict[str, Any], title: str = "Metrics") -> None:
    """Print a readable metrics summary to stdout."""
    line = "-" * 54
    print(f"\n{line}")
    print(f"  {title}")
    print(line)
    skip = {"confusion_matrix", "tp", "fp", "fn", "tn"}
    # find prefix
    for k, v in metrics.items():
        if isinstance(v, float) and "confusion" not in k and k.replace(k.lstrip("abcdefghijklmnopqrstuvwxyz_"), "") not in skip:
            print(f"  {k:<28} {v:.4f}")

    # Confusion matrix
    cm_key = next((k for k in metrics if "confusion_matrix" in k), None)
    pfx = cm_key.replace("confusion_matrix", "") if cm_key else ""
    tn_k, fp_k, fn_k, tp_k = f"{pfx}tn", f"{pfx}fp", f"{pfx}fn", f"{pfx}tp"
    if all(k in metrics for k in (tn_k, fp_k, fn_k, tp_k)):
        tn = metrics[tn_k]; fp = metrics[fp_k]
        fn = metrics[fn_k]; tp = metrics[tp_k]
        print(f"\n  Confusion Matrix:")
        print(f"                     Pred Benign   Pred Injection")
        print(f"  True Benign             {tn:>6}          {fp:>6}")
        print(f"  True Injection          {fn:>6}          {tp:>6}")
        print(line + "\n")


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------

def run_evaluation(
    df: pd.DataFrame,
    label_col: str = "label",
    pred_col: str = "pred_label",
    prob_col: str | None = "pred_prob",
    split_name: str = "test",
) -> dict[str, Any]:
    """
    Run the complete evaluation pipeline on a DataFrame that already
    contains prediction columns (pred_label, pred_prob).

    Returns a nested dict suitable for JSON serialisation.
    """
    report: dict[str, Any] = {"split": split_name}

    # -- Overall metrics ---------------------------------------------------
    y_true = df[label_col].values.astype(np.int32)
    y_pred = df[pred_col].values.astype(np.int32)
    y_prob = df[prob_col].values if (prob_col and prob_col in df.columns) else None

    overall = compute_binary_metrics(y_true, y_pred, y_prob)
    report["overall"] = overall
    print_metrics_table(overall, f"{split_name.upper()} - Overall Metrics")

    # -- Per-category breakdown --------------------------------------------
    if "category" in df.columns:
        report["by_category"] = evaluate_by_group(df, "category", label_col, pred_col, prob_col)

    # -- Per-category-group breakdown --------------------------------------
    if "category_group" in df.columns:
        report["by_category_group"] = evaluate_by_group(
            df, "category_group", label_col, pred_col, prob_col
        )
        print("  Per-Category-Group F1 / Accuracy:")
        for grp, m in sorted(report["by_category_group"].items()):
            print(f"    {grp:<22} n={m['n_samples']:>4}  "
                  f"f1={m.get('f1', 0):.3f}  "
                  f"acc={m.get('accuracy', 0):.3f}")
        print()

    # -- Hard-negative evaluation ------------------------------------------
    hn = evaluate_hard_negatives(df, label_col, pred_col, prob_col)
    if hn:
        report["hard_negatives"] = hn
        print(f"  Hard Negatives - n={hn['n_samples']}  "
              f"f1={hn.get('f1', 0):.3f}  "
              f"acc={hn.get('accuracy', 0):.3f}\n")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PromptShield Evaluation (pure-NumPy)")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--report",      type=Path, required=True)
    parser.add_argument("--split",       default="test")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    preds_df = pd.read_json(args.predictions)
    report = run_evaluation(preds_df, split_name=args.split)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Evaluation report saved -> {args.report}")
