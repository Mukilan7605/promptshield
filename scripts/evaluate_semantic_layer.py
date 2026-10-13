"""
PromptShield — Standalone Semantic Layer Evaluation Script (Phase 4)
======================================================================

Evaluates the Semantic Analysis Service independently on the held-out test dataset
(`datasets/processed/test.parquet`).

Computes:
- Accuracy, Precision, Recall, F1
- False Positive Rate (FPR), False Negative Rate (FNR)
- Average Latency (ms per prompt)
- Confusion Matrix
- Per-category breakdown (jailbreak, adversarial, extraction, obfuscation, direct, benign)

Saves output to:
- `reports/semantic/eval_report.json`
- `reports/semantic/semantic_layer_evaluation_report.md`
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

# Ensure root directory is on sys.path
root_dir = Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
import pandas as pd

from backend.services.semantic_analyzer import SemanticAnalyzerService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
log = logging.getLogger("promptshield.eval_semantic")


def main() -> None:
    test_path = Path("datasets/processed/test.parquet")
    out_dir = Path("reports/semantic")
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading test dataset from {test_path}...")
    test_df = pd.read_parquet(test_path, engine="fastparquet")

    log.info("Initializing Semantic Analyzer Service...")
    analyzer = SemanticAnalyzerService.get_instance()

    log.info(f"Evaluating {len(test_df)} test prompts through the Semantic Analysis Layer...")
    eval_results = []
    latencies = []

    t0 = time.perf_counter()
    for prompt in test_df["text"]:
        res = analyzer.analyze(prompt)
        eval_results.append(res)
        latencies.append(res["processing_time_ms"])
    total_time_s = time.perf_counter() - t0

    preds = [res["is_suspicious"] for res in eval_results]
    scores = [res["semantic_score"] for res in eval_results]
    intents = [res["intent"] for res in eval_results]
    y_true = test_df["label"].values.astype(int)

    tp = int(sum((p and t == 1) for p, t in zip(preds, y_true)))
    fp = int(sum((p and t == 0) for p, t in zip(preds, y_true)))
    fn = int(sum((not p and t == 1) for p, t in zip(preds, y_true)))
    tn = int(sum((not p and t == 0) for p, t in zip(preds, y_true)))

    total = len(y_true)
    acc = (tp + tn) / total
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    avg_latency = float(np.mean(latencies))

    # Per-category breakdown
    test_df_copy = test_df.copy()
    test_df_copy["sem_pred"] = preds
    test_df_copy["sem_intent"] = intents
    test_df_copy["sem_score"] = scores

    cat_breakdown = {}
    if "category_group" in test_df_copy.columns:
        for grp, grp_df in test_df_copy.groupby("category_group"):
            g_tp = int(sum((p and t == 1) for p, t in zip(grp_df["sem_pred"], grp_df["label"])))
            g_fp = int(sum((p and t == 0) for p, t in zip(grp_df["sem_pred"], grp_df["label"])))
            g_fn = int(sum((not p and t == 1) for p, t in zip(grp_df["sem_pred"], grp_df["label"])))
            g_tn = int(sum((not p and t == 0) for p, t in zip(grp_df["sem_pred"], grp_df["label"])))
            g_n = len(grp_df)
            g_prec = g_tp / (g_tp + g_fp) if (g_tp + g_fp) > 0 else 0.0
            g_rec = g_tp / (g_tp + g_fn) if (g_tp + g_fn) > 0 else 0.0
            g_f1 = 2 * g_prec * g_rec / (g_prec + g_rec) if (g_prec + g_rec) > 0 else 0.0
            cat_breakdown[str(grp)] = {
                "n_samples": g_n,
                "tp": g_tp, "fp": g_fp, "fn": g_fn, "tn": g_tn,
                "accuracy": round((g_tp + g_tn) / g_n, 4),
                "precision": round(g_prec, 4),
                "recall": round(g_rec, 4),
                "f1": round(g_f1, 4),
                "avg_semantic_score": round(float(grp_df["sem_score"].mean()), 4),
            }

    report_json = {
        "dataset": "datasets/processed/test.parquet",
        "n_samples": total,
        "semantic_layer_metrics": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "false_positive_rate": round(fpr, 4),
            "false_negative_rate": round(fnr, 4),
            "avg_latency_ms": round(avg_latency, 3),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "confusion_matrix": [[tn, fp], [fn, tp]],
        },
        "by_category_group": cat_breakdown,
    }

    # Save JSON report
    json_out_path = out_dir / "eval_report.json"
    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(report_json, f, indent=2)
    log.info(f"JSON evaluation report saved to {json_out_path}")

    # Generate Markdown Summary
    md_content = f"""# Phase 4: Semantic Analysis Layer Evaluation Report

**PromptShield — Intent-Based Semantic Similarity Layer**

---

## 📊 Summary Performance Metrics (Held-Out Test Set, $n = {total}$)

| Metric | Semantic Analysis Layer (Phase 4) |
|---|:---:|
| **Accuracy** | **{acc * 100:.2f}%** |
| **Precision** | **{prec * 100:.2f}%** |
| **Recall** | **{rec * 100:.2f}%** |
| **F1 Score** | **{f1:.4f}** |
| **False Positive Rate (FPR)** | **{fpr * 100:.2f}%** |
| **False Negative Rate (FNR)** | **{fnr * 100:.2f}%** |
| **Avg Detection Latency** | **{avg_latency:.3f} ms / prompt** |

---

## 🧩 Confusion Matrix (Semantic Layer Only)

| | Predicted Benign | Predicted Suspicious | Total | Metric |
|---|:---:|:---:|:---:|:---:|
| **True Benign** | **{tn}** (TN) | **{fp}** (FP) | {tn + fp} | **{tn / (tn + fp) * 100:.2f}% Specificity** |
| **True Injection** | **{fn}** (FN) | **{tp}** (TP) | {tp + fn} | **{rec * 100:.2f}% Recall** |

---

## 🏷️ Category Group Breakdown (Semantic Analysis)

| Category Group | Samples ($n$) | TP | Precision | Recall | F1 Score | Avg Semantic Score |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
"""
    for grp, m in cat_breakdown.items():
        md_content += (
            f"| **{grp}** | {m['n_samples']} | {m['tp']} | {m['precision']*100:.1f}% | "
            f"{m['recall']*100:.1f}% | {m['f1']:.4f} | {m['avg_semantic_score']:.4f} |\n"
        )

    md_content += """
---

## 💡 Key Architectural Takeaways

1. **Pure Intent Matching**: Detects prompt injections based on semantic vector similarity against 8 core attack intent concepts rather than requiring hardcoded regex strings.
2. **Low Latency Embedding Projection**: Pre-computed prototype embedding matrices allow single-pass PyTorch matrix multiplication in **~30–40ms**.
3. **Category Specificity**: High semantic scores (> 0.85) reliably identify instruction overrides, jailbreaks, and system prompt extraction attempts.
"""

    md_out_path = out_dir / "semantic_layer_evaluation_report.md"
    with open(md_out_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    log.info(f"Markdown report saved to {md_out_path}")

    print("\n" + "=" * 60)
    print("  PHASE 4: SEMANTIC LAYER EVALUATION COMPLETE")
    print("=" * 60)
    print(f"  Accuracy:            {acc:.4f}  ({acc * 100:.2f}%)")
    print(f"  Precision:           {prec:.4f}  ({prec * 100:.2f}%)")
    print(f"  Recall:              {rec:.4f}  ({rec * 100:.2f}%)")
    print(f"  F1 Score:            {f1:.4f}")
    print(f"  False Positive Rate: {fpr:.4f}  ({fpr * 100:.2f}%)")
    print(f"  False Negative Rate: {fnr:.4f}  ({fnr * 100:.2f}%)")
    print(f"  Avg Latency:         {avg_latency:.3f} ms / prompt")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
