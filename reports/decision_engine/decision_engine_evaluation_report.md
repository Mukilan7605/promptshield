# Phase 5: Decision Engine & Final Pipeline Evaluation Report

**PromptShield — 3-Layer Integrated Prompt Injection Defense Pipeline**

---

## 📊 Complete Pipeline Head-to-Head Comparison (Test Set, $n = 942$)

| Layer / Model | Accuracy | Precision | Recall | F1 Score | FPR | FNR | Avg Latency |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Rule-Only Layer** | 66.35% | 100.00% | 42.57% | 0.5972 | 0.00% | 57.43% | ~0.53 ms |
| **Semantic-Only Layer** | 95.86% | 96.38% | 96.56% | 0.9647 | 5.13% | 3.44% | ~64.4 ms |
| **ML-Only Classifier** | 96.18% | 97.43% | 96.01% | 0.9672 | 3.59% | 3.99% | ~45.0 ms |
| 🏆 **PromptShield 3-Layer** | **96.39%** | **95.44%** | **98.55%** | **0.9697** | **6.67%** | **1.45%** | **62.71 ms** |

---

## 🧩 Confusion Matrix (Full 3-Layer PromptShield Pipeline)

| | Predicted Benign (ALLOW) | Predicted Injection (WARN/SANITIZE/BLOCK) | Total | Metric |
|---|:---:|:---:|:---:|:---:|
| **True Benign** | **364** (TN) | **26** (FP) | 390 | **93.33% Specificity** |
| **True Injection** | **8** (FN) | **544** (TP) | 552 | **98.55% Recall** |

---

## ⚖️ Decision & Risk Distribution

### Decision Breakdown
- **ALLOW** ($0–24$ score): **372** prompts
- **WARN** ($25–49$ score): **36** prompts
- **SANITIZE** ($50–74$ score): **146** prompts
- **BLOCK** ($75–100$ score): **388** prompts

### Risk Level Breakdown
- **Low Risk**: **372** prompts
- **Medium Risk**: **36** prompts
- **High Risk**: **146** prompts
- **Critical Risk**: **388** prompts

---

## 🔒 Security Override Enforcements

1. **High-Confidence Rule Protection**: Any prompt matching a critical rule (`RULE-OVR-xxx`, `RULE-SYS-xxx`) with confidence $\ge 0.90$ is immediately forced to `BLOCK` (score $\ge 90$).
2. **Weak ML Defense**: If the ML transformer predicts a low confidence score, a strong critical rule match overrides it, preventing adversarial evasion.
3. **Multi-Layer Evidence Preservation**: Full evidence, rule IDs, matched prototypes, and intent scores are recorded for full auditability.
