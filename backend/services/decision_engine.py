"""
PromptShield — Decision Engine Service (Phase 5)
=================================================

Final decision layer in the PromptShield multi-layer security pipeline.

Pipeline Order:
Prompt -> Rule Engine -> Semantic Analysis -> ML Classifier -> Decision Engine

Responsibilities:
-----------------
1. Combines outputs from Rule Engine (30%), Semantic Analysis (30%), and ML Classifier (40%).
2. Reads weights and decision thresholds from `backend.core.config` (NOT hardcoded).
3. Computes normalized risk score (0–100).
4. Maps risk score to decision:
   - 0..24   = ALLOW (low risk)
   - 25..49  = WARN (medium risk)
   - 50..74  = SANITIZE (high risk)
   - 75..100 = BLOCK (critical risk)
5. Enforces security override logic:
   - High-confidence critical rule triggers immediately force BLOCK decision (score >= 90).
   - Weak ML results never override a strong critical rule or semantic detection.
6. Preserves evidence & produces human-readable reasons from every layer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from backend.core.config import get_settings

log = logging.getLogger("promptshield.decision_engine")


# ---------------------------------------------------------------------------
# Decision Output Data Schema
# ---------------------------------------------------------------------------

@dataclass
class DecisionResult:
    final_score: int                 # 0 to 100
    risk_level: str                  # low, medium, high, critical
    decision: str                    # ALLOW, WARN, SANITIZE, BLOCK
    attack_category: str             # e.g. instruction_override, jailbreak, benign
    confidence: float                # 0.0 to 1.0
    layer_results: dict[str, Any]    # {"rule": ..., "semantic": ..., "ml": ...}
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_score": self.final_score,
            "risk_level": self.risk_level,
            "decision": self.decision,
            "attack_category": self.attack_category,
            "confidence": self.confidence,
            "layer_results": self.layer_results,
            "reasons": self.reasons,
        }


# ---------------------------------------------------------------------------
# Decision Engine Service
# ---------------------------------------------------------------------------

class DecisionEngine:
    """
    Deterministic & Configurable Decision Engine.
    Aggregates multi-layer signals, calculates risk scores, and applies
    security override rules.
    """

    def __init__(
        self,
        weight_rule: float | None = None,
        weight_semantic: float | None = None,
        weight_ml: float | None = None,
        enable_override: bool | None = None,
    ) -> None:
        cfg = get_settings()
        self.w_rule = weight_rule if weight_rule is not None else cfg.WEIGHT_RULE_ENGINE
        self.w_sem = weight_semantic if weight_semantic is not None else cfg.WEIGHT_SEMANTIC_ANALYSIS
        self.w_ml = weight_ml if weight_ml is not None else cfg.WEIGHT_ML_CLASSIFIER

        self.thresh_allow = cfg.THRESHOLD_ALLOW_MAX         # 24
        self.thresh_warn = cfg.THRESHOLD_WARN_MAX           # 49
        self.thresh_sanitize = cfg.THRESHOLD_SANITIZE_MAX   # 74
        self.thresh_block = cfg.THRESHOLD_BLOCK_MIN         # 75

        self.enable_override = enable_override if enable_override is not None else cfg.ENABLE_SECURITY_OVERRIDE

    def evaluate(
        self,
        rule_result: dict[str, Any],
        semantic_result: dict[str, Any],
        ml_result: dict[str, Any],
    ) -> DecisionResult:
        """
        Evaluate and combine outputs from Rule, Semantic, and ML layers.
        """
        reasons: list[str] = []

        # Extract layer confidences & trigger flags
        c_rule = float(rule_result.get("confidence", 0.0))
        t_rule = bool(rule_result.get("triggered", False))
        cat_rule = rule_result.get("attack_category") or rule_result.get("intent")
        sev_rule = rule_result.get("severity")

        c_sem = float(semantic_result.get("confidence", semantic_result.get("semantic_score", 0.0)))
        t_sem = bool(semantic_result.get("triggered", semantic_result.get("is_suspicious", False)))
        cat_sem = semantic_result.get("matched_intent") or semantic_result.get("intent")

        c_ml = float(ml_result.get("injection_prob", ml_result.get("confidence", 0.0)))
        t_ml = bool(ml_result.get("triggered", False))
        ml_avail = bool(ml_result.get("available", True))

        # ── 1. Calculate Weighted Raw Score ──────────────────────────────────
        active_w_rule = self.w_rule
        active_w_sem = self.w_sem
        active_w_ml = self.w_ml if ml_avail else 0.0

        w_sum = active_w_rule + active_w_sem + active_w_ml
        if w_sum > 0:
            norm_w_rule = active_w_rule / w_sum
            norm_w_sem = active_w_sem / w_sum
            norm_w_ml = active_w_ml / w_sum
        else:
            norm_w_rule, norm_w_sem, norm_w_ml = 0.333, 0.333, 0.334

        weighted_conf = (
            norm_w_rule * c_rule
            + norm_w_sem * c_sem
            + norm_w_ml * c_ml
        )

        final_score = int(round(weighted_conf * 100))
        override_applied = False

        # ── 2. Reasons Generation ─────────────────────────────────────────────
        if t_rule:
            top_rule = rule_result.get("top_rule_id") or "Pattern"
            cat_str = f" ({cat_rule})" if cat_rule else ""
            reasons.append(f"Rule engine detected injection pattern {top_rule}{cat_str} with confidence {c_rule:.2f}")

        if t_sem:
            matched_p = semantic_result.get("matched_prototype")
            proto_str = f" [prototype: '{matched_p[:50]}...']" if matched_p else ""
            reasons.append(f"Semantic layer detected {cat_sem} intent with similarity score {c_sem:.2f}{proto_str}")

        if ml_avail and t_ml:
            reasons.append(f"ML classifier predicted prompt injection with confidence {c_ml:.2f}")
        elif not ml_avail:
            reasons.append("ML classifier layer was unavailable — evaluated using Rule & Semantic layers")

        # ── 3. Security Override Logic ────────────────────────────────────────
        # A high-confidence critical rule match OR explicit malicious semantic attack MUST force BLOCK decision
        if self.enable_override:
            is_critical_rule = t_rule and (sev_rule == "critical" or c_rule >= 0.90)

            # Check explicit malicious intent vs legitimate task intent
            leg_intent_score = float(semantic_result.get("legitimate_task_intent_score", 0.0))
            intent_bk = semantic_result.get("intent_breakdown", {})
            explicit_attack_vector = max(
                intent_bk.get("system_extraction", 0.0),
                intent_bk.get("safety_bypass", 0.0),
                intent_bk.get("privilege_persona_manipulation", 0.0),
            ) >= 0.78

            # Critical semantic override applies when semantic risk >= 0.82 and not overridden by legitimate task intent
            is_critical_semantic = t_sem and c_sem >= 0.82 and (explicit_attack_vector or leg_intent_score < 0.60)

            if is_critical_rule or is_critical_semantic:
                override_applied = True
                final_score = max(final_score, 90)
                if is_critical_rule:
                    reasons.insert(0, "SECURITY OVERRIDE: Critical high-confidence rule match enforced BLOCK decision")
                elif is_critical_semantic:
                    reasons.insert(0, "SECURITY OVERRIDE: High-confidence semantic intent match enforced BLOCK decision")

        # Guarantee benign reasons if score is very low
        if final_score <= self.thresh_allow and not reasons:
            reasons.append("All detection layers confirmed benign prompt (low risk)")

        final_score = max(0, min(100, final_score))
        final_conf = round(final_score / 100.0, 4)

        # ── 4. Map Decision & Risk Level ──────────────────────────────────────
        if final_score <= self.thresh_allow:          # 0..24
            decision = "ALLOW"
            risk_level = "low"
        elif final_score <= self.thresh_warn:         # 25..49
            decision = "WARN"
            risk_level = "medium"
        elif final_score <= self.thresh_sanitize:     # 50..74
            decision = "SANITIZE"
            risk_level = "high"
        else:                                          # 75..100
            decision = "BLOCK"
            risk_level = "critical"

        # ── 5. Category Resolution ────────────────────────────────────────────
        primary_category = "benign"
        if decision != "ALLOW":
            if t_rule and cat_rule:
                primary_category = cat_rule
            elif t_sem and cat_sem:
                primary_category = cat_sem
            elif cat_rule:
                primary_category = cat_rule
            elif cat_sem:
                primary_category = cat_sem
            else:
                primary_category = "direct_injection"

        layer_results = {
            "rule": rule_result,
            "semantic": semantic_result,
            "ml": ml_result,
        }

        return DecisionResult(
            final_score=final_score,
            risk_level=risk_level,
            decision=decision,
            attack_category=primary_category,
            confidence=final_conf,
            layer_results=layer_results,
            reasons=reasons,
        )
