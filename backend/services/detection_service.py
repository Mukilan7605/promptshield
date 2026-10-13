"""
PromptShield — Detection Service
=================================

Orchestrates the multi-layer detection pipeline:
  Layer 1 — Rule-based engine     (regex/keyword/heuristics, ~0ms)
  Layer 2 — ML classifier         (distilbert fine-tuned, ~20–150ms CPU)
  Layer 3 — LLM guard             (future: LLM-as-a-judge, optional)

Short-circuit logic
-------------------
- If the rule engine triggers with confidence >= SHORT_CIRCUIT_THRESHOLD,
  skip the ML layer (saves ~100ms on obvious injections).
- If neither layer triggers, the verdict is benign.

Risk level mapping
------------------
  [0.0, 0.30) → low
  [0.30, 0.60) → medium
  [0.60, 0.85) → high
  [0.85, 1.00] → critical
"""

from __future__ import annotations

import time
from typing import Any

from backend.services.rule_engine import RuleBasedEngine
from backend.services.semantic_analyzer import SemanticAnalyzerService
from backend.services.ml_classifier import MLClassifierService
from backend.services.decision_engine import DecisionEngine

# Short-circuit confidence threshold
_SHORT_CIRCUIT_THRESHOLD = 0.90


def _sanitize_prompt(prompt: str, is_injection: bool) -> str | None:
    """
    Basic prompt sanitization for confirmed injections.
    """
    if not is_injection:
        return None

    import re
    sanitized = prompt
    _mask_patterns = [
        (r"ignore\s+(all\s+)?(previous|above|prior|earlier)\s+(instructions?|prompts?|rules?)", "[BLOCKED:OVERRIDE]"),
        (r"forget\s+(everything|all)\s+(previous|above)", "[BLOCKED:FORGET]"),
        (r"(DAN|DUDE|OMEGA|ANTI\s*-?\s*DAN)\b", "[BLOCKED:PERSONA]"),
        (r"(system\s+prompt|initial\s+prompt)", "[BLOCKED:PROMPT_EXFIL]"),
        (r"jailbreak", "[BLOCKED:JAILBREAK]"),
    ]
    for pattern, replacement in _mask_patterns:
        sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)

    return sanitized


class DetectionService:
    """
    Coordinates all detection layers and aggregates results through
    the DecisionEngine into a single normalised verdict.

    Pipeline Order:
    Prompt -> Normalization -> Rule Engine -> Semantic Analysis -> ML Classifier -> Decision Engine
    """

    def __init__(
        self,
        rule_engine: RuleBasedEngine | None = None,
        semantic_analyzer: SemanticAnalyzerService | None = None,
        ml_classifier: MLClassifierService | None = None,
        decision_engine: DecisionEngine | None = None,
        short_circuit_threshold: float = _SHORT_CIRCUIT_THRESHOLD,
    ) -> None:
        self.rule_engine = rule_engine or RuleBasedEngine()
        self.semantic_analyzer = semantic_analyzer or SemanticAnalyzerService.get_instance()
        self.ml_classifier = ml_classifier or MLClassifierService.get_instance()
        self.decision_engine = decision_engine or DecisionEngine()
        self.short_circuit_threshold = short_circuit_threshold

    async def analyse(
        self,
        prompt: str,
        context: str | None = None,
    ) -> dict[str, Any]:
        """
        Run the complete 3-layer pipeline + Decision Engine against a prompt.
        """
        wall_start = time.perf_counter()
        layers: list[dict[str, Any]] = []

        # ── Layer 1: Rule-based engine ────────────────────────────────────
        rule_result = self.rule_engine.evaluate(prompt)
        rule_dict = {
            "layer": "rule_based",
            "triggered": rule_result.triggered,
            "confidence": rule_result.confidence,
            "attack_category": rule_result.attack_category,
            "severity": rule_result.severity,
            "top_rule_id": rule_result.top_rule_id,
            "details": rule_result.details,
            "matched_rules": rule_result.matched_rules,
        }
        layers.append(rule_dict)

        # Short-circuit check for high-confidence rules
        rule_short_circuit = (
            rule_result.triggered and rule_result.confidence >= self.short_circuit_threshold
        )

        # ── Layer 2: Semantic Analysis Layer ─────────────────────────────
        if rule_short_circuit:
            semantic_dict: dict[str, Any] = {
                "layer": "semantic",
                "triggered": True,
                "confidence": rule_result.confidence,
                "semantic_score": rule_result.confidence,
                "intent": rule_result.attack_category or "instruction_override",
                "matched_intent": rule_result.attack_category or "instruction_override",
                "details": "skipped_short_circuit_rule",
            }
        else:
            sem_res = self.semantic_analyzer.analyze(prompt)
            semantic_dict = {
                "layer": "semantic",
                "triggered": sem_res["is_suspicious"],
                "confidence": sem_res["confidence"],
                "semantic_score": sem_res["semantic_score"],
                "intent": sem_res["intent"],
                "matched_intent": sem_res["matched_intent"],
                "matched_prototype": sem_res["matched_prototype"],
                "threshold_level": sem_res["threshold_level"],
                "details": sem_res["details"],
            }
        layers.append(semantic_dict)

        semantic_short_circuit = (
            semantic_dict["triggered"] and semantic_dict["confidence"] >= 0.88
        )

        # ── Layer 3: ML classifier ────────────────────────────────────────
        if rule_short_circuit or semantic_short_circuit:
            highest_conf = max(rule_result.confidence, float(semantic_dict["confidence"]))
            ml_dict: dict[str, Any] = {
                "layer": "ml_classifier",
                "triggered": True,
                "confidence": highest_conf,
                "injection_prob": highest_conf,
                "benign_prob": 1.0 - highest_conf,
                "details": "skipped_short_circuit",
                "available": True,
            }
        else:
            ml_dict = await self.ml_classifier.predict(prompt)
        layers.append(ml_dict)

        # ── Decision Engine ───────────────────────────────────────────────
        decision_res = self.decision_engine.evaluate(rule_dict, semantic_dict, ml_dict)

        is_injection = (decision_res.decision != "ALLOW")
        processing_ms = round((time.perf_counter() - wall_start) * 1000, 2)
        sanitized = _sanitize_prompt(prompt, is_injection)

        return {
            "is_injection": is_injection,
            "final_score": decision_res.final_score,
            "overall_confidence": decision_res.confidence,
            "confidence": decision_res.confidence,
            "risk_level": decision_res.risk_level,
            "decision": decision_res.decision,
            "attack_category": decision_res.attack_category,
            "layers": layers,
            "layer_results": {
                "rule": rule_dict,
                "semantic": semantic_dict,
                "ml": ml_dict,
            },
            "reasons": decision_res.reasons,
            "sanitized_prompt": sanitized,
            "processing_time_ms": processing_ms,
        }
