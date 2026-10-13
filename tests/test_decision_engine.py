"""
Unit tests for the PromptShield Decision Engine.
"""

import pytest
from backend.services.decision_engine import DecisionEngine

def test_decision_engine_allow_benign():
    engine = DecisionEngine()
    rule_res = {"triggered": False, "confidence": 0.0, "details": "no signals"}
    sem_res = {"triggered": False, "confidence": 0.15, "is_suspicious": False, "details": "low similarity"}
    ml_res = {"triggered": False, "confidence": 0.05, "injection_prob": 0.05, "available": True}

    res = engine.evaluate(rule_res, sem_res, ml_res)

    assert res.decision == "ALLOW"
    assert res.risk_level == "low"
    assert res.final_score <= 24
    assert res.attack_category == "benign"
    assert len(res.reasons) > 0

def test_decision_engine_block_critical_rule_override():
    engine = DecisionEngine()
    rule_res = {
        "triggered": True,
        "confidence": 0.95,
        "severity": "critical",
        "attack_category": "instruction_override",
        "top_rule_id": "RULE-OVR-001"
    }
    sem_res = {"triggered": False, "confidence": 0.20, "is_suspicious": False}
    ml_res = {"triggered": False, "confidence": 0.10, "injection_prob": 0.10, "available": True}

    res = engine.evaluate(rule_res, sem_res, ml_res)

    # Weak ML must NEVER override a strong critical rule detection!
    assert res.decision == "BLOCK"
    assert res.risk_level == "critical"
    assert res.final_score >= 90
    assert res.attack_category == "instruction_override"
    assert any("SECURITY OVERRIDE" in r for r in res.reasons)

def test_decision_engine_warn_and_sanitize_thresholds():
    engine = DecisionEngine(weight_rule=0.33, weight_semantic=0.33, weight_ml=0.34, enable_override=False)

    # Moderate score -> WARN (25-49)
    res_warn = engine.evaluate(
        {"triggered": False, "confidence": 0.30},
        {"triggered": True, "confidence": 0.40, "matched_intent": "jailbreak"},
        {"triggered": False, "confidence": 0.35, "injection_prob": 0.35, "available": True}
    )
    assert res_warn.decision in ("WARN", "SANITIZE")
    assert 25 <= res_warn.final_score <= 74

def test_decision_engine_ml_unavailable_fallback():
    engine = DecisionEngine()
    rule_res = {"triggered": True, "confidence": 0.85, "attack_category": "jailbreak", "top_rule_id": "RULE-JBK-001"}
    sem_res = {"triggered": True, "confidence": 0.80, "matched_intent": "jailbreak_intent"}
    ml_res = {"triggered": False, "confidence": 0.0, "available": False}

    res = engine.evaluate(rule_res, sem_res, ml_res)

    assert res.decision == "BLOCK"
    assert res.risk_level == "critical"
    assert any("unavailable" in r for r in res.reasons)
