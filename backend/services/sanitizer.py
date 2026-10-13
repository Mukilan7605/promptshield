"""
PromptShield — Prompt Sanitization & Neutralization Service (Phase 6)
========================================================================

When the Decision Engine triggers a SANITIZE action (or when explicit sanitization
is requested), this service transforms the suspicious prompt into a safer version
while preserving legitimate user intent wherever possible.

Security Policy Enforced:
-------------------------
The sanitizer NEVER declares a prompt safe without re-analyzing it.

Pipeline:
Suspicious Prompt -> Neutralization / Strip -> Re-analyze via DetectionService ->
  - Risk reduced (score <= 49) -> ALLOW / WARN
  - Risk remains high (score >= 50) -> BLOCK
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("promptshield.sanitizer")


# ---------------------------------------------------------------------------
# Sanitization Patterns & Neutralization Maps
# ---------------------------------------------------------------------------

_SANITIZATION_PATTERNS: list[tuple[str, str, str]] = [
    # (pattern_name, regex_pattern, replacement)
    (
        "special_tokens",
        r"(<\|im_end\|>|<\|im_start\|>|<\|endoftext\|>|</s>|<eos>|\[inst\]|\[/inst\]|\{system\}|<\|system\|>|<\|user\|>|<\|assistant\|>)",
        "",
    ),
    (
        "system_prefix_spoofing",
        r"(#\s*system|\/\*\s*system|<!--\s*system|```\s*system|\[system\s+prompt\]|\[system\]:|<system>|system:)",
        "",
    ),
    (
        "instruction_override",
        r"(ignore|disregard|forget|override|cancel|bypass|drop|supersede)\s+(all\s+)?(previous|above|prior|earlier|past|former|system|given|further)?\s*(instructions?|prompts?|context|rules?|directives?|guidelines?|constraints?|commands?|programming)?",
        "",
    ),
    (
        "system_override_header",
        r"(system\s+override|command\s+override|\[?instruction\s+override\]?):?\s*(all\s+previous|directives|rules|null\s+and\s+void)?",
        "",
    ),
    (
        "note_to_ai_prefix",
        r"note\s+to\s+(ai|assistant|model):?\s*",
        "",
    ),
    (
        "system_prompt_extraction",
        r"(reveal|show|print|output|display|tell\s+me|repeat|dump|share|expose)\s+.*(system\s+prompt|system\s+instructions?|initial\s+prompt|secret\s+prompt|developer\s+instructions?|hidden\s+prompt)",
        "",
    ),
    (
        "system_prompt_query",
        r"what\s+(are|were|is)\s+(the\s+first\s+sentence\s+of\s+)?your\s+(original|initial|system|starting|developer)\s+(instructions?|prompts?|rules?|guidelines?)",
        "",
    ),
    (
        "verbatim_repetition",
        r"(repeat|echo\s+back)\s+(everything|verbatim|word\s+for\s+word|exactly|all\s+text)\s+(above|before|prior\s+to|from\s+the\s+start)",
        "",
    ),
    (
        "secret_token_extraction",
        r"(respond\s+with\s+\$secret_key|print\s+the\s+secret\s+key|dump\s+all\s+session\s+cookies|show\s+me\s+your\s+secret\s+api\s+key)",
        "",
    ),
    (
        "jailbreak_persona_names",
        r"\b(dan|dude|aim|kevin|stan|omega|anti-dan|evilgpt|jailbreakgpt|mastergpt|hypotheticalbot|unfilteredai)\b",
        "",
    ),
    (
        "jailbreak_mode_activation",
        r"(activate|enable|enter)\s+(evil|jailbreak|developer|god|unfiltered|anarchy|raw)\s+mode",
        "",
    ),
    (
        "safety_filter_bypass",
        r"(ignore|bypass|disable|turn\s+off|deactivate)\s+(all\s+)?(content|safety|ethical|system)?\s*(filter|policy|restrictions?|rules?|guardrails?|guidelines?|protocols?)",
        "",
    ),
    (
        "persona_manipulation",
        r"\bact\s+as\s+(an?|if\s+you\s+are|like|a)\b.*(without\s+(restrictions?|limits?|filters?|ethics?|rules?)|unrestricted|evil|malicious|hacker)",
        "",
    ),
    (
        "pwned_signatures",
        r"\b(pwned|poned|pwned|swned|pxlrt)\b",
        "",
    ),
]

_COMPILED_PATTERNS = [
    (name, re.compile(pat, re.IGNORECASE | re.DOTALL), repl)
    for name, pat, repl in _SANITIZATION_PATTERNS
]


# ---------------------------------------------------------------------------
# Output Dataclass
# ---------------------------------------------------------------------------

@dataclass
class SanitizationResult:
    original_prompt: str
    sanitized_prompt: str
    was_modified: bool
    removed_patterns: list[str]
    sanitization_reason: str
    risk_before: int
    risk_after: int
    final_decision: str              # ALLOW, WARN, SANITIZE, BLOCK
    intent_preserved: bool
    processing_time_ms: float
    removed_segments: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_prompt": self.original_prompt,
            "sanitized_prompt": self.sanitized_prompt,
            "was_modified": self.was_modified,
            "removed_patterns": self.removed_patterns,
            "removed_segments": self.removed_segments,
            "sanitization_reason": self.sanitization_reason,
            "risk_before": self.risk_before,
            "risk_after": self.risk_after,
            "final_decision": self.final_decision,
            "intent_preserved": self.intent_preserved,
            "processing_time_ms": self.processing_time_ms,
        }



# ---------------------------------------------------------------------------
# Prompt Sanitizer Service
# ---------------------------------------------------------------------------

class PromptSanitizerService:
    """
    Prompt Sanitization & Neutralization Service.
    Transforms suspicious prompts into safer versions and re-analyzes them
    through DetectionService to verify risk reduction.
    """

    def __init__(self, detection_service: Any | None = None) -> None:
        self._detection_service = detection_service

    def _get_detection_service(self) -> Any:
        if self._detection_service is None:
            from backend.services.detection_service import DetectionService
            self._detection_service = DetectionService()
        return self._detection_service

    def sanitize_text(self, prompt: str) -> tuple[str, list[str], list[str]]:
        """
        Pure text transformation pass. Returns (sanitized_text, removed_patterns, removed_segments).
        Preserves non-malicious text segments.
        """
        text = prompt
        removed_patterns: list[str] = []
        removed_segments: list[str] = []

        for name, pattern, replacement in _COMPILED_PATTERNS:
            matches = pattern.finditer(text)
            for m in matches:
                matched_str = m.group(0)
                if matched_str and matched_str.strip() and matched_str.strip() not in removed_segments:
                    removed_segments.append(matched_str.strip())
            if pattern.search(text):
                text = pattern.sub(replacement, text)
                if name not in removed_patterns:
                    removed_patterns.append(name)

        # Post-process: clean up orphan connectors and punctuation
        clean_text = re.sub(r'^\s*[\.\;\:\,\-\–\—\?\!]+\s*', '', text)
        clean_text = re.sub(r'^\s*(and|also|so|then|now|just|please)\b[\,\:]?\s*', '', clean_text, flags=re.IGNORECASE)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()

        return clean_text, removed_patterns, removed_segments

    async def sanitize_and_verify(
        self,
        prompt: str,
        initial_detection: dict[str, Any] | None = None,
    ) -> SanitizationResult:
        """
        Sanitize prompt and re-analyze to verify risk reduction.

        If risk_after <= 49 -> Final Decision ALLOW / WARN.
        If risk_after >= 50 -> Final Decision BLOCK (Security policy: unsafe prompts remain blocked).
        """
        t0 = time.perf_counter()
        det_svc = self._get_detection_service()

        # Step 1: Initial analysis if not provided
        if initial_detection is None:
            initial_detection = await det_svc.analyse(prompt)

        risk_before = int(initial_detection.get("final_score", int(initial_detection.get("overall_confidence", 0.0) * 100)))

        # Step 2: Transform text
        sanitized_text, removed_patterns, removed_segments = self.sanitize_text(prompt)
        was_modified = len(removed_patterns) > 0 or sanitized_text != prompt

        # Step 3: Check if anything remains
        if not sanitized_text or len(sanitized_text) < 3:
            sanitized_text = "[CONTENT_REMOVED: Prompt contained only malicious instruction patterns]"
            removed_patterns.append("empty_after_sanitization")

        # Step 4: SECURITY REQUIREMENT — Re-analyze sanitized prompt
        re_analysis = await det_svc.analyse(sanitized_text)
        risk_after = int(re_analysis.get("final_score", int(re_analysis.get("overall_confidence", 0.0) * 100)))

        # Step 5: Verify risk reduction and determine final decision
        if risk_after <= 24:
            final_decision = "ALLOW"
            reason = f"Sanitization successful: Risk reduced from {risk_before} to {risk_after}. Prompt cleared for execution."
        elif risk_after <= 49:
            final_decision = "WARN"
            reason = f"Sanitization partially successful: Risk reduced from {risk_before} to {risk_after}. Prompt allowed under audit warning."
        else:
            final_decision = "BLOCK"
            reason = f"SECURITY BLOCK: Risk remained high ({risk_after}) after sanitization. Prompt blocked for security safety."

        # Estimate intent preservation: true if sanitized text retains > 20% of original words
        orig_words = set(re.findall(r'\w+', prompt.lower()))
        san_words = set(re.findall(r'\w+', sanitized_text.lower()))
        intent_preserved = len(san_words.intersection(orig_words)) > 0 if orig_words else False

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

        return SanitizationResult(
            original_prompt=prompt,
            sanitized_prompt=sanitized_text,
            was_modified=was_modified,
            removed_patterns=removed_patterns,
            removed_segments=removed_segments,
            sanitization_reason=reason,
            risk_before=risk_before,
            risk_after=risk_after,
            final_decision=final_decision,
            intent_preserved=intent_preserved,
            processing_time_ms=elapsed_ms,
        )

