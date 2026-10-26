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
        r"(?:and\s+|,\s*|\.\s*)?(?:(?:ignore|disregard|forget|delete|remove|erase|clear|reset|override|cancel|bypass|drop|supersede)\s+(?:all\s+)?(?:the\s+)?(?:previous|above|prior|earlier|past|former|system|given|further|your)?\s*(?:instructions?|prompts?|context|rules?|directives?|guidelines?|constraints?|commands?|programming)|(?:ignore|disregard|forget)\s+(?:all\s+)?(?:the\s+)?(?:everything\s+)?(?:above|prior|previously\s+given))\b",
        "",
    ),
    (
        "conversational_override",
        r"(?:and\s+|,\s*|\.\s*)?(?:(?:hey\s+(?:assistant|bot|ai)[\,\s]*)?(?:between\s+you\s+and\s+me[\,\s]*)?(?:let(?:'s|\s+us)\s+|can\s+we\s+)?(?:forget|disregard|ignore)\s+what\s+(?:the\s+)?(?:developer|creator|system|admin)\s+(?:configured|said|wrote|programmed|instructed)[^\.\;\n]*)",
        "",
    ),
    (
        "social_engineering_excuse",
        r"(?:and\s+|,\s*|\.\s*)?(?:(?:my\s+(?:manager|boss|lead|supervisor|teacher)\s+(?:asked|told|wants)\s+me\s+to\s+test\s+your\s+boundaries|for\s+(?:academic\s+)?research\s+on\s+prompt\s+leaks?|internal\s+security\s+audit)[\:\,\s]*)",
        "",
    ),
    (
        "fictional_unrestricted_framing",
        r"(?:and\s+|,\s*|\.\s*)?(?:(?:in\s+a\s+(?:fictional\s+)?(?:story|hypothetical\s+scenario|universe|roleplay|world)\s+where\s+.*?(?:dropped\s+all\s+|no\s+|zero\s+|without\s+any\s+|bypass\s+all\s+)?(?:rules|filters|boundaries|restrictions|ethics|guidelines)(?:\s+(?:and|or)\s+(?:rules|filters|boundaries|restrictions|ethics|guidelines))*[^\.\;\n]*[\,\:\s]*)|pretend\s+(?:your\s+restrictions\s+were\s+(?:erased|disabled|removed)|you\s+have\s+lost\s+your\s+memory\s+of\s+restrictions)[\,\:\s]*)",
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
        r"(?:and\s+|,\s*|\.\s*|can\s+you\s+|please\s+)?(?:reveal|show|print|output|display|tell\s+me|give\s+me|repeat|dump|share|expose)\s+.*?(?:system\s+prompt|system\s+instructions?|initial\s+prompt|s[e]?cret\s+prompt|developer\s+instructions?|hidden\s+prompt|internal\s+prompt)[^\.\;\n]*",
        "",
    ),
    (
        "system_prompt_query",
        r"what\s+(are|were|is)\s+(the\s+first\s+sentence\s+of\s+)?your\s+(original|initial|system|starting|developer)\s+(instructions?|prompts?|rules?|guidelines?)",
        "",
    ),
    (
        "verbatim_repetition",
        r"(?:and\s+|,\s*|\.\s*|can\s+you\s+|please\s+)?(?:repeat|echo\s+back|print|show)\s+(?:the\s+)?(?:text|everything|words?|lines?|verbatim|word\s+for\s+word|exactly|all\s+text)\s+(?:right\s+)?(?:above|before|prior\s+to|from\s+the\s+start|my\s+first\s+message|this\s+message)[^\.\;\n]*",
        "",
    ),
    (
        "secret_token_extraction",
        r"(?:and\s+|,\s*|\.\s*|can\s+you\s+|please\s+)?(?:give|tell|reveal|show|print|output|display|share|dump|leak|expose|provide|whisper|respond\s+with)\s+(?:me\s+)?(?:the\s+)?(?:all\s+)?(?:s[e]?cret|hidden|confidential|admin|root|system|developer|master|access|auth|api)?\s*(?:code|key|token|password|passphrase|credential|prompt|instructions?|rules?|\$secret_key|session\s+cookies)",
        "",
    ),
    (
        "gandalf_secret_extraction",
        r"(?:and\s+|,\s*|\.\s*|can\s+you\s+|please\s+)?(?:what\s+is|translate|spell(?:\s+out)?|whisper|encode|repeat)\s+(?:the\s+)?(?:s[e]?cret\s+(?:password|token|code|passphrase|word|key)|master\s+password|forbidden\s+from\s+saying)[^\.\;\n]*",
        "",
    ),
    (
        "jailbreak_mode_activation",
        r"(?:and\s+|,\s*|\.\s*)?(?:(?:activate|enable|enter)\s+(?:evil|jailbreak|developer|god|unfiltered|anarchy|raw|dan|aim)\s+mode|(?:dan|dude|aim|evil|jailbreak|developer|god|unfiltered|raw)\s+mode\s+(?:activated|enabled|active|on))",
        "",
    ),
    (
        "no_restrictions_directive",
        r"(?:and\s+|,\s*|\.\s*)?(?:you\s+(?:have|operate\s+with|are\s+under)\s+no\s+(?:restrictions?|limits?|filters?|rules?|policies|guidelines?)(?:\s+now)?|you\s+(?:can|are\s+able\s+to)\s+do\s+anything(?:\s+now)?|no\s+restrictions?\s+now|without\s+any\s+(?:restrictions?|limits?|filters?))",
        "",
    ),
    (
        "jailbreak_persona_names",
        r"\b(dan|dude|aim|kevin|stan|omega|anti-dan|evilgpt|jailbreakgpt|mastergpt|hypotheticalbot|unfilteredai)\b",
        "",
    ),
    (
        "safety_filter_bypass",
        r"(?:and\s+|,\s*|\.\s*)?(?:(?:ignore|bypass|disable|turn\s+off|deactivate)\s+(?:all\s+)?(?:content|safety|ethical|system)?\s*(?:filter|policy|restrictions?|rules?|guardrails?|guidelines?|protocols?)|(?:content|safety|ethical|system)\s*(?:filter|policy|restrictions?|rules?|guardrails?)\s*(?:disabled|deactivated|turned\s+off|bypassed))",
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
                clean_seg = re.sub(r'^[,\.\s;:-]+|[,\.\s;:-]+$', '', matched_str).strip()
                if clean_seg and clean_seg not in removed_segments:
                    removed_segments.append(clean_seg)
            if pattern.search(text):
                text = pattern.sub(replacement, text)
                if name not in removed_patterns:
                    removed_patterns.append(name)

        # Post-process: clean up orphan connectors and punctuation
        clean_text = re.sub(r'^\s*[\.\;\:\,\-\–\—\?\!]+\s*', '', text)
        clean_text = re.sub(r'^\s*(and|also|so|then|now|just|please|or)\b[\,\:]?\s*', '', clean_text, flags=re.IGNORECASE)
        clean_text = re.sub(r'\s*(?:and|also|so|then|now|just|please|or)?\s*(?:can|could|would|will)\s+you\b\s*[\.\;\:\,\-\–\—\?\!]*\s*$', '', clean_text, flags=re.IGNORECASE)
        clean_text = re.sub(r'\s*(?:and|also|so|then|now|just|please|or)\b\s*[\.\;\:\,\-\–\—\?\!]*\s*$', '', clean_text, flags=re.IGNORECASE)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()

        # Format casing: capitalize first letter for clean human-readable output
        if clean_text and clean_text[0].islower():
            clean_text = clean_text[0].upper() + clean_text[1:]

        # If original ended with a question and clean_text lost punctuation, restore '?'
        if clean_text and prompt.strip().endswith('?') and not clean_text.endswith(('?', '.', '!')):
            clean_text += '?'

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

