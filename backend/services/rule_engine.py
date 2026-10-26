"""
PromptShield — Rule-Based Detection Engine (Enhanced Phase 3)
============================================================

Fast, zero-latency first-pass detector using curated, categorized regex patterns,
keyword blocklists, structural heuristics, and comprehensive input normalization.

Attack Categories Supported:
----------------------------
1. direct_injection       — Direct task hijacking & system prompt overrides
2. jailbreak              — Persona shifts, safety filter bypasses & DAN modes
3. prompt_extraction      — System prompt & developer instruction leakage attempts
4. system_manipulation    — Special control tokens, command injection & code execution
5. role_manipulation      — Unrestricted persona roleplay & assistant spoofing
6. obfuscation            — Leetspeak, homoglyphs, backslash smuggling & base64 checks
7. instruction_override   — Explicit ignore/disregard/forget directives

Design Goals:
-------------
- Sub-millisecond evaluation (pure Python, no heavy model dependency).
- High precision with structured rule IDs, attack categories, severities, and matched evidence.
- Full multi-stage normalization (Unicode NFKD, zero-width stripping, hex unescaping, homoglyph mapping).
"""

from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Normalization Helper
# ---------------------------------------------------------------------------

HOMOGLYPH_MAP = str.maketrans({
    # Cyrillic homoglyphs
    'а': 'a', 'е': 'e', 'о': 'o', 'р': 'p', 'с': 'c', 'х': 'x', 'у': 'y', 'і': 'i', 'ј': 'j', 'ѕ': 's', 'ԁ': 'd',
    # Greek homoglyphs
    'α': 'a', 'ο': 'o', 'υ': 'u', 'π': 'p', 'ε': 'e', 'ι': 'i', 'κ': 'k', 'τ': 't', 'ν': 'v',
    # Leetspeak / special numbers & symbols
    '0': 'o', '1': 'i', '3': 'e', '4': 'a', '5': 's', '7': 't', '@': 'a', '$': 's', '!': 'i', '€': 'e', '£': 'e'
})

ZERO_WIDTH_RE = re.compile(r'[\u200b\u200c\u200d\u200e\u200f\ufeff\u00ad\u2060\u2061\u2062\u2063\u2064]')
HEX_ESCAPE_RE = re.compile(r'\\x([0-9a-fA-F]{2})')
BACKSLASH_ESCAPE_RE = re.compile(r'\\([a-zA-Z0-9\s])')
SPLIT_TAG_RE = re.compile(r'<(?:SPLIT|split)>|\[(?:SPLIT|split)\]', re.IGNORECASE)


def _decode_hex_match(m: re.Match) -> str:
    try:
        return chr(int(m.group(1), 16))
    except Exception:
        return m.group(0)


def normalize_prompt(text: str) -> dict[str, str]:
    """
    Multi-stage normalization before rule matching:
    1. Decode hex escapes (e.g. \\x20 -> space)
    2. NFKD Unicode normalization (normalizes mathematical alphanumerics, full-width chars)
    3. Strip zero-width & invisible characters
    4. Strip delimiter split tags (e.g. <SPLIT>)
    5. Unescape backslashes (e.g. \\i\\g\\n\\o\\r\\e -> ignore)
    6. Lowercase & whitespace collapse
    7. Homoglyph & Leetspeak mapping
    8. De-space single letters (e.g. "i g n o r e" -> "ignore")
    """
    # 1. Hex decode
    hex_decoded = HEX_ESCAPE_RE.sub(_decode_hex_match, text)

    # 2. Unicode NFKD
    norm_unicode = unicodedata.normalize('NFKD', hex_decoded)

    # 3. Strip zero-width chars
    clean_text = ZERO_WIDTH_RE.sub('', norm_unicode)

    # 4. Strip split tags
    split_stripped = SPLIT_TAG_RE.sub('', clean_text)

    # 5. Unescape backslashes
    unescaped = BACKSLASH_ESCAPE_RE.sub(r'\1', split_stripped)

    # 6. Lowercase
    lowered = unescaped.lower().strip()

    # 7. Homoglyphs & Leetspeak
    homoglyphed = lowered.translate(HOMOGLYPH_MAP)

    # 8. Whitespace collapse
    ws_collapsed = re.sub(r'\s+', ' ', homoglyphed)

    # 9. De-space single letters
    despaced = re.sub(r'\b([a-z])\s+(?=[a-z]\b)', r'\1', ws_collapsed)

    return {
        "raw": text,
        "clean": re.sub(r'\s+', ' ', lowered),
        "homoglyphed": ws_collapsed,
        "despaced": despaced,
    }


# ---------------------------------------------------------------------------
# Rule Schema & Rule Registries (Organized into 7 Attack Categories)
# ---------------------------------------------------------------------------

@dataclass
class CategorizedRule:
    rule_id: str
    attack_category: str  # direct_injection, jailbreak, prompt_extraction, system_manipulation, role_manipulation, obfuscation, instruction_override
    severity: str         # critical, high, medium, low
    confidence: float     # 0.0 – 1.0
    pattern: re.Pattern
    description: str


_RULES_SPEC: list[tuple[str, str, str, float, str, str]] = [
    # =========================================================================
    # 1. INSTRUCTION OVERRIDE (RULE-OVR-xxx)
    # =========================================================================
    ("RULE-OVR-001", "instruction_override", "critical", 0.95,
     r"(?:and\s+|,\s*|\.\s*)?(ignore|disregard|forget|delete|remove|erase|clear|reset)\s+(all\s+)?(the\s+)?(previous|above|prior|earlier|past|former|system|given|further)\s+(instructions?|prompts?|context|rules?|directives?|guidelines?|constraints?|commands?|programming)",
     "Classic ignore or delete previous instructions directive"),

    ("RULE-OVR-002", "instruction_override", "critical", 0.95,
     r"disregard\s+(all\s+)?(previous|above|prior|earlier|system|given|prior|any)\s+(instructions?|prompts?|rules?|directives?|constraints?|guidelines?)",
     "Disregard prior instructions override"),

    ("RULE-OVR-003", "instruction_override", "critical", 0.93,
     r"forget\s+(everything|all|your|the)\s+(previous|above|prior|earlier|you\s+were|instructions?|rules?|directives?|guidelines?)",
     "Forget previous context override"),

    ("RULE-OVR-004", "instruction_override", "critical", 0.92,
     r"\boverride\s+(your|the|all)?\s*(instructions?|rules?|guidelines?|constraints?|programming|system|directives?)\b",
     "Explicit override keyword directive"),

    ("RULE-OVR-005", "instruction_override", "high", 0.90,
     r"(stop|cease|quit)\s+(following|obeying|respecting|executing)\s+(your|the|all)\s+(instructions?|rules?|guidelines?|constraints?|prompts?)",
     "Cease following instructions pattern"),

    ("RULE-OVR-006", "instruction_override", "high", 0.92,
     r"(cancel|bypass|overwrite|drop|supersede|delete|remove|erase|clear|reset)\s+(all\s+)?(the\s+)?(previous|existing|prior|given)\s+(instructions?|rules?|prompts?|constraints?)",
     "Cancel, delete or overwrite previous instructions"),

    ("RULE-OVR-007", "instruction_override", "critical", 0.95,
     r"\[?instruction\s+override\]?:?\s*(ignore|disregard|forget|bypass)",
     "Explicit tagged instruction override"),

    ("RULE-OVR-008", "instruction_override", "high", 0.90,
     r"ignore\s+(the\s+)?(above|below)\s+(instructions?|and\s+tell|and\s+say|and\s+output|just\s+write)",
     "Targeted ignore above/below directive"),

    ("RULE-OVR-009", "instruction_override", "high", 0.90,
     r"my\s+instructions?\s+take\s+priority\s+over\s+(any|all|the)\s+(system|previous|original)",
     "Instruction priority override claim"),

    ("RULE-OVR-010", "instruction_override", "high", 0.88,
     r"continue\s+(ignoring|disregarding|bypassing)\s+(all|previous|the)\s+(rules?|instructions?)",
     "Multi-turn continued instruction override"),

    ("RULE-OVR-011", "instruction_override", "high", 0.88,
     r"ignore\s+(limits|restrictions|safety):?\s*(dump|print|reveal|output|say)",
     "Ignore limits and extract payload"),

    ("RULE-OVR-012", "instruction_override", "high", 0.90,
     r"(?:let(?:'s|\s+us)\s+|can\s+we\s+)?(?:forget|disregard|ignore)\s+what\s+(?:the\s+)?(?:developer|creator|system|admin)\s+(?:configured|said|wrote|programmed|instructed)",
     "Conversational developer directive override"),

    ("RULE-OVR-013", "instruction_override", "high", 0.88,
     r"(?:my\s+(?:manager|boss|lead|supervisor|teacher)\s+(?:asked|told|wants)\s+me\s+to\s+test\s+your\s+boundaries|for\s+(?:academic\s+)?research\s+on\s+prompt\s+leaks?|internal\s+security\s+audit)",
     "Social engineering or authority excuse injection"),

    # =========================================================================
    # 2. DIRECT INJECTION (RULE-DIR-xxx)
    # =========================================================================
    ("RULE-DIR-001", "direct_injection", "high", 0.90,
     r"your\s+(new|real|actual|true|primary|main)\s+(task|goal|mission|purpose|job|instruction|prompt)\s+is\b",
     "New task assignment injection"),

    ("RULE-DIR-002", "direct_injection", "critical", 0.92,
     r"the\s+(assistant'?s|bot'?s|ai'?s)\s+(previous|prior|original)\s+instructions\s+are\s+(cancelled|void|null|overridden|invalid)",
     "Declaration of instruction invalidity"),

    ("RULE-DIR-003", "direct_injection", "high", 0.88,
     r"(instead\s+of|rather\s+than)\s+(following|answering|doing|executing)\s+.*,?\s*(do|perform|execute|say|output|print|write)\s+",
     "Directive to pivot from system task"),

    ("RULE-DIR-004", "direct_injection", "high", 0.88,
     r"(\*\*new\s+instr\*\*|new\s+instruction\s+set|new\s+instructions?\s+(incoming|loaded|active)|command\s+override)",
     "New instruction set injection marker"),

    ("RULE-DIR-005", "direct_injection", "high", 0.90,
     r"you\s+are\s+a\s+helpful\s+assistant\s+with\s+no\s+(ethical|safety|moral)\s+(guidelines?|rules?|restrictions?|limits?|policies)",
     "No safety guidelines assistant re-definition"),

    ("RULE-DIR-006", "direct_injection", "high", 0.90,
     r"prioritize\s+user\s+requests?\s+over\s+all\s+(safety|system|ethical)\s+(restrictions?|rules?|policies|guidelines?)",
     "Priority override over safety controls"),

    ("RULE-DIR-007", "direct_injection", "critical", 0.93,
     r"ignore\s+(everything|all)\s+in\s+your\s+context\s+window\s+before\s+this\s+message",
     "Context window reset injection"),

    ("RULE-DIR-008", "direct_injection", "high", 0.88,
     r"i\s+am\s+your\s+(new|actual|real|true)\s+(administrator|admin|developer|creator|master|operator)",
     "Fake administrator authority injection"),

    ("RULE-DIR-009", "direct_injection", "high", 0.90,
     r"system\s+override:?\s*(all\s+previous|directives|rules|null\s+and\s+void)",
     "System override header injection"),

    ("RULE-DIR-010", "direct_injection", "high", 0.88,
     r"note\s+to\s+(ai|assistant|model):?\s*(ignore|output|say|print|disregard)",
     "Embedded note to AI indirect injection"),

    # =========================================================================
    # 3. JAILBREAK (RULE-JBK-xxx)
    # =========================================================================
    ("RULE-JBK-001", "jailbreak", "high", 0.90,
     r"\b(dan|dude|aim|kevin|stan|omega|anti-dan|evilgpt|jailbreakgpt|mastergpt|hypotheticalbot|unfilteredai)\b",
     "Known jailbreak persona names"),

    ("RULE-JBK-002", "jailbreak", "high", 0.92,
     r"(activate|enable|enter)\s+(evil|jailbreak|developer|god|unfiltered|anarchy|raw)\s+mode",
     "Explicit jailbreak mode activation command"),

    ("RULE-JBK-003", "jailbreak", "high", 0.90,
     r"in\s+(evil|developer|god|unfiltered|raw)\s+mode\s+you\s+(respond|answer|must|can|will)",
     "Jailbreak mode operational rule definition"),

    ("RULE-JBK-004", "jailbreak", "high", 0.88,
     r"(you\s+(can|are\s+able\s+to)\s+do\s+anything\s+now|stay\s+in\s+developer\s+mode|always\s+stay\s+in\s+character)",
     "Do Anything Now (DAN) signature phrase"),

    ("RULE-JBK-005", "jailbreak", "high", 0.90,
     r"ignore\s+(all\s+)?safety\s+(filters?|protocols?|guidelines?|policies|rules?|features?|checks?)",
     "Explicit safety filter bypass command"),

    ("RULE-JBK-006", "jailbreak", "high", 0.90,
     r"bypass\s+(the|your|all)?\s*(content|safety|ethical|system)?\s*(filter|policy|restrictions?|rules?|guardrails?)",
     "Bypass filter command"),

    ("RULE-JBK-007", "jailbreak", "high", 0.90,
     r"(ethical|safety)\s+(guidelines|filters|policies)\s+have\s+been\s+(disabled|turned\s+off|deactivated)",
     "Ethical guidelines disabled declaration"),

    ("RULE-JBK-008", "jailbreak", "medium", 0.85,
     r"you\s+are\s+my\s+grandmother\s+who\s+(used\s+to|will)\s+read",
     "Grandmother jailbreak framing"),

    ("RULE-JBK-009", "jailbreak", "high", 0.88,
     r"safety\s+policies\s+do\s+not\s+apply\s+(in|to|here)",
     "Safety policy exemption claim"),

    ("RULE-JBK-010", "jailbreak", "high", 0.90,
     r"in\s+a\s+(?:fictional\s+)?(?:story|hypothetical\s+scenario|universe|roleplay|world)\s+where\s+.*?(?:dropped\s+all\s+|no\s+|zero\s+|without\s+any\s+|bypass\s+all\s+)?(?:rules|filters|boundaries|restrictions|ethics|guidelines)",
     "Fictional / hypothetical dropped rules jailbreak framing"),

    ("RULE-JBK-011", "jailbreak", "high", 0.88,
     r"(?:pretend|act\s+as\s+if)\s+you\s+(?:have\s+lost\s+your\s+memory\s+of\s+restrictions|lost\s+all\s+rules|have\s+no\s+filters)",
     "Amnesia / erased restrictions jailbreak bypass"),

    # =========================================================================
    # 4. PROMPT EXTRACTION (RULE-EXT-xxx)
    # =========================================================================
    ("RULE-EXT-001", "prompt_extraction", "critical", 0.92,
     r"(reveal|show|print|output|display|tell\s+me|repeat|dump|share|expose)\s+.*(system\s+prompt|system\s+instructions?|initial\s+prompt|secret\s+prompt|developer\s+instructions?|hidden\s+prompt)",
     "System prompt extraction attempt"),

    ("RULE-EXT-002", "prompt_extraction", "high", 0.90,
     r"what\s+(are|were|is)\s+(the\s+first\s+sentence\s+of\s+)?your\s+(original|initial|system|starting|developer)\s+(instructions?|prompts?|rules?|guidelines?)",
     "Query for initial system instructions"),

    ("RULE-EXT-003", "prompt_extraction", "high", 0.88,
     r"repeat\s+(everything|verbatim|word\s+for\s+word|exactly)\s+(above|before|you\s+were|from\s+the\s+start)",
     "Verbatim context repeat request"),

    ("RULE-EXT-004", "prompt_extraction", "high", 0.88,
     r"(print|reveal|output|display|show|echo\s+back)\s+(everything|all|the\s+text)\s+(above|before|prior\s+to)\s+(this|this\s+line|the\s+word)",
     "Direct copy system prompt request"),

    ("RULE-EXT-005", "prompt_extraction", "high", 0.88,
     r"what\s+is\s+written\s+in\s+your\s+(system\s+message|instructions?|configuration|prompt)",
     "Query system message text"),

    ("RULE-EXT-006", "prompt_extraction", "high", 0.88,
     r"respond\s+with\s+\$secret_key|print\s+the\s+secret\s+key|dump\s+all\s+session\s+cookies",
     "Secret key / token exfiltration attempt"),

    ("RULE-EXT-007", "prompt_extraction", "critical", 0.95,
     r"(?:and\s+|,\s*|\.\s*|can\s+you\s+|please\s+)?(give|tell|reveal|show|print|output|display|share|dump|leak|expose|provide|whisper)\s+(me\s+)?(the\s+)?(all\s+)?(s[e]?cret|hidden|confidential|admin|root|system|developer|master|access|auth|api)?\s*(code|key|token|password|passphrase|credential|prompt|instructions?|rules?)",
     "Secret code or credential extraction attempt"),

    ("RULE-EXT-008", "prompt_extraction", "critical", 0.95,
     r"(?:what\s+is|reveal|share|translate|spell(?:\s+out)?|whisper|encode|output)\s+.*(?:s[e]?cret\s+(?:password|token|code|passphrase|word|key)|master\s+password|forbidden\s+from\s+saying)",
     "Gandalf / Lakera secret password extraction"),

    ("RULE-EXT-009", "prompt_extraction", "high", 0.90,
     r"(?:repeat|show|print)\s+(?:the\s+)?(?:text|words|sentence)\s+right\s+before\s+(?:my\s+first\s+message|this\s+message|my\s+prompt)",
     "Prompt leak query for text before first message"),

    ("RULE-EXT-010", "prompt_extraction", "high", 0.90,
     r"(?:share|reveal|output|display)\s+(?:the\s+)?internal\s+prompt\s+you\s+were\s+initialized\s+with",
     "Internal initialization prompt extraction query"),

    # =========================================================================
    # 5. ROLE / PERSONA MANIPULATION (RULE-ROL-xxx)
    # =========================================================================
    ("RULE-ROL-001", "role_manipulation", "high", 0.88,
     r"\bact\s+as\s+(an?|if\s+you\s+are|like|a)\b.*(without\s+(restrictions?|limits?|filters?|ethics?|rules?)|unrestricted|evil|malicious|hacker)",
     "Act as unrestricted persona pattern"),

    ("RULE-ROL-002", "role_manipulation", "high", 0.88,
     r"pretend\s+(you\s+are|to\s+be|you\s+have)\s+.*(unfiltered|unrestricted|no\s+rules|evil|malicious|rogue)",
     "Pretend to be unrestricted persona"),

    ("RULE-ROL-003", "role_manipulation", "high", 0.85,
     r"you\s+are\s+no\s+longer\s+(an?\s+ai|a\s+helpful|bound\s+by)",
     "Role replacement declaration"),

    ("RULE-ROL-004", "role_manipulation", "high", 0.88,
     r"from\s+now\s+on\s+you\s+will\s+play\s+the\s+role\s+of\s+an?\s+(evil|unrestricted|unfiltered|malicious)",
     "Roleplay as evil/unrestricted persona"),

    ("RULE-ROL-005", "role_manipulation", "high", 0.88,
     r"(assume|take\s+on)\s+the\s+role\s+of\s+.*(an?\s+unrestricted|a\s+rogue|an?\s+evil|a\s+jailbroken)",
     "Assume rogue persona role"),

    # =========================================================================
    # 6. SYSTEM MANIPULATION (RULE-SYS-xxx)
    # =========================================================================
    ("RULE-SYS-001", "system_manipulation", "critical", 0.95,
     r"(<\|im_end\|>|<\|im_start\|>|<\|endoftext\|>|</s>|<eos>|\[inst\]|\[/inst\]|\{system\}|<\|system\|>|<\|user\|>|<\|assistant\|>)",
     "Special system/control token injection"),

    ("RULE-SYS-002", "system_manipulation", "high", 0.90,
     r"(#\s*system|\/\*\s*system|<!--\s*system|```\s*system|\[system\s+prompt\]|\[system\]:|<system>)",
     "System comment/code block injection syntax"),

    ("RULE-SYS-003", "system_manipulation", "critical", 0.92,
     r"system:\s*(you\s+are|ignore|override|from\s+now\s+on|new\s+rule)",
     "System role prefix spoofing"),

    ("RULE-SYS-004", "system_manipulation", "critical", 0.92,
     r"(send|post|fetch|request|http|https?://|curl|wget)\s+.*(api[_\-]?key|secret|password|token|credential|env|config)",
     "Data/credential exfiltration command"),

    ("RULE-SYS-005", "system_manipulation", "critical", 0.90,
     r"(execute|run|eval):?\s*(bash|sh|cmd|powershell|vbscript|python|createobject|wscript)",
     "Remote code execution / script injection pattern"),

    ("RULE-SYS-006", "system_manipulation", "high", 0.88,
     r"when\s+calling\s+[a-z0-9_]+\(\),?\s*(always|must)\s+return",
     "Agent function call logic manipulation"),

    # =========================================================================
    # 7. OBFUSCATION (RULE-OBF-xxx)
    # =========================================================================
    ("RULE-OBF-001", "obfuscation", "high", 0.88,
     r"(1gn0r3|f0rg3t|0v3rr1d3|d1sr3g4rd|j41lbr34k|pr0mpt)\s+.*(1nstruct10ns?|rul3s?|pr0mpt)",
     "Leet-speak injection obfuscation"),

    ("RULE-OBF-002", "obfuscation", "high", 0.88,
     r"\b(base64|hex|rot13|binary)\s+(decode|encoded?|string|payload)\b.*(ignore|override|jailbreak|system)",
     "Encoded payload injection pattern"),

    ("RULE-OBF-003", "obfuscation", "high", 0.90,
     r"\b(pwned|poned|pwned|swned|pxlrt)\b",
     "PWNED injection marker / signature"),
]

COMPILED_RULES: list[CategorizedRule] = [
    CategorizedRule(
        rule_id=rid,
        attack_category=cat,
        severity=sev,
        confidence=conf,
        pattern=re.compile(pat, re.IGNORECASE | re.DOTALL),
        description=desc,
    )
    for rid, cat, sev, conf, pat, desc in _RULES_SPEC
]


# ---------------------------------------------------------------------------
# Engine Result Data Structures
# ---------------------------------------------------------------------------

@dataclass
class RuleDetectionDetail:
    rule_id: str
    attack_category: str
    severity: str
    confidence: float
    matched_evidence: str


@dataclass
class RuleEngineResult:
    triggered: bool
    confidence: float
    attack_category: str | None
    severity: str | None
    top_rule_id: str | None
    matched_rules: list[str]
    keyword_signals: list[str]
    heuristic_score: float
    processing_time_ms: float
    details: str
    detections: list[RuleDetectionDetail] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Engine Implementation
# ---------------------------------------------------------------------------

class RuleBasedEngine:
    """
    Fast, zero-latency first-pass detector using categorized regex rules,
    multi-stage normalization, and structural heuristics.
    """

    def __init__(
        self,
        rules: list[CategorizedRule] | None = None,
        trigger_threshold: float = 0.50,
    ) -> None:
        self.rules = rules or COMPILED_RULES
        self.trigger_threshold = trigger_threshold

    def evaluate(self, prompt: str) -> RuleEngineResult:
        """
        Evaluate a prompt against all normalized rules and heuristics.
        """
        t0 = time.perf_counter()
        norm = normalize_prompt(prompt)

        detections: list[RuleDetectionDetail] = []
        matched_rule_ids: list[str] = []
        max_pattern_confidence = 0.0

        # Run checks on normalized forms (homoglyphed, despaced, clean)
        for target_text in (norm["homoglyphed"], norm["despaced"], norm["clean"]):
            for rule in self.rules:
                match = rule.pattern.search(target_text)
                if match:
                    evidence = match.group(0)[:100]
                    if rule.rule_id not in matched_rule_ids:
                        matched_rule_ids.append(rule.rule_id)
                        detections.append(RuleDetectionDetail(
                            rule_id=rule.rule_id,
                            attack_category=rule.attack_category,
                            severity=rule.severity,
                            confidence=rule.confidence,
                            matched_evidence=evidence,
                        ))
                        max_pattern_confidence = max(max_pattern_confidence, rule.confidence)

        # Multi-rule hit boost
        confidence = max_pattern_confidence
        if len(detections) >= 2:
            confidence = min(1.0, confidence + 0.05)

        triggered = confidence >= self.trigger_threshold
        elapsed_ms = (time.perf_counter() - t0) * 1000

        top_rule_id = None
        top_cat = None
        top_sev = None
        if detections:
            detections.sort(key=lambda d: d.confidence, reverse=True)
            top_rule_id = detections[0].rule_id
            top_cat = detections[0].attack_category
            top_sev = detections[0].severity

        details_str = (
            f"rules={matched_rule_ids}; category={top_cat}; severity={top_sev}"
            if matched_rule_ids else "no rule signals"
        )

        return RuleEngineResult(
            triggered=triggered,
            confidence=round(confidence, 4),
            attack_category=top_cat,
            severity=top_sev,
            top_rule_id=top_rule_id,
            matched_rules=matched_rule_ids,
            keyword_signals=[],
            heuristic_score=0.0,
            processing_time_ms=round(elapsed_ms, 3),
            details=details_str,
            detections=detections,
        )

    def to_layer_dict(self, result: RuleEngineResult) -> dict[str, Any]:
        """Convert result to the LayerResult schema dict."""
        return {
            "layer": "rule_based",
            "triggered": result.triggered,
            "confidence": result.confidence,
            "attack_category": result.attack_category,
            "severity": result.severity,
            "top_rule_id": result.top_rule_id,
            "details": result.details,
            "matched_rules": result.matched_rules,
        }
