"""
PromptShield — Secure LLM Gateway (Phase 7)
===========================================

Production-grade Security Gateway controlling whether user prompts are allowed to
reach external or local LLM providers.

Security Policy Enforced:
-------------------------
1. ALLOW    -> Forward original prompt to LLM. Return LLM response. (llm_called=True)
2. WARN     -> Forward original prompt to LLM with recorded audit warning. (llm_called=True)
3. SANITIZE -> NEVER directly forward original prompt. Sanitize & re-analyze.
               - Re-analysis passes (risk <= 49): Forward sanitized prompt. (llm_called=True)
               - Re-analysis fails (risk >= 50): BLOCK request. (llm_called=False, response=None)
4. BLOCK    -> NEVER send prompt to LLM. Return security response. (llm_called=False, response=None)

HARD SECURITY GUARANTEE:
Blocked prompts NEVER reach the LLM provider.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from backend.services.detection_service import DetectionService
from backend.services.sanitizer import PromptSanitizerService
from backend.services.llm_provider import BaseLLMProvider, get_llm_provider

log = logging.getLogger("promptshield.llm_gateway")
sec_log = logging.getLogger("promptshield.security_audit")


@dataclass
class GatewayResponse:
    request_id: str
    original_prompt: str
    final_prompt: str
    decision: str                     # ALLOW | WARN | SANITIZE | BLOCK
    risk_score: int                   # 0 to 100
    was_sanitized: bool
    sanitization_reason: str | None
    blocked_parts: list[str] = field(default_factory=list)
    llm_called: bool = False
    llm_provider: str = "mock"
    response: str | None = None
    processing_time_ms: float = 0.0
    can_sanitize: bool = False
    sanitized_prompt: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "original_prompt": self.original_prompt,
            "final_prompt": self.final_prompt,
            "decision": self.decision,
            "risk_score": self.risk_score,
            "was_sanitized": self.was_sanitized,
            "sanitization_reason": self.sanitization_reason,
            "blocked_parts": self.blocked_parts,
            "llm_called": self.llm_called,
            "llm_provider": self.llm_provider,
            "response": self.response,
            "processing_time_ms": self.processing_time_ms,
            "can_sanitize": self.can_sanitize,
            "sanitized_prompt": self.sanitized_prompt,
        }



class SecureLLMGateway:
    """
    Secure LLM Gateway controlling execution flow to LLM providers.
    """

    def __init__(
        self,
        detection_service: DetectionService | None = None,
        sanitizer_service: PromptSanitizerService | None = None,
        llm_provider: BaseLLMProvider | None = None,
    ) -> None:
        self.detection_service = detection_service or DetectionService()
        self.sanitizer_service = sanitizer_service or PromptSanitizerService(detection_service=self.detection_service)
        self.llm_provider = llm_provider or get_llm_provider()

    def _log_security_audit(
        self,
        request_id: str,
        decision: str,
        risk_score: int,
        orig_len: int,
        final_len: int,
        was_sanitized: bool,
        llm_called: bool,
        latency_ms: float,
    ) -> None:
        """
        Record structured security audit log.
        NOTE: Sensitive prompt contents are NOT logged by default.
        """
        audit_entry = {
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "decision": decision,
            "risk_score": risk_score,
            "original_prompt_length": orig_len,
            "final_prompt_length": final_len,
            "was_sanitized": was_sanitized,
            "llm_called": llm_called,
            "processing_latency_ms": latency_ms,
        }
        sec_log.info(f"[SECURITY AUDIT] {audit_entry}")

    async def process_chat(
        self,
        prompt: str,
        request_id: str | None = None,
        auto_sanitize: bool = False,
    ) -> GatewayResponse:
        """
        Process a user prompt through the complete security gateway pipeline.
        """
        t0 = time.perf_counter()
        req_id = request_id or f"req-{uuid.uuid4().hex[:12]}"
        orig_prompt = prompt.strip()

        # Step 1: Initial 3-layer detection & decision engine pass
        det_result = await self.detection_service.analyse(orig_prompt)

        initial_decision = det_result.get("decision", "ALLOW")
        initial_score = int(det_result.get("final_score", 0))

        final_prompt = orig_prompt
        was_sanitized = False
        sanitization_reason = None
        blocked_parts: list[str] = []
        final_decision = initial_decision
        final_score = initial_score
        can_sanitize = False
        sanitized_prompt_candidate: str | None = None

        # Step 2: Route according to decision
        if initial_decision in ("ALLOW", "WARN"):
            # Direct execution path
            pass

        elif initial_decision in ("SANITIZE", "BLOCK"):
            # Run sanitization verification to see if the malicious parts can be neutralized
            san_result = await self.sanitizer_service.sanitize_and_verify(orig_prompt, initial_detection=det_result)
            is_viable = (
                san_result.was_modified
                and bool(san_result.sanitized_prompt)
                and not san_result.sanitized_prompt.startswith("[CONTENT_REMOVED")
                and san_result.risk_after <= 49
            )

            if is_viable:
                can_sanitize = True
                sanitized_prompt_candidate = san_result.sanitized_prompt

            if auto_sanitize and is_viable:
                # AUTO-SANITIZE & PROCEED: User or system enabled auto-sanitization
                was_sanitized = True
                final_prompt = san_result.sanitized_prompt
                sanitization_reason = "Auto-sanitized: Removed malicious injection tokens. Safe prompt forwarded to LLM."
                blocked_parts = san_result.removed_segments
                final_score = san_result.risk_after
                final_decision = san_result.final_decision  # ALLOW or WARN
            elif initial_decision == "SANITIZE":
                was_sanitized = san_result.was_modified
                final_prompt = san_result.sanitized_prompt
                sanitization_reason = san_result.sanitization_reason
                blocked_parts = san_result.removed_segments
                final_score = san_result.risk_after
                final_decision = san_result.final_decision
            else:
                # BLOCK without auto_sanitize
                final_decision = "BLOCK"
                sanitization_reason = "Prompt contains injection instructions and was blocked."
                blocked_parts = san_result.removed_segments if (is_viable and san_result.removed_segments) else [orig_prompt]

        # Step 3: Enforcement — Should the LLM provider be called?
        llm_called = False
        llm_response_text: str | None = None
        provider_name = getattr(self.llm_provider, "model_name", "mock")

        if final_decision in ("ALLOW", "WARN"):
            # PERMITTED EXECUTION: Forward to LLM Provider
            llm_called = True
            llm_out = await self.llm_provider.generate(final_prompt)
            llm_response_text = llm_out.get("response", "")
            provider_name = llm_out.get("provider", provider_name)

        else:  # BLOCK
            # HARD SECURITY REQUIREMENT: BLOCKED prompts NEVER reach the LLM provider!
            llm_called = False
            llm_response_text = None

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

        # Step 4: Structured Security Audit Logging
        self._log_security_audit(
            request_id=req_id,
            decision=final_decision,
            risk_score=final_score,
            orig_len=len(orig_prompt),
            final_len=len(final_prompt),
            was_sanitized=was_sanitized,
            llm_called=llm_called,
            latency_ms=elapsed_ms,
        )

        return GatewayResponse(
            request_id=req_id,
            original_prompt=orig_prompt,
            final_prompt=final_prompt,
            decision=final_decision,
            risk_score=final_score,
            was_sanitized=was_sanitized,
            sanitization_reason=sanitization_reason,
            blocked_parts=blocked_parts,
            llm_called=llm_called,
            llm_provider=provider_name,
            response=llm_response_text,
            processing_time_ms=elapsed_ms,
            can_sanitize=can_sanitize,
            sanitized_prompt=sanitized_prompt_candidate,
        )

