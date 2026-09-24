"""Guardrails: the rules the agent cannot break, checked before every tool call."""

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from tools import ToolSpec


@dataclass
class GuardrailDecision:
    """The verdict on one proposed tool call."""
    allowed: bool
    reason: str = ""


# Patterns redacted from tool outputs before the planner ever sees them.
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")
_CARD_RE = re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b")


def sanitize_output(text: str) -> str:
    """Redact emails, phone-like numbers, and card-like numbers.

    The planner (LLM) only ever sees the sanitized text — sensitive-looking
    strings from tool results can't leak into the final answer.
    """
    text = _EMAIL_RE.sub("[REDACTED-EMAIL]", text)
    text = _PHONE_RE.sub("[REDACTED-PHONE]", text)
    text = _CARD_RE.sub("[REDACTED-CARD]", text)
    return text


class Guardrails:
    """Four layers of defense around tool execution.

    1. Allowlist: only named tools may run at all.
    2. Loop cap: max_steps bounds how many planner turns a run may take.
    3. Argument validation: args must match the tool's declared contract.
    4. Side-effect policy: tools that change the world outside the process
       are blocked by default; in "approve" mode they need a human callback.
    """

    def __init__(
        self,
        allowed_tools: List[str],
        max_steps: int = 8,
        side_effect_policy: str = "block",  # "block" | "approve"
        approver: Optional[Callable[[str, Dict[str, Any]], bool]] = None,
        validate_args: bool = True,
    ):
        if side_effect_policy not in ("block", "approve"):
            raise ValueError("side_effect_policy must be 'block' or 'approve'")
        self.allowed_tools = set(allowed_tools)
        self.max_steps = max_steps
        self.side_effect_policy = side_effect_policy
        self.approver = approver
        self.validate_args = validate_args

    # -- argument validation -------------------------------------------------
    def _validate_args(self, spec: ToolSpec, args: Dict[str, Any]) -> Optional[str]:
        """Return an error string, or None if args are valid."""
        if not isinstance(args, dict):
            return "args must be a JSON object"
        for name, value in args.items():
            if name not in spec.params:
                return f"unexpected argument '{name}'"
        for name, (expected_type, required) in spec.params.items():
            if required and name not in args:
                return f"missing required argument '{name}'"
            if name in args:
                value = args[name]
                # accept ints where floats are declared (JSON numbers)
                if expected_type is float and isinstance(value, int):
                    continue
                if not isinstance(value, expected_type):
                    return (f"argument '{name}' must be "
                            f"{expected_type.__name__}, got {type(value).__name__}")
        return None

    # -- the check -----------------------------------------------------------
    def check_tool_call(self, name: str, args: Dict[str, Any],
                        registry: Dict[str, ToolSpec]) -> GuardrailDecision:
        if name not in registry:
            return GuardrailDecision(False, f"unknown tool '{name}' — not in registry")
        if name not in self.allowed_tools:
            return GuardrailDecision(
                False, f"tool '{name}' is not on the allowlist "
                       f"(allowed: {sorted(self.allowed_tools)})")
        spec = registry[name]
        if self.validate_args:
            err = self._validate_args(spec, args)
            if err:
                return GuardrailDecision(False, f"argument validation failed: {err}")
        if spec.side_effect:
            if self.side_effect_policy == "block":
                return GuardrailDecision(
                    False, f"tool '{name}' has side effects — blocked by policy "
                           f"(side_effect_policy='block')")
            # approve mode: a human (or a stub in tests) must say yes
            if self.approver is None or not self.approver(name, args):
                return GuardrailDecision(
                    False, f"tool '{name}' has side effects — no approval granted")
        return GuardrailDecision(True, "ok")
