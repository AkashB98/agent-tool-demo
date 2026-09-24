"""AgentRunner: the ReAct loop — reason, act, observe, repeat.

ReAct = Reasoning + Acting: the planner alternates between thinking out loud
(a Thought) and calling tools (a ToolCall), reading each tool's observation
before deciding what to do next, until it produces a FinalAnswer.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List

from cost import BudgetExceeded, CostTracker
from guardrails import Guardrails, sanitize_output
from planner import FinalAnswer, Planner, Thought, ToolCall
from tools import ToolSpec


@dataclass
class RunResult:
    task: str
    status: str  # completed | max_steps_reached | budget_exceeded
    final_answer: str
    trace: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls: int = 0
    blocked_calls: int = 0
    cost: CostTracker = None


class AgentRunner:
    def __init__(self, planner: Planner, guardrails: Guardrails,
                 cost: CostTracker, tools: List[ToolSpec]):
        self.planner = planner
        self.guardrails = guardrails
        self.cost = cost
        self.registry = {t.name: t for t in tools}

    def _prompt_text(self, task: str, history: List[Dict[str, Any]]) -> str:
        """What the planner "saw" this turn — used for token/cost accounting."""
        parts = [f"TASK: {task}"]
        for h in history:
            parts.append(f"STEP: {h['step']}")
            if h.get("observation"):
                parts.append(f"OBSERVATION: {h['observation']}")
        return "\n".join(parts)

    def run(self, task: str) -> RunResult:
        result = RunResult(task=task, status="completed",
                           final_answer="", cost=self.cost)
        history: List[Dict[str, Any]] = []
        try:
            for i in range(self.guardrails.max_steps):
                prompt = self._prompt_text(task, history)
                step = self.planner.next_step(task, history)

                if isinstance(step, Thought):
                    self.cost.record(f"plan-{i} thought", prompt, step.text)
                    result.trace.append({"kind": "thought", "text": step.text})
                    history.append({"step": f"thought: {step.text}",
                                    "observation": None})

                elif isinstance(step, ToolCall):
                    call_repr = json.dumps({"tool": step.name, "args": step.args})
                    decision = self.guardrails.check_tool_call(
                        step.name, step.args, self.registry)
                    if not decision.allowed:
                        result.blocked_calls += 1
                        observation = f"BLOCKED BY GUARDRAIL: {decision.reason}"
                        self.cost.record(f"plan-{i} blocked:{step.name}",
                                         prompt + call_repr, observation)
                        result.trace.append({"kind": "blocked", "tool": step.name,
                                             "reason": decision.reason})
                    else:
                        result.tool_calls += 1
                        try:
                            raw = self.registry[step.name].func(step.args)
                            observation = sanitize_output(json.dumps(raw))
                        except Exception as exc:
                            observation = f"TOOL ERROR: {exc}"
                        self.cost.record(f"plan-{i} tool:{step.name}",
                                         prompt + call_repr, observation)
                        result.trace.append({"kind": "tool_call", "tool": step.name,
                                             "args": step.args,
                                             "observation": observation})
                    history.append({"step": f"tool_call: {call_repr}",
                                    "observation": observation})

                elif isinstance(step, FinalAnswer):
                    self.cost.record(f"plan-{i} final", prompt, step.text)
                    result.trace.append({"kind": "final", "text": step.text})
                    result.final_answer = step.text
                    return result

                else:
                    result.trace.append({"kind": "error",
                                         "text": f"unknown step type: {step!r}"})
            # loop cap hit without a final answer
            result.status = "max_steps_reached"
            result.final_answer = ("(stopped: reached max_steps="
                                   f"{self.guardrails.max_steps} without a final answer)")
        except BudgetExceeded as exc:
            result.status = "budget_exceeded"
            result.final_answer = f"(stopped: {exc})"
        return result
