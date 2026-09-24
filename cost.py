"""Cost tracking: token estimates, per-step pricing, and a budget circuit breaker."""

import math
from dataclasses import dataclass, field
from typing import List


class BudgetExceeded(Exception):
    """Raised when a run's cumulative estimated cost passes the budget cap."""


def estimate_tokens(text: str) -> int:
    """Heuristic token estimate: ~4 characters per token (GPT-style tokenizers).

    A heuristic is not exact, but it is deterministic, dependency-free, and
    good enough for budget enforcement in a demo. Production would swap in
    the model's real tokenizer (e.g. tiktoken).
    """
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


@dataclass
class StepCost:
    label: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


class CostTracker:
    """Records per-step token usage and cost; aborts the run past budget.

    Prices default to GPT-4o-mini-class rates (per 1k tokens).
    """

    def __init__(self, input_per_1k: float = 0.0025,
                 output_per_1k: float = 0.01,
                 budget_usd: float = 0.05):
        self.input_per_1k = input_per_1k
        self.output_per_1k = output_per_1k
        self.budget_usd = budget_usd
        self.steps: List[StepCost] = []

    def record(self, label: str, input_text: str, output_text: str) -> StepCost:
        in_tok = estimate_tokens(input_text)
        out_tok = estimate_tokens(output_text)
        cost = (in_tok / 1000) * self.input_per_1k + (out_tok / 1000) * self.output_per_1k
        step = StepCost(label, in_tok, out_tok, cost)
        self.steps.append(step)
        if self.total_cost_usd > self.budget_usd:
            raise BudgetExceeded(
                f"budget ${self.budget_usd:.4f} exceeded after step "
                f"'{label}' (total ${self.total_cost_usd:.4f})")
        return step

    @property
    def total_tokens(self) -> int:
        return sum(s.input_tokens + s.output_tokens for s in self.steps)

    @property
    def total_cost_usd(self) -> float:
        return sum(s.cost_usd for s in self.steps)

    def table(self) -> str:
        lines = [f"{'step':<28} {'in_tok':>7} {'out_tok':>7} {'cost_usd':>9}",
                 "-" * 56]
        for s in self.steps:
            lines.append(f"{s.label:<28} {s.input_tokens:>7} {s.output_tokens:>7} "
                         f"${s.cost_usd:>8.5f}")
        lines.append("-" * 56)
        lines.append(f"{'TOTAL':<28} {self.total_tokens:>7} {'':>7} "
                     f"${self.total_cost_usd:>8.5f}")
        lines.append(f"budget: ${self.budget_usd:.4f}  "
                     f"({'OK' if self.total_cost_usd <= self.budget_usd else 'EXCEEDED'})")
        return "\n".join(lines)
