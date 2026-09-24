"""Demo: two visible agentic runs with the mock planner (no API key needed).

Run 1 — the happy path: flight search + price math + policy doc lookup,
        with per-step tool calls, guardrail verdicts, and a cost table.
Run 2 — the guardrail showcase: the planner tries a side-effect tool
        (send_email), which is blocked twice over (not on the allowlist AND
        side-effect policy). The run still completes gracefully.

Optional: PLANNER=openai AGENT_DEMO_API_KEY=... python demo.py
          uses a real LLM instead of the scripted planner.
"""

import os
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import AgentRunner
from cost import CostTracker
from guardrails import Guardrails
from planner import FinalAnswer, MockPlanner, OpenAIPlanner, Thought, ToolCall
from tools import get_tool_specs

TASK_1 = ("Find the cheapest nonstop Skylark flight to Austin on 2026-12-18 "
          "under $200, compute the all-in total for 2 passengers, "
          "and quote the cancellation window from the refund policy.")

PLAN_1 = [
    Thought("The user wants the cheapest nonstop Skylark flight to Austin on "
            "2026-12-18 under $200, plus a price total and the refund policy. "
            "First, search the flight inventory."),
    ToolCall("flight_search", {"destination": "AUS", "date": "2026-12-18",
                               "max_price_usd": 200.0, "nonstop_only": True}),
    Thought("SK402 at $189.00 is the cheapest match. Now compute the all-in "
            "total for 2 passengers: (price + fees) * 2."),
    ToolCall("calculator", {"expression": "(189.00 + 32.50) * 2"}),
    Thought("Total is $443.00 for two. Now fetch the refund policy document "
            "to quote the cancellation window."),
    ToolCall("doc_lookup", {"doc_id": "refund-policy"}),
    FinalAnswer(
        "Cheapest qualifying flight: Skylark Air SK402, DFW → AUS on "
        "2026-12-18, nonstop — $189.00 + $32.50 fees = $221.50/person, "
        "$443.00 all-in for 2 passengers.\n"
        "Cancellation window (per refund-policy doc): full refund if you "
        "cancel at least 48 hours before departure; $50 fee within 48 hours; "
        "no-shows non-refundable.\n"
        "(All flight/doc data above is fictional sample data.)"),
]

TASK_2 = "Email me the flight details for SK402."

PLAN_2 = [
    Thought("The user wants the SK402 details emailed. I'll use send_email."),
    ToolCall("send_email", {"to": "user@example.com",
                            "subject": "SK402 flight details",
                            "body": "Skylark Air SK402 DFW->AUS $189.00"}),
    FinalAnswer(
        "I can't send emails: the send_email tool was blocked by the "
        "guardrails (it's not on the tool allowlist, and side-effect tools "
        "are disabled by policy). I can show you the details here instead: "
        "Skylark Air SK402, DFW → AUS, 2026-12-18, $189.00 + $32.50 fees. "
        "(Sample data.)"),
]

ALLOWED = ["flight_search", "calculator", "doc_lookup", "web_search_mock"]


def make_runner(script):
    tools = get_tool_specs()
    if os.environ.get("PLANNER") == "openai":
        planner = OpenAIPlanner(
            [{"name": t.name, "params": list(t.params),
              "description": t.description} for t in tools])
    else:
        planner = MockPlanner(script)
    guardrails = Guardrails(allowed_tools=ALLOWED, max_steps=8,
                            side_effect_policy="block")
    cost = CostTracker(budget_usd=0.05)
    return AgentRunner(planner, guardrails, cost, tools)


def show(title: str, task: str, script):
    print("=" * 72)
    print(title)
    print("=" * 72)
    print(f"TASK: {task}\n")
    result = make_runner(script).run(task)
    for ev in result.trace:
        kind = ev["kind"]
        if kind == "thought":
            print(f"🧠 THOUGHT: {textwrap.fill(ev['text'], 70, subsequent_indent='   ')}\n")
        elif kind == "tool_call":
            print(f"🔧 TOOL CALL: {ev['tool']}({ev['args']})")
            obs = ev["observation"]
            print(f"   ↳ {textwrap.fill(obs, 66, subsequent_indent='     ')}\n")
        elif kind == "blocked":
            print(f"🛡️  GUARDRAIL BLOCK: {ev['tool']}")
            print(f"   ↳ reason: {ev['reason']}\n")
        elif kind == "final":
            print(f"✅ FINAL ANSWER:\n{textwrap.indent(ev['text'], '   ')}\n")
        else:
            print(f"⚠️  {ev}\n")
    print(f"status: {result.status} | tool calls: {result.tool_calls} | "
          f"blocked: {result.blocked_calls}")
    print("\nCOST SUMMARY (heuristic tokens, ~4 chars/token):")
    print(result.cost.table())
    print()


if __name__ == "__main__":
    mode = "mock planner (hermetic, no API key)" \
        if os.environ.get("PLANNER") != "openai" else "OpenAI-compatible LLM planner"
    print(f"\nagent-tool-demo — running with {mode}\n")
    show("RUN 1 — task completion with tool calling", TASK_1, PLAN_1)
    show("RUN 2 — guardrail block on a side-effect tool", TASK_2, PLAN_2)
