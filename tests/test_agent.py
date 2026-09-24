"""Hermetic unit tests for agent-tool-demo.

Everything runs offline against the mock planner and mock tools —
no network, no API keys, no randomness. Run with:

    python -m unittest discover -s tests -v
"""

import json
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import AgentRunner
from cost import BudgetExceeded, CostTracker, estimate_tokens
from guardrails import Guardrails, sanitize_output
from planner import FinalAnswer, MockPlanner, Thought, ToolCall
from tools import get_tool_specs

ALLOWED = ["flight_search", "calculator", "doc_lookup", "web_search_mock"]


def first(trace, kind, n=0):
    """n-th trace event of a given kind (avoids brittle hard indexes)."""
    hits = [e for e in trace if e["kind"] == kind]
    return hits[n]


def make_runner(script, **guard_kw):
    tools = get_tool_specs()
    kw = dict(allowed_tools=ALLOWED, max_steps=8, side_effect_policy="block")
    kw.update(guard_kw)
    return AgentRunner(MockPlanner(script), Guardrails(**kw),
                       CostTracker(budget_usd=0.05), tools)


class TestToolDispatch(unittest.TestCase):
    def test_flight_search_dispatch_and_sort(self):
        r = make_runner([
            ToolCall("flight_search", {"destination": "AUS", "date": "2026-12-18"}),
            FinalAnswer("done"),
        ]).run("t")
        self.assertEqual(r.status, "completed")
        self.assertEqual(r.tool_calls, 1)
        obs = json.loads(first(r.trace, "tool_call")["observation"])
        self.assertTrue(len(obs) >= 3)
        prices = [f["price_usd"] for f in obs]
        self.assertEqual(prices, sorted(prices))  # cheapest first

    def test_flight_search_filters(self):
        r = make_runner([
            ToolCall("flight_search", {"destination": "AUS", "date": "2026-12-18",
                                       "max_price_usd": 200.0, "nonstop_only": True}),
            FinalAnswer("done"),
        ]).run("t")
        obs = json.loads(first(r.trace, "tool_call")["observation"])
        self.assertEqual(len(obs), 1)
        self.assertEqual(obs[0]["flight_no"], "SK402")

    def test_calculator_safe_eval(self):
        r = make_runner([
            ToolCall("calculator", {"expression": "(189 + 32.5) * 2"}),
            FinalAnswer("done"),
        ]).run("t")
        obs = json.loads(first(r.trace, "tool_call")["observation"])
        self.assertAlmostEqual(obs["result"], 443.0)

    def test_calculator_rejects_code_injection(self):
        r = make_runner([
            ToolCall("calculator", {"expression": "__import__('os').system('x')"}),
            FinalAnswer("done"),
        ]).run("t")
        self.assertIn("TOOL ERROR", first(r.trace, "tool_call")["observation"])

    def test_doc_lookup(self):
        r = make_runner([
            ToolCall("doc_lookup", {"doc_id": "refund-policy"}),
            FinalAnswer("done"),
        ]).run("t")
        obs = json.loads(first(r.trace, "tool_call")["observation"])
        self.assertIn("48 hours", obs["text"])
        # planted email must be redacted by the sanitizer
        self.assertNotIn("support@skylark-mock.example", obs["text"])
        self.assertIn("[REDACTED-EMAIL]", obs["text"])

    def test_web_search_mock_labeled(self):
        r = make_runner([
            ToolCall("web_search_mock", {"query": "austin flights"}),
            FinalAnswer("done"),
        ]).run("t")
        obs = json.loads(first(r.trace, "tool_call")["observation"])
        self.assertTrue(obs["mock"])
        self.assertTrue(len(obs["results"]) > 0)


class TestGuardrails(unittest.TestCase):
    def test_unknown_tool_blocked(self):
        r = make_runner([ToolCall("nope_tool", {}), FinalAnswer("done")]).run("t")
        self.assertEqual(r.blocked_calls, 1)
        self.assertIn("unknown tool", first(r.trace, "blocked")["reason"])

    def test_non_allowlisted_tool_blocked(self):
        # send_email exists in the registry but is not in ALLOWED
        r = make_runner([
            ToolCall("send_email", {"to": "a@b.c", "subject": "s", "body": "b"}),
            FinalAnswer("done"),
        ]).run("t")
        self.assertEqual(r.blocked_calls, 1)
        self.assertIn("allowlist", first(r.trace, "blocked")["reason"])
        self.assertEqual(r.tool_calls, 0)  # never dispatched

    def test_side_effect_blocked_even_when_allowlisted(self):
        r = make_runner([
            ToolCall("send_email", {"to": "a@b.c", "subject": "s", "body": "b"}),
            FinalAnswer("done"),
        ], allowed_tools=ALLOWED + ["send_email"]).run("t")
        self.assertEqual(r.blocked_calls, 1)
        self.assertIn("side effects", first(r.trace, "blocked")["reason"])

    def test_side_effect_approve_mode_needs_approver(self):
        r = make_runner([
            ToolCall("send_email", {"to": "a@b.c", "subject": "s", "body": "b"}),
            FinalAnswer("done"),
        ], allowed_tools=ALLOWED + ["send_email"],
            side_effect_policy="approve").run("t")
        self.assertEqual(r.blocked_calls, 1)  # no approver -> denied
        self.assertIn("no approval", first(r.trace, "blocked")["reason"])

    def test_side_effect_approve_mode_with_approver(self):
        r = make_runner([
            ToolCall("send_email", {"to": "a@b.c", "subject": "s", "body": "b"}),
            FinalAnswer("done"),
        ], allowed_tools=ALLOWED + ["send_email"],
            side_effect_policy="approve",
            approver=lambda name, args: True).run("t")
        self.assertEqual(r.blocked_calls, 0)
        self.assertEqual(r.tool_calls, 1)
        obs = json.loads(first(r.trace, "tool_call")["observation"])
        self.assertEqual(obs["status"], "sent")

    def test_missing_required_arg_blocked(self):
        r = make_runner([
            ToolCall("flight_search", {"destination": "AUS"}),  # date missing
            FinalAnswer("done"),
        ]).run("t")
        self.assertEqual(r.blocked_calls, 1)
        self.assertIn("missing required argument 'date'", first(r.trace, "blocked")["reason"])

    def test_wrong_arg_type_blocked(self):
        r = make_runner([
            ToolCall("flight_search", {"destination": "AUS", "date": "2026-12-18",
                                       "nonstop_only": "yes"}),
            FinalAnswer("done"),
        ]).run("t")
        self.assertEqual(r.blocked_calls, 1)
        self.assertIn("must be bool", first(r.trace, "blocked")["reason"])

    def test_unexpected_arg_blocked(self):
        r = make_runner([
            ToolCall("doc_lookup", {"doc_id": "refund-policy", "hack": 1}),
            FinalAnswer("done"),
        ]).run("t")
        self.assertEqual(r.blocked_calls, 1)
        self.assertIn("unexpected argument", first(r.trace, "blocked")["reason"])

    def test_sanitize_output(self):
        s = sanitize_output("mail me at jane.doe@example.com or 555-123-4567, "
                            "card 4111 1111 1111 1111")
        self.assertIn("[REDACTED-EMAIL]", s)
        self.assertIn("[REDACTED-PHONE]", s)
        self.assertIn("[REDACTED-CARD]", s)
        self.assertNotIn("jane.doe", s)

    def test_blocked_run_still_completes(self):
        # A blocked call becomes an observation; the planner can recover.
        r = make_runner([
            ToolCall("send_email", {"to": "a@b.c", "subject": "s", "body": "b"}),
            ToolCall("doc_lookup", {"doc_id": "baggage-policy"}),
            FinalAnswer("recovered"),
        ]).run("t")
        self.assertEqual(r.status, "completed")
        self.assertEqual(r.final_answer, "recovered")
        self.assertEqual(r.blocked_calls, 1)
        self.assertEqual(r.tool_calls, 1)


class TestCostAndTermination(unittest.TestCase):
    def test_estimate_tokens_heuristic(self):
        self.assertEqual(estimate_tokens(""), 0)
        self.assertEqual(estimate_tokens("abcd"), 1)      # 4 chars -> 1
        self.assertEqual(estimate_tokens("abcde"), 2)     # rounds up

    def test_budget_cap_aborts_run(self):
        tools = get_tool_specs()
        runner = AgentRunner(
            MockPlanner([Thought("x" * 5000), FinalAnswer("done")]),
            Guardrails(allowed_tools=ALLOWED),
            CostTracker(budget_usd=0.000001),  # impossibly small
            tools)
        r = runner.run("t")
        self.assertEqual(r.status, "budget_exceeded")
        self.assertIn("budget", r.final_answer)

    def test_loop_cap_terminates(self):
        script = [Thought("still thinking...")] * 50  # never finalizes
        r = make_runner(script, max_steps=3).run("t")
        self.assertEqual(r.status, "max_steps_reached")
        self.assertEqual(len([e for e in r.trace if e["kind"] == "thought"]), 3)

    def test_cost_table_has_rows(self):
        r = make_runner([
            ToolCall("calculator", {"expression": "2+2"}),
            FinalAnswer("four"),
        ]).run("t")
        table = r.cost.table()
        self.assertIn("TOTAL", table)
        self.assertIn("budget", table)
        self.assertGreater(r.cost.total_tokens, 0)
        self.assertLessEqual(r.cost.total_cost_usd, 0.05)


class TestEndToEnd(unittest.TestCase):
    def test_full_task_run(self):
        script = [
            Thought("search flights"),
            ToolCall("flight_search", {"destination": "AUS", "date": "2026-12-18",
                                       "max_price_usd": 200.0, "nonstop_only": True}),
            Thought("math"),
            ToolCall("calculator", {"expression": "(189.00 + 32.50) * 2"}),
            Thought("policy"),
            ToolCall("doc_lookup", {"doc_id": "refund-policy"}),
            FinalAnswer("SK402 $443.00 total; cancel 48h ahead for full refund."),
        ]
        r = make_runner(script).run("find a cheap flight")
        self.assertEqual(r.status, "completed")
        self.assertEqual(r.tool_calls, 3)
        self.assertEqual(r.blocked_calls, 0)
        self.assertIn("SK402", r.final_answer)
        kinds = [e["kind"] for e in r.trace]
        self.assertEqual(kinds,
                         ["thought", "tool_call", "thought", "tool_call",
                          "thought", "tool_call", "final"])


if __name__ == "__main__":
    unittest.main()
