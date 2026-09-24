"""Planners: the "brain" that decides the next step of the ReAct loop.

Two implementations behind one interface:
- MockPlanner: deterministic, scripted, zero-network. The demo and tests run
  on this — fully hermetic, no API key.
- OpenAIPlanner: optional; talks to any OpenAI-compatible chat endpoint
  (OpenAI, or a local server like Ollama/LM Studio). The API key comes ONLY
  from the AGENT_DEMO_API_KEY environment variable — never from code.
"""

import json
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Step types the planner can emit
# ---------------------------------------------------------------------------

@dataclass
class Thought:
    """The planner's internal reasoning, shown in the trace."""
    text: str


@dataclass
class ToolCall:
    """A request to run a tool."""
    name: str
    args: Dict[str, Any]


@dataclass
class FinalAnswer:
    """The planner is done; this is the answer to the user."""
    text: str


class Planner:
    def next_step(self, task: str, history: List[Dict[str, Any]]):
        """Return the next Thought | ToolCall | FinalAnswer."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Mock planner — scripted, deterministic, hermetic
# ---------------------------------------------------------------------------

class MockPlanner(Planner):
    """Plays back a canned script of steps, then ends with a final answer.

    This is how the demo runs with zero API keys: the "reasoning" is a fixed
    script so every run is reproducible and every guardrail path is exercised
    on purpose. The loop machinery (dispatch, guardrails, costs) is all real.
    """

    def __init__(self, script: List):
        self._script = list(script)

    def next_step(self, task, history):
        if not self._script:
            return FinalAnswer("(script exhausted — ending run)")
        return self._script.pop(0)


# ---------------------------------------------------------------------------
# OpenAI-compatible planner — optional, env-key only
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a ReAct agent. On each turn output EXACTLY one JSON object:
{"type": "thought", "text": "..."} to reason,
{"type": "tool_call", "name": "<tool>", "args": {...}} to use a tool,
{"type": "final", "text": "..."} to answer the user and stop.

Available tools (mock data, sample only):
{tools}

Rules: never invent tool results; call tools for facts. If a tool call is
blocked, do NOT retry it — work around it or explain. Keep answers short."""


class OpenAIPlanner(Planner):
    """Uses a real LLM via an OpenAI-compatible /chat/completions endpoint.

    Env vars: AGENT_DEMO_API_KEY (required), AGENT_DEMO_BASE_URL
    (default https://api.openai.com/v1), AGENT_DEMO_MODEL (default gpt-4o-mini).
    """

    def __init__(self, tool_schemas: List[Dict[str, Any]]):
        key = os.environ.get("AGENT_DEMO_API_KEY", "")
        if not key:
            raise RuntimeError(
                "AGENT_DEMO_API_KEY is not set — refusing to run the LLM planner "
                "without an explicit key in the environment.")
        self._key = key
        self._base = os.environ.get("AGENT_DEMO_BASE_URL", "https://api.openai.com/v1")
        self._model = os.environ.get("AGENT_DEMO_MODEL", "gpt-4o-mini")
        tools_desc = "\n".join(
            f"- {t['name']}{t['params']}: {t['description']}"
            for t in tool_schemas)
        self._system = _SYSTEM_PROMPT.format(tools=tools_desc)

    def next_step(self, task, history):
        messages = [{"role": "system", "content": self._system},
                    {"role": "user", "content": task}]
        for h in history:
            messages.append({"role": "assistant", "content": h.get("step", "")})
            if h.get("observation") is not None:
                messages.append({"role": "user",
                                 "content": f"TOOL RESULT: {h['observation']}"})
        body = json.dumps({"model": self._model, "messages": messages,
                           "response_format": {"type": "json_object"},
                           "temperature": 0}).encode()
        req = urllib.request.Request(
            self._base.rstrip("/") + "/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self._key}"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = json.load(resp)
        except Exception as exc:
            raise RuntimeError(f"LLM planner request failed: {exc}")
        raw = payload["choices"][0]["message"]["content"]
        try:
            step = json.loads(raw)
        except json.JSONDecodeError:
            raise RuntimeError(f"LLM planner returned non-JSON: {raw[:200]}")
        kind = step.get("type")
        if kind == "thought":
            return Thought(step.get("text", ""))
        if kind == "tool_call":
            return ToolCall(step.get("name", ""), step.get("args", {}) or {})
        if kind == "final":
            return FinalAnswer(step.get("text", ""))
        raise RuntimeError(f"LLM planner returned unknown step type: {kind!r}")
