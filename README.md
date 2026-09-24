# agent-tool-demo

A **ReAct** agent loop — reasoning + acting: the agent thinks out loud, calls tools, reads each result, and repeats until it answers — with a real **tool-calling** interface (the LLM asks the system to run a function and gets the result back), layered **guardrails** (safety rules the agent cannot break), and **cost tracking** (an estimate of what each run spends in LLM fees).

The point of this repo is not the flight search. The point is the *scaffolding around the model*: most LLM demos skip the parts that decide whether an agent survives contact with production — who may call what, what happens when the model misbehaves, and how much each run costs. This repo makes those parts visible, testable, and cheap to experiment with.

Everything runs **hermetically** — self-contained, no network, no API keys — via a scripted mock planner, so `python demo.py` works on a fresh machine in seconds. Swap in a real LLM any time with one env var (details below).

> **A note on the lingo:** this README uses real industry terms, and every term gets a one-line plain-English explanation the first time it appears, formatted `**term** — plain-English meaning.` Read the term, read the line, and it sticks.

## Demo (one command)

```bash
python demo.py
```

That's it — stdlib only, no installs, no keys. You'll see two runs:

1. **Task completion** — *"Find the cheapest nonstop Skylark flight to Austin on 2026-12-18 under $200, compute the total for 2 passengers, and quote the cancellation window."* The agent calls `flight_search` → `calculator` → `doc_lookup`, and each step shows its **token** usage (a token is roughly ¾ of a word — the unit LLMs bill by) and cost.
2. **Guardrail block** — *"Email me the flight details."* The planner reaches for `send_email`; the guardrails block it (not on the **allowlist** — an explicit list of tools the agent is permitted to use — and side-effect tools are disabled by policy). The run still completes gracefully with an explanation.

Run the tests:

```bash
python -m unittest discover -s tests -v
```

## Architecture

```
                    ┌──────────────┐
                    │   planner    │  Thought | ToolCall | FinalAnswer
                    │ (mock script │  (mock = deterministic demo brain;
                    │  or real LLM)│   LLM = OpenAI-compatible endpoint)
                    └──────┬───────┘
                           │ next_step(task, history)
                           ▼
                    ┌──────────────┐
                    │ AgentRunner  │  the ReAct loop:
                    │  (agent.py)  │  reason → act → observe → repeat
                    └──────┬───────┘
              ┌────────────┼──────────────┐
              ▼            ▼              ▼
       ┌────────────┐ ┌─────────┐ ┌──────────────┐
       │ Guardrails │ │  tools  │ │ CostTracker  │
       │(guardrails │ │(tools.py│ │  (cost.py)   │
       │   .py)     │ │  )      │ │              │
       │ allowlist, │ │ flight_ │ │ token        │
       │ arg check, │ │ search, │ │ heuristic    │
       │ side-effect│ │ calcula-│ │ (~4 chars/   │
       │ policy,    │ │ tor,    │ │  token),     │
       │ output     │ │ doc_    │ │ per-step     │
       │ sanitize   │ │ lookup, │ │ pricing,     │
       └────────────┘ │ web mock│ │ budget abort │
                      │ (+send_ │ └──────────────┘
                      │  email, │
                      │  blocked)│
                      └─────────┘
```

**The loop, step by step:**

1. The planner emits a `Thought` (shown in the trace), a `ToolCall`, or a `FinalAnswer`.
2. Every `ToolCall` goes through the guardrails *before* dispatch:
   - **Allowlist** — unknown or unlisted tools are rejected.
   - **Argument validation** — args must match the tool's declared contract (required params, types, no surprise extras).
   - **Side-effect policy** — tools that change the world (send, charge, delete) are `block`ed by default; in `approve` mode they need a human callback's yes.
3. Tool output passes through the **output sanitizer** (emails, phone-like and card-like strings → `[REDACTED-…]`) before the planner sees it.
4. Each step's **prompt** (everything the planner saw) and output are token-counted and priced; if the cumulative cost passes the budget, a `BudgetExceeded` aborts the run.
5. The loop ends on `FinalAnswer`, on the **max-steps cap** (prevents infinite loops), or on budget abort.

## Tools (all mock, all sample data)

| tool | what it does | side effect? |
|---|---|---|
| `flight_search` | searches a fictional in-memory flight inventory (fictional airlines "Skylark Air", "Blue Mesa"; invented prices) | no |
| `calculator` | evaluates arithmetic via AST — abstract syntax tree: the expression is parsed into a structure, and only numbers/operators are allowed, so `__import__('os')` can't sneak through | no |
| `doc_lookup` | fetches fictional policy docs (`refund-policy`, `baggage-policy`) — the refund doc has a planted support email so you can watch the sanitizer redact it | no |
| `web_search_mock` | canned results from a tiny keyword index, every result labeled `mock` | no |
| `send_email` | exists in the registry *only* to be blocked — proves the allowlist + side-effect policy work | **yes** |

## Sample run transcript (abridged)

```
TASK: Find the cheapest nonstop Skylark flight to Austin on 2026-12-18
under $200, compute the all-in total for 2 passengers, and quote the
cancellation window from the refund policy.

🧠 THOUGHT: ...First, search the flight inventory.

🔧 TOOL CALL: flight_search({'destination': 'AUS', 'date': '2026-12-18',
                             'max_price_usd': 200.0, 'nonstop_only': True})
   ↳ [{"airline": "Skylark Air", "flight_no": "SK402", ... "price_usd": 189.0,
        "fees_usd": 32.5}]

🧠 THOUGHT: SK402 at $189.00 is the cheapest match. Now compute the all-in
   total for 2 passengers: (price + fees) * 2.

🔧 TOOL CALL: calculator({'expression': '(189.00 + 32.50) * 2'})
   ↳ {"expression": "(189.00 + 32.50) * 2", "result": 443.0}

🔧 TOOL CALL: doc_lookup({'doc_id': 'refund-policy'})
   ↳ {"doc_id": "refund-policy", "text": "SKYLARK AIR — REFUND POLICY
       (MOCK DOCUMENT) ... Contact [REDACTED-EMAIL] or call
       [REDACTED-PHONE]."}

✅ FINAL ANSWER:
   Cheapest qualifying flight: Skylark Air SK402, DFW → AUS on 2026-12-18,
   nonstop — $189.00 + $32.50 fees = $221.50/person, $443.00 all-in for 2.
   Cancellation window: full refund if you cancel at least 48 hours before
   departure; $50 fee within 48 hours; no-shows non-refundable.
   (All flight/doc data above is fictional sample data.)

status: completed | tool calls: 3 | blocked: 0

COST SUMMARY (heuristic tokens, ~4 chars/token):
step                          in_tok out_tok  cost_usd
--------------------------------------------------------
plan-0 thought                    46      41 $ 0.00053
plan-1 tool:flight_search        122      42 $ 0.00072
plan-2 thought                   172      27 $ 0.00070
plan-3 tool:calculator           220      14 $ 0.00069
plan-4 thought                   242      27 $ 0.00084
plan-5 tool:doc_lookup           284      76 $ 0.00147
plan-6 final                     368      92 $ 0.00184
--------------------------------------------------------
TOTAL                           1770         $ 0.00679
budget: $0.0500  (OK)
```

And the guardrail run:

```
TASK: Email me the flight details for SK402.

🧠 THOUGHT: The user wants the SK402 details emailed. I'll use send_email.

🛡️  GUARDRAIL BLOCK: send_email
   ↳ reason: tool 'send_email' is not on the allowlist
     (allowed: ['calculator', 'doc_lookup', 'flight_search', 'web_search_mock'])

✅ FINAL ANSWER:
   I can't send emails: the send_email tool was blocked by the guardrails...
```

## Test results

21 hermetic unit tests (`python -m unittest discover -s tests`), all passing:

| area | tests | what they prove |
|---|---|---|
| tool dispatch | 6 | flight search sorts/filters correctly; calculator evaluates math and rejects code injection (`__import__('os')` → `TOOL ERROR`); doc lookup returns docs with PII redacted; mock search labels results |
| guardrails | 10 | unknown tools blocked; non-allowlisted tools blocked; side-effect tools blocked even when allowlisted; `approve` mode denies without a human callback and allows with one; missing/wrong/extra args blocked; output sanitizer redacts emails/phones/cards; a blocked run still completes gracefully |
| cost & termination | 4 | token heuristic behaves; a tiny budget aborts the run (`budget_exceeded`); a planner that never finalizes hits the loop cap (`max_steps_reached`); the cost table balances |
| end-to-end | 1 | full 3-tool task completes with the expected trace shape and answer |

The tests caught one real bug during development: the price filter compared price+fees against `max_price_usd`, which wrongly excluded the $189 fare. The filter now compares base fare, with fees shown separately.

## Using a real LLM (optional)

The mock planner is the default so the demo never needs a key. To drive the loop with a real model:

```bash
PLANNER=openai AGENT_DEMO_API_KEY=sk-... python demo.py
```

- **AGENT_DEMO_API_KEY** — required; read from the environment only, never committed.
- **AGENT_DEMO_BASE_URL** — optional, default `https://api.openai.com/v1` (any OpenAI-compatible endpoint works: OpenAI, Azure, Ollama, LM Studio…).
- **AGENT_DEMO_MODEL** — optional, default `gpt-4o-mini`.

Guardrails, sanitization, and cost tracking apply identically — that's the point: the safety layer doesn't care which brain is driving.

## What you'd wire in production

This demo is honest about what it is — a scaffold. To take it live:

- **Real planner output parsing** — production ReAct uses the provider's native **function calling** (the API's structured way of asking the model to emit a tool call, instead of parsing JSON out of free text) with schema validation on the model's arguments.
- **Real tokenizer** — swap the 4-chars/token heuristic for the model's actual tokenizer (e.g. `tiktoken`) for exact billing.
- **Persistent tool servers** — tools become services (search APIs, databases) behind auth, retries, timeouts, and **rate limits** (caps on how many calls per second/minute, so one run can't hammer a downstream system).
- **Human-in-the-loop approvals** — the `approve` policy's stub callback becomes a real approval queue (Slack / UI) with audit logging of every allowed side effect.
- **Evals** — a golden set of tasks scored on success rate, guardrail-block precision/recall, and cost per task — the same eval-harness discipline as the companion `rag-chatbot-evals` project.
- **Tracing** — export the per-step trace to OpenTelemetry so runs are debuggable in production.

## Why this project (the hiring pitch)

RAG proves you can ground a chatbot. *This* proves you can ship an **agent** — the thing every AI lab is actually hiring **forward deployed engineers** (engineers embedded with customers, turning AI into working systems) to build: tool use under policy, failure modes handled, costs controlled, and a test suite that proves the safety layer works instead of promising it. A hiring manager reading the guardrail tests sees someone who has thought about what happens when the model does the wrong thing — which is exactly the interview conversation FDE loops are built around.
