"""Sample tool registry for the agent demo.

Every tool here operates on FICTIONAL, hard-coded sample data — marked MOCK
where visible. Nothing touches the network, a database, or the user's data.
"""

import ast
import operator
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

# ---------------------------------------------------------------------------
# Tool specification
# ---------------------------------------------------------------------------

@dataclass
class ToolSpec:
    """A callable tool the agent may invoke.

    params: mapping of param name -> (expected python type, required?)
    side_effect: True if calling the tool changes the world outside this
                 process (sends something, charges something, deletes
                 something). Side-effect tools get extra guardrail scrutiny.
    """
    name: str
    description: str
    params: Dict[str, tuple]
    func: Callable[[Dict[str, Any]], Any]
    side_effect: bool = False


# ---------------------------------------------------------------------------
# MOCK DATA — fictional, invented for the demo. Not real flights, prices,
# documents, or search results.
# ---------------------------------------------------------------------------

# MOCK flight inventory (fictional airlines "Skylark Air" and "Blue Mesa").
_FLIGHTS = [
    {"airline": "Skylark Air", "flight_no": "SK402", "origin": "DFW", "destination": "AUS",
     "date": "2026-12-18", "nonstop": True, "price_usd": 189.00, "fees_usd": 32.50},
    {"airline": "Skylark Air", "flight_no": "SK410", "origin": "DFW", "destination": "AUS",
     "date": "2026-12-18", "nonstop": True, "price_usd": 214.00, "fees_usd": 32.50},
    {"airline": "Blue Mesa", "flight_no": "BM77", "origin": "DFW", "destination": "AUS",
     "date": "2026-12-18", "nonstop": False, "price_usd": 129.00, "fees_usd": 28.00},
    {"airline": "Skylark Air", "flight_no": "SK455", "origin": "DFW", "destination": "AUS",
     "date": "2026-12-19", "nonstop": True, "price_usd": 172.00, "fees_usd": 32.50},
    {"airline": "Blue Mesa", "flight_no": "BM12", "origin": "DFW", "destination": "DEN",
     "date": "2026-12-18", "nonstop": True, "price_usd": 149.00, "fees_usd": 30.00},
]

# MOCK document store (fictional policies). Note: the refund policy contains a
# planted support email address — the output sanitizer redacts it, on purpose,
# so the demo visibly shows sanitization working.
_DOCS = {
    "refund-policy": (
        "SKYLARK AIR — REFUND POLICY (MOCK DOCUMENT)\n"
        "Full refund if you cancel at least 48 hours before departure. "
        "Cancellations within 48 hours of departure incur a $50 fee. "
        "No-shows are non-refundable. "
        "Questions? Contact support@skylark-mock.example or call 555-010-2030."
    ),
    "baggage-policy": (
        "SKYLARK AIR — BAGGAGE POLICY (MOCK DOCUMENT)\n"
        "First checked bag: $35. Second checked bag: $45. Carry-on: free. "
        "Checked bags must weigh 50 lbs or less; overweight bags cost $75 extra."
    ),
}

# MOCK web search index: keyword -> canned results.
_SEARCH_INDEX = {
    "austin": [
        {"title": "Visit Austin — official travel guide (MOCK RESULT)",
         "url": "https://example.com/austin-guide",
         "snippet": "Sample snippet: live music, food trucks, and the Capitol. MOCK DATA."},
        {"title": "AUS airport parking rates (MOCK RESULT)",
         "url": "https://example.com/aus-parking",
         "snippet": "Sample snippet: economy parking $9/day. MOCK DATA."},
    ],
    "refund": [
        {"title": "Skylark Air refunds help page (MOCK RESULT)",
         "url": "https://example.com/skylark-refunds",
         "snippet": "Sample snippet: refunds post in 5-7 business days. MOCK DATA."},
    ],
}


# ---------------------------------------------------------------------------
# Tool implementations (pure functions over the mock data)
# ---------------------------------------------------------------------------

def _flight_search(args: Dict[str, Any]) -> List[Dict[str, Any]]:
    dest = args["destination"].upper()
    date = args["date"]
    max_price = args.get("max_price_usd")
    nonstop_only = args.get("nonstop_only", False)
    matches = [f for f in _FLIGHTS if f["destination"] == dest and f["date"] == date]
    if nonstop_only:
        matches = [f for f in matches if f["nonstop"]]
    if max_price is not None:
        # Compare against the base fare (fees are shown separately in results).
        matches = [f for f in matches if f["price_usd"] <= max_price]
    return sorted(matches, key=lambda f: f["price_usd"])


_SAFE_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _safe_eval(node):
    """Evaluate an arithmetic AST — numbers and operators only, no names."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("expression may only contain numbers and + - * / // % **")


def _calculator(args: Dict[str, Any]) -> Dict[str, Any]:
    expr = args["expression"]
    if len(expr) > 200:
        raise ValueError("expression too long")
    try:
        result = _safe_eval(ast.parse(expr, mode="eval"))
    except (SyntaxError, ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"invalid arithmetic expression: {exc}")
    return {"expression": expr, "result": round(float(result), 2)}


def _doc_lookup(args: Dict[str, Any]) -> Dict[str, Any]:
    doc_id = args["doc_id"]
    if doc_id not in _DOCS:
        raise KeyError(f"unknown doc_id '{doc_id}'. Known: {sorted(_DOCS)}")
    return {"doc_id": doc_id, "text": _DOCS[doc_id]}


def _web_search_mock(args: Dict[str, Any]) -> Dict[str, Any]:
    query = args["query"].lower()
    top_k = args.get("top_k", 3)
    hits = []
    for keyword, results in _SEARCH_INDEX.items():
        if keyword in query:
            hits.extend(results)
    return {"query": args["query"], "mock": True,
            "results": hits[:top_k] or
            [{"title": "(no mock results for this query)",
              "url": "https://example.com/",
              "snippet": "Try 'austin' or 'refund' — this is a canned mock index."}]}


def _send_email(args: Dict[str, Any]) -> Dict[str, Any]:
    # Intentionally unreachable in the default demo: the guardrail layer
    # blocks side-effect tools before dispatch. Exists to prove the block.
    return {"status": "sent", "to": args["to"], "subject": args["subject"]}


def get_tool_specs() -> List[ToolSpec]:
    """All tools, with machine-readable parameter contracts."""
    return [
        ToolSpec(
            name="flight_search",
            description="Search MOCK flight inventory. Returns matching flights "
                        "sorted by price. All data fictional.",
            params={"destination": (str, True), "date": (str, True),
                    "max_price_usd": (float, False), "nonstop_only": (bool, False)},
            func=_flight_search,
        ),
        ToolSpec(
            name="calculator",
            description="Evaluate an arithmetic expression safely "
                        "(numbers and + - * / // % ** only).",
            params={"expression": (str, True)},
            func=_calculator,
        ),
        ToolSpec(
            name="doc_lookup",
            description="Fetch a MOCK policy document by id. Known ids: "
                        "refund-policy, baggage-policy.",
            params={"doc_id": (str, True)},
            func=_doc_lookup,
        ),
        ToolSpec(
            name="web_search_mock",
            description="MOCK web search over a tiny canned index. "
                        "Results are labeled mock and fictional.",
            params={"query": (str, True), "top_k": (int, False)},
            func=_web_search_mock,
        ),
        ToolSpec(
            name="send_email",
            description="Send an email. SIDE EFFECT — disabled by default policy.",
            params={"to": (str, True), "subject": (str, True), "body": (str, True)},
            func=_send_email,
            side_effect=True,
        ),
    ]
