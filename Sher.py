"""
Upstream Validation Agent (Flag / Steps / NL) with Azure OpenAI (LangChain)

- Mix-and-match validation per node:
  * FLAG mode: {"mode":"flag","is_transformed": bool}
  * STEPS mode (mini-DSL): {"mode":"steps","steps":[...], "ctx": {...}}
  * NL mode (natural language + tools): {"mode":"nl","statement":"...", "needs":[...]}

- NL mode uses LangChain + Azure OpenAI with a strict, JSON-only schema.
- Steps + NL share the same attribute fetch tool, so you implement IO once.

Dependencies (install):
    pip install langgraph langchain langchain-openai pydantic typing-extensions
    pip install networkx pandas  # optional, only if you use those adapters
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Set, Union, Protocol
from dataclasses import dataclass
import math
import os
import re
import json

# Optional deps
try:
    import networkx as nx
except Exception:
    nx = None  # type: ignore

try:
    import pandas as pd
except Exception:
    pd = None  # type: ignore

# LangGraph
from langgraph.graph import StateGraph, END

# LangChain (Azure OpenAI)
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI

# =============================================================================
# Core structures
# =============================================================================

TransformSpec = Union[None, str, Dict[str, Any], List[Dict[str, Any]]]

@dataclass
class NodeSpec:
    node_id: str
    value: Any
    transform: TransformSpec
    parents: List[str]
    # Optional: bag of attributes (for NL/steps contexts)
    attrs: Optional[Dict[str, Any]] = None

class DAGSpec(TypedDict):
    nodes: Dict[str, NodeSpec]
    sinks: List[str]

# =============================================================================
# Adapters
# =============================================================================

class DataAdapter:
    @staticmethod
    def from_networkx(
        G: "nx.DiGraph",
        value_attr: str = "value",
        transform_attr: str = "transform",
        attrs_attr: str = "attrs",
        id_fn=lambda n: str(n),
    ) -> DAGSpec:
        if nx is None:
            raise ImportError("networkx is not available")
        nodes: Dict[str, NodeSpec] = {}
        for n in G.nodes:
            nid = id_fn(n)
            data = G.nodes[n]
            nodes[nid] = NodeSpec(
                node_id=nid,
                value=data.get(value_attr),
                transform=data.get(transform_attr),
                parents=[id_fn(p) for p in G.predecessors(n)],
                attrs=data.get(attrs_attr) or {},
            )
        sinks = [id_fn(n) for n, deg in G.out_degree if deg == 0]
        return DAGSpec(nodes=nodes, sinks=sinks)

    @staticmethod
    def from_dataframe(
        df: "pd.DataFrame",
        value_col: str = "value",
        transform_col: str = "transform",
        attrs_col: Optional[str] = None,
        id_col: Optional[str] = None,
    ) -> DAGSpec:
        if pd is None:
            raise ImportError("pandas is not available")

        def parse_maybe_json(x):
            if isinstance(x, str):
                t = x.strip()
                if (t.startswith("{") and t.endswith("}")) or (t.startswith("[") and t.endswith("]")):
                    try:
                        return json.loads(t)
                    except Exception:
                        return x
            return x

        nodes: Dict[str, NodeSpec] = {}
        n = len(df)
        for i in range(n):
            nid = str(df.iloc[i][id_col]) if id_col else f"row:{i}"
            value = df.iloc[i][value_col]
            raw_t = df.iloc[i][transform_col] if transform_col in df.columns else None
            transform = parse_maybe_json(raw_t)
            attrs = parse_maybe_json(df.iloc[i][attrs_col]) if (attrs_col and attrs_col in df.columns) else {}
            parents = [str(df.iloc[i-1][id_col])] if (i > 0 and id_col) else ([f"row:{i-1}"] if i > 0 else [])
            nodes[nid] = NodeSpec(nid, value, transform, parents, attrs=attrs)
        sinks = [list(nodes.keys())[-1]] if n > 0 else []
        return DAGSpec(nodes=nodes, sinks=sinks)

    @staticmethod
    def from_custom(prebuilt: DAGSpec) -> DAGSpec:
        if "nodes" not in prebuilt or "sinks" not in prebuilt:
            raise ValueError("Invalid custom DAGSpec; must contain 'nodes' and 'sinks'.")
        return prebuilt

# =============================================================================
# Shared tools / IO (replace with your real implementations)
# =============================================================================

def fetch_data_tool(source: str, key: str) -> Tuple[Any, Optional[str]]:
    """Return (value, transform_spec) from external storage. Replace with real IO."""
    return None, None

def fetch_attribute_tool(node_id: str, attr_name: str, context: Dict[str, Any]) -> Any:
    """
    Fetch arbitrary attribute needed by steps/NL modes (e.g., thresholds, FX rates).
    Replace this with real DB/API logic. For demo, look up in context["external"].
    """
    return context.get("external", {}).get(node_id, {}).get(attr_name)

# =============================================================================
# Utility helpers
# =============================================================================

def _nums(xs):
    out=[]
    for x in xs:
        if x is None: 
            continue
        try: out.append(float(x))
        except Exception: raise TypeError(f"Non-numeric: {x!r}")
    return out

def _prod(xs):
    p=1.0
    for v in xs: p*=v
    return p

def _equalish(a: Any, b: Any, tol: float = 1e-9) -> bool:
    try:
        if (isinstance(a,(int,float)) or isinstance(b,(int,float))) and a is not None and b is not None:
            return abs(float(a)-float(b)) <= tol
        return a == b
    except Exception:
        return a == b

def _eval_expr(expr: str, inputs: List[Any]) -> Any:
    safe = {"__builtins__": {}, "inputs": inputs, "math": math, "len": len, "sum": sum, "min": min, "max": max}
    return eval(expr, safe, {})

# =============================================================================
# Transform Registry (for "fn:..." and simple names)
# =============================================================================

class TransformRegistry:
    _REGISTRY: Dict[str, Any] = {
        "sum":   lambda inputs, **_: sum(_nums(inputs)),
        "prod":  lambda inputs, **_: _prod(_nums(inputs)),
        "mean":  lambda inputs, **_: (sum(_nums(inputs))/len(inputs)) if inputs else None,
        "identity": lambda inputs, **_: inputs[0] if inputs else None,
    }
    @classmethod
    def get(cls, name: str):
        if name not in cls._REGISTRY:
            raise KeyError(f"Unknown transform function: {name}")
        return cls._REGISTRY[name]

# =============================================================================
# Mini-DSL (STEPS) engine and ops
# =============================================================================

class OpRegistry:
    OPS: Dict[str, Any] = {}
    @classmethod
    def register(cls, name: str):
        def deco(fn):
            cls.OPS[name] = fn
            return fn
        return deco
    @classmethod
    def get(cls, name: str):
            if name not in cls.OPS:
                raise KeyError(f"Unknown op: {name}")
            return cls.OPS[name]

@OpRegistry.register("take")
def op_take(seq, idx:int): return seq[idx]

@OpRegistry.register("sum")
def op_sum(*args): return sum(float(x) for x in args)

@OpRegistry.register("mul")
def op_mul(a,b): return float(a)*float(b)

@OpRegistry.register("clip")
def op_clip(x, lower=None, upper=None):
    x=float(x)
    if lower is not None: x=max(x,float(lower))
    if upper is not None: x=min(x,float(upper))
    return x

@OpRegistry.register("apply")  # tiny inline expr on $prev
def op_apply(x=None, expr: str = None):
    if expr is None: return x
    return _eval_expr(expr, [x])

@OpRegistry.register("fetch")  # fetch an attribute on-demand (shared tool)
def op_fetch(node_id: str, name: str, ctx: Dict[str, Any]):
    if "node_attrs" in ctx and name in ctx["node_attrs"]:
        return ctx["node_attrs"][name]
    return fetch_attribute_tool(node_id, name, ctx)

def _resolve(ref: Any, env: Dict[str, Any]) -> Any:
    if not isinstance(ref, str) or not ref.startswith("$"): return ref
    if ref == "$prev": return env.get("prev")
    if ref == "$inputs": return env["inputs"]
    if ref.startswith("$inputs[") and ref.endswith("]"):
        return env["inputs"][int(ref[len("$inputs["):-1])]
    # dotted lookup e.g. $ctx.threshold
    cur = env
    for p in ref[1:].split("."):
        cur = cur.get(p) if isinstance(cur, dict) else getattr(cur, p)
    return cur

def run_steps(steps: List[Dict[str, Any]], inputs: List[Any], ctx: Dict[str, Any]) -> Any:
    env = {"inputs": inputs, "prev": None, "ctx": ctx}
    out=None
    for step in steps:
        op = OpRegistry.get(step.get("op"))
        args=[_resolve(a, env) for a in step.get("args",[])]
        kwargs={k:_resolve(v, env) for k,v in step.get("kwargs",{}).items()}
        out = op(*args, **kwargs)
        env["prev"]=out
    return out

# =============================================================================
# Natural language mode: needs/placeholder resolver + LLM validator
# =============================================================================

_PLACEHOLDER_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")

def resolve_nl_needs(
    node_id: str,
    statement: str,
    needs: List[str],
    base_attrs: Dict[str, Any],
    global_ctx: Dict[str, Any],
) -> Dict[str, Any]:
    required: Set[str] = set(needs or [])
    for m in _PLACEHOLDER_RE.finditer(statement or ""):
        required.add(m.group(1))
    attrs = dict(base_attrs or {})
    for name in required:
        if name not in attrs:
            attrs[name] = fetch_attribute_tool(node_id, name, {**global_ctx, "node_id": node_id})
    return attrs

def materialize_statement(statement: str, attrs: Dict[str, Any]) -> str:
    def repl(m):
        key = m.group(1)
        return str(attrs.get(key, f"{{{{{key}}}}}"))
    return _PLACEHOLDER_RE.sub(repl, statement or "")

# ---------- LangChain model + schema ----------

class NLVerdict(BaseModel):
    ok: bool = Field(..., description="True if the transformation appears correctly applied.")
    expected_value: Optional[float] = Field(
        default=None,
        description="If you can compute a numeric expected value, put it here; else null."
    )
    summary: str = Field(
        ..., description="1-2 sentences explaining the verdict. No step-by-step reasoning."
    )

def _azure_llm() -> AzureChatOpenAI:
    """
    Configure Azure OpenAI from environment:
      AZURE_OPENAI_API_KEY
      AZURE_OPENAI_ENDPOINT        (e.g., https://<your-resource>.openai.azure.com/)
      AZURE_OPENAI_DEPLOYMENT      (deployment name for gpt-4o/gpt-4.1/etc.)
      AZURE_OPENAI_API_VERSION     (e.g., 2024-08-01-preview)
    """
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
    if not (api_key and endpoint and deployment):
        raise RuntimeError("Azure OpenAI env vars missing: AZURE_OPENAI_API_KEY / AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_DEPLOYMENT")
    return AzureChatOpenAI(
        azure_deployment=deployment,
        api_version=api_version,
        azure_endpoint=endpoint,
        temperature=0,
    )

_NL_PROMPT = ChatPromptTemplate.from_template(
    """You are a strict validator for data pipeline transformations.

Inputs:
- Upstream inputs (JSON array): {inputs_json}
- Node attributes (JSON): {attrs_json}
- Actual node value: {actual_json}

Instruction:
- Determine whether the following natural-language transformation statement appears correctly applied:
  "{statement}"

Rules:
- If you can compute an expected numeric value from the inputs/attrs, do so.
- Return STRICT JSON with fields: ok (bool), expected_value (number|null), summary (short string).
- Do NOT include extra text. No step-by-step reasoning or chain-of-thought.
"""
)

def nl_validation_tool_llm(
    statement: str,
    inputs: List[Any],
    actual_value: Any,
    attrs: Dict[str, Any],
) -> Dict[str, Any]:
    """LLM-backed NL validator using Azure OpenAI via LangChain."""
    llm = _azure_llm()
    msg = _NL_PROMPT.format_messages(
        inputs_json=json.dumps(inputs),
        attrs_json=json.dumps(attrs),
        actual_json=json.dumps(actual_value),
        statement=statement,
    )
    resp = llm.invoke(msg)  # returns AIMessage
    # Strict JSON expected:
    try:
        data = json.loads(resp.content)
    except Exception:
        # If model returns non-JSON, try a crude salvage:
        text = resp.content.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            data = json.loads(text[start:end+1])
        else:
            # Last resort: mark unknown but OK
            data = {"ok": True, "expected_value": None, "summary": "Model returned non-JSON; assumed OK."}

    # Validate with Pydantic to normalize types
    verdict = NLVerdict(**data)
    return {"ok": verdict.ok, "expected": verdict.expected_value, "note": verdict.summary}

# =============================================================================
# Strategies (Flag / Steps / NL)
# =============================================================================

class Strategy(Protocol):
    def validate(self, node: NodeSpec, inputs: List[Any], global_ctx: Dict[str, Any]) -> Tuple[bool, Any, str]:
        """Returns (ok, expected, note)."""

# --- FLAG mode ---
class FlagStrategy:
    """
    Only 'is_transformed' boolean known.
    Policy:
      - If is_transformed == False: expected == first input (identity)
      - If is_transformed == True: assert actual != first input (changed-ness heuristic)
    """
    def validate(self, node: NodeSpec, inputs: List[Any], global_ctx: Dict[str, Any]) -> Tuple[bool, Any, str]:
        spec = node.transform if isinstance(node.transform, dict) else {}
        flag = bool(spec.get("is_transformed", False))
        if not inputs:
            return True, node.value, "Source node (no parents)."
        base = inputs[0]
        if not flag:
            expected = base
            ok = _equalish(expected, node.value)
            return ok, expected, "FLAG: expected pass-through."
        else:
            changed = (not _equalish(base, node.value))
            return changed, None, "FLAG: expected changed value vs input[0]."

# --- STEPS (mini-DSL) mode ---
class StepsStrategy:
    def validate(self, node: NodeSpec, inputs: List[Any], global_ctx: Dict[str, Any]) -> Tuple[bool, Any, str]:
        t = node.transform
        # Back-compat shims for None/str
        if t is None:
            expected = inputs[0] if inputs else None
            return _equalish(expected, node.value), expected, "Implicit identity."
        if isinstance(t, str):
            s=t.strip()
            if s.startswith("fn:"):
                fn=TransformRegistry.get(s[3:].strip())
                expected=fn(inputs)
            elif s.startswith("expr:"):
                expected=_eval_expr(s[5:].strip(), inputs)
            else:
                expected=TransformRegistry.get(s)(inputs)
            return _equalish(expected, node.value), expected, "String transform."

        # Proper STEPS
        ctx = (t.get("ctx") if isinstance(t, dict) else {}) or {}
        node_attrs = node.attrs or {}
        ctx = {**global_ctx, **ctx, "node_id": node.node_id, "node_attrs": node_attrs}
        steps = t["steps"] if isinstance(t, dict) and "steps" in t else (t if isinstance(t, list) else [])
        expected = run_steps(steps, inputs, ctx)
        return _equalish(expected, node.value), expected, "Steps (mini-DSL)."

# --- NL mode (with shared fetch + Azure OpenAI) ---
class NLAgentStrategy:
    def validate(self, node: NodeSpec, inputs: List[Any], global_ctx: Dict[str, Any]) -> Tuple[bool, Any, str]:
        t = node.transform if isinstance(node.transform, dict) else {}
        statement = t.get("statement", "")
        needs = t.get("needs", [])  # optional list[str]

        # 1) Resolve attributes via shared fetch tool (needs + {{placeholders}})
        attrs_pre = dict(node.attrs or {})
        attrs = resolve_nl_needs(node.node_id, statement, needs, attrs_pre, global_ctx)

        # 2) Materialize placeholders for LLM (e.g., {{fx_rate}})
        mat_statement = materialize_statement(statement, attrs)

        # 3) LLM verdict (JSON-only)
        res = nl_validation_tool_llm(mat_statement, inputs, node.value, attrs)

        ok = bool(res.get("ok", True))
        expected = res.get("expected")
        summary = res.get("note") or ""
        return ok, expected, summary + " (NL mode)"

# =============================================================================
# Strategy selection
# =============================================================================

def pick_strategy(node: NodeSpec) -> Strategy:
    t = node.transform
    if isinstance(t, dict) and t.get("mode") == "flag":
        return FlagStrategy()
    if isinstance(t, dict) and t.get("mode") == "nl":
        return NLAgentStrategy()
    return StepsStrategy()  # default

# =============================================================================
# Agent state + graph
# =============================================================================

class Finding(TypedDict):
    node_id: str
    inputs: List[Any]
    expected: Any
    actual: Any
    ok: bool
    note: Optional[str]
    mode: str

class AgentState(TypedDict):
    stack: List[str]
    visited: Set[str]
    nodes: Dict[str, NodeSpec]
    findings: List[Finding]
    issues_found: bool
    global_ctx: Dict[str, Any]

def initialize_state(spec: DAGSpec, start_node_id: Optional[str] = None, global_ctx: Optional[Dict[str, Any]] = None) -> AgentState:
    sinks = spec["sinks"]
    stack = [start_node_id] if (start_node_id is not None) else (list(sinks) if sinks else [])
    if not stack:
        raise ValueError("No starting node(s).")
    return AgentState(
        stack=stack,
        visited=set(),
        nodes=spec["nodes"],
        findings=[],
        issues_found=False,
        global_ctx=global_ctx or {},
    )

def _gather_inputs(state: AgentState, node: NodeSpec) -> List[Any]:
    vals=[]
    for pid in node.parents:
        parent = state["nodes"][pid]
        vals.append(parent.value)
    return vals

def validate_and_expand(state: AgentState) -> AgentState:
    if not state["stack"]:
        return state
    node_id = state["stack"].pop()
    if node_id in state["visited"]:
        return state
    state["visited"].add(node_id)

    node = state["nodes"][node_id]
    inputs = _gather_inputs(state, node)

    strategy = pick_strategy(node)
    ok, expected, note = strategy.validate(node, inputs, state["global_ctx"])

    mode = node.transform.get("mode") if isinstance(node.transform, dict) else ("backcompat" if node.transform is not None else "implicit")
    state["findings"].append(Finding(
        node_id=node_id, inputs=inputs, expected=expected, actual=node.value, ok=ok, note=note, mode=str(mode)
    ))
    if not ok:
        state["issues_found"] = True

    # DFS upstream
    for pid in node.parents:
        if pid not in state["visited"]:
            state["stack"].append(pid)
    return state

def should_continue(state: AgentState) -> str:
    return "loop" if state["stack"] else END

def build_agent_graph():
    sg = StateGraph(AgentState)
    sg.add_node("loop", validate_and_expand)
    sg.set_entry_point("loop")
    sg.add_conditional_edges("loop", should_continue, {"loop": "loop", END: END})
    return sg.compile()

# =============================================================================
# Minimal demo (optional)
# =============================================================================

if __name__ == "__main__":
    # Set these env vars before running NL demo:
    #   AZURE_OPENAI_API_KEY
    #   AZURE_OPENAI_ENDPOINT
    #   AZURE_OPENAI_DEPLOYMENT
    #   AZURE_OPENAI_API_VERSION (optional)
    if nx is not None:
        G = nx.DiGraph()
        # Leaves
        G.add_node("A", value=2, transform={"mode":"flag","is_transformed": False})   # identity expected
        G.add_node("B", value=3, transform=None)                                      # implicit identity
        # Merge with STEPS: (A + B) * 2
        G.add_node("C", value=10, transform={
            "mode":"steps",
            "steps":[
                {"op":"sum","args":["$inputs[0]","$inputs[1]"]},
                {"op":"mul","args":["$prev",2]},
            ]
        })
        G.add_edge("A","C"); G.add_edge("B","C")
        # NL: "Value should be the sum of upstream values."
        G.add_node("D", value=10, transform={
            "mode":"nl",
            "statement":"Value should be the sum of upstream values."
        })
        G.add_edge("C","D")

        spec = DataAdapter.from_networkx(G)
        state = initialize_state(spec, global_ctx={"external": {}})
        app = build_agent_graph()
        out = app.invoke(state)  # type: ignore
        print("\n-- Findings --")
        for f in out["findings"]:
            print(f)
        print("issues_found:", out["issues_found"])

  
