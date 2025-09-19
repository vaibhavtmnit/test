"""
LangGraph "Upstream Validation" Agent

- Accepts multiple input types (networkx.DiGraph, pandas.DataFrame, custom).
- Starts from the bottom (sink / last row), walks upstream recursively.
- Validates whether each node's stored value == transform(predecessor values).
- Branches automatically on multiple incoming edges.
- Tracks findings and emits a summary.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Set
from dataclasses import dataclass
import math

# Optional deps (import only if you actually use them)
try:
    import networkx as nx
except Exception:  # pragma: no cover
    nx = None  # type: ignore

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore

# ---- Normalized DAG spec -----------------------------------------------------

@dataclass
class NodeSpec:
    """Normalized node representation."""
    node_id: str
    value: Any
    transform: Optional[str]  # e.g., "fn:sum" or "expr: inputs[0] * 2"
    parents: List[str]        # upstream node_ids

class DAGSpec(TypedDict):
    """Normalized DAG: nodes keyed by id."""
    nodes: Dict[str, NodeSpec]
    sinks: List[str]  # nodes with no outgoing edges (start points for validation)


# ---- Data adapters -----------------------------------------------------------

class DataAdapter:
    """Adapters to normalize various input formats into DAGSpec."""

    @staticmethod
    def from_networkx(
        G: "nx.DiGraph",
        value_attr: str = "value",
        transform_attr: str = "transform",
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
            )

        # sinks = no outgoing edges
        sinks = [id_fn(n) for n, deg in G.out_degree if deg == 0]
        return DAGSpec(nodes=nodes, sinks=sinks)

    @staticmethod
    def from_dataframe(
        df: "pd.DataFrame",
        value_col: str = "value",
        transform_col: str = "transform",
        id_col: Optional[str] = None,
    ) -> DAGSpec:
        """
        Treat rows as a linear pipeline (top -> bottom). Start from the last row.
        parents of row i are [row i-1], except the first row which has [].
        """
        if pd is None:
            raise ImportError("pandas is not available")

        nodes: Dict[str, NodeSpec] = {}
        n = len(df)
        for i in range(n):
            if id_col is None:
                nid = f"row:{i}"
            else:
                nid = str(df.iloc[i][id_col])
            value = df.iloc[i][value_col]
            transform = df.iloc[i][transform_col] if transform_col in df.columns else None
            parents = [f"row:{i-1}"] if i > 0 and id_col is None else ([str(df.iloc[i-1][id_col])] if (i > 0 and id_col) else [])
            nodes[nid] = NodeSpec(
                node_id=nid,
                value=value,
                transform=transform,
                parents=parents,
            )

        sinks = [list(nodes.keys())[-1]] if n > 0 else []
        return DAGSpec(nodes=nodes, sinks=sinks)

    @staticmethod
    def from_custom(prebuilt: DAGSpec) -> DAGSpec:
        """
        Accept a pre-normalized DAGSpec produced by your own code.
        Required keys: {'nodes': {id -> NodeSpec}, 'sinks': [ids]}
        """
        # Basic validation
        if "nodes" not in prebuilt or "sinks" not in prebuilt:
            raise ValueError("Invalid custom DAGSpec; must contain 'nodes' and 'sinks'.")
        return prebuilt


# ---- Transform evaluation (safe) --------------------------------------------

class TransformRegistry:
    """Registry of named, safe transformation functions."""

    _REGISTRY: Dict[str, Any] = {
        # Example primitives; extend freely:
        "sum": lambda inputs, **_: sum(_as_numbers(inputs)),
        "prod": lambda inputs, **_: _prod(_as_numbers(inputs)),
        "max": lambda inputs, **_: max(_as_numbers(inputs)) if inputs else None,
        "min": lambda inputs, **_: min(_as_numbers(inputs)) if inputs else None,
        "mean": lambda inputs, **_: (sum(_as_numbers(inputs)) / len(inputs)) if inputs else None,
        "identity": lambda inputs, **_: inputs[0] if inputs else None,
        # Domain-specific placeholders:
        # "normalize": lambda inputs, mean=0, std=1, **_: (inputs[0] - mean) / std if std else inputs[0],
    }

    @classmethod
    def get(cls, name: str):
        fn = cls._REGISTRY.get(name)
        if fn is None:
            raise KeyError(f"Unknown transform function: {name!r}")
        return fn

def _as_numbers(xs: List[Any]) -> List[float]:
    out = []
    for x in xs:
        if isinstance(x, (int, float)):
            out.append(float(x))
        elif x is None:
            out.append(float("nan"))
        else:
            # try to coerce
            try:
                out.append(float(x))
            except Exception:
                raise TypeError(f"Non-numeric input encountered: {x!r}")
    return out

def _prod(xs: List[float]) -> float:
    p = 1.0
    for v in xs:
        p *= v
    return p

def evaluate_transform(transform_spec: Optional[str], inputs: List[Any]) -> Any:
    """
    Evaluate a transform against upstream inputs.
    Supported forms:
      - None            : passes through first input (identity)
      - "fn:<name>"     : calls a registered function with signature (inputs, **kwargs)
      - "expr:<python>" : evaluates a *restricted* Python expression with 'inputs' in scope
                          e.g., "expr: sum(inputs) / len(inputs)" or "expr: inputs[0] * 2"

    Returns the computed output.
    """
    if transform_spec is None:
        return inputs[0] if inputs else None

    transform_spec = transform_spec.strip()
    if transform_spec.startswith("fn:"):
        name = transform_spec[3:].strip()
        fn = TransformRegistry.get(name)
        return fn(inputs)

    if transform_spec.startswith("expr:"):
        expr = transform_spec[5:].strip()
        # VERY restricted eval: no builtins, only safe math and inputs
        safe_env = {
            "__builtins__": {},
            "inputs": inputs,
            "math": math,
            "len": len,
            "sum": sum,
            "min": min,
            "max": max,
        }
        return eval(expr, safe_env, {})  # Consider replacing with an AST whitelist if needed.

    # Fallback: treat as a registered function name directly
    fn = TransformRegistry.get(transform_spec)
    return fn(inputs)


# ---- Placeholder tool (fetch data) ------------------------------------------

def fetch_data_tool(source: str, node_id: str) -> Tuple[Any, Optional[str]]:
    """
    Placeholder 'tool' you can wire later (e.g., if data & transforms are stored externally).
    Returns (value, transform_spec) for node_id from some 'source'.
    For now, it does nothing—replace with real IO as needed.
    """
    # e.g., query a DB / API by (source, node_id)
    return None, None


# ---- Agent State & LangGraph -------------------------------------------------

from langgraph.graph import StateGraph, END
from typing import TypedDict

class Finding(TypedDict):
    node_id: str
    inputs: List[Any]
    expected: Any
    actual: Any
    ok: bool
    note: Optional[str]

class AgentState(TypedDict):
    """
    Agent runtime state.

    Fields:
      - stack: DFS/LIFO stack of node_ids to process (starts with sink(s))
      - visited: set of node_ids to avoid duplicate work
      - nodes: normalized node specs (id -> NodeSpec)
      - findings: per-node validation results
      - issues_found: True if any mismatch
    """
    stack: List[str]
    visited: Set[str]
    nodes: Dict[str, NodeSpec]
    findings: List[Finding]
    issues_found: bool


def initialize_state(spec: DAGSpec, start_node_id: Optional[str] = None) -> AgentState:
    sinks = spec["sinks"]
    if start_node_id is not None:
        if start_node_id not in spec["nodes"]:
            raise KeyError(f"Start node {start_node_id!r} not found.")
        stack = [start_node_id]
    else:
        if not sinks:
            raise ValueError("No sinks found to start from.")
        stack = list(sinks)  # could be multiple sinks

    return AgentState(
        stack=stack,
        visited=set(),
        nodes=spec["nodes"],
        findings=[],
        issues_found=False,
    )


def _gather_inputs(state: AgentState, node: NodeSpec) -> List[Any]:
    """Collect parent values directly from NodeSpec. If you externalize data, call fetch_data_tool here."""
    inputs: List[Any] = []
    for pid in node.parents:
        parent = state["nodes"][pid]
        val = parent.value
        # If you separate storage: val, _ = fetch_data_tool("my_source", pid)
        inputs.append(val)
    return inputs


def validate_and_expand(state: AgentState) -> AgentState:
    """Pop one node, validate it, push its parents (branching if multiple)."""
    if not state["stack"]:
        return state

    node_id = state["stack"].pop()
    if node_id in state["visited"]:
        return state
    state["visited"].add(node_id)

    node = state["nodes"][node_id]
    inputs = _gather_inputs(state, node)

    # Compute expected from transform(inputs)
    try:
        expected = evaluate_transform(node.transform, inputs)
        ok = _equalish(expected, node.value)
        note = None
    except Exception as e:
        expected = None
        ok = False
        note = f"Transform error: {e}"

    state["findings"].append(Finding(
        node_id=node_id,
        inputs=inputs,
        expected=expected,
        actual=node.value,
        ok=ok,
        note=note
    ))

    if not ok:
        state["issues_found"] = True

    # Push upstream parents (branch naturally via stack)
    for pid in node.parents:
        if pid not in state["visited"]:
            state["stack"].append(pid)

    return state


def should_continue(state: AgentState) -> str:
    """Keep looping until the stack is empty."""
    return "loop" if state["stack"] else END


def _equalish(a: Any, b: Any, tol: float = 1e-9) -> bool:
    """Loose equality for numerics; strict fallback otherwise."""
    try:
        if (isinstance(a, (int, float)) or isinstance(b, (int, float))) and a is not None and b is not None:
            return abs(float(a) - float(b)) <= tol
        return a == b
    except Exception:
        return a == b


def build_agent_graph():
    """
    Returns a compiled LangGraph that:
      - loops 'validate_and_expand' until stack is empty
      - ends and returns final AgentState
    """
    sg = StateGraph(AgentState)
    sg.add_node("loop", validate_and_expand)
    sg.set_entry_point("loop")
    sg.add_conditional_edges("loop", should_continue, {
        "loop": "loop",
        END: END,
    })
    return sg.compile()


# ---- Example usage -----------------------------------------------------------

if __name__ == "__main__":
    # Example A: NetworkX (fan-in demo)
    if nx is not None:
        G = nx.DiGraph()
        # upstream leaves
        G.add_node("A", value=2, transform=None)          # source value
        G.add_node("B", value=3, transform=None)          # source value
        # merge node C = sum(A, B)
        G.add_node("C", value=5, transform="fn:sum")
        G.add_edge("A", "C")
        G.add_edge("B", "C")
        # final node D = C * 2  (via expr)
        G.add_node("D", value=10, transform="expr: inputs[0] * 2")
        G.add_edge("C", "D")

        spec = DataAdapter.from_networkx(G)
        state = initialize_state(spec)  # starts from sink: "D"
        app = build_agent_graph()
        final_state: AgentState = app.invoke(state)  # type: ignore

        print("\n-- NetworkX findings --")
        for f in final_state["findings"]:
            print(f)
        print("issues_found:", final_state["issues_found"])

    # Example B: pandas (linear chain; last row is sink)
    if pd is not None:
        df = pd.DataFrame(
            [
                {"value": 2,  "transform": None},                  # row 0 (source)
                {"value": 4,  "transform": "expr: inputs[0] * 2"}, # row 1
                {"value": 10, "transform": "fn:sum"},              # row 2  <-- will mismatch (expects 4 only)
            ]
        )
        spec = DataAdapter.from_dataframe(df)
        state = initialize_state(spec)
        app = build_agent_graph()
        final_state: AgentState = app.invoke(state)  # type: ignore

        print("\n-- DataFrame findings --")
        for f in final_state["findings"]:
            print(f)
        print("issues_found:", final_state["issues_found"])

    # Example C: custom spec (already normalized)
    custom_spec: DAGSpec = {
        "nodes": {
            "x": NodeSpec("x", 1, None, []).__dict__,
            "y": NodeSpec("y", 2, None, []).__dict__,
            "z": NodeSpec("z", 3, "fn:sum", ["x", "y"]).__dict__,  # expects 3; ok
        },
        "sinks": ["z"],
    }
    state = initialize_state(DataAdapter.from_custom(custom_spec))
    app = build_agent_graph()
    final_state: AgentState = app.invoke(state)  # type: ignore

    print("\n-- Custom findings --")
    for f in final_state["findings"]:
        print(f)
    print("issues_found:", final_state["issues_found"])
