"""
LLM-Only Java Call-Tree Builder (LangGraph)
------------------------------------------

This module builds a DFS-style execution/analysis tree for a Java snippet using
**only an LLM**. No Tree-sitter, no static parsing—every expansion step is guided
by the model, with guardrails to keep the process deterministic and safe.

Key ideas
- The LLM acts like a human code reviewer: given the full code and the current
  focus node, it lists the next "relevant" children to add to the tree.
- We enforce structure by asking for strict JSON and validating it. If the LLM
  fails to produce valid JSON, we fall back to an empty expansion for that step
  (so the loop still terminates cleanly).
- DFS order is achieved via a stack; cycle protection uses stable node ids
  (provided by the LLM, e.g., "method:Foo.start(int,String)").

Dependencies
    pip install langgraph openai networkx

You may inject **any** LLM by supplying a callable that returns the required
JSON; an OpenAI-based implementation is included for convenience.

Usage (quick)
-------------
>>> from openai import OpenAI
>>> from java_calltree_llm_agent import LLMJavaCallTree, openai_next_nodes
>>> code = """
... public class Demo {
...   public void start(int n, String msg){ helper(msg); }
...   void helper(String s){ System.out.println(s); }
... }
... """
>>> client = OpenAI()
>>> agent = LLMJavaCallTree(code, start_symbol="start",
...                         next_nodes_fn=lambda **kw: openai_next_nodes(client, model="gpt-4o", **kw))
>>> g = agent.build(max_nodes=200)
>>> data = agent.to_json(g)
>>> len(data["nodes"]) > 0 and len(data["edges"]) > 0
True

Notes
- The LLM controls which children are added and which of those are expanded.
- If your model is chatty, keep `max_children_per_step` small (default 8).
- For reproducibility, keep your prompt/policy stable.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Callable, TypedDict
import json

import networkx as nx
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# ------------------------------- Data Types -------------------------------

@dataclass(frozen=True)
class NodeData:
    """Metadata stored on each graph node.

    Fields
    ------
    id : str
        LLM-stable identifier (e.g., "method:Demo.start(int,String)"). Used for
        cycle detection and de-duplication.
    kind : str
        e.g., "method", "param", "class", "call", "lambda", "ctrl", etc.
    name : str
        Human-friendly name (e.g., "start", "helper", "msg").
    role : str
        Contextual role (e.g., "root", "param", "call", "receiver", "local").
    signature : Optional[str]
        Signature if known (e.g., "start(int,String)").
    note : Optional[str]
        Short rationale from the LLM (why this node matters).
    confidence : Optional[float]
        Model self-reported confidence in [0,1].
    """
    id: str
    kind: str
    name: str
    role: str
    signature: Optional[str] = None
    note: Optional[str] = None
    confidence: Optional[float] = None


class LLMChild(TypedDict, total=False):
    """Schema for children returned by the LLM."""
    id: str
    kind: str
    name: str
    role: str
    signature: Optional[str]
    note: Optional[str]
    confidence: Optional[float]
    expand: bool


# `next_nodes_fn` contract
#   def next_nodes_fn(code: str, parent: NodeData, seen_ids: List[str], policy: str,
#                     max_children: int) -> List[LLMChild]
NextNodesFn = Callable[[str, NodeData, List[str], str, int], List[LLMChild]]


# ----------------------------- Core Agent API -----------------------------

class LLMJavaCallTree:
    """LLM-only DFS call/analysis tree for Java code.

    Parameters
    ----------
    code : str
        Java source text (single file snippet is fine).
    start_symbol : str
        Name of the method/ctor/class used as the *root* of expansion.
        The LLM must be able to locate it in the code.
    next_nodes_fn : Callable
        A function that calls your LLM and returns a list of children in the
        schema `LLMChild`. You can inject OpenAI, Azure OpenAI, Anthropic, etc.
    policy : str, optional
        A short instruction string handed to the LLM describing the expansion
        priorities. Default mirrors a human reader’s order.
    max_children_per_step : int, default 8
        Hard cap per expansion step; controls token usage and branching.

    Example
    -------
    >>> from openai import OpenAI
    >>> from java_calltree_llm_agent import LLMJavaCallTree, openai_next_nodes
    >>> code = "public class A { void entry(){ helper(); } void helper(){} }"
    >>> client = OpenAI()
    >>> agent = LLMJavaCallTree(code, "entry",
    ...     next_nodes_fn=lambda **kw: openai_next_nodes(client, model="gpt-4o", **kw))
    >>> g = agent.build()
    >>> isinstance(g.number_of_nodes(), int) and g.number_of_nodes() > 0
    True
    """

    def __init__(
        self,
        code: str,
        start_symbol: str,
        *,
        next_nodes_fn: NextNodesFn,
        policy: str = (
            "Relevance policy: parameters (L->R) and their types; receiver class; local new objects; "
            "direct method calls; chained/stream calls; lambdas/method refs; control-flow blocks; returns. "
            "Avoid duplicates; give each child a stable id and set expand=true for nodes that should be expanded."
        ),
        max_children_per_step: int = 8,
    ) -> None:
        self.code = code
        self.start_symbol = start_symbol
        self.next_nodes_fn = next_nodes_fn
        self.policy = policy
        self.max_children_per_step = max_children_per_step

    # ---- Public ----
    def build(self, *, max_nodes: int = 500) -> nx.DiGraph:
        """Run the DFS expansion loop until the stack is exhausted or `max_nodes` reached.

        Returns
        -------
        networkx.DiGraph
            Directed graph with edges labeled `kind='flows_to'`.
        """
        g = nx.DiGraph()

        root = NodeData(
            id=f"root:{self.start_symbol}",
            kind="method",  # heuristic; model can refine children
            name=self.start_symbol,
            role="root",
            note="User-provided starting symbol",
        )
        g.add_node(root.id, data=root)

        seen = {root.id}
        stack: List[str] = [root.id]  # DFS stack of node ids

        while stack and g.number_of_nodes() < max_nodes:
            parent_id = stack.pop()
            parent: NodeData = g.nodes[parent_id]["data"]

            try:
                children = self.next_nodes_fn(
                    code=self.code,
                    parent=parent,
                    seen_ids=list(seen),
                    policy=self.policy,
                    max_children=self.max_children_per_step,
                )
            except Exception:
                children = []  # fail-safe

            # Validate and add
            for ch in children[: self.max_children_per_step]:
                if not _valid_child(ch):
                    continue
                ch_id = ch["id"]
                node = NodeData(
                    id=ch_id,
                    kind=ch.get("kind", "unknown"),
                    name=ch.get("name", ch_id),
                    role=ch.get("role", "child"),
                    signature=ch.get("signature"),
                    note=ch.get("note"),
                    confidence=ch.get("confidence"),
                )

                if ch_id not in g:
                    g.add_node(ch_id, data=node)
                # Always add the edge (it’s okay if it already exists)
                g.add_edge(parent_id, ch_id, kind="flows_to")

                # Cycle protection
                if ch_id in seen:
                    continue
                seen.add(ch_id)

                # DFS expansion if LLM says so
                if bool(ch.get("expand", False)):
                    stack.append(ch_id)

        return g

    def to_json(self, g: nx.DiGraph) -> Dict[str, Any]:
        nodes = []
        id_map: Dict[str, int] = {}
        for i, (node_id, attrs) in enumerate(g.nodes(data=True)):
            id_map[node_id] = i
            nodes.append({"id": i, **asdict(attrs["data"])})
        edges = [{"src": id_map[u], "dst": id_map[v], **attrs} for u, v, attrs in g.edges(data=True)]
        return {"nodes": nodes, "edges": edges}


# ------------------------------- Utilities -------------------------------

def _valid_child(ch: Dict[str, Any]) -> bool:
    try:
        return (
            isinstance(ch.get("id"), str)
            and isinstance(ch.get("kind", ""), str)
            and isinstance(ch.get("name", ""), str)
        )
    except Exception:
        return False


# -------------------------- OpenAI Helper (optional) --------------------------
# You can ignore this and inject your own `next_nodes_fn`.

def openai_next_nodes(client, *, model: str, code: str, parent: NodeData, seen_ids: List[str], policy: str, max_children: int) -> List[LLMChild]:
    """Call OpenAI Responses API to propose next children.

    The prompt asks for a *strict JSON* response with this shape:
        {
          "children": [
             {"id": "method:Demo.helper(String)", "kind": "method", "name": "helper",
              "role": "call", "signature": "helper(String)",
              "note": "Called directly in start(...) with msg", "confidence": 0.82,
              "expand": true}
          ]
        }

    Failures return an empty list.
    """
    system = (
        "You are a precise Java code analysis assistant. Work ONLY with the code provided. "
        "We are building a DFS-style call/analysis tree. Given a PARENT node, propose up to N CHILDREN "
        "that a human would inspect next to understand execution/data flow."
    )
    rules = (
        "Return STRICT JSON with key 'children'. Each child must have: id, kind, name, role. "
        "Optionally include: signature, note, confidence (0..1), expand (true/false). "
        "Ids must be STABLE and DEDUP-friendly, e.g., 'method:Class.method(int,String)'. "
        "Policy: " + policy + " "
        "Do NOT invent code. Only infer from the given snippet."
    )
    user = {
        "code": code,
        "parent": asdict(parent),
        "seen_ids": seen_ids,
        "max_children": max_children,
    }
    try:
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system},
                {"role": "system", "content": rules},
                {"role": "user", "content": json.dumps(user)},
            ],
            temperature=0.2,
        )
        txt = resp.output_text.strip()
        data = json.loads(txt)
        children = data.get("children", [])
        if isinstance(children, list):
            # Basic sanitation: cap length
            return children[:max_children]
    except Exception:
        pass
    return []


# ------------------------------- LangGraph API -------------------------------
# If you prefer a fully LangGraph-managed loop (with checkpoints), use this.

class _State(TypedDict):
    done: bool
    stack: List[str]
    seen: List[str]
    graph: nx.DiGraph
    agent: LLMJavaCallTree


def _step(state: _State) -> _State:
    if not state["stack"]:
        state["done"] = True
        return state
    g = state["graph"]
    agent = state["agent"]

    parent_id = state["stack"].pop()
    parent: NodeData = g.nodes[parent_id]["data"]

    children = agent.next_nodes_fn(
        code=agent.code,
        parent=parent,
        seen_ids=state["seen"],
        policy=agent.policy,
        max_children=agent.max_children_per_step,
    )
    for ch in children:
        if not _valid_child(ch):
            continue
        nid = ch["id"]
        node = NodeData(
            id=nid,
            kind=ch.get("kind", "unknown"),
            name=ch.get("name", nid),
            role=ch.get("role", "child"),
            signature=ch.get("signature"),
            note=ch.get("note"),
            confidence=ch.get("confidence"),
        )
        if nid not in g:
            g.add_node(nid, data=node)
        g.add_edge(parent_id, nid, kind="flows_to")
        if nid not in state["seen"] and bool(ch.get("expand", False)):
            state["stack"].append(nid)
            state["seen"].append(nid)

    if not state["stack"]:
        state["done"] = True
    return state


def build_with_langgraph(agent: LLMJavaCallTree, *, max_nodes: int = 500) -> nx.DiGraph:
    """Run the DFS builder under a LangGraph loop with memory checkpoints.

    Example
    -------
    >>> from openai import OpenAI
    >>> from java_calltree_llm_agent import LLMJavaCallTree, openai_next_nodes, build_with_langgraph
    >>> client = OpenAI()
    >>> code = "public class A { void entry(){ helper(); } void helper(){} }"
    >>> agent = LLMJavaCallTree(code, "entry",
    ...     next_nodes_fn=lambda **kw: openai_next_nodes(client, model="gpt-4o", **kw))
    >>> g = build_with_langgraph(agent)
    >>> g.number_of_nodes() > 0
    True
    """
    # Seed graph with root
    g = nx.DiGraph()
    root = NodeData(id=f"root:{agent.start_symbol}", kind="method", name=agent.start_symbol, role="root")
    g.add_node(root.id, data=root)

    state: _State = {
        "done": False,
        "stack": [root.id],
        "seen": [root.id],
        "graph": g,
        "agent": agent,
    }

    sg = StateGraph(_State)
    sg.add_node("step", _step)
    sg.add_edge(START, "step")
    sg.add_conditional_edges("step", lambda s: "end" if s["done"] or s["graph"].number_of_nodes() >= max_nodes else "continue", {"continue": "step", "end": END})

    app = sg.compile(checkpointer=MemorySaver())
    final = app.invoke(state)
    return final["graph"]


# -------------------------- Model-Specific Helpers --------------------------
"""Drop-in wrappers for **o3-mini** and **o1-preview** using the Responses API.
Use these to plug directly into `LLMJavaCallTree(..., next_nodes_fn=...)`.
"""

def openai_next_nodes_o3mini(client, *, code: str, parent: NodeData, seen_ids: List[str], policy: str, max_children: int, model: str = "o3-mini", temperature: float = 0.2) -> List[LLMChild]:
    """Specialized helper for **o3-mini**.

    - Enforces strict JSON via `response_format={"type":"json_object"}`.
    - Keeps a small temperature for mild diversity (tune or set to 0.0).
    """
    system = (
        "You are a precise Java code analysis assistant. Work ONLY with the code provided. "
        "We are building a DFS-style call/analysis tree. Given a PARENT node, propose up to N CHILDREN "
        "that a human would inspect next to understand execution/data flow."
    )
    rules = (
        "Return STRICT JSON with key 'children'. Each child must have: id, kind, name, role. "
        "Optionally include: signature, note, confidence (0..1), expand (true/false). "
        "Ids must be STABLE and DEDUP-friendly, e.g., 'method:Class.method(int,String)'. "
        "Policy: " + policy + " "
        "Do NOT invent code. Only infer from the given snippet."
    )
    user = {
        "code": code,
        "parent": asdict(parent),
        "seen_ids": seen_ids,
        "max_children": max_children,
    }
    try:
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system},
                {"role": "developer", "content": rules},
                {"role": "user", "content": json.dumps(user)},
            ],
            response_format={"type": "json_object"},
            temperature=temperature,
        )
        data = json.loads(resp.output_text.strip())
        children = data.get("children", [])
        return children[:max_children] if isinstance(children, list) else []
    except Exception:
        return []


def openai_next_nodes_o1preview(client, *, code: str, parent: NodeData, seen_ids: List[str], policy: str, max_children: int, model: str = "o1-preview", reasoning_effort: str = "medium") -> List[LLMChild]:
    """Specialized helper for **o1-preview**.

    - o1-family prioritizes internal reasoning; many sampling params are ignored. We omit `temperature`.
    - We set `reasoning_effort` ("low"|"medium"|"high").
    - Uses the **developer** role for policy instructions.
    - Enforces strict JSON via `response_format={"type":"json_object"}`.
    """
    system = (
        "You are a precise Java code analysis assistant. Work ONLY with the code provided. "
        "We are building a DFS-style call/analysis tree. Given a PARENT node, propose up to N CHILDREN "
        "that a human would inspect next to understand execution/data flow."
    )
    rules = (
        "Return STRICT JSON with key 'children'. Each child must have: id, kind, name, role. "
        "Optionally include: signature, note, confidence (0..1), expand (true/false). "
        "Ids must be STABLE and DEDUP-friendly, e.g., 'method:Class.method(int,String)'. "
        "Policy: " + policy + " "
        "Do NOT invent code. Only infer from the given snippet."
    )
    user = {
        "code": code,
        "parent": asdict(parent),
        "seen_ids": seen_ids,
        "max_children": max_children,
    }
    try:
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system},
                {"role": "developer", "content": rules},
                {"role": "user", "content": json.dumps(user)},
            ],
            response_format={"type": "json_object"},
            reasoning_effort=reasoning_effort,
        )
        data = json.loads(resp.output_text.strip())
        children = data.get("children", [])
        return children[:max_children] if isinstance(children, list) else []
    except Exception:
        return []


# -------------------------- Azure OpenAI (LangChain) --------------------------
"""Helpers to use **azureopenai** via `langchain-openai` as your LLM backend.

These wrappers accept a *constructed* LangChain model (e.g., `AzureChatOpenAI`),
so you control creds, endpoint, api_version, deployment name, temperature, and
**response_format** at initialization time.

Recommended model init (strict JSON):

>>> from langchain_openai import AzureChatOpenAI
>>> llm = AzureChatOpenAI(
...     azure_endpoint="https://YOUR-RESOURCE.openai.azure.com/",
...     api_key="...",
...     api_version="2024-10-21",
...     azure_deployment="o3-mini",  # or your deployment name
...     model_kwargs={"response_format": {"type": "json_object"}},
...     temperature=0.2,
... )
>>> agent = LLMJavaCallTree(CODE, "start",
...     next_nodes_fn=lambda **kw: azurelc_next_nodes(llm, **kw))
>>> g = agent.build()
"""
from typing import Any as _Any
try:  # optional dependency
    from langchain_openai import AzureChatOpenAI  # type: ignore
    from langchain_core.messages import SystemMessage, HumanMessage  # type: ignore
except Exception:  # pragma: no cover
    AzureChatOpenAI = _Any  # type: ignore
    SystemMessage = _Any  # type: ignore
    HumanMessage = _Any  # type: ignore


def azurelc_next_nodes(
    model: "AzureChatOpenAI",
    *,
    code: str,
    parent: NodeData,
    seen_ids: List[str],
    policy: str,
    max_children: int,
) -> List[LLMChild]:
    """Generic LangChain **AzureChatOpenAI** helper.

    Parameters
    ----------
    model : AzureChatOpenAI
        A preconfigured LangChain model instance. For strict JSON, set
        `model_kwargs={"response_format": {"type": "json_object"}}` when
        creating it.

    Returns
    -------
    List[LLMChild]
        Up to `max_children` child dicts.
    """
    system = (
        "You are a precise Java code analysis assistant. Work ONLY with the code provided. "
        "We are building a DFS-style call/analysis tree. Given a PARENT node, propose up to N CHILDREN "
        "that a human would inspect next to understand execution/data flow."
    )
    rules = (
        "Return STRICT JSON with key 'children'. Each child must have: id, kind, name, role. "
        "Optionally include: signature, note, confidence (0..1), expand (true/false). "
        "Ids must be STABLE and DEDUP-friendly, e.g., 'method:Class.method(int,String)'. "
        "Policy: " + policy + " "
        "Do NOT invent code. Only infer from the given snippet."
    )
    user_payload = {
        "code": code,
        "parent": asdict(parent),
        "seen_ids": seen_ids,
        "max_children": max_children,
    }

    try:
        msgs = [
            SystemMessage(content=system + "

" + rules),
            HumanMessage(content=json.dumps(user_payload)),
        ]
        ai_msg = model.invoke(msgs)
        # LangChain AIMessage.content may be a string or list of parts; coerce to str
        content = getattr(ai_msg, "content", ai_msg)
        if isinstance(content, list):
            content = "".join(str(p) for p in content)
        data = json.loads(str(content).strip())
        children = data.get("children", [])
        return children[:max_children] if isinstance(children, list) else []
    except Exception:
        return []


def azurelc_next_nodes_o3mini(
    model: "AzureChatOpenAI",
    *,
    code: str,
    parent: NodeData,
    seen_ids: List[str],
    policy: str,
    max_children: int,
) -> List[LLMChild]:
    """Convenience wrapper for an Azure **o3-mini** deployment.

    Just pass your already-initialized `AzureChatOpenAI` bound to the o3-mini
    deployment. This function simply calls `azurelc_next_nodes`.
    """
    return azurelc_next_nodes(
        model,
        code=code,
        parent=parent,
        seen_ids=seen_ids,
        policy=policy,
        max_children=max_children,
    )


def azurelc_next_nodes_o1preview(
    model: "AzureChatOpenAI",
    *,
    code: str,
    parent: NodeData,
    seen_ids: List[str],
    policy: str,
    max_children: int,
) -> List[LLMChild]:
    """Convenience wrapper for an Azure **o1-preview** deployment.

    Azure chat-completions may not expose `reasoning_effort`; rely on your
    deployment config. For strict JSON, set response_format in `model_kwargs`.
    """
    return azurelc_next_nodes(
        model,
        code=code,
        parent=parent,
        seen_ids=seen_ids,
        policy=policy,
        max_children=max_children,
    )
