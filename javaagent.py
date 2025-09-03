"""
Java Call-Stack DFS Builder with Heuristic "AI Agent"
------------------------------------------------------

This module parses a Java source string and builds a DFS-style execution
analysis tree starting from a user-provided symbol (usually a method).

Design highlights
- Tree-sitter for fast Java parsing (no execution required)
- Heuristic agent that emulates how a human expands the next relevant node:
  parameters → receiver/context → local new objects → method calls → control flow
- Cycle / recursion detection using stable node identities
- Exports a NetworkX DiGraph plus a JSON-friendly dump

Dependencies (install via pip):
    pip install tree-sitter tree-sitter-languages networkx

Notes on Java grammar source:
- We prefer `tree_sitter_languages.get_language('java')` (prebuilt grammars)
- Fallback to `tree_sitter.Language('build/my-languages.so', 'java')` if you
  maintain your own compiled bundle.

This is a pragmatic, extensible baseline. You can plug in your own LLM policy by
supplying a callable to AgentPolicy.llm_decider (optional).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import json
import textwrap

import networkx as nx
from tree_sitter import Parser, Node, Query, QueryCursor

# --- Load Java grammar ---
try:
    from tree_sitter_languages import get_language  # type: ignore
    JAVA = get_language("java")
except Exception:  # pragma: no cover - fallback path
    from tree_sitter import Language  # type: ignore
    # Expect user to have built a languages bundle; see Tree-sitter docs
    JAVA = Language("build/my-languages.so", "java")

# --- Helpers ---

def _q(pattern: str) -> Query:
    """Compile a Tree-sitter query for Java, with dedent and robust errors."""
    try:
        return Query(JAVA, textwrap.dedent(pattern))
    except Exception as e:
        raise RuntimeError(f"Query compile failed: {e}\nPattern:\n{pattern}")


def _text(node: Node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _span(node: Node) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Return ((start_row, start_col), (end_row, end_col)) for stable identity/debug."""
    return ((node.start_point[0], node.start_point[1]), (node.end_point[0], node.end_point[1]))


def _body_node(n: Node) -> Optional[Node]:
    """Return the executable body node for declarations (method, ctor, class bodies, etc.)."""
    if n.type in ("block", "class_body", "enum_body", "interface_body", "record_body"):
        return n
    child = n.child_by_field_name("body")
    if child is not None:
        return child
    # Some declarations may store a body under a different field for lambdas
    if n.type == "lambda_expression":
        b = n.child_by_field_name("body")
        return b
    return None


# --- Query Pack (kept narrow and composable to avoid QueryError explosions) ---
Q_METHOD_DECL = _q(
    r"""
    (method_declaration
      name: (identifier) @method.name
    ) @method.decl
    """
)

Q_CONSTRUCTOR_DECL = _q(
    r"""
    (constructor_declaration
      name: (identifier) @ctor.name
    ) @ctor.decl
    """
)

Q_CLASS_DECL = _q(
    r"""
    (class_declaration name: (identifier) @class.name) @class.decl
    (interface_declaration name: (identifier) @iface.name) @iface.decl
    (enum_declaration name: (identifier) @enum.name) @enum.decl
    (record_declaration name: (identifier) @record.name) @record.decl
    """
)

Q_FORMAL_PARAMS = _q(
    r"""
    ; Typical formal parameter (explicit type)
    (formal_parameter
      (variable_declarator_id (identifier) @param.name)
      type: (_) @param.type
    ) @param.decl

    ; Varargs
    (spread_parameter
      (variable_declarator_id (identifier) @param.name)
      type: (_) @param.type
    ) @param.decl
    """
)

Q_LOCAL_VARS = _q(
    r"""
    (local_variable_declaration
      type: (_) @var.type
      (variable_declarator
        name: (identifier) @var.name
        value: (_) @var.init)?
      ) @var.decl
    """
)

Q_OBJECT_CREATION = _q(
    r"""
    ; new ClassName(...)
    (object_creation_expression
      (type_identifier) @new.class
      (argument_list) @new.args
    ) @new.node

    ; new pkg.ClassName(...)
    (object_creation_expression
      (scoped_type_identifier
        (identifier) @new.class)
      (argument_list) @new.args
    ) @new.node
    """
)

Q_METHOD_INVOKE = _q(
    r"""
    (method_invocation
      name: (identifier) @call.name
      (argument_list) @call.args
    ) @call.node
    """
)

Q_METHOD_CHAIN = _q(
    r"""
    ; capture chained calls like a.b().c().d()
    (field_access
      object: (_) @chain.object
      field: (identifier) @chain.field
    ) @chain.node
    """
)

Q_LAMBDA = _q(
    r"""
    (lambda_expression) @lambda.node
    """
)

Q_CTRL = _q(
    r"""
    (if_statement) @ctrl.if
    (for_statement) @ctrl.for
    (enhanced_for_statement) @ctrl.foreach
    (while_statement) @ctrl.while
    (switch_expression) @ctrl.switch
    (switch_statement) @ctrl.switch
    (try_statement) @ctrl.try
    """
)

Q_METHOD_REF = _q(
    r"""
    (method_reference) @mref.node
    """
)

Q_RETURN = _q(
    r"""
    (return_statement (expression) @ret.expr) @ret.node
    """
)

Q_IMPORTS = _q(
    r"""
    (import_declaration (scoped_identifier) @imp.fqn) @imp.node
    (import_declaration (identifier) @imp.fqn) @imp.node
    """
)

# --- Data Model ---
@dataclass(frozen=True)
class NodeKey:
    kind: str
    name: str
    start: Tuple[int, int]
    end: Tuple[int, int]


@dataclass
class NodeData:
    kind: str  # method|ctor|class|var|new|call|lambda|ctrl|mref|return|param
    name: str
    role: str  # root|param|receiver|local|call|ctor|lambda|ctrl|arg|return
    fqn: Optional[str] = None
    signature: Optional[str] = None
    static_type: Optional[str] = None
    origin: str = "local"  # local|external
    source_range: Tuple[Tuple[int, int], Tuple[int, int]] = field(default_factory=lambda: ((-1, -1), (-1, -1)))
    meta: Dict[str, Any] = field(default_factory=dict)


class AgentPolicy:
    """Heuristic policy that picks next relevant children in a *strict, deterministic*
    order that mirrors human reading, with optional LLM re-ranking.

    The default deterministic order is:
        params → receiver → locals → news → calls → chains → lambdas → ctrl → returns → mrefs

    If an LLM decider is provided, we first apply the deterministic sort and then
    let the model return a permutation over those candidates. If the call fails,
    we keep the deterministic order.
    """

    # Strict expansion order (category names are attached in NodeData.meta["category"])
    ORDER = ("params", "receiver", "locals", "news", "calls", "chains", "lambdas", "ctrl", "returns", "mrefs")

    def __init__(self, llm_decider=None):
        self.llm_decider = llm_decider

    def _cat_index(self, meta: Dict[str, Any]) -> int:
        cat = (meta or {}).get("category", "zzz")
        try:
            return self.ORDER.index(cat)
        except ValueError:
            return len(self.ORDER)

    def choose(self, candidates: List[Tuple[NodeData, Node]], src: bytes) -> List[Tuple[NodeData, Node]]:
        # 1) Deterministic: by category then by source position
        candidates.sort(key=lambda t: (self._cat_index(t[0].meta), t[1].start_byte))
        if self.llm_decider is None:
            return candidates
        # 2) Optional: allow LLM to reorder (returns indices into the *current* list)
        try:  # pragma: no cover - optional branch
            order = self.llm_decider(candidates, src)
            if order and all(isinstance(i, int) and 0 <= i < len(candidates) for i in order):
                candidates = [candidates[i] for i in order]
        except Exception:
            pass
        return candidates


class JavaAnalyzer:
    """Parse Java and build a DFS call/analysis tree from a starting symbol.

    Parameters
    ----------
    code : str
        Java source text (single file snippet is fine).
    start_symbol : str
        Name of the method/ctor/class used as the *root* of expansion.
    file_name : str, optional
        Only used for metadata/debug.
    policy : AgentPolicy, optional
        Custom ordering policy. By default, uses deterministic category order.

    Notes
    -----
    The expansion is *static* (no execution). We resolve local declarations and
    approximate types/arity. External calls become leaf nodes unless a matching
    declaration exists in the same source.

    Examples
    --------
    >>> az = JavaAnalyzer(EXAMPLE_JAVA, start_symbol="start")
    >>> g = az.build_call_tree()  # deterministic (no LLM)
    >>> out = az.to_json(g)
    >>> isinstance(out["nodes"], list) and isinstance(out["edges"], list)
    True
    """
    def __init__(self, code: str, start_symbol: str, *, file_name: str = "Main.java", policy: Optional[AgentPolicy] = None):
        self.code = code
        self.src = code.encode("utf-8")
        self.start_symbol = start_symbol
        self.file_name = file_name
        self.policy = policy or AgentPolicy()

        p = Parser()
        p.set_language(JAVA)
        self.tree = p.parse(self.src)
        self.root = self.tree.root_node

    # ------------- Public API -------------
    def build_call_tree(self) -> nx.DiGraph:
        """Build a DFS call/analysis tree starting from ``start_symbol``.

        Returns
        -------
        networkx.DiGraph
            Directed graph where edges are ``kind='flows_to'`` and each node has
            ``data``: :class:`NodeData`.

        Example
        -------
        >>> az = JavaAnalyzer(EXAMPLE_JAVA, start_symbol="start")
        >>> g = az.build_call_tree()
        >>> j = az.to_json(g)
        >>> len(j["nodes"]) > 0 and len(j["edges"]) > 0
        True
        """
        start_decl = self._find_start_decl(self.start_symbol)
        if start_decl is None:
            raise ValueError(f"Start symbol '{self.start_symbol}' not found as method/ctor/class")

        g = nx.DiGraph()
        root_data = self._node_data_for_decl(start_decl)
        root_key = self._key_for(start_decl, root_data.kind, (root_data.name + (f"/{root_data.signature}" if root_data.signature else "")))
        g.add_node(root_key, data=root_data)

        visited: Set[NodeKey] = {root_key}
        stack: List[Tuple[NodeKey, Node]] = [(root_key, start_decl)]

        while stack:
            parent_key, parent_node = stack.pop()
            parent_data: NodeData = g.nodes[parent_key]["data"]

            # Determine relevant children for parent
            children = self._relevant_children(parent_node, parent_data)
            chosen = self.policy.choose(children, self.src)

            for child_data, child_node in chosen:
                key = self._key_for(child_node, child_data.kind, (child_data.name + (f"/{child_data.signature}" if child_data.signature else "")))
                if key in visited:
                    # Just add edge to existing node to avoid duplication
                    if not g.has_node(key):
                        # In rare cases if key was visited but node removed (not expected)
                        g.add_node(key, data=child_data)
                    g.add_edge(parent_key, key, kind="flows_to")
                    continue

                g.add_node(key, data=child_data)
                g.add_edge(parent_key, key, kind="flows_to")
                visited.add(key)

                # Only expand entities that introduce local executable context
                if child_data.kind in {"method", "ctor", "lambda", "ctrl"}:
                    stack.append((key, child_node))
                elif child_data.kind == "call":
                    # If target method is declared locally, expand into its body; otherwise leaf
                    decl = self._resolve_local_method(child_node)
                    if decl is not None:
                        stack.append((key, decl))

        return g

    def to_json(self, g: nx.DiGraph) -> Dict[str, Any]:
        nodes = []
        id_map: Dict[NodeKey, int] = {}
        for i, (n, attrs) in enumerate(g.nodes(data=True)):
            id_map[n] = i
            nodes.append({"id": i, **asdict(attrs["data"])})
        edges = [{"src": id_map[u], "dst": id_map[v], **attrs} for u, v, attrs in g.edges(data=True)]
        return {"nodes": nodes, "edges": edges}

    # ------------- Internals -------------
    def _find_start_decl(self, name: str) -> Optional[Node]:
        # Try method
        for m in self._captures(Q_METHOD_DECL, self.root):
            if _text(m["method.name"], self.src) == name:
                return m["method.decl"]
        # Try constructor
        for c in self._captures(Q_CONSTRUCTOR_DECL, self.root):
            if _text(c["ctor.name"], self.src) == name:
                return c["ctor.decl"]
        # Try class/interface/enum/record
        for c in self._captures(Q_CLASS_DECL, self.root):
            for tag in ("class", "iface", "enum", "record"):
                nm = c.get(f"{tag}.name")
                if nm and _text(nm, self.src) == name:
                    return c[f"{tag}.decl"]
        return None

    def _node_data_for_decl(self, decl: Node) -> NodeData:
        if decl.type == "method_declaration":
            name = self._field_text(decl, "name")
            sig = self._signature_for_method(decl)
            return NodeData(kind="method", name=name, role="root", signature=sig, source_range=_span(decl))
        if decl.type == "constructor_declaration":
            name = self._field_text(decl, "name")
            sig = self._signature_for_method(decl)
            return NodeData(kind="ctor", name=name, role="root", signature=sig, source_range=_span(decl))
        # Class-like
        name = self._field_text(decl, "name")
        return NodeData(kind="class", name=name, role="root", source_range=_span(decl))

    def _relevant_children(self, node: Node, parent_data: NodeData) -> List[Tuple[NodeData, Node]]:
        out: List[Tuple[NodeData, Node]] = []

        # 1) Parameters for methods/ctors/lambdas
        if node.type in {"method_declaration", "constructor_declaration"}:
            for cap in self._captures(Q_FORMAL_PARAMS, node):
                p_name = _text(cap["param.name"], self.src)
                p_type = _text(cap["param.type"], self.src)
                data = NodeData(kind="param", name=p_name, role="param", static_type=p_type, source_range=_span(cap["param.decl"]), meta={"category": "params"})
                out.append((data, cap["param.decl"]))

        if node.type == "lambda_expression":
            # Represent lambda parameters as a single node for brevity
            data = NodeData(kind="param", name="lambda.params", role="param", source_range=_span(node), meta={"category": "params"})
            out.append((data, node))

        # 2) Receiver context (class of 'this') for methods inside classes
        # We approximate by adding the enclosing class declaration if present
        encl_class = self._enclosing_class(node)
        if encl_class is not None and node.type in {"method_declaration", "constructor_declaration", "lambda_expression"}:
            c_name = self._field_text(encl_class, "name")
            data = NodeData(kind="class", name=c_name, role="receiver", source_range=_span(encl_class), meta={"category": "receiver"})
            out.append((data, encl_class))

        # For any node with a body, scan the body block for locals/news/calls/ctrl
        body = _body_node(node)
        block = body if body is not None else node

        # 3) Local variable declarations
        for cap in self._captures(Q_LOCAL_VARS, block):
            vname = _text(cap["var.name"], self.src)
            vtype = _text(cap.get("var.type", cap["var.decl"]) or cap["var.decl"], self.src)
            data = NodeData(kind="var", name=vname, role="local", static_type=vtype, source_range=_span(cap["var.decl"]), meta={"category": "locals"})
            out.append((data, cap["var.decl"]))

        # 4) Object creations (constructors)
        for cap in self._captures(Q_OBJECT_CREATION, block):
            cname = _text(cap["new.class"], self.src)
            argc = self._arity_of_args(cap["new.args"])
            sig = f"{cname}({argc})"
            data = NodeData(kind="new", name=cname, role="ctor", signature=sig, source_range=_span(cap["new.node"]), meta={"category": "news"})
            out.append((data, cap["new.node"]))

        # 5) Method invocations
        for cap in self._captures(Q_METHOD_INVOKE, block):
            mname = _text(cap["call.name"], self.src)
            argc = self._arity_of_args(cap["call.args"])
            sig = f"{mname}({argc})"
            data = NodeData(kind="call", name=mname, role="call", signature=sig, source_range=_span(cap["call.node"]), meta={"category": "calls"})
            out.append((data, cap["call.node"]))

        # 6) Chains (field accesses in fluent chains) - heuristic only
        for cap in self._captures(Q_METHOD_CHAIN, block):
            fname = _text(cap["chain.field"], self.src)
            data = NodeData(kind="call", name=fname, role="call", source_range=_span(cap["chain.node"]), meta={"category": "chains"})
            out.append((data, cap["chain.node"]))

        # 7) Lambdas
        for cap in self._captures(Q_LAMBDA, block):
            data = NodeData(kind="lambda", name="lambda", role="lambda", source_range=_span(cap["lambda.node"]), meta={"category": "lambdas"})
            out.append((data, cap["lambda.node"]))

        # 8) Control flow regions
        for cap in self._captures(Q_CTRL, block):
            # Choose a stable name based on type
            tag = next((k for k in cap.keys() if k.startswith("ctrl.")), None)
            if tag:
                ctrl_node = cap[tag]
                data = NodeData(kind="ctrl", name=ctrl_node.type, role="ctrl", source_range=_span(ctrl_node), meta={"category": "ctrl"})
                out.append((data, ctrl_node))

        # 9) Returns
        for cap in self._captures(Q_RETURN, block):
            data = NodeData(kind="return", name="return", role="return", source_range=_span(cap["ret.node"]), meta={"category": "returns"})
            out.append((data, cap["ret.node"]))

        # 10) Method references
        for cap in self._captures(Q_METHOD_REF, block):
            data = NodeData(kind="mref", name="method_ref", role="call", source_range=_span(cap["mref.node"]), meta={"category": "mrefs"})
            out.append((data, cap["mref.node"]))

        return out

    def _resolve_local_method(self, call_node: Node) -> Optional[Node]:
        """Best-effort static resolution of a method call to a local declaration.
        We match by name and arity within the same file. Overloads are matched by arity.
        """
        name_field = call_node.child_by_field_name("name")
        args_field = call_node.child_by_field_name("arguments") or call_node.child_by_field_name("argument_list")
        if not name_field:
            return None
        name = _text(name_field, self.src)
        argc = self._arity_of_args(args_field) if args_field else 0

        # Search methods
        best: Optional[Node] = None
        for cap in self._captures(Q_METHOD_DECL, self.root):
            if _text(cap["method.name"], self.src) != name:
                continue
            if self._arity_of_method(cap["method.decl"]) == argc:
                return cap["method.decl"]
            best = best or cap["method.decl"]
        # Constructors
        for cap in self._captures(Q_CONSTRUCTOR_DECL, self.root):
            if _text(cap["ctor.name"], self.src) != name:
                continue
            if self._arity_of_method(cap["ctor.decl"]) == argc:
                return cap["ctor.decl"]
            best = best or cap["ctor.decl"]
        return best

    def _enclosing_class(self, n: Node) -> Optional[Node]:
        cur = n
        while cur is not None:
            if cur.type in {"class_declaration", "interface_declaration", "enum_declaration", "record_declaration"}:
                return cur
            cur = cur.parent
        return None

    def _field_text(self, n: Node, field: str) -> str:
        c = n.child_by_field_name(field)
        return _text(c, self.src) if c is not None else ""

    def _arity_of_method(self, decl: Node) -> int:
        params = decl.child_by_field_name("parameters")
        if params is None:
            return 0
        # Count commas + 1 within parameter list; simple heuristic
        txt = _text(params, self.src)
        # Remove generics and annotations for stability
        return 0 if txt.strip() in ("()", "") else txt.count(",") + 1

    def _arity_of_args(self, arglist: Optional[Node]) -> int:
        if arglist is None:
            return 0
        txt = _text(arglist, self.src)
        content = txt.strip()
        if not content or content in ("()", "<>"):
            return 0
        # handle nested generics crudely by counting top-level commas
        depth = 0
        count = 1
        for ch in content:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(depth - 1, 0)
            elif ch == "," and depth <= 1:
                count += 1
        # Rough fallback
        return max(count, 1)

    def _key_for(self, n: Node, kind: str, name: str) -> NodeKey:
        """Create a stable identity for de-duplication and cycle detection.

        We include the source span so that two different sites with the same
        symbol don't collide; and we optionally suffix the name with a
        signature upstream (caller passes name+"/"+signature when present).
        """
        (sr, sc), (er, ec) = _span(n)
        return NodeKey(kind=kind, name=name, start=(sr, sc), end=(er, ec))

    # Query exec that returns a list of dicts mapping capture name -> Node
    def _captures(self, query: Query, root: Node) -> List[Dict[str, Node]]:
        cursor = QueryCursor()
        res: List[Dict[str, Node]] = []
        for m, caps in cursor.exec(query, root):  # type: ignore[attr-defined]
            d: Dict[str, Node] = {}
            for i, cap in enumerate(caps):
                name = query.captures[cap.index]
                d[name] = cap.node
            res.append(d)
        return res


# ------------- Example usage (for quick manual testing) -------------
EXAMPLE_JAVA = r"""
import java.util.*;

public class Demo {
    public static void main(String[] args) {
        Demo d = new Demo();
        d.start(42, "hi");
    }

    public void start(int n, String msg) {
        Helper h = new Helper(msg);
        int x = compute(n).map(v -> v + 1).orElse(0);
        if (x > 10) {
            log(msg);
        } else {
            for (int i = 0; i < x; i++) doThing(i);
        }
    }

    static int compute(int z) { return z * 2; }
    void doThing(int k) { System.out.println(k); }
    void log(String s) { System.out.println(s); }
}

class Helper {
    Helper(String s) {}
}
"""

if __name__ == "__main__":  # pragma: no cover
    analyzer = JavaAnalyzer(EXAMPLE_JAVA, start_symbol="start")
    g = analyzer.build_call_tree()
    data = analyzer.to_json(g)
    print(json.dumps(data, indent=2))


# -------------------- LangGraph + OpenAI Wiring --------------------
"""
Optional integration with an agentic orchestration library. We use LangGraph to
coordinate a looped DFS step that calls an LLM (OpenAI Responses API) to
prioritize candidates. This keeps the control flow deterministic while letting
an LLM nudge ordering when ties exist.

Install:
    pip install langgraph openai

Env:
    export OPENAI_API_KEY=sk-...

Usage (example):
    from openai import OpenAI
    from langgraph.checkpoint.memory import MemorySaver

    client = OpenAI()
    analyzer = JavaAnalyzer(EXAMPLE_JAVA, start_symbol="start")
    g = build_call_tree_with_langgraph(analyzer, client, model="gpt-4o")
    print(json.dumps(analyzer.to_json(g), indent=2))
"""
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from openai import OpenAI


class TreeState(TypedDict):
    done: bool
    # A simple stack of (NodeKey, Node) encoded as tuples we can serialize
    stack: List[Tuple[NodeKey, Node]]  # type: ignore[type-arg]
    visited: Set[NodeKey]  # type: ignore[type-arg]
    graph: nx.DiGraph
    analyzer: "JavaAnalyzer"
    model: str
    client: OpenAI
    # Optional maximum nodes to avoid runaway in adversarial inputs
    max_nodes: int


class LLMSelector:
    """Reorders candidate children using an OpenAI model via Responses API.

    If the call fails for any reason, returns candidates unchanged.
    """

    def __init__(self, client: OpenAI, model: str = "gpt-4o") -> None:
        self.client = client
        self.model = model

    def reorder(self, code: str, parent_summary: str, candidates: List[Tuple[NodeData, Node]]) -> List[int]:
        # Build a compact prompt with numbered options
        def summarize(cd: NodeData, node: Node) -> str:
            seg = code[node.start_byte:node.end_byte]
            seg = seg[:200].decode("utf-8", errors="replace")
            return f"kind={cd.kind}, role={cd.role}, name={cd.name}, sig={cd.signature or ''}, type={cd.static_type or ''}, code_snip={seg!r}"

        options = [summarize(cd, n) for (cd, n) in candidates]
        system = (
            "You are a precise Java code analysis assistant. Given a parent entity and multiple "
            "candidate next nodes, return the best processing order to mimic a human DFS analysis. "
            "Prioritize: parameters (L->R), receiver class context, local var decls, new object ctors, "
            "direct calls in source order, chained calls, lambdas, control-flow blocks, returns, method refs. "
            "Avoid duplicates or obviously irrelevant picks. Return ONLY a JSON array of indices."
        )
        user = (
            "Parent: " + parent_summary + "
" +
            "Candidates (0-based):
" +
            "
".join(f"[{i}] {opt}" for i, opt in enumerate(options)) +
            "
Output MUST be JSON array of integers (e.g., [0,2,1])."
        )
        try:
            resp = self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            text = resp.output_text.strip()
            # Very defensive parse
            import json as _json
            order = _json.loads(text)
            if isinstance(order, list) and all(isinstance(i, int) and 0 <= i < len(candidates) for i in order):
                return order
        except Exception:
            pass
        # Fallback: identity order
        return list(range(len(candidates)))


def _parent_summary(analyzer: "JavaAnalyzer", node: Node, data: NodeData) -> str:
    return f"kind={data.kind}, name={data.name}, role={data.role}, sig={data.signature or ''}, span={_span(node)}"


def _encode_key(key: NodeKey) -> Tuple[str, str, Tuple[int, int], Tuple[int, int]]:
    return (key.kind, key.name, key.start, key.end)


def _decode_key(t: Tuple[str, str, Tuple[int, int], Tuple[int, int]]) -> NodeKey:
    return NodeKey(kind=t[0], name=t[1], start=t[2], end=t[3])


def _make_initial_state(analyzer: "JavaAnalyzer", client: OpenAI, model: str, *, max_nodes: int = 500) -> TreeState:
    start_decl = analyzer._find_start_decl(analyzer.start_symbol)
    if start_decl is None:
        raise ValueError(f"Start symbol '{analyzer.start_symbol}' not found")
    g = nx.DiGraph()
    root_data = analyzer._node_data_for_decl(start_decl)
    root_key = analyzer._key_for(start_decl, root_data.kind, root_data.name)
    g.add_node(root_key, data=root_data)
    return TreeState(
        done=False,
        stack=[(root_key, start_decl)],
        visited={root_key},
        graph=g,
        analyzer=analyzer,
        model=model,
        client=client,
        max_nodes=max_nodes,
    )


def _step(state: TreeState) -> TreeState:
    if not state["stack"]:
        state["done"] = True
        return state

    parent_key, parent_node = state["stack"].pop()
    analyzer = state["analyzer"]
    g = state["graph"]
    parent_data: NodeData = g.nodes[parent_key]["data"]

    candidates = analyzer._relevant_children(parent_node, parent_data)

    # LangGraph state is Python; call OpenAI reordering if available
    selector = LLMSelector(state["client"], state["model"]) if state.get("client") else None
    order = list(range(len(candidates)))
    if selector and candidates:
        order = selector.reorder(analyzer.code, _parent_summary(analyzer, parent_node, parent_data), candidates)

    for idx in order:
        child_data, child_node = candidates[idx]
        key = analyzer._key_for(child_node, child_data.kind, child_data.name)
        if key in state["visited"]:
            if not g.has_node(key):
                g.add_node(key, data=child_data)
            g.add_edge(parent_key, key, kind="flows_to")
            continue
        # Guardrail: prevent runaway graphs
        if g.number_of_nodes() >= state["max_nodes"]:
            state["done"] = True
            break
        g.add_node(key, data=child_data)
        g.add_edge(parent_key, key, kind="flows_to")
        state["visited"].add(key)

        if child_data.kind in {"method", "ctor", "lambda", "ctrl"}:
            state["stack"].append((key, child_node))
        elif child_data.kind == "call":
            decl = analyzer._resolve_local_method(child_node)
            if decl is not None:
                state["stack"].append((key, decl))

    if not state["stack"]:
        state["done"] = True
    return state


def _should_continue(state: TreeState) -> Literal["continue", "end"]:
    return "end" if state["done"] else "continue"


def build_call_tree_with_langgraph(analyzer: "JavaAnalyzer", client: OpenAI, model: str = "gpt-4o", *, max_nodes: int = 500) -> nx.DiGraph:
    """Run the DFS builder under a LangGraph loop with optional OpenAI re-ordering.

    Parameters
    ----------
    analyzer : JavaAnalyzer
        Initialized analyzer with parsed tree and start symbol.
    client : openai.OpenAI
        OpenAI client (Responses API).
    model : str, default "gpt-4o"
        OpenAI model name for ranking candidates.
    max_nodes : int, default 500
        Hard cap to prevent runaway expansion on adversarial inputs.

    Returns
    -------
    networkx.DiGraph
        Built call/analysis graph.

    Example
    -------
    >>> from openai import OpenAI
    >>> client = OpenAI()
    >>> az = JavaAnalyzer(EXAMPLE_JAVA, start_symbol="start")
    >>> g = build_call_tree_with_langgraph(az, client, model="gpt-4o")
    >>> isinstance(g.number_of_nodes(), int) and g.number_of_nodes() > 0
    True
    """
    sg = StateGraph(TreeState)
    sg.add_node("step", _step)
    sg.add_edge(START, "step")
    sg.add_conditional_edges("step", _should_continue, {"continue": "step", "end": END})

    memory = MemorySaver()
    app = sg.compile(checkpointer=memory)

    init = _make_initial_state(analyzer, client, model, max_nodes=max_nodes)

    # Execute until end
    final_state = app.invoke(init)
    return final_state["graph"]
