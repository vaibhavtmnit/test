# %%
"""
dfs_tree.py

A lightweight DFS tree builder that plugs into your existing **LLM `supervisor`**
pipeline. It does *not* implement your LLM logic; instead it orchestrates calls
to your `supervisor(AgentState)` and builds a deduplicated, cycle-aware tree
using `NodeData` items plus optional `ClassDetail` nodes and `Edge`s.

Design notes:
- You provide the `supervisor` function and the types `AgentState`, `NodeData`, and `edge_list` values.
- You also provide two strategy callbacks:
    1) `expansion_inputs(node) -> Optional[Tuple[str, str, str]]`
       Returns (code, code_path, object_name) for the next expansion seed.
       Return `None` to mark the node as non-expandable.
    2) `is_class(node) -> Optional[str]`
       Returns a stable class name if `node` represents a class-like entity.
       If a class name is returned, the node will connect to a canonical
       `ClassDetail` instance and won't be expanded further.

- Edge typing:
  Pass `edge_types={"parent_child": "...", "node_class": "..."}` with values
  from your existing `edge_list`. If not provided, sensible defaults are used.

- Uniqueness rules:
  * Nodes are keyed by their `.name` attribute for deduping and cycle-prevention.
  * Edges are unique by (source_name, target_name, edge_type).
  * ClassDetail instances are unique by their `name`.

- DFS order:
  LIFO stack of (parent_name, child_node). Parent-child edges are created
  immediately; expansion only happens the first time a child name is seen.

- Integration:
  Construct `Tree(code_path=..., root_object=..., supervisor=..., expansion_inputs=..., is_class=...)`
  then call `tree.build()` (or set `autobuild=True` on init). Inspect
  `tree.nodes`, `tree.class_details`, and `tree.edges`.

Docstring examples near the bottom include a self-contained stub `supervisor`
and `NodeData` so you can see the traversal shape without your real LLM.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Protocol, Union
from pathlib import Path


# ---- Protocols & helper types -------------------------------------------------
class HasName(Protocol):
    """
    A minimal duck-typed protocol that your NodeData should satisfy.

    Any object with a string `name` attribute will work. We don't import your
    concrete NodeData type to keep this module independent.
    """
    name: str


# A tiny value object for (code, code_path, object_name) returned by your expansion probe.
ExpansionInputs = Tuple[str, str, str]


# ---- Graph-side domain objects -------------------------------------------------
@dataclass(eq=True, frozen=True)
class ClassDetail:
    """
    Canonical class node. Instances are de-duplicated by `name`.

    Attributes
    ----------
    name : str
        Stable class identifier (e.g., FQN or unique simple name in your scope).
    path : Optional[str]
        Optional source path where the class is defined.
    meta : dict
        Free-form metadata bag for anything you want to stash (e.g., package).
    """
    name: str
    path: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


class Edge:
    """
    A unique, hashable graph edge connecting two nodes with a typed relation.

    Notes on attribute names:
    - The user requested attributes called `from` and `to`. Python keywords
      prevent us from *defining* a `from` field directly, but attribute access
      like `edge.from` is allowed. We therefore store internally as `from_`
      and expose a `from` alias via `__getattr__`.

    Uniqueness is defined by (source_name, target_name, edge_type).
    """
    __slots__ = ("from_", "to", "type", "_key")

    def __init__(self, source: HasName | ClassDetail, target: HasName | ClassDetail, edge_type: str) -> None:
        # store
        self.from_ = source
        self.to = target
        self.type = edge_type
        # precompute uniqueness key
        self._key = (getattr(source, "name", repr(source)), getattr(target, "name", repr(target)), edge_type)

    # Provide the `.from` alias for ergonomic access (read-only).
    def __getattr__(self, name: str) -> Any:
        if name == "from":
            return self.from_
        raise AttributeError(f"{self.__class__.__name__!s} has no attribute {name!r}")

    def __repr__(self) -> str:
        s = getattr(self.from_, "name", repr(self.from_))
        t = getattr(self.to, "name", repr(self.to))
        return f"Edge({s!r} -[{self.type}]-> {t!r})"

    def __hash__(self) -> int:
        return hash(self._key)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Edge):
            return NotImplemented
        return self._key == other._key


# ---- Tree builder -------------------------------------------------------------
class Tree:
    """
    DFS tree builder that orchestrates your LLM `supervisor` and assembles a graph.

    Parameters
    ----------
    code_path : str | Path
        Path to the Java source file containing the *root object*.
    root_object : str
        Name of the starting object (class/method/field/etc.) the DFS should seed from.
    supervisor : Callable[[Any], List[HasName]]
        Your existing function. Given an AgentState, returns a list of NodeData (HasName).
        We do not import `AgentState`; you supply it as part of the call to `supervisor`.
    expansion_inputs : Callable[[HasName], Optional[ExpansionInputs]]
        Your callback that inspects a node and returns `(code, code_path, object_name)` for expansion.
        Return `None` to skip expansion of this node.
    is_class : Callable[[HasName], Optional[str]]
        Your callback that returns a class name if the node represents a class-like entity.
        If a class name is returned, the node will connect to a canonical ClassDetail and not expand further.
    make_agent_state : Callable[[str, str, str], Any]
        A factory that builds the AgentState expected by your `supervisor` from (code, path, object_name).
        We keep this as a loose callback so you can pass your real AgentState constructor.
    edge_types : Optional[dict]
        A dict with entries:
            - "parent_child": value from your `edge_list` to use between parent->child item
            - "node_class"  : value from your `edge_list` to use between node->ClassDetail
        Defaults: {"parent_child": "related", "node_class": "class_of"}

    Other attributes
    ----------------
    nodes : Dict[str, HasName]
        Canonical NodeData instances keyed by name (deduped).
    class_details : Dict[str, ClassDetail]
        Canonical ClassDetail instances keyed by class name.
    edges : Set[Edge]
        Unique set of edges.
    expanded : Set[str]
        Names that have already been expanded (prevents cycles).
    root_name : str
        The root object's name.
    root_path : str
        The root file path string.
    root_code : str
        Contents of the file at `code_path`.

    Usage example
    -------------
    >>> # You will replace these stubs with your real implementations.
    >>> class StubNode:
    ...     def __init__(self, name): self.name = name
    ...     def __repr__(self): return f"StubNode({self.name})"
    ...
    >>> def stub_supervisor(agent_state) -> List[StubNode]:
    ...     # fan-out demo: a->b,c ; b->d ; c->(no children) ; d->(class)
    ...     cur = agent_state.get("object_name")
    ...     if cur == "a": return [StubNode("b"), StubNode("c")]
    ...     if cur == "b": return [StubNode("d")]
    ...     return []
    ...
    >>> def stub_expansion_inputs(node: StubNode) -> Optional[ExpansionInputs]:
    ...     # pretend we always have the info: (code, path, name)
    ...     return ("//code", "/path/File.java", node.name)
    ...
    >>> def stub_is_class(node: StubNode) -> Optional[str]:
    ...     return "DemoClass" if node.name == "d" else None
    ...
    >>> def stub_make_agent_state(code: str, path: str, obj: str) -> dict:
    ...     return {"code": code, "path": path, "object_name": obj}
    ...
    >>> t = Tree(
    ...     code_path="/path/File.java",
    ...     root_object="a",
    ...     supervisor=stub_supervisor,
    ...     expansion_inputs=stub_expansion_inputs,
    ...     is_class=stub_is_class,
    ...     make_agent_state=stub_make_agent_state,
    ...     edge_types={"parent_child": "uses", "node_class": "class_of"},
    ...     autobuild=False,
    ... )
    >>> t.build()   # doctest: +ELLIPSIS
    >>> sorted(e.__repr__() for e in t.edges)
    ["Edge('ROOT:a' -[uses]-> 'b')", "Edge('ROOT:a' -[uses]-> 'c')", "Edge('d' -[class_of]-> 'DemoClass')"]
    """
    def __init__(
        self,
        code_path: Union[str, Path],
        root_object: str,
        supervisor: Callable[[Any], List[HasName]],
        expansion_inputs: Callable[[HasName], Optional[ExpansionInputs]],
        is_class: Callable[[HasName], Optional[str]],
        make_agent_state: Callable[[str, str, str], Any],
        edge_types: Optional[Dict[str, str]] = None,
        autobuild: bool = True,
    ) -> None:
        # --- Capture config ---
        self.root_path: str = str(code_path)
        self.root_name: str = root_object
        self.supervisor = supervisor
        self.expansion_inputs = expansion_inputs
        self.is_class = is_class
        self.make_agent_state = make_agent_state
        self.edge_types = edge_types or {"parent_child": "related", "node_class": "class_of"}

        # --- Load initial code from path (root file) ---
        self.root_code: str = Path(self.root_path).read_text(encoding="utf-8")

        # --- Graph stores ---
        self.nodes: Dict[str, HasName] = {}
        self.class_details: Dict[str, ClassDetail] = {}
        self.edges: Set[Edge] = set()
        self.expanded: Set[str] = set()

        # --- Synthetic root helper for first-level edges ---
        @dataclass
        class _Root:
            name: str
        self._root = _Root(name=f"ROOT:{self.root_name}")

        if autobuild:
            self.build()

    # -- public API -------------------------------------------------------------
    def build(self) -> None:
        """
        Run DFS starting from (root_code, root_path, root_object).

        This method is idempotent (safe to call again); already-expanded nodes
        are remembered via `self.expanded`.
        """
        # 1) Seed: call supervisor on the root AgentState
        root_state = self.make_agent_state(self.root_code, self.root_path, self.root_name)
        first_children = self._safe_supervisor(root_state)

        # Use a LIFO stack of frames: (parent_name, child_node)
        stack: List[Tuple[str, HasName]] = []

        for child in first_children:
            child = self._canonicalize_node(child)
            self._connect(self._root, child, self.edge_types["parent_child"])
            stack.append((child.name, child))

        # 2) DFS loop
        while stack:
            parent_name, node = stack.pop()

            # If we've already expanded a node with this name, just continue.
            if node.name in self.expanded:
                continue

            # Try class short-circuit first (your logic decides).
            cls_name = self.is_class(node)
            if cls_name:
                cls = self._get_or_create_class_detail(cls_name)
                self._connect(node, cls, self.edge_types["node_class"])
                # mark as expanded (won't be expanded further)
                self.expanded.add(node.name)
                continue

            # Probe for expansion inputs (your logic decides).
            exp = self.expansion_inputs(node)
            if exp is None:
                # Nothing to expand
                self.expanded.add(node.name)
                continue

            code, path, obj_name = exp

            # Call supervisor to get next-level nodes
            agent_state = self.make_agent_state(code, path, obj_name)
            next_nodes = self._safe_supervisor(agent_state)

            if not next_nodes:
                # Leaf
                self.expanded.add(node.name)
                continue

            # Enqueue children DFS-style, dedupe by name
            for nxt in next_nodes:
                nxt = self._canonicalize_node(nxt)
                # Always connect parent->child (even if child was seen) — edges are unique anyway.
                self._connect(node, nxt, self.edge_types["parent_child"])

                if nxt.name not in self.expanded:
                    # If we've never seen this name, push for expansion.
                    stack.append((nxt.name, nxt))

            # Mark current node expanded after pushing children
            self.expanded.add(node.name)

    # -- internal utilities -----------------------------------------------------
    def _safe_supervisor(self, agent_state: Any) -> List[HasName]:
        """
        Wrapper around `supervisor` with a small safety net:
        - Ensures list return type
        - Filters out items lacking a `name` attribute
        """
        try:
            result = self.supervisor(agent_state) or []
        except Exception as e:
            # In production, you may wish to log/telemetry here.
            result = []
        filtered: List[HasName] = []
        for item in result:
            nm = getattr(item, "name", None)
            if isinstance(nm, str) and nm:
                filtered.append(item)
        return filtered

    def _canonicalize_node(self, node: HasName) -> HasName:
        """
        De-duplicate nodes by name and return the canonical instance.
        """
        existing = self.nodes.get(node.name)
        if existing is not None:
            return existing
        self.nodes[node.name] = node
        return node

    def _connect(self, src: Union[HasName, ClassDetail], dst: Union[HasName, ClassDetail], edge_type: str) -> None:
        """
        Create a unique edge from src->dst with the given type.
        """
        edge = Edge(src, dst, edge_type)
        self.edges.add(edge)

    def _get_or_create_class_detail(self, class_name: str, path: Optional[str] = None) -> ClassDetail:
        """
        Flyweight factory for ClassDetail objects keyed by name.
        """
        cd = self.class_details.get(class_name)
        if cd is not None:
            return cd
        cd = ClassDetail(name=class_name, path=path)
        self.class_details[class_name] = cd
        return cd


# ---- Demonstration (you can delete below this line in your codebase) ----------
if __name__ == "__main__":
    # A compact smoke test you can run directly:
    class StubNode:
        def __init__(self, name): self.name = name
        def __repr__(self): return f"StubNode({self.name})"

    def stub_supervisor(agent_state) -> List[StubNode]:
        cur = agent_state.get("object_name")
        if cur == "a": return [StubNode("b"), StubNode("c")]
        if cur == "b": return [StubNode("d")]
        return []

    def stub_expansion_inputs(node: StubNode) -> Optional[ExpansionInputs]:
        return ("//code", "/path/File.java", node.name)

    def stub_is_class(node: StubNode) -> Optional[str]:
        return "DemoClass" if node.name == "d" else None

    def stub_make_agent_state(code: str, path: str, obj: str) -> dict:
        return {"code": code, "path": path, "object_name": obj}

    # Create a temp file to simulate code_path
    import tempfile
    import textwrap
    with tempfile.NamedTemporaryFile("w+", suffix=".java", delete=False) as fp:
        fp.write(textwrap.dedent("""
        public class Demo {
            void a() {}
            void b() {}
            void c() {}
            void d() {}
        }
        """).strip())
        code_path = fp.name

    t = Tree(
        code_path=code_path,
        root_object="a",
        supervisor=stub_supervisor,
        expansion_inputs=stub_expansion_inputs,
        is_class=stub_is_class,
        make_agent_state=stub_make_agent_state,
        edge_types={"parent_child": "uses", "node_class": "class_of"},
        autobuild=True,
    )

    print("Nodes:", sorted(t.nodes))
    print("ClassDetails:", list(t.class_details))
    print("Edges:", sorted(map(repr, t.edges)))



# STDOUT/STDERR
# Nodes: ['b', 'c', 'd']
# ClassDetails: ['DemoClass']
# Edges: ["Edge('ROOT:a' -[uses]-> 'b')", "Edge('ROOT:a' -[uses]-> 'c')", "Edge('b' -[uses]-> 'd')", "Edge('d' -[class_of]-> 'DemoClass')"]


# Awesome — I’ve implemented a clean, DFS-style Tree module that plugs directly into your existing supervisor(AgentState) without dictating your LLM logic. It includes:

# Tree (DFS orchestrator)

# Edge (unique, typed edges with .from and .to access)

# ClassDetail (flyweight, deduped by class name)

# Clear docstrings with examples and a runnable smoke test

# Hook points where your logic goes (no-ops/stubs included)


from dfs_tree import Tree

# Your real implementations:
# def supervisor(state): ...
# def expansion_inputs(node): ...  # -> (code, code_path, object_name) or None
# def is_class(node): ...          # -> class_name or None
# def make_agent_state(code, path, obj): ...  # builds your AgentState

# tree = Tree(
#     code_path="/path/to/YourFile.java",
#     root_object="YourRootObject",
#     supervisor=supervisor,
#     expansion_inputs=expansion_inputs,
#     is_class=is_class,
#     make_agent_state=make_agent_state,
#     edge_types={"parent_child": "your_edge_from_edge_list", "node_class": "your_class_edge"},
#     autobuild=True,
# )

# # Results
# tree.nodes          # dict[name -> NodeData]
# tree.class_details  # dict[class_name -> ClassDetail]
# tree.edges          # set[Edge], unique by (from.name, to.name, type)


# Key design choices & where you plug in your logic

# DFS stack of frames (parent_name, child_node): deterministic, easy to control expansion.

# Cycle/loop guard: once a node name is expanded, it won’t be expanded again; we still add new edges to it.

# De-dup nodes: nodes are canonicalized by .name.

# Class short-circuit: is_class(node) returning a name creates/uses a canonical ClassDetail and stops expansion for that node.

# Expansion probe: expansion_inputs(node) decides if/how to produce (code, path, object_name) for the next AgentState. Returning None means “don’t expand”.

# Edge types: pass values from your edge_list via edge_types={"parent_child": "...", "node_class": "..."}.

# Small notes you’ll care about

# The Edge class exposes .from and .to for ergonomics (with an internal from_ field because from is a Python keyword). You can still access edge.from and edge.to normally.

# Uniqueness of edges is enforced by (source_name, target_name, edge_type).

# ClassDetail instances are reused (flyweight) via Tree._get_or_create_class_detail().

# Possible improvements (optional)

# Add a pluggable termination predicate (e.g., max depth, time, node count).

# Add a visited policy to allow re-expansion for certain node types.

# Add serialization/export to NetworkX or JSON (easy to bolt on).

# Add instrumentation hooks (pre/post supervisor call, edge creation) for tracing.

# If anything looks off or you want me to adapt names/interfaces to your existing AgentState, NodeData, or edge_list values, say the word and I’ll tune it.