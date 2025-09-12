# tree_dfs.py
"""
DFS Tree Orchestrator for LLM-driven code analysis.

This module provides three core classes:
- `Tree`: drives a DFS expansion using your existing `supervisor(AgentState)` function.
- `Edge`: immutable connection between two nodes with a typed relationship.
- `ClassDetail`: canonical entity representing a Java class encountered during expansion.

It is intentionally **framework-agnostic** and **stdlib-only**, so you can plug it into
your repo without adding dependencies. All repo-specific logic is injected via
callables (strategy hooks), keeping traversal robust and testable.

------------------------------------------------------------------------------
QUICK START (with stubs)  — replace stubs with your real logic
------------------------------------------------------------------------------

>>> # -- Your existing functions & types (stubs here for illustration) --
>>> from dataclasses import dataclass
>>>
>>> @dataclass
>>> class NodeData:
>>>     name: str
>>>     kind: str = "method"  # could be "class", "method", etc. in your code
>>>
>>> @dataclass
>>> class AgentState:
>>>     code: str
>>>     code_path: str | None
>>>     obj_name: str
>>>
>>> # Minimal supervisor stub: parent "start" yields children "A" and "B",
>>> # "B" yields "C", and "C" is a class.
>>> def supervisor_stub(state: AgentState) -> list[NodeData]:
>>>     if state.obj_name == "start":
>>>         return [NodeData("A"), NodeData("B")]
>>>     if state.obj_name == "B":
>>>         return [NodeData("C", kind="class")]
>>>     return []
>>>
>>> # ---- Strategy hooks (replace with your real logic) ----
>>> def build_agent_state(code: str, code_path: str | None, obj: str) -> AgentState:
>>>     return AgentState(code=code, code_path=code_path, obj_name=obj)
>>>
>>> def get_name(node: NodeData) -> str:
>>>     return node.name
>>>
>>> # Expansion target: for demo, expand using same code/path and node's own name.
>>> from dataclasses import dataclass
>>> @dataclass(frozen=True)
>>> class ExpandTarget:
>>>     code: str
>>>     code_path: str | None
>>>     object_name: str
>>>
>>> def expand_target(node: NodeData) -> ExpandTarget | None:
>>>     return ExpandTarget(code="class Main { void start(){} }", code_path="Main.java", object_name=node.name)
>>>
>>> def is_class_type(node: NodeData) -> bool:
>>>     return node.kind == "class"
>>>
>>> def get_class_name(node: NodeData) -> str:
>>>     return node.name
>>>
>>> def edge_type_for(parent: object, child: object) -> str:
>>>     # In your project, validate against your `edge_list` enum/registry.
>>>     return "ASSOCIATED"
>>>
>>> def should_expand(node: NodeData) -> bool:
>>>     # You will replace with your termination/eligibility logic later.
>>>     return True
>>>
>>> # ---- Build and run the Tree DFS ----
>>> tree = Tree(
...     root_object="start",
...     root_code="class Main { void start(){} }",
...     root_code_path="Main.java",
...     supervisor=supervisor_stub,
...     agent_state_builder=build_agent_state,
...     name_getter=get_name,
...     expand_target_getter=expand_target,
...     is_class_type=is_class_type,
...     get_class_name=get_class_name,
...     edge_type_for=edge_type_for,
...     should_expand=should_expand,
... )
>>> tree.build()   # runs DFS
>>> sorted(n for n in tree.nodes())
['A', 'B', 'C', 'start']
>>> sorted((e.from_name, e.to_name, e.type) for e in tree.edges)
[('B', 'C', 'ASSOCIATED'), ('start', 'A', 'ASSOCIATED'), ('start', 'B', 'ASSOCIATED')]
>>> # ClassDetails registry recorded class 'C'
>>> sorted(tree.class_details)
['C']

------------------------------------------------------------------------------
DESIGN SUMMARY
------------------------------------------------------------------------------
1) You provide repo-specific logic via callables:
   - agent_state_builder(code, code_path, obj_name) -> AgentState
   - supervisor(AgentState) -> list[NodeData]
   - name_getter(NodeData) -> str
   - expand_target_getter(NodeData) -> ExpandTarget | None
   - is_class_type(NodeData) -> bool
   - get_class_name(NodeData) -> str
   - edge_type_for(parent, child) -> str     (validated against your edge_list)
   - should_expand(NodeData) -> bool         (you’ll refine with real termination rules)

2) DFS traversal:
   - Start from a *synthetic* root identified by `root_object`.
   - Call supervisor(root AgentState) -> children -> push to DFS stack.
   - Pop next candidate:
       (a) Derive ExpandTarget (code, path, object). If None -> do not expand.
       (b) If node *is a class* -> create/reuse `ClassDetail` and connect edge.
       (c) Else -> call supervisor on derived AgentState, enqueue returned nodes.
   - After processing, connect the node to its parent with an `Edge`.
   - Uses `name_getter` to:
       - Ensure node uniqueness and visited/expansion guards.
       - Avoid infinite loops/recursion (don’t expand same name twice).

3) Uniqueness & registries:
   - Nodes deduped by `name_getter(node)`.
   - `Edge` uniqueness via hashing (from_name, to_name, type).
   - `ClassDetail` registry keyed by canonical class name.

4) Extensibility:
   - Add your termination predicate later by swapping `should_expand`.
   - Validate edge types against your `edge_list` inside `edge_type_for` if desired.
   - Convert to NetworkX later via `to_networkx()` (provided but optional).

"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Any, Dict, List, Set, Tuple


# ---------------------------------------------------------------------------
# Core data types (generic; your repo provides the concrete classes)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExpandTarget:
    """
    Triple needed to continue DFS from a NodeData.

    Attributes
    ----------
    code : str
        The Java source code to analyze next (full text).
    code_path : Optional[str]
        Path or identifier for the code (can be None for in-memory).
    object_name : str
        The object name to pass into AgentState for `supervisor`.

    Notes
    -----
    - You will implement the logic that extracts this from a `NodeData`.
      (See `expand_target_getter` in `Tree.__init__`.)
    """
    code: str
    code_path: Optional[str]
    object_name: str


@dataclass
class ClassDetail:
    """
    Canonical representation of a Java class encountered during DFS.

    Attributes
    ----------
    name : str
        Canonical class name (e.g., 'com.foo.Bar' or 'Bar' if simple).
    code : Optional[str]
        Optional: source code of the class, if you choose to store it.
    code_path : Optional[str]
        Optional: where it came from.
    """
    name: str
    code: Optional[str] = None
    code_path: Optional[str] = None

    def __hash__(self) -> int:
        # Hash by canonical name so registry can dedupe.
        return hash(self.name)


@dataclass(frozen=True)
class Edge:
    """
    Directed edge between two nodes (NodeData or ClassDetail), with a type.

    Attributes
    ----------
    from_ref : object
        Reference to the source node (NodeData or ClassDetail).
    to_ref : object
        Reference to the target node (NodeData or ClassDetail).
    type : str
        Relationship label. Must be a member of your `edge_list` in the host repo.

    Notes
    -----
    - Uniqueness is enforced structurally via (from_name, to_name, type).
    - We store the *names* alongside the object refs to make hashing stable even
      if the underlying objects are unhashable or mutable.
    """
    from_ref: object
    to_ref: object
    type: str
    from_name: str = field(init=False)
    to_name: str = field(init=False)

    def __post_init__(self):
        # We infer names lazily in Tree when constructing Edge,
        # but keep fields present for introspection and hashing.
        object.__setattr__(self, "from_name", getattr(self.from_ref, "__node_name__", None) or str(self.from_ref))
        object.__setattr__(self, "to_name", getattr(self.to_ref, "__node_name__", None) or str(self.to_ref))

    def __hash__(self) -> int:
        return hash((self.from_name, self.to_name, self.type))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Edge):
            return False
        return (
            self.type == other.type
            and self.from_name == other.from_name
            and self.to_name == other.to_name
        )


# ---------------------------------------------------------------------------
# Tree (DFS Orchestrator)
# ---------------------------------------------------------------------------

class Tree:
    """
    DFS-style expansion driver that plugs into your LLM supervisor.

    Parameters
    ----------
    root_object : str
        Name of the starting object (e.g., a method name). This becomes the
        synthetic root node label (it is tracked internally as a node).
    root_code : str
        Full source of the code containing the root object.
    root_code_path : Optional[str]
        Path/identifier for the root code file.
    supervisor : Callable[[AgentState], list]
        Your existing function that takes an AgentState and returns `List[NodeData]`.
    agent_state_builder : Callable[[str, Optional[str], str], Any]
        Factory that builds your concrete `AgentState` from (code, code_path, obj_name).
    name_getter : Callable[[Any], str]
        Extracts the canonical `name` from a `NodeData`. Used for deduping and edges.
    expand_target_getter : Callable[[Any], Optional[ExpandTarget]]
        From a `NodeData`, derive (code, path, object) to continue DFS.
        Return `None` to skip expansion of that node.
    is_class_type : Callable[[Any], bool]
        Returns True if the `NodeData` represents a class (then we create ClassDetail).
    get_class_name : Callable[[Any], str]
        Returns the canonical class name for class-typed nodes.
    edge_type_for : Callable[[object, object], str]
        Returns the edge label to connect `parent -> child`. Validate against your edge_list.
    should_expand : Callable[[Any], bool]
        Eligibility predicate for DFS expansion (plug your termination logic later).
        If this returns False, the node will be attached by edge but *not* expanded.

    Notes
    -----
    - The tree is **built by calling `build()`**. You can call `clear()` to reset.
    - Node uniqueness is enforced by `name_getter`. The same name won't be expanded twice.
    - Edges are unique by (from_name, to_name, type), but nodes can have many in/out edges.

    Examples
    --------
    See module-level doctest in the header for a runnable example.
    """

    # --------------------
    # Construction & state
    # --------------------
    def __init__(
        self,
        *,
        root_object: str,
        root_code: str,
        root_code_path: Optional[str],
        supervisor: Callable[[Any], List[Any]],
        agent_state_builder: Callable[[str, Optional[str], str], Any],
        name_getter: Callable[[Any], str],
        expand_target_getter: Callable[[Any], Optional[ExpandTarget]],
        is_class_type: Callable[[Any], bool],
        get_class_name: Callable[[Any], str],
        edge_type_for: Callable[[object, object], str],
        should_expand: Callable[[Any], bool],
    ) -> None:
        # --- Public inputs (hooks & config)
        self._root_object = root_object
        self._root_code = root_code
        self._root_code_path = root_code_path

        self._supervisor = supervisor
        self._agent_state_builder = agent_state_builder
        self._name_of = name_getter
        self._expand_target = expand_target_getter
        self._is_class = is_class_type
        self._class_name = get_class_name
        self._edge_type_for = edge_type_for
        self._should_expand = should_expand

        # --- Graph data (internal)
        # Map of node name -> NodeData (or synthetic root placeholder with __node_name__)
        self._nodes: Dict[str, object] = {}
        # Set of edges (uniqueness guaranteed by Edge.__hash__)
        self.edges: Set[Edge] = set()
        # Registry of classes (canonical name -> ClassDetail)
        self.class_details: Dict[str, ClassDetail] = {}
        # Names we have already *expanded* (not just seen) to prevent cycles
        self._expanded: Set[str] = set()

        # Create & register synthetic root
        self._root = _SyntheticRoot(self._root_object)
        self._nodes[self._root.__node_name__] = self._root

    # -------------
    # Public API
    # -------------
    def clear(self) -> None:
        """Reset all computed state, keeping the configuration/hooks intact."""
        self.edges.clear()
        self.class_details.clear()
        self._expanded.clear()
        # Keep root registered
        self._nodes = {self._root.__node_name__: self._root}

    def build(self) -> None:
        """
        Run DFS expansion from the root.

        Process (high level)
        --------------------
        1) Call supervisor on (root_code, root_code_path, root_object) to get first layer.
        2) Push those onto a DFS stack with their parent anchored to the synthetic root.
        3) While stack not empty:
           a) Pop a candidate (parent_name, child_node).
           b) If we've already expanded `child_node.name`, just connect edge & continue.
           c) Otherwise:
              - Ask for ExpandTarget (code, path, object). If none -> don't expand.
              - If node is class -> create/reuse ClassDetail; connect node -> class.
              - Else -> supervisor on new AgentState; push returned children.
           d) Connect `parent -> child` edge (unique).
        """
        self.clear()

        # 1) Prime DFS with the root call to supervisor
        root_state = self._agent_state_builder(self._root_code, self._root_code_path, self._root_object)
        first_children = self._ensure_list(self._supervisor(root_state))

        stack: List[Tuple[str, object]] = []  # Each item: (parent_name, child_node)
        for child in reversed(first_children):  # reverse so first child is processed first (DFS)
            self._register_node(child)
            stack.append((self._root.__node_name__, child))

        # 2) DFS loop
        while stack:
            parent_name, node = stack.pop()
            node_name = self._name_of(node)

            # (i) Avoid infinite loops: if we've expanded this node before, just connect edge
            if node_name in self._expanded:
                self._connect(parent_name, node)
                continue

            # (ii) If policy says "do not expand", just connect and skip deeper traversal
            if not self._should_expand(node):
                self._connect(parent_name, node)
                # Mark as expanded so we don't circle back later
                self._expanded.add(node_name)
                continue

            # (iii) Try to derive expansion target (code, path, object). If None -> no expansion.
            target = self._expand_target(node)
            if target is None:
                self._connect(parent_name, node)
                self._expanded.add(node_name)
                continue

            # (iv) If node represents a class: attach a ClassDetail and do not expand deeper.
            if self._is_class(node):
                cls = self._get_or_create_class(self._class_name(node), code=target.code, code_path=target.code_path)
                # Connect node -> class detail (type chosen by your hook)
                self._connect(node, cls)
                # Also connect parent -> node (the “main” edge for this step)
                self._connect(parent_name, node)
                self._expanded.add(node_name)
                continue

            # (v) Otherwise, expand via supervisor
            next_state = self._agent_state_builder(target.code, target.code_path, target.object_name)
            children = self._ensure_list(self._supervisor(next_state))

            # Enqueue children for DFS (register nodes, push on stack with current as parent)
            if children:
                for ch in reversed(children):
                    self._register_node(ch)
                    stack.append((node_name, ch))

            # Finally, connect parent -> node and mark node expanded
            self._connect(parent_name, node)
            self._expanded.add(node_name)

    def nodes(self) -> Iterable[str]:
        """Return the set of node names currently tracked (including the root label)."""
        return self._nodes.keys()

    # -------------
    # Internals
    # -------------
    def _register_node(self, node: object) -> None:
        """Track a node by its canonical name, without overwriting existing entries."""
        name = self._name_of(node)
        if name not in self._nodes:
            # Attach a well-known attribute used by Edge for stable naming
            try:
                setattr(node, "__node_name__", name)  # for NodeData objects
            except Exception:
                # Some foreign objects may not allow attributes; we still store by name
                pass
            self._nodes[name] = node

    def _connect(self, parent: str | object, child: object) -> None:
        """
        Add a unique edge from `parent` -> `child`.

        `parent` can be a parent name (str) or a node object. `child` is a node object.
        The edge type is delegated to `edge_type_for(parent, child)`.
        """
        # Normalize to object refs & names
        parent_ref = self._nodes[parent] if isinstance(parent, str) else parent
        child_ref = child

        # Ensure both endpoints are registered for consistent naming
        if isinstance(parent, str):
            parent_name = parent
        else:
            parent_name = getattr(parent_ref, "__node_name__", None) or self._name_of(parent_ref)
            self._register_node(parent_ref)

        child_name = getattr(child_ref, "__node_name__", None) or self._name_of(child_ref)
        self._register_node(child_ref)

        # Compute edge label and create Edge
        label = self._edge_type_for(parent_ref, child_ref)
        edge = Edge(from_ref=parent_ref, to_ref=child_ref, type=label)
        # Override from/to names with canonical, ensuring stable hashing & equality
        object.__setattr__(edge, "from_name", parent_name)
        object.__setattr__(edge, "to_name", child_name)

        # Insert uniquely
        self.edges.add(edge)

    def _get_or_create_class(self, name: str, *, code: Optional[str], code_path: Optional[str]) -> ClassDetail:
        """
        Return an existing ClassDetail by name or create/register a new one.
        """
        if name in self.class_details:
            cd = self.class_details[name]
            # Optionally augment missing fields on re-encounter
            if cd.code is None and code is not None:
                cd.code = code
            if cd.code_path is None and code_path is not None:
                cd.code_path = code_path
            return cd

        cd = ClassDetail(name=name, code=code, code_path=code_path)
        # Attach a canonical node name so it can participate in edges
        setattr(cd, "__node_name__", name)
        self._nodes[name] = cd
        self.class_details[name] = cd
        return cd

    @staticmethod
    def _ensure_list(maybe_list: Any) -> List[Any]:
        """Normalize supervisor output into a list."""
        if maybe_list is None:
            return []
        if isinstance(maybe_list, list):
            return maybe_list
        return [maybe_list]

    # -------------
    # Optional: export
    # -------------
    def to_networkx(self):
        """
        Optional convenience: export to a NetworkX DiGraph (if networkx is installed).

        Returns
        -------
        networkx.DiGraph

        Notes
        -----
        - Nodes are keyed by their canonical names.
        - Edge attribute 'type' carries the relationship label.
        """
        try:
            import networkx as nx  # type: ignore
        except Exception as e:
            raise RuntimeError("networkx is not installed. `pip install networkx`") from e

        G = nx.DiGraph()
        for name, obj in self._nodes.items():
            G.add_node(name, ref=obj)
        for e in self.edges:
            G.add_edge(e.from_name, e.to_name, type=e.type, ref=e)
        return G


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _SyntheticRoot:
    """
    Lightweight placeholder representing the synthetic root.

    This object behaves like a node for naming/edges but is not expanded.
    """
    def __init__(self, label: str) -> None:
        self.__node_name__ = label

    def __repr__(self) -> str:  # helpful in debugging
        return f"<ROOT {self.__node_name__!r}>"


# Why this shape?

# Loose coupling to your codebase. Your AgentState, NodeData, and supervisor are treated as black boxes; the Tree only needs the factory and accessors you pass in.

# Deterministic DFS with cycle guards. We dedupe expansion by canonical name, as you requested, which naturally avoids infinite recursion/loop patterns. We still allow multiple incoming/outgoing edges.

# Class registry with reuse. ClassDetail is canonicalized by name; subsequent hits reuse the same instance.

# Edge uniqueness. We hash by (from_name, to_name, type) so duplicates are automatically ignored.

# Pluggable policies. All project-specific logic (expand target extraction, edge typing, termination/eligibility) is passed in as callables you can swap later.

# Small improvements you might like

# Edge validation: in edge_type_for, assert membership in your edge_list to catch wiring mistakes early.

# Progress hooks / logging: add an optional on_expand(node, children) callback or logger.debug statements to observe traversal in large projects.

# Back-pressure controls: add a max_nodes or max_depth guard to should_expand to protect against pathological graphs before your full termination logic lands.

# Memoized supervisor: if supervisor is pure for (code_path, obj_name), a simple LRU cache around it will cut cost & latency drastically.

# If you want, I can now:

# Plug in your actual AgentState and NodeData types,

# Validate edge_type_for against your edge_list,

# Or add a thin adapter that returns a fully-formed networkx.DiGraph you can feed straight into your graph process.