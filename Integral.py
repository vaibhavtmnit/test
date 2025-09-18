Awesome—this is a clean add-on without changing your core design. We’ll keep your DFS exactly as-is and bolt on an optional cross-check against a NetworkX “analytical” tree:
	•	you pass an analytical_tree provider (returns a networkx.DiGraph)
	•	you pass two callbacks: f1 (match) and f2 (mismatch)
	•	at the root and at every expansion, we compare LLM-children vs analytical-children by node name and dispatch f1/f2 accordingly
	•	tiny NetworkX helpers make “get children by name” robust across different node key/attr conventions

Below are the new pieces + the minimal insert points.

⸻

1) NetworkX helpers (drop in anywhere, e.g., near “Helpers”)

def nx_find_node_key_by_name(G, name: str):
    """
    Resolve a NetworkX node *key* from a human name.

    Tries, in order:
      1) key == name
      2) node attribute 'name' == name
      3) node attribute 'ref'.__node_name__ == name   (matches Tree nodes)
    Returns the node key or None if not found.
    """
    # 1) direct key match
    if name in G:
        return name

    # 2/3) attribute-based match
    for k, data in G.nodes(data=True):
        n = data.get("name")
        if n == name:
            return k
        ref = data.get("ref")
        ref_name = getattr(ref, "__node_name__", None)
        if ref_name == name:
            return k
    return None


def nx_children_of(G, parent_name: str):
    """
    Return a list of child *node keys* for a named parent.

    Works for DiGraph (successors) and undirected graphs (neighbors).
    If the parent is missing, returns [].
    """
    pk = nx_find_node_key_by_name(G, parent_name)
    if pk is None:
        return []

    # Prefer directed successors when available
    if hasattr(G, "successors"):
        return list(G.successors(pk))

    # Fallback: neighbors across all edges
    return [v for u, v in G.edges() if u == pk]


def nx_child_names_of(G, parent_name: str):
    """
    Return a set of child *names* for a named parent.

    Derives a human-readable name for each child using, in order:
      1) node attr 'name'
      2) node attr 'ref'.__node_name__
      3) str(key)
    """
    child_keys = nx_children_of(G, parent_name)
    names = set()
    for ck in child_keys:
        data = G.nodes[ck]
        n = data.get("name")
        if not n:
            ref = data.get("ref")
            n = getattr(ref, "__node_name__", None)
        if not n:
            n = str(ck)
        names.add(n)
    return names

Docstring notes are inlined and each line is commented for intent.

⸻

2) Tiny dispatcher used by Tree (drop into Tree as a private helper)

def _compare_and_dispatch(self, parent_name: str, children: list[object]) -> None:
    """
    Compare LLM-discovered `children` against analytical graph children for `parent_name`.
    For each LLM child:
      - if its name appears among analytical children -> call f1 (on_match)
      - else -> call f2 (on_mismatch)

    No design changes to nodes/edges; this is side-effect-only via callbacks.
    """
    G = getattr(self, "_analytical_graph", None)
    f1 = getattr(self, "_on_match", None)
    f2 = getattr(self, "_on_mismatch", None)
    if G is None or (f1 is None and f2 is None):
        return

    # Names from analytical tree for this parent
    ana_child_names = nx_child_names_of(G, parent_name)

    # Dispatch per LLM child
    for ch in children:
        ch_name = self._name_of(ch)
        if ch_name in ana_child_names:
            if f1:  # match
                f1(parent_name, ch)
        else:
            if f2:  # mismatch
                f2(parent_name, ch)


⸻

3) Add optional constructor hooks (one line each + attributes)

In Tree.__init__ signature, append optional params (keeps design intact; defaults are no-ops):

# new optional hooks (add at the end of the parameter list)
analytical_tree_provider: Optional[Callable[[str, Optional[str], str], "nx.DiGraph"]] = None,
on_match: Optional[Callable[[str, object], None]] = None,
on_mismatch: Optional[Callable[[str, object], None]] = None,

Then store them:

self._analytical_tree_provider = analytical_tree_provider
self._on_match = on_match
self._on_mismatch = on_mismatch
self._analytical_graph = None  # will be set in build() if provider is given

Type note: the "nx.DiGraph" string annotation avoids a hard import; you can import networkx at the call-site that provides the graph.

⸻

4) Wire it in at two points inside build()

A. Right after you compute root_state (before priming DFS), obtain the analytical tree:

# Build() — after root_state is created, before first_children:
if self._analytical_tree_provider is not None:
    # Call your external analytical builder (e.g., tree-sitter pipeline)
    # Signature: (root_code, root_code_path, root_object) -> nx.DiGraph
    self._analytical_graph = self._analytical_tree_provider(
        self._root_code, self._root_code_path, self._root_object
    )
else:
    self._analytical_graph = None

B. After you fetch children from supervisor, compare & dispatch.
	•	For the root layer (right after first_children is produced):

self._compare_and_dispatch(self._root.__node_name__, first_children)

	•	And for every expansion, immediately after children = ... is obtained for a node:

self._compare_and_dispatch(node_name, children)

These calls are side-effects only (your DFS, nodes, edges remain unchanged).
If you prefer to run the comparison before pushing to the stack, keep the call right where children is first known (as shown).

⸻

5) How you provide analytical_tree, f1, f2 (usage)

import networkx as nx

def analytical_tree(code: str, code_path: str | None, root_obj: str) -> nx.DiGraph:
    """
    Your tree-sitter based builder.
    Must return a networkx.DiGraph (or Graph) whose nodes are *either* keyed by
    canonical names, or have node attr 'name' (and optionally 'ref').
    """
    # ... your existing implementation ...
    return G

def f1_on_match(parent_name: str, child_node: object) -> None:
    """
    Called when the LLM-discovered child also exists under the same parent
    in the analytical graph.
    Typical uses: mark confidence, skip rework, attach metadata, log parity.
    """
    # e.g., child_node could be enriched: setattr(child_node, "analytical_match", True)
    pass

def f2_on_mismatch(parent_name: str, child_node: object) -> None:
    """
    Called when the LLM-discovered child is *not* present under the same parent
    in the analytical graph.
    Typical uses: flag review, record discrepancy, trigger deeper checks, etc.
    """
    pass

tree = Tree(
    # ... your existing args ...
    analytical_tree_provider=analytical_tree,
    on_match=f1_on_match,     # optional
    on_mismatch=f2_on_mismatch,  # optional
)
tree.build()


⸻

Why this fits your design
	•	No change to node identity, dedupe, edges, or DFS order.
	•	The analytical tree is advisory only; we never mutate your graph from it.
	•	You can keep your sophisticated name_getter (e.g., canonical names) so matching is accurate.
The helpers also tolerate graphs keyed by something else (using node attributes).

⸻

Extra: one-line docs for each new line
	•	self._analytical_tree_provider = ... — stores the optional function used once at build() start to get G.
	•	self._on_match / self._on_mismatch — callbacks you control; we never assume behavior.
	•	self._analytical_graph — cached G so we don’t rebuild per node.
	•	nx_find_node_key_by_name — resolves a node key from a human name across common conventions.
	•	nx_children_of — returns child keys given a parent name (successors > neighbors).
	•	nx_child_names_of — converts those child keys to human names.
	•	_compare_and_dispatch — computes analytical child name set, then for each LLM child runs f1 or f2.
	•	Two insert points in build() — one for root layer, one inside the DFS loop.

If you want me to also attach the analytical child set (or the match boolean) onto edges (e.g., edge.meta["analytical_match"]=True), say the word and I’ll show the 3-line patch for _connect.
