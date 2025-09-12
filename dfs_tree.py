
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Protocol, Union
from pathlib import Path

class HasName(Protocol):
    name: str

ExpansionInputs = Tuple[str, str, str]

@dataclass(eq=True, frozen=True)
class ClassDetail:
    name: str
    path: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

class Edge:
    __slots__ = ("from_", "to", "type", "_key")
    def __init__(self, source: HasName | ClassDetail, target: HasName | ClassDetail, edge_type: str) -> None:
        self.from_ = source
        self.to = target
        self.type = edge_type
        self._key = (getattr(source, "name", repr(source)), getattr(target, "name", repr(target)), edge_type)
    def __getattr__(self, name: str):
        if name == "from":
            return self.from_
        raise AttributeError(f"{self.__class__.__name__} has no attribute {name!r}")
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

class Tree:
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
        self.root_path: str = str(code_path)
        self.root_name: str = root_object
        self.supervisor = supervisor
        self.expansion_inputs = expansion_inputs
        self.is_class = is_class
        self.make_agent_state = make_agent_state
        self.edge_types = edge_types or {"parent_child": "related", "node_class": "class_of"}
        self.root_code: str = Path(self.root_path).read_text(encoding="utf-8")
        self.nodes: Dict[str, HasName] = {}
        self.class_details: Dict[str, ClassDetail] = {}
        self.edges: Set[Edge] = set()
        self.expanded: Set[str] = set()
        @dataclass
        class _Root:
            name: str
        self._root = _Root(name=f"ROOT:{self.root_name}")
        if autobuild:
            self.build()

    def build(self) -> None:
        root_state = self.make_agent_state(self.root_code, self.root_path, self.root_name)
        first_children = self._safe_supervisor(root_state)
        stack: List[Tuple[str, HasName]] = []
        for child in first_children:
            child = self._canonicalize_node(child)
            self._connect(self._root, child, self.edge_types["parent_child"])
            stack.append((child.name, child))
        while stack:
            parent_name, node = stack.pop()
            if node.name in self.expanded:
                continue
            cls_name = self.is_class(node)
            if cls_name:
                cls = self._get_or_create_class_detail(cls_name)
                self._connect(node, cls, self.edge_types["node_class"])
                self.expanded.add(node.name)
                continue
            exp = self.expansion_inputs(node)
            if exp is None:
                self.expanded.add(node.name)
                continue
            code, path, obj_name = exp
            agent_state = self.make_agent_state(code, path, obj_name)
            next_nodes = self._safe_supervisor(agent_state)
            if not next_nodes:
                self.expanded.add(node.name)
                continue
            for nxt in next_nodes:
                nxt = self._canonicalize_node(nxt)
                self._connect(node, nxt, self.edge_types["parent_child"])
                if nxt.name not in self.expanded:
                    stack.append((nxt.name, nxt))
            self.expanded.add(node.name)

    def _safe_supervisor(self, agent_state: Any) -> List[HasName]:
        try:
            result = self.supervisor(agent_state) or []
        except Exception:
            result = []
        filtered: List[HasName] = []
        for item in result:
            nm = getattr(item, "name", None)
            if isinstance(nm, str) and nm:
                filtered.append(item)
        return filtered

    def _canonicalize_node(self, node: HasName) -> HasName:
        existing = self.nodes.get(node.name)
        if existing is not None:
            return existing
        self.nodes[node.name] = node
        return node

    def _connect(self, src: Union[HasName, ClassDetail], dst: Union[HasName, ClassDetail], edge_type: str) -> None:
        edge = Edge(src, dst, edge_type)
        self.edges.add(edge)

    def _get_or_create_class_detail(self, class_name: str, path: Optional[str] = None) -> ClassDetail:
        cd = self.class_details.get(class_name)
        if cd is not None:
            return cd
        cd = ClassDetail(name=class_name, path=path)
        self.class_details[class_name] = cd
        return cd
