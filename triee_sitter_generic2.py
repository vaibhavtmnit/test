"""
Java Call-Graph Builder (tree-sitter) — Indexed, Typed, and Lambda-Aware
========================================================================

This module builds a directional **call/usage graph** from Java code in **two passes**:

1) ProjectIndexer  (FIRST PASS — "index first")
   - Indexes packages, imports (wildcards & static), classes/interfaces/enums/records,
     fields, methods (name, params, return type, static).
   - Stores results in a project-level structure:
       - class_index: {FQN -> ClassInfo(fields, methods, inheritance, ...)}
       - simple_to_fqn: {'List' -> {'java.util.List', ...}}
   - ✔ Implements improvement #1 and #4.

2) CodeAnalyzer    (SECOND PASS — "analyze second")
   - Walks statements/expressions from a chosen start method.
   - Performs **type inference**:
       - var inference (beyond 'new'), assignments, casts
       - array access element type, method return types (from index)
       - propagates generic element type T across common Stream ops
   - Resolves `obj.method(...)` by inferred receiver type (arity-aware);
     walks inheritance to locate definitions.
   - Covers many Java nodes: loops, try/catch/finally, switch, method references,
     member references, array/index, assignments.
   - ✔ Implements improvements #2, #3, #5, #6, #7 (precompiled queries via helper cache-ready).

Testing scaffolding (#8):
   - See `run_smoke_demo()` and the demo in __main__.
   - For “golden tests”, create a `tests/java/` folder with inputs + expected edges,
     then assert on graph nodes/edges per sample (docstring below shows how).

Compatibility:
   - Works with older *and* newer py-tree-sitter query APIs.
   - Parser initialization handles both constructor styles.

Dependencies:
   pip install tree-sitter tree-sitter-java networkx
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable, Set, Union, DefaultDict
from collections import defaultdict, deque

import networkx as nx

# --- tree-sitter setup ------------------------------------------------------------
#   Keep imports explicit; we use Query and QueryCursor for API compatibility.
import tree_sitter_java as tsj
from tree_sitter import Language, Parser, Node, Query, QueryCursor


# ==================================================================================
#                                   Language & Parser
# ==================================================================================

try:
    # Newer py-tree-sitter: Language(...) accepts a compiled language object.
    JAVA = Language(tsj.language())
except Exception as e:
    raise RuntimeError(
        "Error loading tree-sitter Java language. Ensure 'tree-sitter' and 'tree-sitter-java' are installed."
    ) from e

# Parser compatibility: some versions require set_language; others allow Parser(JAVA)
try:
    PARSER = Parser(JAVA)  # newer style
except TypeError:          # older style
    PARSER = Parser()
    PARSER.set_language(JAVA)


# ==================================================================================
#                             Small helpers & Constants
# ==================================================================================

JAVA_LANG_IMPLICIT = {
    # java.lang.* is implicitly imported; resolve these without explicit import lines.
    "String": "java.lang.String",
    "Object": "java.lang.Object",
    "Boolean": "java.lang.Boolean",
    "Byte": "java.lang.Byte",
    "Character": "java.lang.Character",
    "Class": "java.lang.Class",
    "Double": "java.lang.Double",
    "Float": "java.lang.Float",
    "Integer": "java.lang.Integer",
    "Long": "java.lang.Long",
    "Math": "java.lang.Math",
    "System": "java.lang.System",
    "Runnable": "java.lang.Runnable",
}

# Minimal rules for common Stream APIs to propagate element types (improvement #6).
STREAM_RULES = {
    # method_name: (return_type_template, lambda_param_from_receiver_generic?)
    # '$T' = element type of receiver generic; '$R' = lambda result type (unknown here)
    "stream": ("java.util.stream.Stream<$T>", False),
    "map": ("java.util.stream.Stream<$R>", True),
    "flatMap": ("java.util.stream.Stream<$R>", True),
    "filter": ("java.util.stream.Stream<$T>", False),
    "anyMatch": ("boolean", False),
    "allMatch": ("boolean", False),
    "noneMatch": ("boolean", False),
    "collect": ("java.lang.Object", False),  # simplified
    "forEach": ("void", False),
    "toList": ("java.util.List<$T>", False),
}


def _text(src: str, node: Optional[Node]) -> str:
    """
    Return the exact source text for a node.

    Args:
        src: Full file text.
        node: Tree-sitter Node.

    Returns:
        The substring for this node, or "" if node is None.

    Example:
        >>> _text("abcd", None)
        ''
    """
    if node is None:
        return ""
    return src[node.start_byte: node.end_byte]


def _field(node: Optional[Node], name: str) -> Optional[Node]:
    """
    Safe wrapper around child_by_field_name.

    Example:
        >>> # returns None (no crash) if node is None
        >>> _field(None, "name") is None
        True
    """
    if node is None:
        return None
    return node.child_by_field_name(name)


def _children_of_type(node: Node, t: Union[str, Tuple[str, ...]]) -> Iterable[Node]:
    """
    Yield named children whose type equals `t` (or is inside tuple of types).

    Example:
        >>> # Used internally to iterate 'catch_clause' nodes inside try_statement
        >>> # for c in _children_of_type(try_node, "catch_clause"): ...
    """
    for c in node.named_children:
        if (isinstance(t, str) and c.type == t) or (isinstance(t, tuple) and c.type in t):
            yield c


def _has_modifier(src: str, node: Node, mod: str) -> bool:
    """
    Check if a declaration node has a particular modifier (e.g., 'static').

    Example:
        >>> # In a 'method_declaration', returns True if 'static' appears in modifiers
        >>> # _has_modifier(src, decl_node, "static")
    """
    mods = _field(node, "modifiers")
    if mods:
        for c in mods.children:
            if _text(src, c) == mod:
                return True
    return False


def _type_from_generic(src: str, type_node: Optional[Node]) -> Tuple[str, Optional[str]]:
    """
    Extract ('raw_type', 'first_generic_arg') from a type node.

    Example:
        'List<Predicate>' -> ('List', 'Predicate')
        'String'          -> ('String', None)

    Returns:
        (raw, generic_arg_or_none)
    """
    if type_node is None:
        return "Unknown", None
    if type_node.type == "generic_type":
        raw = type_node.child(0)
        targ = _field(type_node, "type_arguments")
        gen = None
        if targ and targ.named_child_count > 0:
            gen = _text(src, targ.named_child(0))
        return _text(src, raw), gen
    return _text(src, type_node), None


def query_capture_map(lang: Language, pattern: str, root: Node) -> Dict[str, List[Node]]:
    """
    Run a tree-sitter query and return a dict: {capture_name: [Node, ...]}.

    Compatibility wrapper for multiple py-tree-sitter API shapes:
      - New API:  lang.query(pattern).captures(root) -> list[(Node, "cap")]
      - Old APIA: Query(lang, pattern); QueryCursor().captures(root, query) -> list[(Node, "cap")]
      - Old APIB: Query(lang, pattern); QueryCursor(query).captures(root) -> {"cap":[Node,...]}

    Example:
        >>> # capmap = query_capture_map(JAVA, "(import_declaration) @import", root)
        >>> # for imp_node in capmap.get("import", []): ...
    """
    # NEW style
    try:
        q_new = lang.query(pattern)
        res = q_new.captures(root)
        if isinstance(res, dict):
            return res
        out: Dict[str, List[Node]] = {}
        for n, cap in res:
            out.setdefault(cap, []).append(n)
        return out
    except Exception:
        pass

    # OLD styles
    q = Query(lang, pattern)
    try:
        # Old Style B
        cursor_b = QueryCursor(q)
        res_b = cursor_b.captures(root)
        if isinstance(res_b, dict):
            return res_b
        out_b: Dict[str, List[Node]] = {}
        for n, cap in res_b:
            out_b.setdefault(cap, []).append(n)
        return out_b
    except TypeError:
        # Old Style A
        cursor_a = QueryCursor()
        res_a = cursor_a.captures(root, q)
        if isinstance(res_a, dict):
            return res_a
        out_a: Dict[str, List[Node]] = {}
        for n, cap in res_a:
            out_a.setdefault(cap, []).append(n)
        return out_a


# ==================================================================================
#                               Index data structures
# ==================================================================================

@dataclass
class MethodSig:
    """
    Represents a method signature in a type.

    Attributes:
        name: Method simple name.
        params: List of parameter type names (text as parsed).
        return_type: Return type name (text as parsed).
        is_static: Whether method is static.
        decl_node: AST node of the declaration.
        line: 1-based source line number.

    Example:
        >>> MethodSig("processRequest", ["String"], "String", False).name
        'processRequest'
    """
    name: str
    params: List[str]
    return_type: str
    is_static: bool
    decl_node: Optional[Node] = None
    line: Optional[int] = None


@dataclass
class ClassInfo:
    """
    Represents a type (class/interface/enum/record) with members and relationships.

    Attributes:
        fqn: Fully qualified name (e.g., 'com.example.DummyTest').
        kind: 'class' | 'interface' | 'enum' | 'record'
        package: Package name or ''.
        simple_name: Simple type name.
        extends: Superclass name (as parsed; FQN resolution is best-effort later).
        implements: List of interface names (as parsed).
        fields: Map var_name -> type_name
        methods: Map method_name -> list[MethodSig] (to support overloads)
        node: AST node for the declaration.
    """
    fqn: str
    kind: str
    package: str
    simple_name: str
    extends: Optional[str] = None
    implements: List[str] = field(default_factory=list)
    fields: Dict[str, str] = field(default_factory=dict)
    methods: DefaultDict[str, List[MethodSig]] = field(default_factory=lambda: defaultdict(list))
    node: Optional[Node] = None


@dataclass
class FileIndex:
    """
    Per-file index: package, imports, wildcards, static imports, and classes found.

    Example:
        >>> fi = FileIndex(Path("A.java"))
        >>> isinstance(fi.imports, dict), isinstance(fi.classes, dict)
        (True, True)
    """
    path: Path
    package: str = ""
    imports: Dict[str, str] = field(default_factory=dict)        # simple -> FQN
    wildcard_imports: List[str] = field(default_factory=list)    # 'com.pkg.*'
    static_imports: List[str] = field(default_factory=list)      # e.g., 'java.lang.Math.max' or 'pkg.Class.*'
    classes: Dict[str, ClassInfo] = field(default_factory=dict)  # fqn -> ClassInfo
    src: str = ""


class ProjectIndexer:
    """
    FIRST PASS (Improvement #1 & #4):
    Index packages/imports/types/members across one or more files. Results feed the analyzer.

    Usage:
        >>> idx = ProjectIndexer().index_files(["A.java", "B.java"])
        >>> cls = next(iter(idx.class_index.values()))
        >>> isinstance(cls, ClassInfo)
        True
    """

    def __init__(self):
        self.file_indexes: List[FileIndex] = []
        self.class_index: Dict[str, ClassInfo] = {}              # FQN -> ClassInfo
        self.simple_to_fqn: DefaultDict[str, Set[str]] = defaultdict(set)  # 'List' -> {'java.util.List', ...}

    def index_files(self, file_paths: List[Union[str, Path]]) -> "ProjectIndexer":
        """
        Parse and index each provided .java file.

        Args:
            file_paths: List of paths.

        Returns:
            Self (fluent API).

        Example:
            >>> ProjectIndexer().index_files([]).__class__.__name__
            'ProjectIndexer'
        """
        for p in file_paths:
            self._index_one(Path(p))
        # Build global maps
        for fidx in self.file_indexes:
            for fqn, ci in fidx.classes.items():
                self.class_index[fqn] = ci
                self.simple_to_fqn[ci.simple_name].add(fqn)
        return self

    def _index_one(self, path: Path) -> None:
        """
        Index a single file: package, imports, and type declarations.

        Notes:
            Compatible with older/newer py-tree-sitter query APIs via query_capture_map().

        Example:
            >>> # Internal use. Called by index_files()
        """
        if not path.exists() or path.suffix.lower() != ".java":
            return

        src = path.read_text(encoding="utf-8")
        tree = PARSER.parse(src.encode("utf-8"))
        root = tree.root_node

        fidx = FileIndex(path=path, src=src)

        # --- package_declaration ----------------------------------------------------
        for node in root.named_children:
            if node.type == "package_declaration":
                name_node = _field(node, "name")
                fidx.package = _text(src, name_node)
                break

        # --- imports (including wildcards & static) --------------------------------
        capmap = query_capture_map(JAVA, "(import_declaration) @import", root)
        for node in capmap.get("import", []):
            text = _text(src, node).strip().rstrip(";")
            if text.startswith("import "):
                text = text[len("import "):].strip()
            is_static = False
            if text.startswith("static "):
                is_static = True
                text = text[len("static "):].strip()
            if text.endswith(".*"):
                if is_static:
                    fidx.static_imports.append(text)   # e.g., java.util.Collections.*
                else:
                    fidx.wildcard_imports.append(text) # e.g., com.pkg.*
            else:
                if is_static:
                    fidx.static_imports.append(text)   # e.g., java.lang.Math.max
                else:
                    simple = text.split(".")[-1]
                    fidx.imports[simple] = text
                    self.simple_to_fqn[simple].add(text)

        # --- type declarations (classes/interfaces/enums/records) -------------------
        capmap = query_capture_map(JAVA, r"""
            [
              (class_declaration) @cls
              (interface_declaration) @intf
              (enum_declaration) @enm
              (record_declaration) @rec
            ]
        """, root)

        def _process_type_decl(node: Node, kind: str) -> None:
            """Extract ClassInfo: name, inheritance, fields, methods."""
            name_node = _field(node, "name")
            simple = _text(src, name_node)
            fqn = f"{fidx.package}.{simple}" if fidx.package else simple

            ci = ClassInfo(
                fqn=fqn,
                kind=kind,
                package=fidx.package,
                simple_name=simple,
                node=node,
            )

            # extends / implements (tolerant of grammar variants)
            sc = _field(node, "superclass")
            if sc:
                ci.extends = _text(src, sc)

            si = (
                _field(node, "super_interfaces")
                or _field(node, "extends_interfaces")
                or _field(node, "interfaces")
            )
            if si:
                for idn in si.named_children:
                    ci.implements.append(_text(src, idn))

            # members (fields + methods) inside body
            body = _field(node, "body")
            if body:
                for mem in body.named_children:
                    if mem.type == "field_declaration":
                        # e.g., List<T> x, y;
                        tnode = _field(mem, "type")
                        raw_t, gen = _type_from_generic(src, tnode) if tnode else ("Unknown", None)
                        # Support one 'declarator' or multiple 'variable_declarator' nodes
                        decls = []
                        single = _field(mem, "declarator")
                        if single is not None:
                            decls = [single]
                        else:
                            decls = [ch for ch in mem.named_children if ch.type == "variable_declarator"]
                        for dnode in decls:
                            nm = _field(dnode, "name")
                            var_name = _text(src, nm)
                            ci.fields[var_name] = gen or raw_t

                    elif mem.type == "method_declaration":
                        # name, params, return type, static?
                        mname = _text(src, _field(mem, "name"))
                        params: List[str] = []
                        pnode = _field(mem, "parameters")
                        if pnode:
                            for ch in pnode.children:
                                if ch.type == "formal_parameter":
                                    ptn = _field(ch, "type")
                                    ptype = _text(src, ptn) if ptn else "void"
                                    params.append(ptype)
                        rtype_node = _field(mem, "type")
                        rtype = _text(src, rtype_node) if rtype_node else "void"
                        is_static = _has_modifier(src, mem, "static")
                        ci.methods[mname].append(
                            MethodSig(
                                name=mname,
                                params=params,
                                return_type=rtype,
                                is_static=is_static,
                                decl_node=mem,
                                line=mem.start_point[0] + 1,
                            )
                        )

            fidx.classes[fqn] = ci
            self.simple_to_fqn[simple].add(fqn)

        for node in capmap.get("cls", []):
            _process_type_decl(node, kind="class")
        for node in capmap.get("intf", []):
            _process_type_decl(node, kind="interface")
        for node in capmap.get("enm", []):
            _process_type_decl(node, kind="enum")
        for node in capmap.get("rec", []):
            _process_type_decl(node, kind="record")

        # Register this file’s index
        self.file_indexes.append(fidx)

    def resolve_simple(self, name: str, using_file: FileIndex) -> Optional[str]:
        """
        Resolve a simple type name to a fully-qualified name (FQN), using:
          - explicit imports in the file
          - wildcard imports
          - java.lang implicit imports
          - global unique simple-name map

        Returns:
            FQN if uniquely resolvable; otherwise returns the input name.

        Example:
            >>> # If file has 'import java.util.List;', resolve_simple("List", fidx) -> "java.util.List"
        """
        if not name or "." in name:
            return name

        if name in using_file.imports:
            return using_file.imports[name]
        if name in JAVA_LANG_IMPLICIT:
            return JAVA_LANG_IMPLICIT[name]

        # wildcard imports (search project index for matches)
        candidates = set()
        for pkg_star in using_file.wildcard_imports:
            pkg = pkg_star[:-2]  # strip '.*'
            fqn = f"{pkg}.{name}"
            if fqn in self.class_index:
                candidates.add(fqn)
        if len(candidates) == 1:
            return next(iter(candidates))

        # globally unique
        if name in self.simple_to_fqn and len(self.simple_to_fqn[name]) == 1:
            return next(iter(self.simple_to_fqn[name]))

        return name

    def super_chain(self, fqn: str) -> Iterable[str]:
        """
        Yield the class and its superclasses upwards, as best-effort resolution.

        Example:
            >>> # for fqn in indexer.super_chain("com.example.Child"): ...
        """
        seen = set()
        cur = fqn
        while cur and cur not in seen:
            seen.add(cur)
            yield cur
            ci = self.class_index.get(cur)
            if not ci or not ci.extends:
                break
            ext = ci.extends
            if "." not in ext:
                pkg = ci.package
                xfqn = f"{pkg}.{ext}" if pkg else ext
                cur = xfqn if xfqn in self.class_index else None
            else:
                cur = ext if ext in self.class_index else None

    def find_methods(self, recv_fqn: str, name: str, arity: int) -> List[MethodSig]:
        """
        Locate method candidates (by name + arity) along the inheritance chain.

        Example:
            >>> # indexer.find_methods("java.util.List", "stream", 0)
        """
        out: List[MethodSig] = []
        for cfqn in self.super_chain(recv_fqn):
            ci = self.class_index.get(cfqn)
            if not ci:
                continue
            for ms in ci.methods.get(name, []):
                if len(ms.params) == arity:
                    out.append(ms)
        return out

    def _file_of_class(self, fqn: str) -> Optional[FileIndex]:
        """
        Return the FileIndex that declared a given class FQN.

        Example:
            >>> # idx._file_of_class("com.example.DummyTest")
        """
        for f in self.file_indexes:
            if fqn in f.classes:
                return f
        return None


# ==================================================================================
#                                Graph data structure
# ==================================================================================

@dataclass
class CodeNode:
    """
    A node in the output graph (nx.DiGraph), with metadata.

    Attributes:
        name: Symbol name (method, variable, class, etc.).
        node_type: "method" | "class" | "variable" | "field" | "method_call" | ...
        class_name: Enclosing class (for decls) or receiver class (for calls).
        code_path: File where the node occurs.
        line: 1-based line number.
        import_statement: FQN if applicable (for type nodes).
        expandable: Whether node could expand to more nodes (e.g., a call).
        parents: Breadcrumbs (upstream node names).
        generic_type: For containers, the element type T (best-effort).
        return_type: For method_call nodes, inferred return type (best-effort).
    """
    name: str
    node_type: str
    class_name: Optional[str] = None
    code_path: Optional[Path] = None
    line: Optional[int] = None
    import_statement: Optional[str] = None
    expandable: bool = False
    parents: List[str] = field(default_factory=list)
    generic_type: Optional[str] = None
    return_type: Optional[str] = None


# ==================================================================================
#                                     Analyzer
# ==================================================================================

class CodeAnalyzer:
    """
    SECOND PASS (Improvements #2, #3, #5, #6, #7):
    Analyze from a start method and build a call/usage graph with type inference.

    Usage:
        >>> analyzer = CodeAnalyzer(["DummyTest.java"], start_method="start")
        >>> graph = analyzer.analyze()
        >>> analyzer.print_graph_structure()
        >>> analyzer.print_chains()
    """

    def __init__(self, file_paths: Union[str, List[str]], start_method: str, start_class: Optional[str] = None):
        """
        Args:
            file_paths: One or many .java files to index.
            start_method: The method name to start traversal from (e.g., 'start').
            start_class: Optional (simple or FQN) to disambiguate the start method.

        Example:
            >>> CodeAnalyzer(["DummyTest.java"], "start")  # doctest: +SKIP
        """
        if isinstance(file_paths, (str, os.PathLike)):
            file_paths = [file_paths]
        self.paths = [Path(p) for p in file_paths]
        self.index = ProjectIndexer().index_files(self.paths)  # ✔ multi-pass

        self.primary_file = self.index.file_indexes[0] if self.index.file_indexes else None
        self.start_method = start_method
        self.start_class = start_class

        self.graph = nx.DiGraph()

    def _add_node(self, name: str, **kwargs) -> None:
        """
        Add/update a CodeNode in the graph by name.

        Guard:
          - Avoid overwriting a meaningful class_name with "UnknownClass".

        Example:
            >>> g = CodeAnalyzer.__new__(CodeAnalyzer)  # doctest: +SKIP
        """
        if not name:
            return
        if name not in self.graph:
            self.graph.add_node(name, data=CodeNode(name=name, **kwargs))
        else:
            cn: CodeNode = self.graph.nodes[name]["data"]
            for k, v in kwargs.items():
                if k == "class_name" and v == "UnknownClass" and cn.class_name not in (None, "UnknownClass"):
                    continue
                setattr(cn, k, v)

    def _resolve_start(self) -> Optional[Tuple[ClassInfo, MethodSig, FileIndex]]:
        """
        Pick a (class, method) to start from; uses start_class if provided, otherwise
        the first matching method name across indexed files.

        Returns:
            (ClassInfo, MethodSig, FileIndex) or None.
        """
        candidates: List[Tuple[ClassInfo, MethodSig, FileIndex]] = []

        if self.start_class:
            if "." in self.start_class:
                fqn_opts = [self.start_class] if self.start_class in self.index.class_index else []
            else:
                fqn_opts = list(self.index.simple_to_fqn.get(self.start_class, []))
            for fqn in fqn_opts:
                ci = self.index.class_index.get(fqn)
                if not ci:
                    continue
                for ms in ci.methods.get(self.start_method, []):
                    fidx = self.index._file_of_class(fqn)
                    if fidx:
                        candidates.append((ci, ms, fidx))

        if not candidates:
            for fidx in self.index.file_indexes:
                for ci in fidx.classes.values():
                    for ms in ci.methods.get(self.start_method, []):
                        candidates.append((ci, ms, fidx))

        return candidates[0] if candidates else None

    def analyze(self) -> Optional[nx.DiGraph]:
        """
        Build the call/usage graph from the chosen start method.

        Returns:
            The populated graph, or None if the start method could not be resolved.

        Example:
            >>> # g = CodeAnalyzer(["DummyTest.java"], "start").analyze()  # doctest: +SKIP
        """
        start = self._resolve_start()
        if not start:
            print("Could not resolve starting method. Provide 'start_class' or check files.")
            return None

        start_ci, start_ms, start_file = start

        # Root node for the traversal
        self._add_node(
            name=self.start_method,
            node_type="method",
            class_name=start_ci.fqn,
            code_path=start_file.path,
            line=start_ms.line,
            expandable=True,
        )

        # Scope symbol table seeded with class fields and parameters (improvement #2).
        scope_symbols: Dict[str, str] = {}
        scope_symbols.update(start_ci.fields)

        params_node = _field(start_ms.decl_node, "parameters")
        if params_node:
            for p in params_node.children:
                if p.type == "formal_parameter":
                    ptype = _text(start_file.src, _field(p, "type"))
                    pname = _text(start_file.src, _field(p, "name"))
                    scope_symbols[pname] = ptype
                    self._add_node(pname, node_type="parameter", class_name=start_ci.fqn, line=p.start_point[0] + 1)
                    self.graph.add_edge(self.start_method, pname, relation="is an input")
                    self._add_node(ptype, node_type="class", import_statement=self._import_of(ptype, start_file))
                    self.graph.add_edge(pname, ptype, relation="is an instance of")

        # Walk the method body
        body = _field(start_ms.decl_node, "body")
        if body:
            self._walk_scope(body, self.start_method, start_ci, start_file, scope_symbols)

        return self.graph

    def _import_of(self, simple_or_fqn: str, using_file: FileIndex) -> Optional[str]:
        """
        Return the FQN import path (if available) for a given type name.

        Example:
            >>> # _import_of("List", file_index) -> "java.util.List" (if imported)
        """
        if "." in simple_or_fqn:
            return simple_or_fqn
        if simple_or_fqn in using_file.imports:
            return using_file.imports[simple_or_fqn]
        if simple_or_fqn in JAVA_LANG_IMPLICIT:
            return JAVA_LANG_IMPLICIT[simple_or_fqn]
        return self.index.resolve_simple(simple_or_fqn, using_file)

    def _walk_scope(self, scope: Node, parent_name: str, cls: ClassInfo, fidx: FileIndex, scope_symbols: Dict[str, str]) -> None:
        """
        Process statements and nested blocks inside a scope/body.

        Handles:
            - local_variable_declaration
            - expression_statement
            - control-flow (if/else, loops, try/catch/finally, switch)
            - return_statement
            - nested blocks

        Example:
            >>> # internal traversal used by analyze()
        """
        src = fidx.src
        for ch in scope.named_children:
            t = ch.type

            if t == "local_variable_declaration":
                self._handle_local_var(ch, parent_name, cls, fidx, scope_symbols)

            elif t == "expression_statement" and ch.named_child_count > 0:
                self._handle_expression(ch.named_child(0), parent_name, cls, fidx, scope_symbols)

            elif t == "if_statement":
                cond = _field(ch, "condition")
                if cond:
                    node = cond
                    if node.type == "parenthesized_expression" and node.named_child_count > 0:
                        node = node.named_child(0)
                    self._handle_expression(node, parent_name, cls, fidx, scope_symbols)
                cons = _field(ch, "consequence")
                if cons:
                    self._walk_scope(cons, parent_name, cls, fidx, scope_symbols)
                alt = _field(ch, "alternative")
                if alt:
                    self._walk_scope(alt, parent_name, cls, fidx, scope_symbols)

            elif t in ("for_statement", "enhanced_for_statement", "while_statement", "do_statement"):
                # Improvement #3: loop constructs
                for name in ("init", "condition", "update", "body"):
                    n = _field(ch, name)
                    if n:
                        if name == "body":
                            self._walk_scope(n, parent_name, cls, fidx, scope_symbols)
                        else:
                            self._handle_expression(n, parent_name, cls, fidx, scope_symbols)

            elif t == "try_statement":
                # Improvement #3: exception flow
                res = _field(ch, "resource")
                if res:
                    self._handle_expression(res, parent_name, cls, fidx, scope_symbols)
                tryb = _field(ch, "body")
                if tryb:
                    self._walk_scope(tryb, parent_name, cls, fidx, scope_symbols)
                for cblock in _children_of_type(ch, "catch_clause"):
                    parm = _field(cblock, "parameter")
                    if parm:
                        et = _field(parm, "type")
                        en = _field(parm, "name")
                        if et and en:
                            scope_symbols[_text(src, en)] = _text(src, et)
                    cbody = _field(cblock, "body")
                    if cbody:
                        self._walk_scope(cbody, parent_name, cls, fidx, scope_symbols)
                fin = _field(ch, "finally_clause")
                if fin:
                    fbody = _field(fin, "body")
                    if fbody:
                        self._walk_scope(fbody, parent_name, cls, fidx, scope_symbols)

            elif t in ("switch_statement", "switch_expression"):
                # Improvement #3: switch
                disc = _field(ch, "value")
                if disc:
                    self._handle_expression(disc, parent_name, cls, fidx, scope_symbols)
                for scase in _children_of_type(ch, ("switch_block", "switch_block_statement_group", "switch_rule")):
                    self._walk_scope(scase, parent_name, cls, fidx, scope_symbols)

            elif t == "return_statement" and ch.named_child_count > 0:
                self._handle_expression(ch.named_child(0), parent_name, cls, fidx, scope_symbols)

            elif t in ("block", "else_clause"):
                self._walk_scope(ch, parent_name, cls, fidx, scope_symbols)

    def _handle_local_var(self, node: Node, parent: str, cls: ClassInfo, fidx: FileIndex, scope_symbols: Dict[str, str]) -> None:
        """
        Handle a local variable declaration (including generics and 'var').

        - Adds variable node and link to its declared/inferred type.
        - If initializer exists, analyze its expression.

        Example:
            'ProcessingService service = new ProcessingService();'
              service -> ProcessingService ("is an instance of")
              parent -> service; parent -> ProcessingService ("instantiates")
        """
        src = fidx.src
        tnode = _field(node, "type")
        decl = _field(node, "declarator")
        decls = [decl] if decl is not None else [d for d in node.named_children if d.type == "variable_declarator"]

        raw_type, gen_arg = _type_from_generic(src, tnode) if tnode else ("Unknown", None)
        for d in decls:
            name_node = _field(d, "name")
            val_node = _field(d, "value")
            var_name = _text(src, name_node)

            # Improvement #2: var inference beyond 'new' (use method return types, etc.)
            vtype = raw_type
            if raw_type == "var" and val_node:
                vtype = self._infer_type_from_expression(val_node, cls, fidx, scope_symbols) or "Unknown"

            scope_symbols[var_name] = gen_arg or vtype

            self._add_node(var_name, node_type="variable", class_name=cls.fqn, code_path=fidx.path,
                           line=node.start_point[0] + 1, expandable=True, parents=[parent], generic_type=gen_arg)
            self.graph.add_edge(parent, var_name)
            self._add_node(vtype, node_type="class", import_statement=self._import_of(vtype, fidx))
            self.graph.add_edge(var_name, vtype, relation="is an instance of")

            if val_node:
                self._handle_expression(val_node, var_name, cls, fidx, scope_symbols)

    def _infer_type_from_expression(self, node: Node, cls: ClassInfo, fidx: FileIndex, scope_symbols: Dict[str, str]) -> Optional[str]:
        """
        Best-effort type inference for expressions (Improvement #2).

        Handles:
            - constructor calls ('new T(...)') -> 'T'
            - identifier -> type from scope
            - method_invocation -> return type (via index + arity)
            - parenthesized / cast / array access element type

        Returns:
            Type name (text) if inferred, else None.
        """
        src = fidx.src
        t = node.type

        if t == "object_creation_expression":
            typ = _field(node, "type")
            if typ:
                return _text(src, typ)

        if t == "identifier":
            nm = _text(src, node)
            return scope_symbols.get(nm)

        if t == "method_invocation":
            obj = _field(node, "object")
            mname = _text(src, _field(node, "name"))
            args = _field(node, "arguments")
            arity = len(args.named_children) if args else 0

            recv_type = cls.fqn
            if obj:
                recv_sym = self._handle_expression(obj, cls.simple_name, cls, fidx, scope_symbols)
                recv_type = scope_symbols.get(recv_sym, recv_type)

            rt = self._resolve_method_return(recv_type, mname, arity)
            if rt:
                return rt

        if t == "parenthesized_expression" and node.named_child_count > 0:
            return self._infer_type_from_expression(node.named_child(0), cls, fidx, scope_symbols)

        if t == "cast_expression":
            tn = _field(node, "type")
            return _text(src, tn) if tn else None

        if t == "array_access":
            arr = _field(node, "array")
            arr_name = self._handle_expression(arr, cls.simple_name, cls, fidx, scope_symbols)
            arr_t = scope_symbols.get(arr_name)
            if arr_t and arr_t.endswith("[]"):
                return arr_t[:-2]

        return None

    def _resolve_method_return(self, recv_type: str, name: str, arity: int) -> Optional[str]:
        """
        Return a best-effort return type for a receiver method (by arity),
        including Stream pipeline rules (Improvement #6) and inheritance walk (Improvement #5).
        """
        recv_fqn = recv_type
        if "." not in recv_type and self.primary_file:
            recv_fqn = self.index.resolve_simple(recv_type, self.primary_file) or recv_type

        # Stream rules (map/filter/flatMap/anyMatch/etc.)
        if name in STREAM_RULES:
            template, _ = STREAM_RULES[name]
            # Derive $T if recv_type looks generic (e.g., List<T>, Stream<T>)
            el = None
            if "<" in recv_type and ">" in recv_type:
                el = recv_type[recv_type.find("<")+1: recv_type.rfind(">")]
            if "$T" in template:
                return template.replace("$T", el or "java.lang.Object")
            if "$R" in template:
                return template.replace("$R", "java.lang.Object")
            return template

        # Inheritance-aware lookup
        if recv_fqn:
            methods = self.index.find_methods(recv_fqn, name, arity)
            if methods:
                return methods[0].return_type

        return None

    def _handle_arguments(self, arg_node: Optional[Node], call_parent_name: str, cls: ClassInfo, fidx: FileIndex, scope_symbols: Dict[str, str]) -> None:
        """
        Handle argument_list: identifiers (link), nested calls, lambdas, constructors.

        Example:
            f(x, g(y))  -> links: f -> x, f -> g, g -> y
        """
        if not arg_node or arg_node.type != "argument_list":
            return
        for arg in arg_node.named_children:
            if arg.type == "identifier":
                name = _text(fidx.src, arg)
                if name not in self.graph:
                    self._add_node(name, node_type="argument")
                self.graph.add_edge(call_parent_name, name, relation="uses as argument")
            else:
                self._handle_expression(arg, call_parent_name, cls, fidx, scope_symbols)

    def _handle_expression(self, node: Node, parent_name: str, cls: ClassInfo, fidx: FileIndex, scope_symbols: Dict[str, str]) -> Optional[str]:
        """
        Evaluate an expression node; add graph nodes/edges; return a terminal symbol name when appropriate.

        Covers:
            - identifier
            - field_access
            - object_creation_expression (new)
            - method_invocation (chained)
            - lambda_expression  (Improvement #6: param type inferred from generic context)
            - method_reference   (obj::method / Type::staticMethod)
            - member_reference   (Type.NAME via scoped identifiers)
            - array_access
            - assignment_expression
            - cast_expression / parenthesized_expression
        """
        src = fidx.src
        t = node.type

        # identifier -> returns variable/param name
        if t == "identifier":
            return _text(src, node)

        # a.b -> field access
        if t == "field_access":
            obj = _field(node, "object")
            fld = _field(node, "field")
            parent_obj = self._handle_expression(obj, parent_name, cls, fidx, scope_symbols)
            fld_name = _text(src, fld)
            self._add_node(fld_name, node_type="field_access", parents=[parent_obj])
            if parent_obj:
                self.graph.add_edge(parent_obj, fld_name)
            return fld_name

        # arr[idx] -> array access; return arr symbol and record the access
        if t == "array_access":
            arr = _field(node, "array")
            idx = _field(node, "index")
            arr_name = self._handle_expression(arr, parent_name, cls, fidx, scope_symbols)
            if idx:
                self._handle_expression(idx, parent_name, cls, fidx, scope_symbols)
            if arr_name:
                self._add_node(f"{arr_name}[]", node_type="array_access")
                self.graph.add_edge(arr_name, f"{arr_name}[]", relation="index")
            return arr_name

        # x = expr   (assignment; propagate type)
        if t == "assignment_expression":
            left = _field(node, "left")
            right = _field(node, "right")
            lval = self._handle_expression(left, parent_name, cls, fidx, scope_symbols)
            rval = self._handle_expression(right, parent_name, cls, fidx, scope_symbols)
            if lval and rval:
                scope_symbols[lval] = scope_symbols.get(rval, scope_symbols.get(lval, "Unknown"))
                self.graph.add_edge(lval, rval, relation="assigned from")
            return lval

        # (T) expr  (cast; update the variable's type if identifiable)
        if t == "cast_expression":
            typ = _field(node, "type")
            expr = _field(node, "expression")
            sym = self._handle_expression(expr, parent_name, cls, fidx, scope_symbols)
            if sym and typ:
                scope_symbols[sym] = _text(src, typ)
            return sym

        # new T(args)
        if t == "object_creation_expression":
            typ = _field(node, "type")
            obj_type = _text(src, typ) if typ else "UnknownClass"
            self._add_node(obj_type, node_type="class_instantiation", parents=[parent_name])
            self.graph.add_edge(parent_name, obj_type, relation="instantiates")
            self._handle_arguments(_field(node, "arguments"), parent_name, cls, fidx, scope_symbols)
            return obj_type

        # lambda: params -> body (infer parameter type from generic context)
        if t == "lambda_expression":
            params = _field(node, "parameters")
            body = _field(node, "body")
            lambda_param_name = None

            if params:
                if params.type in ("lambda_parameters", "inferred_parameters") and params.named_child_count == 1 and params.named_child(0).type == "identifier":
                    lambda_param_name = _text(src, params.named_child(0))
                elif params.type == "identifier":
                    lambda_param_name = _text(src, params)

            if lambda_param_name and parent_name in self.graph:
                parent_data: CodeNode = self.graph.nodes[parent_name]["data"]
                inferred = parent_data.generic_type
                if inferred:
                    scope_symbols[lambda_param_name] = inferred

            if body:
                self._walk_scope(body, parent_name, cls, fidx, scope_symbols)
            return None

        # obj::method or Type::staticMethod
        if t == "method_reference":
            q = _field(node, "object") or _field(node, "type")
            nm = _field(node, "name")
            qual = _text(src, q) if q else ""
            mname = _text(src, nm)
            ref_name = f"{qual}::{mname}" if qual else f"::{mname}"
            self._add_node(ref_name, node_type="method_reference", parents=[parent_name])
            self.graph.add_edge(parent_name, ref_name, relation="refers to")
            return ref_name

        # Type.NAME (scoped identifiers)
        if t in ("scoped_identifier", "scoped_type_identifier"):
            ref = _text(src, node)
            self._add_node(ref, node_type="member_reference", parents=[parent_name])
            self.graph.add_edge(parent_name, ref, relation="refers to")
            return ref

        # call or obj.call(...)
        if t == "method_invocation":
            mname = _text(src, _field(node, "name"))
            obj = _field(node, "object")
            args = _field(node, "arguments")
            arity = len(args.named_children) if args else 0

            chain_parent = parent_name
            recv_type = cls.fqn  # default receiver = current class

            if obj:
                chain_parent = self._handle_expression(obj, parent_name, cls, fidx, scope_symbols)
                if chain_parent:
                    recv_type = scope_symbols.get(chain_parent, recv_type)

            self._add_node(
                mname, node_type="method_call", class_name=cls.fqn, code_path=fidx.path,
                line=node.start_point[0] + 1, expandable=True, parents=[chain_parent]
            )
            if chain_parent:
                self.graph.add_edge(chain_parent, mname, relation="calls method")

            # Propagate generic element type to help lambdas downstream
            parent_data = self.graph.nodes.get(chain_parent, {}).get("data")
            if parent_data and parent_data.generic_type:
                self.graph.nodes[mname]["data"].generic_type = parent_data.generic_type

            # Resolve and store return type
            ret_type = self._resolve_method_return(recv_type, mname, arity)
            if ret_type:
                self.graph.nodes[mname]["data"].return_type = ret_type
                # If it looks generic, propagate inner type for chaining
                if "<" in ret_type and ">" in ret_type:
                    inner = ret_type[ret_type.find("<")+1: ret_type.rfind(">")]
                    self.graph.nodes[mname]["data"].generic_type = inner

            # Analyze arguments
            self._handle_arguments(args, mname, cls, fidx, scope_symbols)
            return mname

        # Parentheses: pass through
        if t == "parenthesized_expression" and node.named_child_count > 0:
            return self._handle_expression(node.named_child(0), parent_name, cls, fidx, scope_symbols)

        return None

    def print_graph_structure(self) -> None:
        """
        Pretty-print the graph in a readable tree, with relation labels.

        Example output:
            -> start (type=method, expandable=True)
              |--[is an input]-->
              -> reporter (type=parameter, ...)
              |--[calls method]-->
              -> processRequest (type=method_call, return=String)
        """
        if not self.graph:
            print("Graph is empty. Run analyze() first.")
            return

        print("\n--- Call Graph Structure ---")
        printed = set()

        def show(n: str, indent: str = "") -> None:
            if n not in self.graph:
                return
            data: CodeNode = self.graph.nodes[n].get("data")
            if not data:
                return
            details = [f"type={data.node_type}"]
            if data.expandable:
                details.append("expandable=True")
            if data.import_statement:
                details.append(f"import='{data.import_statement}'")
            if data.return_type:
                details.append(f"return={data.return_type}")
            print(f"{indent}-> {data.name} ({', '.join(details)})")

            for succ in sorted(self.graph.successors(n)):
                edge = (n, succ)
                if edge in printed:
                    continue
                printed.add(edge)
                rel = (self.graph.get_edge_data(n, succ) or {}).get("relation")
                if rel:
                    print(f"{indent}  |--[{rel}]-->")
                show(succ, indent + "  ")

        roots = [self.start_method] if self.start_method in self.graph else list(self.graph.nodes)
        for r in roots:
            show(r)
        print("--------------------------")

    def print_chains(self) -> None:
        """
        Print all simple paths from the start method to leaves.

        Example:
            start -> service -> processRequest
            start -> defaultCalc -> isOf -> isValid
        """
        if not self.graph:
            print("Graph is empty. Run analyze() first.")
            return

        print("\n--- Call Chains ---")
        leaves = [n for n, outd in self.graph.out_degree() if outd == 0]
        paths = []
        if self.start_method in self.graph:
            for lf in leaves:
                for p in nx.all_simple_paths(self.graph, source=self.start_method, target=lf):
                    paths.append(p)

        if not paths:
            print("No complete chains found from the start object.")
        else:
            for p in sorted(paths):
                print(" -> ".join(p))
        print("-------------------")


# ==================================================================================
#                                 Test scaffolding (8)
# ==================================================================================

def run_smoke_demo() -> None:
    """
    Minimal smoke test to verify parsing & traversal.

    How to build "golden tests" (suggested):
      1) Create a folder 'tests/java/' with small Java files, each illustrating a feature
         (constructors, generics, lambdas, method references, inner classes, records,
          enums, switch expressions, try-with-resources, var, arrays/maps).
      2) For each file, run the analyzer and assert on key nodes/edges:
           - that 'varName' exists and has edge 'is an instance of' -> 'Type'
           - that call chains include expected sequences
           - that lambdas propagate element type to parameter names
      3) Store expected edges in simple CSV/JSON and compare to graph edges.

    This function only runs a tiny in-memory example and prints results.
    """
    dummy_java_code = """
package com.example;

import com.x.y.z.DefaultCalc;
import com.x.y.z.TransEvent;
import com.x.y.z.Reporter;
import java.util.List;
import java.util.Optional;
import java.util.stream.Stream;

public class DummyTest {

    private final List<ApplicablePredicate> predicates = null;

    public void start(Reporter reporter) {
        System.out.println("Main process started.");
        ProcessingService service = new ProcessingService();
        String result = service.processRequest("user-123");
        DefaultCalc defaultCalc = new DefaultCalc();
        defaultCalc.isOf(result).isValid(service);
        TransEvent ts = new TransEvent(reporter);

        if (isapp(ts)){
            Optional.of(ts);
        } else {
            Optional.empty();
        }
    }

    private boolean isapp(TransEvent event){
        String ispord = event.isdaprodval().getornull();
        String ptemp = event.gettemp().getornull();

        return predicates.stream()
            .anyMatch(predicate->predicate.isapp(ispord,ptemp));
    }

    public static class ApplicablePredicate {
        public static final String NA = "NA";
        public static final String Any = "ANY";

        String ispord;
        String ptemp;

        private ApplicablePredicate(String ispord, String ptemp){
            this.ispord = ispord;
            this.ptemp = ptemp;
        }

        public boolean isapp(String p1, String p2) {
            return true;
        }
    }
}

class ProcessingService {
    public String processRequest(String user) { return "data"; }
}

class Reporter {
    public String isdaprodval() { return ""; }
    public String gettemp() { return ""; }
}
class TransEvent {
    public TransEvent(Reporter r) {}
    public String isdaprodval() { return ""; }
    public String gettemp() { return ""; }
}
""".lstrip()

    dummy_file = Path("DummyTest.java")
    dummy_file.write_text(dummy_java_code, encoding="utf-8")
    print(f"Created dummy file '{dummy_file}' for analysis.")

    analyzer = CodeAnalyzer([str(dummy_file)], start_method="start")
    graph = analyzer.analyze()
    if graph:
        analyzer.print_graph_structure()
        analyzer.print_chains()

    dummy_file.unlink(missing_ok=True)
    print(f"\nCleaned up dummy file '{dummy_file}'.")


# ==================================================================================
#                                        Demo
# ==================================================================================

if __name__ == "__main__":
    # Run the built-in smoke demo. Replace with your project files as needed.
    run_smoke_demo()
