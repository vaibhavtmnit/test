"""
Java Call-Graph Builder with Multi-File Indexing + Strong Type Inference (tree-sitter)

This module builds a directional call/usage graph from Java code. It works in two passes:

1) ProjectIndexer (FIRST PASS)
   - Indexes packages, imports, classes (incl. inheritance), fields, and methods across one or many files.
   - Produces fast lookups: simple name -> FQN, class -> methods, class -> fields, inheritance graph.

2) CodeAnalyzer (SECOND PASS)
   - Walks statements/expressions from a chosen start method.
   - Infers types (var, assignments, casts, constructor return, method return).
   - Resolves method calls by receiver type (with simple overload & inheritance walk).
   - Handles common Java constructs, including lambdas, method references, loops, try/catch, switch, arrays/maps, etc.

Install:
    pip install tree-sitter tree-sitter-java networkx

Tested with py-tree-sitter >= 0.21.

Author’s note:
    Designed to be *robust yet explainable*. It favors clear heuristics and upgrade points over complex full-Java typing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable, Set, Union
from collections import defaultdict, deque

import networkx as nx

import tree_sitter_java as tsj
from tree_sitter import Language, Parser, Node


# ======================================================================================
#                                  Tree-sitter setup
# ======================================================================================

try:
    JAVA = Language(tsj.language())
except Exception as e:
    print(f"Error loading tree-sitter Java language: {e}")
    print("Ensure 'tree-sitter' and 'tree-sitter-java' are installed and importable.")
    raise

PARSER = Parser(JAVA)


# ======================================================================================
#                                Small helper utilities
# ======================================================================================

JAVA_LANG_IMPLICIT = {
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
    # add more as needed
}

STREAM_RULES = {
    # method_name: (return_type_template, lambda_param_from_receiver_generic)
    # template '$T' = element type of receiver generic; '$R' = lambda result type (unknown here)
    "stream": ("java.util.stream.Stream<$T>", False),
    "map": ("java.util.stream.Stream<$R>", True),          # map(T -> R) => Stream<R>
    "flatMap": ("java.util.stream.Stream<$R>", True),      # flatMap(T -> Stream<R>) simplified
    "filter": ("java.util.stream.Stream<$T>", False),
    "anyMatch": ("boolean", False),
    "allMatch": ("boolean", False),
    "noneMatch": ("boolean", False),
    "collect": ("java.lang.Object", False),                # simplified
    "forEach": ("void", False),
    "toList": ("java.util.List<$T>", False),               # Stream<T>.toList()
}

# --------------------------------------------------------------------------------------

def _text(src: str, node: Optional[Node]) -> str:
    """Return source text for a node or empty string if None."""
    if node is None:
        return ""
    return src[node.start_byte: node.end_byte]

def _field(node: Node, name: str) -> Optional[Node]:
    """Safe child_by_field_name wrapper."""
    if node is None:
        return None
    return node.child_by_field_name(name)

def _children_of_type(node: Node, t: Union[str, Tuple[str, ...]]) -> Iterable[Node]:
    """Yield named children with given type(s)."""
    for c in node.named_children:
        if (isinstance(t, str) and c.type == t) or (isinstance(t, tuple) and c.type in t):
            yield c

def _has_modifier(src: str, node: Node, mod: str) -> bool:
    """
    Returns True if a declaration node has a given modifier (e.g., 'static', 'public').
    Works by scanning a 'modifiers' child when present; otherwise scans leading tokens.
    """
    mods = _field(node, "modifiers")
    if mods:
        # modifiers: sequence of modifier nodes or tokens
        for c in mods.children:
            if _text(src, c) == mod:
                return True
    else:
        # Fallback heuristic: look left siblings for raw text 'static'
        # Not perfect; grammar should provide 'modifiers' usually.
        pass
    return False

def _split_qualified(name: str) -> Tuple[str, Optional[str]]:
    """Return (package, simple_name) if qualified; else ('', name)."""
    if "." in name:
        *pkg, simple = name.split(".")
        return ".".join(pkg), simple
    return "", name

def _type_from_generic(src: str, type_node: Node) -> Tuple[str, Optional[str]]:
    """
    Given a type node, return (raw_type, generic_arg) where generic_arg is the first type argument if present.
    Example:
        'List<Predicate>' -> ('List', 'Predicate')
        'Map<K,V>'        -> ('Map', 'K')  # simplified (first arg)
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

def _arity_of_parameters(src: str, params_node: Optional[Node]) -> int:
    """Return the number of formal parameters for a method_declaration parameters node."""
    if not params_node:
        return 0
    count = 0
    for ch in params_node.children:
        if ch.type == "formal_parameter":
            count += 1
    return count


# ======================================================================================
#                                    Index structures
# ======================================================================================

@dataclass
class MethodSig:
    """
    Represents a method signature attached to a class/interface.

    Attributes:
        name: Method name (simple).
        params: Parameter type names (simple or FQN, as parsed).
        return_type: Return type name (simple or FQN).
        is_static: Whether method is static.
        decl_node: The AST node for the method_declaration (for navigation).
        line: 1-based line number.
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
        simple_name: Simple class name.
        extends: Superclass FQN or simple (if unresolved).
        implements: List of interface names (FQN or simple).
        fields: Map var_name -> type_name (first generic arg propagated when obvious).
        methods: Map method_name -> list[MethodSig] (support overloads).
        node: AST node for the declaration (for context).
    """
    fqn: str
    kind: str
    package: str
    simple_name: str
    extends: Optional[str] = None
    implements: List[str] = field(default_factory=list)
    fields: Dict[str, str] = field(default_factory=dict)
    methods: Dict[str, List[MethodSig]] = field(default_factory=lambda: defaultdict(list))
    node: Optional[Node] = None


@dataclass
class FileIndex:
    """
    Per-file index: package, imports, wildcard imports, static imports, and classes found.
    """
    path: Path
    package: str = ""
    imports: Dict[str, str] = field(default_factory=dict)        # simple -> FQN
    wildcard_imports: List[str] = field(default_factory=list)    # 'com.pkg.*'
    static_imports: List[str] = field(default_factory=list)      # static imports (members or wildcards)
    classes: Dict[str, ClassInfo] = field(default_factory=dict)  # fqn -> ClassInfo
    src: str = ""


class ProjectIndexer:
    """
    FIRST PASS: Build a symbol table across one or more files.

    Usage:
        >>> idx = ProjectIndexer().index_files(["A.java", "B.java"])
        >>> fqn = idx.simple_to_fqn.get("DummyTest")
        >>> cls = idx.class_index.get(fqn)

    What is indexed:
        - Packages (package_declaration)
        - Imports (including static and wildcard)
        - Type declarations (class/interface/enum/record) with fields and methods
        - Inheritance links (extends/implements)
    """

    def __init__(self):
        self.file_indexes: List[FileIndex] = []
        self.class_index: Dict[str, ClassInfo] = {}     # FQN -> ClassInfo
        self.simple_to_fqn: Dict[str, Set[str]] = defaultdict(set)  # 'List' -> {'java.util.List', ...}

    # -----------------------------------

    def index_files(self, file_paths: List[Union[str, Path]]) -> "ProjectIndexer":
        """Index all given files. Returns self for chaining."""
        for p in file_paths:
            self._index_one(Path(p))
        # Accumulate cross-file quick lookups
        for fidx in self.file_indexes:
            for fqn, ci in fidx.classes.items():
                self.class_index[fqn] = ci
                self.simple_to_fqn[ci.simple_name].add(fqn)
        return self

    # -----------------------------------

    def _index_one(self, path: Path) -> None:
        """Index a single .java file: package, imports, classes, members."""
        if not path.exists() or path.suffix.lower() != ".java":
            return
        src = path.read_text(encoding="utf-8")
        tree = PARSER.parse(src.encode("utf-8"))
        root = tree.root_node

        fidx = FileIndex(path=path, src=src)

        # Package
        for node in root.named_children:
            if node.type == "package_declaration":
                name_node = _field(node, "name")
                fidx.package = _text(src, name_node)
                break

        # Imports
        for (node, _) in JAVA.query("(import_declaration) @imp").captures(root):
            text = _text(src, node).strip().rstrip(";")
            if text.startswith("import "):
                text = text[len("import "):].strip()
            is_static = False
            if text.startswith("static "):
                is_static = True
                text = text[len("static "):].strip()
            if text.endswith(".*"):
                if is_static:
                    fidx.static_imports.append(text)
                else:
                    fidx.wildcard_imports.append(text)
            else:
                if is_static:
                    # static single-member import (e.g., java.lang.Math.max)
                    fidx.static_imports.append(text)
                else:
                    simple = text.split(".")[-1]
                    fidx.imports[simple] = text

                # Also opportunity to map the simple name to its fqn
                simple = text.split(".")[-1]
                self.simple_to_fqn[simple].add(text)

        # Classes/interfaces/enums/records
        decl_query = JAVA.query(r"""
            [
              (class_declaration) @cls
              (interface_declaration) @intf
              (enum_declaration) @enm
              (record_declaration) @rec
            ]
        """)
        for (node, cap) in decl_query.captures(root):
            kind = {"cls": "class", "intf": "interface", "enm": "enum", "rec": "record"}[cap]
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

            # Extends / implements (best-effort)
            sc = _field(node, "superclass")
            if sc:
                ci.extends = _text(src, sc)
            si = _field(node, "super_interfaces")
            if si:
                for idn in si.named_children:
                    ci.implements.append(_text(src, idn))

            # Members in body
            body = _field(node, "body")
            if body:
                for mem in body.named_children:
                    if mem.type == "field_declaration":
                        tnode = _field(mem, "type")
                        raw_t, gen = _type_from_generic(src, tnode)
                        # One or many declarators (variable_declarator)
                        for d in mem.named_children:
                            if d.type == "variable_declarator":
                                nm = _field(d, "name")
                                var_name = _text(src, nm)
                                ci.fields[var_name] = gen or raw_t

                    elif mem.type == "method_declaration":
                        # method name, params types, return type, static?
                        name_n = _field(mem, "name")
                        mname = _text(src, name_n)
                        params = []
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
                        ms = MethodSig(
                            name=mname, params=params, return_type=rtype,
                            is_static=is_static, decl_node=mem, line=mem.start_point[0] + 1
                        )
                        ci.methods[mname].append(ms)

            fidx.classes[fqn] = ci

        self.file_indexes.append(fidx)

    # -----------------------------------

    def resolve_simple(self, name: str, using_file: FileIndex) -> Optional[str]:
        """
        Resolve a simple type name to a FQN using:
          - File imports
          - File wildcard imports
          - java.lang implicit
          - Any globally indexed classes by simple name (if unique)
        """
        if not name or "." in name:
            return name

        # Exact file-import
        if name in using_file.imports:
            return using_file.imports[name]

        # java.lang implicit
        if name in JAVA_LANG_IMPLICIT:
            return JAVA_LANG_IMPLICIT[name]

        # Wildcards: try to find a unique class under any wildcard package
        candidates = set()
        for pkg_star in using_file.wildcard_imports:
            pkg = pkg_star[:-2]  # drop '.*'
            fqn = f"{pkg}.{name}"
            if fqn in self.class_index:
                candidates.add(fqn)
        if len(candidates) == 1:
            return next(iter(candidates))

        # Global unique simple-name
        if name in self.simple_to_fqn and len(self.simple_to_fqn[name]) == 1:
            return next(iter(self.simple_to_fqn[name]))

        # Unresolved -> return simple
        return name

    def super_chain(self, fqn: str) -> Iterable[str]:
        """
        Yield the class and its superclasses upwards (best-effort, by names as indexed).
        """
        seen = set()
        cur = fqn
        while cur and cur not in seen:
            seen.add(cur)
            yield cur
            ci = self.class_index.get(cur)
            if not ci or not ci.extends:
                break
            # Resolve extends if simple
            ext = ci.extends
            if "." not in ext:
                pkg, simple = _split_qualified(cur)
                # same package?
                xfqn = f"{pkg}.{ext}" if pkg else ext
                cur = self.class_index.get(xfqn) and xfqn or self.resolve_simple(ext, self._file_of_class(cur))
            else:
                cur = ext if ext in self.class_index else None

    def find_methods(self, recv_fqn: str, name: str, arity: int) -> List[MethodSig]:
        """
        Locate method candidates by name + arity along the inheritance chain.
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
        """Find the FileIndex that defined a given class FQN."""
        for f in self.file_indexes:
            if fqn in f.classes:
                return f
        return None


# ======================================================================================
#                                   Graph structures
# ======================================================================================

@dataclass
class CodeNode:
    """
    Node placed in the output graph (nx.DiGraph).

    Attributes:
        name: Symbol name (variable/method/class/etc).
        node_type: "method" | "class" | "variable" | "field" | "method_call" | ...
        class_name: Enclosing class (for declarations) or receiver class (for calls).
        code_path: File where the node occurs.
        line: 1-based line number.
        import_statement: FQN import if known for type nodes.
        expandable: Whether this node might expand to more nodes.
        parents: Upstream nodes (breadcrumbs).
        generic_type: For generic containers, tracked 'T' (e.g., List<T> -> T).
        return_type: For method_call nodes, the inferred return type (if resolved).
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


# ======================================================================================
#                                      Analyzer
# ======================================================================================

class CodeAnalyzer:
    """
    SECOND PASS: Analyze from a start method and build a call/usage graph.

    Create with one or more files:
        >>> analyzer = CodeAnalyzer(["DummyTest.java"], start_method="start")
        >>> graph = analyzer.analyze()

    You can optionally provide `start_class` (simple or FQN) to disambiguate.
    """

    def __init__(self, file_paths: Union[str, List[str]], start_method: str, start_class: Optional[str] = None):
        if isinstance(file_paths, (str, os.PathLike)):
            file_paths = [file_paths]

        self.paths = [Path(p) for p in file_paths]
        self.index = ProjectIndexer().index_files(self.paths)

        # Choose a primary file for starting context (first path).
        self.primary_file = self.index.file_indexes[0] if self.index.file_indexes else None

        self.start_method = start_method
        self.start_class = start_class  # simple or FQN; optional

        self.graph = nx.DiGraph()

    # -----------------------------------

    def _add_node(self, name: str, **kwargs) -> None:
        """Graph helper: add/update CodeNode by name."""
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

    # -----------------------------------

    def _resolve_start(self) -> Optional[Tuple[ClassInfo, MethodSig, FileIndex]]:
        """
        Pick a (class, method) pair to start from, using start_class if provided,
        otherwise the first method named start_method across indexed files.

        Returns:
            (ClassInfo, MethodSig, FileIndex) or None
        """
        candidates: List[Tuple[ClassInfo, MethodSig, FileIndex]] = []

        # With start_class
        if self.start_class:
            # Resolve start_class to FQN(s)
            fqn_opts = []
            if "." in self.start_class:
                fqn_opts = [self.start_class] if self.start_class in self.index.class_index else []
            else:
                fqn_opts = list(self.index.simple_to_fqn.get(self.start_class, []))

            for fqn in fqn_opts:
                ci = self.index.class_index.get(fqn)
                if not ci:
                    continue
                ms_list = ci.methods.get(self.start_method, [])
                for ms in ms_list:
                    fidx = self.index._file_of_class(fqn)
                    if fidx:
                        candidates.append((ci, ms, fidx))

        # Without start_class: scan all
        if not candidates:
            for fidx in self.index.file_indexes:
                for ci in fidx.classes.values():
                    for ms in ci.methods.get(self.start_method, []):
                        candidates.append((ci, ms, fidx))

        # Return first candidate (or choose by heuristic; here first is fine)
        return candidates[0] if candidates else None

    # -----------------------------------

    def analyze(self) -> Optional[nx.DiGraph]:
        """
        Build the call/usage graph from the chosen start method.

        Returns:
            The populated graph, or None if start could not be resolved.

        Example:
            >>> analyzer = CodeAnalyzer(["DummyTest.java"], "start")
            >>> g = analyzer.analyze()
            >>> isinstance(g, nx.DiGraph)
            True
        """
        start = self._resolve_start()
        if not start:
            print("Could not resolve starting method. Provide 'start_class' or check files.")
            return None

        start_ci, start_ms, start_file = start

        # Root node: the start method
        self._add_node(
            name=self.start_method,
            node_type="method",
            class_name=start_ci.fqn,
            code_path=start_file.path,
            line=start_ms.line,
            expandable=True,
        )

        # Start traversal within the method body
        scope_symbols: Dict[str, str] = {}
        # Seed class fields
        scope_symbols.update(start_ci.fields)

        # Seed parameters
        params_node = _field(start_ms.decl_node, "parameters")
        if params_node:
            for p in params_node.children:
                if p.type == "formal_parameter":
                    ptype = _text(start_file.src, _field(p, "type"))
                    pname = _text(start_file.src, _field(p, "name"))
                    scope_symbols[pname] = ptype
                    self._add_node(pname, node_type="parameter", class_name=start_ci.fqn, line=p.start_point[0] + 1)
                    # link method -> parameter
                    self.graph.add_edge(self.start_method, pname, relation="is an input")
                    # ensure class node exists for ptype
                    self._add_node(ptype, node_type="class", import_statement=self._import_of(ptype, start_file))
                    self.graph.add_edge(pname, ptype, relation="is an instance of")

        # Walk method body
        body = _field(start_ms.decl_node, "body")
        if body:
            self._walk_scope(body, self.start_method, start_ci, start_file, scope_symbols)

        return self.graph

    # -----------------------------------

    def _import_of(self, simple_or_fqn: str, using_file: FileIndex) -> Optional[str]:
        """Return the FQN import path (if available) for a given type name."""
        if "." in simple_or_fqn:
            return simple_or_fqn
        if simple_or_fqn in using_file.imports:
            return using_file.imports[simple_or_fqn]
        # implicit java.lang
        if simple_or_fqn in JAVA_LANG_IMPLICIT:
            return JAVA_LANG_IMPLICIT[simple_or_fqn]
        # wildcard resolution if unique
        return self.index.resolve_simple(simple_or_fqn, using_file)

    # -----------------------------------

    def _walk_scope(self, scope: Node, parent_name: str, cls: ClassInfo, fidx: FileIndex, scope_symbols: Dict[str, str]) -> None:
        """
        Process statements and nested blocks inside a scope/body.

        Handles:
            - local_variable_declaration
            - expression_statement
            - control-flow (if/else, loops, try/catch/finally, switch)
            - return_statement
            - nested blocks
        """
        src = fidx.src

        for ch in scope.named_children:
            t = ch.type

            # --- Declarations: local variables ---
            if t == "local_variable_declaration":
                self._handle_local_var(ch, parent_name, cls, fidx, scope_symbols)

            # --- Expression statements ---
            elif t == "expression_statement" and ch.named_child_count > 0:
                self._handle_expression(ch.named_child(0), parent_name, cls, fidx, scope_symbols)

            # --- If/Else ---
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

            # --- Loops ---
            elif t in ("for_statement", "enhanced_for_statement", "while_statement", "do_statement"):
                # handle inits/conditions/updates as expressions if present
                for name in ("init", "condition", "update", "body"):
                    n = _field(ch, name)
                    if n:
                        if name == "body":
                            self._walk_scope(n, parent_name, cls, fidx, scope_symbols)
                        else:
                            self._handle_expression(n, parent_name, cls, fidx, scope_symbols)

            # --- Try/Catch/Finally ---
            elif t == "try_statement":
                res = _field(ch, "resource")
                if res:
                    self._handle_expression(res, parent_name, cls, fidx, scope_symbols)
                tryb = _field(ch, "body")
                if tryb:
                    self._walk_scope(tryb, parent_name, cls, fidx, scope_symbols)
                # catches
                for cblock in _children_of_type(ch, "catch_clause"):
                    parm = _field(cblock, "parameter")
                    if parm:
                        # exception variable comes into scope
                        et = _field(parm, "type")
                        en = _field(parm, "name")
                        if et and en:
                            scope_symbols[_text(src, en)] = _text(src, et)
                    cbody = _field(cblock, "body")
                    if cbody:
                        self._walk_scope(cbody, parent_name, cls, fidx, scope_symbols)
                # finally
                fin = _field(ch, "finally_clause")
                if fin:
                    fbody = _field(fin, "body")
                    if fbody:
                        self._walk_scope(fbody, parent_name, cls, fidx, scope_symbols)

            # --- Switch (statement/expression) ---
            elif t in ("switch_statement", "switch_expression"):
                disc = _field(ch, "value")
                if disc:
                    self._handle_expression(disc, parent_name, cls, fidx, scope_symbols)
                for scase in _children_of_type(ch, ("switch_block", "switch_block_statement_group", "switch_rule")):
                    self._walk_scope(scase, parent_name, cls, fidx, scope_symbols)

            # --- Return ---
            elif t == "return_statement" and ch.named_child_count > 0:
                self._handle_expression(ch.named_child(0), parent_name, cls, fidx, scope_symbols)

            # --- Nested blocks / else_clause ---
            elif t in ("block", "else_clause"):
                self._walk_scope(ch, parent_name, cls, fidx, scope_symbols)

    # -----------------------------------

    def _handle_local_var(self, node: Node, parent: str, cls: ClassInfo, fidx: FileIndex, scope_symbols: Dict[str, str]) -> None:
        """Process a local variable declaration: type name = value; (incl. generics, 'var', initializer)."""
        src = fidx.src
        tnode = _field(node, "type")
        decl = _field(node, "declarator")
        # Some grammars use multiple 'variable_declarator' children instead of field 'declarator'
        decls = []
        if decl:
            decls = [decl]
        else:
            decls = [d for d in node.named_children if d.type == "variable_declarator"]

        raw_type, gen_arg = _type_from_generic(src, tnode) if tnode else ("Unknown", None)
        for d in decls:
            name_node = _field(d, "name")
            val_node = _field(d, "value")
            var_name = _text(src, name_node)

            # var inference
            vtype = raw_type
            if raw_type == "var" and val_node:
                vtype = self._infer_type_from_expression(val_node, cls, fidx, scope_symbols) or "Unknown"

            # record in-scope types
            scope_symbols[var_name] = gen_arg or vtype

            # add variable node + edges
            self._add_node(var_name, node_type="variable", class_name=cls.fqn, code_path=fidx.path,
                           line=node.start_point[0] + 1, expandable=True, parents=[parent], generic_type=gen_arg)
            self.graph.add_edge(parent, var_name)
            self._add_node(vtype, node_type="class", import_statement=self._import_of(vtype, fidx))
            self.graph.add_edge(var_name, vtype, relation="is an instance of")

            # initializer expression extends the chain
            if val_node:
                self._handle_expression(val_node, var_name, cls, fidx, scope_symbols)

    # -----------------------------------

    def _infer_type_from_expression(self, node: Node, cls: ClassInfo, fidx: FileIndex, scope_symbols: Dict[str, str]) -> Optional[str]:
        """Heuristics to infer a type from an expression."""
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
            # derive receiver + resolve method -> return type
            obj = _field(node, "object")
            mname = _text(src, _field(node, "name"))
            args = _field(node, "arguments")
            arity = len(args.named_children) if args else 0

            recv_type = cls.fqn  # default if no explicit object
            if obj:
                recv_sym = self._handle_expression(obj, cls.simple_name, cls, fidx, scope_symbols)  # propagate chain
                recv_type = scope_symbols.get(recv_sym, recv_type)

            rt = self._resolve_method_return(recv_type, mname, arity)
            if rt:
                return rt

        if t == "parenthesized_expression" and node.named_child_count > 0:
            return self._infer_type_from_expression(node.named_child(0), cls, fidx, scope_symbols)

        if t == "cast_expression":
            tn = _field(node, "type")
            return _text(src, tn) if tn else None

        # array access returns element type if known
        if t == "array_access":
            arr = _field(node, "array")
            arr_name = self._handle_expression(arr, cls.simple_name, cls, fidx, scope_symbols)
            arr_t = scope_symbols.get(arr_name)
            if arr_t and arr_t.endswith("[]"):
                return arr_t[:-2]

        return None

    # -----------------------------------

    def _resolve_method_return(self, recv_type: str, name: str, arity: int) -> Optional[str]:
        """
        Return the best-effort return type for a receiver method (by arity),
        including basic Stream/lambda rules when applicable.
        """
        # Resolve receiver type to FQN
        recv_fqn = recv_type
        if "." not in recv_type and self.primary_file:
            recv_fqn = self.index.resolve_simple(recv_type, self.primary_file) or recv_type

        # Try rules for common stream pipeline methods
        if name in STREAM_RULES:
            template, _ = STREAM_RULES[name]
            # Extract $T if recv_type has generic T
            el = None
            if "<" in recv_type and ">" in recv_type:
                # Already FQN-like generic "List<Something>"
                el = recv_type[recv_type.find("<")+1: recv_type.rfind(">")]
            # Or if we track generic_type in caller, omitted here; best-effort:
            if "$T" in template:
                return template.replace("$T", el or "java.lang.Object")
            if "$R" in template:
                return template.replace("$R", "java.lang.Object")  # unknown lambda result
            return template

        # Look up methods on the class and supertypes
        if recv_fqn:
            methods = self.index.find_methods(recv_fqn, name, arity)
            if methods:
                # Pick the first match (could refine with param types later)
                return methods[0].return_type

        return None

    # -----------------------------------

    def _handle_arguments(self, arg_node: Node, call_parent_name: str, cls: ClassInfo, fidx: FileIndex, scope_symbols: Dict[str, str]) -> None:
        """Process an argument_list: identifiers, nested calls, lambdas, constructor calls, literals."""
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

    # -----------------------------------

    def _handle_expression(self, node: Node, parent_name: str, cls: ClassInfo, fidx: FileIndex, scope_symbols: Dict[str, str]) -> Optional[str]:
        """
        Evaluate an expression node; add graph nodes/edges; return a terminal symbol name if applicable.

        Handles:
            - identifier
            - field_access
            - object_creation_expression (new)
            - method_invocation (chained)
            - lambda_expression
            - method_reference  (obj::method / Type::method)
            - member_reference  (Type.NAME) via 'scoped_identifier' shapes
            - array_access
            - assignment_expression
            - cast_expression / parenthesized_expression (passthrough)
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

        # arr[idx] -> array access; link as an access; return arr symbol
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

        # x = expr, possibly compound
        if t == "assignment_expression":
            left = _field(node, "left")
            right = _field(node, "right")
            lval = self._handle_expression(left, parent_name, cls, fidx, scope_symbols)
            rval = self._handle_expression(right, parent_name, cls, fidx, scope_symbols)
            if lval and rval:
                # propagate type: type(lval) = type(rval)
                scope_symbols[lval] = scope_symbols.get(rval, scope_symbols.get(lval, "Unknown"))
                self.graph.add_edge(lval, rval, relation="assigned from")
            return lval

        # (T) expr -> cast; return expr symbol but update type if possible
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

        # lambda: params -> body   (infer parameter type from generic context when possible)
        if t == "lambda_expression":
            params = _field(node, "parameters")
            body = _field(node, "body")
            lambda_param_name = None

            if params:
                # Common shapes: lambda_parameters | inferred_parameters | identifier
                if params.type in ("lambda_parameters", "inferred_parameters") and params.named_child_count == 1 and params.named_child(0).type == "identifier":
                    lambda_param_name = _text(src, params.named_child(0))
                elif params.type == "identifier":
                    lambda_param_name = _text(src, params)

            if lambda_param_name and parent_name in self.graph:
                # Try to infer generic T from the call chain context (e.g., 'predicates.stream().anyMatch(predicate -> ...)').
                parent_data: CodeNode = self.graph.nodes[parent_name]["data"]
                inferred = parent_data.generic_type
                if inferred:
                    scope_symbols[lambda_param_name] = inferred

            # Recurse into body
            if body:
                self._walk_scope(body, parent_name, cls, fidx, scope_symbols)
            return None

        # obj::method or Type::staticMethod
        if t == "method_reference":
            # Best-effort: capture qualifier and name, add a node, link
            q = _field(node, "object") or _field(node, "type")
            nm = _field(node, "name")
            qual = _text(src, q) if q else ""
            mname = _text(src, nm)
            ref_name = f"{qual}::{mname}" if qual else f"::{mname}"
            self._add_node(ref_name, node_type="method_reference", parents=[parent_name])
            self.graph.add_edge(parent_name, ref_name, relation="refers to")
            return ref_name

        # Scoped identifier for static member: Type.NAME (some grammars expose as 'scoped_identifier')
        if t in ("scoped_identifier", "scoped_type_identifier"):
            # Attach a member reference node
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
            recv_type = cls.fqn  # default to current class if no explicit receiver

            if obj:
                chain_parent = self._handle_expression(obj, parent_name, cls, fidx, scope_symbols)
                if chain_parent:
                    recv_type = scope_symbols.get(chain_parent, recv_type)

            # Create method_call node
            self._add_node(
                mname, node_type="method_call", class_name=cls.fqn, code_path=fidx.path,
                line=node.start_point[0] + 1, expandable=True, parents=[chain_parent]
            )

            if chain_parent:
                self.graph.add_edge(chain_parent, mname, relation="calls method")

            # Assign/propagate generic_type to help lambdas downstream
            parent_data = self.graph.nodes.get(chain_parent, {}).get("data")
            if parent_data and parent_data.generic_type:
                self.graph.nodes[mname]["data"].generic_type = parent_data.generic_type

            # Resolve return type for this call (best-effort)
            ret_type = self._resolve_method_return(recv_type, mname, arity)
            if ret_type:
                self.graph.nodes[mname]["data"].return_type = ret_type
                # If the receiver is a generic container and rule yields Stream<$T> etc.,
                # try to derive generic_type for chaining.
                if "<" in ret_type and ">" in ret_type:
                    inner = ret_type[ret_type.find("<")+1: ret_type.rfind(">")]
                    self.graph.nodes[mname]["data"].generic_type = inner

            # Process arguments (identifiers, nested calls, lambdas, constructors)
            self._handle_arguments(args, mname, cls, fidx, scope_symbols)

            # If the method is declared in current file set, we can expand it (optional, name-only)
            # Here we expand only when it belongs to the same class for simplicity.
            # (Could be extended to look up by resolved FQN + arity.)
            for ms in cls.methods.get(mname, []):
                if len(ms.params) == arity:
                    # enqueue for expansion: not implemented as BFS here to avoid cycles;
                    # you can add a worklist if you want cross-method traversal.
                    pass

            return mname

        # Parentheses: pass through
        if t == "parenthesized_expression" and node.named_child_count > 0:
            return self._handle_expression(node.named_child(0), parent_name, cls, fidx, scope_symbols)

        # Literals, binary, conditional, etc. can be added similarly as needed.
        return None

    # -----------------------------------

    def print_graph_structure(self) -> None:
        """
        Pretty-print the graph: node plus children, with relation labels.

        Example:
            >>> analyzer.print_graph_structure()
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

        # Start from start_method if present; otherwise print entire graph roots.
        roots = [self.start_method] if self.start_method in self.graph else list(self.graph.nodes)
        for r in roots:
            show(r)
        print("--------------------------")

    # -----------------------------------

    def print_chains(self) -> None:
        """
        List all simple paths from the start method to graph leaves.

        Example:
            >>> analyzer.print_chains()
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


# ======================================================================================
#                                        Demo
# ======================================================================================

if __name__ == "__main__":
    """
    Demo: create a dummy Java file, run the indexer + analyzer, print the graph,
    then clean up. Extend to multiple files by adding more .java files to 'files'.
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

    files = [str(dummy_file)]
    analyzer = CodeAnalyzer(files, start_method="start")  # optionally: start_class="DummyTest" or "com.example.DummyTest"
    graph = analyzer.analyze()

    if graph:
        analyzer.print_graph_structure()
        analyzer.print_chains()

    dummy_file.unlink(missing_ok=True)
    print(f"\nCleaned up dummy file '{dummy_file}'.")





Software Engineer (Python) – AI Workflows & Multi-Agent Systems

Regulatory Transaction Reporting (RTR) | London (Hybrid)

What you’ll do

We’re looking for someone who can:

Design and build Python services and tools that power RTR transformation and digitization—focusing on reliability, testability, and performance.

Develop AI workflows and multi-agent systems (e.g., with LangGraph) to automate investigation, exception triage, QA, and reporting processes across MiFID II, EMIR, SFTR and FinfraG.

Ship production-grade APIs (REST/gRPC) and libraries that integrate with upstream/downstream platforms in Operations and Technology.

Prototype quickly (POCs) and convert successful experiments into hardened, maintainable software with CI/CD, observability, and documentation.

Collaborate with data scientists to turn foundational DS/ML ideas into robust services (feature computation, model execution, evaluation, and monitoring).

Champion engineering excellence: code reviews, testing culture, automation, and continuous improvement.

Contribute to solution architecture and roadmaps in partnership with Business and Technology stakeholders.

You don’t need to be a data-science specialist, but you should be comfortable working with DS/ML colleagues and have foundational knowledge of DS/ML concepts so you can build the software that operationalizes them.

Your expertise (must-haves)

5+ years of professional Python software development, delivering production systems.

Strong grasp of software engineering fundamentals:

Testing: unit, integration, contract tests (e.g., pytest, fixtures, coverage).

APIs: building and consuming REST/gRPC services (e.g., FastAPI), pagination, auth, versioning.

Packaging & dependency management: pyproject.toml, virtualenv/poetry/pip, semantic versioning.

Code quality: type hints (mypy/pyright), linters/formatters (ruff/black), logging/metrics.

CI/CD: GitHub/GitLab/Azure DevOps pipelines, artifact management, release automation.

Runtime: Docker/containerization, environment configuration, secrets management.

Hands-on with AI workflow orchestration and multi-agent patterns, ideally LangGraph (tools/function-calling, memory/state, guards, retries, evaluation).

Foundational DS/ML knowledge: data wrangling, basic modeling concepts, evaluation metrics, and model lifecycle awareness (even if models are built by others).

Experience integrating with data platforms and eventing (files/DBs, queues, batch/stream, schemas/validation).

Clear, pragmatic communication and the ability to work across global teams and stakeholders.

Nice to have (pluses)

Azure Machine Learning (jobs, endpoints, model registry) and/or MLflow for tracking.

Frontend exposure: React (TypeScript), building simple UIs to operationalize tooling.

Vector stores/RAG tooling, prompt/evaluation frameworks, or policy/guardrails experience.

Experience in financial services or regulatory reporting (MiFID II, EMIR, SFTR, FinfraG).

Databricks/Azure data stack familiarity.

Your team

You’ll be working in the Regulatory Transaction Reporting (RTR) team in London, which is part of UBS Investment Bank Operations. We’re principally responsible for providing governance, exceptions management, QA, risk management and client services for MiFID II, EMIR, SFTR and FinfraG transaction reporting. We collaborate with UBS Investment Bank, Wealth Management and group functions globally.
With over 120 specialists and experts working in the UK, Switzerland, Poland, US and Asia, we are a talent powerhouse that attracts and develops the best people by driving career growth in and outside RTR.

About UBS

UBS is the world's largest and the only truly global wealth manager. We operate through four business divisions: Global Wealth Management, Personal & Corporate Banking, Asset Management and the Investment Bank. Our global reach and the breadth of our expertise set us apart from our competitors. We have a presence in all major financial centers in more than 50 countries.

Join us

At UBS, we embrace flexible ways of working when the role permits. We offer different working arrangements like part-time, job-sharing and hybrid (office and home) working. Our purpose-led culture and global infrastructure help us connect, collaborate and work together in agile ways to meet all our business needs. From gaining new experiences in different roles to acquiring fresh knowledge and skills, we know that great work is never done alone. We know that it's our people, with their unique backgrounds, skills, experience levels and interests, who drive our ongoing success. Together we're more than ourselves. Ready to be part of #teamUBS and make an impact?

Disclaimer / Policy Statements

UBS is an Equal Opportunity Employer. We respect and seek to empower each individual and support the diverse cultures, perspectives, skills and experiences within our workforce.

