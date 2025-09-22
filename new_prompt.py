# analysis_stack.py
"""
LLM-guided Java analyzer: prompts + schemas + deterministic tools (single file).

Requires:
  pip install pydantic tree_sitter tree_sitter_java

What you get:
  • Pydantic models for all LLM calls (structured responses)
  • Focus-aware prompt templates (system + user) for:
      - method-call finder
      - object/variable discovery
      - object typing
      - relationship validation
  • Deterministic helpers using tree-sitter-java:
      - scan_direct_calls(...)
      - list_scope_symbols(..., filter_chainable=True)
      - resolve_type_hints(...)
      - validate_relationship(...)
  • Focus descriptors (method | object | call_result)
  • Model config presets for o3-mini / o1-preview

Notes:
  - Prompts are written to *prefer empty lists over guesses*.
  - Tools are deliberately conservative; UNKNOWN beats hallucination.
  - Spans are 0-based (line/col).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field
from tree_sitter import Language, Parser, Node
import tree_sitter_java as tsjava


# ──────────────────────────────────────────────────────────────────────────────
# Model/config presets (tune per your infra)
# ──────────────────────────────────────────────────────────────────────────────

MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    # Use o3-mini for fast extraction/typing
    "method_call_finder": {"model": "o3-mini", "temperature": 0, "reasoning_effort": "low", "max_output_tokens": 1200},
    "object_discovery":   {"model": "o3-mini", "temperature": 0, "reasoning_effort": "low", "max_output_tokens": 1200},
    "object_typing":      {"model": "o3-mini", "temperature": 0, "reasoning_effort": "medium", "max_output_tokens": 1400},
    # Use o1-preview for adjudication/validation only when needed
    "relationship_validator": {"model": "o1-preview", "temperature": 0, "reasoning_effort": "medium", "max_output_tokens": 1200},
}


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic schemas (shared by tools & LLM responses)
# ──────────────────────────────────────────────────────────────────────────────

class Span(BaseModel):
    file: str
    start_line: int
    start_col: int
    end_line: int
    end_col: int


class MethodCall(BaseModel):
    name: str
    receiver: Optional[str] = None   # "this" | "super" | object var | "<call_result>"
    occurrence: Span
    chain_index: int                 # 0 for method focus; 1 for one-hop on object/call_result
    confidence: float
    evidence: str


class MethodCallFinderOut(BaseModel):
    calls: List[MethodCall] = Field(default_factory=list)
    notes: Optional[str] = None


class ObjectItem(BaseModel):
    name: str
    kind: Literal["parameter", "local"]
    decl_span: Span
    first_use_span: Optional[Span] = None
    notes: Optional[str] = None
    confidence: float


class ObjectDiscoveryOut(BaseModel):
    objects: List[ObjectItem] = Field(default_factory=list)
    excluded: List[Dict[str, str]] = Field(default_factory=list)


class TypeOut(BaseModel):
    variable: str
    type: str  # FQN if known, else simple, else "UNKNOWN"
    evidence: List[Span] = Field(default_factory=list)
    confidence: float
    unknown_reason: Optional[str] = None
    suggested_followup: Optional[Dict[str, Any]] = None


class Verdict(BaseModel):
    child: Dict[str, str]            # {"name": "...", "kind": "..."}
    valid: bool
    reason: str
    confidence: float
    normalized_child_name: Optional[str] = None


class RelationshipValidationOut(BaseModel):
    verdicts: List[Verdict] = Field(default_factory=list)
    summary: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────────
# Prompt packs (focus-aware; empty lists allowed; strict JSON)
# ──────────────────────────────────────────────────────────────────────────────

PROMPTS: Dict[str, Dict[str, str]] = {
    # 1) Method-call finder (focus-aware: method | object | call_result)
    "method_call_finder_system": (
        "You extract method calls RELATIVE to the FOCUS.\n"
        "- If FOCUS.kind == 'method': return only direct calls declared in the same class, "
        "invoked in the focus method's body (e.g., 'isapp(...)'). Exclude calls on other objects, chained calls, and JDK/library calls.\n"
        "- If FOCUS.kind == 'object': return one-hop calls whose receiver is exactly the object variable (e.g., 'defaultCalc.isOf(...)'). "
        "Do NOT include further chained calls.\n"
        "- If FOCUS.kind == 'call_result': return one-hop calls where the receiver is the result of the specified call site "
        "(e.g., the next '.isValid(...)' after '.isOf(...)').\n"
        "It is valid to return ZERO results; never guess. Return STRICT JSON matching the schema. No prose."
    ),
    "method_call_finder_user": (
        "FOCUS:\n{json_focus}\n\n"
        "FILE: {file_path}\n\n"
        "CANDIDATE_CALL_SPANS:\n{json_candidate_calls}\n\n"
        "SNIPPET:\n{code_snippet}\n\n"
        "RULES:\n"
        "- Follow the behavior for FOCUS.kind exactly.\n"
        "- Report each qualifying call with the precise name-token span.\n"
        "- Do NOT include deeper chained calls for 'object' or 'call_result' focus.\n"
        "- Returning an empty list is valid.\n\n"
        "OUTPUT SCHEMA (JSON):\n"
        "{{\n"
        '  "calls": [\n'
        '    {{\n'
        '      "name": "string",\n'
        '      "receiver": "string|null",\n'
        '      "occurrence": {{"file":"string","start_line":0,"start_col":0,"end_line":0,"end_col":0}},\n'
        '      "chain_index": 0,\n'
        '      "confidence": 0.0,\n'
        '      "evidence": "short reason tied to span"\n'
        "    }}\n"
        "  ],\n"
        '  "notes": "string"\n'
        "}}\n"
        "Return ONLY a JSON object."
    ),

    # 2) Object/variable discovery (arguments + locals, with optional chainable filter)
    "object_discovery_system": (
        "You extract method arguments and directly scoped local variables that participate in the immediate execution line "
        "of the focus method. Exclude class fields and literals. If filter_chainable=true is indicated in the context, "
        "only include locals that are receivers of a call whose result is immediately chained within the method body.\n"
        "Empty results are valid. Return STRICT JSON only."
    ),
    "object_discovery_user": (
        "FOCUS_METHOD: {fqn}\n"
        "FILE: {file_path}\n"
        "FILTER_CHAINABLE: {filter_chainable}\n\n"
        "AST_SYMBOLS (params/locals):\n{symbol_table_json}\n\n"
        "SNIPPET:\n{code_snippet}\n\n"
        "RULES:\n"
        "- Include all method parameters.\n"
        "- Include local variables declared in the focus method body (not fields). "
        "If FILTER_CHAINABLE is true, keep only locals that are receivers of a call whose result is immediately chained.\n"
        "- Exclude literals and variables only introduced inside nested lambdas/inner classes.\n"
        "- It is valid to return [].\n\n"
        "OUTPUT SCHEMA:\n"
        "{{\n"
        '  "objects": [\n'
        '    {{\n'
        '      "name": "string",\n'
        '      "kind": "parameter|local",\n'
        '      "decl_span": {{"file":"string","start_line":0,"start_col":0,"end_line":0,"end_col":0}},\n'
        '      "first_use_span": {{"file":"string","start_line":0,"start_col":0,"end_line":0,"end_col":0}} | null,\n'
        '      "notes": "string" | null,\n'
        '      "confidence": 0.0\n'
        "    }}\n"
        "  ],\n"
        '  "excluded": [{{"name":"string","reason":"literal|field|nested-scope|not-direct"}}]\n'
        "}}\n"
        "Return ONLY a JSON object."
    ),

    # 3) Object typing (determine class/FQN for each discovered object)
    "object_typing_system": (
        "Determine the Java type for the target variable using only index facts, AST spans, and code snippets provided. "
        "Prefer explicit declarations; infer from constructors; otherwise return type='UNKNOWN' with a reason. "
        "Empty evidence is allowed. Return STRICT JSON only; no guessing."
    ),
    "object_typing_user": (
        "TARGET: {variable_name}\n"
        "CONTEXT_METHOD: {fqn}\n\n"
        "INDEX_FACTS:\n{index_slice_json}\n\n"
        "EVIDENCE_SPANS:\n{spans_json}\n\n"
        "SNIPPET:\n{code_snippet}\n\n"
        "RULES:\n"
        "- If parameter: use declared type.\n"
        "- If local with explicit type: use it. If 'var': infer from constructor on RHS when present.\n"
        "- If assigned from method call with unknown return: type='UNKNOWN'.\n"
        "- Provide confidence in [0,1]. It is valid to return UNKNOWN.\n\n"
        "OUTPUT SCHEMA:\n"
        "{{\n"
        '  "variable": "string",\n'
        '  "type": "string",\n'
        '  "evidence": [{{"file":"string","start_line":0,"start_col":0,"end_line":0,"end_col":0}}],\n'
        '  "confidence": 0.0,\n'
        '  "unknown_reason": "string" | null,\n'
        '  "suggested_followup": {{"tool":"resolve_imports|index_search|scan_constructors","args":{{}}}} | null\n'
        "}}\n"
        "Return ONLY a JSON object."
    ),

    # 4) Relationship validator (parent → child validity)
    "relationship_validator_system": (
        "Validate parent→child relationships for a code analysis tree. "
        "A child is valid if: (a) it's a direct intra-method call issued by the parent method; or "
        "(b) it's a parameter/local declared in that method; or "
        "(c) for parent=object, it's a one-hop call on that object; or "
        "(d) for parent=call_result, it's the one-hop chained call right after that call site. "
        "Reject literals, fields, and deeper chains. Return STRICT JSON only. Empty lists allowed."
    ),
    "relationship_validator_user": (
        "PARENT:\n{parent_json}\n\n"
        "CANDIDATES:\n{candidates_json}\n\n"
        "INDEX_FACTS:\n{index_slice_json}\n\n"
        "SNIPPET:\n{code_snippet}\n\n"
        "DECISION RULES:\n"
        "- child.kind='call' under method: must be unqualified or this/super qualified direct call.\n"
        "- child.kind in {parameter,local} under method: must be declared in that method.\n"
        "- child.kind='call' under object: one-hop call whose receiver is exactly the object variable.\n"
        "- child.kind='call' under call_result: one-hop chained call off that call site.\n"
        "- Provide reason + confidence for each verdict. It is valid to return [].\n\n"
        "OUTPUT SCHEMA:\n"
        "{{\n"
        '  "verdicts": [\n'
        '    {{"child": {{"name":"string","kind":"call|parameter|local"}}, "valid": true|false, "reason":"string", "confidence": 0.0, "normalized_child_name": "string|null"}}\n'
        "  ],\n"
        '  "summary":"string"\n'
        "}}\n"
        "Return ONLY a JSON object."
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# Focus descriptors
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FocusMethod:
    """Focus on a method by (body) span; used for direct intra-method calls & scope discovery."""
    site_span: Span             # span covering the method body (block) or method decl
    symbol: Optional[str] = None


@dataclass
class FocusObject:
    """Focus on a variable declared at method scope."""
    name: str
    decl_span: Span             # span of the variable declarator (identifier token)


@dataclass
class FocusCallResult:
    """Focus on the result of a specific call (by name-token span)."""
    from_call_name: str
    site_span: Span             # span of the callee name token (method_invocation.name)


Focus = Union[FocusMethod, FocusObject, FocusCallResult]


# ──────────────────────────────────────────────────────────────────────────────
# Tree-sitter setup & AST utilities
# ──────────────────────────────────────────────────────────────────────────────

JAVA = Language(tsjava.language())
_PARSER = Parser(JAVA)

def _parse(code: str):
    tree = _PARSER.parse(code.encode("utf-8"))
    return tree, code.encode("utf-8")

def _node_span(node: Node, file_path: str) -> Span:
    (sl, sc) = node.start_point
    (el, ec) = node.end_point
    return Span(file=file_path, start_line=sl, start_col=sc, end_line=el, end_col=ec)

def _text(code_bytes: bytes, node: Node) -> str:
    return code_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

def _inside(a: Span, b: Span) -> bool:
    if a.file != b.file: return False
    if (a.start_line, a.start_col) < (b.start_line, b.start_col): return False
    if (a.end_line, a.end_col) > (b.end_line, b.end_col): return False
    return True

def _node_inside_span(node: Node, span: Span, file_path: str) -> bool:
    return _inside(_node_span(node, file_path), span)

def _has_ancestor_in(node: Node, types: Tuple[str, ...]) -> bool:
    cur = node.parent
    while cur is not None:
        if cur.type in types: return True
        cur = cur.parent
    return False

def _find_method_decl_by_span(root: Node, span: Span, file_path: str) -> Optional[Node]:
    stack = [root]
    while stack:
        n = stack.pop()
        if n.type == "method_declaration" and _node_inside_span(n, span, file_path):
            return n
        for i in range(n.child_count - 1, -1, -1): stack.append(n.child(i))
    return None

def _method_body_node(method_decl: Node) -> Optional[Node]:
    if method_decl.type == "block":
        return method_decl
    return method_decl.child_by_field_name("body")

def _extract_params(method_decl: Node, file_path: str) -> List[ObjectItem]:
    items: List[ObjectItem] = []
    params = method_decl.child_by_field_name("parameters")
    if not params: return items
    for child in params.children:
        if child.type in ("formal_parameter", "receiver_parameter", "spread_parameter"):
            name_node = child.child_by_field_name("name")
            if not name_node: continue
            items.append(ObjectItem(
                name=_text(b"", name_node),
                kind="parameter",
                decl_span=_node_span(name_node, file_path),
                confidence=0.99
            ))
    return items

def _extract_locals_in_method_body(body: Node, file_path: str, code_bytes: bytes) -> List[ObjectItem]:
    out: List[ObjectItem] = []
    stack = [body]
    while stack:
        n = stack.pop()
        if n is not body and n.type in ("method_declaration","class_body","interface_body","enum_body","record_body"):
            continue
        if _has_ancestor_in(n, ("lambda_expression",)):
            continue
        if n.type == "local_variable_declaration":
            for c in n.children:
                if c.type == "variable_declarator":
                    name_node = c.child_by_field_name("name")
                    if not name_node: continue
                    out.append(ObjectItem(
                        name=_text(code_bytes, name_node),
                        kind="local",
                        decl_span=_node_span(name_node, file_path),
                        confidence=0.95
                    ))
        for i in range(n.child_count - 1, -1, -1): stack.append(n.child(i))
    return out

def _iter_method_invocations(root: Node):
    stack = [root]
    while stack:
        n = stack.pop()
        if n.type == "method_invocation": yield n
        for i in range(n.child_count - 1, -1, -1): stack.append(n.child(i))

def _method_invocation_parts(mi: Node, code_bytes: bytes) -> Dict[str, Any]:
    name_node = mi.child_by_field_name("name")
    obj_node = mi.child_by_field_name("object")
    name = _text(code_bytes, name_node) if name_node else ""
    receiver = _text(code_bytes, obj_node).strip() if obj_node else None
    return {"name": name, "receiver": receiver, "name_node": name_node or mi, "mi_node": mi}

def _same_identifier(receiver_text: Optional[str], var_name: str) -> bool:
    return receiver_text == var_name if receiver_text else False

def _find_method_invocation_by_name_span(root: Node, span: Span, file_path: str) -> Optional[Node]:
    # Return method_invocation whose 'name' field matches the span
    stack = [root]
    while stack:
        n = stack.pop()
        if n.type == "method_invocation":
            name_node = n.child_by_field_name("name")
            if name_node and _inside(_node_span(name_node, file_path), span):
                return n
        for i in range(n.child_count - 1, -1, -1): stack.append(n.child(i))
    return None

def _chainable_receivers_in_method_body(body: Node, code_bytes: bytes) -> set[str]:
    names: set[str] = set()
    stack = [body]
    while stack:
        n = stack.pop()
        if n.type == "method_invocation":
            parent = n.parent
            if parent and parent.type == "method_invocation" and parent.child_by_field_name("object") == n:
                recv = n.child_by_field_name("object")
                if recv and recv.type == "identifier":
                    names.add(_text(code_bytes, recv))
        for i in range(n.child_count - 1, -1, -1): stack.append(n.child(i))
    return names


# ──────────────────────────────────────────────────────────────────────────────
# Deterministic tools (focus-aware)
# ──────────────────────────────────────────────────────────────────────────────

def scan_direct_calls(code: str, focus: Focus, file_path: str = "File.java") -> MethodCallFinderOut:
    """
    Focus-aware direct call scanner:
      - FocusMethod: unqualified or this/super direct calls in method body.
      - FocusObject: one-hop calls whose receiver == object name (exclude if this MI is an intermediate 'object' of a parent MI).
      - FocusCallResult: one-hop chained call using the given call site's result as receiver (parent MI whose 'object' == anchor MI).
    Returns empty list if nothing qualifies.
    """
    tree, code_bytes = _parse(code)
    root = tree.root_node
    out = MethodCallFinderOut(calls=[])

    def mk(name_node: Node, mi_node: Node, receiver: Optional[str], chain_idx: int, why: str, conf: float) -> MethodCall:
        return MethodCall(
            name=_text(code_bytes, name_node),
            receiver=receiver,
            occurrence=_node_span(name_node, file_path),
            chain_index=chain_idx,
            confidence=conf,
            evidence=why,
        )

    # Method focus
    if isinstance(focus, FocusMethod):
        method_decl = _find_method_decl_by_span(root, focus.site_span, file_path)
        body = _method_body_node(method_decl) if method_decl else None
        if not (method_decl and body):
            out.notes = "No method body found for span."
            return out
        for mi in _iter_method_invocations(body):
            parts = _method_invocation_parts(mi, code_bytes)
            if parts["receiver"] is None or parts["receiver"] in ("this", "super"):
                out.calls.append(mk(parts["name_node"], mi, parts["receiver"] or "this", 0,
                                    "Unqualified/this/super call in focus method body.", 0.96))
        return out

    # Object focus
    if isinstance(focus, FocusObject):
        containing_method = _find_method_decl_by_span(root, focus.decl_span, file_path)
        body = _method_body_node(containing_method) if containing_method else None
        if not body:
            out.notes = "Cannot locate enclosing method body for object decl."
            return out
        for mi in _iter_method_invocations(body):
            parts = _method_invocation_parts(mi, code_bytes)
            if _same_identifier(parts["receiver"], focus.name):
                parent_mi = mi.parent
                if parent_mi and parent_mi.type == "method_invocation" and parent_mi.child_by_field_name("object") == mi:
                    # intermediate hop; the parent owns the next call
                    continue
                out.calls.append(mk(parts["name_node"], mi, focus.name, 1,
                                    f"One-hop call on object '{focus.name}'.", 0.95))
        return out

    # Call-result focus
    if isinstance(focus, FocusCallResult):
        anchor = _find_method_invocation_by_name_span(root, focus.site_span, file_path)
        if not anchor:
            out.notes = "Could not locate anchor invocation for call_result span."
            return out
        parent_mi = anchor.parent
        if parent_mi and parent_mi.type == "method_invocation" and parent_mi.child_by_field_name("object") == anchor:
            parts = _method_invocation_parts(parent_mi, code_bytes)
            out.calls.append(mk(parts["name_node"], parent_mi, "<call_result>", 1,
                                "Parent invocation directly chained off focused call result.", 0.92))
        else:
            out.notes = "No one-hop chained invocation off this call result."
        return out

    out.notes = "Unsupported focus type."
    return out


def list_scope_symbols(code: str, method_site_span: Span, file_path: str = "File.java",
                       filter_chainable: bool = True) -> ObjectDiscoveryOut:
    """
    Return parameters + method-scope locals. If filter_chainable=True,
    keep only locals that are receivers of a call whose result is immediately chained.
    """
    tree, code_bytes = _parse(code)
    root = tree.root_node
    out = ObjectDiscoveryOut()

    method_decl = _find_method_decl_by_span(root, method_site_span, file_path)
    if not method_decl:
        out.excluded.append({"name": "<all>", "reason": "no-method-for-span"})
        return out

    # Params (always kept)
    out.objects.extend(_extract_params(method_decl, file_path))

    body = _method_body_node(method_decl)
    if not body:
        return out

    locals_items = _extract_locals_in_method_body(body, file_path, code_bytes)
    if filter_chainable:
        chainables = _chainable_receivers_in_method_body(body, code_bytes)
        locals_items = [it for it in locals_items if it.name in chainables]
        # mark excluded locals
        for it in _extract_locals_in_method_body(body, file_path, code_bytes):
            if it.name not in chainables:
                out.excluded.append({"name": it.name, "reason": "not-direct"})
    out.objects.extend(locals_items)
    return out


def resolve_type_hints(code: str, variable_name: str,
                       context_method_span: Optional[Span], file_path: str = "File.java") -> TypeOut:
    """
    Best-effort type resolution within a single file/context:
      1) Parameter → declared type.
      2) Local with explicit type → that type.
      3) 'var' or no type with constructor on RHS → constructor type.
      4) Otherwise → UNKNOWN with reason.
    """
    tree, code_bytes = _parse(code)
    root = tree.root_node

    def mk_span(node: Node) -> Span: return _node_span(node, file_path)

    method_decl = _find_method_decl_by_span(root, context_method_span, file_path) if context_method_span else None

    # Parameters
    if method_decl:
        params = method_decl.child_by_field_name("parameters")
        if params:
            for child in params.children:
                if child.type in ("formal_parameter", "receiver_parameter", "spread_parameter"):
                    name_node = child.child_by_field_name("name")
                    if name_node and _text(code_bytes, name_node) == variable_name:
                        tnode = child.child_by_field_name("type")
                        if tnode:
                            return TypeOut(variable=variable_name, type=_text(code_bytes, tnode).strip(),
                                           evidence=[mk_span(tnode)], confidence=0.99)
                        return TypeOut(variable=variable_name, type="UNKNOWN", evidence=[mk_span(child)],
                                       confidence=0.4, unknown_reason="Parameter without explicit type node.")

    # Locals (in body)
    search_root = _method_body_node(method_decl) if method_decl else root
    if search_root:
        stack = [search_root]
        while stack:
            n = stack.pop()
            if n.type == "local_variable_declaration":
                tnode = n.child_by_field_name("type")
                for c in n.children:
                    if c.type == "variable_declarator":
                        name_node = c.child_by_field_name("name")
                        if name_node and _text(code_bytes, name_node) == variable_name:
                            if tnode is not None:
                                declared = _text(code_bytes, tnode).strip()
                                if declared == "var":
                                    init = c.child_by_field_name("value")
                                    if init and init.type == "object_creation_expression":
                                        t1 = init.child_by_field_name("type")
                                        if t1 is not None:
                                            return TypeOut(variable=variable_name, type=_text(code_bytes, t1).strip(),
                                                           evidence=[mk_span(t1)], confidence=0.9)
                                    return TypeOut(variable=variable_name, type="UNKNOWN", evidence=[mk_span(c)],
                                                   confidence=0.5, unknown_reason="var init not resolvable")
                                return TypeOut(variable=variable_name, type=declared, evidence=[mk_span(tnode)], confidence=0.95)
                            else:
                                init = c.child_by_field_name("value")
                                if init and init.type == "object_creation_expression":
                                    t1 = init.child_by_field_name("type")
                                    if t1 is not None:
                                        return TypeOut(variable=variable_name, type=_text(code_bytes, t1).strip(),
                                                       evidence=[mk_span(t1)], confidence=0.85)
                                return TypeOut(variable=variable_name, type="UNKNOWN", evidence=[mk_span(c)],
                                               confidence=0.4, unknown_reason="No type & initializer not a constructor.")
            for i in range(n.child_count - 1, -1, -1): stack.append(n.child(i))

    return TypeOut(variable=variable_name, type="UNKNOWN", evidence=[], confidence=0.2,
                   unknown_reason="Variable not found in provided context.")


def validate_relationship(code: str, parent: Dict[str, Any], child: Dict[str, Any],
                          file_path: str = "File.java") -> RelationshipValidationOut:
    """
    Validate parent→child under your rules (method | object | call_result).
    """
    tree, code_bytes = _parse(code)
    root = tree.root_node
    verdicts: List[Verdict] = []

    def rej(reason: str, conf: float = 0.9):
        verdicts.append(Verdict(child={"name": child.get("name",""), "kind": child.get("kind","")},
                                valid=False, reason=reason, confidence=conf,
                                normalized_child_name=child.get("name","")))

    def ok(reason: str, conf: float = 0.95):
        verdicts.append(Verdict(child={"name": child.get("name",""), "kind": child.get("kind","")},
                                valid=True, reason=reason, confidence=conf,
                                normalized_child_name=child.get("name","")))

    if child.get("name","").startswith(("\"", "'")):
        rej("Child appears to be a literal.")
        return RelationshipValidationOut(verdicts=verdicts, summary="literal rejected")

    pk = parent.get("kind")
    ck = child.get("kind")

    if pk == "method":
        span_dict = parent.get("site_span")
        if not span_dict:
            rej("Parent method missing site_span.")
            return RelationshipValidationOut(verdicts=verdicts, summary="missing method span")
        mspan = Span(**span_dict)
        mdecl = _find_method_decl_by_span(root, mspan, file_path)
        if not mdecl:
            rej("Method not found for span.")
            return RelationshipValidationOut(verdicts=verdicts, summary="no method")
        body = _method_body_node(mdecl)

        if ck == "call":
            if not body:
                rej("Method has no body; cannot contain direct calls.")
                return RelationshipValidationOut(verdicts=verdicts, summary="no body")
            target = child.get("name","")
            found = False
            for mi in _iter_method_invocations(body):
                parts = _method_invocation_parts(mi, code_bytes)
                if parts["name"] == target and (parts["receiver"] is None or parts["receiver"] in ("this","super")):
                    found = True
                    break
            if found: ok("Direct unqualified/this/super call in parent method body.", 0.96)
            else:     rej("No matching direct call in parent method body.", 0.9)
            return RelationshipValidationOut(verdicts=verdicts, summary="method→call checked")

        if ck in ("parameter","local"):
            if ck == "parameter":
                present = any(p.name == child.get("name") for p in _extract_params(mdecl, file_path))
            else:
                present = False
                if body:
                    for it in _extract_locals_in_method_body(body, file_path, code_bytes):
                        if it.name == child.get("name"):
                            present = True; break
            if present: ok(f"{ck} is declared in parent method.", 0.95)
            else:       rej(f"{ck} not declared in parent method (likely field/out-of-scope).", 0.92)
            return RelationshipValidationOut(verdicts=verdicts, summary="method→symbol checked")

        rej(f"Unsupported child.kind for method parent: {ck}")
        return RelationshipValidationOut(verdicts=verdicts, summary="unsupported child")

    if pk == "object":
        name = parent.get("name")
        decl_span = parent.get("decl_span")
        if not (name and decl_span):
            rej("Object parent missing name/decl_span.")
            return RelationshipValidationOut(verdicts=verdicts, summary="missing object info")
        mdecl = _find_method_decl_by_span(root, Span(**decl_span), file_path)
        body = _method_body_node(mdecl) if mdecl else None
        if not body:
            rej("Cannot locate enclosing method body for object.")
            return RelationshipValidationOut(verdicts=verdicts, summary="no body")
        if ck != "call":
            rej("For object parent, only 'call' children are valid.")
            return RelationshipValidationOut(verdicts=verdicts, summary="invalid child kind")
        target = child.get("name","")
        for mi in _iter_method_invocations(body):
            parts = _method_invocation_parts(mi, code_bytes)
            if parts["name"] == target and _same_identifier(parts["receiver"], name):
                parent_mi = mi.parent
                if parent_mi and parent_mi.type == "method_invocation" and parent_mi.child_by_field_name("object") == mi:
                    continue
                ok("One-hop call on object receiver in enclosing method.", 0.95)
                return RelationshipValidationOut(verdicts=verdicts, summary="object→call ok")
        rej("No one-hop call on the specified object receiver found.")
        return RelationshipValidationOut(verdicts=verdicts, summary="object→call not found")

    if pk == "call_result":
        site_span = parent.get("site_span")
        if not site_span:
            rej("call_result parent missing site_span.")
            return RelationshipValidationOut(verdicts=verdicts, summary="missing site span")
        focused = scan_direct_calls(
            code=code,
            focus=FocusCallResult(from_call_name=parent.get("from_call_name",""), site_span=Span(**site_span)),
            file_path=file_path
        )
        if ck != "call":
            rej("For call_result parent, only 'call' children are valid.")
            return RelationshipValidationOut(verdicts=verdicts, summary="invalid child kind")
        ok_flag = any(c.name == child.get("name") for c in focused.calls)
        if ok_flag: ok("One-hop call chained off focused call_result.", 0.93)
        else:       rej("No one-hop chained call matching child found.", 0.9)
        return RelationshipValidationOut(verdicts=verdicts, summary="call_result→call checked")

    rej(f"Unsupported parent.kind: {pk}")
    return RelationshipValidationOut(verdicts=verdicts, summary="unsupported parent kind")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers to build user prompts from tool outputs (for your LangGraph nodes)
# ──────────────────────────────────────────────────────────────────────────────

def build_user_prompt_method_call_finder(json_focus: str, file_path: str,
                                         json_candidate_calls: str, code_snippet: str) -> str:
    return PROMPTS["method_call_finder_user"].format(
        json_focus=json_focus, file_path=file_path,
        json_candidate_calls=json_candidate_calls, code_snippet=code_snippet
    )

def build_user_prompt_object_discovery(fqn: str, file_path: str, filter_chainable: bool,
                                       symbol_table_json: str, code_snippet: str) -> str:
    return PROMPTS["object_discovery_user"].format(
        fqn=fqn, file_path=file_path, filter_chainable=str(filter_chainable).lower(),
        symbol_table_json=symbol_table_json, code_snippet=code_snippet
    )

def build_user_prompt_object_typing(variable_name: str, fqn: str, index_slice_json: str,
                                    spans_json: str, code_snippet: str) -> str:
    return PROMPTS["object_typing_user"].format(
        variable_name=variable_name, fqn=fqn, index_slice_json=index_slice_json,
        spans_json=spans_json, code_snippet=code_snippet
    )

def build_user_prompt_relationship_validator(parent_json: str, candidates_json: str,
                                             index_slice_json: str, code_snippet: str) -> str:
    return PROMPTS["relationship_validator_user"].format(
        parent_json=parent_json, candidates_json=candidates_json,
        index_slice_json=index_slice_json, code_snippet=code_snippet
    )


# ──────────────────────────────────────────────────────────────────────────────
# (Optional) tiny self-test — remove or keep for sanity checks
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample = r'''
    package com.example;
    import java.util.List;
    import java.util.Optional;
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
            return predicates.stream().anyMatch(p -> p.isapp(ispord, ptemp));
        }

        public static class ApplicablePredicate {
            public boolean isapp(String p1, String p2) { return true; }
        }
    }
    class ProcessingService { public String processRequest(String user) { return "x"; } }
    class DefaultCalc { public DefaultCalc(){} public X isOf(String s){ return new X(); } }
    class X { public boolean isValid(ProcessingService s){ return true; } }
    class Reporter {}
    class TransEvent { public TransEvent(Reporter r){} public Y isdaprodval(){return new Y();} public Y gettemp(){return new Y();} }
    class Y { public String getornull(){return ""; } }
    '''
    file = "DummyTest.java"

    # Focus: start method body span (approx; adjust if needed)
    start_span = Span(file=file, start_line=6, start_col=8, end_line=23, end_col=9)
    fm = FocusMethod(site_span=start_span, symbol="com.example.DummyTest.start(Reporter)")

    # 1) Calls for FOCUS=method (should find only isapp)
    calls_start = scan_direct_calls(sample, fm, file)
    print("scan_direct_calls (FocusMethod):", calls_start.model_dump())

    # 2) Vars for start (chainable locals → should include defaultCalc, exclude others)
    objs = list_scope_symbols(sample, start_span, file, filter_chainable=True)
    print("list_scope_symbols (chainable):", objs.model_dump())

    # 3) Focus object defaultCalc → one-hop call isOf
    # Find a plausible decl_span for defaultCalc quick-and-dirty (for demo)
    # In production, keep spans from AST when listing locals.
    defaultcalc_decl = Span(file=file, start_line=12, start_col=12, end_line=12, end_col=39)
    fo = FocusObject(name="defaultCalc", decl_span=defaultcalc_decl)
    calls_dc = scan_direct_calls(sample, fo, file)
    print("scan_direct_calls (FocusObject=defaultCalc):", calls_dc.model_dump())

    # 4) Focus call_result of isOf (span approx for demo)
    isof_span = Span(file=file, start_line=13, start_col=23, end_line=13, end_col=27)  # name token "isOf"
    fcr = FocusCallResult(from_call_name="isOf", site_span=isof_span)
    calls_isof = scan_direct_calls(sample, fcr, file)
    print("scan_direct_calls (FocusCallResult=isOf):", calls_isof.model_dump())

    # 5) Type hints (service)
    print("resolve_type_hints(service):", resolve_type_hints(sample, "service", start_span, file).model_dump())

    # 6) Relationship validation examples
    parent_m = {"kind": "method", "site_span": start_span.model_dump()}
    child_isapp = {"name": "isapp", "kind": "call"}
    print("validate (start→isapp):", validate_relationship(sample, parent_m, child_isapp, file).model_dump())

    parent_o = {"kind": "object", "name": "defaultCalc", "decl_span": defaultcalc_decl.model_dump()}
    child_isof = {"name": "isOf", "kind": "call"}
    print("validate (defaultCalc→isOf):", validate_relationship(sample, parent_o, child_isof, file).model_dump())
