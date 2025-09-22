# analysis_stack_min.py
"""
End-to-end prompt pack + tools for your Java analyzer (no spans; names only).

What this gives you
-------------------
• Pydantic schemas with names-only outputs.
• Focus-aware prompts for four LLM calls:
    1) method_call_finder      → list[str]    (names of calls based on focus)
    2) object_discovery        → {parameters, locals} (names only)
    3) object_typing           → {variable, type, confidence, reason?}
    4) relationship_validator  → verdicts for parent→child names
• Deterministic tools using tree-sitter-java that do NOT require spans.
• LangChain AzureChatOpenAI wiring with structured outputs.

Focus policy (your requirement)
-------------------------------
- FOCUS = method:<name>           → method_call_finder returns direct intra-method calls in SAME class
- FOCUS = object:<var_name>       → method_call_finder returns one-hop calls on that variable (e.g., defaultCalc → isOf)
- FOCUS = call_result:<call_name> → method_call_finder returns the immediate next hop after that call (e.g., isOf → isValid)

Install
-------
pip install pydantic tree_sitter tree_sitter_java langchain langchain-openai

Set Azure env (example)
-----------------------
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_ENDPOINT="https://<your-endpoint>.openai.azure.com/"
export AZURE_OPENAI_API_VERSION="2024-02-15-preview"  # or your deployed version
# You'll also need your deployment names below.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field
from tree_sitter import Language, Parser, Node
import tree_sitter_java as tsjava

# LangChain (Azure OpenAI)
from langchain_openai import AzureChatOpenAI
from langchain.schema import SystemMessage, HumanMessage


# ──────────────────────────────────────────────────────────────────────────────
# Azure LLM clients (configure once)
# ──────────────────────────────────────────────────────────────────────────────

class LLMClients:
    """
    Wraps AzureChatOpenAI models with convenient structured-output helpers.
    Configure deployment names as per your Azure resource.
    """

    def __init__(
        self,
        fast_deployment: str,     # e.g., "o3-mini"
        strong_deployment: str,   # e.g., "o1-preview"
        api_version: Optional[str] = None,  # defaults from env if None
    ):
        self.llm_fast = AzureChatOpenAI(
            azure_deployment=fast_deployment,
            api_version=api_version,
            temperature=0,
        )
        self.llm_strong = AzureChatOpenAI(
            azure_deployment=strong_deployment,
            api_version=api_version,
            temperature=0,
        )

    # Generic helper for structured outputs
    def run_structured(self, llm: AzureChatOpenAI, pydantic_model: Any, system: str, user: str):
        # LC supports pydantic structured output via with_structured_output
        return llm.with_structured_output(pydantic_model).invoke(
            [SystemMessage(content=system), HumanMessage(content=user)]
        )

    # Convenience wrappers
    def method_call_finder(self, system: str, user: str) -> "MethodCallsOut":
        return self.run_structured(self.llm_fast, MethodCallsOut, system, user)

    def object_discovery(self, system: str, user: str) -> "ObjectDiscoveryOut":
        return self.run_structured(self.llm_fast, ObjectDiscoveryOut, system, user)

    def object_typing(self, system: str, user: str) -> "TypeOut":
        # try fast model first, escalate is your choice outside
        return self.run_structured(self.llm_fast, TypeOut, system, user)

    def relationship_validator(self, system: str, user: str) -> "RelationshipValidationOut":
        return self.run_structured(self.llm_strong, RelationshipValidationOut, system, user)


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic schemas (names-only, span-free)
# ──────────────────────────────────────────────────────────────────────────────

class MethodCallsOut(BaseModel):
    calls: List[str] = Field(default_factory=list)
    notes: Optional[str] = None

class ObjectDiscoveryOut(BaseModel):
    parameters: List[str] = Field(default_factory=list)
    locals: List[str] = Field(default_factory=list)
    excluded: List[str] = Field(default_factory=list)

class TypeOut(BaseModel):
    variable: str
    type: str                 # FQN or simple; "UNKNOWN" if not resolvable
    confidence: float
    reason: Optional[str] = None

class Verdict(BaseModel):
    child_name: str
    kind: Literal["call","parameter","local"]
    valid: bool
    confidence: float
    reason: Optional[str] = None

class RelationshipValidationOut(BaseModel):
    verdicts: List[Verdict] = Field(default_factory=list)
    summary: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────────
# Prompt templates (NO spans; code + focus only; empties allowed)
# ──────────────────────────────────────────────────────────────────────────────

METHOD_CALL_FINDER_SYSTEM = (
    "You extract method calls RELATIVE to a FOCUS.\n"
    "- If FOCUS.kind == 'method': return only direct calls issued by that method body in the same class "
    "(e.g., 'isapp'). Exclude calls on other objects (like 'service.processRequest') and chained calls.\n"
    "- If FOCUS.kind == 'object': return only one-hop calls where the receiver is exactly that variable name "
    "(e.g., 'defaultCalc.isOf' → return 'isOf'). Do not include the next chained call.\n"
    "- If FOCUS.kind == 'call_result': return the immediate next method called on the result of the named call "
    "at any site in the code (e.g., after '.isOf(...)' return 'isValid' if directly chained).\n"
    "If nothing qualifies, return an empty list. Return STRICT JSON per schema. No prose."
)

METHOD_CALL_FINDER_USER_TMPL = (
    "FOCUS:\n"
    "{focus_json}\n\n"
    "CODE:\n"
    "{code}\n\n"
    "RULES:\n"
    "- Return only names of methods per the focus behavior above.\n"
    "- Returning [] is valid.\n\n"
    "OUTPUT SCHEMA (JSON): {{\"calls\": [\"string\", ...], \"notes\": \"string|null\"}}\n"
    "Return ONLY a JSON object."
)

OBJECT_DISCOVERY_SYSTEM = (
    "You extract method arguments and method-scope locals for the given focus METHOD NAME.\n"
    "Return names only. Exclude class fields and literals. If FILTER_CHAINABLE=true, include only locals that are receivers "
    "of a call whose result is immediately chained in that method (e.g., 'defaultCalc' in 'defaultCalc.isOf(...).isValid(...)').\n"
    "Empty lists are valid. Strict JSON only."
)

OBJECT_DISCOVERY_USER_TMPL = (
    "FOCUS_METHOD: {method_name}\n"
    "FILTER_CHAINABLE: {filter_chainable}\n\n"
    "CODE:\n{code}\n\n"
    "OUTPUT SCHEMA (JSON): {{\"parameters\": [\"string\",...], \"locals\": [\"string\",...], \"excluded\": [\"string\",...]}}\n"
    "Return ONLY a JSON object."
)

OBJECT_TYPING_SYSTEM = (
    "Determine the Java type of the given variable name using only the provided code. Prefer explicit declarations; "
    "otherwise infer from constructors (e.g., 'new Foo()' → 'Foo'); otherwise return 'UNKNOWN'. Provide a confidence in [0,1] "
    "and a short reason. Strict JSON only."
)

OBJECT_TYPING_USER_TMPL = (
    "VARIABLE: {var_name}\n\n"
    "CODE:\n{code}\n\n"
    "OUTPUT SCHEMA (JSON): {{\"variable\":\"string\",\"type\":\"string\",\"confidence\":0.0,\"reason\":\"string|null\"}}\n"
    "Return ONLY a JSON object."
)

REL_VALIDATOR_SYSTEM = (
    "Validate parent→child relationships for the analysis tree, using ONLY the code provided. Rules:\n"
    "- parent.kind='method': child.kind='call' must be a direct unqualified call in that method; child.kind in {parameter,local} must be declared there.\n"
    "- parent.kind='object': child.kind='call' must be a one-hop call on that variable as receiver in its enclosing method.\n"
    "- parent.kind='call_result': child.kind='call' must be the immediate next chained call after the named call.\n"
    "Reject literals/fields/deeper chains. Strict JSON only; allow empty verdicts."
)

REL_VALIDATOR_USER_TMPL = (
    "PARENT: {parent_json}\n"
    "CANDIDATES: {candidates_json}\n\n"
    "CODE:\n{code}\n\n"
    "OUTPUT SCHEMA (JSON): {{\"verdicts\": [{{\"child_name\":\"string\",\"kind\":\"call|parameter|local\",\"valid\":true|false,\"confidence\":0.0,\"reason\":\"string|null\"}}], \"summary\":\"string|null\"}}\n"
    "Return ONLY a JSON object."
)


# ──────────────────────────────────────────────────────────────────────────────
# Focus descriptors (names only)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FocusMethod:
    kind: Literal["method"] = "method"
    name: str = ""

@dataclass
class FocusObject:
    kind: Literal["object"] = "object"
    name: str = ""

@dataclass
class FocusCallResult:
    kind: Literal["call_result"] = "call_result"
    name: str = ""  # the method name whose result is the receiver of the next hop


# ──────────────────────────────────────────────────────────────────────────────
# Deterministic tools (no spans)
# ──────────────────────────────────────────────────────────────────────────────

JAVA = Language(tsjava.language())
_PARSER = Parser(JAVA)

def _parse(code: str):
    return _PARSER.parse(code.encode("utf-8")), code.encode("utf-8")

def _txt(bs: bytes, n: Node) -> str:
    return bs[n.start_byte:n.end_byte].decode("utf-8", errors="replace")

def _iter(root: Node, t: str):
    stack=[root]
    while stack:
        n=stack.pop()
        if n.type==t: yield n
        for i in range(n.child_count-1, -1, -1):
            stack.append(n.child(i))

def _class_bodies(root: Node) -> List[Node]:
    return list(_iter(root, "class_body"))

def _methods(root: Node) -> List[Node]:
    return list(_iter(root, "method_declaration"))

def _method_name(bs: bytes, m: Node) -> Optional[str]:
    n = m.child_by_field_name("name"); return _txt(bs,n) if n else None

def _method_body(m: Node) -> Optional[Node]:
    b = m.child_by_field_name("body")
    return b if b is not None else None

def _formal_params(bs: bytes, m: Node) -> List[str]:
    out=[]
    p = m.child_by_field_name("parameters")
    if not p: return out
    for ch in p.children:
        if ch.type in ("formal_parameter","receiver_parameter","spread_parameter"):
            nm = ch.child_by_field_name("name")
            if nm: out.append(_txt(bs,nm))
    return out

def _locals_in_body(bs: bytes, body: Node) -> List[str]:
    out=[]
    stack=[body]
    while stack:
        n=stack.pop()
        if n is not body and n.type in ("method_declaration","class_body","interface_body","enum_body","record_body"):
            continue
        # skip locals declared only inside lambdas (we don't include lambda-scoped vars)
        cur=n.parent
        skip=False
        while cur is not None:
            if cur.type=="lambda_expression":
                skip=True; break
            cur=cur.parent
        if skip: 
            continue
        if n.type=="local_variable_declaration":
            for c in n.children:
                if c.type=="variable_declarator":
                    nm = c.child_by_field_name("name")
                    if nm: out.append(_txt(bs,nm))
        for i in range(n.child_count-1,-1,-1):
            stack.append(n.child(i))
    return out

def _calls_in_body(bs: bytes, body: Node) -> List[Tuple[str, Optional[str], Node]]:
    """Return list of (callee_name, receiver_text, node)."""
    out=[]
    for mi in _iter(body,"method_invocation"):
        name = mi.child_by_field_name("name")
        obj  = mi.child_by_field_name("object")
        out.append((_txt(bs,name) if name else "", _txt(bs,obj).strip() if obj else None, mi))
    return out

def _chainable_receivers(bs: bytes, body: Node) -> List[str]:
    """Identifiers that start a chain: X.y(...).z(...)."""
    out=set()
    for mi in _iter(body,"method_invocation"):
        parent=mi.parent
        if parent and parent.type=="method_invocation" and parent.child_by_field_name("object")==mi:
            obj = mi.child_by_field_name("object")
            if obj and obj.type=="identifier":
                out.add(_txt(bs,obj))
    return list(sorted(out))

def tool_method_call_finder_names(code: str, focus: FocusMethod|FocusObject|FocusCallResult) -> List[str]:
    """Deterministic name finder per your focus policy (no spans)."""
    tree, bs = _parse(code)
    root = tree.root_node

    # FOCUS: method
    if isinstance(focus, FocusMethod):
        for m in _methods(root):
            if _method_name(bs,m) == focus.name:
                b=_method_body(m)
                if not b: return []
                names=[]
                for callee, recv, _ in _calls_in_body(bs,b):
                    if recv is None or recv in ("this","super"):
                        names.append(callee)
                return names
        return []

    # FOCUS: object
    if isinstance(focus, FocusObject):
        # find method that declares this local
        for m in _methods(root):
            b=_method_body(m)
            if not b: continue
            locals_ = _locals_in_body(bs,b)
            if focus.name in locals_:
                names=[]
                for callee, recv, mi in _calls_in_body(bs,b):
                    if recv == focus.name:
                        # include even if this mi is intermediate in a chain
                        names.append(callee)
                # dedupe keeping order
                seen=set(); out=[]
                for n in names:
                    if n not in seen:
                        seen.add(n); out.append(n)
                return out
        return []

    # FOCUS: call_result
    if isinstance(focus, FocusCallResult):
        next_calls=set()
        # Find patterns: X.isOf(...).<NEXT>(...)
        for m in _methods(root):
            b=_method_body(m)
            if not b: continue
            for mi in _iter(b,"method_invocation"):
                name_node = mi.child_by_field_name("name")
                if not name_node: continue
                name = _txt(bs,name_node)
                if name != focus.name:
                    continue
                # check parent
                parent = mi.parent
                if parent and parent.type=="method_invocation" and parent.child_by_field_name("object")==mi:
                    next_name_node = parent.child_by_field_name("name")
                    if next_name_node:
                        next_calls.add(_txt(bs,next_name_node))
        return list(sorted(next_calls))

    return []


def tool_object_discovery_names(code: str, method_name: str, chainable_only: bool = True) -> Dict[str,List[str]]:
    """Return {parameters:[...], locals:[...]}, optionally filtering locals to chain starters."""
    tree, bs = _parse(code)
    root = tree.root_node
    for m in _methods(root):
        if _method_name(bs,m) == method_name:
            params = _formal_params(bs,m)
            b=_method_body(m)
            if not b: 
                return {"parameters": params, "locals": []}
            locals_ = _locals_in_body(bs,b)
            if chainable_only:
                starters = set(_chainable_receivers(bs,b))
                locals_ = [x for x in locals_ if x in starters]
            return {"parameters": params, "locals": locals_}
    return {"parameters": [], "locals": []}


def tool_type_hint(code: str, variable: str) -> TypeOut:
    """Try to resolve a variable type (param/local) by scanning the whole file (no spans)."""
    tree, bs = _parse(code)
    root = tree.root_node

    # Check params
    for m in _methods(root):
        # params
        p = m.child_by_field_name("parameters")
        if p:
            for child in p.children:
                if child.type in ("formal_parameter","receiver_parameter","spread_parameter"):
                    nm = child.child_by_field_name("name")
                    if nm and _txt(bs,nm)==variable:
                        t = child.child_by_field_name("type")
                        if t:
                            return TypeOut(variable=variable, type=_txt(bs,t).strip(), confidence=0.99)
                        return TypeOut(variable=variable, type="UNKNOWN", confidence=0.4, reason="Parameter has no explicit type node.")
        # locals
        b=_method_body(m)
        if not b: 
            continue
        stack=[b]
        while stack:
            n=stack.pop()
            if n.type=="local_variable_declaration":
                tnode = n.child_by_field_name("type")
                for c in n.children:
                    if c.type=="variable_declarator":
                        nm = c.child_by_field_name("name")
                        if nm and _txt(bs,nm)==variable:
                            if tnode is not None:
                                declared = _txt(bs,tnode).strip()
                                if declared=="var":
                                    init = c.child_by_field_name("value")
                                    if init and init.type=="object_creation_expression":
                                        t1 = init.child_by_field_name("type")
                                        if t1:
                                            return TypeOut(variable=variable, type=_txt(bs,t1).strip(), confidence=0.9, reason="var inferred from constructor")
                                    return TypeOut(variable=variable, type="UNKNOWN", confidence=0.5, reason="var initializer not a constructor")
                                return TypeOut(variable=variable, type=declared, confidence=0.95, reason="explicit local type")
                            else:
                                init = c.child_by_field_name("value")
                                if init and init.type=="object_creation_expression":
                                    t1 = init.child_by_field_name("type")
                                    if t1:
                                        return TypeOut(variable=variable, type=_txt(bs,t1).strip(), confidence=0.85, reason="constructor initializer")
                                return TypeOut(variable=variable, type="UNKNOWN", confidence=0.4, reason="no explicit type; non-ctor initializer")
            for i in range(n.child_count-1,-1,-1):
                stack.append(n.child(i))
    return TypeOut(variable=variable, type="UNKNOWN", confidence=0.2, reason="variable not found")


def tool_relationship_validate(code: str, parent: Dict[str,str], child: Dict[str,str]) -> Verdict:
    """
    parent: {"kind":"method|object|call_result", "name":"..."}
    child:  {"kind":"call|parameter|local", "name":"..."}
    Returns a single Verdict (names only).
    """
    pk, pn = parent.get("kind"), parent.get("name")
    ck, cn = child.get("kind"), child.get("name")

    # method → call / parameter / local
    if pk == "method":
        # calls
        if ck == "call":
            calls = tool_method_call_finder_names(code, FocusMethod(name=pn))
            valid = cn in calls
            return Verdict(child_name=cn, kind="call", valid=valid, confidence=0.96 if valid else 0.9,
                           reason="direct unqualified call" if valid else "not a direct unqualified call")
        # parameter/local
        od = tool_object_discovery_names(code, pn, chainable_only=False)
        if ck == "parameter":
            valid = cn in od["parameters"]
            return Verdict(child_name=cn, kind="parameter", valid=valid, confidence=0.95 if valid else 0.9,
                           reason="declared parameter" if valid else "not declared parameter of method")
        if ck == "local":
            # locals regardless of chainability
            valid = cn in od["locals"]
            return Verdict(child_name=cn, kind="local", valid=valid, confidence=0.95 if valid else 0.9,
                           reason="method-scope local" if valid else "not a method-scope local")
        return Verdict(child_name=cn, kind=ck, valid=False, confidence=0.9, reason="unsupported child kind for method parent")

    # object → call
    if pk == "object" and ck == "call":
        calls = tool_method_call_finder_names(code, FocusObject(name=pn))
        valid = cn in calls
        return Verdict(child_name=cn, kind="call", valid=valid, confidence=0.95 if valid else 0.9,
                       reason="one-hop call on object" if valid else "no one-hop call on object")

    # call_result → call
    if pk == "call_result" and ck == "call":
        calls = tool_method_call_finder_names(code, FocusCallResult(name=pn))
        valid = cn in calls
        return Verdict(child_name=cn, kind="call", valid=valid, confidence=0.93 if valid else 0.9,
                       reason="immediate chained call" if valid else "no immediate chained call after parent")

    return Verdict(child_name=cn, kind=ck, valid=False, confidence=0.9, reason="unsupported parent kind")


# ──────────────────────────────────────────────────────────────────────────────
# LLM runners (build prompts from code + focus ONLY)
# ──────────────────────────────────────────────────────────────────────────────

def build_user_method_call_finder(focus_json: str, code: str) -> str:
    return METHOD_CALL_FINDER_USER_TMPL.format(focus_json=focus_json, code=code)

def build_user_object_discovery(method_name: str, code: str, filter_chainable: bool = True) -> str:
    return OBJECT_DISCOVERY_USER_TMPL.format(method_name=method_name, code=code, filter_chainable=str(filter_chainable).lower())

def build_user_object_typing(var_name: str, code: str) -> str:
    return OBJECT_TYPING_USER_TMPL.format(var_name=var_name, code=code)

def build_user_rel_validator(parent_json: str, candidates_json: str, code: str) -> str:
    return REL_VALIDATOR_USER_TMPL.format(parent_json=parent_json, candidates_json=candidates_json, code=code)


# ──────────────────────────────────────────────────────────────────────────────
# Example usage (run this file directly to sanity check)
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

    # Deterministic tools demo (no LLM calls needed for these)
    print("TOOL method(start) →", tool_method_call_finder_names(sample, FocusMethod(name="start")))
    print("TOOL object(defaultCalc) →", tool_method_call_finder_names(sample, FocusObject(name="defaultCalc")))
    print("TOOL call_result(isOf) →", tool_method_call_finder_names(sample, FocusCallResult(name="isOf")))
    print("TOOL object_discovery(start, chainable_only=True) →", tool_object_discovery_names(sample, "start", chainable_only=True))
    print("TOOL type_hint('defaultCalc') →", tool_type_hint(sample, "defaultCalc").model_dump())

    # If you want to actually call Azure LLMs, configure deployments and uncomment below:
    """
    clients = LLMClients(
        fast_deployment="o3-mini-deploy-name",     # <-- your Azure deployment name
        strong_deployment="o1-preview-deploy-name",# <-- your Azure deployment name
        api_version=None,                          # or "2024-02-15-preview"
    )

    # 1) Method-call finder (start → ["isapp"])
    focus = FocusMethod(name="start")
    user = build_user_method_call_finder(focus_json='{"kind":"method","name":"start"}', code=sample)
    out1 = clients.method_call_finder(METHOD_CALL_FINDER_SYSTEM, user)
    print("LLM method_call_finder(start) →", out1.model_dump())

    # 2) Object-call finder (defaultCalc → ["isOf"])
    focus_o = FocusObject(name="defaultCalc")
    user = build_user_method_call_finder(focus_json='{"kind":"object","name":"defaultCalc"}', code=sample)
    out2 = clients.method_call_finder(METHOD_CALL_FINDER_SYSTEM, user)
    print("LLM method_call_finder(defaultCalc) →", out2.model_dump())

    # 3) Call-result (isOf → ["isValid"])
    focus_cr = FocusCallResult(name="isOf")
    user = build_user_method_call_finder(focus_json='{"kind":"call_result","name":"isOf"}', code=sample)
    out3 = clients.method_call_finder(METHOD_CALL_FINDER_SYSTEM, user)
    print("LLM method_call_finder(call_result=isOf) →", out3.model_dump())

    # 4) Object discovery (start, chainable_only=True → locals likely ["defaultCalc"])
    user = build_user_object_discovery(method_name="start", code=sample, filter_chainable=True)
    out4 = clients.object_discovery(OBJECT_DISCOVERY_SYSTEM, user)
    print("LLM object_discovery(start) →", out4.model_dump())

    # 5) Typing (variable → type)
    user = build_user_object_typing(var_name="defaultCalc", code=sample)
    out5 = clients.object_typing(OBJECT_TYPING_SYSTEM, user)
    print("LLM object_typing(defaultCalc) →", out5.model_dump())

    # 6) Relationship validator
    parent_json = '{"kind":"method","name":"start"}'
    candidates_json = '[{"child_name":"isapp","kind":"call"},{"child_name":"defaultCalc","kind":"local"}]'
    user = build_user_rel_validator(parent_json=parent_json, candidates_json=candidates_json, code=sample)
    out6 = clients.relationship_validator(REL_VALIDATOR_SYSTEM, user)
    print("LLM relationship_validator →", out6.model_dump())
    """
