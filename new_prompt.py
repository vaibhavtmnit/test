# prompt_builders.py
# Tiny f-string prompt builders (names-only; no spans; no tools).

from typing import Dict

# 1) Method-call finder (focus-aware)
def method_call_finder_prompts(*, focus_json: str, code: str) -> Dict[str, str]:
    system = (
        "You extract method calls RELATIVE to a FOCUS. Return STRICT JSON only.\n"
        "Rules:\n"
        "• If FOCUS.kind=='method': return only direct calls issued by that method body that are unqualified "
        "  or qualified by 'this'/'super'. Exclude calls on other objects (e.g., 'svc.run()'), chained calls "
        "  (e.g., 'x.a().b()'), and calls inside lambdas/anonymous classes. Ignore JDK/library calls unless "
        "  they are unqualified and declared in the same class.\n"
        "• If FOCUS.kind=='object': return one-hop calls where the receiver is exactly that variable name "
        "  (e.g., 'calc.isOf' → return 'isOf' only). Do NOT include further chained calls.\n"
        "• If FOCUS.kind=='call_result': return the immediate next chained call after the named call "
        "  (e.g., for 'x.a().b()', FOCUS='a' → return 'b').\n"
        "• It is VALID to return []. Never guess.\n"
        'Output schema: {"calls": ["string", ...], "notes": "string|null"}'
    )
    user = (
        f"FOCUS (JSON): {focus_json}\n\n"
        f"CODE:\n{code}\n\n"
        "Few-shot examples:\n"
        "1) FOCUS.method\n"
        "class C { void m(){ helper(); svc.run(); y.a().b(); } void helper(){} }\n"
        "→ calls=['helper']   # exclude svc.run (object call), exclude chained y.a().b()\n"
        "2) FOCUS.object\n"
        "void m(){ X x=new X(); x.a().b(); x.c(); Y y=new Y(); y.d(); }\n"
        'FOCUS={"kind":"object","name":"x"} → calls=[\'a\',\'c\'] (not \'b\')\n'
        "3) FOCUS.call_result\n"
        "void m(){ X x=new X(); x.a().b(); x.a().e(); }\n"
        'FOCUS={"kind":"call_result","name":"a"} → calls=[\'b\',\'e\']\n\n'
        "Return ONLY a JSON object per schema."
    )
    return {"system": system, "user": user}


# 2) Object/variable discovery (parameters + method-scope locals; optional chainable-only)
def object_discovery_prompts(*, method_name: str, code: str, chainable_only: bool = True) -> Dict[str, str]:
    system = (
        "You list method parameters and method-scope locals (names only) for the given METHOD NAME. Strict JSON only.\n"
        "Rules:\n"
        "• Include ALL parameters.\n"
        "• Include locals declared in the method body (exclude class fields and literals). "
        "  Exclude variables introduced only inside lambda bodies/anonymous classes.\n"
        "• If FILTER_CHAINABLE==true, keep only locals that start a chain within the method "
        "  (e.g., for 'foo.a().b()', 'foo' is chainable). Otherwise include all method-scope locals.\n"
        "• Empty lists are valid.\n"
        'Output schema: {"parameters":[...], "locals":[...], "excluded":[...]}'
    )
    user = (
        f"FOCUS_METHOD: {method_name}\n"
        f"FILTER_CHAINABLE: {str(chainable_only).lower()}\n\n"
        f"CODE:\n{code}\n\n"
        "Few-shot examples:\n"
        "1) Params + locals\n"
        "class C { void m(A a){ B b=new B(); int k=0; svc.run(); } }\n"
        "→ parameters=['a']; locals=['b','k'] (unless chainable_only=true ⇒ likely [] if no chains)\n"
        "2) Chainable-only\n"
        "void m(){ S s=new S(); s.a().b(); T t=new T(); t.c(); u.d(); }\n"
        "FILTER_CHAINABLE=true → locals=['s','t']  # 'u' not declared; ignore\n\n"
        "Return ONLY a JSON object per schema."
    )
    return {"system": system, "user": user}


# 3) Object typing (best-effort from code; return UNKNOWN if unclear)
def object_typing_prompts(*, var_name: str, code: str) -> Dict[str, str]:
    system = (
        "Determine the Java type of a single variable using ONLY the provided code. Strict JSON only. Rules:\n"
        "• If it's a parameter, use its declared type.\n"
        "• If it's a local with an explicit type (not 'var'), use that type.\n"
        "• If it's 'var' initialized with 'new Type(...)', infer 'Type'.\n"
        "• If assigned from a method call or otherwise unclear, return 'UNKNOWN' with a short reason.\n"
        "• Confidence in [0,1]. Do not guess.\n"
        'Output schema: {"variable":"...","type":"...","confidence":0.0,"reason":"...|null"}'
    )
    user = (
        f"VARIABLE: {var_name}\n\n"
        f"CODE:\n{code}\n\n"
        "Few-shot examples:\n"
        "1) int k=0; → ('k','int',~0.95)\n"
        "2) var sb=new StringBuilder(); → ('sb','StringBuilder',~0.90)\n"
        "3) String x=svc.run(); → ('x','UNKNOWN',reason='assigned from method call')\n\n"
        "Return ONLY a JSON object per schema."
    )
    return {"system": system, "user": user}


# 4) Relationship validator (parent→child, names only)
def relationship_validator_prompts(*, parent_json: str, candidates_json: str, code: str) -> Dict[str, str]:
    system = (
        "Validate parent→child relationships for the analysis tree using ONLY the provided code. Strict JSON only. Rules:\n"
        "• parent.kind='method': child.kind='call' must be a direct unqualified/this/super call in that method body; "
        "  child.kind in {parameter,local} must be declared in that method.\n"
        "• parent.kind='object': child.kind='call' must be a one-hop call whose receiver == object name (in its enclosing method).\n"
        "• parent.kind='call_result': child.kind='call' must be the immediate next chained call after the named call.\n"
        "• Reject literals, fields, nested-only vars, deeper chains. Empty verdicts allowed.\n"
        'Output schema: {"verdicts":[{"child_name":"string","kind":"call|parameter|local","valid":true|false,"confidence":0.0,"reason":"string|null"}], "summary":"string|null"}'
    )
    user = (
        f"PARENT (JSON): {parent_json}\n"
        f"CANDIDATES (JSON): {candidates_json}\n\n"
        f"CODE:\n{code}\n\n"
        "Few-shot examples:\n"
        "1) parent.method=m, candidate=('helper','call'): valid if 'helper()' is unqualified in m.\n"
        "2) parent.object=x, candidate=('a','call'): valid if code has 'x.a(...)' (one hop), not if only 'y.a(...)'.\n"
        "3) parent.call_result='a', candidate=('b','call'): valid if 'x.a(...).b(...)' appears.\n"
        "4) parent.method=m, candidate=('field','local'): invalid if 'field' is a class field, not declared in m.\n\n"
        "Return ONLY a JSON object per schema."
    )
    return {"system": system, "user": user}



from langchain_openai import AzureChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from pydantic import BaseModel
from prompt_builders import (
    method_call_finder_prompts,
    object_discovery_prompts,
    object_typing_prompts,
    relationship_validator_prompts,
)

# define your output schemas once
class MethodCallsOut(BaseModel):
    calls: list[str]
    notes: str | None = None

class ObjectDiscoveryOut(BaseModel):
    parameters: list[str]
    locals: list[str]
    excluded: list[str]

class TypeOut(BaseModel):
    variable: str
    type: str          # "UNKNOWN" allowed
    confidence: float
    reason: str | None = None

class Verdict(BaseModel):
    child_name: str
    kind: str
    valid: bool
    confidence: float
    reason: str | None = None

class RelationshipValidationOut(BaseModel):
    verdicts: list[Verdict]
    summary: str | None = None

# azure deployments
fast = AzureChatOpenAI(azure_deployment="o3-mini", temperature=0)
strong = AzureChatOpenAI(azure_deployment="o1-preview", temperature=0)

# example: method-call finder
prompts = method_call_finder_prompts(
    focus_json='{"kind":"method","name":"start"}',
    code="class C { void start(){ helper(); svc.run(); } void helper(){} }",
)
resp = fast.with_structured_output(MethodCallsOut).invoke(
    [SystemMessage(content=prompts["system"]), HumanMessage(content=prompts["user"])]
)
print(resp)

# example: relationship validator
prompts = relationship_validator_prompts(
    parent_json='{"kind":"object","name":"calc"}',
    candidates_json='[{"child_name":"isOf","kind":"call"}]',
    code="class C { void m(){ Calc calc=new Calc(); calc.isOf(x).isValid(); } }",
)
resp = strong.with_structured_output(RelationshipValidationOut).invoke(
    [SystemMessage(content=prompts["system"]), HumanMessage(content=prompts["user"])]
)
print(resp)
