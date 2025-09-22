# llm_prompts_no_tools.py
"""
LLM-only prompt pack for Java analysis (no tools, no spans). Names-only I/O.

What you get
------------
• Pydantic schemas (names only) for all four calls.
• Focus-aware prompts that capture your rules:
    - FOCUS.method: only direct, unqualified calls (this/super) in that method body.
    - FOCUS.object: one-hop calls where receiver == that variable name.
    - FOCUS.call_result: the immediate next chained call after the named call.
• Object discovery: parameters + method-scope locals, with optional "chainable_only"
  (locals that start a chain like  foo.a().b()  ⇒ foo is chainable).
• Object typing: best-effort from code; return "UNKNOWN" if not resolvable.
• Relationship validator: sanity-check parent→child edges using only code + names.

Azure setup
-----------
pip install pydantic langchain langchain-openai
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_ENDPOINT="https://<your-endpoint>.openai.azure.com/"
export AZURE_OPENAI_API_VERSION="2024-02-15-preview"   # or your deployed version names below
"""

from __future__ import annotations
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain.schema import SystemMessage, HumanMessage


# ──────────────────────────────────────────────────────────────────────────────
# Names-only Pydantic schemas (match your pipeline)
# ──────────────────────────────────────────────────────────────────────────────

class MethodCallsOut(BaseModel):
    calls: List[str] = Field(default_factory=list)
    notes: Optional[str] = None

class ObjectDiscoveryOut(BaseModel):
    parameters: List[str] = Field(default_factory=list)
    locals: List[str] = Field(default_factory=list)
    excluded: List[str] = Field(default_factory=list)  # literals/fields/nested-only, etc.

class TypeOut(BaseModel):
    variable: str
    type: str                  # FQN or simple; "UNKNOWN" if not resolvable
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
# AzureChatOpenAI clients + tiny retry wrapper
# ──────────────────────────────────────────────────────────────────────────────

class LLMClients:
    def __init__(self, fast_deployment: str, strong_deployment: str, api_version: Optional[str] = None):
        self.fast = AzureChatOpenAI(azure_deployment=fast_deployment, api_version=api_version, temperature=0)
        self.strong = AzureChatOpenAI(azure_deployment=strong_deployment, api_version=api_version, temperature=0)

    @staticmethod
    def _invoke_with_retry(llm: AzureChatOpenAI, schema, system: str, user: str, max_retries: int = 1):
        for attempt in range(max_retries + 1):
            try:
                return llm.with_structured_output(schema).invoke([SystemMessage(content=system),
                                                                  HumanMessage(content=user)])
            except Exception as e:
                if attempt == max_retries:
                    raise
                # Nudge the model once to comply strictly
                user = user + "\n\nREMINDER: Return ONLY a JSON object matching the schema. If no items, return empty lists."

    # convenience methods
    def method_call_finder(self, system: str, user: str) -> MethodCallsOut:
        return self._invoke_with_retry(self.fast, MethodCallsOut, system, user)

    def object_discovery(self, system: str, user: str) -> ObjectDiscoveryOut:
        return self._invoke_with_retry(self.fast, ObjectDiscoveryOut, system, user)

    def object_typing(self, system: str, user: str) -> TypeOut:
        return self._invoke_with_retry(self.fast, TypeOut, system, user)

    def relationship_validator(self, system: str, user: str) -> RelationshipValidationOut:
        return self._invoke_with_retry(self.strong, RelationshipValidationOut, system, user)


# ──────────────────────────────────────────────────────────────────────────────
# PROMPTS — concise, general, and few-shot (no spans, code + names only)
# ──────────────────────────────────────────────────────────────────────────────

# 1) METHOD-CALL FINDER (focus-aware)
METHOD_CALL_FINDER_SYSTEM = (
    "You extract method calls RELATIVE to a FOCUS. Return STRICT JSON only.\n"
    "Rules:\n"
    "• If FOCUS.kind=='method': return only direct calls issued by that method body that are unqualified or qualified by 'this'/'super'. "
    "Exclude calls on other objects (e.g., 'svc.run()'), chained calls (e.g., 'x.a().b()'), and calls inside lambdas/anonymous classes. "
    "JDK/library calls should be ignored unless they are unqualified and declared in the same class.\n"
    "• If FOCUS.kind=='object': return one-hop calls where the receiver is exactly that variable name (e.g., 'calc.isOf'→'isOf'). "
    "Do NOT include further chained calls.\n"
    "• If FOCUS.kind=='call_result': return the immediate next chained call after the named call (e.g., 'x.a().b()' with FOCUS 'a'→ return 'b').\n"
    "• It is VALID to return []. Never guess.\n"
    "Output schema: {\"calls\": [\"string\", ...], \"notes\": \"string|null\"}"
)

METHOD_CALL_FINDER_USER_TMPL = (
    "FOCUS (JSON): {focus_json}\n\n"
    "CODE:\n{code}\n\n"
    "Few-shot examples:\n"
    "1) FOCUS.method\n"
    "class C { void m(){ helper(); svc.run(); y.a().b(); } void helper(){} }\n"
    "→ calls=['helper']   # exclude svc.run (object call), exclude chained y.a().b()\n"
    "2) FOCUS.object\n"
    "void m(){ X x=new X(); x.a().b(); x.c(); Y y=new Y(); y.d(); }\n"
    "FOCUS={\"kind\":\"object\",\"name\":\"x\"} → calls=['a','c'] (not 'b')\n"
    "3) FOCUS.call_result\n"
    "void m(){ X x=new X(); x.a().b(); x.a().e(); }\n"
    "FOCUS={\"kind\":\"call_result\",\"name\":\"a\"} → calls=['b','e']\n\n"
    "Return ONLY a JSON object per schema."
)

# 2) OBJECT / VARIABLE DISCOVERY (method parameters + locals; optional chainable_only)
OBJECT_DISCOVERY_SYSTEM = (
    "You list method parameters and method-scope locals (names only) for the given METHOD NAME. Strict JSON only.\n"
    "Rules:\n"
    "• Include ALL parameters.\n"
    "• Include locals declared in the method body (exclude class fields and literals). "
    "Exclude variables introduced only inside lambda bodies/anonymous classes.\n"
    "• If FILTER_CHAINABLE==true, keep only locals that start a chain within the method (e.g., 'foo.a().b()' ⇒ 'foo' is chainable). "
    "Otherwise include all method-scope locals.\n"
    "• Empty lists are valid.\n"
    "Output schema: {\"parameters\":[...], \"locals\":[...], \"excluded\":[...]}"
)

OBJECT_DISCOVERY_USER_TMPL = (
    "FOCUS_METHOD: {method_name}\n"
    "FILTER_CHAINABLE: {filter_chainable}\n\n"
    "CODE:\n{code}\n\n"
    "Few-shot examples:\n"
    "1) Params + locals\n"
    "class C { void m(A a){ B b=new B(); int k=0; svc.run(); } }\n"
    "→ parameters=['a']; locals=['b','k'] (unless chainable_only=true ⇒ likely [] if no chains)\n"
    "2) Chainable-only\n"
    "void m(){ S s=new S(); s.a().b(); T t=new T(); t.c(); u.d(); }\n"
    "FILTER_CHAINABLE=true → locals=['s','t']  # 'u' not declared; ignore\n\n"
    "Return ONLY a JSON object per schema."
)

# 3) OBJECT TYPING (best-effort from code)
OBJECT_TYPING_SYSTEM = (
    "Determine the Java type of a single variable using ONLY the provided code. Strict JSON only. Rules:\n"
    "• If it's a parameter, use its declared type.\n"
    "• If it's a local with an explicit type (not 'var'), use that type.\n"
    "• If it's 'var' initialized with 'new Type(...)', infer 'Type'.\n"
    "• If assigned from a method call or otherwise unclear, return 'UNKNOWN' with a short reason.\n"
    "• Confidence in [0,1]. Do not guess.\n"
    "Output schema: {\"variable\":\"...\",\"type\":\"...\",\"confidence\":0.0,\"reason\":\"...|null\"}"
)

OBJECT_TYPING_USER_TMPL = (
    "VARIABLE: {var_name}\n\n"
    "CODE:\n{code}\n\n"
    "Few-shot examples:\n"
    "1) int k=0; → ('k','int',~0.95)\n"
    "2) var sb=new StringBuilder(); → ('sb','StringBuilder',~0.90)\n"
    "3) String x=svc.run(); → ('x','UNKNOWN',reason='assigned from method call')\n\n"
    "Return ONLY a JSON object per schema."
)

# 4) RELATIONSHIP VALIDATOR (parent→child, names only)
REL_VALIDATOR_SYSTEM = (
    "Validate parent→child relationships for the analysis tree using ONLY the provided code. Strict JSON only. Rules:\n"
    "• parent.kind='method': child.kind='call' must be a direct unqualified/this/super call in that method body; "
    "child.kind in {parameter,local} must be declared in that method.\n"
    "• parent.kind='object': child.kind='call' must be a one-hop call whose receiver == object name (in its enclosing method).\n"
    "• parent.kind='call_result': child.kind='call' must be the immediate next chained call after the named call.\n"
    "• Reject literals, fields, nested-only vars, deeper chains. Empty verdicts allowed.\n"
    "Output schema: {\"verdicts\":[{...}], \"summary\":\"...|null\"}"
)

REL_VALIDATOR_USER_TMPL = (
    "PARENT (JSON): {parent_json}\n"
    "CANDIDATES (JSON): {candidates_json}\n\n"
    "CODE:\n{code}\n\n"
    "Few-shot examples:\n"
    "1) parent.method=m, candidate=('helper','call'): valid if 'helper()' is unqualified in m.\n"
    "2) parent.object=x, candidate=('a','call'): valid if code has 'x.a(...)' (one hop), not if only 'y.a(...)'.\n"
    "3) parent.call_result='a', candidate=('b','call'): valid if 'x.a(...).b(...)' appears.\n"
    "4) parent.method=m, candidate=('field','local'): invalid if 'field' is a class field, not declared in m.\n\n"
    "Return ONLY a JSON object per schema."
)


# ──────────────────────────────────────────────────────────────────────────────
# Convenience builders for user prompts
# ──────────────────────────────────────────────────────────────────────────────

def build_user_method_call_finder(focus_json: str, code: str) -> str:
    return METHOD_CALL_FINDER_USER_TMPL.format(focus_json=focus_json, code=code)

def build_user_object_discovery(method_name: str, code: str, filter_chainable: bool = True) -> str:
    return OBJECT_DISCOVERY_USER_TMPL.format(method_name=method_name, code=code,
                                             filter_chainable=str(filter_chainable).lower())

def build_user_object_typing(var_name: str, code: str) -> str:
    return OBJECT_TYPING_USER_TMPL.format(var_name=var_name, code=code)

def build_user_rel_validator(parent_json: str, candidates_json: str, code: str) -> str:
    return REL_VALIDATOR_USER_TMPL.format(parent_json=parent_json, candidates_json=candidates_json, code=code)


# ──────────────────────────────────────────────────────────────────────────────
# Minimal usage demo (commented; wire into your LangGraph nodes)
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    SAMPLE = """
    class C {
        void m(A a){
            System.out.println(\"log\");
            S s=new S(); T t=new T();
            String r = svc.run();
            s.a().b();
            t.c();
            if (ok(a)) { /* ... */ }
            java.util.List<L> list=null;
            boolean any = list.stream().anyMatch(e -> check(e));  // lambda call 'check' MUST be ignored here
        }
        boolean ok(A x){ return true; }
        boolean check(L l){ return true; }
    }
    """

    # Configure with your Azure deployment names
    # clients = LLMClients(fast_deployment="o3-mini-deploy", strong_deployment="o1-preview-deploy")

    # 1) Method-call finder — focus method "m" → expect ["ok"] only
    # user1 = build_user_method_call_finder('{"kind":"method","name":"m"}', SAMPLE)
    # out1 = clients.method_call_finder(METHOD_CALL_FINDER_SYSTEM, user1)
    # print(out1)

    # 2) Method-call finder — focus object "s" → expect ["a"]
    # user2 = build_user_method_call_finder('{"kind":"object","name":"s"}', SAMPLE)
    # out2 = clients.method_call_finder(METHOD_CALL_FINDER_SYSTEM, user2)
    # print(out2)

    # 3) Method-call finder — focus call_result "a" → expect ["b"]
    # user3 = build_user_method_call_finder('{"kind":"call_result","name":"a"}', SAMPLE)
    # out3 = clients.method_call_finder(METHOD_CALL_FINDER_SYSTEM, user3)
    # print(out3)

    # 4) Object discovery — method "m", chainable_only=true → expect locals like ["s","t"]; parameters ["a"]
    # user4 = build_user_object_discovery("m", SAMPLE, filter_chainable=True)
    # out4 = clients.object_discovery(OBJECT_DISCOVERY_SYSTEM, user4)
    # print(out4)

    # 5) Typing — e.g., var not shown; try 's' → UNKNOWN without full declaration types in this synthetic snippet
    # user5 = build_user_object_typing("s", SAMPLE)
    # out5 = clients.object_typing(OBJECT_TYPING_SYSTEM, user5)
    # print(out5)

    # 6) Relationship validator — parent=method m, candidates=[('ok','call'),('s','local')]
    # parent_json = '{"kind":"method","name":"m"}'
    # candidates_json = '[{"child_name":"ok","kind":"call"},{"child_name":"s","kind":"local"}]'
    # user6 = build_user_rel_validator(parent_json, candidates_json, SAMPLE)
    # out6 = clients.relationship_validator(REL_VALIDATOR_SYSTEM, user6)
    # print(out6)
