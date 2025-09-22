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


#
