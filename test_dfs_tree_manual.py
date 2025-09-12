
import unittest, tempfile, textwrap, os, sys
sys.path.append("/mnt/data")

from dfs_tree import Tree, ClassDetail

class Node:
    def __init__(self, name): self.name = name
    def __repr__(self): return f"Node({self.name})"

def mk_agent_state(code: str, path: str, obj: str):
    return {"code": code, "path": path, "object_name": obj}

class BadItem:
    def __repr__(self): return "BadItem()"

class EmptyName:
    def __init__(self): self.name = ""

class TestManualBuildAndEdgeCases(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile("w+", suffix=".java", delete=False)
        self.tmp.write(textwrap.dedent("""
        package p;
        public class Root {
            void a(){}
            void b(){}
        }
        """).strip())
        self.tmp.flush()
        self.code_path = self.tmp.name

    def tearDown(self):
        try:
            os.unlink(self.code_path)
        except Exception:
            pass

    def test_manual_build(self):
        def supervisor(state): return [Node("child")]
        def expansion_inputs(node): return None
        def is_class(node): return None

        t = Tree(
            code_path=self.code_path,
            root_object="a",
            supervisor=supervisor,
            expansion_inputs=expansion_inputs,
            is_class=is_class,
            make_agent_state=mk_agent_state,
            autobuild=False,
        )
        self.assertEqual(len(t.edges), 0)
        t.build()
        self.assertEqual({k for k in t.nodes.keys()}, {"child"})
        edge_strs = set(map(repr, t.edges))
        self.assertIn("Edge('ROOT:a' -[related]-> 'child')", edge_strs)

    def test_expansion_inputs_none_leaf(self):
        def supervisor(state): return [Node("leaf")]
        def expansion_inputs(node): return None
        def is_class(node): return None

        t = Tree(self.code_path, "a", supervisor, expansion_inputs, is_class, mk_agent_state, autobuild=False)
        t.build()
        self.assertIn("leaf", t.nodes)
        self.assertIn("leaf", t.expanded)

    def test_supervisor_exception_safety(self):
        def supervisor(state):
            raise RuntimeError("boom")
        def expansion_inputs(node): return None
        def is_class(node): return None

        t = Tree(self.code_path, "a", supervisor, expansion_inputs, is_class, mk_agent_state, autobuild=False)
        t.build()
        self.assertEqual(len(t.nodes), 0)

    def test_cycle_prevention_a_to_b_b_to_a(self):
        def supervisor(state):
            cur = state["object_name"]
            if cur == "a": return [Node("b")]
            if cur == "b": return [Node("a")]
            return []

        def expansion_inputs(node):
            return ("//code", self.code_path, node.name)

        def is_class(node): return None

        t = Tree(self.code_path, "a", supervisor, expansion_inputs, is_class, mk_agent_state, autobuild=False)
        t.build()
        self.assertTrue({"a", "b"}.issubset(set(t.nodes.keys())))
        edge_strs = set(map(repr, t.edges))
        self.assertIn("Edge('ROOT:a' -[related]-> 'b')", edge_strs)
        self.assertIn("Edge('b' -[related]-> 'a')", edge_strs)
        self.assertTrue({"a", "b"}.issubset(t.expanded))

    def test_filter_nameless_items(self):
        def supervisor(state): return [BadItem(), EmptyName()]
        def expansion_inputs(node): return None
        def is_class(node): return None

        t = Tree(self.code_path, "a", supervisor, expansion_inputs, is_class, mk_agent_state, autobuild=False)
        t.build()
        self.assertEqual(len(t.nodes), 0)

    def test_classdetail_flyweight_reuse(self):
        def supervisor(state):
            cur = state["object_name"]
            if cur == "a": return [Node("x"), Node("y")]
            return []

        def expansion_inputs(node): return ("//code", self.code_path, node.name)
        def is_class(node): return "SameClass"  if node.name in ("x", "y") else None

        t = Tree(self.code_path, "a", supervisor, expansion_inputs, is_class, mk_agent_state, autobuild=False)
        t.build()
        self.assertEqual(set(t.class_details.keys()), {"SameClass"})
        edge_strs = set(map(repr, t.edges))
        self.assertIn("Edge('x' -[class_of]-> 'SameClass')", edge_strs)
        self.assertIn("Edge('y' -[class_of]-> 'SameClass')", edge_strs)

    def test_root_code_loaded(self):
        t = Tree(self.code_path, "a", supervisor=lambda s: [], expansion_inputs=lambda n: None,
                 is_class=lambda n: None, make_agent_state=mk_agent_state, autobuild=False)
        with open(self.code_path, "r", encoding="utf-8") as f:
            expected = f.read()
        self.assertEqual(t.root_code, expected)

if __name__ == "__main__":
    unittest.main(verbosity=2)
