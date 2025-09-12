
import unittest, tempfile, textwrap, os, sys
sys.path.append("/mnt/data")

from dfs_tree import Tree, ClassDetail, Edge

class Node:
    def __init__(self, name): self.name = name
    def __repr__(self): return f"Node({self.name})"

def mk_agent_state(code: str, path: str, obj: str):
    return {"code": code, "path": path, "object_name": obj}

class TestAutobuildDFS(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile("w+", suffix=".java", delete=False)
        self.tmp.write(textwrap.dedent("""
        public class Demo {
            void a() {}
            void b() {}
            void c() {}
            void d() {}
            void e() {}
        }
        """).strip())
        self.tmp.flush()
        self.code_path = self.tmp.name

    def tearDown(self):
        try:
            os.unlink(self.code_path)
        except Exception:
            pass

    def test_basic_dfs_with_autobuild(self):
        def supervisor(state):
            cur = state["object_name"]
            if cur == "a": return [Node("b"), Node("c")]
            if cur == "b": return [Node("d")]
            if cur == "c": return [Node("e")]
            return []

        def expansion_inputs(node):
            if node.name == "e":
                return None
            return ("//code", self.code_path, node.name)

        def is_class(node):
            return "DemoClass" if node.name == "d" else None

        t = Tree(
            code_path=self.code_path,
            root_object="a",
            supervisor=supervisor,
            expansion_inputs=expansion_inputs,
            is_class=is_class,
            make_agent_state=mk_agent_state,
            edge_types={"parent_child": "uses", "node_class": "class_of"},
            autobuild=True,
        )

        self.assertEqual(set(t.nodes.keys()), {"b", "c", "d", "e"})
        self.assertEqual(set(t.class_details.keys()), {"DemoClass"})
        edge_strs = set(map(repr, t.edges))
        self.assertIn("Edge('ROOT:a' -[uses]-> 'b')", edge_strs)
        self.assertIn("Edge('ROOT:a' -[uses]-> 'c')", edge_strs)
        self.assertIn("Edge('b' -[uses]-> 'd')", edge_strs)
        self.assertIn("Edge('c' -[uses]-> 'e')", edge_strs)
        self.assertIn("Edge('d' -[class_of]-> 'DemoClass')", edge_strs)

    def test_edge_uniqueness_and_from_alias(self):
        def supervisor(state):
            return [Node("x"), Node("x")]

        def expansion_inputs(node):
            return None

        def is_class(node): return None

        t = Tree(
            code_path=self.code_path,
            root_object="a",
            supervisor=supervisor,
            expansion_inputs=expansion_inputs,
            is_class=is_class,
            make_agent_state=mk_agent_state,
            edge_types={"parent_child": "uses", "node_class": "class_of"},
            autobuild=True,
        )
        self.assertEqual(set(t.nodes.keys()), {"x"})
        self.assertEqual(len(t.edges), 1)
        edge = next(iter(t.edges))
        self.assertIs(getattr(edge, 'from'), edge.from_)
        self.assertEqual(edge.type, "uses")

    def test_default_edge_types(self):
        def supervisor(state): return [Node("k")]
        def expansion_inputs(node): return None
        def is_class(node): return "Klass" if node.name == "k" else None

        t = Tree(
            code_path=self.code_path,
            root_object="a",
            supervisor=supervisor,
            expansion_inputs=expansion_inputs,
            is_class=is_class,
            make_agent_state=mk_agent_state,
            autobuild=True,
        )
        edge_strs = set(map(repr, t.edges))
        self.assertIn("Edge('ROOT:a' -[related]-> 'k')", edge_strs)
        self.assertIn("Edge('k' -[class_of]-> 'Klass')", edge_strs)

if __name__ == "__main__":
    unittest.main(verbosity=2)
