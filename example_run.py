
import sys, tempfile
sys.path.append("/mnt/data")
from dfs_tree import Tree

class Node: 
    def __init__(self, name): self.name = name

def mk_agent_state(code, path, obj): return {"code": code, "path": path, "object_name": obj}

def supervisor(state):
    cur = state["object_name"]
    if cur == "start": return [Node("parse"), Node("analyze")]
    if cur == "parse": return [Node("classify")]
    return []

def expansion_inputs(node): return ("//code", "/tmp/Demo.java", node.name)
def is_class(node): return "FinalClass" if node.name == "classify" else None

with tempfile.NamedTemporaryFile("w+", suffix=".java", delete=False) as fp:
    fp.write("public class Demo {}")
    path = fp.name

t = Tree(path, "start", supervisor, expansion_inputs, is_class, mk_agent_state, edge_types={"parent_child":"uses","node_class":"class_of"})
print("Nodes:", sorted(t.nodes.keys()))
print("ClassDetails:", sorted(t.class_details.keys()))
print("Edges:", sorted(map(repr, t.edges)))
