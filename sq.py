import sqlglot
from sqlglot import exp
import networkx as nx

class SQLStructuralMapper:
    def __init__(self, file_path):
        self.file_path = file_path
        self.dependency_graph = nx.DiGraph()
        self.segments = {}

    def parse_and_map(self):
        with open(self.file_path, "r") as f:
            # Splitting by '/' is common for large SQL/PLSQL scripts
            raw_blocks = [b.strip() for b in f.read().split('/') if b.strip()]
        
        for idx, block in enumerate(raw_blocks):
            try:
                # Parse using the Oracle/PLSQL dialect
                expressions = sqlglot.parse(block, read="oracle")
                for expr in expressions:
                    # Logic to identify the scope (Package/Procedure)
                    scope_name = "ANONYMOUS_BLOCK"
                    if isinstance(expr, exp.Create):
                        scope_name = expr.this.this.sql()
                    
                    # Search for table modifications
                    for mod in expr.find_all((exp.Insert, exp.Update, exp.Merge, exp.Delete)):
                        table_node = mod.find(exp.Table)
                        if table_node:
                            table_name = table_node.name.upper()
                            node_id = f"mod_{idx}_{table_name}"
                            
                            # Build the segment record
                            self.segments[node_id] = {
                                "scope": scope_name,
                                "table": table_name,
                                "code": mod.sql(dialect="oracle"),
                                "full_context": block[:1000] # Snippet of the parent block
                            }
                            
                            # Add to Graph
                            self.graph_add(table_name, scope_name, node_id)
            except:
                continue # Skip unparseable noise

    def graph_add(self, table, scope, node_id):
        self.dependency_graph.add_edge(scope, table, type="modifies")
        self.dependency_graph.add_node(node_id, attr="modification_instance")
        self.dependency_graph.add_edge(table, node_id)

    def get_table_report(self, table_name):
        table_name = table_name.upper()
        results = []
        if table_name in self.dependency_graph:
            # Find all modification nodes linked to this table
            for neighbor in self.dependency_graph.neighbors(table_name):
                if neighbor in self.segments:
                    results.append(self.segments[neighbor])
        return results



# pip install deepagents langchain-openai sqlglot networkx
from deepagents import create_deep_agent
from langchain_core.tools import tool
import os

# 1. Initialize our Mapper (Global or passed via context)
mapper = SQLStructuralMapper("prod_script_3k.sql")
mapper.parse_and_map()

# 2. Define the Specialized Tool
@tool
def analyze_sql_dependencies(table_name: str) -> str:
    """Queries the structural map to find all Packages/Procedures 
    modifying a table and returns the exact modification SQL code."""
    data = mapper.get_table_report(table_name)
    if not data:
        return f"No modifications found for {table_name}."
    
    report = f"Analysis for Table: {table_name}\n"
    for item in data:
        report += f"--- Scope: {item['scope']} ---\n"
        report += f"Code:\n{item['code']}\n\n"
    return report

# 3. Create the Deep Agent
# The 'deepagents' harness provides write_todos and sub-agents automatically
agent = create_deep_agent(
    model="openai:gpt-4o",
    tools=[analyze_sql_dependencies],
    system_prompt=(
        "You are a Database Migration Expert. Your task is to analyze large SQL files. "
        "Use the analyze_sql_dependencies tool to find table modifications. "
        "Always explain the parent package/procedure context for every modification found."
    )
)

# 4. Invoke with a complex query
query = "Identify all modifications to the 'ORDERS' table and list their parent procedures."
# result = agent.invoke({"messages": [{"role": "user", "content": query}]})


def create_test_sql():
    with open("prod_script_3k.sql", "w") as f:
        f.write("CREATE OR REPLACE PACKAGE BODY PKG_SALES AS\n")
        for i in range(20):
            f.write(f"  PROCEDURE PROC_{i}(p_val IN NUMBER) IS\n  BEGIN\n")
            f.write("    -- 100 lines of noise\n" * 100)
            if i % 5 == 0:
                f.write("    MERGE INTO ORDERS o USING (SELECT p_val FROM DUAL) s ")
                f.write("ON (o.id = s.id) WHEN MATCHED THEN UPDATE SET status = 'DONE';\n")
            f.write(f"  END PROC_{i};\n/\n")
        f.write("END PKG_SALES;\n/")

create_test_sql()
