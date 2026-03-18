import sqlglot
from sqlglot import exp
import networkx as nx
Help me brainstorm strategy to solve a  prodblem which involves intelligently finding information from .sql code files having code lines in the order of 10k.
The use case is as follows:
There are files with 10k codelines and i want to build a system to retrive information about a table which is being modified in the script the modification is most adding data to a table or creating and merging.
The return should have all the instances where the table of goven name is modified but the challenge is these tables might be linked to procedures or packages defined at the starting of the file thus if the table of interest is at line 4000 its doffocult to bring outs its dependencies as well. I thought of using deepagents by langchain but i am unable to come up with the retrieval strategy. The requirement is all eligible canddiated to be return with information related to their packages and procedures and one of the attributes of returned object should be only that part of the code which shows modifications.


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


import sqlglot
from sqlglot import exp
import networkx as nx
import re

class SQLStructuralMapper:
    def __init__(self, file_path):
        self.file_path = file_path
        self.dependency_graph = nx.DiGraph()
        self.segments = {}

    def _clean_block(self, block):
        """Removes SQL*Plus noise that causes sqlglot to fail."""
        # Remove comments, SHOW ERRORS, SET, and trailing slashes
        block = re.sub(r'--.*', '', block)
        block = re.sub(r'(?i)^\s*(SHOW|SET|PROMPT|SPOOL).*$', '', block, flags=re.MULTILINE)
        return block.strip()

    def parse_and_map(self):
        with open(self.file_path, "r") as f:
            content = f.read()
            # Split by '/' followed by a newline (standard PL/SQL block separator)
            raw_blocks = re.split(r'\n/\s*\n', content)
        
        for idx, raw_block in enumerate(raw_blocks):
            clean_block = self._clean_block(raw_block)
            if not clean_block: continue
            
            try:
                # Use 'oracle' or 'postgres' depending on your DB
                # 'read=None' allows it to try and guess if oracle fails
                expressions = sqlglot.parse(clean_block, read="oracle")
                
                # Tracking the "Parent" (Package or Procedure)
                current_scope = "GLOBAL"
                
                for expr in expressions:
                    # Logic to identify the scope (Package/Procedure/Function)
                    if isinstance(expr, (exp.Create, exp.Alter)):
                        # Safely extract the name of the object being created
                        found_name = expr.find(exp.Table) or expr.find(exp.Identifier)
                        if found_name:
                            current_scope = str(found_name).upper()

                    # Find modifications: INSERT, UPDATE, MERGE, DELETE
                    for mod in expr.find_all((exp.Insert, exp.Update, exp.Merge, exp.Delete)):
                        table_node = mod.find(exp.Table)
                        if table_node:
                            table_name = table_node.name.upper()
                            node_id = f"mod_{idx}_{table_name}"
                            
                            self.segments[node_id] = {
                                "scope": current_scope,
                                "table": table_name,
                                "code": mod.sql(dialect="oracle"),
                                "line_hint": idx # Tracking block index
                            }
                            
                            # Build the Graph: Scope -> Table -> Modification Instance
                            self.dependency_graph.add_edge(current_scope, table_name, type="contains_logic_for")
                            self.dependency_graph.add_edge(table_name, node_id, type="has_instance")
            except Exception as e:
                # Fallback: If strict parsing fails, use Regex as a safety net for "Modifications"
                self._fallback_regex_parse(clean_block, idx)

    def _fallback_regex_parse(self, block, idx):
        """If sqlglot fails, we still want to find modifications via Regex."""
        # Pattern for MERGE INTO table_name or INSERT INTO table_name
        pattern = r'(?i)(MERGE|INSERT|UPDATE|DELETE)\s+(?:INTO\s+)?([a-zA-Z0-9_.]+)'
        matches = re.findall(pattern, block)
        for action, table in matches:
            table_name = table.upper()
            node_id = f"fallback_{idx}_{table_name}"
            self.segments[node_id] = {
                "scope": "PARSING_ERROR_BLOCK",
                "table": table_name,
                "code": block[:500] + "... [Truncated due to parse error]",
                "note": "Extracted via Regex fallback"
            }
            self.dependency_graph.add_edge("UNKNOWN_SCOPE", table_name)
            self.dependency_graph.add_edge(table_name, node_id)

    def get_table_report(self, table_name):
        table_name = table_name.upper()
        results = []
        if table_name in self.dependency_graph:
            for neighbor in self.dependency_graph.neighbors(table_name):
                if neighbor in self.segments:
                    results.append(self.segments[neighbor])
        return results


