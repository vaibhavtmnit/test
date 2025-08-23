import tree_sitter_java as tsj
from tree_sitter import Parser, Language
import sys # Import sys for printing directly to stderr if needed, though print is fine.

def print_ast_recursive(node, indent=0):
    """
    Recursively prints the AST node by node, showing type and text.
    """
    node_text = node.text.decode('utf-8').strip().replace('\n', '\\n')
    print(f"{'  ' * indent}{node.type} [start={node.start_point}, end={node.end_point}] -> '{node_text}'")
    for child in node.children:
        print_ast_recursive(child, indent + 1)


def test_java_parsing(java_code: str):
    """
    Parses a given Java code string using tree-sitter-java and prints its AST.
    """
    try:
        # --- Updated Language and Parser Initialization ---
        # Initialize the Java Language object from tree_sitter_java
        lng = Language(tsj.language())
        # Create a parser instance and set its language
        parser = Parser(language=lng)
        # --- End Updated Initialization ---

        # Parse the Java code
        tree = parser.parse(java_code.encode('utf-8'))

        # Print the AST in a human-readable format
        print("Successfully parsed Java code. Here's the AST:")
        print_ast_recursive(tree.root_node)

    except Exception as e:
        # Catch any unexpected errors during parsing
        print(f"An unexpected error occurred during parsing: {e}", file=sys.stderr)
        print("Please ensure 'tree-sitter' and 'tree_sitter_java' are correctly installed", file=sys.stderr)
        print("and your Python environment is restarted.", file=sys.stderr)

# Example Java code to test
java_snippet = """
package com.example.app;

import java.util.List;

public class MyService {
    private String name;

    public MyService(String name) {
        this.name = name;
    }

    public void processData(List<String> data) {
        for (String item : data) {
            System.out.println("Processing: " + item + " by " + this.name);
        }
    }
}
"""

if __name__ == "__main__":
    print("--- Testing tree-sitter-java ---")
    test_java_parsing(java_snippet)
    print("\nTest complete.")
