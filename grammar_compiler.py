from tree_sitter import Parser
from tree_sitter_java import language as java_language # Import the Java language object

def test_java_parsing(java_code: str):
    """
    Parses a given Java code string using tree-sitter-java and prints its AST.
    """
    try:
        # Create a parser instance
        parser = Parser()
        # Set the language to Java using the imported language object
        parser.set_language(java_language)

        # Parse the Java code
        tree = parser.parse(java_code.encode('utf-8'))

        # Print the S-expression representation of the AST
        print("Successfully parsed Java code. Here's the AST (S-expression):")
        print(tree.root_node.sexp())

        # You can also iterate through the tree for more detailed inspection
        # For example, to find all class declarations:
        # print("\nNodes of type 'class_declaration':")
        # for node in tree.root_node.children:
        #     if node.type == 'class_declaration':
        #         print(f"  Class Name: {node.child_by_field_name('name').text.decode('utf-8')}")
        #         print(f"  Range: {node.start_point} to {node.end_point}")

    except Exception as e:
        print(f"An error occurred during parsing: {e}")
        print("Please ensure 'tree-sitter' and 'tree-sitter-java' are correctly installed.")
        print("You might need to restart your Python environment if you just installed them.")

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
