from tree_sitter import Language, Parser
import os

# Define the path where you cloned the tree-sitter-java grammar
# Make sure this path points to the 'tree-sitter-java' directory you just cloned.
GRAMMAR_REPO_PATH = 'path/to/your/cloned/tree-sitter-java' # e.g., './tree-sitter-java'

# Define where you want to store the compiled library
# A common choice is to put it in the same directory as your Python script
BUILD_DIR = '.'
JAVA_LANGUAGE_FILE = os.path.join(BUILD_DIR, 'java.so') # For Linux/macOS
# For Windows, it might be 'java.dll' or 'java.pyd' depending on environment

# Ensure the grammar repository exists
if not os.path.exists(GRAMMAR_REPO_PATH):
    print(f"Error: tree-sitter-java grammar not found at {GRAMMAR_REPO_PATH}")
    print("Please clone it first: git clone https://github.com/tree-sitter/tree-sitter-java")
else:
    try:
        Language.build_library(
            # Store the library in the BUILD_DIR
            JAVA_LANGUAGE_FILE,
            # Path to the tree-sitter-java grammar directory
            [
                GRAMMAR_REPO_PATH
            ]
        )
        print(f"Successfully built Java grammar library at {JAVA_LANGUAGE_FILE}")
    except Exception as e:
        print(f"Error building Java grammar library: {e}")
        print("Ensure you have a C/C++ compiler installed (like GCC or Clang for Linux/macOS, MSVC for Windows).")

# Now you can load and use it
if os.path.exists(JAVA_LANGUAGE_FILE):
    JAVA_LANGUAGE = Language(JAVA_LANGUAGE_FILE, 'java')
    parser = Parser()
    parser.set_language(JAVA_LANGUAGE)
    print("Tree-sitter Java parser is ready!")
else:
    print("Java grammar library not found. Please check the build process.")
