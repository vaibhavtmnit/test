import os
import re

def find_java_class_path(java_file_path: str, import_statement: str) -> str | None:
    """
    Finds the absolute path of a Java class file based on an import statement.

    This function assumes a standard Java project layout where the directory
    structure mirrors the package structure (e.g., class com.example.MyClass
    is in com/example/MyClass.java).

    Args:
        java_file_path: The absolute or relative path to the Java file containing the import.
        import_statement: The full import statement string for the class to find.
                          (e.g., "import com.example.project.MyClass;").

    Returns:
        The absolute path to the imported Java file if the source root can be
        determined, otherwise None. This function does not guarantee that the
        returned file path exists.
    """
    if not os.path.exists(java_file_path):
        raise FileNotFoundError(f"The file '{java_file_path}' does not exist.")

    # 1. Parse the import statement to get the fully qualified class name.
    # Example: "import com.example.project.MyClass;" -> "com.example.project.MyClass"
    match = re.search(r'import\s+([\w\.]+);', import_statement)
    if not match:
        # Handles cases like wildcard imports or static imports, which are not supported.
        print(f"Warning: Could not parse a valid class from import statement: '{import_statement}'")
        return None
    
    qualified_name = match.group(1)

    # 2. Convert the qualified name to a relative file path.
    # Example: "com.example.project.MyClass" -> "com/example/project/MyClass.java"
    relative_path_of_import = qualified_name.replace('.', os.path.sep) + ".java"

    # 3. Determine the source root directory from the initial Java file.
    source_root = _find_source_root(java_file_path)
    
    if source_root is None:
        print(f"Warning: Could not determine the source root for '{java_file_path}'.")
        return None

    # 4. Combine the source root and the relative path to get the final path.
    return os.path.join(source_root, relative_path_of_import)


def _find_source_root(java_file_path: str) -> str | None:
    """Helper function to find the source root (e.g., 'src/main/java')."""
    package_path = ""
    try:
        with open(java_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Find the package declaration line.
                package_match = re.search(r'^\s*package\s+([\w\.]+);', line)
                if package_match:
                    package_name = package_match.group(1)
                    package_path = package_name.replace('.', os.path.sep)
                    break # Assuming only one package declaration per file
    except IOError as e:
        print(f"Error reading file {java_file_path}: {e}")
        return None

    # Get the absolute directory of the java file.
    file_dir = os.path.dirname(os.path.abspath(java_file_path))

    # The source root is the file's directory path minus the package path.
    if package_path and file_dir.endswith(package_path):
        # Cut the package path from the end of the directory path.
        return file_dir[:-len(package_path)].rstrip(os.path.sep)
    elif not package_path:
        # File is in the default package, so its directory is the source root.
        return file_dir
    
    # If the directory structure does not match the package declaration.
    return None

if __name__ == '__main__':
    # --- Example Usage ---
    # Create a dummy project structure for demonstration.
    # src/main/java/
    # └── com
    #     └── example
    #         ├── App.java
    #         └── util
    #             └── Helper.java

    # Create directories
    os.makedirs("src/main/java/com/example/util", exist_ok=True)

    # Create App.java
    app_java_content = """
package com.example;

import com.example.util.Helper; // We want to find the path for this import

public class App {
    public static void main(String[] args) {
        Helper.doSomething();
    }
}
"""
    with open("src/main/java/com/example/App.java", "w") as f:
        f.write(app_java_content)

    # Create Helper.java
    helper_java_content = """
package com.example.util;

public class Helper {
    public static void doSomething() {
        System.out.println("Hello from Helper!");
    }
}
"""
    with open("src/main/java/com/example/util/Helper.java", "w") as f:
        f.write(helper_java_content)
    
    # --- Function Call ---
    path_to_app_java = "src/main/java/com/example/App.java"
    import_to_find = "import com.example.util.Helper;"

    found_path = find_java_class_path(path_to_app_java, import_to_find)

    if found_path:
        # Normalize paths for consistent comparison across OS
        normalized_found = os.path.normpath(found_path)
        expected_path = os.path.abspath("src/main/java/com/example/util/Helper.java")
        normalized_expected = os.path.normpath(expected_path)

        print(f"Starting from: {path_to_app_java}")
        print(f"Looking for import: '{import_to_find}'")
        print("-" * 20)
        print(f"Found path: {normalized_found}")
        print(f"Expected path: {normalized_expected}")
        print(f"Success: {normalized_found == normalized_expected}")

