import os

def find_java_class_path(directory_path, important_statement):
    """
    Searches for a Java file containing a specific statement within a directory.

    Args:
        directory_path (str): The absolute or relative path to the directory
                              containing the Java project.
        important_statement (str): A unique line or statement from a Java class
                                   to search for.

    Returns:
        str: The full path to the .java file containing the statement,
             or None if the statement is not found in any .java file.
    """
    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found at '{directory_path}'")
        return None

    # os.walk recursively traverses the directory tree (top-down).
    for root, _, files in os.walk(directory_path):
        for file_name in files:
            # We are only interested in Java source files.
            if file_name.endswith(".java"):
                file_path = os.path.join(root, file_name)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        # Read the entire file content.
                        content = f.read()
                        # Check if the important statement exists in the content.
                        if important_statement in content:
                            return file_path
                except (IOError, UnicodeDecodeError) as e:
                    # Handle potential file reading errors.
                    print(f"Could not read file {file_path}: {e}")
                    continue

    # If the loop completes without finding the statement, return None.
    return Noneimport os

def find_java_class_path(directory_path, important_statement):
    """
    Searches for a Java file containing a specific statement within a directory.

    Args:
        directory_path (str): The absolute or relative path to the directory
                              containing the Java project.
        important_statement (str): A unique line or statement from a Java class
                                   to search for.

    Returns:
        str: The full path to the .java file containing the statement,
             or None if the statement is not found in any .java file.
    """
    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found at '{directory_path}'")
        return None

    # os.walk recursively traverses the directory tree (top-down).
    for root, _, files in os.walk(directory_path):
        for file_name in files:
            # We are only interested in Java source files.
            if file_name.endswith(".java"):
                file_path = os.path.join(root, file_name)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        # Read the entire file content.
                        content = f.read()
                        # Check if the important statement exists in the content.
                        if important_statement in content:
                            return file_path
                except (IOError, UnicodeDecodeError) as e:
                    # Handle potential file reading errors.
                    print(f"Could not read file {file_path}: {e}")
                    continue

    # If the loop completes without finding the statement, return None.
    return None
