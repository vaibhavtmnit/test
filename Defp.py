import re

def check_java_class(java_code: str, class_name: str) -> str:
    """
    Checks if a Java class is defined or imported in a given Java code string.

    This function first strips all single-line (//) and multi-line (/* */)
    comments from the code before performing the checks. It does not detect
    classes included via wildcard imports (e.g., import java.util.*;).

    Args:
        java_code: A string containing the Java code.
        class_name: The name of the class to check for.

    Returns:
        'defined' if the class is defined in the code.
        'imported' if the class is explicitly imported.
        'not found' otherwise.
    """
    # Helper function to remove comments from the Java code string
    def _strip_comments(code: str) -> str:
        # Remove multi-line comments /* ... */
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        # Remove single-line comments // ...
        code = re.sub(r'//.*', '', code)
        return code

    # Escape class_name to safely use it in a regex pattern
    safe_class_name = re.escape(class_name)
    code_without_comments = _strip_comments(java_code)

    # 1. Pattern to find a class definition, e.g., 'public class MyClass {'
    # The \b ensures we match the whole word for the class name.
    definition_pattern = re.compile(r'\bclass\s+' + safe_class_name + r'\b')
    if definition_pattern.search(code_without_comments):
        return 'defined'

    # 2. Pattern to find an explicit import, e.g., 'import java.util.ArrayList;'
    import_pattern = re.compile(r'\bimport\s+[\w\.]*\.' + safe_class_name + r'\s*;')
    if import_pattern.search(code_without_comments):
        return 'imported'

    # 3. If neither pattern is found
    return 'not found'

