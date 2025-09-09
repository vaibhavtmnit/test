from typing import Optional
from tree_sitter import Node

def find_import_statement_for_class(code: str, class_name: str) -> Optional[str]:
    """
    Find the import statement for `class_name` in a Java source `code`.
    Returns:
      - 'SAME' if `class_name` is declared in this file (class/interface/enum/record),
      - the exact import line (including trailing semicolon) if present,
      - otherwise None.

    Precedence:
      1) Explicit non-static import of the class:   import a.b.ClassName;
      2) Static imports that reference the class:    import static a.b.ClassName.member; / import static a.b.ClassName.*;
      3) Package wildcard imports:                   import a.b.*;
    """
    src = code.encode("utf-8", errors="ignore")
    tree = parser.parse(src)
    root = tree.root_node

    def _text(n: Node) -> str:
        return src[n.start_byte:n.end_byte].decode("utf-8", errors="ignore")

    def _is_same_file_declaration() -> bool:
        kinds = ("class_declaration", "interface_declaration", "enum_declaration", "record_declaration")
        stack = [root]
        while stack:
            n = stack.pop()
            if n.type in kinds:
                name_node = n.child_by_field_name("name")
                if name_node is not None and _text(name_node) == class_name:
                    return True
            for i in range(n.child_count - 1, -1, -1):
                stack.append(n.child(i))
        return False

    if _is_same_file_declaration():
        return "SAME"

    explicit_imports = []        # import a.b.ClassName;
    static_exact_imports = []    # import static a.b.ClassName.member;
    static_wildcard_imports = [] # import static a.b.ClassName.*;
    package_wildcards = []       # import a.b.*;

    # Scan all import_declaration nodes
    stack = [root]
    while stack:
        n = stack.pop()
        if n.type == "import_declaration":
            line = _text(n).strip()
            # Normalize & parse the path
            # Examples:
            #   "import a.b.ClassName;"
            #   "import a.b.*;"
            #   "import static a.b.ClassName.member;"
            #   "import static a.b.ClassName.*;"
            body = line
            if body.endswith(";"):
                body = body[:-1].strip()
            assert body.startswith("import")
            body = body[len("import"):].strip()
            is_static = False
            if body.startswith("static"):
                is_static = True
                body = body[len("static"):].strip()

            # body is now like "a.b.ClassName" or "a.b.*" or "a.b.ClassName.member" or "a.b.ClassName.*"
            parts = body.split(".")
            if not parts:
                # malformed; skip
                pass
            else:
                if not is_static:
                    # Non-static: exact or package wildcard
                    if parts[-1] == "*":
                        package_wildcards.append(line)
                    else:
                        last_seg = parts[-1]
                        if last_seg == class_name:
                            explicit_imports.append(line)
                else:
                    # Static imports: the class name is the *second last* segment (last is either member or '*')
                    if len(parts) >= 2:
                        if parts[-1] == "*":
                            cls = parts[-2]
                            if cls == class_name:
                                static_wildcard_imports.append(line)
                        else:
                            cls = parts[-2]
                            if cls == class_name:
                                static_exact_imports.append(line)

        # DFS
        for i in range(n.child_count - 1, -1, -1):
            stack.append(n.child(i))

    # Choose the best match by precedence
    if explicit_imports:
        return explicit_imports[0]
    if static_exact_imports:
        return static_exact_imports[0]
    if static_wildcard_imports:
        return static_wildcard_imports[0]
    if package_wildcards:
        # A wildcard package import *might* bring the class into scope; return the first one.
        return package_wildcards[0]

    return None
