"""
Builds a call-graph-ish structure from Java source using tree-sitter.

Key improvements in this revision:
- Fixed import extraction (uses Query correctly and handles wildcards/static imports).
- Much clearer inline comments + expanded docstrings with examples.
- More robust lambda handling (`lambda_parameters` vs `inferred_parameters`).
- Safer graph updates and symbol propagation.
- Light type inference for `var` and `new` expressions.
"""

import networkx as nx
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict
from collections import deque

# To install the required libraries:
#   pip install tree-sitter tree-sitter-java networkx
import tree_sitter_java as tsj
from tree_sitter import Language, Parser, Node

# --- Parser Initialization ---
# Create a Language object for Java from tree_sitter_java, then build a Parser with it.
try:
    JAVA_LANGUAGE = Language(tsj.language())
except Exception as e:
    print(f"Error loading tree-sitter Java language: {e}")
    print("Please ensure 'tree-sitter' and 'tree-sitter-java' are installed correctly.")
    raise

# Newer py-tree-sitter supports passing the language in the constructor.
parser = Parser(JAVA_LANGUAGE)


@dataclass
class CodeNode:
    """
    Represents a node in the custom call graph with rich metadata.

    Attributes:
        name: The unique string identifier (variable name, method name, class name, etc.).
        node_type: A coarse type (e.g., "method", "class", "variable", "method_call", "field").
        class_name: The enclosing class name (if relevant).
        code_path: Path to the .java file where the node originates.
        line: 1-based line number where the node originates.
        import_statement: Fully-qualified import if known (e.g., 'java.util.List').
        expandable: Whether this node potentially expands to more nodes (e.g., a method call).
        parents: Names of immediate upstream nodes in a chain (for breadcrumbs/context).
        generic_type: If this node is a generic container (e.g., List<T>), the T we inferred.

    Examples:
        >>> n = CodeNode(name="start", node_type="method", class_name="DummyTest", line=12)
        >>> n.name, n.node_type, n.class_name, n.line
        ('start', 'method', 'DummyTest', 12)
    """
    name: str
    node_type: str  # e.g., "method", "class", "variable", "method_call", "field_access"
    class_name: Optional[str] = None
    code_path: Optional[Path] = None
    line: Optional[int] = None
    import_statement: Optional[str] = None
    expandable: bool = False
    parents: List[str] = field(default_factory=list)
    generic_type: Optional[str] = None  # Store generic type like 'ApplicablePredicate' for a List


class CodeAnalyzer:
    """
    Analyzes Java source code to build a directional graph (nx.DiGraph) representing
    a hierarchical call/usage structure starting from a given method or symbol.

    Typical usage:
        >>> analyzer = CodeAnalyzer("DummyTest.java", "start")
        >>> graph = analyzer.analyze()
        >>> analyzer.print_graph_structure()
        >>> analyzer.print_chains()

    Args:
        file_path: Path to the Java file to analyze.
        start_object: Entry method/symbol name to start traversal (e.g., 'start').

    Notes:
        - This is intentionally single-file. See the suggestions section for a multi-file indexer.
    """

    def __init__(self, file_path: str, start_object: str):
        # Normalize path and store the user's desired starting symbol.
        self.file_path = Path(file_path)
        self.start_object = start_object

        # Graph to hold CodeNode objects (stored under node attribute 'data').
        self.graph = nx.DiGraph()

        # Read source and parse into an AST.
        self.code = self._read_file()
        self.tree = parser.parse(self.code.encode("utf-8")) if self.code else None

        # Maps simple class name -> fully-qualified import path ('List' -> 'java.util.List')
        # Also keeps '*' wildcard imports and 'static' imports (in dedicated sets).
        self.imports: Dict[str, str] = {}
        self.wildcard_imports: List[str] = []  # e.g., ['com.x.y.*']
        self.static_imports: List[str] = []    # e.g., ['java.util.Collections.*', 'java.lang.Math.max']

        # Symbol table at class scope: class -> { var -> type }
        self.class_symbol_table: Dict[str, Dict[str, str]] = {}

        if self.tree:
            self._extract_imports()

    def _read_file(self) -> Optional[str]:
        """
        Reads the Java file as UTF-8 text with basic validation.

        Returns:
            The file contents as a string, or None if invalid.

        Examples:
            >>> from pathlib import Path
            >>> p = Path("Example.java")
            >>> _ = p.write_text("class Example {}", encoding="utf-8")
            >>> CodeAnalyzer("Example.java", "main")._read_file()[:5]
            'class'
            >>> p.unlink()
        """
        if not self.file_path.exists():
            print(f"Error: File not found at '{self.file_path}'")
            return None
        if self.file_path.suffix.lower() != ".java":
            print(f"Error: Not a Java file: '{self.file_path}'")
            return None
        return self.file_path.read_text(encoding="utf-8")

    def _extract_imports(self) -> None:
        """
        Parses import statements and populates:
          - self.imports (simple name -> FQN),
          - self.wildcard_imports (list of 'pkg.*'),
          - self.static_imports (list of static imports).

        Implementation detail:
          Uses Language.query(...).captures(node) which returns a list of (Node, capture_name).

        Examples:
            Given:
                import java.util.List;
                import com.x.y.z.DefaultCalc;
                import static java.util.Collections.*;
                import com.pkg.*;
            Produces:
                imports = {'List':'java.util.List', 'DefaultCalc':'com.x.y.z.DefaultCalc'}
                wildcard_imports = ['com.pkg.*']
                static_imports = ['java.util.Collections.*']
        """
        if not self.tree:
            return

        # Find import_declaration nodes and capture the whole declaration.
        q = JAVA_LANGUAGE.query(
            r"""
            (import_declaration) @imp
            """
        )
        captures = q.captures(self.tree.root_node)

        for node, _name in captures:
            text = self._get_node_text(node).strip().rstrip(";")
            # text like: "import java.util.List" | "import com.a.b.*" | "import static java.lang.Math.max"
            # Normalize by removing leading 'import '.
            if text.startswith("import "):
                text = text[len("import "):].strip()

            is_static = text.startswith("static ")
            if is_static:
                text = text[len("static "):].strip()

            # Separate wildcard vs concrete.
            if text.endswith(".*"):
                if is_static:
                    self.static_imports.append(text)
                else:
                    self.wildcard_imports.append(text)
                continue

            # At this point we have a fully-qualified class or member (FQN or FQN.member).
            # If it's a static import like "java.lang.Math.max" we keep it in static_imports.
            if is_static and "." in text:
                self.static_imports.append(text)
                # Also map the simple member for potential lookups.
                # But do not add to class imports map (it’s a member).
                continue

            # Split fully qualified class, map simple name -> FQN
            simple = text.split(".")[-1]
            self.imports[simple] = text

    def _get_node_text(self, node: Optional[Node]) -> str:
        """
        Utility to slice source text of a node.

        Args:
            node: A tree-sitter Node (or None).

        Returns:
            Exact source substring for the given node, or empty string if None.

        Example:
            >>> # requires a parsed AST, shown here as intent rather than a doctest
            >>> # analyzer._get_node_text(some_node)  # -> 'List<ApplicablePredicate>'
        """
        if node is None:
            return ""
        return self.code[node.start_byte:node.end_byte]

    def _add_node_to_graph(self, name: str, **kwargs) -> None:
        """
        Adds or updates a CodeNode in the graph by name.

        Behavior:
            - If the node doesn't exist, create it with CodeNode(**kwargs).
            - If it exists, update fields (with a guard to not overwrite a non-Unknown class_name).

        Examples:
            >>> g = nx.DiGraph()
            >>> cn = CodeAnalyzer.__new__(CodeAnalyzer)  # bypass init for the example
            >>> cn.graph = g
            >>> cn._add_node_to_graph("A", node_type="class")
            >>> cn._add_node_to_graph("A", import_statement="java.lang.A")
            >>> g.nodes["A"]["data"].import_statement
            'java.lang.A'
        """
        if not name:
            return

        if name not in self.graph:
            self.graph.add_node(name, data=CodeNode(name=name, **kwargs))
            return

        node_data = self.graph.nodes[name].get("data")
        if not node_data:
            self.graph.nodes[name]["data"] = CodeNode(name=name, **kwargs)
            return

        # Merge fields carefully
        for key, value in kwargs.items():
            if key == "class_name" and value == "UnknownClass" and node_data.class_name not in (None, "UnknownClass"):
                continue
            setattr(node_data, key, value)

    def analyze(self) -> Optional[nx.DiGraph]:
        """
        Main entry point: builds the graph starting from self.start_object.

        Returns:
            The populated directed graph, or None on error.

        Examples:
            >>> analyzer = CodeAnalyzer("DummyTest.java", "start")
            >>> graph = analyzer.analyze()
            >>> isinstance(graph, nx.DiGraph)
            True
        """
        if not self.tree:
            print("Analysis aborted due to file reading/parsing errors.")
            return None

        # Worklist BFS/DFS hybrid (queue).
        methods_to_analyze = deque([self.start_object])
        analyzed_methods = set()

        while methods_to_analyze:
            current = methods_to_analyze.popleft()
            if current in analyzed_methods:
                continue

            # Find the declaration node for the method/identifier we're expanding.
            method_decl = self._find_object_in_ast(self.tree.root_node, current)
            if not method_decl:
                # Not found in this file; we might still capture usages elsewhere in scope.
                continue

            analyzed_methods.add(current)

            # Walk up to the enclosing class or enum to annotate where this lives.
            enclosing = method_decl
            while enclosing and enclosing.type not in ("class_declaration", "enum_declaration"):
                enclosing = enclosing.parent

            class_name = (
                self._get_node_text(enclosing.child_by_field_name("name"))
                if enclosing else "UnknownClass"
            )

            # Register the root "method" node for this chunk.
            self._add_node_to_graph(
                name=current,
                node_type="method",
                class_name=class_name,
                code_path=self.file_path,
                line=method_decl.start_point[0] + 1,
            )

            # Build a scope symbol table seeded by class fields (and later, params & locals).
            scope_symbols: Dict[str, str] = {}
            if enclosing:
                self._handle_class_fields(enclosing, class_name)
                scope_symbols.update(self.class_symbol_table.get(class_name, {}))

            # Method parameters become in-scope variables.
            self._handle_parameters(method_decl, current, class_name, scope_symbols)

            # Traverse the method body for statements and expressions.
            body = method_decl.child_by_field_name("body")
            if body:
                self._build_graph_from_scope(
                    body, current, class_name, methods_to_analyze, scope_symbols
                )

        return self.graph

    def _build_graph_from_scope(
        self,
        scope_node: Node,
        parent_name: str,
        class_name: str,
        methods_to_analyze: deque,
        scope_symbols: Dict[str, str],
    ) -> None:
        """
        Recursively traverses a block/scope building edges and nodes from statements.

        Covers:
          - Variable declarations, expressions, if/else branches, returns, nested blocks.

        Example:
            (internal helper, called by analyze())
        """
        for child in scope_node.named_children:
            t = child.type

            if t == "local_variable_declaration":
                self._handle_variable_declaration(
                    child, parent_name, class_name, methods_to_analyze, scope_symbols
                )

            elif t == "expression_statement" and child.named_child_count > 0:
                self._handle_expression(
                    child.named_child(0), parent_name, class_name, methods_to_analyze, scope_symbols
                )

            elif t == "if_statement":
                # Handle condition
                cond = child.child_by_field_name("condition")
                if cond:
                    expr = cond
                    if expr.type == "parenthesized_expression" and expr.named_child_count > 0:
                        expr = expr.named_child(0)
                    self._handle_expression(
                        expr, parent_name, class_name, methods_to_analyze, scope_symbols
                    )

                # Then handle branches
                cons = child.child_by_field_name("consequence")
                if cons:
                    self._build_graph_from_scope(
                        cons, parent_name, class_name, methods_to_analyze, scope_symbols
                    )
                alt = child.child_by_field_name("alternative")
                if alt:
                    self._build_graph_from_scope(
                        alt, parent_name, class_name, methods_to_analyze, scope_symbols
                    )

            elif t == "return_statement" and child.named_child_count > 0:
                self._handle_expression(
                    child.named_child(0), parent_name, class_name, methods_to_analyze, scope_symbols
                )

            elif t in ("block", "else_clause"):
                self._build_graph_from_scope(
                    child, parent_name, class_name, methods_to_analyze, scope_symbols
                )

            # (Optional) Extend here: for_statement, enhanced_for_statement, while_statement,
            #   do_statement, try_statement (try/catch/finally), switch_statement/expression.

    def _handle_class_fields(self, class_node: Node, class_name: str) -> None:
        """
        Parses class-level fields and records them in the class_symbol_table.
        Also links field -> type and (if generic) type -> generic parameter.

        Examples:
            For:
                private final List<ApplicablePredicate> predicates = null;

            Produces:
                class_symbol_table['DummyTest']['predicates'] = 'ApplicablePredicate'
        """
        if class_name not in self.class_symbol_table:
            self.class_symbol_table[class_name] = {}

        body = class_node.child_by_field_name("body")
        if not body:
            return

        for member in body.children:
            if member.type != "field_declaration":
                continue

            field_type_node = member.child_by_field_name("type")
            if not field_type_node:
                continue

            type_arg = None
            # Heuristic: if field is generic, peel T from List<T>.
            if field_type_node.type == "generic_type":
                raw_type_node = field_type_node.child(0)
                field_type = self._get_node_text(raw_type_node)
                type_args = field_type_node.child_by_field_name("type_arguments")
                if type_args and type_args.named_child_count > 0:
                    type_arg = self._get_node_text(type_args.named_child(0))
            else:
                field_type = self._get_node_text(field_type_node)

            declarator = member.child_by_field_name("declarator")
            if not declarator:
                continue

            field_name = self._get_node_text(declarator.child_by_field_name("name"))

            # Register in symbol table and graph
            self.class_symbol_table[class_name][field_name] = type_arg or field_type
            self._add_node_to_graph(
                name=field_name, node_type="field", class_name=class_name, generic_type=type_arg
            )
            self._add_node_to_graph(name=field_type, node_type="class")
            self.graph.add_edge(class_name, field_name)  # containment
            self.graph.add_edge(field_name, field_type, relation="is an instance of")

            if type_arg:
                self._add_node_to_graph(name=type_arg, node_type="class")
                self.graph.add_edge(field_type, type_arg, relation="of type")

    def _handle_parameters(
        self,
        method_node: Node,
        method_name: str,
        class_name: str,
        scope_symbols: Dict[str, str],
    ) -> None:
        """
        Records formal parameters as in-scope variables and links to their types.

        Example:
            For 'void start(Reporter reporter)', adds:
              - param node 'reporter'
              - class node 'Reporter' (with import if available)
              - edge reporter -> Reporter ("is an instance of")
        """
        params_node = method_node.child_by_field_name("parameters")
        if not params_node:
            return

        for param in params_node.children:
            if param.type != "formal_parameter":
                continue

            p_type_node = param.child_by_field_name("type")
            p_name_node = param.child_by_field_name("name")
            if not (p_type_node and p_name_node):
                continue

            param_type = self._get_node_text(p_type_node)
            param_name = self._get_node_text(p_name_node)
            scope_symbols[param_name] = param_type

            self._add_node_to_graph(
                name=param_name, node_type="parameter", class_name=class_name, line=param.start_point[0] + 1
            )
            self.graph.add_edge(method_name, param_name, relation="is an input")

            self._add_node_to_graph(
                name=param_type,
                node_type="class",
                import_statement=self.imports.get(param_type),
            )
            self.graph.add_edge(param_name, param_type, relation="is an instance of")

    def _handle_variable_declaration(
        self,
        node: Node,
        parent_name: str,
        class_name: str,
        methods_to_analyze: deque,
        scope_symbols: Dict[str, str],
    ) -> None:
        """
        Handles 'T x = ...;' declarations (including basic generic inference and 'var').

        - Adds variable node, links var -> type.
        - If initializer exists, processes its expression to extend the chain.

        Example:
            'ProcessingService service = new ProcessingService();'
              service -> ProcessingService ("is an instance of")
              parent_name -> service (containment/flow)
              parent_name -> ProcessingService ("instantiates") via init expression
        """
        var_type_node = node.child_by_field_name("type")
        declarator = node.child_by_field_name("declarator")
        if not (var_type_node and declarator):
            return

        # Resolve the declared type (with mild support for generics and 'var').
        var_type_text = self._get_node_text(var_type_node)
        generic_type = None

        if var_type_node.type == "generic_type":
            type_args_node = var_type_node.child_by_field_name("type_arguments")
            if type_args_node and type_args_node.named_child_count > 0:
                generic_type = self._get_node_text(type_args_node.named_child(0))

        # Java 10: local var type inference
        if var_type_text == "var":
            init = declarator.child_by_field_name("value")
            # Heuristic: if initializing with 'new T(...)', infer T
            if init and init.type == "object_creation_expression":
                tnode = init.child_by_field_name("type")
                if tnode:
                    var_type_text = self._get_node_text(tnode)

        var_name = self._get_node_text(declarator.child_by_field_name("name"))

        # Record variable and its type in the current scope.
        scope_symbols[var_name] = generic_type or var_type_text

        self._add_node_to_graph(
            name=var_name,
            node_type="variable",
            class_name=class_name,
            code_path=self.file_path,
            line=node.start_point[0] + 1,
            expandable=True,
            parents=[parent_name],
            generic_type=generic_type,
        )
        self.graph.add_edge(parent_name, var_name)

        self._add_node_to_graph(
            name=var_type_text,
            node_type="class",
            class_name=class_name,
            code_path=self.file_path,
            import_statement=self.imports.get(var_type_text),
            parents=[var_name],
        )
        self.graph.add_edge(var_name, var_type_text, relation="is an instance of")

        # If there is an initializer, analyze it as an expression.
        var_value = declarator.child_by_field_name("value")
        if var_value:
            self._handle_expression(
                var_value, var_name, class_name, methods_to_analyze, scope_symbols
            )

    def _handle_arguments(
        self,
        arg_node: Node,
        call_parent_name: str,
        methods_to_analyze: deque,
        scope_symbols: Dict[str, str],
    ) -> None:
        """
        Handles the argument_list in method invocations / constructors.

        - If identifier, links call -> identifier with 'uses as argument'.
        - If nested invocation/lambda, recursively analyze.

        Example:
            f(x, g(y))   # links: f -> x, f -> g, g -> y
        """
        if not arg_node or arg_node.type != "argument_list":
            return

        for arg in arg_node.named_children:
            t = arg.type
            if t == "identifier":
                arg_name = self._get_node_text(arg)
                if arg_name not in self.graph:
                    self._add_node_to_graph(name=arg_name, node_type="argument")
                self.graph.add_edge(call_parent_name, arg_name, relation="uses as argument")

            elif t in ("method_invocation", "lambda_expression", "object_creation_expression"):
                self._handle_expression(arg, call_parent_name, "UnknownClass", methods_to_analyze, scope_symbols)

            # Optionally handle literals/binary/conditional/etc.
            # elif t in (...): pass  # can be extended to attach literal nodes if desired

    def _handle_expression(
        self,
        node: Node,
        parent_name: str,
        class_name: str,
        methods_to_analyze: deque,
        scope_symbols: Dict[str, str],
    ) -> Optional[str]:
        """
        Handles expressions and returns the terminal symbol name when appropriate.
        Supports: identifier, field_access, object_creation_expression,
                  lambda_expression, method_invocation (chained), and more.

        Returns:
            A symbol name (e.g. 'service', 'processRequest') when useful, otherwise None.

        Example:
            - For 'defaultCalc.isOf(result).isValid(service)', this will:
                parent -> 'defaultCalc' (var), then -> 'isOf', then -> 'isValid'
        """
        t = node.type

        # Plain variable or parameter reference.
        if t == "identifier":
            return self._get_node_text(node)

        # obj.field -> link parent object to a "field_access" node (name=field identifier)
        if t == "field_access":
            obj_node = node.child_by_field_name("object")
            field_node = node.child_by_field_name("field")
            parent_obj_name = self._handle_expression(
                obj_node, parent_name, class_name, methods_to_analyze, scope_symbols
            )
            field_name = self._get_node_text(field_node)
            self._add_node_to_graph(name=field_name, node_type="field_access", parents=[parent_obj_name])
            if parent_obj_name:
                self.graph.add_edge(parent_obj_name, field_name)
            return field_name

        # new T(args)
        if t == "object_creation_expression":
            obj_type_node = node.child_by_field_name("type")
            obj_type = self._get_node_text(obj_type_node) if obj_type_node else "UnknownClass"
            self._add_node_to_graph(name=obj_type, node_type="class_instantiation", parents=[parent_name])
            self.graph.add_edge(parent_name, obj_type, relation="instantiates")
            self._handle_arguments(node.child_by_field_name("arguments"), parent_name, methods_to_analyze, scope_symbols)
            return obj_type

        # Lambda expressions: parameters -> body. We propagate any inferred T for the parameter.
        if t == "lambda_expression":
            lambda_body = node.child_by_field_name("body")
            params_node = node.child_by_field_name("parameters")
            lambda_param_name = None

            # Two shapes are common in tree-sitter-java:
            #   - 'lambda_parameters' (may contain 'identifier' or 'formal_parameter' children)
            #   - 'inferred_parameters' (identifiers without declared types)
            if params_node:
                if params_node.type in ("lambda_parameters", "inferred_parameters"):
                    # Single identifier: `x -> ...` or `(x) -> ...`
                    if params_node.named_child_count == 1 and params_node.named_child(0).type == "identifier":
                        lambda_param_name = self._get_node_text(params_node.named_child(0))
                elif params_node.type == "identifier":
                    # Rare: parameters node is the identifier itself
                    lambda_param_name = self._get_node_text(params_node)

            if lambda_body and lambda_param_name:
                parent_node_data = self.graph.nodes.get(parent_name, {}).get("data")
                inferred_type = parent_node_data.generic_type if parent_node_data else None

                lambda_scope = scope_symbols.copy()
                if inferred_type:
                    # Example: predicates.stream().anyMatch(predicate -> ...)
                    # If 'predicates' was List<ApplicablePredicate>, bind predicate: ApplicablePredicate
                    lambda_scope[lambda_param_name] = inferred_type

                # Recurse into body (can be block or expression)
                self._build_graph_from_scope(lambda_body, parent_name, class_name, methods_to_analyze, lambda_scope)
            return None

        # obj.call(args) or call(args)
        if t == "method_invocation":
            method_name = self._get_node_text(node.child_by_field_name("name"))
            obj_node = node.child_by_field_name("object")

            # If chained like a.b().c(), we first resolve the object part, then attach.
            chain_parent = parent_name
            if obj_node:
                chain_parent = self._handle_expression(
                    obj_node, parent_name, class_name, methods_to_analyze, scope_symbols
                )

            self._add_node_to_graph(
                name=method_name,
                node_type="method_call",
                class_name=class_name,
                code_path=self.file_path,
                line=node.start_point[0] + 1,
                expandable=True,
                parents=[chain_parent],
            )

            if chain_parent:
                self.graph.add_edge(chain_parent, method_name, relation="calls method")

                # Propagate generic_type down a call chain (helps inferring lambda T).
                parent_data = self.graph.nodes.get(chain_parent, {}).get("data")
                if parent_data and parent_data.generic_type:
                    self.graph.nodes[method_name]["data"].generic_type = parent_data.generic_type

            # Opportunistic: if a method with this simple name exists in file, queue it for expansion.
            if self._find_object_in_ast(self.tree.root_node, method_name):
                if method_name not in methods_to_analyze:
                    methods_to_analyze.append(method_name)

            # Process arguments (identifiers, nested calls, lambdas).
            self._handle_arguments(node.child_by_field_name("arguments"), method_name, methods_to_analyze, scope_symbols)
            return method_name

        # TODO: Handle method references: obj::method or Type::staticMethod (tree-sitter node: 'method_reference').
        # TODO: Handle assignment, conditional, binary, array_access, cast, etc., if needed.

        return None

    def _find_object_in_ast(self, node: Node, target: str) -> Optional[Node]:
        """
        Recursively search for a method_declaration named `target`.

        NOTE:
            This is a simple name-based search and does not consider overloads or
            the receiver type. See the suggestions section for a typed resolution.

        Returns:
            The method_declaration Node if found, otherwise None.

        Examples:
            >>> # Given a parsed class with 'void start() {...}', this returns that node:
            >>> # analyzer._find_object_in_ast(analyzer.tree.root_node, "start")
        """
        if node.type == "method_declaration":
            name_node = node.child_by_field_name("name")
            if name_node and self._get_node_text(name_node) == target:
                return node

        for child in node.children:
            found = self._find_object_in_ast(child, target)
            if found:
                return found
        return None

    def print_graph_structure(self) -> None:
        """
        Prints a readable structure of the graph:
            node -> children (with relation labels where available).

        Example:
            >>> # analyzer.print_graph_structure()
            # -> start (type=method, expandable=True)
            #   |--[is an input]-->
            #   -> reporter (type=parameter, ...)
            #   ...
        """
        if not self.graph:
            print("Graph is empty. Run analyze() first.")
            return

        print("\n--- Call Graph Structure ---")
        printed_edges = set()

        def print_node_and_children(node_name: str, indent: str = "") -> None:
            if node_name not in self.graph.nodes:
                return

            node_data = self.graph.nodes[node_name].get("data")
            if not node_data:
                return

            details = f"type={node_data.node_type}, expandable={node_data.expandable}"
            if node_data.import_statement:
                details += f", import='{node_data.import_statement}'"
            print(f"{indent}-> {node_data.name} ({details})")

            # Walk outgoing edges
            for child in sorted(self.graph.successors(node_name)):
                edge = (node_name, child)
                if edge in printed_edges:
                    continue
                printed_edges.add(edge)

                edge_data = self.graph.get_edge_data(node_name, child) or {}
                relation = edge_data.get("relation")
                if relation:
                    print(f"{indent}  |--[{relation}]-->")
                print_node_and_children(child, indent + "  ")

        print_node_and_children(self.start_object)
        print("--------------------------")

    def print_chains(self) -> None:
        """
        Finds all simple paths from the start_object to every leaf (a node with out-degree 0)
        and prints them in a readable 'A -> B -> C' format.

        Example:
            >>> # analyzer.print_chains()
            # start -> service -> processRequest
            # start -> defaultCalc -> isOf -> isValid
        """
        if not self.graph:
            print("Graph is empty. Run analyze() first.")
            return

        print("\n--- Call Chains ---")
        leaf_nodes = [n for n, outd in self.graph.out_degree() if outd == 0]
        all_paths = []
        for leaf in leaf_nodes:
            for path in nx.all_simple_paths(self.graph, source=self.start_object, target=leaf):
                all_paths.append(path)

        if not all_paths:
            print("No complete chains found from the start object.")
            return

        for path in sorted(all_paths):
            print(" -> ".join(path))
        print("-------------------")


if __name__ == "__main__":
    # ---- Demo: write a temporary Java file, analyze it, show the graph, then clean up. ----
    dummy_java_code = """
package com.example;

import com.x.y.z.DefaultCalc;
import com.x.y.z.TransEvent;
import com.x.y.z.Reporter;
import java.util.List;
import java.util.Optional;
import java.util.stream.Stream;

public class DummyTest {

    private final List<ApplicablePredicate> predicates = null;

    public void start(Reporter reporter) {
        System.out.println("Main process started.");
        ProcessingService service = new ProcessingService();
        String result = service.processRequest("user-123");
        DefaultCalc defaultCalc = new DefaultCalc();
        defaultCalc.isOf(result).isValid(service);
        TransEvent ts = new TransEvent(reporter);

        if (isapp(ts)){
            Optional.of(ts);
        } else {
            Optional.empty();
        }
    }

    private boolean isapp(TransEvent event){
        String ispord = event.isdaprodval().getornull();
        String ptemp = event.gettemp().getornull();

        return predicates.stream()
            .anyMatch(predicate->predicate.isapp(ispord,ptemp));
    }

    public static class ApplicablePredicate {
        public static final String NA = "NA";
        public static final String Any = "ANY";

        String ispord;
        String ptemp;

        private ApplicablePredicate(String ispord, String ptemp){
            this.ispord = ispord;
            this.ptemp = ptemp;
        }

        public boolean isapp(String p1, String p2) {
            return true;
        }
    }
}

class ProcessingService {
    public String processRequest(String user) { return "data"; }
}

class Reporter {
    public String isdaprodval() { return ""; }
    public String gettemp() { return ""; }
}
class TransEvent {
    public TransEvent(Reporter r) {}
    public String isdaprodval() { return ""; }
    public String gettemp() { return ""; }
}
"""
    dummy_file_name = "DummyTest.java"
    with open(dummy_file_name, "w", encoding="utf-8") as f:
        f.write(dummy_java_code)

    print(f"Created dummy file '{dummy_file_name}' for analysis.")
    analyzer = CodeAnalyzer(dummy_file_name, "start")
    graph = analyzer.analyze()

    if graph:
        analyzer.print_graph_structure()
        analyzer.print_chains()

    import os
    os.remove(dummy_file_name)
    print(f"\nCleaned up dummy file '{dummy_file_name}'.")
