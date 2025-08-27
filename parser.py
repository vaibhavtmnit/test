"""
Java Execution Flow Analysis Tree using Tree-sitter

This builds a tree that follows execution flow and data dependencies,
showing how a human would trace through code step by step.

Installation requirements:
pip install tree-sitter tree-sitter-java

Usage:
analyzer = JavaExecutionAnalyzer()
tree = analyzer.build_execution_tree(java_code, "start")
analyzer.print_tree(tree)
"""

import tree_sitter_java as tsjava
from tree_sitter import Language, Parser, Node
from enum import Enum
from typing import List, Optional, Dict, Set, Union
from dataclasses import dataclass, field
import re

class EdgeType(Enum):
    ARGUMENT_OF = "argument of"
    INSTANCE_OF = "instance of"
    CALLS = "calls"
    RETURNS = "returns"
    ASSIGNED_TO = "assigned to"
    PARAMETER_OF = "parameter of"
    FIELD_ACCESS = "field access"
    METHOD_CALL = "method call"
    CONDITION = "condition"
    LAMBDA_PARAM = "lambda parameter"
    STREAM_OPERATION = "stream operation"
    CLASS_MEMBER = "class member"
    EXTENDS = "extends"
    IMPLEMENTS = "implements"

class NodeCategory(Enum):
    METHOD = "Method"
    CLASS = "Class"
    VARIABLE = "Variable"
    PARAMETER = "Parameter"
    FIELD = "Field"
    METHOD_CALL = "Method Call"
    LAMBDA = "Lambda"
    CONDITION = "Condition"
    ASSIGNMENT = "Assignment"
    CONSTANT = "Constant"
    STREAM_OP = "Stream Operation"

@dataclass
class ExecutionNode:
    name: str
    category: NodeCategory
    details: str
    line_number: int
    children: List['ExecutionNode'] = field(default_factory=list)
    edge_type: Optional[EdgeType] = None  # Relationship to parent
    source_text: str = ""
    ts_node: Optional[Node] = None
    is_external: bool = False  # True if defined outside current file
    
    def add_child(self, child: 'ExecutionNode', edge_type: EdgeType):
        child.edge_type = edge_type
        self.children.append(child)
    
    def __str__(self):
        edge_str = f" ({self.edge_type.value})" if self.edge_type else ""
        external_str = " [external]" if self.is_external else ""
        return f"{self.name}{edge_str} - {self.details}{external_str} (Line: {self.line_number})"

class JavaExecutionAnalyzer:
    def __init__(self):
        self.JAVA_LANGUAGE = Language(tsjava.language())
        self.parser = Parser(self.JAVA_LANGUAGE)
        self.source_code = ""
        self.analyzed_methods = set()  # To prevent infinite recursion
        self.class_info = {}  # Store class definitions
        self.method_info = {}  # Store method definitions
        self.field_info = {}  # Store field definitions
        
    def build_execution_tree(self, java_code: str, starting_point: str) -> Optional[ExecutionNode]:
        """Build execution flow tree starting from a method or class"""
        self.source_code = java_code.encode('utf-8')
        tree = self.parser.parse(self.source_code)
        
        # First pass: collect all definitions
        self._collect_definitions(tree.root_node)
        
        # Find and analyze the starting point
        start_node = self._find_method_or_class(tree.root_node, starting_point)
        if not start_node:
            return None
            
        # Build execution tree
        root = self._create_execution_node(start_node, NodeCategory.METHOD)
        self._analyze_execution_flow(start_node, root)
        
        return root
    
    def _collect_definitions(self, node: Node):
        """Collect all class, method, and field definitions"""
        if node.type == 'class_declaration':
            class_name = self._get_identifier_name(node)
            if class_name:
                self.class_info[class_name] = node
                
        elif node.type == 'method_declaration':
            method_name = self._get_identifier_name(node)
            if method_name:
                self.method_info[method_name] = node
                
        elif node.type == 'field_declaration':
            # Handle multiple field declarations
            for declarator in self._find_all_children_by_type(node, 'variable_declarator'):
                field_name = self._get_identifier_name(declarator)
                if field_name:
                    self.field_info[field_name] = node
        
        # Recursively collect from children
        for child in node.children:
            self._collect_definitions(child)
    
    def _analyze_execution_flow(self, ts_node: Node, exec_node: ExecutionNode):
        """Analyze execution flow from a method"""
        if exec_node.name in self.analyzed_methods:
            return
        self.analyzed_methods.add(exec_node.name)
        
        if ts_node.type == 'method_declaration':
            self._analyze_method_execution(ts_node, exec_node)
        elif ts_node.type == 'class_declaration':
            self._analyze_class_members(ts_node, exec_node)
    
    def _analyze_method_execution(self, method_node: Node, exec_node: ExecutionNode):
        """Analyze execution flow within a method"""
        # 1. First add parameters
        params_node = self._find_child_by_type(method_node, 'formal_parameters')
        if params_node:
            for param in self._find_all_children_by_type(params_node, 'parameter'):
                param_name = self._get_identifier_name(param)
                param_type = self._get_parameter_type(param)
                if param_name:
                    param_exec = ExecutionNode(
                        name=param_name,
                        category=NodeCategory.PARAMETER,
                        details=f"Type: {param_type}",
                        line_number=param.start_point[0] + 1,
                        ts_node=param
                    )
                    exec_node.add_child(param_exec, EdgeType.PARAMETER_OF)
                    
                    # If parameter has a class type, add class info
                    if param_type in self.class_info:
                        class_exec = self._create_class_node(param_type)
                        param_exec.add_child(class_exec, EdgeType.INSTANCE_OF)
        
        # 2. Find method body and analyze statements
        body = self._find_child_by_type(method_node, 'block')
        if body:
            self._analyze_block_execution(body, exec_node)
    
    def _analyze_block_execution(self, block_node: Node, parent_exec: ExecutionNode):
        """Analyze execution flow in a code block"""
        for statement in block_node.children:
            if statement.type == 'local_variable_declaration':
                self._analyze_variable_declaration(statement, parent_exec)
            elif statement.type == 'expression_statement':
                self._analyze_expression_statement(statement, parent_exec)
            elif statement.type == 'if_statement':
                self._analyze_if_statement(statement, parent_exec)
            # Add more statement types as needed
    
    def _analyze_variable_declaration(self, var_decl: Node, parent_exec: ExecutionNode):
        """Analyze variable declaration and assignment"""
        type_node = self._find_child_by_type(var_decl, 'type')
        var_type = self._get_node_text(type_node) if type_node else "unknown"
        
        for declarator in self._find_all_children_by_type(var_decl, 'variable_declarator'):
            var_name = self._get_identifier_name(declarator)
            if not var_name:
                continue
                
            var_exec = ExecutionNode(
                name=var_name,
                category=NodeCategory.VARIABLE,
                details=f"Type: {var_type}",
                line_number=declarator.start_point[0] + 1,
                ts_node=declarator
            )
            parent_exec.add_child(var_exec, EdgeType.ASSIGNED_TO)
            
            # Check if it's a class type
            if var_type in self.class_info:
                class_exec = self._create_class_node(var_type)
                var_exec.add_child(class_exec, EdgeType.INSTANCE_OF)
            
            # Analyze initializer (e.g., new ProcessingService())
            initializer = self._find_child_by_type(declarator, 'object_creation_expression')
            if initializer:
                self._analyze_object_creation(initializer, var_exec)
            else:
                # Check for method call initializer
                method_call = self._find_child_by_type(declarator, 'method_invocation')
                if method_call:
                    self._analyze_method_invocation(method_call, var_exec)
    
    def _analyze_expression_statement(self, expr_stmt: Node, parent_exec: ExecutionNode):
        """Analyze expression statements"""
        # Look for method invocations
        for method_call in self._find_all_descendants_by_type(expr_stmt, 'method_invocation'):
            self._analyze_method_invocation(method_call, parent_exec)
        
        # Look for assignments
        for assignment in self._find_all_descendants_by_type(expr_stmt, 'assignment_expression'):
            self._analyze_assignment(assignment, parent_exec)
    
    def _analyze_method_invocation(self, method_call: Node, parent_exec: ExecutionNode):
        """Analyze method call and its chain"""
        # Get the object being called on
        object_node = method_call.children[0] if method_call.children else None
        method_name = self._get_method_name_from_invocation(method_call)
        
        if not method_name:
            return
            
        # Create method call node
        call_exec = ExecutionNode(
            name=method_name,
            category=NodeCategory.METHOD_CALL,
            details="Method call",
            line_number=method_call.start_point[0] + 1,
            ts_node=method_call
        )
        parent_exec.add_child(call_exec, EdgeType.CALLS)
        
        # Analyze object being called on
        if object_node:
            obj_name = self._get_node_text(object_node).split('.')[0]  # Get base object
            obj_exec = ExecutionNode(
                name=obj_name,
                category=NodeCategory.VARIABLE,
                details="Method receiver",
                line_number=object_node.start_point[0] + 1,
                ts_node=object_node
            )
            call_exec.add_child(obj_exec, EdgeType.CALLS)
        
        # Analyze arguments
        args_node = self._find_child_by_type(method_call, 'argument_list')
        if args_node:
            for arg in args_node.children:
                if arg.type != ',' and arg.type != '(' and arg.type != ')':
                    arg_name = self._get_node_text(arg)
                    arg_exec = ExecutionNode(
                        name=arg_name,
                        category=NodeCategory.VARIABLE,
                        details="Method argument",
                        line_number=arg.start_point[0] + 1,
                        ts_node=arg
                    )
                    call_exec.add_child(arg_exec, EdgeType.ARGUMENT_OF)
        
        # If method is defined locally, analyze it
        if method_name in self.method_info and method_name not in self.analyzed_methods:
            local_method = self.method_info[method_name]
            method_exec = self._create_execution_node(local_method, NodeCategory.METHOD)
            call_exec.add_child(method_exec, EdgeType.CALLS)
            self._analyze_execution_flow(local_method, method_exec)
    
    def _analyze_object_creation(self, obj_creation: Node, parent_exec: ExecutionNode):
        """Analyze object creation (new ClassName())"""
        type_node = self._find_child_by_type(obj_creation, 'type_identifier')
        if type_node:
            class_name = self._get_node_text(type_node)
            class_exec = self._create_class_node(class_name)
            parent_exec.add_child(class_exec, EdgeType.INSTANCE_OF)
            
            # Analyze constructor arguments
            args_node = self._find_child_by_type(obj_creation, 'argument_list')
            if args_node:
                for arg in args_node.children:
                    if arg.type not in [',', '(', ')']:
                        arg_name = self._get_node_text(arg)
                        arg_exec = ExecutionNode(
                            name=arg_name,
                            category=NodeCategory.VARIABLE,
                            details="Constructor argument",
                            line_number=arg.start_point[0] + 1,
                            ts_node=arg
                        )
                        class_exec.add_child(arg_exec, EdgeType.ARGUMENT_OF)
    
    def _analyze_if_statement(self, if_stmt: Node, parent_exec: ExecutionNode):
        """Analyze if statement condition and branches"""
        condition = self._find_child_by_type(if_stmt, 'parenthesized_expression')
        if condition:
            cond_text = self._get_node_text(condition)
            cond_exec = ExecutionNode(
                name=f"if({cond_text})",
                category=NodeCategory.CONDITION,
                details="Conditional branch",
                line_number=condition.start_point[0] + 1,
                ts_node=condition
            )
            parent_exec.add_child(cond_exec, EdgeType.CONDITION)
            
            # Analyze condition content
            for method_call in self._find_all_descendants_by_type(condition, 'method_invocation'):
                self._analyze_method_invocation(method_call, cond_exec)
        
        # Analyze then block
        then_block = self._find_child_by_type(if_stmt, 'block')
        if then_block:
            self._analyze_block_execution(then_block, parent_exec)
        
        # Analyze else block if present
        else_stmt = self._find_child_by_type(if_stmt, 'else_clause')
        if else_stmt:
            else_block = self._find_child_by_type(else_stmt, 'block')
            if else_block:
                self._analyze_block_execution(else_block, parent_exec)
    
    def _analyze_assignment(self, assignment: Node, parent_exec: ExecutionNode):
        """Analyze assignment expressions"""
        if len(assignment.children) >= 3:
            left = assignment.children[0]
            right = assignment.children[2]  # Skip the '=' operator
            
            left_name = self._get_node_text(left)
            assign_exec = ExecutionNode(
                name=left_name,
                category=NodeCategory.ASSIGNMENT,
                details=f"Assigned: {self._get_node_text(right)[:30]}",
                line_number=assignment.start_point[0] + 1,
                ts_node=assignment
            )
            parent_exec.add_child(assign_exec, EdgeType.ASSIGNED_TO)
            
            # Analyze right side
            for method_call in self._find_all_descendants_by_type(right, 'method_invocation'):
                self._analyze_method_invocation(method_call, assign_exec)
    
    def _create_class_node(self, class_name: str) -> ExecutionNode:
        """Create a node for a class"""
        is_external = class_name not in self.class_info
        line_num = 1
        
        if not is_external:
            class_node = self.class_info[class_name]
            line_num = class_node.start_point[0] + 1
        
        class_exec = ExecutionNode(
            name=class_name,
            category=NodeCategory.CLASS,
            details="Class definition",
            line_number=line_num,
            is_external=is_external,
            ts_node=self.class_info.get(class_name)
        )
        
        # Add class members if defined locally
        if not is_external:
            self._analyze_class_members(self.class_info[class_name], class_exec)
        
        return class_exec
    
    def _analyze_class_members(self, class_node: Node, class_exec: ExecutionNode):
        """Analyze class fields and methods"""
        for child in class_node.children:
            if child.type == 'field_declaration':
                for declarator in self._find_all_children_by_type(child, 'variable_declarator'):
                    field_name = self._get_identifier_name(declarator)
                    if field_name:
                        field_type = self._get_field_type(child)
                        field_exec = ExecutionNode(
                            name=field_name,
                            category=NodeCategory.FIELD,
                            details=f"Type: {field_type}",
                            line_number=declarator.start_point[0] + 1,
                            ts_node=declarator
                        )
                        class_exec.add_child(field_exec, EdgeType.CLASS_MEMBER)
            
            elif child.type == 'method_declaration':
                method_name = self._get_identifier_name(child)
                if method_name:
                    method_exec = self._create_execution_node(child, NodeCategory.METHOD)
                    class_exec.add_child(method_exec, EdgeType.CLASS_MEMBER)
    
    def _create_execution_node(self, ts_node: Node, category: NodeCategory) -> ExecutionNode:
        """Create an ExecutionNode from a tree-sitter node"""
        name = self._get_identifier_name(ts_node) or "unknown"
        details = self._get_node_details(ts_node, category)
        
        return ExecutionNode(
            name=name,
            category=category,
            details=details,
            line_number=ts_node.start_point[0] + 1,
            ts_node=ts_node
        )
    
    def _get_node_details(self, node: Node, category: NodeCategory) -> str:
        """Get detailed information about a node"""
        if category == NodeCategory.METHOD:
            return_type = self._get_return_type(node)
            params = self._get_parameter_list(node)
            return f"Returns: {return_type}, Params: {params}"
        elif category == NodeCategory.CLASS:
            return "Class definition"
        else:
            return self._get_node_text(node)[:50]
    
    # Helper methods
    def _find_method_or_class(self, root: Node, name: str) -> Optional[Node]:
        """Find a method or class by name"""
        return self.method_info.get(name) or self.class_info.get(name)
    
    def _find_child_by_type(self, node: Node, target_type: str) -> Optional[Node]:
        """Find first child of specific type"""
        for child in node.children:
            if child.type == target_type:
                return child
        return None
    
    def _find_all_children_by_type(self, node: Node, target_type: str) -> List[Node]:
        """Find all children of specific type"""
        return [child for child in node.children if child.type == target_type]
    
    def _find_all_descendants_by_type(self, node: Node, target_type: str) -> List[Node]:
        """Find all descendants of specific type"""
        result = []
        if node.type == target_type:
            result.append(node)
        for child in node.children:
            result.extend(self._find_all_descendants_by_type(child, target_type))
        return result
    
    def _get_identifier_name(self, node: Node) -> Optional[str]:
        """Get identifier name from various node types"""
        if node.type == 'identifier':
            return self._get_node_text(node)
        
        # Look for identifier child
        for child in node.children:
            if child.type == 'identifier':
                return self._get_node_text(child)
        
        return None
    
    def _get_method_name_from_invocation(self, method_call: Node) -> Optional[str]:
        """Extract method name from method invocation"""
        # Handle chained calls like obj.method() or Class.method()
        text = self._get_node_text(method_call)
        if '(' in text:
            method_part = text.split('(')[0]
            if '.' in method_part:
                return method_part.split('.')[-1]
            return method_part
        return None
    
    def _get_parameter_type(self, param_node: Node) -> str:
        """Get parameter type"""
        type_node = self._find_child_by_type(param_node, 'type')
        return self._get_node_text(type_node) if type_node else "unknown"
    
    def _get_field_type(self, field_node: Node) -> str:
        """Get field type"""
        type_node = self._find_child_by_type(field_node, 'type')
        return self._get_node_text(type_node) if type_node else "unknown"
    
    def _get_return_type(self, method_node: Node) -> str:
        """Get method return type"""
        type_node = self._find_child_by_type(method_node, 'type')
        return self._get_node_text(type_node) if type_node else "void"
    
    def _get_parameter_list(self, method_node: Node) -> str:
        """Get method parameter list"""
        params_node = self._find_child_by_type(method_node, 'formal_parameters')
        return self._get_node_text(params_node) if params_node else "()"
    
    def _get_node_text(self, node: Node) -> str:
        """Get source text for a node"""
        if not node:
            return ""
        return self.source_code[node.start_byte:node.end_byte].decode('utf-8')
    
    def print_tree(self, node: ExecutionNode, indent: int = 0):
        """Print the execution tree"""
        prefix = "  " * indent
        arrow = "→ " if indent > 0 else ""
        print(f"{prefix}{arrow}{node}")
        
        for child in node.children:
            self.print_tree(child, indent + 1)

# Example usage with your provided code
def test_with_example():
    java_code = '''
package com.example;

import com.x.y.z.DefaultCalc;
import com.x.y.z.TransEvent;
import com.x.y.z.Reporter;
import java.util.List;
import java.util.Optional;
import java.util.stream.Stream;
import com.x.y.ProcessingService;

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
    }
}
    '''
    
    analyzer = JavaExecutionAnalyzer()
    tree = analyzer.build_execution_tree(java_code, "start")
    
    if tree:
        print("Execution Flow Analysis Tree for 'start' method:")
        print("=" * 60)
        analyzer.print_tree(tree)
    else:
        print("Could not find starting point 'start'")

if __name__ == "__main__":
    test_with_example()
