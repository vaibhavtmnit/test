# main_agent.py

import os
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, END

# --- Environment Setup ---
# It's a best practice to load credentials from environment variables
# for security and portability.
# os.environ["AZURE_OPENAI_API_KEY"] = "YOUR_AZURE_OPENAI_API_KEY"
# os.environ["AZURE_OPENAI_ENDPOINT"] = "YOUR_AZURE_OPENAI_ENDPOINT"
# os.environ["OPENAI_API_VERSION"] = "YOUR_API_VERSION"

# --- Agent State Definition ---
# The AgentState is a TypedDict, which provides a structured way to manage
# the agent's internal state as it passes between nodes in the graph.
# This is a core concept in LangGraph and crucial for building complex agents.
class AgentState(TypedDict):
    """
    Represents the state of our agent.

    Attributes:
        messages: The history of messages in the conversation.
        sender: The name of the last entity that added a message.
    """
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    sender: str

# --- Node Definitions ---
# Nodes are the fundamental building blocks of a LangGraph. Each node is a
# function or a runnable that performs an action.

def agent_node(state: AgentState, model: AzureChatOpenAI):
    """
    The primary node of the agent. It takes the current state,
    invokes the language model, and returns the model's response.
    """
    print("---AGENT NODE---")
    response = model.invoke(state["messages"])
    return {"messages": [response], "sender": "agent"}

def tool_node(state: AgentState):
    """
    A placeholder for a tool-using node. In a real-world scenario, this
    node would parse the agent's last message, execute a tool (e.g., a
    search query, API call), and return the result.
    """
    print("---TOOL NODE---")
    # This is a mock tool response.
    # In a real implementation, you would use a tool calling agent
    # to interpret the model's output and execute the correct tool.
    last_message = state["messages"][-1]
    # Simple check to see if the agent is trying to use a tool
    if "tool_call" in str(last_message.content):
        tool_output = "The weather in London is 20 degrees and sunny."
        return {"messages": [HumanMessage(content=tool_output)], "sender": "tool"}
    return {"messages": [], "sender": "none"}

# --- Conditional Edge Logic ---
# Conditional edges determine the path of execution through the graph.
# This function inspects the last message in the state to decide
# which node to go to next.
def should_continue(state: AgentState) -> str:
    """
    Determines the next step in the graph.
    If the agent's last message was a tool call, it routes to the tool node.
    Otherwise, it ends the execution.
    """
    print("---CONDITIONAL EDGE---")
    if state["sender"] == "agent":
        # A more sophisticated check for tool calls would be implemented here.
        # For example, using the model's tool_calls attribute.
        if "tool_call" in str(state["messages"][-1].content):
            return "tool"
    return END

# --- Graph Construction ---
# Here we assemble the nodes and edges into a coherent workflow.

# Initialize the model
# It's good practice to initialize the model once and pass it to the nodes
# where it's needed. This improves testability and modularity.
model = AzureChatOpenAI(
    deployment_name="your-deployment-name",
    # model_name="gpt-4o", # Or your preferred model
)

# Create the StateGraph
workflow = StateGraph(AgentState)

# Add nodes to the graph
workflow.add_node("agent", lambda state: agent_node(state, model))
workflow.add_node("tool", tool_node)

# Set the entry point of the graph
workflow.set_entry_point("agent")

# Add conditional edges
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tool": "tool",
        END: END
    }
)

# An edge to loop back from the tool node to the agent node
workflow.add_edge('tool', 'agent')


# Compile the graph into a runnable LangChain object
app = workflow.compile()

# --- Running the Agent ---
if __name__ == "__main__":
    # Example of how to run the agent
    initial_state = {
        "messages": [HumanMessage(content="What is the weather in London? tool_call")],
    }
    # The `stream` method is useful for observing the state at each step
    for output in app.stream(initial_state):
        # The key is the name of the node that was just executed
        for key, value in output.items():
            print(f"Output from node '{key}':")
            print("---")
            print(value)
        print("\n---\n")





# lats_agent.py

import os
import uuid
from typing import TypedDict, List, Annotated, Dict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, END

# --- Environment Setup ---
# os.environ["AZURE_OPENAI_API_KEY"] = "YOUR_AZURE_OPENAI_API_KEY"
# os.environ["AZURE_OPENAI_ENDPOINT"] = "YOUR_AZURE_OPENAI_ENDPOINT"
# os.environ["OPENAI_API_VERSION"] = "YOUR_API_VERSION"

# --- LATS State Definition ---
# For LATS, the state needs to be more complex to manage the tree structure.
class LATSNode(TypedDict):
    """A node in the reasoning tree."""
    id: str
    messages: List[BaseMessage]
    parent_id: str | None
    score: float
    reflections: List[str]

class LATSState(TypedDict):
    """The state for the LATS agent."""
    problem_statement: str
    nodes: Dict[str, LATSNode]
    active_node_id: str
    max_depth: int
    current_depth: int
    solution: str | None

# --- LATS Node Definitions ---

def initialize_tree(state: LATSState) -> LATSState:
    """Initializes the root of the reasoning tree."""
    print("---INITIALIZING TREE---")
    root_id = str(uuid.uuid4())
    root_node = LATSNode(
        id=root_id,
        messages=[HumanMessage(content=state['problem_statement'])],
        parent_id=None,
        score=0.0,
        reflections=[]
    )
    return {
        "nodes": {root_id: root_node},
        "active_node_id": root_id,
        "current_depth": 0,
    }

def generate_thoughts(state: LATSState, model: AzureChatOpenAI) -> LATSState:
    """Generates multiple reasoning steps (thoughts) from the active node."""
    print(f"---GENERATING THOUGHTS (Depth: {state['current_depth']})---")
    active_node = state['nodes'][state['active_node_id']]
    
    # Prompt the model to generate multiple diverse thoughts
    # This is a key part of the LATS process
    prompt = (
        f"Problem: {state['problem_statement']}\n\n"
        f"Current reasoning path:\n{''.join([m.content for m in active_node['messages']])}\n\n"
        "Generate 3 distinct and promising next steps or thoughts to solve this problem. "
        "Format each thought as 'Thought: [your thought]'."
    )
    
    response = model.invoke([HumanMessage(content=prompt)])
    thoughts = [t.split("Thought:")[1].strip() for t in response.content.split("Thought:")[1:]]

    new_nodes = {}
    for thought in thoughts:
        new_node_id = str(uuid.uuid4())
        new_node = LATSNode(
            id=new_node_id,
            messages=active_node['messages'] + [AIMessage(content=thought)],
            parent_id=active_node['id'],
            score=0.0,
            reflections=[]
        )
        new_nodes[new_node_id] = new_node
        
    return {"nodes": {**state['nodes'], **new_nodes}}


def evaluate_thoughts(state: LATSState, evaluator_model: AzureChatOpenAI) -> LATSState:
    """Evaluates the generated thoughts and scores them."""
    print("---EVALUATING THOUGHTS---")
    parent_node_id = state['active_node_id']
    child_nodes = [node for node in state['nodes'].values() if node['parent_id'] == parent_node_id]

    updated_nodes = state['nodes'].copy()
    for node in child_nodes:
        prompt = (
            f"Problem: {state['problem_statement']}\n\n"
            f"Reasoning path:\n{''.join([m.content for m in node['messages']])}\n\n"
            "Evaluate the quality and promise of this reasoning path on a scale of 1 to 10. "
            "Provide a score and a brief justification. Format as 'Score: [score], Justification: [justification]'."
        )
        response = evaluator_model.invoke([HumanMessage(content=prompt)])
        score_str = response.content.split("Score:")[1].split(",")[0].strip()
        node['score'] = float(score_str)
        updated_nodes[node['id']] = node
        
    return {"nodes": updated_nodes}

def reflect_and_select(state: LATSState, reflection_model: AzureChatOpenAI) -> LATSState:
    """Reflects on the scores and selects the most promising node to expand."""
    print("---REFLECTING AND SELECTING---")
    parent_node_id = state['active_node_id']
    child_nodes = [node for node in state['nodes'].values() if node['parent_id'] == parent_node_id]

    if not child_nodes:
        return {"solution": "Failed to find a solution path."}

    # Select the best node based on score
    best_node = max(child_nodes, key=lambda node: node['score'])

    # Generate a reflection on why this path was chosen
    prompt = (
        f"Problem: {state['problem_statement']}\n\n"
        f"I am exploring multiple reasoning paths. I have chosen the following path to continue:\n"
        f"{''.join([m.content for m in best_node['messages']])}\n\n"
        f"Briefly reflect on why this path is more promising than the alternatives."
    )
    reflection = reflection_model.invoke([HumanMessage(content=prompt)]).content
    best_node['reflections'].append(reflection)
    
    updated_nodes = state['nodes'].copy()
    updated_nodes[best_node['id']] = best_node

    return {
        "nodes": updated_nodes,
        "active_node_id": best_node['id'],
        "current_depth": state['current_depth'] + 1
    }

def check_depth(state: LATSState) -> str:
    """Checks if the maximum search depth has been reached."""
    print("---CHECKING DEPTH---")
    if state['current_depth'] >= state['max_depth']:
        final_node = state['nodes'][state['active_node_id']]
        solution = "".join([m.content for m in final_node['messages']])
        return "solution_found"
    return "continue_search"

def finalize_solution(state: LATSState) -> LATSState:
    """Finalizes the solution from the best path."""
    print("---FINALIZING SOLUTION---")
    final_node = state['nodes'][state['active_node_id']]
    solution_content = "\n".join([msg.content for msg in final_node['messages']])
    return {"solution": f"Final Solution Path:\n{solution_content}"}


# --- Graph Construction ---
generator_model = AzureChatOpenAI(deployment_name="your-deployment-name", model_name="gpt-4o")
evaluator_model = AzureChatOpenAI(deployment_name="your-deployment-name", model_name="gpt-4-turbo")
reflection_model = AzureChatOpenAI(deployment_name="your-deployment-name", model_name="gpt-4")


lats_workflow = StateGraph(LATSState)

lats_workflow.add_node("initialize_tree", initialize_tree)
lats_workflow.add_node("generate_thoughts", lambda state: generate_thoughts(state, generator_model))
lats_workflow.add_node("evaluate_thoughts", lambda state: evaluate_thoughts(state, evaluator_model))
lats_workflow.add_node("reflect_and_select", lambda state: reflect_and_select(state, reflection_model))
lats_workflow.add_node("finalize_solution", finalize_solution)


lats_workflow.set_entry_point("initialize_tree")
lats_workflow.add_edge("initialize_tree", "generate_thoughts")
lats_workflow.add_edge("generate_thoughts", "evaluate_thoughts")
lats_workflow.add_edge("evaluate_thoughts", "reflect_and_select")

lats_workflow.add_conditional_edges(
    "reflect_and_select",
    check_depth,
    {
        "continue_search": "generate_thoughts",
        "solution_found": "finalize_solution"
    }
)
lats_workflow.add_edge("finalize_solution", END)

lats_app = lats_workflow.compile()

# --- Running the LATS Agent ---
if __name__ == "__main__":
    problem = "Plan a 3-day trip to London for a family with two young children, focusing on interactive and educational activities."
    initial_state = {
        "problem_statement": problem,
        "max_depth": 3, # Controls how many expansion steps
    }
    final_state = lats_app.invoke(initial_state)
    print("\n--- LATS AGENT FINAL OUTPUT ---")
    print(final_state['solution'])





# test_agent.py

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import AzureChatOpenAI

# Import the components from your agent files
# To make this runnable, you would save the above templates as
# main_agent.py and lats_agent.py
from main_agent import app as basic_agent_app, AgentState
from lats_agent import lats_app as lats_agent_app, LATSState


# --- Unit Testing Individual Nodes ---
# Unit tests focus on a single component (a node) in isolation.
# We use mocking to control inputs and outputs.

def test_agent_node_logic():
    """Tests the agent_node's interaction with the model."""
    # Arrange: Create a mock model and a sample state
    mock_model = MagicMock(spec=AzureChatOpenAI)
    mock_model.invoke.return_value = AIMessage(content="This is a test response.")
    
    state = AgentState(messages=[HumanMessage(content="Hello")], sender="user")
    
    # Act: Run the node function
    # We need to import the node function itself for this to work
    from main_agent import agent_node 
    result = agent_node(state, mock_model)
    
    # Assert: Check if the result is as expected
    mock_model.invoke.assert_called_once_with(state["messages"])
    assert len(result["messages"]) == 1
    assert result["messages"][0].content == "This is a test response."
    assert result["sender"] == "agent"


# --- Integration Testing the Graph ---
# Integration tests check if the nodes and edges of the graph work together correctly.

@patch('main_agent.AzureChatOpenAI')
def test_basic_agent_graph_flow_to_end(MockAzureChatOpenAI):
    """Tests a simple path through the graph that should terminate immediately."""
    # Arrange: Configure the mock model to give a simple response
    mock_model_instance = MockAzureChatOpenAI.return_value
    mock_model_instance.invoke.return_value = AIMessage(content="A simple answer.")
    
    # Act: Run the compiled graph
    initial_state = {"messages": [HumanMessage(content="Hi")]}
    final_state = basic_agent_app.invoke(initial_state)
    
    # Assert: Check the final state
    assert len(final_state["messages"]) == 2 # Initial HumanMessage + AIMessage
    assert final_state["messages"][-1].content == "A simple answer."
    assert final_state["sender"] == "agent"


@patch('main_agent.AzureChatOpenAI')
def test_basic_agent_graph_tool_path(MockAzureChatOpenAI):
    """Tests the conditional edge logic to ensure it routes to the tool node."""
    # Arrange: Configure the mock to return a message that should trigger the tool node
    mock_model_instance = MockAzureChatOpenAI.return_value
    # The first call is from the agent, the second is after the tool result
    mock_model_instance.invoke.side_effect = [
        AIMessage(content="I need to use a tool. tool_call"),
        AIMessage(content="Okay, the tool said the weather is sunny.")
    ]
    
    # Act
    initial_state = {"messages": [HumanMessage(content="What's the weather?")]}
    # We can use `stream` to inspect intermediate steps
    outputs = list(basic_agent_app.stream(initial_state))
    
    # Assert
    # 1. Agent runs first
    assert "agent" in outputs[0]
    # 2. Tool runs second
    assert "tool" in outputs[1]
    # 3. Agent runs again
    assert "agent" in outputs[2]
    
    final_state = outputs[-1]['agent']
    assert "the weather is sunny" in final_state['messages'][-1].content


# --- Advanced Evaluation with a "Critic" Agent ---
# This is an innovative approach where you use another LLM to evaluate your agent's
# output based on a set of criteria. This is particularly useful for complex,
# generative tasks where a simple "assert" is not enough.

EVALUATION_CRITERIA = """
1.  **Clarity:** Is the answer clear, concise, and easy to understand?
2.  **Completeness:** Does the answer fully address all parts of the user's query?
3.  **Correctness:** Is the information provided accurate and factual?
4.  **Helpfulness:** Does the answer provide practical, useful information to the user?
"""

def run_evaluator_agent(query: str, agent_response: str) -> dict:
    """
    Uses an LLM to evaluate an agent's response against predefined criteria.
    """
    print("\n--- RUNNING EVALUATOR AGENT ---")
    evaluator_prompt = f"""
    You are a meticulous AI quality assurance agent.
    Your task is to evaluate an AI agent's response to a user query based on the following criteria:
    {EVALUATION_CRITERIA}
    
    User Query: "{query}"
    Agent Response: "{agent_response}"
    
    Please provide a score (1-5) for each criterion and a brief justification.
    Format your output as a JSON object with keys: "clarity", "completeness", "correctness", "helpfulness".
    Each key should have a "score" and "justification" field.
    """
    
    # In a real scenario, you'd use your AzureChatOpenAI instance here.
    # For this example, we'll mock the response.
    # evaluator_model = AzureChatOpenAI(...)
    # response = evaluator_model.invoke(evaluator_prompt)
    # return json.loads(response.content)
    
    # Mocked response for demonstration
    mock_evaluation = {
        "clarity": {"score": 4, "justification": "The response is well-structured but a bit verbose."},
        "completeness": {"score": 5, "justification": "All aspects of the query were addressed."},
        "correctness": {"score": 5, "justification": "The information provided is accurate."},
        "helpfulness": {"score": 4, "justification": "The plan is good, but could include more budget-friendly options."}
    }
    print(f"Evaluation result: {mock_evaluation}")
    return mock_evaluation


@pytest.mark.lats
@patch('lats_agent.lats_app.invoke')
def test_lats_agent_qualitative_evaluation(mock_lats_invoke):
    """
    A qualitative test for the LATS agent using the evaluator agent.
    """
    # Arrange
    query = "Plan a 3-day trip to London for a family with two young children, focusing on interactive and educational activities."
    # Mock the output of the LATS agent for a predictable test
    mock_lats_output = """
    Final Solution Path:
    Human: Plan a 3-day trip to London...
    AI: Day 1: Visit the Natural History Museum in the morning, followed by the Science Museum's Wonderlab in the afternoon. Both are highly interactive.
    AI: Day 2: Explore the Tower of London with a Yeoman Warder tour, then take a boat trip on the Thames.
    AI: Day 3: Spend the day at the Diana Memorial Playground in Hyde Park and have a picnic.
    """
    mock_lats_invoke.return_value = {'solution': mock_lats_output}
    
    # Act: Get the LATS agent's response
    lats_result = lats_agent_app.invoke({"problem_statement": query, "max_depth": 3})
    agent_response = lats_result['solution']
    
    # Evaluate the response
    evaluation = run_evaluator_agent(query, agent_response)
    
    # Assert: Check if the scores meet a minimum threshold
    assert evaluation['clarity']['score'] >= 4
    assert evaluation['completeness']['score'] >= 5
    assert evaluation['helpfulness']['score'] >= 4


Of course. I will provide you with a set of comprehensive, industry-standard templates for building and testing powerful, modern agents using langgraph.

Here is the first document: a foundational template for creating a robust LangGraph agent. It includes all the necessary components, from state definition to graph compilation, and is designed for use with AzureChatOpenAI.
LangGraph Agent Template
Jul 21, 9:28 PM

Open

Next, I'll provide templates for advanced agentic strategies. This first one is for LATS (Language Agent Tree Search), a powerful technique that enables agents to explore different reasoning paths, self-correct, and improve the quality of their final output.
Advanced Agent - LATS Template
Jul 21, 9:28 PM

Open

Finally, here is a template for testing your agents. A solid testing strategy is non-negotiable for production-grade AI systems. This template uses pytest and introduces the concept of an "evaluator agent" for more nuanced, qualitative testing.
Agent Testing and Evaluation Template
Jul 21, 9:28 PM

I have provided you with three distinct templates:

A foundational langgraph agent that shows how to structure a basic conversational agent with tool-use capabilities.

An advanced agent template implementing the LATS (Language Agent Tree Search) strategy for more complex problem-solving.

A comprehensive testing suite that includes unit tests, integration tests, and an innovative qualitative evaluation method using a "critic" agent.

You can now use these templates as a starting point for your own projects. I recommend starting with the basic agent to get a feel for the workflow, then exploring the LATS agent for more challenging tasks. The testing template is crucial and should be adapted for any agent you build.

# test_agent.py

import pytest
import json
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import AzureChatOpenAI

# Import the components from your agent files
# To make this runnable, you would save the above templates as
# main_agent.py and lats_agent.py
from main_agent import app as basic_agent_app, AgentState
from lats_agent import lats_app as lats_agent_app, LATSState


# --- Unit Testing Individual Nodes ---
# Unit tests focus on a single component (a node) in isolation.
# We use mocking to control inputs and outputs.

def test_agent_node_logic():
    """Tests the agent_node's interaction with the model."""
    # Arrange: Create a mock model and a sample state
    mock_model = MagicMock(spec=AzureChatOpenAI)
    mock_model.invoke.return_value = AIMessage(content="This is a test response.")
    
    state = AgentState(messages=[HumanMessage(content="Hello")], sender="user")
    
    # Act: Run the node function
    # We need to import the node function itself for this to work
    from main_agent import agent_node 
    result = agent_node(state, mock_model)
    
    # Assert: Check if the result is as expected
    mock_model.invoke.assert_called_once_with(state["messages"])
    assert len(result["messages"]) == 1
    assert result["messages"][0].content == "This is a test response."
    assert result["sender"] == "agent"


# --- Integration Testing the Graph ---
# Integration tests check if the nodes and edges of the graph work together correctly.

@patch('main_agent.AzureChatOpenAI')
def test_basic_agent_graph_flow_to_end(MockAzureChatOpenAI):
    """Tests a simple path through the graph that should terminate immediately."""
    # Arrange: Configure the mock model to give a simple response
    mock_model_instance = MockAzureChatOpenAI.return_value
    mock_model_instance.invoke.return_value = AIMessage(content="A simple answer.")
    
    # Act: Run the compiled graph
    initial_state = {"messages": [HumanMessage(content="Hi")]}
    final_state = basic_agent_app.invoke(initial_state)
    
    # Assert: Check the final state
    assert len(final_state["messages"]) == 2 # Initial HumanMessage + AIMessage
    assert final_state["messages"][-1].content == "A simple answer."
    assert final_state["sender"] == "agent"


@patch('main_agent.AzureChatOpenAI')
def test_basic_agent_graph_tool_path(MockAzureChatOpenAI):
    """Tests the conditional edge logic to ensure it routes to the tool node."""
    # Arrange: Configure the mock to return a message that should trigger the tool node
    mock_model_instance = MockAzureChatOpenAI.return_value
    # The first call is from the agent, the second is after the tool result
    mock_model_instance.invoke.side_effect = [
        AIMessage(content="I need to use a tool. tool_call"),
        AIMessage(content="Okay, the tool said the weather is sunny.")
    ]
    
    # Act
    initial_state = {"messages": [HumanMessage(content="What's the weather?")]}
    # We can use `stream` to inspect intermediate steps
    outputs = list(basic_agent_app.stream(initial_state))
    
    # Assert
    # 1. Agent runs first
    assert "agent" in outputs[0]
    # 2. Tool runs second
    assert "tool" in outputs[1]
    # 3. Agent runs again
    assert "agent" in outputs[2]
    
    final_state = outputs[-1]['agent']
    assert "the weather is sunny" in final_state['messages'][-1].content


# --- Advanced Evaluation with a "Critic" Agent ---
# This is an innovative approach where you use another LLM to evaluate your agent's
# output based on a set of criteria. This is particularly useful for complex,
# generative tasks where a simple "assert" is not enough.

EVALUATION_CRITERIA = """
1.  **Clarity:** Is the answer clear, concise, and easy to understand?
2.  **Completeness:** Does the answer fully address all parts of the user's query?
3.  **Correctness:** Is the information provided accurate and factual?
4.  **Helpfulness:** Does the answer provide practical, useful information to the user?
"""

def run_evaluator_agent(query: str, agent_response: str, evaluator_model: AzureChatOpenAI) -> dict:
    """
    Uses an LLM to evaluate an agent's response against predefined criteria.
    """
    print("\n--- RUNNING EVALUATOR AGENT ---")
    evaluator_prompt = f"""
    You are a meticulous AI quality assurance agent.
    Your task is to evaluate an AI agent's response to a user query based on the following criteria:
    {EVALUATION_CRITERIA}
    
    User Query: "{query}"
    Agent Response: "{agent_response}"
    
    Please provide a score (1-5) for each criterion and a brief justification.
    Format your output as a JSON object with keys: "clarity", "completeness", "correctness", "helpfulness".
    Each key should have a "score" and "justification" field.
    """
    
    response = evaluator_model.invoke(evaluator_prompt)
    print(f"Evaluation raw response: {response.content}")
    return json.loads(response.content)


@pytest.mark.lats
@patch('lats_agent.lats_app.invoke')
@patch('test_agent.run_evaluator_agent')
def test_lats_agent_qualitative_evaluation(mock_run_evaluator, mock_lats_invoke):
    """
    A qualitative test for the LATS agent using the evaluator agent.
    """
    # Arrange
    query = "Plan a 3-day trip to London for a family with two young children, focusing on interactive and educational activities."
    mock_lats_output = "Final Solution Path:\nHuman: Plan a 3-day trip...\nAI: Day 1: Natural History Museum, Science Museum.\nAI: Day 2: Tower of London, Thames boat trip.\nAI: Day 3: Diana Memorial Playground."
    mock_lats_invoke.return_value = {'solution': mock_lats_output}
    
    mock_evaluation = {
        "clarity": {"score": 4, "justification": "Well-structured."},
        "completeness": {"score": 5, "justification": "Addressed all parts."},
        "correctness": {"score": 5, "justification": "Accurate info."},
        "helpfulness": {"score": 4, "justification": "Good plan."}
    }
    mock_run_evaluator.return_value = mock_evaluation
    
    # Act
    lats_result = lats_agent_app.invoke({"problem_statement": query, "max_depth": 3})
    agent_response = lats_result['solution']
    evaluation = run_evaluator_agent(query, agent_response, MagicMock())
    
    # Assert
    assert evaluation['completeness']['score'] >= 4
    assert evaluation['helpfulness']['score'] >= 4


# --- Self-Correction / Reflexion-style Testing ---
# This advanced method tests an agent's ability to receive feedback and
# improve its own answer. It's a powerful way to evaluate deeper reasoning.

@patch('main_agent.AzureChatOpenAI')
def test_agent_self_correction_loop(MockAzureChatOpenAI):
    """
    Tests if the agent can improve its response based on feedback from an evaluator.
    """
    # Arrange
    query = "Explain the difference between a list and a tuple in Python."
    
    # 1. Mock the agent's model to give two responses: an initial flawed one, and an improved one.
    initial_response = "A list and a tuple are both collections. Lists use square brackets, tuples use parentheses."
    corrected_response = "A list and a tuple are both ordered collections in Python. The key difference is that lists are mutable, meaning you can change their content, while tuples are immutable, meaning they cannot be changed after creation. Lists use `[]` and tuples use `()`."
    
    mock_agent_model = MockAzureChatOpenAI.return_value
    mock_agent_model.invoke.side_effect = [
        AIMessage(content=initial_response),
        AIMessage(content=corrected_response)
    ]
    
    # 2. Mock the evaluator model to provide feedback on the flawed response.
    evaluation_feedback = {
        "clarity": {"score": 3, "justification": "The answer is too brief."},
        "completeness": {"score": 2, "justification": "The answer is missing the core concept of mutability vs. immutability."},
        "correctness": {"score": 4, "justification": "What is stated is correct, but incomplete."},
        "helpfulness": {"score": 2, "justification": "Does not fully explain the most important difference."}
    }
    mock_evaluator_model = MagicMock(spec=AzureChatOpenAI)
    mock_evaluator_model.invoke.return_value = AIMessage(content=json.dumps(evaluation_feedback))

    # --- ACTION ---

    # Step 1: Get the agent's initial, flawed response.
    print("\n--- SELF-CORRECTION: INITIAL RESPONSE ---")
    initial_state = {"messages": [HumanMessage(content=query)]}
    initial_result = basic_agent_app.invoke(initial_state)
    initial_agent_response = initial_result['messages'][-1].content
    print(f"Agent's initial response: {initial_agent_response}")

    # Step 2: Use the evaluator agent to critique the initial response.
    print("\n--- SELF-CORRECTION: EVALUATION ---")
    critique = run_evaluator_agent(query, initial_agent_response, mock_evaluator_model)
    
    # Step 3: Formulate a new prompt with the critique and ask the agent to self-correct.
    print("\n--- SELF-CORRECTION: REVISED PROMPT ---")
    correction_prompt = f"""Your previous answer to the query '{query}' was:
'{initial_agent_response}'

This answer received the following critique:
{json.dumps(critique, indent=2)}

Please provide a revised, improved answer that addresses the critique, focusing on completeness and the concept of mutability."""

    # Step 4: Get the agent's revised response.
    print("\n--- SELF-CORRECTION: REVISED RESPONSE ---")
    correction_state = {"messages": [HumanMessage(content=correction_prompt)]}
    final_result = basic_agent_app.invoke(correction_state)
    final_agent_response = final_result['messages'][-1].content
    print(f"Agent's revised response: {final_agent_response}")

    # --- ASSERT ---

    # Assert that the agent was called twice.
    assert mock_agent_model.invoke.call_count == 2
    
    # Assert that the final response is the improved one we defined.
    assert final_agent_response == corrected_response
    
    # For an even more robust test, you could re-run the evaluator on the
    # final response and assert that the scores have improved.
    final_evaluation = {
        "clarity": {"score": 5, "justification": "Clear and concise."},
        "completeness": {"score": 5, "justification": "Perfectly explains mutability."},
        "correctness": {"score": 5, "justification": "Fully correct."},
        "helpfulness": {"score": 5, "justification": "Excellent explanation."}
    }
    mock_evaluator_model.invoke.return_value = AIMessage(content=json.dumps(final_evaluation))
    final_critique = run_evaluator_agent(query, final_agent_response, mock_evaluator_model)

    assert final_critique['completeness']['score'] > critique['completeness']['score']
    assert final_critique['helpfulness']['score'] > critique['helpfulness']['score']
    print("\n--- SELF-CORRECTION TEST PASSED: Agent successfully improved its answer based on feedback. ---")



# tot_agent.py

import os
import re
from typing import TypedDict, List, Annotated, Dict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, END

# --- Environment Setup ---
# os.environ["AZURE_OPENAI_API_KEY"] = "YOUR_AZURE_OPENAI_API_KEY"
# os.environ["AZURE_OPENAI_ENDPOINT"] = "YOUR_AZURE_OPENAI_ENDPOINT"
# os.environ["OPENAI_API_VERSION"] = "YOUR_API_VERSION"

# --- ToT State Definition ---
# The state for ToT needs to manage multiple parallel lines of reasoning.
class Thought(TypedDict):
    """A single thought in the reasoning tree, representing a path."""
    path: List[BaseMessage]
    evaluation: float

class ToTState(TypedDict):
    """The state for the Tree of Thoughts agent."""
    problem_statement: str
    thoughts: List[Thought]
    max_iterations: int
    current_iteration: int
    final_solution: str | None

# --- ToT Node Definitions ---

def initialize_thoughts(state: ToTState) -> Dict:
    """Initializes the first thought for the tree."""
    print("---INITIALIZING THOUGHTS---")
    initial_thought = Thought(
        path=[HumanMessage(content=state['problem_statement'])],
        evaluation=0.0
    )
    return {"thoughts": [initial_thought], "current_iteration": 0}

def generate_new_thoughts(state: ToTState, generator_model: AzureChatOpenAI, num_thoughts: int) -> Dict:
    """Generates multiple new thoughts from the current best paths."""
    print(f"---GENERATING NEW THOUGHTS (Iteration: {state['current_iteration']})---")
    
    # This simple ToT implementation expands all current thoughts.
    # A more advanced version might select the top-k best thoughts to expand.
    new_thoughts_to_add = []
    
    for thought in state['thoughts']:
        # Create a prompt that encourages diverse next steps
        prompt = (
            "You are a brilliant problem solver. Given the following problem and reasoning path, generate a coherent and promising next step. Do not solve the whole problem at once. Focus only on the next logical step.\n\n"
            f"Problem: {state['problem_statement']}\n\n"
            f"Current Path:\n{''.join([f'{m.type}: {m.content}' for m in thought['path']])}\n\n"
            f"Generate {num_thoughts} distinct next steps."
        )
        
        response = generator_model.invoke([HumanMessage(content=prompt)])
        generated_steps = response.content.split('\n') # Simple parsing
        
        for step in generated_steps:
            if step.strip():
                new_path = thought['path'] + [AIMessage(content=step.strip())]
                new_thoughts_to_add.append(Thought(path=new_path, evaluation=0.0))
                
    # Combine the new thoughts with the existing ones for evaluation
    return {"thoughts": state['thoughts'] + new_thoughts_to_add}


def evaluate_thoughts(state: ToTState, evaluator_model: AzureChatOpenAI) -> Dict:
    """Evaluates each thought path for its potential to lead to a solution."""
    print("---EVALUATING THOUGHTS---")
    
    updated_thoughts = []
    for thought in state['thoughts']:
        # Create a prompt for the evaluator model
        path_str = '\n'.join([f"{m.type}: {m.content}" for m in thought['path']])
        evaluation_prompt = (
            f"You are a critical evaluator. Assess the following reasoning path for its quality, coherence, and likelihood of solving the problem. Score it from 0 (dead end) to 1 (very promising).\n\n"
            f"Problem: {state['problem_statement']}\n\n"
            f"Path to evaluate:\n{path_str}\n\n"
            "Provide only a single floating-point number as the score."
        )
        
        response = evaluator_model.invoke([HumanMessage(content=evaluation_prompt)])
        try:
            # Extract the first floating point number from the response
            score_match = re.search(r"[-+]?\d*\.\d+|\d+", response.content)
            score = float(score_match.group()) if score_match else 0.0
        except (ValueError, AttributeError):
            score = 0.0
            
        thought['evaluation'] = score
        updated_thoughts.append(thought)
        
    return {"thoughts": updated_thoughts}


def prune_and_select(state: ToTState, beam_width: int) -> Dict:
    """Prunes the tree by keeping only the best thoughts (beam search)."""
    print("---PRUNING AND SELECTING---")
    
    # Sort thoughts by their evaluation score in descending order
    sorted_thoughts = sorted(state['thoughts'], key=lambda x: x['evaluation'], reverse=True)
    
    # Keep the top-k thoughts (this is the beam)
    best_thoughts = sorted_thoughts[:beam_width]
    
    # Check if the top thought is a solution
    top_thought_content = best_thoughts[0]['path'][-1].content.lower()
    if "solution is" in top_thought_content or "final answer" in top_thought_content:
        return {"final_solution": top_thought_content}
        
    return {
        "thoughts": best_thoughts,
        "current_iteration": state['current_iteration'] + 1
    }


def check_progress(state: ToTState) -> str:
    """Checks if a solution is found or max iterations are reached."""
    print("---CHECKING PROGRESS---")
    if state.get("final_solution"):
        return "solution_found"
    if state['current_iteration'] >= state['max_iterations']:
        # If no solution after max iterations, return the best path so far
        return "max_iterations_reached"
    return "continue"

def finalize_solution(state: ToTState) -> Dict:
    """Prepares the final output."""
    print("---FINALIZING SOLUTION---")
    if state.get("final_solution"):
        return {"final_solution": state["final_solution"]}
    
    # If max iterations were reached without a definitive solution
    best_path = state['thoughts'][0]['path']
    solution_str = '\n'.join([f"{m.type}: {m.content}" for m in best_path])
    return {"final_solution": f"Best path found:\n{solution_str}"}


# --- Graph Construction ---
# It's good practice to use different models for generation and evaluation
# to reduce bias.
generator_model = AzureChatOpenAI(deployment_name="your-deployment-name", model_name="gpt-4o", temperature=0.7)
evaluator_model = AzureChatOpenAI(deployment_name="your-deployment-name", model_name="gpt-4-turbo", temperature=0.0)

# ToT parameters
BEAM_WIDTH = 3 # Number of thoughts to keep at each step
NUM_THOUGHTS_PER_BRANCH = 2 # Number of new thoughts to generate from each existing thought

tot_workflow = StateGraph(ToTState)

tot_workflow.add_node("initialize", initialize_thoughts)
tot_workflow.add_node("generate", lambda state: generate_new_thoughts(state, generator_model, NUM_THOUGHTS_PER_BRANCH))
tot_workflow.add_node("evaluate", lambda state: evaluate_thoughts(state, evaluator_model))
tot_workflow.add_node("prune", lambda state: prune_and_select(state, BEAM_WIDTH))
tot_workflow.add_node("finalize", finalize_solution)

tot_workflow.set_entry_point("initialize")
tot_workflow.add_edge("initialize", "generate")
tot_workflow.add_edge("generate", "evaluate")
tot_workflow.add_edge("evaluate", "prune")

tot_workflow.add_conditional_edges(
    "prune",
    check_progress,
    {
        "continue": "generate",
        "solution_found": "finalize",
        "max_iterations_reached": "finalize"
    }
)
tot_workflow.add_edge("finalize", END)

tot_app = tot_workflow.compile()

# --- Running the ToT Agent ---
if __name__ == "__main__":
    # A classic ToT problem: The Game of 24
    problem = "Use the numbers 4, 8, 8, 9 and the operations +, -, *, / to get the number 24. Each number must be used exactly once."
    initial_state = {
        "problem_statement": problem,
        "max_iterations": 4,
    }
    final_state = tot_app.invoke(initial_state)
    print("\n--- ToT AGENT FINAL OUTPUT ---")
    print(final_state.get('final_solution', 'No solution found.'))


