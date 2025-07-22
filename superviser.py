# agent/supervisor.py

import os
import operator
from typing import TypedDict, Annotated, Sequence, Optional, Dict
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_openai import AzureChatOpenAI

# Import the original worker functions
from agent.query_parser import parse_user_query
from agent.data_fetcher import fetch_and_process_data

# --- 1. Define the State for the Supervisor ---
# The state is now much simpler. It's just the conversation history.
# The agent will use this history to decide what to do next.
class SupervisorState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

# --- 2. Convert Worker Agents into Tools ---
# We wrap our existing agent functions with the `@tool` decorator.
# This makes them available for the supervisor LLM to call.

@tool
def query_parser_tool(user_query: str) -> str:
    """
    Use this tool first to parse and validate a user's initial query.
    It extracts key details like regulator, UTI, and the field to investigate.
    Input is the raw user query.
    Output is a success message with the extracted data or an error message.
    """
    print("--- SUPERVISOR: Calling Query Parser Tool ---")
    parsed_details, impacted_field, error = parse_user_query(user_query)
    
    if error:
        return f"Error during parsing: {error}"
    
    # On success, return a structured string for the LLM to process.
    # We combine the details into a single dictionary for the next step.
    success_data = {**parsed_details, "impacted_field": impacted_field}
    return f"Successfully parsed query. Details: {success_data}"

@tool
def data_fetcher_tool(parsed_query_details: Dict) -> str:
    """
    Use this tool AFTER the user's query has been successfully parsed.
    It takes the extracted details and fetches the corresponding trade data.
    Input is a dictionary containing all parsed details, including 'impacted_field'.
    Output is a success message with the fetched data or a failure message.
    """
    print("--- SUPERVISOR: Calling Data Fetcher Tool ---")
    
    # The LLM will pass the dictionary from the previous step's output.
    impacted_field = parsed_query_details.pop("impacted_field", None)
    
    success, data = fetch_and_process_data(parsed_query_details, impacted_field)
    
    if success:
        return f"Successfully fetched data: {data}"
    else:
        return f"Failed to fetch data for UTI: {parsed_query_details.get('uti')}."

# --- 3. Build the Supervisor Agent Graph ---
def build_supervisor_graph():
    """Builds the intelligent, tool-calling supervisor agent."""
    
    # Define the tools the supervisor can use
    tools = [query_parser_tool, data_fetcher_tool]
    
    # Set up the agent model with the tools
    llm = AzureChatOpenAI(
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
        temperature=0
    )
    llm_with_tools = llm.bind_tools(tools)

    # Define the agent node: this calls the LLM to decide the next action
    def supervisor_agent_node(state: SupervisorState):
        print("--- SUPERVISOR: Deciding next action ---")
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    # Define the tool node: this executes the chosen tool
    tool_node = ToolNode(tools)

    # Define the routing logic
    def after_agent_router(state: SupervisorState):
        """Routes to the tool node if the agent decides to use a tool."""
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "call_tool"
        return END

    # Assemble the graph
    graph = StateGraph(SupervisorState)
    graph.add_node("agent", supervisor_agent_node)
    graph.add_node("call_tool", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        after_agent_router,
        {"call_tool": "call_tool", END: END}
    )
    graph.add_edge("call_tool", "agent") # Loop back to the agent after a tool is called

    return graph.compile()

# Create a single runnable instance for your application to use
supervisor_app = build_supervisor_graph()
