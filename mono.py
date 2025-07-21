# supervisor_setup.py

import os
import operator
from typing import TypedDict, Annotated, Sequence

from langchain_core.messages import BaseMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from tools.regulatory_tools import get_regulatory_fields # We still use our external tool

# --- Pydantic Model for Structured Data ---
# This helps the agent understand what information to extract.
class TradeDetails(BaseModel):
    """The structured details of a trade for investigation."""
    uti: str = Field(..., description="The Unique Transaction Identifier (UTI) of the trade.")
    field_to_investigate: str = Field(..., description="The specific field the user wants to investigate.")
    regulator: str = Field(..., description="The relevant regulator, e.g., EMIR, ASIC.")

# --- Define the Agent's State ---
# This is much simpler now. It's just the history of messages.
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

# --- Define the Agent's Tools ---
# We give the agent functions it can decide to call.

# Tool #1: The regulatory fields tool from our other file.
# We wrap it with the @tool decorator so the agent can use it.
@tool
def get_regulatory_reporting_fields(regulator: str) -> list[str]:
    """
    Retrieves the list of mandatory reporting fields for a given financial regulator.
    Use this to validate if a field name provided by the user is valid for a regulator.
    """
    return get_regulatory_fields(regulator)

# Tool #2: A tool to "finalize" the confirmed details.
@tool
def confirm_trade_details_for_investigation(trade_details: TradeDetails) -> str:
    """
    Call this function ONLY when you have successfully extracted and confirmed all required details
    (UTI, field_to_investigate, and regulator) with the user.
    This signals that the validation process is complete.
    """
    return f"Confirmed. The investigation for UTI {trade_details.uti} on field '{trade_details.field_to_investigate}' under {trade_details.regulator} is ready to proceed."


# --- Build the Agent ---
def build_supervisor_graph():
    """
    Builds a stateful, conversational agent that can use tools.
    """
    
    # 1. Define the LLM and Tools
    llm = AzureChatOpenAI(
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
        temperature=0
    )
    tools = [get_regulatory_reporting_fields, confirm_trade_details_for_investigation]
    
    # 2. Bind the tools to the LLM. This creates the "agent" component.
    llm_with_tools = llm.bind_tools(tools)
    
    # 3. Define the graph nodes
    # The 'agent' node will be the LLM with tools bound
    def agent_node(state):
        """Calls the LLM to decide the next action or respond to the user."""
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    # The 'tool' node will execute the tools chosen by the agent
    tool_node = ToolNode(tools)

    # 4. Define the routing logic
    def should_continue(state):
        """Determines whether to continue with a tool call or end the turn."""
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "tools"  # Route to the tool node
        return "end"  # End the conversation for this turn

    # 5. Assemble the graph
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END}
    )
    graph.add_edge("tools", "agent") # After executing tools, loop back to the agent

    # 6. Compile the graph
    # The checkpointer is now attached at runtime in the main application file,
    # not during compilation.
    return graph.compile()
