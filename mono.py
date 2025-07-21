# supervisor_setup.py

import os
import operator
from typing import TypedDict, Annotated, Sequence

from langchain_core.messages import BaseMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_tool_calling_agent

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
    
    # 2. Create the Agent using a pre-built helper
    # This function wires up the LLM, tools, and memory for us.
    # It will automatically create a graph that reasons, calls tools, and responds.
    system_prompt = (
        "You are an expert assistant for financial regulatory reporting. Your job is to interact with the user to get all the necessary details for a trade investigation.\n"
        "You must collect and confirm three pieces of information: the UTI, the regulator, and the specific field to investigate.\n"
        "1. Start by asking the user for the details if they haven't provided them.\n"
        "2. If the user provides a field name and regulator, use the `get_regulatory_reporting_fields` tool to check if the field is valid.\n"
        "3. Ask for confirmation from the user before proceeding. Be explicit, for example: 'Is it correct that you want to investigate the Notional Amount for UTI 123 under EMIR?'\n"
        "4. Once you have confirmed all details, and only then, call the `confirm_trade_details_for_investigation` tool to finalize the process."
    )

    agent_graph = create_tool_calling_agent(
        llm=llm,
        tools=tools,
        system_prompt=system_prompt,
        state_schema=AgentState,
        checkpointer=SqliteSaver.from_conn_string(":memory:"),
    )
    
    return agent_graph
