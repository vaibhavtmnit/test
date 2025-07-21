# supervisor_setup.py

import os
import operator
from typing import TypedDict, Annotated, Sequence, List, Literal, Optional

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from langchain_openai import AzureChatOpenAI
from tools.regulatory_tools import get_regulatory_fields

# --- Unified State for the Entire Graph ---
# This single state now holds all the information needed for any step in the process.
class GraphState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # Fields previously in the worker's state are now here
    parsed_query: Optional[ParsedQuery]
    regulatory_fields: Optional[List[str]]
    final_field: Optional[str]
    # This will determine the next route for the supervisor
    next_route: str

# --- Pydantic Models (previously in worker) ---
class ParsedQuery(BaseModel):
    """Structured representation of the user's initial query."""
    uti: str = Field(..., description="The Unique Transaction Identifier (UTI) of the trade.")
    field_to_investigate: Optional[str] = Field(None, description="The specific field the user wants to investigate.")
    regulator: Optional[str] = Field(None, description="The relevant regulator, e.g., EMIR, ASIC.")
    trade_type: Optional[Literal['collateral_valuation', 'general']] = Field(None, description="The type of the trade.")
    reporting_type: Optional[Literal['house', 'delegated']] = Field(None, description="The reporting type, house or delegated.")
    dealstore: Optional[str] = Field(None, description="The source system or dealstore of the trade.")

# --- Node Functions (previously worker methods) ---
# These are now standalone functions that operate on the unified GraphState.

def parse_and_validate_input(state: GraphState):
    print("--- NODE: Parsing and Validating Input ---")
    user_query = state["messages"][-1].content
    llm = AzureChatOpenAI(azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"], temperature=0)
    
    parsed_query = llm.with_structured_output(ParsedQuery).invoke(f"Parse the user query to extract trade details. Query: \"{user_query}\"")
    
    ai_message = AIMessage(content=f"Okay, I'm starting the investigation for UTI: **{parsed_query.uti}**.")
    
    if not parsed_query.field_to_investigate:
        ai_message = AIMessage(content="It looks like you haven't specified which field to investigate. Please tell me the field name (e.g., 'Notional Amount').")
        return {"messages": [ai_message], "next_route": "end"} # End here and wait for user to re-prompt
        
    return {"messages": [ai_message], "parsed_query": parsed_query, "next_route": "fetch_fields"}

def fetch_regulatory_fields(state: GraphState):
    print("--- NODE: Fetching Regulatory Fields ---")
    regulator = state["parsed_query"].regulator
    if not regulator:
        ai_message = AIMessage(content="You haven't specified a regulator (e.g., EMIR, ASIC). I can't validate the field without it. Please clarify.")
        return {"messages": [ai_message], "next_route": "end"}
        
    fields = get_regulatory_fields(regulator)
    if not fields:
        ai_message = AIMessage(content=f"Sorry, I don't have a schema for the regulator '{regulator}'.")
        return {"messages": [ai_message], "next_route": "end"}
        
    return {"regulatory_fields": fields, "next_route": "match_and_confirm_field"}

def match_and_confirm_field(state: GraphState):
    print("--- NODE: Matching and Confirming Field ---")
    user_field = state["parsed_query"].field_to_investigate
    official_fields = state["regulatory_fields"]
    matched_field = next((f for f in official_fields if f.lower() == user_field.lower()), None)
    
    if matched_field:
        ai_message = AIMessage(content=f"I have identified the field as **'{matched_field}'**. Is this correct? (yes/no)")
        final_field = matched_field
    else:
        fields_list_str = "\n- ".join(official_fields)
        ai_message = AIMessage(content=f"I couldn't find a direct match for '{user_field}'. Please choose the correct field from this list for **{state['parsed_query'].regulator}**:\n- {fields_list_str}")
        final_field = None
        
    return {"messages": [ai_message], "final_field": final_field, "next_route": "process_user_confirmation"}

def process_user_confirmation(state: GraphState):
    print("--- NODE: Processing User Confirmation ---")
    last_message = state["messages"][-1].content.lower()
    
    if last_message == 'yes' and state["final_field"]:
        ai_message = AIMessage(content=f"Great! The field **'{state['final_field']}'** is confirmed. I will now proceed.")
        return {"messages": [ai_message], "next_route": "end"} # End of this flow
        
    official_fields_lower = [f.lower() for f in state["regulatory_fields"]]
    if last_message in official_fields_lower:
        original_field = state["regulatory_fields"][official_fields_lower.index(last_message)]
        ai_message = AIMessage(content=f"Understood. The field has been set to **'{original_field}'** and I will now proceed.")
        return {"messages": [ai_message], "final_field": original_field, "next_route": "end"}
    
    # If confirmation fails, ask again.
    fields_list_str = "\n- ".join(state["regulatory_fields"])
    ai_message = AIMessage(content=f"My apologies. Please reply with 'yes', 'no', or the correct field name from the list:\n- {fields_list_str}")
    return {"messages": [ai_message], "next_route": "process_user_confirmation"} # Loop back to this node

# --- Supervisor / Router ---
def supervisor_router(state: GraphState):
    """
    This is the main router. For now, it always starts the validation workflow.
    In the future, it can be expanded to route to different worker flows.
    """
    print("--- SUPERVISOR: Routing request ---")
    # This is where you would add logic to route to different agents.
    # For now, we only have one workflow.
    if len(state['messages']) == 1: # First user message
        return "parse_and_validate_input"
    
    # Use the 'next_route' field set by the previous node to decide where to go.
    return state.get("next_route", "end")

# --- Build the Graph ---
def build_supervisor_graph():
    graph = StateGraph(GraphState)
    
    # Add all nodes to the single graph
    graph.add_node("parse_and_validate_input", parse_and_validate_input)
    graph.add_node("fetch_fields", fetch_regulatory_fields)
    graph.add_node("match_and_confirm_field", match_and_confirm_field)
    graph.add_node("process_user_confirmation", process_user_confirmation)
    
    # The entry point is the supervisor router
    graph.set_entry_point("parse_and_validate_input")
    
    # Define the routing logic using conditional edges
    graph.add_conditional_edges(
        "parse_and_validate_input",
        lambda x: x["next_route"],
        {"fetch_fields": "fetch_fields", "end": END}
    )
    graph.add_conditional_edges(
        "fetch_fields",
        lambda x: x["next_route"],
        {"match_and_confirm_field": "match_and_confirm_field", "end": END}
    )
    graph.add_conditional_edges(
        "match_and_confirm_field",
        lambda x: x["next_route"],
        {"process_user_confirmation": "process_user_confirmation"}
    )
    graph.add_conditional_edges(
        "process_user_confirmation",
        lambda x: x["next_route"],
        {"process_user_confirmation": "process_user_confirmation", "end": END}
    )
    
    memory = SqliteSaver.from_conn_string(":memory:")
    return graph.compile(checkpointer=memory)
