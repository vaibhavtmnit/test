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

# --- Pydantic Models for Data Structure ---
class ParsedQuery(BaseModel):
    """Structured representation of the user's initial query."""
    uti: str = Field(..., description="The Unique Transaction Identifier (UTI) of the trade.")
    field_to_investigate: Optional[str] = Field(None, description="The specific field the user wants to investigate.")
    regulator: Optional[str] = Field(None, description="The relevant regulator, e.g., EMIR, ASIC.")
    trade_type: Optional[Literal['collateral_valuation', 'general']] = Field(None, description="The type of the trade.")
    reporting_type: Optional[Literal['house', 'delegated']] = Field(None, description="The reporting type, house or delegated.")
    dealstore: Optional[str] = Field(None, description="The source system or dealstore of the trade.")

# --- Unified State for the Entire Graph ---
# This single state now holds all the information needed for any step in the process.
class GraphState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    parsed_query: Optional[ParsedQuery]
    regulatory_fields: Optional[List[str]]
    # We add a flag to know if we are waiting for the user's confirmation
    confirmation_pending: bool
    final_field: Optional[str]

# --- Node Functions ---
# Each node performs one action and updates the state. The graph will END after each node,
# waiting for the next user input.

def parse_and_validate_input(state: GraphState):
    print("--- NODE: Parsing and Validating Input ---")
    user_query = state["messages"][-1].content
    llm = AzureChatOpenAI(azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"], temperature=0)
    
    parsed_query = llm.with_structured_output(ParsedQuery).invoke(f"Parse the user query to extract trade details. Query: \"{user_query}\"")
    
    if not parsed_query.field_to_investigate:
        ai_message = AIMessage(content="It looks like you haven't specified which field to investigate. Please tell me the field name (e.g., 'Notional Amount').")
        return {"messages": [ai_message]} # End here and wait for user to re-prompt
        
    ai_message = AIMessage(content=f"Okay, I'm starting the investigation for UTI: **{parsed_query.uti}**.")
    return {"messages": [ai_message], "parsed_query": parsed_query}

def fetch_regulatory_fields(state: GraphState):
    print("--- NODE: Fetching Regulatory Fields ---")
    regulator = state["parsed_query"].regulator
    if not regulator:
        ai_message = AIMessage(content="You haven't specified a regulator (e.g., EMIR, ASIC). I can't validate the field without it. Please clarify.")
        return {"messages": [ai_message]}
        
    fields = get_regulatory_fields(regulator)
    if not fields:
        ai_message = AIMessage(content=f"Sorry, I don't have a schema for the regulator '{regulator}'.")
        return {"messages": [ai_message]}
        
    return {"regulatory_fields": fields}

def match_and_ask_for_confirmation(state: GraphState):
    print("--- NODE: Matching and Asking for Confirmation ---")
    user_field = state["parsed_query"].field_to_investigate
    official_fields = state["regulatory_fields"]
    matched_field = next((f for f in official_fields if f.lower() == user_field.lower()), None)
    
    if matched_field:
        ai_message = AIMessage(content=f"I have identified the field as **'{matched_field}'**. Is this correct? (yes/no)")
        # We tentatively set the final_field and set the flag that we are waiting for confirmation
        return {"messages": [ai_message], "final_field": matched_field, "confirmation_pending": True}
    else:
        fields_list_str = "\n- ".join(official_fields)
        ai_message = AIMessage(content=f"I couldn't find a direct match for '{user_field}'. Please choose the correct field from this list for **{state['parsed_query'].regulator}**:\n- {fields_list_str}")
        return {"messages": [ai_message], "confirmation_pending": True}

def process_user_confirmation(state: GraphState):
    print("--- NODE: Processing User Confirmation ---")
    last_message = state["messages"][-1].content.lower()
    
    # Case 1: User confirmed the matched field
    if last_message == 'yes' and state["final_field"]:
        ai_message = AIMessage(content=f"Great! The field **'{state['final_field']}'** is confirmed. What would you like to do next?")
        # Confirmation is no longer pending
        return {"messages": [ai_message], "confirmation_pending": False}
        
    # Case 2: User selected a field from the list
    official_fields_lower = [f.lower() for f in state["regulatory_fields"]]
    if last_message in official_fields_lower:
        original_field = state["regulatory_fields"][official_fields_lower.index(last_message)]
        ai_message = AIMessage(content=f"Understood. The field has been set to **'{original_field}'**. What's next?")
        return {"messages": [ai_message], "final_field": original_field, "confirmation_pending": False}
    
    # Case 3: Confirmation failed, ask again.
    fields_list_str = "\n- ".join(state["regulatory_fields"])
    ai_message = AIMessage(content=f"My apologies. Please reply with 'yes', 'no', or the correct field name from the list:\n- {fields_list_str}")
    # We remain in a confirmation_pending state
    return {"messages": [ai_message], "confirmation_pending": True}

def end_of_workflow(state: GraphState):
    print("--- NODE: Workflow Complete ---")
    # This node can be used to signal the end of this specific task
    # and potentially hand off to another agent in a more complex system.
    return {}


# --- The Router: This is the brain of the graph ---
def router(state: GraphState):
    """
    This is the main router. It checks the state and decides which node to run next.
    This function is called at the beginning of every turn.
    """
    print("--- ROUTER: Analyzing state ---")
    
    if state.get("confirmation_pending"):
        print("ROUTE: -> process_user_confirmation")
        return "process_user_confirmation"
        
    if not state.get("parsed_query"):
        print("ROUTE: -> parse_and_validate_input")
        return "parse_and_validate_input"
        
    if not state.get("regulatory_fields"):
        print("ROUTE: -> fetch_regulatory_fields")
        return "fetch_regulatory_fields"
        
    if not state.get("final_field"):
        print("ROUTE: -> match_and_ask_for_confirmation")
        return "match_and_ask_for_confirmation"
    
    print("ROUTE: -> end_of_workflow")
    return "end_of_workflow"

# --- Build the Graph ---
def build_supervisor_graph():
    graph = StateGraph(GraphState)
    
    # Add all nodes to the single graph
    graph.add_node("parse_and_validate_input", parse_and_validate_input)
    graph.add_node("fetch_regulatory_fields", fetch_regulatory_fields)
    graph.add_node("match_and_ask_for_confirmation", match_and_ask_for_confirmation)
    graph.add_node("process_user_confirmation", process_user_confirmation)
    graph.add_node("end_of_workflow", end_of_workflow)
    
    # The entry point is the router, which decides the first step.
    graph.set_entry_point(router)
    
    # After any node runs, the graph ends for that turn. The router will pick up
    # the correct next step on the next user message.
    graph.add_edge("parse_and_validate_input", END)
    graph.add_edge("fetch_regulatory_fields", END)
    graph.add_edge("match_and_ask_for_confirmation", END)
    graph.add_edge("process_user_confirmation", END)
    graph.add_edge("end_of_workflow", END)
    
    memory = SqliteSaver.from_conn_string(":memory:")
    return graph.compile(checkpointer=memory)
