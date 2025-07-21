# tools/regulatory_tools.py

def get_regulatory_fields(regulator: str) -> list[str]:
    """
    Retrieves the list of mandatory reporting fields for a given financial regulator.

    Args:
        regulator: The acronym of the regulator (e.g., 'EMIR', 'ASIC').

    Returns:
        A list of field names.
    """
    print(f"--- TOOL: Fetching fields for regulator: {regulator} ---")
    
    # In a real application, this would query a database or a configuration service.
    regulatory_schemas = {
        "EMIR": [
            "Trade ID", "Product ID", "Notional Amount", "Currency", 
            "Effective Date", "Maturity Date", "Counterparty ID"
        ],
        "ASIC": [
            "Transaction Reference Number", "Instrument Identifier", "Execution Timestamp",
            "Quantity", "Price", "Venue", "Trader ID"
        ],
        "MAS": [
            "Reporting Entity ID", "Trade Party 1", "Trade Party 2", "Asset Class",
            "Notional Quantity", "Price Notation", "Cleared Status"
        ]
    }
    
    return regulatory_schemas.get(regulator.upper(), [])


# agent/query_validator_class.py

import os
from typing import TypedDict, Annotated, List, Literal, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from langchain_openai import AzureChatOpenAI
from tools.regulatory_tools import get_regulatory_fields

# --- Pydantic Model for Structured Data ---
class ParsedQuery(BaseModel):
    """Structured representation of the user's initial query."""
    uti: str = Field(..., description="The Unique Transaction Identifier (UTI) of the trade.")
    field_to_investigate: Optional[str] = Field(None, description="The specific field the user wants to investigate.")
    regulator: Optional[str] = Field(None, description="The relevant regulator, e.g., EMIR, ASIC.")
    trade_type: Optional[Literal['collateral_valuation', 'general']] = Field(None, description="The type of the trade.")
    reporting_type: Optional[Literal['house', 'delegated']] = Field(None, description="The reporting type, house or delegated.")
    dealstore: Optional[str] = Field(None, description="The source system or dealstore of the trade.")

# --- Agent State Definition ---
class AgentState(TypedDict):
    messages: Annotated[list, lambda x, y: x + y]
    parsed_query: ParsedQuery
    regulatory_fields: List[str]
    final_field: str
    needs_user_input: bool

# --- Agent Class ---
class QueryValidatorAgent:
    """An agent that validates and enriches a user's query about a transaction."""
    def __init__(self):
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("parser_validator", self._parse_and_validate_input)
        workflow.add_node("fetch_fields", self._fetch_regulatory_fields)
        workflow.add_node("matcher_confirmer", self._match_and_confirm_field)
        workflow.add_node("process_confirmation", self._process_user_confirmation)
        workflow.add_node("end_early", lambda state: {"needs_user_input": True})
        workflow.add_node("final_output", lambda state: {"needs_user_input": False})

        workflow.set_entry_point("parser_validator")

        workflow.add_conditional_edges("parser_validator", self._route_after_validation)
        workflow.add_edge("fetch_fields", "matcher_confirmer")
        workflow.add_edge("matcher_confirmer", "end_early")
        workflow.add_conditional_edges("process_confirmation", self._route_after_confirmation)
        workflow.add_edge("final_output", END)
        
        memory = SqliteSaver.from_conn_string(":memory:")
        return workflow.compile(checkpointer=memory)

    def _parse_and_validate_input(self, state: AgentState):
        print("--- NODE: Parsing and Validating Input ---")
        user_query = state["messages"][-1].content
        llm_with_tools = AzureChatOpenAI(
            azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            temperature=0
        ).with_structured_output(ParsedQuery)
        prompt = f"Parse the user query to extract trade details. The user MUST specify a 'field_to_investigate'. Query: \"{user_query}\""
        parsed_query = llm_with_tools.invoke(prompt)
        ai_message = AIMessage(content=f"Okay, I'm starting the investigation for UTI: **{parsed_query.uti}**.")
        return {"parsed_query": parsed_query, "messages": [ai_message], "needs_user_input": parsed_query.field_to_investigate is None}

    def _fetch_regulatory_fields(self, state: AgentState):
        print("--- NODE: Fetching Regulatory Fields ---")
        regulator = state["parsed_query"].regulator
        if not regulator:
            ai_message = AIMessage(content="You haven't specified a regulator (e.g., EMIR, ASIC). I can't validate the field without it.")
            return {"messages": [ai_message], "needs_user_input": True}
        fields = get_regulatory_fields(regulator)
        if not fields:
            ai_message = AIMessage(content=f"Sorry, I don't have a schema for the regulator '{regulator}'.")
            return {"messages": [ai_message], "needs_user_input": True}
        return {"regulatory_fields": fields}

    def _match_and_confirm_field(self, state: AgentState):
        print("--- NODE: Matching and Confirming Field ---")
        user_field = state["parsed_query"].field_to_investigate
        official_fields = state["regulatory_fields"]
        matched_field = next((f for f in official_fields if f.lower() == user_field.lower()), None)
        if matched_field:
            ai_message = AIMessage(content=f"I have identified the field as **'{matched_field}'**. Is this correct? (yes/no)")
            final_field = matched_field
        else:
            fields_list_str = "\n- ".join(official_fields)
            ai_message = AIMessage(content=f"I couldn't find a direct match for '{user_field}'. Please choose from this list for **{state['parsed_query'].regulator}**:\n- {fields_list_str}")
            final_field = None
        return {"messages": [ai_message], "final_field": final_field, "needs_user_input": True}

    def _process_user_confirmation(self, state: AgentState):
        print("--- NODE: Processing User Confirmation ---")
        last_message = state["messages"][-1].content.lower()
        if last_message == 'yes' and state["final_field"]:
            ai_message = AIMessage(content="Great! I will proceed with the investigation on this field.")
            return {"messages": [ai_message], "needs_user_input": False}
        
        official_fields_lower = [f.lower() for f in state["regulatory_fields"]]
        if last_message in official_fields_lower:
            original_field = state["regulatory_fields"][official_fields_lower.index(last_message)]
            ai_message = AIMessage(content=f"Understood. I have set the field to **'{original_field}'** and will proceed.")
            return {"messages": [ai_message], "final_field": original_field, "needs_user_input": False}
        
        fields_list_str = "\n- ".join(state["regulatory_fields"])
        ai_message = AIMessage(content=f"My apologies. Please select the correct field from the list:\n- {fields_list_str}")
        return {"messages": [ai_message], "needs_user_input": True, "final_field": None}

    def _route_after_validation(self, state: AgentState):
        return "end_early" if state["needs_user_input"] else "fetch_fields"

    def _route_after_confirmation(self, state: AgentState):
        return "end_early" if state["needs_user_input"] else "end_complete"



# supervisor_setup.py

import os
from typing import TypedDict, Annotated
from langchain_core.messages import AIMessage
from langchain_openai import AzureChatOpenAI
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_supervisor
from agent.query_validator_class import QueryValidatorAgent

# --- Define a placeholder for a future worker agent ---
class DataFetcherAgent:
    def invoke(self, state):
        return {"messages": [AIMessage(content="This is a placeholder for a future data fetching agent.")]}

# --- Supervisor State ---
class SupervisorState(TypedDict):
    messages: Annotated[list, add_messages]

def build_supervisor_graph():
    """Builds and returns the main supervisor agent graph."""
    llm = AzureChatOpenAI(
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
        temperature=0
    )
    
    # Instantiate agents
    validator_agent = QueryValidatorAgent()
    fetcher_agent = DataFetcherAgent()

    members = {
        "validator": {
            "graph": validator_agent.graph,
            "description": "Validates and enriches a user's initial query. Use this first for all new requests."
        },
        "fetcher": {
            "graph": fetcher_agent.invoke,
            "description": "Fetches detailed trade data after the query has been validated."
        }
    }

    return create_supervisor(
        llm=llm,
        members=members,
        state_schema=SupervisorState,
        interrupt_after_members=["validator", "fetcher"]
    )



# main.py

import uuid
from fastapi import FastAPI
from pydantic import BaseModel
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage
from supervisor_setup import build_supervisor_graph

# --- Pydantic Models for API ---
class InvokeRequest(BaseModel):
    input: str
    thread_id: str | None = None

class InvokeResponse(BaseModel):
    output: str
    thread_id: str

# --- App Setup ---
app = FastAPI(
    title="Multi-Agent Application Server",
    description="Backend server for the LangGraph multi-agent system."
)

# Build the supervisor graph when the app starts
supervisor_graph = build_supervisor_graph()

# --- API Endpoint ---
@app.post("/invoke", response_model=InvokeResponse)
async def invoke_agent(request: InvokeRequest):
    """
    Receives a user query, invokes the supervisor agent,
    and returns the agent's response.
    """
    # Generate a new thread_id if one is not provided
    thread_id = request.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    # Invoke the supervisor graph
    events = supervisor_graph.stream(
        {"messages": [HumanMessage(content=request.input)]}, config=config
    )
    
    # The final response is in the last event
    final_response = ""
    for event in events:
        if "messages" in event.get("supervisor", {}):
            final_response = event["supervisor"]["messages"][-1].content
    
    return InvokeResponse(output=final_response, thread_id=thread_id)


# chainlit_app.py

import chainlit as cl
import httpx

FASTAPI_AGENT_URL = "http://127.0.0.1:8000/invoke"

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("thread_id", None)
    await cl.Message(content="Financial reporting agent is ready. How can I help you today?").send()

@cl.on_message
async def on_message(message: cl.Message):
    thread_id = cl.user_session.get("thread_id")
    
    payload = {"input": message.content, "thread_id": thread_id}
    
    response_msg = cl.Message(content="")
    await response_msg.send()

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(FASTAPI_AGENT_URL, json=payload)
            response.raise_for_status()
            agent_response = response.json()
            
            final_content = agent_response.get("output", "Error: No output received.")
            cl.user_session.set("thread_id", agent_response.get("thread_id"))

    except httpx.RequestError as e:
        final_content = f"Error: Could not connect to the agent server. Details: {e}"
    except Exception as e:
        final_content = f"An unexpected error occurred: {e}"

    response_msg.content = final_content
    await response_msg.update()
