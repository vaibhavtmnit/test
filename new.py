# agent/supervisor.py

import os
import operator
from typing import TypedDict, Annotated, Sequence, Optional, Dict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from langchain_openai import AzureChatOpenAI

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver

# Import the original worker functions
from agent.query_parser import parse_user_query
from agent.data_fetcher import fetch_and_process_data

# --- 1. Define the NEW, richer State ---
# <<< CHANGED >>>
# This state now holds structured data, not just messages.
# It's the "shared project brief" for all our agents.
class AgentState(TypedDict):
    # The user's original query
    user_query: str
    
    # We still keep messages for conversational context if needed
    messages: Annotated[Sequence[BaseMessage], operator.add]
    
    # Structured data from the parser agent
    parsed_data: Optional[Dict]
    
    # The final data fetched by the data agent
    final_response: Optional[str]
    
    # A status flag to report errors
    error: Optional[str]


# --- 2. Define Agent Nodes (Instead of Tools) ---
# <<< CHANGED >>>
# These are no longer "tools" for one agent to call. They are independent
# nodes in the graph that read from and write to our shared AgentState.

def query_parser_node(state: AgentState) -> Dict:
    """
    Parses the user query and updates the state with structured data or an error.
    This is a self-contained "specialist" agent.
    """
    print("--- NODE: Running Query Parser ---")
    user_query = state["user_query"]
    parsed_details, impacted_field, error = parse_user_query(user_query)
    
    if error:
        print(f"--- PARSER ERROR: {error} ---")
        return {"error": error} # Update the state with the error
    
    # On success, update the state with the structured, parsed data
    success_data = {**parsed_details, "impacted_field": impacted_field}
    print(f"--- PARSER SUCCESS: {success_data} ---")
    return {"parsed_data": success_data}


def data_fetcher_node(state: AgentState) -> Dict:
    """
    Reads parsed data from the state, fetches data, and updates the state.
    This node ONLY runs if the parser was successful.
    """
    print("--- NODE: Running Data Fetcher ---")
    parsed_query_details = state["parsed_data"]
    
    if not parsed_query_details:
        return {"error": "Cannot fetch data, query was not parsed successfully."}

    impacted_field = parsed_query_details.pop("impacted_field", None)
    
    success, data = fetch_and_process_data(parsed_query_details, impacted_field)
    
    if success:
        print(f"--- FETCHER SUCCESS ---")
        return {"final_response": f"Successfully fetched data: {data}"}
    else:
        print(f"--- FETCHER FAILURE ---")
        error_msg = f"Failed to fetch data for UTI: {parsed_query_details.get('uti')}."
        return {"error": error_msg}


# --- 3. The Supervisor's NEW Role: The Router ---
# <<< CHANGED >>>
# The Supervisor no longer has a complex prompt to call tools.
# Its only job is to look at the state and decide which node to run next.
# For now, this logic is simple enough to be a Python function, no LLM needed!

def supervisor_router(state: AgentState) -> str:
    """
    This is the core of our supervisor. It inspects the state
    and decides which node to route to next.
    """
    print("--- SUPERVISOR: Routing ---")
    if state.get("error"):
        return "handle_error"
    if state.get("parsed_data") is None:
        return "query_parser"
    if state.get("final_response") is None:
        return "data_fetcher"
    return "end"

# We can also create simple nodes for handling entry and exit
def handle_error_node(state: AgentState) -> Dict:
    """A simple node to formalize the end state on error."""
    print("--- NODE: Handling Error ---")
    return {"final_response": state["error"]}


# --- 4. Build the NEW Graph with Conditional Routing ---
# <<< CHANGED >>>
# The graph structure now explicitly defines the workflow.
def build_supervisor_graph():
    """Builds the new, state-driven orchestrator graph."""
    
    graph = StateGraph(AgentState)
    
    # Add all our nodes to the graph
    graph.add_node("query_parser", query_parser_node)
    graph.add_node("data_fetcher", data_fetcher_node)
    graph.add_node("handle_error", handle_error_node)

    # The entry point is now the supervisor router
    graph.set_entry_point("supervisor_router")

    # Add conditional edges from the supervisor router
    graph.add_conditional_edges(
        "supervisor_router",
        supervisor_router,
        {
            "query_parser": "query_parser",
            "data_fetcher": "data_fetcher",
            "handle_error": "handle_error",
            "end": END
        }
    )

    # Add edges from worker nodes back to the supervisor to re-evaluate
    graph.add_edge("query_parser", "supervisor_router")
    graph.add_edge("data_fetcher", "supervisor_router")
    graph.add_edge("handle_error", END)

    # Compile the graph
    # For a real application, you'd use a persistent checkpointer like Postgres or Redis
    checkpointer = InMemorySaver()
    return graph.compile(checkpointer=checkpointer)

# --- How to Run It ---
# To use this new graph, you would invoke it with the initial state:
# app = build_supervisor_graph()
# initial_state = {"user_query": "The user's question here", "messages": []}
# for s in app.stream(initial_state, {"recursion_limit": 5}):
#     print(s)
# The final result will be in the 'final_response' key of the state.







# You'll need these imports
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

# --- 1. Define a structured output model (Best Practice) ---
# This forces the LLM to return clean, predictable JSON.
class ParsedQuery(BaseModel):
    """Structured representation of a parsed user query."""
    regulator: Optional[str] = Field(description="The financial regulator, e.g., 'CME'.")
    uti: Optional[str] = Field(description="The Unique Trade Identifier.")
    impacted_field: Optional[str] = Field(description="The specific field under investigation.")
    is_complete: bool = Field(description="Set to true if all mandatory fields (regulator, uti) are present.")
    follow_up_question: Optional[str] = Field(description="A question to ask the user if information is missing.")

# --- 2. Create the new, smarter query parser node ---
def conversational_query_parser_node(state: AgentState) -> Dict:
    """
    An intelligent parser that reads the entire conversation to extract details.
    """
    print("--- NODE: Running Conversational Query Parser ---")

    # This is the prompt that makes the magic happen ✨
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert at parsing user requests from a conversation. Your goal is to extract key details for a financial data query. Analyze the entire conversation history provided below and fill in the `ParsedQuery` structure. If mandatory information is missing, ask a clear follow-up question."),
        ("human", "Conversation History:\n{conversation}")
    ])
    
    # Set up the LLM to return structured output
    llm = AzureChatOpenAI(
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
        temperature=0
    )
    structured_llm = llm.with_structured_output(ParsedQuery)
    
    # Format the conversation history for the prompt
    conversation_text = "\n".join([f"{msg.type}: {msg.content}" for msg in state["messages"]])
    
    # Invoke the LLM to parse the full conversation
    parsed_result = structured_llm.invoke(prompt.format(conversation=conversation_text))

    if not parsed_result.is_complete:
        # If info is missing, we can set up a loop to ask the user
        # For now, we'll just report an error with the follow-up question
        return {"error": parsed_result.follow_up_question}
    
    # On success, update the state with the structured data
    # We convert the Pydantic model to a dict for the state
    success_data = parsed_result.dict()
    print(f"--- CONVERSATIONAL PARSER SUCCESS: {success_data} ---")
    return {"parsed_data": success_data}



{
  "user_query": "I need data for UTI 123 and regulator CME",
  "messages": [ ... ],
  "parsed_data": null,
  "final_response": null,
  "error": null
}


