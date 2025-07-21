# tools/trade_data_tools.py

import json
from typing import Optional, Dict

def fetch_trade_data(trade_details: dict, impacted_field: str) -> Optional[Dict]:
    """
    Simulates fetching detailed trade data from a database or another system.

    In a real application, this would connect to a data source. Here, we
    simulate success or failure based on the input UTI.

    Args:
        trade_details: A dictionary of validated details from the query parser.
        impacted_field: The specific field the user is interested in.

    Returns:
        A dictionary (JSON object) if the trade is found, otherwise None.
    """
    print(f"\n--- TOOL: Fetching data for UTI: {trade_details.get('uti')} ---")
    
    # Simulate a database lookup. We'll pretend we can only find data for UTI '123ABC'.
    if trade_details.get("uti") == "123ABC":
        print("✅ Trade found in data source.")
        # Return a sample JSON object
        return {
            "uti": "123ABC",
            "regulator": trade_details.get("regulator"),
            "status": "Reported",
            "reporting_timestamp": "2023-10-27T10:00:00Z",
            "fields": {
                "Notional Amount": 1000000,
                "Currency": "USD",
                "Effective Date": "2023-10-28",
                "Maturity Date": "2024-10-28"
            },
            "lineage": "Originated in London Dealstore, processed via Reporting Engine v2."
        }
    else:
        print("❌ Trade not found in data source.")
        return None


# agent/data_fetcher.py

from typing import Optional, Dict, Tuple
from tools.trade_data_tools import fetch_trade_data

def fetch_and_process_data(validated_details: dict, impacted_field: str) -> Tuple[bool, Optional[Dict]]:
    """
    An agent function that takes validated query details, fetches data using a tool,
    and processes the result.

    Args:
        validated_details: The dictionary of entities from the query_parser agent.
        impacted_field: The specific field to be analyzed.

    Returns:
        A tuple containing:
        - A boolean indicating success (True if data was found, False otherwise).
        - The JSON data as a dictionary if successful, otherwise None.
    """
    print(f"\n--- AGENT: Data Fetcher running for field '{impacted_field}' ---")
    
    # Call the tool to fetch the data
    json_data = fetch_trade_data(validated_details, impacted_field)
    
    if json_data:
        print("✅ Success! Data fetched and processed.")
        return True, json_data
    else:
        print("❌ Failure. No data returned from the tool.")
        return False, None



# tests/test_data_fetcher.py

from agent.data_fetcher import fetch_and_process_data

def run_tests():
    """
    A simple utility to run test cases against the data fetcher agent.
    """
    print("=====================================")
    print("= Running Data Fetcher Test Suite =")
    print("=====================================")

    # --- Test Case 1: Simulating a successful data fetch ---
    # This simulates the output we would get from the query_parser for a successful parse.
    print("\n--- Test Case 1: Happy Path (Data Found) ---")
    successful_parse_details = {
        'regulator': 'EMIR', 
        'uti': '123ABC', 
        'trade_type': 'General', 
        'reporting_type': 'House'
    }
    impacted_field_success = 'Notional Amount'
    
    success, data = fetch_and_process_data(successful_parse_details, impacted_field_success)
    
    if success and data is not None:
        print(f"🟢 PASSED: Agent correctly returned success status and data.")
        assert data['uti'] == '123ABC'
    else:
        print(f"🔴 FAILED: Expected success, but agent returned status={success}.")


    # --- Test Case 2: Simulating a failed data fetch ---
    # Here, the UTI is different, so our mock tool will return None.
    print("\n--- Test Case 2: Failure Path (Data Not Found) ---")
    failed_parse_details = {
        'regulator': 'ASIC', 
        'uti': '999XYZ', 
        'trade_type': 'General', 
        'reporting_type': 'Delegated'
    }
    impacted_field_fail = 'Price'

    success, data = fetch_and_process_data(failed_parse_details, impacted_field_fail)

    if not success and data is None:
        print(f"🟢 PASSED: Agent correctly returned failure status and no data.")
    else:
        print(f"🔴 FAILED: Expected failure, but agent returned status={success}.")

    print("\n--- Test Suite Complete ---")


if __name__ == "__main__":
    run_tests()





# agent/supervisor.py

import operator
from typing import TypedDict, Annotated, Optional, Dict

from langgraph.graph import StateGraph, END
from agent.query_parser import parse_user_query
from agent.data_fetcher import fetch_and_process_data

# --- 1. Define the State for the Supervisor Graph ---
# This dictionary holds all the data that moves between the steps of our workflow.
class SupervisorState(TypedDict):
    # The initial user query
    user_query: str
    
    # The output from the query parser agent
    parsed_details: Optional[Dict]
    impacted_field: Optional[str]
    parser_error: Optional[str]
    
    # The final result of the workflow
    final_message: str

# --- 2. Define the Nodes of the Graph ---
# Each node is a function that performs a step in the workflow.

def call_query_parser(state: SupervisorState):
    """Node that runs the first agent: the query parser."""
    print("--- SUPERVISOR: Calling Query Parser ---")
    user_query = state["user_query"]
    parsed_details, impacted_field, error = parse_user_query(user_query)
    return {
        "parsed_details": parsed_details,
        "impacted_field": impacted_field,
        "parser_error": error,
    }

def call_data_fetcher(state: SupervisorState):
    """Node that runs the second agent: the data fetcher."""
    print("--- SUPERVISOR: Calling Data Fetcher ---")
    parsed_details = state["parsed_details"]
    impacted_field = state["impacted_field"]
    
    success, data = fetch_and_process_data(parsed_details, impacted_field)
    
    if success:
        message = f"Workflow complete. Data successfully fetched for UTI {parsed_details.get('uti')}."
    else:
        message = f"Workflow complete, but no data could be found for UTI {parsed_details.get('uti')}."
        
    return {"final_message": message}

# --- 3. Define the Conditional Routing Logic ---
# This function decides which path to take after the query parser runs.

def after_parsing_router(state: SupervisorState):
    """
    Checks the output of the query parser. If there was an error, the workflow ends.
    Otherwise, it proceeds to the data fetcher.
    """
    if state["parser_error"]:
        print("--- SUPERVISOR: Parser failed. Halting workflow. ---")
        # We can set the final message here and end the graph
        error_message = f"Workflow halted at parsing stage. Reason: {state['parser_error']}"
        # To end the graph, we return the name of the special END node.
        # We also need to update the state with the final message.
        return "end_with_error" 
    else:
        print("--- SUPERVISOR: Parser succeeded. Proceeding. ---")
        return "call_data_fetcher"

def handle_error_and_end(state: SupervisorState):
    """A simple node to set the final message in case of a parsing error."""
    return {"final_message": f"Workflow halted at parsing stage. Reason: {state['parser_error']}"}

# --- 4. Assemble the Graph ---
def build_supervisor_graph():
    """Builds and compiles the supervisor StateGraph."""
    graph = StateGraph(SupervisorState)

    # Add the nodes to the graph
    graph.add_node("query_parser", call_query_parser)
    graph.add_node("data_fetcher", call_data_fetcher)
    graph.add_node("error_handler", handle_error_and_end)

    # Set the entry point of the graph
    graph.set_entry_point("query_parser")

    # Add the conditional edge for routing after the parser
    graph.add_conditional_edges(
        "query_parser",
        after_parsing_router,
        {
            "call_data_fetcher": "data_fetcher",
            "end_with_error": "error_handler",
        },
    )
    
    # After the data fetcher or error handler runs, the workflow is finished.
    graph.add_edge("data_fetcher", END)
    graph.add_edge("error_handler", END)

    # Compile the graph into a runnable application
    return graph.compile()

# You can create a single runnable instance for your application to use
supervisor_app = build_supervisor_graph()




# tests/test_supervisor.py

import os
from agent.supervisor import supervisor_app

def run_end_to_end_tests():
    """
    A utility to run end-to-end tests for the new LangGraph-based supervisor.
    """
    # --- Set your Azure environment variables for the test ---
    # os.environ["AZURE_OPENAI_API_KEY"] = "YOUR_KEY"
    # os.environ["AZURE_OPENAI_ENDPOINT"] = "YOUR_ENDPOINT"
    # os.environ["OPENAI_API_VERSION"] = "YOUR_API_VERSION"
    # os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = "YOUR_DEPLOYMENT_NAME"
    
    print("=====================================")
    print("= Running Supervisor Test Suite     =")
    print("=====================================")

    # --- Test Case 1: Full successful workflow ---
    print("\n--- Test Case 1: End-to-End Success ---")
    query_success = "Analyze the Notional Amount for UTI 123ABC under EMIR."
    # We invoke the graph by passing the initial state
    final_state_success = supervisor_app.invoke({"user_query": query_success})
    result_success = final_state_success.get("final_message")
    print(f"\nFinal Outcome for Test 1: {result_success}")
    assert "Data successfully fetched" in result_success


    # --- Test Case 2: End-to-End, Data Not Found ---
    print("\n\n--- Test Case 2: End-to-End, Data Not Found ---")
    query_data_fail = "Check the Currency for trade 999XYZ for ASIC."
    final_state_data_fail = supervisor_app.invoke({"user_query": query_data_fail})
    result_data_fail = final_state_data_fail.get("final_message")
    print(f"\nFinal Outcome for Test 2: {result_data_fail}")
    assert "no data could be found" in result_data_fail


    # --- Test Case 3: End-to-End, Parser Failure ---
    print("\n\n--- Test Case 3: End-to-End, Parser Failure ---")
    query_parser_fail = "Look at this trade for EMIR."
    final_state_parser_fail = supervisor_app.invoke({"user_query": query_parser_fail})
    result_parser_fail = final_state_parser_fail.get("final_message")
    print(f"\nFinal Outcome for Test 3: {result_parser_fail}")
    assert "Workflow halted" in result_parser_fail

    print("\n--- Test Suite Complete ---")

if __name__ == "__main__":
    run_end_to_end_tests()




You are absolutely right. The current supervisor is just a simple Python script. Let's elevate it into a proper, intelligent LangGraph agent that can make decisions.

This new supervisor will be a StateGraph that orchestrates the workflow, deciding whether to proceed or halt based on the output of the first agent. This is the correct and modern way to implement the logic you're describing.

I will update the supervisor.py file in your Canvas with this new LangGraph implementation. I will also update the corresponding test file so you can see how to run this new agent.

1. The New Supervisor Agent (as a LangGraph)
This is the updated supervisor.py. It now defines a stateful graph that manages the entire process.

2. The Updated Testing Utility
This test file is now updated to import the supervisor_app from the new supervisor.py and invoke it correctly.






how give me a template on how can I add more worker agents to the superviser and what change would be required in supeviser




Show thinking
Of course. Extending the supervisor with more worker agents is the natural next step. Here is a clear template and explanation of the changes required in the supervisor.py file in your Canvas.

The general pattern involves three steps:

Create your new worker agent function.

Add it as a new node in the supervisor's graph.

Update the graph's routing logic (the edges) to include this new node in the workflow.

Let's imagine we want to add a new worker called Data Analyzer. Its job is to take the JSON data from the data_fetcher and produce a simple analysis.

### Step 1: Create the New Worker Agent Function
First, you would create the function for your new agent. It would look something like this. You could place this function within your supervisor.py file or import it from its own file.

Python

# --- Template for a New Worker Agent ---

def call_data_analyzer(state: SupervisorState):
    """
    A new worker that analyzes the data fetched by the previous step.
    """
    print("--- SUPERVISOR: Calling Data Analyzer ---")
    # This agent would receive the 'data' through the state, but since
    # our current state doesn't pass it, we'll just use the final_message
    # for this example. A real implementation would add 'data' to the state.
    
    analysis_result = "Analysis complete: The 'Notional Amount' field is within expected limits."
    
    # It updates the final_message in the state
    return {"final_message": analysis_result}
### Step 2: Update the Supervisor Graph
Next, you need to modify the build_supervisor_graph function inside your supervisor.py file in the Canvas to integrate this new worker.

Here are the required changes:

Add the New Node: You add the call_data_analyzer function as a new node in the graph.

Update the Edges: Instead of the data_fetcher node going directly to END, you change its edge to point to your new analyzer node. The analyzer node will then point to END.

Here is what the modified build_supervisor_graph function would look like.

Python

# --- Modified section within agent/supervisor.py ---

def build_supervisor_graph():
    """Builds and compiles the supervisor StateGraph."""
    graph = StateGraph(SupervisorState)

    # Add the existing nodes
    graph.add_node("query_parser", call_query_parser)
    graph.add_node("data_fetcher", call_data_fetcher)
    graph.add_node("error_handler", handle_error_and_end)
    
    # --- 1. Add your new worker as a node ---
    graph.add_node("analyzer", call_data_analyzer) # New line

    # Set the entry point (this doesn't change)
    graph.set_entry_point("query_parser")

    # The conditional edge for the parser doesn't change
    graph.add_conditional_edges(
        "query_parser",
        after_parsing_router,
        {
            "call_data_fetcher": "data_fetcher",
            "end_with_error": "error_handler",
        },
    )
    
    # --- 2. Update the graph edges ---
    # The data_fetcher no longer ends the graph. It now goes to the analyzer.
    graph.add_edge("data_fetcher", "analyzer") # Changed line
    
    # The new analyzer node now marks the end of the successful workflow.
    graph.add_edge("analyzer", END) # New line
    
    # The error handler edge doesn't change
    graph.add_edge("error_handler", END)

    # Compile the graph into a runnable application
    return graph.compile()
By following this pattern, you can chain as many worker agents as you need, creating complex workflows that are managed and orchestrated by the supervisor graph
