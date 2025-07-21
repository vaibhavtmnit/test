# agent/query_parser.py

import os
from typing import Literal, Optional
from langchain_core.pydantic_v1 import BaseModel, Field, ValidationError
from langchain_openai import AzureChatOpenAI

# --- 1. Define the Data Structure with Pydantic ---
# This class defines the exact information we want to extract.
# The mandatory fields (`regulator` and `field_to_investigate`) are not Optional.
class ParsedQuery(BaseModel):
    """Structured representation of all entities in the user's query."""
    regulator: str = Field(..., description="The financial regulator, e.g., EMIR, ASIC, MAS.")
    field_to_investigate: str = Field(..., description="The specific field the user wants to investigate, e.g., 'Notional Amount'.")
    uti: Optional[str] = Field(None, description="The Unique Transaction Identifier (UTI) of the trade.")
    trade_type: Optional[Literal['General', 'Collateral & Valuation']] = Field(None, description="The type of the trade.")
    reporting_type: Optional[Literal['House', 'Delegated']] = Field(None, description="The reporting type, either House or Delegated.")
    
    class Config:
        # Pydantic configuration to handle enum-like string matching
        use_enum_values = True

# --- 2. Create the Agent Function ---
def parse_user_query(user_query: str) -> tuple[Optional[dict], Optional[str], Optional[str]]:
    """
    Parses a user's query to extract trade details using an LLM.

    This function uses langchain's `.with_structured_output` method, which leverages
    LLM tool-calling to reliably extract information into the `ParsedQuery` Pydantic model.

    Args:
        user_query: The raw text input from the user.

    Returns:
        A tuple containing:
        - A dictionary of found entities (if successful), otherwise None.
        - The impacted field name (if successful), otherwise None.
        - An error message string (if unsuccessful), otherwise None.
    """
    print(f"\n--- Parsing Query: '{user_query}' ---")
    
    # Ensure Azure OpenAI environment variables are set
    if "AZURE_OPENAI_DEPLOYMENT_NAME" not in os.environ:
        return None, None, "Error: Azure OpenAI environment variables are not set."

    llm = AzureChatOpenAI(
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
        temperature=0
    )
    
    # The modern way to do reliable extraction is by binding a Pydantic model
    # to the LLM, which will use its tool-calling capabilities.
    extractor_agent = llm.with_structured_output(ParsedQuery)
    
    prompt = (
        "You are an expert at extracting specific financial transaction details from user text. "
        "Extract the required entities from the following query. Remember that 'regulator' and "
        "'field_to_investigate' are mandatory.\n\n"
        f"User Query: \"{user_query}\""
    )
    
    try:
        # Invoke the agent to get the structured data
        parsed_data = extractor_agent.invoke(prompt)
        
        # If successful, prepare the output as requested
        # Create a dictionary of all found entities
        output_dict = parsed_data.dict()
        
        # Pop the impacted field from the dictionary to return it separately
        impacted_field = output_dict.pop('field_to_investigate', None)
        
        print(f"✅ Success! Extracted: {output_dict}, Field: {impacted_field}")
        return output_dict, impacted_field, None

    except ValidationError as e:
        # This block catches errors if the LLM fails to extract the mandatory fields.
        error_message = f"Validation Error: The query is missing mandatory information. Details: {e}"
        print(f"❌ {error_message}")
        return None, None, error_message
    except Exception as e:
        # Catch any other unexpected errors during the API call
        error_message = f"An unexpected error occurred: {e}"
        print(f"❌ {error_message}")
        return None, None, error_message



# tests/test_query_parser.py

import os
from agent.query_parser import parse_user_query

def run_tests():
    """
    A simple utility to run a series of test cases against the query parser.
    """
    # --- Set your Azure environment variables for the test ---
    # You can set these directly or use a .env file and python-dotenv
    # os.environ["AZURE_OPENAI_API_KEY"] = "YOUR_KEY"
    # os.environ["AZURE_OPENAI_ENDPOINT"] = "YOUR_ENDPOINT"
    # os.environ["OPENAI_API_VERSION"] = "YOUR_API_VERSION"
    # os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = "YOUR_DEPLOYMENT_NAME"

    test_cases = [
        {
            "description": "Happy Path: All details provided",
            "query": "Please analyze the Notional Amount for UTI 123ABC under the EMIR regulator. It was a General, House reporting trade.",
            "expects_success": True,
        },
        {
            "description": "Mandatory Only: Only regulator and field provided",
            "query": "I need to check the Currency field for an ASIC trade.",
            "expects_success": True,
        },
        {
            "description": "Failure Case: Missing field to investigate",
            "query": "Can you look at UTI 987ZYX for EMIR?",
            "expects_success": False,
        },
        {
            "description": "Failure Case: Missing regulator",
            "query": "Check the Maturity Date for trade 555XYZ.",
            "expects_success": False,
        },
        {
            "description": "Conversational Style",
            "query": "Hi there, for trade ID GHI456, could you look into the 'Product ID' field? This is for MAS and it's a Delegated, Collateral & Valuation trade.",
            "expects_success": True,
        }
    ]

    print("===================================")
    print("= Running Query Parser Test Suite =")
    print("===================================")

    for i, test in enumerate(test_cases):
        print(f"\n--- Test Case {i+1}: {test['description']} ---")
        
        details_dict, impacted_field, error = parse_user_query(test["query"])
        
        if test["expects_success"]:
            if error:
                print(f"🔴 FAILED: Expected success, but got an error: {error}")
            else:
                print(f"🟢 PASSED: Successfully parsed.")
                # You can add more specific assertions here if needed
                # assert details_dict['regulator'] is not None
        else: # Expects failure
            if not error:
                print(f"🔴 FAILED: Expected an error, but the parsing succeeded.")
            else:
                print(f"🟢 PASSED: Correctly failed as expected.")

    print("\n--- Test Suite Complete ---")

if __name__ == "__main__":
    run_tests()
