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

