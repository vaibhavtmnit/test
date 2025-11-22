import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage

# Assuming 'app' is your compiled LangGraph workflow
# app = workflow.compile(checkpointer=memory)

async def event_generator(user_query: str, thread_id: str):
    """
    Generates Server-Sent Events (SSE) for the UI.
    """
    config = {"configurable": {"thread_id": thread_id}}
    
    # We stream events to capture both tokens (text) and node state changes
    async for event in app.astream_events(
        {"messages": [HumanMessage(content=user_query)]},
        config=config,
        version="v2"
    ):
        event_type = event["event"]
        node_name = event.get("name")
        
        # --- 1. HANDLE TEXT STREAMING (Standard Chat) ---
        # This ensures the user sees the text being typed out in real-time
        if event_type == "on_chat_model_stream":
            chunk_content = event["data"]["chunk"].content
            if chunk_content:
                payload = {
                    "type": "message_chunk",
                    "content": chunk_content,
                    "node": node_name # Optional: show who is talking
                }
                yield f"data: {json.dumps(payload)}\n\n"

        # --- 2. HANDLE NODE COMPLETION (Logic & Special Payloads) ---
        elif event_type == "on_chain_end":
            # 'data' contains the return value of the node function
            output_data = event["data"].get("output")
            
            # We only care if the output is a dict containing our specific keys
            # (LangGraph internal nodes might return other things, so we filter)
            if output_data and isinstance(output_data, dict) and "processing_status" in output_data:
                
                status = output_data.get("processing_status")
                ui_payload = output_data.get("ui_payload")
                
                # --- SCENARIO A: KARAKORAM SUCCESS (Special Trigger) ---
                if node_name == AgentNames.KARAKORAM and status == AgentStatus.SUCCESS:
                    response_payload = {
                        "type": "special_event",
                        "event_name": "karakoram_success",
                        "node": node_name,
                        "data": ui_payload, # The specific data from the agent
                        "message": "Karakoram analysis complete."
                    }
                    yield f"data: {json.dumps(response_payload)}\n\n"

                # --- SCENARIO B: NEED INFO (Any Agent) ---
                elif status == AgentStatus.NEED_INFO:
                    response_payload = {
                        "type": "interaction_request",
                        "node": node_name,
                        "status": "NEED_INFO",
                        "message": "Please provide the requested information."
                    }
                    yield f"data: {json.dumps(response_payload)}\n\n"

                # --- SCENARIO C: UNKNOWN ERROR ---
                elif status == AgentStatus.UNKNOWN_ERROR:
                    response_payload = {
                        "type": "error",
                        "node": node_name,
                        "message": "An unknown error occurred. Please restart."
                    }
                    yield f"data: {json.dumps(response_payload)}\n\n"
                
                # --- SCENARIO D: GENERIC SUCCESS (Agent A, etc.) ---
                # Optional: If you want to signal the UI that a step finished
                elif status == AgentStatus.SUCCESS and node_name != AgentNames.KARAKORAM:
                    response_payload = {
                        "type": "step_complete",
                        "node": node_name,
                        "status": "SUCCESS"
                    }
                    yield f"data: {json.dumps(response_payload)}\n\n"

# --- FastAPI Endpoint ---
@app_fastapi.post("/chat")
async def chat_endpoint(request: dict):
    # request example: {"query": "run karakoram", "thread_id": "123"}
    return StreamingResponse(
        event_generator(request["query"], request["thread_id"]),
        media_type="text/event-stream"
    )
