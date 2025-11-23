async def event_generator(user_query: str, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    
    # We define which nodes are allowed to "speak" to the user directly.
    # We EXCLUDE the Supervisor because its output is internal routing logic.
    ALLOWED_STREAMING_NODES = {
        AgentNames.AGENT_A, 
        AgentNames.DATA_VALIDATOR, 
        AgentNames.KARAKORAM
    }

    async for event in app.astream_events(
        {"messages": [HumanMessage(content=user_query)]},
        config=config,
        version="v2"
    ):
        event_type = event["event"]
        
        # --- CRITICAL FIX 2: IDENTIFYING THE NODE ---
        # We look at metadata to find the actual Graph Node Name.
        # If the event happens outside a node (e.g. graph start), this might be None.
        current_node = event.get("metadata", {}).get("langgraph_node")

        # --- 1. HANDLE TEXT STREAMING ---
        if event_type == "on_chat_model_stream":
            
            # --- CRITICAL FIX 1: PREVENT LEAKAGE ---
            # Only stream if the current node is in our "Allowed" list.
            # This effectively silences the Supervisor's internal JSON generation.
            if current_node not in ALLOWED_STREAMING_NODES:
                continue
            
            # Get the content
            chunk = event["data"]["chunk"]
            if hasattr(chunk, "content") and chunk.content:
                payload = {
                    "type": "message_chunk",
                    "content": chunk.content,
                    "node": current_node
                }
                yield f"data: {json.dumps(payload)}\n\n"

        # --- 2. HANDLE NODE COMPLETION (Logic & Payloads) ---
        elif event_type == "on_chain_end":
            # We still use metadata to identify the node here
            if current_node is None:
                continue

            output_data = event["data"].get("output")
            
            # Ensure it's a dict and has our specific keys
            if output_data and isinstance(output_data, dict) and "processing_status" in output_data:
                
                status = output_data.get("processing_status")
                ui_payload = output_data.get("ui_payload")
                
                # --- KARAKORAM LOGIC ---
                if current_node == AgentNames.KARAKORAM and status == AgentStatus.SUCCESS:
                    response_payload = {
                        "type": "special_event",
                        "event_name": "karakoram_success",
                        "data": ui_payload
                    }
                    yield f"data: {json.dumps(response_payload)}\n\n"

                # --- NEED INFO LOGIC ---
                elif status == AgentStatus.NEED_INFO:
                    # We can send a signal to UI to unlock input or show a form
                    response_payload = {
                        "type": "interaction_request",
                        "node": current_node,
                        "status": "NEED_INFO"
                    }
                    yield f"data: {json.dumps(response_payload)}\n\n"
