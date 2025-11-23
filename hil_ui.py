import json
import httpx
import chainlit as cl

# Configuration
BACKEND_URL = "http://localhost:8000/stream"  # Replace with your actual API endpoint

@cl.on_chat_start
async def start():
    # Generate or retrieve a thread_id for this session
    thread_id = cl.user_session.get("id")
    cl.user_session.set("thread_id", thread_id)
    await cl.Message(content="System ready. How can I help you?").send()

@cl.on_message
async def main(message: cl.Message):
    thread_id = cl.user_session.get("thread_id")
    
    # Prepare the payload for your API
    # Adjust keys based on your exact API definition (e.g., query params vs json body)
    payload = {
        "user_query": message.content,
        "thread_id": thread_id
    }

    async with httpx.AsyncClient(timeout=300) as client:
        try:
            # Connect to the streaming endpoint
            async with client.stream("POST", BACKEND_URL, json=payload) as response:
                
                # Track the current active message and the node sending it
                current_msg = None
                current_node_name = None

                # Iterate over the Server-Sent Events (SSE)
                async for line in response.aiter_lines():
                    line = line.strip()
                    
                    # Parse SSE format: "data: {...}"
                    if not line.startswith("data: "):
                        continue
                    
                    raw_data = line[6:]  # Remove "data: " prefix
                    if not raw_data:
                        continue

                    try:
                        data = json.loads(raw_data)
                    except json.JSONDecodeError:
                        continue

                    event_type = data.get("type")

                    # --- CASE 1: TEXT STREAMING ---
                    if event_type == "message_chunk":
                        node_name = data.get("node")
                        content_chunk = data.get("content")

                        # If the node has changed (or it's the first message), start a new bubble
                        if node_name != current_node_name:
                            # If there was a previous message open, send/close it first
                            if current_msg:
                                await current_msg.send()
                            
                            # Create a new message, optionally setting the author to the Node Name
                            current_node_name = node_name
                            current_msg = cl.Message(content="", author=node_name)

                        # Stream the token
                        await current_msg.stream_token(content_chunk)

                    # --- CASE 2: SPECIAL EVENTS (KARAKORAM) ---
                    elif event_type == "special_event":
                        # Close any pending text stream first
                        if current_msg:
                            await current_msg.send()
                            current_msg = None
                            current_node_name = None

                        event_name = data.get("event_name")
                        ui_payload = data.get("data")

                        if event_name == "karakoram_success":
                            # Render the UI payload. 
                            # If it's JSON data, we can use cl.Json, or cl.Text for raw output
                            await cl.Message(
                                content=f"**Success:** Process completed.",
                                elements=[
                                    cl.Json(name="Result Data", display="inline", content=ui_payload)
                                ]
                            ).send()

                    # --- CASE 3: INTERACTION REQUEST (NEED INFO) ---
                    elif event_type == "interaction_request":
                        # Close any pending text stream
                        if current_msg:
                            await current_msg.send()
                            current_msg = None
                        
                        # Notify the user they need to provide input
                        # You could also use cl.AskUserMessage here if you want to block execution
                        await cl.Message(
                            content="⚠️ **Input Required**: The system needs more information to proceed.",
                            author="System"
                        ).send()

                # End of stream: Ensure the last message is sent
                if current_msg:
                    await current_msg.send()

        except Exception as e:
            await cl.Message(content=f"An error occurred: {str(e)}").send()
