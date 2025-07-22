# main.py

import uuid
import json
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage

# Import the application instance from your supervisor file
from agent.supervisor import supervisor_app

# --- Pydantic Model for the API Request ---
class InvokeRequest(BaseModel):
    input: str
    thread_id: str | None = None

# --- FastAPI App Setup ---
app = FastAPI(
    title="Multi-Agent Supervisor API (Streaming)",
    description="Backend server that runs the LangGraph supervisor agent with streaming."
)

# --- Streaming Endpoint Generator ---
async def stream_generator(request: InvokeRequest):
    """
    An async generator that streams the agent's process back to the client.
    """
    # 1. Configure the conversation thread
    thread_id = request.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    input_messages = {"messages": [HumanMessage(content=request.input)]}
    
    # 2. Use .astream_events() to get a real-time stream of graph events
    # The version="v1" parameter ensures a stable event schema.
    async for event in supervisor_app.astream_events(input_messages, config=config, version="v1"):
        kind = event["event"]
        
        # Stream the content from the LLM as it's being generated
        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if chunk.content:
                # Yield a JSON object for each content chunk
                yield f"data: {json.dumps({'type': 'content', 'data': chunk.content})}\n\n"
        
        # Stream the output of any tool as soon as it finishes
        elif kind == "on_tool_end":
            tool_output = event["data"].get("output")
            if tool_output:
                yield f"data: {json.dumps({'type': 'tool_output', 'data': tool_output})}\n\n"

    # 3. After the stream is complete, send a final message with the thread_id
    yield f"data: {json.dumps({'type': 'end', 'thread_id': thread_id})}\n\n"


@app.post("/invoke_stream")
async def invoke_supervisor_stream(request: InvokeRequest):
    """
    This endpoint receives a user message and streams the agent's
    response back to the client using Server-Sent Events (SSE).
    """
    return StreamingResponse(stream_generator(request), media_type="text/event-stream")



# chainlit_app.py

import chainlit as cl
import httpx
import json

FASTAPI_STREAM_URL = "http://127.0.0.1:8000/invoke_stream"

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
        async with httpx.AsyncClient(timeout=300) as client:
            # Use client.stream() to handle the streaming response from FastAPI
            async with client.stream("POST", FASTAPI_STREAM_URL, json=payload) as response:
                response.raise_for_status()
                # Read the stream line by line
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        # Parse the JSON data from the Server-Sent Event
                        data_str = line[len("data:"):].strip()
                        if data_str:
                            data = json.loads(data_str)
                            
                            # Handle the different types of events from the stream
                            if data["type"] == "content":
                                # Stream the LLM's response token by token
                                await response_msg.stream_token(data["data"])
                            elif data["type"] == "tool_output":
                                # You can decide how to display tool output.
                                # Here, we'll just add it to the message with some formatting.
                                await response_msg.stream_token(f"\n\n*Tool Output:*\n```\n{data['data']}\n```\n")
                            elif data["type"] == "end":
                                # The stream is finished, save the final thread_id
                                cl.user_session.set("thread_id", data["thread_id"])

    except httpx.RequestError as e:
        await response_msg.stream_token(f"\nError: Could not connect to the agent server. Details: {e}")
    except Exception as e:
        await response_msg.stream_token(f"\nAn unexpected error occurred: {e}")

    await response_msg.update()
