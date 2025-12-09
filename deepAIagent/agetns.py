# agents.py
from deepagents import async_create_deep_agent
from langchain_core.tools import BaseTool
from config import get_chat_model
from tools import LogEvidenceTool, ReadEvidenceTool, FetchChunkTool

def create_vertical_researcher(topic_id: str, topic: str, total_chunks: int, chunks: list[str]):
    """
    Creates a Deep Agent dedicated to a specific research topic.
    """
    model = get_chat_model()
    
    # Initialize tools
    # We pass the specific chunks to the Fetch tool for this instance
    tools = [
        LogEvidenceTool(),
        ReadEvidenceTool(),
        FetchChunkTool(chunks=chunks)
    ]

    # Strict System Prompt to enforce behavior
    system_instructions = f"""
    You are a Deep Research Worker. Your ID is '{topic_id}'.
    Your Goal: Research the topic "{topic}" within the provided document.
    
    You have access to {total_chunks} document chunks (indices 0 to {total_chunks - 1}).
    
    Follow this STRICT process:
    
    PHASE A: DATA INGESTION (Sequential)
    1. Iterate through every chunk index from 0 to {total_chunks - 1}.
    2. For EACH chunk:
       a. Call `fetch_chunk(chunk_index)`.
       b. Analyze the text specifically for "{topic}".
       c. If relevant info is found, call `log_evidence` immediately.
       d. DO NOT try to remember the text. Offload it to the file via the tool.
    
    PHASE B: SYNTHESIS
    1. Once ALL chunks are processed, call `read_evidence_file(topic_id='{topic_id}')`.
    2. Read the YAML data returned.
    3. Synthesize a comprehensive report answering the research goal.
    4. Your final output must be the synthesized report only.
    
    Constraints:
    - Do not skip chunks.
    - Do not hallucinate content not in the chunks.
    - Use the tools for memory.
    """

    # Create the agent using deepagents library
    # This returns a CompiledGraph (Runnable)
    agent = async_create_deep_agent(
        model=model,
        tools=tools,
        instructions=system_instructions
    )
    
    return agent
