# graph.py
import asyncio
from typing import Annotated, List, TypedDict
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send
import operator

from agents import create_vertical_researcher

# --- State Definitions ---

class ResearchState(TypedDict):
    """Global state for the horizontal manager."""
    document_chunks: List[str]
    research_goals: List[str]
    final_reports: Annotated[List[str], operator.add] # Append-only list

class WorkerState(TypedDict):
    """State passed to a specific worker."""
    topic_id: str
    topic: str
    chunks: List[str]

# --- Nodes ---

async def dispatch_research(state: ResearchState):
    """
    This node doesn't do work itself; it prepares the data 
    to be 'Sent' to the workers.
    """
    # In a more complex system, this might use an LLM to refine goals.
    # Here, we just pass through.
    pass 

async def vertical_worker_node(state: WorkerState):
    """
    Executes the Deep Agent for a single topic.
    """
    topic_id = state["topic_id"]
    topic = state["topic"]
    chunks = state["chunks"]
    
    print(f"--- [Horizontal Graph] Spawning Worker for: {topic} ---")
    
    # Create the agent factory
    agent = create_vertical_researcher(
        topic_id=topic_id,
        topic=topic,
        total_chunks=len(chunks),
        chunks=chunks
    )
    
    # Run the agent
    # We trigger it with a simple start message. The System Prompt drives the rest.
    input_message = HumanMessage(content=f"Begin research on topic: {topic}")
    
    # deepagents returns a graph, so we invoke it.
    # The result usually contains the conversation history.
    # We assume the final message is the answer.
    result = await agent.ainvoke({"messages": [input_message]})
    
    final_response = result["messages"][-1].content
    
    report = f"## Report: {topic}\n\n{final_response}\n"
    return {"final_reports": [report]}

def map_goals_to_workers(state: ResearchState):
    """
    Mapper function that generates Send objects for parallel execution.
    """
    goals = state["research_goals"]
    chunks = state["document_chunks"]
    
    tasks = []
    for i, goal in enumerate(goals):
        tasks.append(
            Send(
                "vertical_worker", 
                {
                    "topic_id": str(i),
                    "topic": goal,
                    "chunks": chunks
                }
            )
        )
    return tasks

# --- Graph Construction ---

def create_horizontal_orchestrator():
    builder = StateGraph(ResearchState)
    
    # Nodes
    builder.add_node("dispatcher", dispatch_research)
    builder.add_node("vertical_worker", vertical_worker_node)
    
    # Edges
    builder.add_edge(START, "dispatcher")
    
    # Conditional Edge (Map-Reduce pattern)
    builder.add_conditional_edges(
        "dispatcher",
        map_goals_to_workers,
        ["vertical_worker"] 
    )
    
    builder.add_edge("vertical_worker", END)
    
    return builder.compile()
