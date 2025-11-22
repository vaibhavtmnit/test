from typing import Annotated, List, Optional, Literal, TypedDict
from enum import Enum
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import END

# --- Enums for Configuration ---
class AgentStatus(str, Enum):
    SUCCESS = "SUCCESS"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"
    NEED_INFO = "NEED_INFO"
    # Initial state when no agent has run yet
    NONE = "NONE" 

class AgentNames(str, Enum):
    SUPERVISOR = "supervisor"
    AGENT_A = "agent_a"
    AGENT_B = "agent_b" # Added B as per requirement example
    KARAKORAM = "karakoram"

# --- Graph State ---
class AgentState(TypedDict):
    # Stores the full conversation history (User + AI messages)
    messages: List[BaseMessage] 
    
    # The ID of the agent that last executed (e.g., "agent_a")
    active_agent: Optional[str]
    
    # The status returned by the last agent (SUCCESS, NEED_INFO, etc.)
    processing_status: AgentStatus
    
    # The decision made by the supervisor on where to go next
    next_node: Optional[str]
    
    # Special payload for UI (Karakoram)
    ui_payload: Optional[dict]


from enum import Enum

# 1. Define your Agent Names (The identifiers)
class AgentNames(str, Enum):
    SUPERVISOR = "supervisor"
    AGENT_A = "agent_a"          # The default entry point
    DATA_VALIDATOR = "data_validator"
    KARAKORAM = "karakoram"
    # Add new agents here as needed

# 2. The Routing Manifest (The Rubric)
# This maps the Agent Name to the "Instruction" for the Supervisor.
ROUTING_MANIFEST = {
    AgentNames.AGENT_A: (
        "Primary Entry Point. Handles general queries and the initial user request. "
        "If the user's intent is unclear or generic, default here."
    ),
    AgentNames.DATA_VALIDATOR: (
        "Run this ONLY when the user explicitly asks to validate data formats, "
        "check schemas, or verify file integrity."
    ),
    AgentNames.KARAKORAM: (
        "Run this for special reporting tasks or when the user mentions 'Karakoram' analysis."
    )
}

def get_agent_descriptions():
    """Helper to format the manifest for the System Prompt."""
    return "\n".join([f"- '{name.value}': {desc}" for name, desc in ROUTING_MANIFEST.items()])



from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage
from langgraph.graph import END

# We define the output schema. 
# Note: In a dynamic system, ensure this Literal includes all agents in your Enum.
class RouterOutput(BaseModel):
    next_node: Literal[
        AgentNames.AGENT_A, 
        AgentNames.DATA_VALIDATOR, 
        AgentNames.KARAKORAM, 
        "FINISH"
    ] = Field(description="The next node to execute.")
    
    reasoning: str = Field(description="Brief reason for the routing decision.")

def supervisor_node(state: AgentState):
    """
    Supervisor Node with Dynamic Routing Manifest.
    """
    messages = state.get("messages", [])
    last_status = state.get("processing_status", AgentStatus.NONE)
    active_agent = state.get("active_agent")
    
    # --- RULE 1: INITIAL REQUEST ENFORCEMENT ---
    # "Initially send the request to node A"
    # If this is the very first message (or state was reset), we force Agent A.
    # We don't even need to burn LLM tokens for this obvious rule.
    if last_status == AgentStatus.NONE or len(messages) <= 1:
        return {
            "next_node": AgentNames.AGENT_A,
            "active_agent": AgentNames.AGENT_A
        }

    # --- DYNAMIC PROMPT CONSTRUCTION ---
    
    # Get the descriptions from our configuration object
    agent_rubric = get_agent_descriptions()

    system_prompt = (
        "You are the Supervisor AI. You manage a team of agents.\n\n"
        
        "### CURRENT STATE\n"
        f"Last Active Agent: {active_agent}\n"
        f"Last Status: {last_status}\n\n"
        
        "### AVAILABLE AGENTS & TRIGGERS\n"
        f"{agent_rubric}\n\n"
        
        "### ROUTING LOGIC (Follow Strictly)\n"
        
        "1. **CONTINUATION (Need Info)**:\n"
        "   IF 'Last Status' is 'NEED_INFO':\n"
        "   - The user has just answered a question from the previous agent.\n"
        "   - YOU MUST route back to the 'Last Active Agent' ({active_agent}).\n\n"
        
        "2. **NEW TASK (Success)**:\n"
        "   IF 'Last Status' is 'SUCCESS':\n"
        "   - The previous task is complete.\n"
        "   - Analyze the User's LATEST message.\n"
        "   - Match the user's intent to the 'AVAILABLE AGENTS' descriptions above.\n"
        "   - If the user wants to perform a specific action defined in the list, route there.\n"
        "   - If the user's intent is ambiguous or they are just acknowledging, output 'FINISH' to ask for the next step.\n\n"
        
        "3. **ERROR RECOVERY**:\n"
        "   IF 'Last Status' is 'UNKNOWN_ERROR':\n"
        "   - Output 'FINISH' to pause and allow the user to restart.\n"
    )

    # --- EXECUTION ---
    
    # Bind the structured output
    router = llm.with_structured_output(RouterOutput)
    
    # Run the chain
    chain = SystemMessage(content=system_prompt) | router
    response = chain.invoke(messages)
    
    decision = response.next_node
    
    # Map FINISH to LangGraph END
    if decision == "FINISH":
        next_dest = END
    else:
        next_dest = decision

    return {
        "next_node": next_dest,
        # Update active agent only if we are actually going to a node
        "active_agent": decision if decision != "FINISH" else active_agent
    }
