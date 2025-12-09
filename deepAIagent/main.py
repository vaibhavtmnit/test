# main.py
import asyncio
from graph import create_horizontal_orchestrator

# Mock Data for Demonstration
MOCK_DOCUMENT = [
    "Chunk 0: Project Apollo was a series of spaceflights undertaken by NASA.",
    "Chunk 1: The goal was to land the first humans on the Moon.",
    "Chunk 2: Apollo 11 achieved this goal in July 1969.",
    "Chunk 3: Budget constraints eventually led to the cancellation of later missions.",
    "Chunk 4: Technologies developed during Apollo include freeze-dried food and cordless drills."
]

RESEARCH_GOALS = [
    "What were the primary objectives of the project?",
    "What were the technological spin-offs?"
]

async def main():
    print("Initializing Deep Research Document Analyzer...")
    
    # 1. Initialize the Horizontal Graph
    orchestrator = create_horizontal_orchestrator()
    
    # 2. Prepare Input
    initial_state = {
        "document_chunks": MOCK_DOCUMENT,
        "research_goals": RESEARCH_GOALS,
        "final_reports": []
    }
    
    # 3. Execute
    # streaming mode allows us to see steps if configured, here we just await result
    print("Starting Orchestration (Parallel Execution)...")
    final_state = await orchestrator.ainvoke(initial_state)
    
    # 4. Output Results
    print("\n\n=== FINAL AGGREGATED REPORT ===\n")
    for report in final_state["final_reports"]:
        print(report)
        print("-" * 40)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Execution stopped.")
