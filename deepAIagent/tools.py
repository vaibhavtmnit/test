# tools.py
import os
import yaml
from typing import Optional, Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

# --- Tool: Log Evidence ---

class LogEvidenceInput(BaseModel):
    chunk_id: int = Field(..., description="The index of the document chunk being analyzed.")
    quote: str = Field(..., description="Direct quote from the text supporting the finding.")
    analysis: str = Field(..., description="Reasoning on why this is relevant to the topic.")
    topic_id: str = Field(..., description="The ID of the topic being researched.")

class LogEvidenceTool(BaseTool):
    name: str = "log_evidence"
    description: str = (
        "Appends structured evidence to a YAML file. "
        "Use this during Phase A to store findings."
    )
    args_schema: Type[BaseModel] = LogEvidenceInput

    def _run(self, chunk_id: int, quote: str, analysis: str, topic_id: str) -> str:
        filename = f"topic_{topic_id}_evidence.yaml"
        entry = {
            "chunk_id": chunk_id,
            "quote": quote,
            "analysis": analysis
        }
        
        # Append to YAML file safely
        try:
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    data = yaml.safe_load(f) or []
            else:
                data = []
            
            data.append(entry)
            
            with open(filename, 'w') as f:
                yaml.dump(data, f)
                
            return f"Evidence logged for chunk {chunk_id} to {filename}."
        except Exception as e:
            return f"Error logging evidence: {e}"

# --- Tool: Read Evidence File ---

class ReadEvidenceInput(BaseModel):
    topic_id: str = Field(..., description="The ID of the topic to read notes for.")

class ReadEvidenceTool(BaseTool):
    name: str = "read_evidence_file"
    description: str = "Reads the accumulated notes/evidence for a specific topic."
    args_schema: Type[BaseModel] = ReadEvidenceInput

    def _run(self, topic_id: str) -> str:
        filename = f"topic_{topic_id}_evidence.yaml"
        if not os.path.exists(filename):
            return "No evidence found yet."
        with open(filename, 'r') as f:
            return f.read()

# --- Tool: Fetch Chunk ---
# To avoid context overflow, we let the agent 'fetch' chunks one by one.

class FetchChunkInput(BaseModel):
    chunk_index: int = Field(..., description="Index of the chunk to read (0-based).")

class FetchChunkTool(BaseTool):
    name: str = "fetch_chunk"
    description: str = "Retrieves the text content of a specific document chunk."
    args_schema: Type[BaseModel] = FetchChunkInput
    
    def __init__(self, chunks: list[str], **kwargs):
        super().__init__(**kwargs)
        self.chunks_store = chunks

    # Pydantic private attribute to store the chunks
    chunks_store: list[str] = Field(default_factory=list, exclude=True)

    def _run(self, chunk_index: int) -> str:
        if 0 <= chunk_index < len(self.chunks_store):
            return self.chunks_store[chunk_index]
        return "Error: Chunk index out of bounds."
