import uuid
import time
from typing import List, Dict, Tuple
from collections import defaultdict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_openai import OpenAIEmbeddings

# --- 1. The Hit Memory Component ---
class SessionHitMemory:
    """
    Keeps track of how many times specific chunks have been retrieved 
    during a specific analysis session.
    """
    def __init__(self):
        # Maps chunk_id -> count
        self.hit_counts: Dict[str, int] = defaultdict(int)
        # Maps chunk_id -> timestamp of last access (for potential decay logic)
        self.last_accessed: Dict[str, float] = {}
        # Total chunks in the system (set during ingestion)
        self.total_chunks = 0

    def register_hit(self, chunk_id: str):
        """Increment the hit counter for a chunk."""
        self.hit_counts[chunk_id] += 1
        self.last_accessed[chunk_id] = time.time()

    def get_stats(self) -> Dict:
        """Return coverage statistics."""
        unique_hits = len(self.hit_counts)
        coverage = (unique_hits / self.total_chunks * 100) if self.total_chunks > 0 else 0
        return {
            "unique_chunks_accessed": unique_hits,
            "total_chunks_available": self.total_chunks,
            "coverage_percentage": f"{coverage:.2f}%",
            "most_accessed_chunks": sorted(self.hit_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        }

# --- 2. The Sophisticated RAG System ---
class SophisticatedRAGTool:
    def __init__(self, raw_text: str):
        self.memory = SessionHitMemory()
        self.embeddings = OpenAIEmbeddings() # Requires OPENAI_API_KEY env var
        
        # A. Ingestion & Chunking
        print("--- Ingesting and Chunking ---")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100, # Critical for "scattered info" at boundaries
            add_start_index=True,
        )
        
        raw_docs = text_splitter.create_documents([raw_text])
        
        # Assign deterministic IDs to chunks for tracking
        self.docs = []
        for i, doc in enumerate(raw_docs):
            # Create a simple ID. In production, use a hash of content.
            doc_id = f"chunk_{i}"
            doc.metadata["chunk_id"] = doc_id
            doc.metadata["hit_count"] = 0 
            self.docs.append(doc)
            
        self.memory.total_chunks = len(self.docs)
        print(f"Generated {len(self.docs)} chunks.")

        # B. Hybrid Indexing (Vector + BM25)
        print("--- Building Hybrid Index ---")
        
        # 1. Vector Index (Semantic)
        self.vector_store = FAISS.from_documents(self.docs, self.embeddings)
        self.vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        
        # 2. Keyword Index (Lexical/Sparse)
        self.bm25_retriever = BM25Retriever.from_documents(self.docs)
        self.bm25_retriever.k = 5
        
        # 3. Ensemble (Hybrid)
        # Weights: 0.6 for Semantic, 0.4 for Keyword (adjustable)
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.vector_retriever, self.bm25_retriever],
            weights=[0.6, 0.4]
        )

    def query(self, user_query: str) -> str:
        """
        The main public method to use this as a tool.
        """
        print(f"\nQuerying: '{user_query}'")
        
        # 1. Retrieve
        results = self.ensemble_retriever.invoke(user_query)
        
        # 2. Update Memory & Format Output
        formatted_response = []
        
        for doc in results:
            c_id = doc.metadata.get("chunk_id")
            
            # TRACKING: Register the hit
            self.memory.register_hit(c_id)
            current_hits = self.memory.hit_counts[c_id]
            
            # Formatting the output chunk
            # We add a warning if this chunk has been seen too many times
            usage_note = "(New Info)" if current_hits == 1 else f"(Seen {current_hits} times)"
            
            chunk_out = (
                f"--- [ID: {c_id}] {usage_note} ---\n"
                f"{doc.page_content}\n"
            )
            formatted_response.append(chunk_out)
            
        return "\n".join(formatted_response)

    def get_session_stats(self):
        return self.memory.get_stats()

# --- 3. Example Usage Simulation ---

# Mock Data (Scattered Info about Topic A and B)
mock_text = """
[Start] The financial results for Q3 were promising. Project Alpha has officially started. 
However, there are risks associated with the supply chain. 
(Topic B info starts here) The HR policy was updated yesterday. 
Project Alpha is expected to deliver results by 2025. 
The updated HR policy includes 4 remote days. 
(Back to Topic A) The budget for Alpha is $5M. 
The supply chain risks for Alpha are mitigated by new vendors.
"""

# Initialize
rag_system = SophisticatedRAGTool(mock_text)

# Simulation: Deep Agent asking questions
print(rag_system.query("What is the budget for Project Alpha?"))
print(rag_system.query("Tell me about HR policies"))

# Check Memory State
print("\n--- Session Stats ---")
print(rag_system.get_session_stats())









import re
from typing import List, Dict, Optional

# ... (Previous imports: Document, FAISS, etc. remain the same) ...

class SophisticatedRAGTool:
    def __init__(self, raw_text: str):
        # --- Memory & Ingestion (Same as before) ---
        self.memory = SessionHitMemory()
        self.embeddings = OpenAIEmbeddings()
        
        print("--- Ingesting and Chunking ---")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50, add_start_index=True
        )
        
        raw_docs = text_splitter.create_documents([raw_text])
        
        # We maintain a list for index-based access (Crucial for neighbor lookup)
        self.docs_list = [] 
        
        for i, doc in enumerate(raw_docs):
            # We explicitly store the integer index 'i' in metadata
            doc_id = f"chunk_{i}"
            doc.metadata["chunk_id"] = doc_id
            doc.metadata["index"] = i 
            doc.metadata["hit_count"] = 0
            self.docs_list.append(doc)
            
        self.memory.total_chunks = len(self.docs_list)
        
        # --- Indexing (Same as before) ---
        print("--- Building Indices ---")
        self.vector_store = FAISS.from_documents(self.docs_list, self.embeddings)
        self.vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        self.bm25_retriever = BM25Retriever.from_documents(self.docs_list)
        self.bm25_retriever.k = 5
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.vector_retriever, self.bm25_retriever],
            weights=[0.6, 0.4]
        )

    # ... (Previous query method remains here) ...

    def query_structural_context(self, identifier: str, expansion: int = 1) -> str:
        """
        Search for a specific structural identifier (e.g., 'Section 3.2', 'Table 1')
        and return the matching chunks PLUS their neighbors.
        
        Args:
            identifier: The string to look for (e.g., "3.2", "Figure A")
            expansion: How many chunks before/after to retrieve (default 1)
        """
        print(f"\n[Structural Search] Looking for anchor: '{identifier}' with context ±{expansion}")
        
        # 1. Find Anchor Chunks (Simple Linear Regex Search)
        # We scan documents because structural markers might be ignored by vectors.
        matched_indices = []
        for doc in self.docs_list:
            if identifier.lower() in doc.page_content.lower():
                matched_indices.append(doc.metadata["index"])
        
        if not matched_indices:
            return f"No structure found matching identifier: '{identifier}'"

        # 2. Context Expansion Logic
        # We use a set to avoid duplicate chunks if anchors are close together
        indices_to_fetch = set()
        
        for idx in matched_indices:
            # Add the anchor itself
            indices_to_fetch.add(idx)
            
            # Add neighbors (handling boundaries 0 and max_length)
            start = max(0, idx - expansion)
            end = min(len(self.docs_list) - 1, idx + expansion)
            
            # Add range to set
            for neighbor_idx in range(start, end + 1):
                indices_to_fetch.add(neighbor_idx)

        # 3. Retrieve and Format
        sorted_indices = sorted(list(indices_to_fetch))
        formatted_response = []
        
        for idx in sorted_indices:
            doc = self.docs_list[idx]
            c_id = doc.metadata["chunk_id"]
            
            # Register Hit
            self.memory.register_hit(c_id)
            
            # Visual marker to show if this is the Anchor or Context
            is_anchor = idx in matched_indices
            role_label = "**ANCHOR**" if is_anchor else "Context"
            
            chunk_out = (
                f"--- [ID: {c_id}] [{role_label}] ---\n"
                f"{doc.page_content}\n"
            )
            formatted_response.append(chunk_out)

        return "\n".join(formatted_response)

# --- Usage Example ---

mock_text_structure = """
(Chunk 10) This is the introduction.
(Chunk 11) We will discuss the methodology now.
(Chunk 12) Section 3.2: Dependency Injection. The system relies on DI for modularity.
(Chunk 13) This allows for easier testing and decoupling of components.
(Chunk 14) Furthermore, the configuration is stored in YAML files.
(Chunk 15) Section 3.3: Database Schema. The schema is normalized.
"""

rag = SophisticatedRAGTool(mock_text_structure)

# SCENARIO: The agent sees a reference to "Section 3.2" and wants to read it + context.
# It asks for expansion=1 to see the chunk before and after.
result = rag.query_structural_context("Section 3.2", expansion=1)
print(result)


