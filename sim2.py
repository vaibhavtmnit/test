# pip install ragatouille
from ragatouille import RAGPretrainedModel

# 1. Load SOTA ColBERTv2 model
RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

# 2. Data Preparation (Mock)
docs = [
    "Apple released the iPhone 15 with titanium edges and USB-C.",
    "The stock market saw a dip in tech shares today.",
    "A guide to baking flourless chocolate cake."
]
doc_ids = ["doc1", "doc2", "doc3"]

# 3. Indexing (In-memory for this demo)
# In production, this saves to disk automatically
index_path = RAG.index(index_name="demo_index", collection=docs, document_ids=doc_ids)

# 4. Retrieval
# Note: ColBERT handles the "extra info" penalty natively via MaxSim
query = "titanium iPhone"
results = RAG.search(query, k=1)

print(results)
# Output includes 'score', 'content', 'document_id'



# pip install sentence-transformers einops
from sentence_transformers import SentenceTransformer

# 1. Load Model with Trust Remote Code (Required for Nomic/Jina architectures)
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

# 2. Asymmetric Prefixing
# Crucial: 'search_query' vs 'search_document' tells the model "This is asymmetric"
query_text = "search_query: baking cake"
doc_text = "search_document: " + ("This is a long article about cooking... " * 50) + "flourless chocolate cake recipe."

# 3. Embedding & Similarity
emb_query = model.encode(query_text, convert_to_tensor=True)
emb_doc = model.encode(doc_text, convert_to_tensor=True)

# Use standard cosine, but the *model* has already adjusted for the length difference
score = model.similarity(emb_query, emb_doc)
print(f"Match Score: {score.item():.4f}")



import torch
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-mpnet-base-v2')

def get_max_sim_score(query, long_document, chunk_size=200, overlap=50):
    # 1. Chunk the long document
    tokens = long_document.split() # Simple whitespace splitter for demo
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = " ".join(tokens[i : i + chunk_size])
        chunks.append(chunk)
    
    # 2. Encode Query and ALL Chunks
    query_emb = model.encode(query, convert_to_tensor=True)
    chunk_embs = model.encode(chunks, convert_to_tensor=True)
    
    # 3. Calculate Cosine Similarity for every chunk
    cosine_scores = util.cos_sim(query_emb, chunk_embs)
    
    # 4. Take the MAXIMUM score (The "Needle in Haystack" match)
    # This ignores the low scores of irrelevant chunks
    best_score = torch.max(cosine_scores)
    best_chunk_idx = torch.argmax(cosine_scores).item()
    
    return best_score.item(), chunks[best_chunk_idx]

query = "financial report"
# Document has noise at start and end
long_doc = "random noise " * 100 + "The Q3 financial report shows strong growth." + " random noise" * 100

score, match = get_max_sim_score(query, long_doc)
print(f"Max Score: {score:.4f} (Matched Chunk: '{match[:50]}...')")



# pip install fastembed
from fastembed import SparseTextEmbedding

# 1. Load SPLADE model
model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")

# 2. Define Data
documents = [
    "Apple iPhone 15 features",
    "A very long document about fruit agriculture involving apples and oranges...",
    "Tesla Q3 financial report"
]
query = "Apple iPhone"

# 3. Generate Sparse Vectors (Dictionaries of Token ID -> Weight)
# Note: 'batch_size' is for processing
doc_embeddings = list(model.embed(documents))
query_embedding = list(model.embed([query]))[0]

# 4. Sparse Dot Product (Manual Implementation)
def sparse_dot_product(vec_a, vec_b):
    # Only multiply indices that exist in BOTH (Intersection)
    # Non-overlapping indices contribute 0 (No Penalty)
    score = 0
    common_indices = set(vec_a.indices) & set(vec_b.indices)
    
    # Create lookups
    val_a = dict(zip(vec_a.indices, vec_a.values))
    val_b = dict(zip(vec_b.indices, vec_b.values))
    
    for idx in common_indices:
        score += val_a[idx] * val_b[idx]
    return score

for i, doc_emb in enumerate(doc_embeddings):
    score = sparse_dot_product(query_embedding, doc_emb)
    print(f"Doc {i} Score: {score:.4f}")




# pip install pot transformers torch
import torch
import ot
from transformers import AutoTokenizer, AutoModel

# Load standard BERT
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

def get_bert_wmd(text1, text2):
    # 1. Get Token Embeddings (Not sentence embeddings)
    inputs1 = tokenizer(text1, return_tensors="pt")
    inputs2 = tokenizer(text2, return_tensors="pt")
    
    with torch.no_grad():
        out1 = model(**inputs1).last_hidden_state[0] # [Len1, 768]
        out2 = model(**inputs2).last_hidden_state[0] # [Len2, 768]
    
    # 2. Cost Matrix (Euclidean distance between every token pair)
    # cdist computes distance between every vector in out1 and out2
    M = torch.cdist(out1, out2).cpu().numpy()
    
    # 3. Weights (Uniform - every word has equal weight)
    # In production, use IDF weights here to ignore 'the', 'is', etc.
    n1, n2 = M.shape
    a = torch.ones(n1) / n1
    b = torch.ones(n2) / n2
    
    # 4. Earth Mover's Distance
    # "How much work to move distribution 'a' to distribution 'b'?"
    emd_score = ot.emd2(a.numpy(), b.numpy(), M)
    return emd_score

score = get_bert_wmd("iPhone titanium", "The new iPhone 15 has a titanium body")
print(f"Transport Cost (Lower is better): {score:.4f}")




# pip install sentence-transformers
from sentence_transformers import CrossEncoder

# 1. Load NLI Model
# This model outputs 3 scores: [Contradiction, Entailment, Neutral]
model = CrossEncoder('cross-encoder/nli-deberta-v3-base')

query = "Stock market crash"
doc = "The financial exchange saw a massive downturn appearing like a collapse today."

# 2. Predict Entailment
# Input is a pair: (Premise, Hypothesis)
scores = model.predict([(doc, query)]) # Note: Doc is premise, Query is hypothesis

# 3. Extract 'Entailment' Score (Index 1 usually, check model card)
# Label mapping: 0: Contradiction, 1: Entailment, 2: Neutral
entailment_score = scores[0][1] 

# Convert logit to probability
import numpy as np
prob = 1 / (1 + np.exp(-entailment_score))

print(f"Entailment Probability: {prob:.4f}")



# This combines Approach 4 (SPLADE) and Approach 6 (Cross-Encoder)
from fastembed import SparseTextEmbedding
from sentence_transformers import CrossEncoder

# Setup Models
retriever = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
reranker = CrossEncoder('cross-encoder/nli-deberta-v3-base')

def hybrid_match(query, documents):
    # STEP 1: Fast Retrieval (SPLADE)
    # Efficiently filters 100k docs down to top 10
    print("Step 1: SPLADE Retrieval...")
    query_vec = list(retriever.embed([query]))[0]
    doc_vecs = list(retriever.embed(documents))
    
    candidates = []
    for idx, doc_vec in enumerate(doc_vecs):
        # (Use the sparse_dot_product function from Approach 4)
        score = sparse_dot_product(query_vec, doc_vec) 
        candidates.append({'id': idx, 'score': score, 'text': documents[idx]})
    
    # Get Top 5 candidates
    top_candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)[:5]
    
    # STEP 2: Deep Re-ranking (Cross-Encoder)
    # Precision scoring on the small subset
    print("Step 2: Cross-Encoder Re-ranking...")
    pairs = [(c['text'], query) for c in top_candidates]
    ce_scores = reranker.predict(pairs)
    
    final_results = []
    for i, candidate in enumerate(top_candidates):
        entailment_logit = ce_scores[i][1] # Extract entailment
        final_results.append({
            'text': candidate['text'],
            'initial_score': candidate['score'],
            'final_entailment_score': entailment_logit
        })
        
    return sorted(final_results, key=lambda x: x['final_entailment_score'], reverse=True)

# Run
docs = ["Stock market crash", "Banana bread recipe", "Financial downturn report"]
results = hybrid_match("Market collapse", docs)
import json
print(json.dumps(results, indent=2))



