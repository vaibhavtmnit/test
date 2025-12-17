import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

# ---------------------------------------------------------
# 1. SETUP & CONFIGURATION
# ---------------------------------------------------------
# Load the SOTA model. 'all-mpnet-base-v2' is widely considered
# the best trade-off between speed and accuracy for semantic search.
model = SentenceTransformer('all-mpnet-base-v2')

# Configuration for weights (customize importance of each column)
# If unsure, keep them equal (1.0, 1.0, 1.0)
WEIGHTS = {
    'summary': 0.4,
    'description': 0.2,
    'entities': 0.4
}

# ---------------------------------------------------------
# 2. DATA PREPARATION (Mock Data)
# ---------------------------------------------------------
# In production, load these via pd.read_csv('file1.csv')
data1 = {
    'id': ['A001', 'A002'],
    'summary': ['AI algorithm for stock prediction', 'New baking recipe for chocolate cake'],
    'description': ['Uses LSTM and Transformer models to predict market trends.', 'A flourless cake using dark chocolate and eggs.'],
    'entities': ['finance, stock market, LSTM', 'baking, chocolate, dessert']
}

data2 = {
    'id': ['B001', 'B002', 'B003', 'B004'],
    'summary': ['Market trend analysis tool', 'Pastry cooking guide', 'Automated trading bot', 'Car engine repair'],
    'description': ['Predicts stock prices using deep learning.', 'How to make brownies and cakes.', 'High frequency trading algo.', 'Fixing pistons and oil leaks.'],
    'entities': ['stock market, prediction, finance', 'dessert, baking, sugar', 'finance, trading, bot', 'mechanic, pistons']
}

df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

# ---------------------------------------------------------
# 3. THE SOTA MATCHING PROCESS
# ---------------------------------------------------------

def generate_embeddings(df, columns):
    """
    Generates a dictionary of embeddings for specified columns.
    """
    embeddings = {}
    for col in columns:
        print(f"Embedding column: {col}...")
        # encode(convert_to_tensor=True) moves data to GPU if available for speed
        embeddings[col] = model.encode(df[col].tolist(), convert_to_tensor=True)
    return embeddings

# Generate embeddings for both files
cols_to_match = ['summary', 'description', 'entities']
emb1 = generate_embeddings(df1, cols_to_match)
emb2 = generate_embeddings(df2, cols_to_match)

print("\nCalculating Similarity Matrices...")

# We will store the similarity matrices for each column
# shape of each matrix will be (len(df1), len(df2))
sim_matrices = {}

for col in cols_to_match:
    # Cosine Similarity is the industry standard for normalized vectors
    sim_matrices[col] = util.cos_sim(emb1[col], emb2[col])

# ---------------------------------------------------------
# 4. SCORING & FUSION
# ---------------------------------------------------------

# Calculate Weighted Average Score
# Initialize zero matrix of shape (len(df1), len(df2))
total_score_matrix = torch.zeros(len(df1), len(df2))
total_weight = sum(WEIGHTS.values())

for col in cols_to_match:
    weight = WEIGHTS.get(col, 1.0)
    # Add weighted similarity to total
    total_score_matrix += sim_matrices[col] * weight

# Normalize by total weight to keep score between 0 and 1 (approx)
final_scores = total_score_matrix / total_weight

# ---------------------------------------------------------
# 5. RETRIEVAL & REPORTING (Top N)
# ---------------------------------------------------------
TOP_N = 2
results = []

# Iterate through each item in File 1
for idx1, row1 in df1.iterrows():
    # Get scores for this specific item against all items in File 2
    scores = final_scores[idx1]
    
    # Get top N indices in File 2 that match this item
    # torch.topk returns values and indices
    top_scores, top_indices = torch.topk(scores, k=min(TOP_N, len(df2)))
    
    matches = []
    for score, idx2_tensor in zip(top_scores, top_indices):
        idx2 = idx2_tensor.item()
        
        # EXTRACTING THE "WHY" (Explainability)
        # We retrieve the specific similarity score for each column
        details = {}
        for col in cols_to_match:
            # Convert tensor score to percentage (0-100)
            col_score = sim_matrices[col][idx1][idx2].item() * 100
            details[f"{col}_match"] = f"{col_score:.1f}%"
        
        matches.append({
            'match_id': df2.iloc[idx2]['id'],
            'match_summary': df2.iloc[idx2]['summary'],
            'overall_score': f"{score.item() * 100:.1f}%",
            'explainability': details
        })
    
    results.append({
        'source_id': row1['id'],
        'source_summary': row1['summary'],
        'top_matches': matches
    })

# ---------------------------------------------------------
# 6. OUTPUT
# ---------------------------------------------------------
import json
print(json.dumps(results, indent=2))
